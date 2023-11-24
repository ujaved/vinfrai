import streamlit as st
import os
import openai
from pathlib import Path
import shutil
import boto3
from io import StringIO
from utils.terraform import download_terraform, validate_template
import uuid
import datetime
from dataclasses import dataclass, field
from chatbot import MAX_QUESTIONS, Chatbot, MAX_ERROR_RETRIES, MAX_TOKENS, VALIDATE_ERR_MSG, Question

openai.api_key = os.getenv("OPENAI_API_KEY")
openai_model_id = os.getenv("OPENAI_MODEL_ID")
s3_bucket = os.getenv("S3_BUCKET")
display_options = ['template', 'llm notes']

def parse_llm_response(resp: str, llm_notes: list[str]) -> tuple[str, str]:
    template = ''
    if 'hcl' in resp:
        fields = resp.split('```hcl')
        preface = fields[0]
        fields = fields[1].split('```')
        template = fields[0]
        if len(fields) >= 2:
            llm_notes.append(fields[1])
    else:
        fields = resp.split('```')
        preface = fields[0]
        if len(fields) >= 2:
            template = fields[1]
        if len(fields) >= 3:
            llm_notes.append(fields[2])

    return (preface, template)


@dataclass
class Session:
    # id is supposed to be the index in an array of sessions
    id: int
    messages: list[str] = field(default_factory=lambda: [
                                {'role': 'assistant', 'content': 'Please select a provider from the sidebar and provide your Terraform specification'}])
    in_progress: bool = True
    terraform_template: str = ''
    llm_notes: list[str] = field(default_factory=list)
    uuid_str: uuid = uuid.uuid4()
    terraform_dir_name: str = f'terraform_{uuid_str}'
    time_str: str = datetime.datetime.now().strftime("%Y/%m/%d/%H")
    s3_key: str = f'{time_str}/{uuid_str}'
    show_chat_input: str = True
    chatbot: Chatbot = Chatbot(os.getenv("OPENAI_MODEL_ID"), 0)
    tab = None
    initial_spec: str = ''
    user_q_a: list[tuple[Question, str]] = field(default_factory=list)
    user_q_id_to_display: int = 0

    def create_prompt_from_qa(self) -> str:
        prompt = f'For provider {st.session_state.provider} give me a terraform template for {self.initial_spec} with the following additional specifications \n\n'
        for q in self.user_q_a:
            prompt += f'question {q[0].question}. answer: {q[1]}\n'
        return prompt

    def try_validate(self) -> str:
        with st.spinner('validating template'):
            err, err_source = validate_template(
                self.terraform_template, self.terraform_dir_name)

        num_retry = 0
        while err and num_retry < MAX_ERROR_RETRIES and self.chatbot.num_tokens < MAX_TOKENS:
            with self.tab:
                with st.chat_message('assistant'):
                    response = self.chatbot.response(
                        f'while executing {err_source} there were these errors:\n{err}\n Fix the template by correcting these errors')
                    _, self.terraform_template = parse_llm_response(
                        response, self.llm_notes)
            with st.spinner('validating template'):
                err, err_source = validate_template(
                    self.terraform_template, self.terraform_dir_name)
            num_retry += 1

    def get_terraform_template(self, spec: str):
        with self.tab:
            with st.chat_message('assistant'):
                # chatbot response will write streaming responses in an empty container inside the chat_message container
                response = self.chatbot.response(spec)
        preface, self.terraform_template = parse_llm_response(
            response, self.llm_notes)
        preface = response

        if st.session_state.validate_template:
            err = self.try_validate()
            if err:
                preface = VALIDATE_ERR_MSG
                
        self.messages.append({'role': 'assistant', 'content': preface})

    def render_question_radios(self) -> bool:
        for i in range(self.user_q_id_to_display+1):
            if i == MAX_QUESTIONS:
                if not self.terraform_template:
                    self.get_terraform_template(self.create_prompt_from_qa())
                    return True
                break
            q = self.user_q_a[i][0]
            key = f'{i}_{self.id}_user_q'
            st.radio(f'**{q.topic}**: {q.question}', q.possible_answers, horizontal=True, key=key,
                     index=None, on_change=user_question_radio_cb, args=(key,), disabled=i < self.user_q_id_to_display)
        return False

    def render_messages(self):
        for m in self.messages:
            if m['role'] == 'user_questions':
                # if rendering the question radios results in the rendering of the template, 
                # break since otherwise the template will be rendered twice
                if self.render_question_radios():
                    break
            else:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])

    def render(self):
        with self.tab:
            st.metric(
                label=f'Number of tokens used out of a max of {MAX_TOKENS}', value=self.chatbot.num_tokens, delta=self.chatbot.num_tokens_delta)
            self.render_messages()
        if self.in_progress:
            self.show_chat_input = True
        if self.show_chat_input:
            key = f'{self.id}_terraform_description'
            st.chat_input('user input', key=key,
                          on_submit=terraform_description_callback, args=(key,))


def user_question_radio_cb(key: str):
    cur_session = st.session_state.sessions[-1]
    q = cur_session.user_q_a[cur_session.user_q_id_to_display][0] 
    cur_session.user_q_a[cur_session.user_q_id_to_display] = (q, st.session_state[key])
    cur_session.user_q_id_to_display += 1


def terraform_description_callback(chat_input_key: str):
    spec = st.session_state[chat_input_key]
    if len(spec) == 0:
        return
    session_id = int(chat_input_key.split("_")[0])
    session = st.session_state.sessions[session_id]
    with session.tab:
        with st.chat_message('user'):
            st.markdown(spec)
    session.messages.append({'role': 'user', 'content': spec})

    chatbot = session.chatbot
    if chatbot.num_tokens > MAX_TOKENS:
        session.messages.append(
            {'role': 'assistant', 'content': f'Exceeded the token limit of {MAX_TOKENS}. Please start a new session'})
        end_session()
    else:
        if not session.initial_spec:
            session.initial_spec = spec
            ai_starter_msg = "Certainly! Let's start with a few questions that will help me generate your initial template."
            session.messages.append(
                {'role': 'assistant', 'content': ai_starter_msg})
            with session.tab:
                with st.chat_message('assistant'):
                    st.markdown(ai_starter_msg)
            with st.spinner('generating clarifying questions'):
                session.user_q_a = [(q, '')
                                    for q in chatbot.spec_gathering_response(spec)]
                session.messages.append({'role': 'user_questions'})
        else:
            session.get_terraform_template(spec)

    session.show_chat_input = False


def start_session():
    # end the previous session
    end_session()

    id = len(st.session_state.sessions)
    st.session_state.sessions.append(Session(id=id))


def end_session():
    # the last session is the current session
    if len(st.session_state.sessions) == 0:
        return
    session_id = len(st.session_state.sessions)-1
    st.session_state.sessions[session_id].in_progress = False
    st.session_state.sessions[session_id].show_chat_input = False

    # storage and cleanup
    if (Path.cwd()/f'{st.session_state.sessions[session_id].terraform_dir_name}').exists():
        shutil.rmtree(
            f'./{st.session_state.sessions[session_id].terraform_dir_name}')
        client = boto3.client('s3')
        client.put_object(Bucket=s3_bucket, Key=st.session_state.sessions[session_id].s3_key, Body=StringIO(
            st.session_state.sessions[session_id].chatbot.chain.memory.buffer).getvalue())


def validate_template_cb():
    if st.session_state.validate_template_prev:
        st.session_state.validate_template_prev = st.session_state.validate_template
        return
    st.session_state.validate_template_prev = st.session_state.validate_template
    if not st.session_state.validate_template or len(st.session_state.sessions) == 0:
        return
    cur_session = st.session_state.sessions[-1]
    if not cur_session.in_progress or not cur_session.terraform_template:
        return

    err = cur_session.try_validate()
    if err:
        cur_session.messages.append(
            {'role': 'assistant', 'content': VALIDATE_ERR_MSG})


def main():

    st.set_page_config(
        page_title="InfraBot", page_icon="👋", layout="wide"
    )
    st.sidebar.header('InfraBot')
    st.sidebar.write(
        'InfraBot is an AI-powered Terraform template builder. It generates and validates a terraform template for your provider of choice based on a conversation with you. It is built as an interface on top of ChatGPT.')
    if 'sessions' not in st.session_state:
        st.session_state.sessions = []

    if len(st.session_state.sessions) == 0:
        st.video(
            "https://ai-infra-demo-video-1.s3.amazonaws.com/streamlit-Home-2023-08-05-05-08.mp4")
    else:
        st.sidebar.video(
            "https://ai-infra-demo-video-1.s3.amazonaws.com/streamlit-Home-2023-08-05-05-08.mp4")

    with st.chat_message("assistant"):
        st.markdown(
            'Welcome to InfraBot! Your bespoke AI-powered Terraform IaaS builder!')
        st.markdown('To start a new session, press the \'Start session\' button in the sidebar. When you are satisfied with your template you can press \'End session\' ')

    st.sidebar.button('Start Session', key='start_session',
                      type="primary", on_click=start_session)
    st.sidebar.radio("provider", ["aws", "gcp"], key="provider")
    st.sidebar.checkbox("validate template", value=False,
                        key="validate_template", on_change=validate_template_cb)
    if 'validate_template_prev' not in st.session_state:
        st.session_state.validate_template_prev = False

    st.sidebar.button('End Session', key=end_session, on_click=end_session)
    
    download_terraform(os.getenv("TERRAFORM_VERSION"))

    session_tab_names = ["session " + str(s.id)
                         for s in st.session_state.sessions]
    session_tab_names.reverse()
    session_tabs = []
    if len(session_tab_names) > 0:
        session_tabs = st.tabs(session_tab_names)

    for i, t in enumerate(reversed(session_tabs)):
        st.session_state.sessions[i].tab = t
        st.session_state.sessions[i].render()


if __name__ == "__main__":
    main()
