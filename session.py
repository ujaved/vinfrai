from dataclasses import dataclass, field

import uuid
from typing import Callable
import datetime

from chatbot import MAX_TOKENS, MAX_QUESTIONS, VALIDATE_ERR_MSG, Chatbot, MAX_ERROR_RETRIES, Question, OpenAIChatbot, StreamlitStreamHandler, LLamaChatbot
import streamlit as st
import os
from utils.llm import parse_llm_response, parse_llm_response_mult_lang
from utils.terraform import validate_terratest
from contextlib import AbstractContextManager, contextmanager
from utils.terraform import validate_template
from pathlib import Path
import shutil
import boto3
from io import StringIO
from yaspin import yaspin


PROMPT_FROM_QA_PREFIX = """For provider aws give me a terraform template for {initial_spec} with the following additional specifications. \n\n"""
PROMPT_FROM_QA_SUFFIX = """\n Create random names for resources. Create output variables. \n"""
AI_STARTER_MSG = "Certainly! Let's start with a few questions that will help me generate your initial template."

s3_bucket = os.getenv("S3_BUCKET")


@contextmanager
def st_chat_cm(**kwargs):
    try:
        with kwargs['container']:
            yield st.chat_message('assistant')
    finally:
        pass


def start_session():
    # end the previous session
    end_session()

    id = len(st.session_state.sessions)
    session = StreamlitSession(id=id)
    if st.session_state.llm == "codellama":
        session.chatbot = LLamaChatbot()
    st.session_state.sessions.append(session)


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


def user_question_radio_cb(key: str):
    cur_session = st.session_state.sessions[-1]
    q = cur_session.user_q_a[cur_session.user_q_id_to_display][0]
    cur_session.user_q_a[cur_session.user_q_id_to_display] = (
        q, st.session_state[key])
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
            session.messages.append(
                {'role': 'assistant', 'content': AI_STARTER_MSG})
            with session.tab:
                with st.chat_message('assistant'):
                    st.markdown(AI_STARTER_MSG)
            with st.spinner('generating clarifying questions'):
                session.user_q_a = [(q, '')
                                    for q in chatbot.spec_gathering_response(spec)]
                session.messages.append({'role': 'user_questions'})
        else:
            session.get_terraform_template(
                spec=spec, validate=st.session_state.validate_template, **{"container": session.tab})

    session.show_chat_input = False


# session is used both in streamlit ui and cli
@dataclass
class Session:
    id: int

    validate_ctx: Callable[..., AbstractContextManager]
    generate_ctx: Callable[..., AbstractContextManager]

    chatbot: Chatbot

    messages: list[str] = field(default_factory=lambda: [
                                {'role': 'assistant', 'content': 'Please select an llm from the sidebar and provide your Terraform specification'}])
    in_progress: bool = True
    terraform_template: str = ''
    llm_notes: list[str] = field(default_factory=list)
    uuid_str: uuid = uuid.uuid4()
    terraform_dir_name: str = f'terraform_{uuid_str}'
    time_str: str = datetime.datetime.now().strftime("%Y/%m/%d/%H")
    s3_key: str = f'{time_str}/{uuid_str}'
    initial_spec: str = ''
    user_q_a: list[tuple[Question, str]] = field(default_factory=list)
    user_q_id_to_display: int = 0
    terratest: bool = False

    def create_prompt_from_qa(self) -> str:
        prompt = PROMPT_FROM_QA_PREFIX.format(initial_spec=self.initial_spec)
        for q in self.user_q_a:
            prompt += f'{q[0].question} answer: {q[1]}\n'
        prompt += PROMPT_FROM_QA_SUFFIX
        if self.terratest:
            prompt += 'Generate a terratest file for the template that tests important features of the template. For terraform.Options the value of TerraformDir should be "../" \n'
        return prompt

    def try_validate(self, **kwargs) -> str:
        with self.validate_ctx(text='performing static validation of template'):
            err, err_source = validate_template(
                self.terraform_template, self.terraform_dir_name)

        num_retry = 0
        while err and num_retry < MAX_ERROR_RETRIES and self.chatbot.num_tokens < MAX_TOKENS:
            with self.generate_ctx(**kwargs):
                response = self.chatbot.response(
                    f'while executing {err_source} there were these errors:\n{err}\n Fix the template by correcting these errors')
                _, self.terraform_template = parse_llm_response(
                    response, self.llm_notes)
            with self.validate_ctx(text='performing static validation of template'):
                err, err_source = validate_template(
                    self.terraform_template, self.terraform_dir_name)
            num_retry += 1
        return err

    def get_terraform_template(self, spec: str, validate: bool, **kwargs) -> None:
        with self.generate_ctx(**kwargs):
            # chatbot response will write streaming responses in an empty container inside the chat_message container
            response = self.chatbot.response(spec)
        preface, self.terraform_template = parse_llm_response(
            response, self.llm_notes)
        preface = response

        if validate:
            err = self.try_validate(**kwargs)
            if err:
                preface = VALIDATE_ERR_MSG

        self.messages.append({'role': 'assistant', 'content': preface})


class CliSession(Session):

    def __init__(self, id: int, validate_ctx, generate_ctx, chatbot, terratest: bool):
        super().__init__(id=id, validate_ctx=validate_ctx,
                         generate_ctx=generate_ctx, chatbot=chatbot)
        self.terratest = terratest
        self.terratest_dir_name = f'terratest_{self.uuid_str}'

    def get_terraform_template(self, spec: str, validate: bool, **kwargs) -> None:
        with self.generate_ctx(**kwargs):
            response = self.chatbot.response(spec)
            if response.isdigit():
                raise Exception(
                    "error code from server: {code}".format(code=response))
            if self.terratest:
                parsed = parse_llm_response_mult_lang(
                    resp=response, langs=['hcl', 'go'])
                self.terraform_template = parsed[0]
                self.terratest_go = parsed[1]
            else:
                _, self.terraform_template = parse_llm_response(
                    response, self.llm_notes)

        err = ''
        if validate or self.terratest:
            err = self.try_validate(**kwargs)
        if self.terratest and not err:
            err = self.try_terratest_validate(**kwargs)
        if err:
            print(VALIDATE_ERR_MSG)

    def try_terratest_validate(self, **kwargs) -> str:
        with self.validate_ctx(text='validating template with terratest'):
            err = validate_terratest(
                go_code=self.terratest_go, terraform_dir_name=self.terraform_dir_name, terratest_dir_name=self.terratest_dir_name)

        num_retry = 0
        while err and num_retry < MAX_ERROR_RETRIES:
            with self.generate_ctx(text='regenerating terratest code'):
                response = self.chatbot.response(
                    f'while compiling the go test code there were this error:\n{err}\n Fix the test code template by correcting the error')
                parsed = parse_llm_response_mult_lang(
                    resp=response, langs=['go'])
                self.terratest_go = parsed[0]
            with self.validate_ctx(text='validating template with terratest'):
                err = validate_terratest(
                    go_code=self.terratest_go, terraform_dir_name=self.terraform_dir_name, terratest_dir_name=self.terratest_dir_name)
            num_retry += 1

        return err


class StreamlitSession(Session):
    def __init__(self, id: int):
        super().__init__(id=id)

        self.validate_ctx = st.spinner
        self.generate_ctx = st_chat_cm
        self.show_chat_input: str = True
        self.chatbot = OpenAIChatbot(model_id=os.getenv(
            "OPENAI_MODEL_ID"), temperature=0, stream_handler_class=StreamlitStreamHandler)
        self.tab = None

    def render_question_radios(self) -> bool:
        for i in range(self.user_q_id_to_display+1):
            if i == MAX_QUESTIONS:
                if not self.terraform_template:
                    self.get_terraform_template(spec=self.create_prompt_from_qa(), validate=st.session_state.validate_template, **{"container": self.tab})
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
                    # force rerun for codellama to render chat messages
                    if st.session_state.llm == 'codellama':
                        st.rerun()
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
