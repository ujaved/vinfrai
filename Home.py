from xml.dom import VALIDATION_ERR
import streamlit as st
import os
import openai
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.callbacks import OpenAICallbackHandler
from langchain.callbacks import get_openai_callback
from dataclasses import dataclass, field
from pathlib import Path
import shutil
import subprocess
import uuid
import platform
import urllib.request
import zipfile
import stat
import datetime
import boto3
from io import StringIO
from langchain.prompts.prompt import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler

from streamlit_option_menu import option_menu

openai.api_key = os.getenv("OPENAI_API_KEY")
openai_model_id = os.getenv("OPENAI_MODEL_ID")
s3_bucket = os.getenv("S3_BUCKET")
terraform_version = os.getenv("TERRAFORM_VERSION")
display_options = ['template', 'llm notes']

MAX_TOKENS = 3500
MAX_ERROR_RETRIES = 2

VALIDATE_ERR_MSG = "I'm sorry attempts to validate the template has resulted in the request exceeding max tokens available"

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


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


def validate_template(template: str, terraform_dir_name: str) -> tuple[str, str]:

    # remove output directory for safety
    if (Path.cwd()/terraform_dir_name).exists():
        shutil.rmtree(f'./{terraform_dir_name}')

    os.mkdir(f'./{terraform_dir_name}')
    with open(f'./{terraform_dir_name}/main.tf', 'w') as f:
        f.write(template)
    os.chdir(f'./{terraform_dir_name}')

    result = subprocess.run(['../terraform', 'init'],
                            capture_output=True, text=True)

    # only run plan if init has no errors
    err_source = "terraform plan"
    if len(result.stderr) == 0:
        result = subprocess.run(['../terraform', 'plan'],
                                capture_output=True, text=True)
    else:
        err_source = "terraform init"

    os.chdir('../')
    return (result.stderr, err_source)


@dataclass
class Chatbot:
    model_id: str
    temperature: float
    start: bool = True
    chain: any = None
    num_tokens: int = 0
    num_tokens_delta: int = 0

    def response(self, input: str) -> str:
        if self.chain is None:
            llm = ChatOpenAI(model_name=self.model_id,
                             temperature=self.temperature, streaming=True)
            template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context.
                          Current conversation: {history}
                          Human: {input}
                          AI:"""
            PROMPT = PromptTemplate(
                input_variables=["history", "input"], template=template)
            self.chain = ConversationChain(
                prompt=PROMPT, llm=llm, memory=ConversationBufferMemory(), verbose=True)

        if self.start:
            self.start = False
            prompt = f'For provider {st.session_state.provider} give me a terraform template for {input}'
            if st.session_state.provider == 'gcp':
                prompt += 'In the "provider" block do not include the "credentials" and "project" properties'
        else:
            prompt = f'For the above terraform template, {input}'

        with get_openai_callback() as cb:
            # for every response, we create a new stream handler; if not, response would use the old container
            self.chain.llm.callbacks = [StreamHandler(st.empty())]
            resp = self.chain.run(prompt)
            self.num_tokens_delta = cb.total_tokens - self.num_tokens
            self.num_tokens = cb.total_tokens
        return resp


@dataclass
class session:
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
    chatbot: Chatbot = Chatbot(openai_model_id, 0)
    tab = None

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

    def render_messages(self):
        for m in self.messages:
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
            st.chat_input('user input', key=key, on_submit=terraform_description_callback, args=(key,))


def terraform_description_callback(chat_input_key: str):
    spec = st.session_state[chat_input_key]
    if len(spec) == 0:
        return
    session_id = int(chat_input_key.split("_")[0])
    with st.session_state.sessions[session_id].tab:
        with st.chat_message('user'):
            st.markdown(spec)
    st.session_state.sessions[session_id].messages.append(
        {'role': 'user', 'content': spec})
    chatbot = st.session_state.sessions[session_id].chatbot

    if chatbot.num_tokens > MAX_TOKENS:
        st.session_state.sessions[session_id].messages.append(
            {'role': 'assistant', 'content': f'Exceeded the token limit of {MAX_TOKENS}. Please start a new session'})
        end_session()
    else:
        st.session_state.sessions[session_id].get_terraform_template(spec)

    st.session_state.sessions[session_id].show_chat_input = False


def start_session():
    # end the previous session
    end_session()

    id = len(st.session_state.sessions)
    st.session_state.sessions.append(session(id=id))


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


def download_terraform():
    if (Path.cwd()/'terraform').exists():
        return
    platform_name = platform.system().lower()
    base_url = f"https://releases.hashicorp.com/terraform/{terraform_version}"
    zip_file = f"terraform_{terraform_version}_{platform_name}_amd64.zip"
    download_url = f"{base_url}/{zip_file}"

    urllib.request.urlretrieve(download_url, zip_file)

    with zipfile.ZipFile(zip_file) as terraform_zip_archive:
        terraform_zip_archive.extractall('.')

    os.remove(zip_file)
    executable_path = './terraform'
    executable_stat = os.stat(executable_path)
    os.chmod(executable_path, executable_stat.st_mode | stat.S_IEXEC)


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
        cur_session.messages.append({'role': 'assistant', 'content': VALIDATE_ERR_MSG})


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
    download_terraform()

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
