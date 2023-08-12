import streamlit as st
import os
import openai
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from streamlit_chat import message
from langchain.callbacks import get_openai_callback
from langchain.callbacks.base import BaseCallbackHandler
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
from typing import ClassVar, Dict, Any

from streamlit_option_menu import option_menu

openai.api_key = os.getenv("OPENAI_API_KEY")
openai_model_id = os.getenv("OPENAI_MODEL_ID")
s3_bucket = os.getenv("S3_BUCKET")
terraform_version = os.getenv("TERRAFORM_VERSION")
display_options = ['template', 'llm notes']

MAX_TOKENS = 3500
MAX_ERROR_RETRIES = 3
DEMO_VIDEO_URL = "https://ai-infra-demo-video.s3.us-west-2.amazonaws.com/demo-video"


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
class Session:
    # id is supposed to be the index in an array of sessions
    id: int
    messages: list[str] = field(default_factory=lambda: [
                                {'role': 'assistant', 'content': 'Please select a provider from the sidebar and provide your Terraform specification'}])
    in_progress: bool = True
    chat_start: bool = True
    uuid_str: uuid = uuid.uuid4()
    terraform_dir_name: str = f'terraform_{uuid_str}'
    time_str: str = datetime.datetime.now().strftime("%Y/%m/%d/%H")
    s3_key: str = f'{time_str}/{uuid_str}'
    show_text_area: str = True
    chain: Any = field(init=False)
    tab: Any = None
    num_tokens: int = 0
    num_tokens_delta: int = 0

    def __post_init__(self):
        llm = ChatOpenAI(model_name=openai_model_id,
                         streaming=True, temperature=0)
        template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context.
                          Current conversation: {history}
                          Human: {input}
                          AI:"""
        PROMPT = PromptTemplate(
            input_variables=["history", "input"], template=template)
        self.chain = ConversationChain(
            prompt=PROMPT, llm=llm, memory=ConversationBufferMemory(), verbose=True)

    def get_terraform_template(self, spec: str):
        '''
        self.chatbot.response(self.stream_handler, spec)
        err, err_source = validate_template(
            self.terraform_template, self.terraform_dir_name)

        num_retry = 0
        while err and num_retry < MAX_ERROR_RETRIES and self.chatbot.num_tokens < MAX_TOKENS:
            with st.spinner('generating template'):
                _, self.terraform_template = parse_llm_response(self.chatbot.response(
                    f'while executing {err_source} there were these errors:\n{err}\n Fix the template by correcting these errors'), self.llm_notes)
            with st.spinner('validating template'):
                err, err_source = validate_template(
                    self.terraform_template, self.terraform_dir_name)
            num_retry += 1

        if err:
            preface = "I'm sorry attempts to validate the template has resulted in the request exceeding max tokens available. Here is your most recent template."
        self.add_message({'role': 'assistant', 'content': preface})
        '''

    def add_message(self, m: dict):
        self.messages.append(m)

    def render_messages(self):
        for i, m in enumerate(self.messages):
            message(m['content'], is_user=True if m['role'] ==
                    'user' else False, key=f'{self.id}_{i}_chat')

    def render(self):
        with self.tab:
            st.metric(
                label=f'Number of tokens used out of a max of {MAX_TOKENS}', value=self.num_tokens, delta=self.num_tokens_delta)

            self.render_messages()
            if self.in_progress:
                self.show_text_area = True

            if self.show_text_area:
                key = f'{self.id}_terraform_description'
                st.text_area('IaC', key=key, on_change=terraform_description_callback, args=(
                    key,), label_visibility="hidden")


@dataclass
class StreamHandler(BaseCallbackHandler):
    buf_length: ClassVar[str] = 10
    session: Session
    dynamic_template_container: Any

    preface: str = ""
    template: str = ""
    # this buffer is necessary to gather enough text to make any sense of the parsing
    buffer: list[str] = field(default_factory=list)

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.buffer.append(token)
        if len(self.buffer) < StreamHandler.buf_length:
            return

        text = ''.join(self.buffer)
        self.buffer = []

        preface_just_finished = False
        if 'hcl' in text:
            fields = text.split('hcl')
            self.preface += fields[0]
            preface_just_finished = True
            self.template += fields[1]
        elif '```' in text:
            fields = text.split('```')
            if len(self.template) > 0:
                self.template += fields[0]
            else:
                self.preface += fields[0]
                preface_just_finished = True
                self.template += fields[1]
        else:
            if len(self.template) > 0:
                self.template += text
            else:
                self.preface += text
        if preface_just_finished:
            self.session.add_message(
                {'role': 'assistant', 'content': self.preface})
            st.experimental_rerun()

        if len(self.template) > 0:
            with self.session.tab:
                self.dynamic_template_container.code(
                    self.template, language="hcl", line_numbers=True)


def chat_response(session: Session, input: str):
    if session.chat_start:
        session.chat_start = False
        prompt = f'For provider {st.session_state.provider} give me a terraform template for {input}'
    else:
        prompt = f'For the above terraform template, {input}'

    stream_handler = StreamHandler(
        session=session, dynamic_template_container=st.empty())
    
    # run this async
    with get_openai_callback() as cb:
        session.chain.run(prompt, callbacks=[stream_handler])
        session.num_tokens_delta = cb.total_tokens - session.num_tokens
        session.num_tokens = cb.total_tokens


def terraform_description_callback(text_area_key: str):
    spec = st.session_state[text_area_key]
    if len(spec) == 0:
        return
    session = st.session_state.sessions[int(text_area_key.split("_")[0])]
    session.add_message({'role': 'user', 'content': spec})

    if session.num_tokens > MAX_TOKENS:
        session.add_message(
            {'role': 'assistant', 'content': f'Exceeded the token limit of {MAX_TOKENS}. Please start a new session'})
        end_session()
    else:
        chat_response(session, spec)

    # hide the text box
    session.show_text_area = False


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
    st.session_state.sessions[session_id].show_text_area = False

    # storage and cleanup
    if (Path.cwd()/f'{st.session_state.sessions[session_id].terraform_dir_name}').exists():
        shutil.rmtree(
            f'./{st.session_state.sessions[session_id].terraform_dir_name}')
        client = boto3.client('s3')
        client.put_object(Bucket=s3_bucket, Key=st.session_state.sessions[session_id].s3_key, Body=StringIO(
            st.session_state.sessions[session_id].chain.memory.buffer).getvalue())


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


def main():

    st.set_page_config(
        page_title="InfraBot", page_icon="👋", layout="wide"
    )
    st.sidebar.header('InfraBot')
    st.sidebar.write(
        'InfraBot is an AI-powered Terraform template builder. It generates and validates a terraform template for your provider of choice based on a conversation with you. It is built as an interface on top of ChatGPT.')
    if 'sessions' not in st.session_state:
        st.session_state.sessions = []
        st.video(DEMO_VIDEO_URL)
    else:
        st.sidebar.video(DEMO_VIDEO_URL)
    download_terraform()
    message('Welcome to InfraBot! Your bespoke AI-powered Terraform IaaS builder!')
    message('To start a new session, press the \'Start session\' button in the sidebar. When you are satisfied with your template you can press \'End session\' ')

    st.sidebar.button('Start Session', key='start_session',
                      type="primary", on_click=start_session)
    st.sidebar.radio("provider", ["aws", "gcp"], key="provider", disabled=True)
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
