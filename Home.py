import streamlit as st
import os
import openai
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from streamlit_chat import message
from langchain.callbacks import get_openai_callback
from dataclasses import dataclass
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

from streamlit_option_menu import option_menu

openai.api_key = os.getenv("OPENAI_API_KEY")
openai_model_id = os.getenv("OPENAI_MODEL_ID")
s3_bucket = os.getenv("S3_BUCKET")
terraform_version = os.getenv("TERRAFORM_VERSION")
display_options = ['template', 'llm notes']

MAX_TOKENS = 3500
MAX_ERROR_RETRIES = 3


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
                             temperature=self.temperature)
            self.chain = ConversationChain(
                llm=llm, memory=ConversationBufferMemory(), verbose=True)

        if self.start:
            # set by the radio button
            provider = st.session_state.provider
            st.session_state.chatbot.start = False
            prompt = f'For provider {provider} write a terraform template for {input}'
        else:
            # for subsequent chat messages, always use the base, not the finetuned llm
            if self.model_id != openai_model_id:
                self.model_id = openai_model_id
                self.chain = ConversationChain(llm=ChatOpenAI(
                    model_name=self.model_id, temperature=self.temperature), memory=self.chain.memory, verbose=True)
            prompt = f'For the above terraform template, {input}'

        with get_openai_callback() as cb:
            resp = self.chain.run(prompt)
            self.num_tokens_delta = cb.total_tokens - self.num_tokens
            self.num_tokens = cb.total_tokens
        return resp


def parse_llm_response(resp: str) -> tuple[str, str]:
    llm_notes = ''
    if 'hcl' in resp:
        fields = resp.split('```hcl')
        preface = fields[0]
        fields = fields[1].split('```')
        template = fields[0]
        if len(fields) >= 2:
            llm_notes = fields[1]
    else:
        fields = resp.split('```')
        preface = fields[0]
        template = fields[1]
        if len(fields) >= 3:
            llm_notes = fields[2]

    st.session_state.llm_notes.append(llm_notes)
    return (preface, template)


def get_terraform_template(spec: str) -> tuple[str, str]:

    with st.spinner('generating template'):
        preface, template = parse_llm_response(
            st.session_state.chatbot.response(spec))
    with st.spinner('validating template'):
        err, err_source = validate_template(template)

    num_retry = 0
    while err and num_retry < MAX_ERROR_RETRIES and st.session_state.chatbot.num_tokens < MAX_TOKENS:
        with st.spinner('generating template'):
            _, template = parse_llm_response(st.session_state.chatbot.response(
                f"while executing {err_source} there were these errors:\n{err}\n Fix the template by correcting these errors"))
        with st.spinner('validating template'):
            err, err_source = validate_template(template)
        num_retry += 1

    if err:
        preface = "I'm sorry the request has exceeded max tokens available. Here's is your most recent template."

    return (preface, template)


def validate_template(template: str) -> tuple[str, str]:

    # remove output directory for safety
    if (Path.cwd()/st.session_state.terraform_dir_name).exists():
        shutil.rmtree(f'./{st.session_state.terraform_dir_name}')

    os.mkdir(f'./{st.session_state.terraform_dir_name}')
    with open(f'./{st.session_state.terraform_dir_name}/main.tf', 'w') as f:
        f.write(template)
    os.chdir(f'./{st.session_state.terraform_dir_name}')

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


def terraform_description_callback(text_area_key: str):
    if len(st.session_state[text_area_key]) == 0:
        return
    st.session_state.messages.append(
        {'role': 'user', 'content': st.session_state[text_area_key]})

    if st.session_state.chatbot.num_tokens > MAX_TOKENS:
        st.session_state.messages.append(
            {'role': 'assistant', 'content': f'Exceeded the token limit of {MAX_TOKENS}. Please start a new session'})
        end_session()

    else:
        preface, st.session_state.terraform_template = get_terraform_template(
            st.session_state[text_area_key])
        st.session_state.messages.append(
            {'role': 'assistant', 'content': preface})
        st.session_state.messages.append({'role': 'terraform_options'})

    # hide the text box
    st.session_state.show_text_area = False


def fn_model_id_callback(key: str):
    if len(st.session_state[key]) == 0:
        return
    st.session_state.chatbot.model_id = st.session_state[key]


def start_session():
    st.session_state.messages.append(
        {'role': 'assistant', 'content': 'Please select a provider from the sidebar and provide your Terraform specification'})
    st.session_state.chatbot.start = True
    st.session_state.show_text_area = True
    st.session_state.show_fn_model_id_input = True
    st.session_state.show_provider_radio = True
    st.session_state.llm_notes = []
    uuid_str = uuid.uuid4()
    st.session_state.terraform_dir_name = f'terraform_{uuid_str}'
    time_str = datetime.datetime.now().strftime("%Y/%m/%d/%H")
    st.s3_key = f'{time_str}/{uuid_str}'


def end_session():
    st.session_state.show_text_area = False

    # storage and cleanup
    shutil.rmtree(f'./{st.session_state.terraform_dir_name}')
    client = boto3.client('s3')
    client.put_object(Bucket=s3_bucket, Key=st.s3_key, Body=StringIO(st.session_state.chatbot.chain.memory.buffer).getvalue())


def terraform_options_callback(key: str):
    for o in display_options:
        if st.session_state[key] == o:
            st.session_state[o] = True
        else:
            st.session_state[o] = False


def render_messages():
    for i, m in enumerate(st.session_state.messages):
        if m['role'] == 'terraform_options':
            option_menu(None, display_options,
                        on_change=terraform_options_callback, key=f'{i}_terraform_options', orientation="horizontal")
        else:
            message(m['content'], is_user=True if m['role']
                    == 'user' else False, key=f'{i}_chat')


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
        page_title="InfraBot",
        page_icon="👋",
        layout="wide"
    )
    download_terraform()

    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {'role': 'assistant',
             'content': 'Welcome to InfraBot! Your bespoke AI-powered Terraform IaaS builder!'},
            {'role': 'assistant', 'content': 'To start a new session, press the \'Start session\' button in the sidebar. When you are satisfied with your template you can press \'End session\' '}]

    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = Chatbot(openai_model_id, 0)
    if 'show_text_area' not in st.session_state:
        st.session_state.show_text_area = False
    if 'show_fn_model_id_input' not in st.session_state:
        st.session_state.show_fn_model_id_input = False
    if 'show_provider_radio' not in st.session_state:
        st.session_state.show_provider_radio = False

    st.sidebar.button('Start session', key='start_session',
                      type="primary", on_click=start_session)
    st.sidebar.metric(label=f'Number of tokens used out of a max of {MAX_TOKENS}', value=st.session_state.chatbot.num_tokens,
                      delta=st.session_state.chatbot.num_tokens_delta)
    st.sidebar.button('End session', key='end_session', on_click=end_session)
    # st.sidebar.progress(int(
    #    100*st.session_state.chatbot.num_tokens/MAX_TOKENS), text=f'% of {MAX_TOKENS} tokens used')

    if st.session_state.show_provider_radio:
        st.sidebar.radio("provider", ["aws", "gcp"],
                         key="provider", disabled=True)

    # if st.session_state.show_fn_model_id_input:
    #    key = 'openai_fn_model_id'
    #    st.sidebar.text_input('model id', key=key,
    #                          on_change=fn_model_id_callback, args=(key,))

    for o in display_options:
        if o not in st.session_state:
            st.session_state[o] = False

    render_messages()
    if st.session_state['template']:
        st.code(st.session_state.terraform_template,
                language="hcl", line_numbers=True)
        st.session_state.show_text_area = True
    elif st.session_state['llm notes']:
        st.text('\n'.join(st.session_state.llm_notes))
        st.session_state.show_text_area = True

    if st.session_state.show_text_area:
        with st.empty():
            key = 'terraform_description'
            st.text_area('IaC', key=key, on_change=terraform_description_callback, args=(
                key,), label_visibility="hidden")


if __name__ == "__main__":
    main()
