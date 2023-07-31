import streamlit as st
import os
import openai
from langchain.chat_models import ChatOpenAI
import streamlit as st
import subprocess
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from streamlit_chat import message
from streamlit_option_menu import option_menu
import shutil
from pathlib import Path
from PIL import Image

# from modules.chatbot import Chatbot


class Chatbot:
    def __init__(self, model_name, temperature):
        self.model_name = model_name
        self.temperature = temperature
        self.llm = ChatOpenAI(model_name=self.model_name,
                              temperature=self.temperature)
        self.chain = ConversationChain(
            llm=self.llm, memory=ConversationBufferMemory(), verbose=True)
        self.start = True
        self.cur_iac = 'cfn'

    def response(self, input: str):
        if self.start:
            if self.cur_iac == 'cfn':
                return self.chain.run(f'Write a cloudformation template in yaml for {input}')
            elif self.cur_iac == 'terraform':
                return self.chain.run(f'Write a valid terraform template for {input}')
        else:
            # modify
            return self.chain.run(f'For the above template, {input}')


openai.api_key = os.getenv("OPENAI_API_KEY")
openai_model_id = os.getenv("OPENAI_MODEL_ID")
options = ['template', 'diagram', 'llm notes']


def get_iac_template(spec: str):
    with st.spinner('generating cfn template'):
        response = st.session_state.chatbot.response(spec)
        if st.session_state.chatbot.cur_iac == 'cfn':
            if 'yaml' in response:
                fields = response.split('```yaml')
            else:
                fields = response.split('```')
        elif st.session_state.chatbot.cur_iac == 'terraform':
            if 'hcl' in response:
                fields = response.split('```hcl')
            else:
                fields = response.split('```')
        preface = fields[0]
        fields = fields[1].split('```')
        template = fields[0]
        llm_notes = ''
        if len(fields) >= 2:
            llm_notes = fields[1]
        return (preface, template, llm_notes)


def get_cfn_visualization():
    with open('template.yaml', 'w') as f:
        f.write(st.session_state.iac_template)

    # remove output directory for safety
    if (Path.cwd()/'cfn-diagram').exists():
        shutil.rmtree('./cfn-diagram')

    subprocess.run(['cfn-dia', 'h', '-t', 'template.yaml', '-o', './cfn-diagram', '-c', '-sa'],
                   stdout=subprocess.PIPE,
                   universal_newlines=True)

    with open('cfn-diagram/index.html', 'r') as f:
        st.session_state.template_vis = f.read()


def get_terraform_visualization():
    with open('template.tf', 'w') as f:
        f.write(st.session_state.iac_template)

    subprocess.run(['terraform', 'graph', '|', 'dot', '-Tpng', '>>', 'graph.png'],
                   stdout=subprocess.PIPE,
                   universal_newlines=True)


def iac_description_callback():
    # key of the user text box
    key = 'iac_description_' + str(st.session_state.num_sessions)
    if len(st.session_state[key]) == 0:
        return
    st.session_state.messages.append(
        {'role': 'user', 'content': st.session_state[key]})

    preface, st.session_state.iac_template, st.session_state.llm_notes = get_iac_template(
        st.session_state[key])
    st.session_state.messages.append({'role': 'assistant', 'content': preface})
    st.session_state.messages.append({'role': 'iac_template_diagram'})

    if st.session_state.chatbot.start:
        st.session_state.chatbot.start = False

    # hide the text box
    st.session_state.show_text_area = False

    if st.session_state.chatbot.cur_iac == 'cfn':
        get_cfn_visualization()
    elif st.session_state.chatbot.cur_iac == 'terraform':
        get_terraform_visualization()


def iac_tool_options_callback():
    st.session_state.chatbot.cur_iac = st.session_state.iac_tool_options


def start_session():
    st.session_state.messages.append(
        {'role': 'assistant', 'content': 'Please give me your template specification'})
    st.session_state.chatbot.start = True
    st.session_state.session_started = True
    st.session_state.show_text_area = True
    st.session_state.num_sessions += 1


def iac_template_diagram_callback(key: str):
    for o in options:
        if st.session_state[key] == o:
            st.session_state[o] = True
        else:
            st.session_state[o] = False


def render_messages():
    for i, m in enumerate(st.session_state.messages):
        if m['role'] == 'iac_template_diagram':
            option_menu(None, options,
                        on_change=iac_template_diagram_callback, key=f'{i}_iac_template_diagram_options', orientation="horizontal")
        else:
            message(m['content'], is_user=True if m['role']
                    == 'user' else False, key=f'{i}_chat')


def main():

    st.set_page_config(
        page_title="InfraBot",
        page_icon="👋",
        layout="wide"
    )

    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {'role': 'assistant',
             'content': 'Welcome to InfraBot! Your bespoke AI-powered IaaS builder!'},
            {'role': 'assistant', 'content': 'To start a new session, press the \'Start session\' button in the sidebar.'}]

    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = Chatbot(openai_model_id, 0)
    if 'show_text_area' not in st.session_state:
        st.session_state.show_text_area = False
    if 'num_sessions' not in st.session_state:
        st.session_state.num_sessions = 0
    if 'session_started' not in st.session_state:
        st.session_state.session_started = False

    st.sidebar.button('Start session', key='start_session',
                      type="primary", on_click=start_session)

    if st.session_state.session_started:
        st.sidebar.selectbox('IaC tool', ['', 'cfn', 'terraform'], key='iac_tool_options',
                             on_change=iac_tool_options_callback, label_visibility="hidden")

    for o in options:
        if o not in st.session_state:
            st.session_state[o] = False

    render_messages()
    if st.session_state['template']:
        if st.session_state.chatbot.cur_iac == 'cfn':
            st.code(st.session_state.iac_template,
                    language="yaml", line_numbers=True)
        elif st.session_state.chatbot.cur_iac == 'terraform':
            st.code(st.session_state.iac_template,
                    language="hcl", line_numbers=True)
        st.session_state.show_text_area = True
    elif st.session_state['diagram']:
        if st.session_state.chatbot.cur_iac == 'cfn':
            st.components.v1.html(
                st.session_state.template_vis, height=600, scrolling=True)
        elif st.session_state.chatbot.cur_iac == 'terraform':
            st.image(Image.open('graph.png'))
        st.session_state.show_text_area = True
    elif st.session_state['llm notes']:
        st.text(st.session_state.llm_notes)
        st.session_state.show_text_area = True

    if st.session_state.show_text_area:
        with st.empty():
            st.text_area('iac', key='iac_description_' + str(st.session_state.num_sessions),
                         on_change=iac_description_callback, label_visibility="hidden")


if __name__ == "__main__":
    main()
