import enum
from langchain.callbacks import get_openai_callback
from dataclasses import dataclass
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from pydantic import BaseModel, Field
from langchain.prompts.prompt import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
import streamlit as st
MAX_TOKENS = 3500
MAX_ERROR_RETRIES = 2
MAX_QUESTIONS = 5

VALIDATE_ERR_MSG = "I'm sorry attempts to validate the template has resulted in the request exceeding max tokens available"


class Question(BaseModel):
    topic: str = Field(
        description="The topic of the template about which the question is asked")
    question: str = Field(description="The actual question")
    possible_answers: list[str] = Field(
        description="The list of possible answers for the question")

    def __str__(self) -> str:
        s = self.question + "\n"
        for i, a in enumerate(self.possible_answers):
            s += f'{i+1}) {a}\n'
        return s


class TemplateSpecQuestionsParams(BaseModel):
    """get clarifying questions for the terraform template"""
    questions: list[Question] = Field(
        description="List of clarifying questions about what the user wants")


class StreamlitStreamHandler(BaseCallbackHandler):
    def __init__(self):
        self.text = ""
        self.container = st.empty()

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

class NoopStreamHandler(BaseCallbackHandler):
    pass


class Chatbot:
    def __init__(self, model_id: str, temperature: float, stream_handler_class: any) -> None:
        self.model_id = model_id
        self.temperature = temperature
        self.start = True
        self.num_tokens = 0
        self.num_tokens_delta = 0
        self.stream_handler_class = stream_handler_class


class OpenAIChatbot(Chatbot):

    def __init__(self, model_id: str, temperature: float, stream_handler_class: any) -> None:
        super().__init__(model_id=model_id, temperature=temperature,
                         stream_handler_class=stream_handler_class)
        self.llm = ChatOpenAI(model_name=self.model_id,
                              temperature=self.temperature, streaming=True)
        self.llm_with_functions = ChatOpenAI(
            model_name=self.model_id, temperature=self.temperature).bind(functions=[convert_pydantic_to_openai_function(TemplateSpecQuestionsParams)])
        self.spec_chain = ChatPromptTemplate.from_messages(
            [("user", "{prompt}")]) | self.llm_with_functions | JsonOutputFunctionsParser()

        template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context.
                    Current conversation: {history}
                    Human: {input}
                    AI:"""
        PROMPT = PromptTemplate(
            input_variables=["history", "input"], template=template)
        self.chain = ConversationChain(
            prompt=PROMPT, llm=self.llm, memory=ConversationBufferMemory())

    def spec_gathering_response(self, user_input: str) -> list[Question]:
        prompt = f"I want to create {user_input}. Give me a terraform template. But before doing so ask me at most {MAX_QUESTIONS} clarifying questions. Each question should have a fixed number of possible answers."
        resp = self.spec_chain.invoke({"prompt": prompt})
        return TemplateSpecQuestionsParams(**resp).questions

        """
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=self.model_id,
            messages=messages,
            functions=[
                {
                    "name": "get_clarifying_questions",
                    "description": "get clarifying questions for the terraform template",
                    "parameters": TemplateSpecQuestionsParams.schema()
                }
            ],
        )
        arguments = json.loads(
            response.choices[0].message["function_call"]["arguments"])
        return TemplateSpecQuestionsParams(**arguments).questions
        """

    def response(self, prompt: str) -> str:
        with get_openai_callback() as cb:
            # for every response, we create a new stream handler; if not, response would use the old container
            self.chain.llm.callbacks = [self.stream_handler_class()]
            resp = self.chain.run(prompt)
            self.num_tokens_delta = cb.total_tokens - self.num_tokens
            self.num_tokens = cb.total_tokens
        return resp
