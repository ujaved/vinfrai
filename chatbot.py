from langchain.callbacks import get_openai_callback, FileCallbackHandler
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
from langchain.llms.base import LLM
import os
import replicate
from utils import logger
from langchain.llms import Replicate

MAX_TOKENS = 3500
MAX_ERROR_RETRIES = 3
MAX_QUESTIONS = 5

VALIDATE_ERR_MSG = "I'm sorry attempts to validate the template has resulted in the request exceeding max tokens available"
PROMPT_TEMPLATE = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context.
                    Current conversation: {history}
                    Human: {input}
                    AI:"""

SPEC_TEMPLATE = """I want to create {user_input}. Give me a terraform template. But before doing so ask me at most {MAX_QUESTIONS} clarifying questions. No question should ask about resource naming. Each question should have a fixed number of possible answers."""

REPLICATE_CODE_LLAMA_ENDPOINT = "meta/codellama-34b-instruct:b17fdb44c843000741367ae3d73e2bb710d7428a662238ddebbf4302db2b5422"


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


class LlamaLLM(LLM):

    @property
    def _llm_type(self) -> str:
        return "llama"

    # this is the implemented langchain method so that the chain's invoke method works
    def _call(self, prompt: str, stop=None) -> str:
        resp = ""
        container = st.empty()
        output = replicate.run(REPLICATE_CODE_LLAMA_ENDPOINT, input={
                               "prompt": prompt, "max_tokens": 4000})
        for token in output:
            resp += token
            container.markdown(resp)
        return resp


class Chatbot:
    def __init__(self, model_id: str, temperature: float, stream_handler_class: any) -> None:
        self.model_id = model_id
        self.temperature = temperature
        self.start = True
        self.num_tokens = 0
        self.num_tokens_delta = 0
        self.stream_handler_class = stream_handler_class

    def response(self, prompt: str) -> str:
        raise NotImplementedError

    def spec_gathering_response(self, user_input: str) -> list[Question]:
        raise NotImplementedError


class OpenAIChatbot(Chatbot):

    def __init__(self, model_id: str, temperature: float, stream_handler_class: any) -> None:
        super().__init__(model_id=model_id, temperature=temperature,
                         stream_handler_class=stream_handler_class)
        self.llm = ChatOpenAI(model_name=self.model_id,
                              temperature=self.temperature, streaming=True)
        PROMPT = PromptTemplate(
            input_variables=["history", "input"], template=PROMPT_TEMPLATE)
        self.chain = ConversationChain(prompt=PROMPT, llm=self.llm, callbacks=[
                                       FileCallbackHandler(os.getenv("LOGFILE"))], memory=ConversationBufferMemory())

        self.llm_with_functions = ChatOpenAI(model_name=self.model_id, temperature=self.temperature).bind(
            functions=[convert_pydantic_to_openai_function(TemplateSpecQuestionsParams)])
        self.spec_chain = ChatPromptTemplate.from_messages(
            [("user", "{prompt}")]) | self.llm_with_functions | JsonOutputFunctionsParser()

    def spec_gathering_response(self, user_input: str) -> list[Question]:
        resp = self.spec_chain.invoke({"prompt": SPEC_TEMPLATE.format(
            user_input=user_input, MAX_QUESTIONS=MAX_QUESTIONS)})
        logger.info(resp)
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
        logger.info(resp)
        return resp


class LLamaChatbot(OpenAIChatbot):

    def __init__(self, temperature: float, stream_handler_class: any) -> None:
        # initialize openai chatbot for q_a prompt
        super().__init__(model_id=os.getenv("OPENAI_MODEL_ID"),
                         temperature=temperature, stream_handler_class=stream_handler_class)
        self.llm = Replicate(streaming=True, model=REPLICATE_CODE_LLAMA_ENDPOINT, model_kwargs={
                             "temperature": temperature, "max_length": MAX_TOKENS})
        self.chain.llm = self.llm

    def response(self, prompt: str) -> str:
        self.chain.llm.callbacks = [self.stream_handler_class()]
        resp = self.chain.run(prompt)
        logger.info(resp)
        return resp
