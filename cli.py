import argparse
from utils.terraform import download_terraform
from Home import WELCOME_MSG, AI_STARTER_MSG, Session
from prompt_toolkit import PromptSession
from prompt_toolkit.lexers import PygmentsLexer
from pygments.lexers.sql import SqlLexer
import os
from yaspin import yaspin
from chatbot import OpenAIChatbot, LLamaChatbot, NoopStreamHandler
from collections import Counter


def get_qa_spec_prompt(user_spec: str, chat_session: Session, prompt_session: PromptSession, provider: str) -> str:
    print(AI_STARTER_MSG)
    with yaspin(text="generating clarifying questions", color="yellow"):
        chat_session.user_q_a = [
            (q, '') for q in chat_session.chatbot.spec_gathering_response(user_spec)]
    for i, q_a in enumerate(chat_session.user_q_a):
        user_answer = prompt_session.prompt(str(q_a[0])).strip()
        if user_answer.isdigit():
            user_answer = q_a[0].possible_answers[int(user_answer)-1]
        chat_session.user_q_a[i] = (q_a[0], user_answer)

    return chat_session.create_prompt_from_qa(provider)
     


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-id', type=str,
                        required=True, choices=["gpt-4", "codellama_7b"])
    parser.add_argument('-p', '--provider', type=str,
                        required=True, choices=["aws", "gcp"])
    parser.add_argument('-v', '--validate', action='store_true')
    args = parser.parse_args()

    download_terraform(os.getenv("TERRAFORM_VERSION"))

    if args.model_id == "gpt-4":
        chat_session = Session(id=0, chatbot=OpenAIChatbot(model_id=os.getenv(
            "OPENAI_MODEL_ID"), temperature=0, stream_handler_class=NoopStreamHandler), validate_ctx=yaspin, generate_ctx=yaspin)
    elif args.model_id == "codellama_7b":
        chat_session = Session(id=0, chatbot=LLamaChatbot(), validate_ctx=yaspin, generate_ctx=yaspin)

    prompt_session = PromptSession(lexer=PygmentsLexer(SqlLexer))
    user_spec = prompt_session.prompt(
        WELCOME_MSG + '\nPlease specify what you would like in your template. >')
    
    prompt = get_qa_spec_prompt(user_spec, chat_session, prompt_session, args.provider)

    while True:
        try:
            while len(prompt.strip()) == 0:
              prompt = prompt_session.prompt('> ')    
            chat_session.get_terraform_template(spec=prompt, validate=args.validate, **{
                                        "text": "generating terraform template", "color": "green"})
            print(chat_session.terraform_template)
            prompt = prompt_session.prompt('> ')
        except KeyboardInterrupt:
            continue
        except EOFError:
            break

    return


if __name__ == '__main__':
    main()
