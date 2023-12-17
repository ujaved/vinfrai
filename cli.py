import argparse
from utils.terraform import download_terraform
from Home import WELCOME_MSG, AI_STARTER_MSG, Session
from prompt_toolkit import PromptSession
from prompt_toolkit.lexers import PygmentsLexer
from pygments.lexers.sql import SqlLexer
import os
from yaspin import yaspin
from chatbot import OpenAIChatbot, NoopStreamHandler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-id', type=str,
                        required=True, choices=["gpt-4", "codellama_7b"])
    parser.add_argument('-p', '--provider', type=str,
                        required=True, choices=["aws", "gcp"])
    parser.add_argument('-v', '--validate', action='store_true')
    args = parser.parse_args()

    download_terraform(os.getenv("TERRAFORM_VERSION"))

    chat_session = Session(id=0, chatbot=OpenAIChatbot(model_id=os.getenv(
        "OPENAI_MODEL_ID"), temperature=0, stream_handler_class=NoopStreamHandler), validate_ctx=yaspin, generate_ctx=yaspin)

    prompt_session = PromptSession(lexer=PygmentsLexer(SqlLexer))
    initial_spec = prompt_session.prompt(
        WELCOME_MSG + '\nPlease specify what you would like in your template. >')
    print(AI_STARTER_MSG)
    with yaspin(text="generating clarifying questions", color="yellow"):
        chat_session.user_q_a = [
            (q, '') for q in chat_session.chatbot.spec_gathering_response(initial_spec)]
    for i, q_a in enumerate(chat_session.user_q_a):
        user_answer = prompt_session.prompt(str(q_a[0])).strip()
        if user_answer.isdigit():
            user_answer = q_a[0].possible_answers[int(user_answer)-1]
        chat_session.user_q_a[i] = (q_a[0], user_answer)

    prompt = chat_session.create_prompt_from_qa(args.provider)

    while True:
        try:
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
