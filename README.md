Vinfrai is a tool to interactively generate and verify cloud infrastructure. It does this by generating and verifying [Terraform](https://www.terraform.io/) templates. Currently only AWS is supported. In addition to static validation, it can also generate automated tests for the template written using the [Terratest](https://terratest.gruntwork.io/) library. 

# Features
* Interactive cli providing a chat experience.
* Asks clarifying questions before generating the initial template.
* Choice of `gpt-4` or `codellama-34b` as the llm.
* Option to statically validate the template.
* Option to generate and run automated tests in Golang using the [Terratest](https://terratest.gruntwork.io/) library.



https://github.com/ujaved/vinfrai/assets/1680045/ba7dd85f-ab09-4d87-9b04-4c11675f98df

  
# Cli
* Install [Pipenv](https://docs.pipenv.org/install/#installing-pipenv), if not already installed.
* Install dependencies with `pipenv install`.
* Start the virtual environment shell with `pipenv shell`.
* Set the follwoing environment variables with appropriate values:
   
    *  `OPENAI_API_KEY` [OpenAI API key](https://platform.openai.com/api-keys)
    *  `AWS_ACCESS_KEY_ID` [AWS access key Id](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-envvars.html) (This is required only for running Terratest automated tests)
    *  `AWS_SECRET_ACCESS_KEY` [AWS secret access key](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-envvars.html) (This is required only for running Terratest automated tests)
    *  `LOGFILE` (Path to a log file relative to the current directory. Default is `output.log`.)
    *  `REPLICATE_API_TOKEN` [Replicate](https://replicate.com/) API token. (The replicate API token is required for `codellama-34b` inferences and hence optional.)
    *  `TERRAFORM_VERSION` (The Terraform version to download. Default is "1.5.4".)

    These could be set with a `.env` file (An example `.env.example` file is provided.)
* Run `python3 cli.py --help` for cli usage options.
* For a template without validation, the cli only requires the llm: with `gpt-4` as the llm, run `python3 cli.py -m gpt-4 -d` and start chatting away!
* For a template with static validation, specify the llm and `-v`: with `gpt-4` as the llm, run `python3 cli.py -m gpt-4 -v`. The validated template will be saved as `main.tf` in subdirectory `terraform_<uuid>` when the generate-validate cycle ends.
* For a template with automated tests, specify the llm and `-t`: with `gpt-4` as the llm, run `python3 cli.py -m gpt-4 -t`. The validated template will be saved as `main.tf` in subdirectory `terraform_<uuid>` when the generate-validate cycle ends. This subdirectory will also contain a `go` module `terratest_<uuid>` containing the `terratest` automated test file.   


# GUI

There's also a slightly limited version of the tool available as a Streamlit [GUI](https://ai-infra.streamlit.app/).

