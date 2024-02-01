import platform
import urllib.request
import zipfile
import stat
from pathlib import Path
import os
import shutil
import subprocess
from . import logger

def download_terraform(terraform_version: str):
    
    if not (Path.cwd()/'terraform').exists():
        
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
        
    if str(Path.cwd()) not in os.environ['PATH']:
        os.environ["PATH"] = str(Path.cwd()) + os.pathsep + os.environ["PATH"]
    
    
def validate_template(template: str, terraform_dir_name: str) -> tuple[str, str]:

    # remove output directory for safety
    if (Path.cwd()/terraform_dir_name).exists():
        shutil.rmtree(f'./{terraform_dir_name}')

    os.mkdir(f'./{terraform_dir_name}')
    with open(f'./{terraform_dir_name}/main.tf', 'w') as f:
        f.write(template)
    os.chdir(f'./{terraform_dir_name}')

    # assume terraform binary is present in PATH
    logger.info('running terraform init')
    result = subprocess.run(['../terraform', 'init'],
                            capture_output=True, text=True)
    logger.info('terraform init result: ' + str(result.__repr__()))

    # only run plan if init has no errors
    err_source = "terraform plan"
    if len(result.stderr) == 0:
        logger.info('running terraform plan')
        result = subprocess.run(['../terraform', 'plan'],
                                capture_output=True, text=True)
        logger.info('terraform plan result: ' + str(result.__repr__()))
    else:
        err_source = "terraform init"

    os.chdir('../')
    return (result.stderr, err_source)


# this validation is performed in the directory where the terraform template is located
# the reason is the '../' that the terratest code checks, which we explicitly specify in the prompt
def validate_terratest(go_code: str, terraform_dir_name: str, terratest_dir_name: str) -> str:

    os.chdir(f'./{terraform_dir_name}')
    
    # remove output directory for safety
    if (Path.cwd()/terratest_dir_name).exists():
        shutil.rmtree(f'./{terratest_dir_name}')

    os.mkdir(f'./{terratest_dir_name}')
    filename = f'./{terratest_dir_name}/{terratest_dir_name}_test.go'
    with open(filename, 'w') as f:
        f.write(go_code)
    os.chdir(f'./{terratest_dir_name}')

    subprocess.run(['go', 'mod', 'init', f'{terratest_dir_name}.com'], capture_output=True, text=True)
    subprocess.run(['go', 'mod', 'tidy'], capture_output=True, text=True)
    
    # compile without running
    result = subprocess.run(['go', 'test', '-c'], capture_output=True, text=True)
    rv = result.stderr 
    if result.stderr:
        fields = result.stderr.split(f'{terratest_dir_name}_test.go')
        rv = fields[-1]
    else:
        result = subprocess.run(['go', 'test', '-v'], capture_output=True, text=True)
        rv = result.stderr     
    os.chdir('../../')
    logger.info("error running go code: " + str(rv))
    return rv