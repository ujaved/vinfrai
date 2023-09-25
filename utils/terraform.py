import platform
import urllib.request
import zipfile
import stat
from pathlib import Path
import os
import shutil
import subprocess

def download_terraform(terraform_version: str):
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