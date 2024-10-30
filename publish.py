import os
import subprocess
import time
import venv
import tempfile
from pathlib import Path
from dotenv import load_dotenv
import shutil

def load_credentials():
    load_dotenv()
    return {
        'test': {
            'username': os.getenv('PYPI_TEST_USERNAME'),
            'password': os.getenv('PYPI_TEST_PASSWORD')
        },
        'prod': {
            'username': os.getenv('PYPI_USERNAME'),
            'password': os.getenv('PYPI_PASSWORD')
        }
    }

def build_distribution():
    print("Building distribution...")
    if Path("dist").exists():
        shutil.rmtree("dist")
    if Path("build").exists():
        shutil.rmtree("build")
    subprocess.run(["python", "setup.py", "sdist", "bdist_wheel"], check=True)

def verify_installation(package_name, version, is_test=True):
    print(f"Verifying installation from {'TestPyPI' if is_test else 'PyPI'}...")
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create virtual environment
        venv.create(temp_dir, with_pip=True)
        
        # Get pip path
        pip_path = str(Path(temp_dir) / "bin" / "pip") if os.name != 'nt' else str(Path(temp_dir) / "Scripts" / "pip")
        
        # Install package
        repo_url = "--index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/" if is_test else ""
        try:
            subprocess.run(
                f"{pip_path} install {repo_url} {package_name}=={version}",
                shell=True,
                check=True,
                capture_output=True,
                text=True
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"Installation failed: {e.stderr}")
            return False

def upload_to_pypi(credentials, is_test=True):
    env = "test" if is_test else "prod"
    repo_url = "https://test.pypi.org/legacy/" if is_test else "https://upload.pypi.org/legacy/"
    
    print(f"Uploading to {'TestPyPI' if is_test else 'PyPI'}...")
    try:
        subprocess.run([
            "twine", "upload",
            "--repository-url", repo_url,
            "--username", credentials[env]['username'],
            "--password", credentials[env]['password'],
            "dist/*"
        ], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Upload failed: {e}")
        return False

def get_version():
    with open("setup.py", "r") as f:
        for line in f:
            if "version=" in line:
                return line.split('"')[1]
    raise ValueError("Version not found in setup.py")

def main():
    # Install required packages
    subprocess.run(["pip", "install", "twine", "python-dotenv", "wheel"], check=True)
    
    # Load credentials
    credentials = load_credentials()
    if not all(credentials['test'].values()) or not all(credentials['prod'].values()):
        raise ValueError("Missing PyPI credentials in .env file")
    
    # Get package version
    version = get_version()
    package_name = "llmcore"
    
    # Build distribution
    build_distribution()
    
    # Upload to TestPyPI
    if not upload_to_pypi(credentials, is_test=True):
        print("Failed to upload to TestPyPI. Aborting.")
        return
    
    # Wait for TestPyPI to process the upload
    print("Waiting for TestPyPI to process the upload...")
    time.sleep(30)
    
    # Verify TestPyPI installation
    if not verify_installation(package_name, version, is_test=True):
        print("Failed to verify TestPyPI installation. Aborting.")
        return
    
    # Upload to PyPI
    if not upload_to_pypi(credentials, is_test=False):
        print("Failed to upload to PyPI.")
        return
    
    print(f"Successfully published {package_name} version {version} to PyPI!")

if __name__ == "__main__":
    main()