import os
import subprocess
import shutil
import tempfile
import venv
from pathlib import Path

class MinimalEnvBuilder(venv.EnvBuilder):
    def __init__(self):
        super().__init__(with_pip=True, system_site_packages=False)
    
    def post_setup(self, context):
        # Skip pip and setuptools installation
        pass

def calculate_deployment_size():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create minimal virtual environment
        builder = MinimalEnvBuilder()
        builder.create(temp_dir)
        
        # Determine pip path
        pip_path = str(Path(temp_dir) / "bin" / "pip") if os.name != 'nt' else str(Path(temp_dir) / "Scripts" / "pip")
        
        # First, build the package if it doesn't exist
        if not list(Path("dist").glob("*.whl")):
            subprocess.run(["python", "setup.py", "bdist_wheel"], check=True)
        
        # Get the latest wheel file
        wheel_file = sorted(Path("dist").glob("*.whl"))[-1]
        
        # Install only the wheel file without dependencies first
        subprocess.run([pip_path, "install", "--no-deps", str(wheel_file)], check=True)
        
        # Then install only the required dependencies
        subprocess.run([pip_path, "install", "--no-deps", "aiohttp>=3.8.0,<4.0.0"], check=True)
        subprocess.run([pip_path, "install", "--no-deps", "tiktoken>=0.3.3,<0.4.0"], check=True)
        
        # Calculate total size
        site_packages = Path(temp_dir) / "lib" / "python3.8" / "site-packages"
        if os.name == 'nt':
            site_packages = Path(temp_dir) / "Lib" / "site-packages"
            
        total_size = sum(f.stat().st_size for f in site_packages.rglob('*') if f.is_file())
        
        # Convert to MB
        size_mb = total_size / (1024 * 1024)
        print(f"Total deployment size: {size_mb:.2f} MB")
        
        # List individual package sizes
        for package_dir in site_packages.iterdir():
            if package_dir.is_dir():
                package_size = sum(f.stat().st_size for f in package_dir.rglob('*') if f.is_file())
                print(f"{package_dir.name}: {package_size / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    # Clean up old distributions first
    if Path("dist").exists():
        shutil.rmtree("dist")
    if Path("build").exists():
        shutil.rmtree("build")
    if Path("llmcore.egg-info").exists():
        shutil.rmtree("llmcore.egg-info")
        
    calculate_deployment_size()