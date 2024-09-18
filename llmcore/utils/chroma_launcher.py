import subprocess
import sys
import time

def is_docker_installed():
    try:
        subprocess.run(['docker', '--version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception:
        return False

def is_container_running(container_name):
    result = subprocess.run(['docker', 'ps', '--filter', f'name={container_name}', '--format', '{{.Names}}'], stdout=subprocess.PIPE)
    containers = result.stdout.decode().strip().split('\n')
    return container_name in containers

def launch_chromadb():
    docker_image = "ghcr.io/chroma-core/chroma:latest"
    container_name = "chromadb_local"
    port = 8000

    if not is_docker_installed():
        print("Docker is not installed or not running. Please install Docker and try again.")
        sys.exit(1)

    if is_container_running(container_name):
        print(f"ChromaDB container '{container_name}' is already running.")
        print(f"Access it at http://localhost:{port}")
        return

    try:
        print(f"Pulling Docker image '{docker_image}'...")
        subprocess.run(['docker', 'pull', docker_image], check=True)
        print("Docker image pulled successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to pull Docker image: {e}")
        sys.exit(1)

    try:
        print(f"Launching ChromaDB container '{container_name}' on port {port}...")
        subprocess.run([
            'docker', 'run', '--name', container_name, '-d',
            '-p', f"{port}:8000",
            docker_image
        ], check=True)
        print(f"ChromaDB is running at http://localhost:{port}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to launch ChromaDB container: {e}")
        sys.exit(1)

if __name__ == "__main__":
    launch_chromadb()