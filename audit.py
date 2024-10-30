import os
import tarfile
import zipfile
import pkg_resources

def get_package_size(package_name):
    try:
        package = pkg_resources.get_distribution(package_name)
        metadata_files = []
        for line in package.get_metadata_lines('RECORD'):
            parts = line.strip().split(',')
            if len(parts) > 0:
                metadata_files.append(parts[0])
        return sum(os.path.getsize(os.path.join(package.location, f))
                   for f in metadata_files if os.path.isfile(os.path.join(package.location, f)))
    except pkg_resources.DistributionNotFound:
        return 0

def audit_sdist(sdist_path):
    total_size = 0
    with tarfile.open(sdist_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.isfile():
                print(f"{member.size} bytes - {member.name}")
                total_size += member.size
    print(f"\nTotal sdist size: {total_size} bytes ({total_size / 1024 / 1024:.2f} MB)")

def audit_wheel(wheel_path):
    total_size = 0
    with zipfile.ZipFile(wheel_path, 'r') as zipf:
        for file_info in zipf.infolist():
            if not file_info.is_dir():
                print(f"{file_info.file_size} bytes - {file_info.filename}")
                total_size += file_info.file_size
    print(f"\nTotal wheel size: {total_size} bytes ({total_size / 1024 / 1024:.2f} MB)")

def audit_dependencies():
    dependencies = [
        "aiohttp", "tiktoken"
    ]
    print("\nDependency Sizes:")
    for dep in dependencies:
        size = get_package_size(dep)
        print(f"{dep}: {size / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    sdist = "dist/llmcore-0.0.4.tar.gz"
    wheel = "dist/llmcore-0.0.4-py3-none-any.whl"
    
    print("SDist Contents:")
    audit_sdist(sdist)
    
    print("\nWheel Contents:")
    audit_wheel(wheel)
    
    audit_dependencies()