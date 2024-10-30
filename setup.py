from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="llmcore",
    version="0.0.5",
    author="Sunny Singh",
    author_email="ishy.singh@gmail.com",
    description="LLMCore: Essential tools for LLM development - models, prompts, embeddings, agents, chains, and more.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(
        include=["llmcore", "llmcore.*"],
        exclude=["tests", "tests.*", "run.py", "audit.py", "size.py", "publish.py"] 
    ),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "aiohttp>=3.8.0,<4.0.0",
        "tiktoken>=0.3.3,<0.4.0"
    ],
    extras_require={
        "dev": [
            "pytest==7.3.1",
            "pytest-asyncio",
            "pytest-mock",
            "wheel"
        ],
    },
    exclude_package_data = {
        '': ['*.pyc', '*.pyo', '*.pyd', '__pycache__', '*.egg-info', '*.dist-info', '*.so'],
        'pip': ['*'],
        'setuptools': ['*'],
        'llmcore': ['tests/*', 'docs/*', '*.md'],
        'llmcore.egg-info': ['*'],
        'wheel': ['*'],
    }
)