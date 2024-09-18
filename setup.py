from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="llmcore",
    version="0.0.1",
    author="Sunny Singh",
    author_email="ishy.singh@gmail.com",
    description="LLMCore: Essential tools for LLM development - models, prompts, embeddings, agents, chains, and more.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
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
        "aiohttp",
        "numpy",
        "tiktoken",
        "chromadb",
        "networkx",
        "scipy",
    ],
    extras_require={
        "dev": [
            "pytest==7.3.1",
            "pytest-asyncio"
        ],
    }
)