from setuptools import setup, find_packages

setup(
    name="gllm",
    version="0.1",
    packages=find_packages(),  # automatically finds gllm/
    install_requires=[
        "torch",
        "transformers",
        # add other dependencies here
    ],
)