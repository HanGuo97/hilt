from setuptools import find_packages, setup

LIBRARY_NAME = "cutedsl_utilities"


setup(
    name=LIBRARY_NAME,
    version="0.0.1",
    packages=find_packages(),
    install_requires=["torch"],
    description="CuTeDSL Utilities",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/HanGuo97/cutedsl_utilities",
)
