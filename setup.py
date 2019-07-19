"""
Setup the xenith package.
"""
import setuptools

with open("README.md", "r") as readme:
    LONG_DESC = readme.read()

DESC = ("Enhanced cross-linked peptide detection using pretrained models")

CATAGORIES = ["Programming Language :: Python :: 3",
              "License :: OSI Approved :: Apache Software License",
              "Operating System :: OS Independent",
              "Topic :: Scientific/Engineering :: Bio-Informatics"]

setuptools.setup(
    name="xenith",
    version="0.0.1",
    author="William E. Fondrie",
    author_email="fondriew@gmail.com",
    description=DESC,
    long_description=LONG_DESC,
    long_description_content_type="text/markdown",
    url="https://github.com/wfondrie/xenith",
    packages=setuptools.find_packages(),
    license="Apache 2.0",
    entry_points={"console_scripts": ["xenith = xenith.xenith:main"]},
    classifiers=CATAGORIES,
    install_requires=[
        "numpy",
        "pandas",
        "torch"
    ],
    use_scm_version=True,
    setup_requires=["setuptools_scm"]
)
