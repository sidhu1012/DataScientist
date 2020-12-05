import setuptools

with open("README.md", "r" , encoding= "utf8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="DataScientist",
    version="0.0.1",
    author="Sudeep Sidhu",
    author_email="sudeepsidhu102@gmail.com",
    description="A python library to do almost all data science tasks automatically.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sidhu1012/DataScientist",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
