import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ml-indie-tools",
    version="0.3.24",
    author="Dominik Schlösser",
    author_email="dsc@dosc.net",
    description="A collection of tools for low-resource indie machine learning development",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/domschl/ml-indie-tools",
    project_urls={"Bug Tracker": "https://github.com/domschl/ml-indie-tools/issues"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
