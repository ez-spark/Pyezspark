import setuptools

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "Pyezspark",
    packages=['pyezspark'],
    version = "1.0.1",
    install_requires=['setuptools>=18.0','wheel','Cython', 'numpy', 'Ezclient', 'requests'],
    author = "Riccardo Viviano",
    author_email = "riccardo.viviano@ezspark.ai",
    description = "Ezspark python package",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/ez-spark/Pyezspark",
    project_urls = {
        "Bug Tracker": "https://github.com/ez-spark/Pyezspark",
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires = ">=3.6"
)
