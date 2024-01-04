import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Pylyze",
    version="0.1",
    author="Kannmu",
    author_email="Kannmu@163.com",
    description="Pylyze: A data processing tool for python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Kannmu/Pylyze",
    packages = setuptools.find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'scipy>=1.7.0'
    ],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
)