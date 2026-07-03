from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="ffspaces",
    version="0.2.0",
    author="Nathan Hoehndorf",
    description="A computational laboratory for finite field vector spaces, Hamming balls, and sumset structural analysis.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nathanhoehndorf/finite-field-spaces",
    package_dir={"": "src"},
    packages=find_packages(where="src", include=["ffspaces*"]),
    python_requires=">=3.10",
    install_requires=["numpy>=1.24.0"],
    extras_require={"test": ["pytest>=7.0"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
