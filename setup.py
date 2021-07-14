import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="easynlp",
    author="Ben Trevett",
    author_email="bentrevett@gmail.com",
    description="Performing NLP without getting your hands dirty.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/easynlp/easynlp",
    project_urls={
        "Bug Tracker": "https://github.com/easynlp/easynlp/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
    install_requires=[
        "datasets>=1.9.0",
        "fastapi>=0.66",
        "pandas>=1.3.0",
        "torch>=1.9.0",
        "transformers>=4.8.2",
        "uvicorn>=0.14",
    ],
)
