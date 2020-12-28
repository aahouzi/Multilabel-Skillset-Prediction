import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="MultilabelSkillset-aahouzi",
    version="0.0.1.1",
    author="Anas Ahouzi",
    author_email="ahouzi2000@hotmail.fr",
    description="Multilabel classification of skill set based on a job description",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aahouzi/Multilabel-Skillset-Prediction.git ",
    packages=setuptools.find_packages(),
    classifiers=["Programming Language :: Python :: 3",
                 "License :: OSI Approved :: MIT License",
                 "Operating System :: OS Independent"],
    python_requires='>=3.6'
)