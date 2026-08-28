from setuptools import setup, find_packages

setup(
    name="fairpp",
    version="0.1",
    description="Fairness threshold post-processing",
    author="Anderson Lucas",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "scikit-learn"
    ],
)