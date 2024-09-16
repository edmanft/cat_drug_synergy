from setuptools import setup, find_packages

setup(
    name="cat_drug_syn",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastai",
        "dtreeviz",
        "fastbook",
        "pandas",
        "scikit-learn",
        "xgboost",
        "seaborn",
        "IPython",
        "auto-sklearn",
        "pytorch-tabnet",
    ],
    author="Manuel González Lastre",
    author_email="manuel.e.g.l1999@gmail.com",
    description="A package to identify synergic drug combinations using deep learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/edmanft/cat_drug_synergy",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
