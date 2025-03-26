from setuptools import setup, find_packages

setup(
    name="fuzzy-decision-tree",  # Usa direttamente il trattino per evitare conversioni
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "simpful"
    ],
    author="Riccardo",
    description="Un albero decisionale fuzzy per classificazione",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tuonome/fuzzy_tree",  # Aggiungi URL se disponibile
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="fuzzy, decision tree, machine learning",
    python_requires=">=3.6",
)