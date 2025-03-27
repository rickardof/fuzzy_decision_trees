from setuptools import setup, find_packages

setup(
    name="fuzzy_decision_tree",
    version="1.0.4",
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
    keywords="fuzzy, decision tree, machine learning",
    python_requires=">=3.6",
)