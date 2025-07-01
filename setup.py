from setuptools import setup, find_packages

setup(
    name="spotify_analysis_app",
    version="1.0",
    packages=find_packages(),
    python_requires=">=3.10, <3.14",
    install_requires=[
        "streamlit>=1.32.0",
        "pandas>=2.2.0",
        "numpy>=1.26.3",
        "matplotlib>=3.8.2",
        "seaborn>=0.13.0",
        "scikit-learn>=1.3.2",
        "mlxtend>=0.23.0",
    ],
)
