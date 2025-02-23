from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="numeraifold",
    version="0.1.0",
    author="whit3rabbit",
    author_email="whiterabbit@protonmail.com",
    description="AlphaFold-inspired pipeline for Numerai",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/whit3rabbit/numeraifold",
    project_urls={
        "Bug Tracker": "https://github.com/whit3rabbit/numeraifold/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.0",
        "umap-learn>=0.5.7",
        "hdbscan>=0.8.0",
        "lightgbm>=3.2.0",
        "tqdm>=4.62.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "numerapi>=2.8.0",
        "numerblox>=0.1.0",
        "plotly>=5.3.0",
        "PyYAML>=6.0",
        "psutil>=5.8.0",
        "pyarrow>=6.0.0"
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "isort>=5.0",
            "flake8>=3.9",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
            "myst-parser>=0.15",
        ],
    },
)
