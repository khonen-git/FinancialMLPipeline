"""Setup script for FinancialMLPipeline."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="financial-ml-pipeline",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A professional ML trading pipeline with session-aware backtesting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/FinancialMLPipeline",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "polars>=0.19.0",
        "pyarrow>=13.0.0",
        "scikit-learn>=1.3.0",
        "hmmlearn>=0.3.0",
        "backtrader>=1.9.76",
        "hydra-core>=1.3.0",
        "mlflow>=2.8.0",
        "jinja2>=3.1.0",
    ],
    extras_require={
        "gpu": [
            "cudf-cu11",
            "cuml-cu11",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "mypy>=1.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fml-prepare=scripts.prepare_data:main",
            "fml-experiment=scripts.run_experiment:main",
            "fml-backtest=scripts.run_backtest:main",
            "fml-predict=scripts.predict_cli:main",
        ],
    },
)

