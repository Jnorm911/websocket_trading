from setuptools import setup, find_packages

setup(
    name="websocket_trading",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "scikit-learn",
        "pandas",
        "kucoin-python",
    ],
    python_requires=">=3.6",
)
