from setuptools import setup, find_packages

# Read the README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dlcopilot",
    version="0.1.0",
    author="Kiren",
    author_email="kiren2612@gmail.com",
    description="An assistant for training and analyzing deep learning models in PyTorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KIREN2612/dlcopilot",
    project_urls={
        "Documentation": "https://github.com/KIREN2612/dlcopilot/tree/main/docs",
        "Source": "https://github.com/KIREN2612/dlcopilot",
        "Tracker": "https://github.com/KIREN2612/dlcopilot/issues",
    },
    packages=find_packages(exclude=["tests*", "examples*", "docs*"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10",
        "matplotlib>=3.5",
        "numpy>=1.20",
        "scikit-learn>=1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black",
            "flake8",
            "isort"
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    include_package_data=True,
    license="MIT",
    keywords="deep learning pytorch training analysis copilot",
)
