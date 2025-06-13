#!/usr/bin/env python3
"""Setup script for STRATO-PEFT experimental framework."""

import os
from pathlib import Path
from typing import List

from setuptools import find_packages, setup

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements from requirements.txt
def read_requirements(filename: str) -> List[str]:
    """Read requirements from a file."""
    requirements_path = this_directory / filename
    if requirements_path.exists():
        with open(requirements_path, "r", encoding="utf-8") as f:
            return [
                line.strip()
                for line in f
                if line.strip() and not line.startswith("#")
            ]
    return []

# Read version from __init__.py
def get_version() -> str:
    """Get version from package __init__.py."""
    version_file = this_directory / "src" / "__init__.py"
    if version_file.exists():
        with open(version_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip('"').strip("'")
    return "0.1.0"

# Package metadata
setup(
    name="strato-peft",
    version=get_version(),
    author="STRATO-PEFT Team",
    author_email="strato-peft@example.com",
    description="Strategic Parameter-Efficient Fine-Tuning Experimental Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/strato-peft",
    project_urls={
        "Bug Reports": "https://github.com/your-org/strato-peft/issues",
        "Source": "https://github.com/your-org/strato-peft",
        "Documentation": "https://strato-peft.readthedocs.io/",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": read_requirements("requirements-dev.txt"),
        "cuda": [
            "torch>=2.0.0+cu118",
            "torchvision>=0.15.0+cu118",
            "torchaudio>=2.0.0+cu118",
        ],
        "rocm": [
            "torch>=2.0.0+rocm5.4",
            "torchvision>=0.15.0+rocm5.4",
            "torchaudio>=2.0.0+rocm5.4",
        ],
        "mps": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "torchaudio>=2.0.0",
        ],
        "all": read_requirements("requirements.txt") + read_requirements("requirements-dev.txt"),
    },
    entry_points={
        "console_scripts": [
            "strato-peft=main:main",
            "strato-eval=scripts.eval:main",
            "strato-compare=scripts.compare:main",
        ],
    },
    include_package_data=True,
    package_data={
        "strato_peft": [
            "configs/*.yaml",
            "configs/**/*.yaml",
            "templates/*.yaml",
            "templates/**/*.yaml",
        ],
    },
    zip_safe=False,
    keywords=[
        "machine learning",
        "deep learning",
        "parameter efficient fine-tuning",
        "PEFT",
        "LoRA",
        "transformers",
        "large language models",
        "reinforcement learning",
        "optimization",
    ],
    # Additional metadata
    platforms=["any"],
    license="MIT",
    # Test configuration
    test_suite="tests",
    tests_require=[
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "pytest-xdist>=3.3.0",
    ],
    # Build configuration
    cmdclass={},
    # Dependency links (for development versions)
    dependency_links=[],
    # Namespace packages
    namespace_packages=[],
)