from setuptools import setup, find_packages

setup(
    name="loom",
    version="0.1.0",
    description="A persistent database that feels like Python",
    author="Maixent Chenebaux",
    packages=find_packages(
        exclude=["tests", "tests.*", "benchmarks", "examples", "tutorials"]
    ),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "mmh3>=4.0.0",
        "lru-dict>=1.2.0",
        "brotli>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
