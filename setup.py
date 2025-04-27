from setuptools import setup, find_packages

setup(
    name="dog-breed-recognition",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.5.0",
        "numpy>=1.19.5",
        "matplotlib>=3.4.0",
        "pillow>=8.2.0",
        "scikit-learn>=0.24.0",
        "seaborn>=0.11.0",
        "tensorflow-datasets>=4.4.0",
    ],
    python_requires=">=3.7",
    author="Your Name",
    author_email="your.email@example.com",
    description="Dog breed classification using MobileNetV2",
    keywords="dog, breed, classification, deep learning, mobilenet",
    url="https://github.com/yourusername/dog-breed-recognition",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
