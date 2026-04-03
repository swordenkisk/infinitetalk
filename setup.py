from setuptools import setup, find_packages

setup(
    name="infinitetalk",
    version="1.0.0",
    description="Sparse-Frame Video Dubbing with 4-Step Inference",
    author="swordenkisk",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "diffusers>=0.24.0",
        "transformers>=4.35.0",
        "accelerate>=0.24.0",
        "peft>=0.6.0",
        "xformers>=0.0.22",
        "fastapi>=0.104.0",
        "gradio>=4.0.0",
        "einops>=0.7.0",
        "facenet-pytorch>=2.5.0",
        "transformer-engine>=1.0; platform_machine=='x86_64'",
    ],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "infinitetalk=infinitetalk.cli:main",
        ],
    },
)
