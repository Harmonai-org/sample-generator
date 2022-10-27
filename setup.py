from setuptools import setup, find_packages

setup(
    name='sample-generator',
    version='1.0.0',
    url='https://github.com/harmonai-org/sample-generator.git',
    author='Zach Evans',
    packages=find_packages(),    
    install_requires=[
        'einops',
        'pandas',
        'prefigure', 
        'pytorch_lightning',
        'scipy',
        'torch',
        'torchaudio',
        'tqdm',
        'wandb',
    ],
)
