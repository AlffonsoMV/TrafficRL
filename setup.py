from setuptools import setup, find_packages

setup(
    name="traffic_rl",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.19.0',
        'torch>=2.0.0',
        'matplotlib>=3.3.0',
        'gymnasium>=0.26.0',
        'pygame>=2.0.0',
        'pandas>=1.1.0',
        'seaborn>=0.11.0',
        'tqdm>=4.65.0',
    ],
    entry_points={
        "console_scripts": [
            "traffic_rl=traffic_rl.cli:main",
        ],
    },
    python_requires=">=3.7",
    description="Reinforcement Learning for Traffic Light Control",
    author="Traffic RL Team",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)