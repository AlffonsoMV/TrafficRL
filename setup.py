from setuptools import setup, find_packages

setup(
    name="traffic_rl",
    version="1.0.0",
    description="Traffic Light Management with Reinforcement Learning",
    author_email="",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.18.0",
        "torch>=1.7.0",
        "gymnasium>=0.28.0",
        "pygame>=2.0.0",
        "matplotlib>=3.3.0",
        "tqdm>=4.50.0",
    ],
    python_requires='>=3.7',
    entry_points={
        'console_scripts': [
            'traffic-rl=traffic_rl.main:main',
        ],
    },
)