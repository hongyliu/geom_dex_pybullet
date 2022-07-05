from setuptools import setup, find_packages


setup(
    name="shadowhand_gym",
    description="OpenAI Gym Shadow Dexterous Hand robot environment based on PyBullet.",
    author="szahlner",
    url="https://github.com/szahlner/shadowhand-gym",
    packages=find_packages(),
    include_package_data=True,
    package_data={},
    version="1.0.0",
    install_requires=["gym", "numpy", "pybullet"],
    classifiers=[
        "License :: MIT License",
        "Programming Language :: Python :: 3.7",
    ],
)
