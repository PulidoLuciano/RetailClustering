from setuptools import find_packages, setup

setup(
    name="RetailClustering",
    packages=find_packages(exclude=["RetailClustering_tests"]),
    install_requires=[
        "dagster",
        "dagster-cloud"
    ],
    extras_require={"dev": ["dagster-webserver", "pytest"]},
)
