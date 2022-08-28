from setuptools import find_packages, setup

# pip install pip -e .
setup(
    name="deepcloth_photography_helper",
    packages=find_packages(exclude=("tests*",)),
    entry_points={
        "console_scripts": [
            "deepcloth_photography_helper=deepcloth_photography_helper.cli:app"
        ]
    },
)
