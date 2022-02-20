import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tfyolor",
    version="0.0.1",
    author="Jacob Lubecki",
    author_email="jacoblubecki@gmail.com",
    description="An attempt at implementing YOLOR in TensorFlow.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jacoblubecki/tfyolor",
    project_urls={
        "Bug Tracker": "https://github.com/jacoblubecki/tfyolor/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
)
