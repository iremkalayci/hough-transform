from setuptools import find_packages, setup


setup(
    name="hough-transform-capsule",
    version="0.1.0",
    description="NovaVision capsule for OpenCV line and circle detection",
    packages=find_packages(include=("src", "src.*")),
    install_requires=["sdk", "opencv-python-headless"],
    python_requires=">=3.9",
)
