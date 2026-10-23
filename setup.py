from setuptools import find_namespace_packages, setup


setup(
    name="hough-transform-capsule",
    version="0.1.0",
    description="NovaVision capsule for OpenCV line and circle detection",
    # NovaVision capsules are loaded as namespace packages; do not add
    # __init__.py files that alter the capsule layout in the platform.
    packages=find_namespace_packages(include=("src", "src.*")),
    install_requires=["sdk", "opencv-python-headless"],
    python_requires=">=3.6",
)
