from setuptools import find_namespace_packages, setup

CAPSULE_PACKAGE = "capsules.HoughTransform"
source_packages = find_namespace_packages(include=("src", "src.*"))
packages = [CAPSULE_PACKAGE] + [
    f"{CAPSULE_PACKAGE}.{package}" for package in source_packages
]

setup(
    name="hough-transform-capsule",
    version="0.1.0",
    description="NovaVision capsule for OpenCV line and circle detection",
    # Match the import path used by NovaVision's capsule loader without
    # shipping synthetic __init__.py files in the capsule source tree.
    packages=packages,
    package_dir={CAPSULE_PACKAGE: "."},
    # NovaVision's private ``sdks`` package is provided by the host runtime;
    # the unrelated public PyPI package named ``sdk`` must not be installed.
    install_requires=["numpy", "opencv-python-headless"],
    python_requires=">=3.8",
)
