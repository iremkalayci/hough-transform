from setuptools import setup

setup(
    name="hough-transform-capsule",
    version="0.1.0",
    description="NovaVision capsule for OpenCV line and circle detection",
    # Match the import path used by NovaVision's capsule loader without
    # shipping synthetic __init__.py files in the capsule source tree.
    # Keep metadata values literal: NovaVision's repository loader reads
    # setup.py metadata without executing arbitrary Python expressions.
    packages=[
        "capsules.HoughTransform",
        "capsules.HoughTransform.src",
        "capsules.HoughTransform.src.executors",
        "capsules.HoughTransform.src.models",
        "capsules.HoughTransform.src.utils",
    ],
    package_dir={"capsules.HoughTransform": "."},
    # NovaVision's private ``sdks`` package is provided by the host runtime;
    # the unrelated public PyPI package named ``sdk`` must not be installed.
    install_requires=["numpy", "opencv-python-headless"],
    python_requires=">=3.8",
)
