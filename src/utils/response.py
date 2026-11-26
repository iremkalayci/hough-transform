from sdks.novavision.src.helper.package import PackageHelper

from capsules.HoughTransform.src.models.PackageModel import (
    ConfigExecutor,
    DetectionOutputs,
    DetectionResponse,
    OutputDetections,
    OutputImage,
    PackageConfigs,
    PackageModel,
    LineDetectionExecutor,
    CircleDetectionExecutor,
    NoiseDetectionExecutor,
    BrokenDetectionExecutor,
    PartialEdgeDetectionExecutor,
    EllipseDetectionExecutor,
    ShapeDetectionExecutor,
)

EXECUTOR_MODELS = {
    "LineDetection": LineDetectionExecutor,
    "CircleDetection": CircleDetectionExecutor,
    "NoiseDetection": NoiseDetectionExecutor,
    "BrokenDetection": BrokenDetectionExecutor,
    "PartialEdgeDetection": PartialEdgeDetectionExecutor,
    "EllipseDetection": EllipseDetectionExecutor,
    "ShapeDetection": ShapeDetectionExecutor,
}


def build_response(context):
    executor_model = EXECUTOR_MODELS.get(context.task)
    if executor_model is None:
        raise ValueError("Unsupported Hough detection task: {!r}".format(context.task))

    outputs = DetectionOutputs(
        outputImage=OutputImage(value=context.image),
        outputDetections=OutputDetections(value=context.detections),
    )
    response = DetectionResponse(outputs=outputs)
    executor = ConfigExecutor(value=executor_model(value=response))
    package = PackageHelper(
        packageModel=PackageModel,
        packageConfigs=PackageConfigs(executor=executor),
    )
    return package.build_model(context)