
from sdks.novavision.src.helper.package import PackageHelper

from capsules.HoughTransform.src.models.PackageModel import (
    PackageModel,
    HoughConfigs,
    ConfigExecutor,
    PackageExecutor,
    PackageResponse,
    PackageOutputs,
    OutputImage,
    OutputDetections,
)


def build_response(context):
    output_image = OutputImage(value=context.image)
    output_detections = OutputDetections(value=context.detect)

    outputs = PackageOutputs(
        outputImage=output_image,
        outputDetections=output_detections,
    )

    response = PackageResponse(outputs=outputs)
    package_executor = PackageExecutor(value=response)
    executor = ConfigExecutor(value=package_executor)

    configs = HoughConfigs(
        executor=executor,
        task=context.request.get_param("Task"),
        cannyLow=context.request.get_param("CannyLow"),
        cannyHigh=context.request.get_param("CannyHigh"),
        houghThreshold=context.request.get_param("HoughThreshold"),
        minLineLength=context.request.get_param("MinLineLength"),
        maxLineGap=context.request.get_param("MaxLineGap"),
        dp=context.request.get_param("DP"),
        minDist=context.request.get_param("MinDist"),
        param1=context.request.get_param("Param1"),
        param2=context.request.get_param("Param2"),
        minRadius=context.request.get_param("MinRadius"),
        maxRadius=context.request.get_param("MaxRadius"),
    )

    package = PackageHelper(
        packageModel=PackageModel,
        packageConfigs=configs,
    )

    return package.build_model(context)
