
from sdks.novavision.src.helper.package import PackageHelper

try:  # NovaVision loads capsules under its capsules.<name> namespace.
    from capsules.HoughTransform.src.models.PackageModel import (
        PackageModel, HoughConfigs, ConfigExecutor, PackageExecutor,
        PackageResponse, PackageOutputs, OutputImage, OutputDetections,
    )
except ModuleNotFoundError:  # Also support an ordinary installation of this repository.
    from src.models.PackageModel import (
        PackageModel, HoughConfigs, ConfigExecutor, PackageExecutor,
        PackageResponse, PackageOutputs, OutputImage, OutputDetections,
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
    )

    package = PackageHelper(
        packageModel=PackageModel,
        packageConfigs=configs,
    )

    return package.build_model(context)
