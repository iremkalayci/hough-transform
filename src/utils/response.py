from sdks.novavision.src.helper.package import PackageHelper

if __package__ and __package__.startswith("capsules."):
    from capsules.HoughTransform.src.models.PackageModel import (
        CircleDetectionExecutor as CircleDetectionExecutorModel,
        CircleDetectionOutputs,
        CircleDetectionResponse,
        ConfigExecutor,
        LineDetectionExecutor as LineDetectionExecutorModel,
        LineDetectionOutputs,
        LineDetectionResponse,
        OutputDetections,
        OutputImage,
        PackageConfigs,
        PackageModel,
    )
else:
    from src.models.PackageModel import (
        CircleDetectionExecutor as CircleDetectionExecutorModel,
        CircleDetectionOutputs,
        CircleDetectionResponse,
        ConfigExecutor,
        LineDetectionExecutor as LineDetectionExecutorModel,
        LineDetectionOutputs,
        LineDetectionResponse,
        OutputDetections,
        OutputImage,
        PackageConfigs,
        PackageModel,
    )


def _build_response(context, executor_model, response_model, outputs_model):
    output_image = OutputImage(value=context.image)
    output_detections = OutputDetections(value=context.detections)
    outputs = outputs_model(
        outputImage=output_image,
        outputDetections=output_detections,
    )
    response = response_model(outputs=outputs)
    task_executor = executor_model(value=response)
    executor = ConfigExecutor(value=task_executor)
    package_configs = PackageConfigs(executor=executor)
    package = PackageHelper(
        packageModel=PackageModel,
        packageConfigs=package_configs,
    )
    return package.build_model(context)


def build_response_line(context):
    return _build_response(
        context,
        LineDetectionExecutorModel,
        LineDetectionResponse,
        LineDetectionOutputs,
    )


def build_response_circle(context):
    return _build_response(
        context,
        CircleDetectionExecutorModel,
        CircleDetectionResponse,
        CircleDetectionOutputs,
    )


def build_response(context):
    if context.task == "LineDetection":
        return build_response_line(context)
    if context.task == "CircleDetection":
        return build_response_circle(context)
    raise ValueError("Unsupported Hough detection task: {!r}".format(context.task))
