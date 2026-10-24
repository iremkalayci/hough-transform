from typing import List, Literal, Optional, Union

from pydantic import validator

from sdks.novavision.src.base.model import (
    Config,
    Configs,
    Detection as SDKDetection,
    Image,
    Input,
    Inputs,
    Output,
    Outputs,
    Package,
    Request,
    Response,
)


class InputImage(Input):
    name: Literal["inputImage"] = "inputImage"
    value: Union[List[Image], Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        image_value = values.get("value")
        if isinstance(image_value, Image):
            return "object"
        if isinstance(image_value, list):
            return "list"
        return value

    class Config:
        title = "Image"


class OutputImage(Output):
    name: Literal["outputImage"] = "outputImage"
    value: Union[List[Image], Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        image_value = values.get("value")
        if isinstance(image_value, Image):
            return "object"
        if isinstance(image_value, list):
            return "list"
        return value

    class Config:
        title = "Output Image"


class Detection(SDKDetection):
    confidence: Optional[float] = None
    classLabel: Optional[str] = None
    classId: Optional[int] = None
    imgUID: str


class OutputDetections(Output):
    name: Literal["outputDetections"] = "outputDetections"
    value: List[Detection]
    type: Literal["list"] = "list"

    class Config:
        title = "Detections"


class PackageInputs(Inputs):
    inputImage: InputImage


class LineDetectionConfigs(Configs):
    """Keep detector tuning internal; expose only the Task selector in the UI."""
    pass


class CircleDetectionConfigs(Configs):
    """Use capsule defaults instead of exposing OpenCV tuning controls."""
    pass


class LineDetectionOutputs(Outputs):
    outputImage: OutputImage
    outputDetections: OutputDetections


class CircleDetectionOutputs(Outputs):
    outputImage: OutputImage
    outputDetections: OutputDetections


class LineDetectionRequest(Request):
    inputs: Optional[PackageInputs] = None
    configs: LineDetectionConfigs

    @validator("configs", pre=True)
    def normalize_empty_configs(cls, value):
        # NovaVision serializes a task with no visible tuning fields as [].
        return {} if isinstance(value, list) and not value else value

    class Config:
        json_schema_extra = {"target": "configs"}


class CircleDetectionRequest(Request):
    inputs: Optional[PackageInputs] = None
    configs: CircleDetectionConfigs

    @validator("configs", pre=True)
    def normalize_empty_configs(cls, value):
        return {} if isinstance(value, list) and not value else value

    class Config:
        json_schema_extra = {"target": "configs"}


class LineDetectionResponse(Response):
    outputs: LineDetectionOutputs


class CircleDetectionResponse(Response):
    outputs: CircleDetectionOutputs


class LineDetectionExecutor(Config):
    name: Literal["LineDetection"] = "LineDetection"
    value: Union[LineDetectionRequest, LineDetectionResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Line Detection"
        json_schema_extra = {"target": {"value": 0}}


class CircleDetectionExecutor(Config):
    name: Literal["CircleDetection"] = "CircleDetection"
    value: Union[CircleDetectionRequest, CircleDetectionResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Circle Detection"
        json_schema_extra = {"target": {"value": 0}}


class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[LineDetectionExecutor, CircleDetectionExecutor]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Task"


class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["capsule"] = "capsule"
    name: Literal["HoughTransform"] = "HoughTransform"
