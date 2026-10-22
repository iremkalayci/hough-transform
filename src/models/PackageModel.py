
from typing import Literal, Optional, Union, List

from pydantic import Field, validator

from sdks.novavision.src.base.model import (
    Package,
    Image,
    Inputs,
    Configs,
    Outputs,
    Response,
    Request,
    Output,
    Input,
    Config,
    Detection,
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
        if isinstance(value, list):
            return "list"
        return value

    class Config:
        title = "Output Image"


class OutputDetections(Output):
    name: Literal["outputDetections"] = "outputDetections"
    value: list
    type: Literal["list"] = "list"

    class Config:
        title = "Detections"


class Task(Config):
    name: Literal["Task"] = "Task"
    value: Literal["LineDetection", "CircleDetection"] = "LineDetection"
    type: Literal["str"] = "str"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Detection Task"


class CannyLow(Config):
    name: Literal["CannyLow"] = "CannyLow"
    value: int = Field(default=50, ge=0, le=255)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Canny Low Threshold"


class CannyHigh(Config):
    name: Literal["CannyHigh"] = "CannyHigh"
    value: int = Field(default=150, ge=1, le=255)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Canny High Threshold"


class HoughThreshold(Config):
    name: Literal["HoughThreshold"] = "HoughThreshold"
    value: int = Field(default=50, ge=1)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Hough Threshold"


class MinLineLength(Config):
    name: Literal["MinLineLength"] = "MinLineLength"
    value: int = Field(default=50, ge=1)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Minimum Line Length"


class MaxLineGap(Config):
    name: Literal["MaxLineGap"] = "MaxLineGap"
    value: int = Field(default=10, ge=0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Maximum Line Gap"


class DP(Config):
    name: Literal["DP"] = "DP"
    value: float = Field(default=1.2, gt=0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Inverse Accumulator Ratio"


class MinDist(Config):
    name: Literal["MinDist"] = "MinDist"
    value: float = Field(default=20.0, gt=0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Minimum Circle Distance"


class Param1(Config):
    name: Literal["Param1"] = "Param1"
    value: float = Field(default=100.0, gt=0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Circle Edge Threshold"


class Param2(Config):
    name: Literal["Param2"] = "Param2"
    value: float = Field(default=30.0, gt=0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Circle Accumulator Threshold"


class MinRadius(Config):
    name: Literal["MinRadius"] = "MinRadius"
    value: int = Field(default=0, ge=0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Minimum Radius"


class MaxRadius(Config):
    name: Literal["MaxRadius"] = "MaxRadius"
    value: int = Field(default=0, ge=0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Maximum Radius"


class PackageInputs(Inputs):
    inputImage: InputImage


class PackageOutputs(Outputs):
    outputImage: OutputImage
    outputDetections: OutputDetections


class PackageRequest(Request):
    inputs: Optional[PackageInputs] = None
    configs: Configs

    class Config:
        json_schema_extra = {"target": "configs"}


class PackageResponse(Response):
    outputs: PackageOutputs


class PackageExecutor(Config):
    name: Literal["HoughTransform"] = "HoughTransform"
    value: Union[PackageRequest, PackageResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Hough Transform"
        json_schema_extra = {"target": {"value": 0}}


class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[PackageExecutor]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Task"
        json_schema_extra = {"target": "value"}


class HoughConfigs(Configs):
    executor: ConfigExecutor
    task: Task
    cannyLow: CannyLow
    cannyHigh: CannyHigh
    houghThreshold: HoughThreshold
    minLineLength: MinLineLength
    maxLineGap: MaxLineGap
    dp: DP
    minDist: MinDist
    param1: Param1
    param2: Param2
    minRadius: MinRadius
    maxRadius: MaxRadius


class PackageModel(Package):
    configs: HoughConfigs
    type: Literal["capsule"] = "capsule"
    name: Literal["HoughTransform"] = "HoughTransform"
