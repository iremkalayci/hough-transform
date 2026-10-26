from typing import List, Literal, Optional, Union

from pydantic import Field, root_validator, validator

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


def _description(short, long):
    """Metadata used by NovaVision forms and package-aware LLM tooling."""
    return {"shortDescription": short, "longDescription": long}


class CannyLow(Config):
    name: Literal["cannyLow"] = "cannyLow"
    value: int = Field(default=50, ge=0, le=254)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Canny Low Threshold"
        json_schema_extra = _description(
            "Weak-edge threshold",
            "Gradients below this value are rejected. Lower values preserve faint edges but may introduce texture noise. This value must stay below Canny High Threshold.",
        )


class CannyHigh(Config):
    name: Literal["cannyHigh"] = "cannyHigh"
    value: int = Field(default=150, ge=1, le=255)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Canny High Threshold"
        json_schema_extra = _description(
            "Strong-edge threshold",
            "Gradients at or above this value seed accepted Canny edges. Higher values reduce weak texture; lower values can recover low-contrast lines.",
        )


class HoughThreshold(Config):
    name: Literal["houghThreshold"] = "houghThreshold"
    value: int = Field(default=50, ge=1, le=1000)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Hough Vote Threshold"
        json_schema_extra = _description(
            "Minimum accumulator votes",
            "A candidate needs this many Hough accumulator votes. Raise it for fewer, stronger lines; lower it for faint or short lines.",
        )


class MinLineLength(Config):
    name: Literal["minLineLength"] = "minLineLength"
    value: int = Field(default=30, ge=1, le=10000)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Minimum Line Length"
        json_schema_extra = _description(
            "Shortest accepted segment",
            "Hough segments shorter than this pixel length are ignored. Reduce it for small shapes and dashes; increase it to suppress fine texture.",
        )


class MaxLineGap(Config):
    name: Literal["maxLineGap"] = "maxLineGap"
    value: int = Field(default=20, ge=0, le=10000)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Maximum Line Gap"
        json_schema_extra = _description(
            "Broken-line linking",
            "Largest pixel gap that can be linked into one Hough segment. Increase it for interrupted paint; reduce it to avoid joining unrelated edges.",
        )


class MergeDistance(Config):
    name: Literal["mergeDistance"] = "mergeDistance"
    value: float = Field(default=4.0, ge=0.0, le=1000.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Segment Merge Distance"
        json_schema_extra = _description(
            "Collinear consolidation",
            "Nearby, similarly oriented Hough segments within this pixel distance are merged. Set 0 to disable capsule-level merging.",
        )


class MaxOutputLines(Config):
    name: Literal["maxOutputLines"] = "maxOutputLines"
    value: int = Field(default=100, ge=1, le=1000)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Maximum Output Lines"
        json_schema_extra = _description(
            "Response-size limit",
            "Caps the number of longest merged segments returned to NovaVision so dense textures cannot create an unbounded response.",
        )


class GeneralLineProfile(Config):
    name: Literal["GeneralLines"] = "GeneralLines"
    value: Literal["GeneralLines"] = "GeneralLines"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "General Lines (Default)"
        json_schema_extra = _description(
            "Balanced straight-line detection",
            "Balanced Canny and probabilistic Hough values for ordinary straight segments in documents, industrial imagery, architecture, and roads.",
        )


class BrokenLineProfile(Config):
    name: Literal["BrokenLines"] = "BrokenLines"
    value: Literal["BrokenLines"] = "BrokenLines"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Broken / Dashed Lines"
        json_schema_extra = _description(
            "Interrupted-line recovery",
            "Uses lower vote and length requirements plus a larger gap tolerance. Suitable for worn paint, dash markings, and partially occluded edges.",
        )


class StrictLineProfile(Config):
    name: Literal["StrictLines"] = "StrictLines"
    value: Literal["StrictLines"] = "StrictLines"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Long, Strong Lines"
        json_schema_extra = _description(
            "Noise-resistant segments",
            "Requires longer and stronger evidence. Use for textured images that produce too many valid but unhelpful line candidates.",
        )


class RoadLaneProfile(Config):
    name: Literal["RoadLaneMarkings"] = "RoadLaneMarkings"
    value: Literal["RoadLaneMarkings"] = "RoadLaneMarkings"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Road Lane Markings"
        json_schema_extra = _description(
            "Road-specific lane-marking extraction",
            "Uses a lower-road trapezoid, white/yellow marking evidence, and perspective-consistent segment filtering. This profile suppresses trees, sky, and asphalt cracks, but is intentionally for forward-facing road scenes rather than arbitrary line images.",
        )


class CustomLineProfile(Config):
    name: Literal["CustomLineSettings"] = "CustomLineSettings"
    cannyLow: CannyLow = Field(default_factory=CannyLow)
    cannyHigh: CannyHigh = Field(default_factory=CannyHigh)
    houghThreshold: HoughThreshold = Field(default_factory=HoughThreshold)
    minLineLength: MinLineLength = Field(default_factory=MinLineLength)
    maxLineGap: MaxLineGap = Field(default_factory=MaxLineGap)
    mergeDistance: MergeDistance = Field(default_factory=MergeDistance)
    maxOutputLines: MaxOutputLines = Field(default_factory=MaxOutputLines)
    value: Literal["CustomLineSettings"] = "CustomLineSettings"
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Custom"
        json_schema_extra = _description(
            "Manual line settings",
            "Exposes every Canny, Hough, merge, and output-size control. Values are validated before OpenCV runs.",
        )


class LineProfile(Config):
    name: Literal["lineProfile"] = "lineProfile"
    value: Union[
        GeneralLineProfile,
        BrokenLineProfile,
        StrictLineProfile,
        RoadLaneProfile,
        CustomLineProfile,
    ] = Field(default_factory=GeneralLineProfile)
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Line Detection Profile"
        json_schema_extra = _description(
            "Preset or custom controls",
            "Choose a tuned profile for common sources, or select Custom to tune the underlying Canny and Hough parameters.",
        )


class CircleDp(Config):
    name: Literal["dp"] = "dp"
    value: float = Field(default=1.2, gt=0.0, le=10.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Accumulator Resolution (dp)"
        json_schema_extra = _description(
            "Circle accumulator scale",
            "Inverse image-to-accumulator resolution. Values near 1 are more precise but cost more computation.",
        )


class MinCircleDistance(Config):
    name: Literal["minDist"] = "minDist"
    value: float = Field(default=0.0, ge=0.0, le=100000.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Minimum Circle Distance"
        json_schema_extra = _description(
            "Minimum centre separation",
            "Minimum distance between circle centres in pixels. Set 0 to use the capsule's image-size-aware default.",
        )


class CircleEdgeThreshold(Config):
    name: Literal["param1"] = "param1"
    value: float = Field(default=100.0, gt=0.0, le=10000.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Circle Edge Threshold"
        json_schema_extra = _description(
            "Internal Canny high threshold",
            "HoughCircles' internal Canny high threshold. Lower it for weak circular edges; raise it to reject texture and shadows.",
        )


class CircleAccumulatorThreshold(Config):
    name: Literal["param2"] = "param2"
    value: float = Field(default=30.0, gt=0.0, le=10000.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Circle Accumulator Threshold"
        json_schema_extra = _description(
            "Circle acceptance threshold",
            "Lower values return more candidates, including false positives. Higher values require stronger circular evidence.",
        )


class MinRadius(Config):
    name: Literal["minRadius"] = "minRadius"
    value: int = Field(default=0, ge=0, le=100000)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Minimum Circle Radius"
        json_schema_extra = _description(
            "Smallest accepted circle",
            "Reject circles smaller than this pixel radius. Use 0 when the expected radius is unknown.",
        )


class MaxRadius(Config):
    name: Literal["maxRadius"] = "maxRadius"
    value: int = Field(default=0, ge=0, le=100000)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Maximum Circle Radius"
        json_schema_extra = _description(
            "Largest accepted circle",
            "Reject circles larger than this pixel radius. Use 0 to disable the upper radius limit.",
        )


class MaxOutputCircles(Config):
    name: Literal["maxOutputCircles"] = "maxOutputCircles"
    value: int = Field(default=100, ge=1, le=1000)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Maximum Output Circles"
        json_schema_extra = _description(
            "Response-size limit",
            "Caps the number of non-overlapping circles returned to NovaVision for dense images.",
        )


class GeneralCircleProfile(Config):
    name: Literal["GeneralCircles"] = "GeneralCircles"
    value: Literal["GeneralCircles"] = "GeneralCircles"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "General Circles (Default)"
        json_schema_extra = _description(
            "Balanced circle detection",
            "General HoughCircles defaults for circular edges with unknown radius and spacing.",
        )


class CoinCircleProfile(Config):
    name: Literal["CoinCircles"] = "CoinCircles"
    value: Literal["CoinCircles"] = "CoinCircles"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Coins / Circular Parts"
        json_schema_extra = _description(
            "Separated round objects",
            "Tuned for well-separated objects such as coins and industrial parts, favouring stronger circles and fewer near-duplicate candidates.",
        )


class CustomCircleProfile(Config):
    name: Literal["CustomCircleSettings"] = "CustomCircleSettings"
    dp: CircleDp = Field(default_factory=CircleDp)
    minDist: MinCircleDistance = Field(default_factory=MinCircleDistance)
    param1: CircleEdgeThreshold = Field(default_factory=CircleEdgeThreshold)
    param2: CircleAccumulatorThreshold = Field(default_factory=CircleAccumulatorThreshold)
    minRadius: MinRadius = Field(default_factory=MinRadius)
    maxRadius: MaxRadius = Field(default_factory=MaxRadius)
    maxOutputCircles: MaxOutputCircles = Field(default_factory=MaxOutputCircles)
    value: Literal["CustomCircleSettings"] = "CustomCircleSettings"
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Custom"
        json_schema_extra = _description(
            "Manual circle settings",
            "Exposes all HoughCircles controls for sources that do not suit the General or Coins / Circular Parts presets.",
        )


class CircleProfile(Config):
    name: Literal["circleProfile"] = "circleProfile"
    value: Union[
        GeneralCircleProfile,
        CoinCircleProfile,
        CustomCircleProfile,
    ] = Field(default_factory=GeneralCircleProfile)
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Circle Detection Profile"
        json_schema_extra = _description(
            "Preset or custom controls",
            "Choose a tuned profile for common circle sources, or select Custom to tune accumulator, distance, and radius settings.",
        )


class LineDetectionConfigs(Configs):
    lineProfile: LineProfile = Field(default_factory=LineProfile)


class CircleDetectionConfigs(Configs):
    circleProfile: CircleProfile = Field(default_factory=CircleProfile)


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
        json_schema_extra = _description(
            "Detection algorithm",
            "Choose Line Detection for straight segments using probabilistic Hough transform, or Circle Detection for circular objects using gradient Hough transform.",
        )


class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["capsule"] = "capsule"
    name: Literal["HoughTransform"] = "HoughTransform"

    @root_validator(pre=True)
    def normalize_empty_nested_configs(cls, values):
        """Normalize NovaVision's empty-list form before nested union parsing."""
        def normalize(value):
            if isinstance(value, dict):
                return {
                    key: (
                        {}
                        if key == "configs" and isinstance(item, list) and not item
                        else normalize(item)
                    )
                    for key, item in value.items()
                }
            if isinstance(value, list):
                return [normalize(item) for item in value]
            return value

        return normalize(values)
