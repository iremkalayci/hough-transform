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


# ---------------------------------------------------------
# LINE DETECTION PARAMETERS
# ---------------------------------------------------------

class CannyLow(Config):
    """
    Sets the lower Canny edge threshold.

    Lower values preserve faint edges but may also introduce
    texture noise. This value must remain below Canny High Threshold.
    """

    name: Literal["cannyLow"] = "cannyLow"
    value: int = Field(default=50, ge=0, le=254)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Canny Low Threshold"
        json_schema_extra = {
            "shortDescription": "Weak-edge threshold"
        }


class CannyHigh(Config):
    """
    Sets the upper Canny edge threshold.

    Higher values suppress weaker edges, while lower values
    can recover lines with low contrast.
    """

    name: Literal["cannyHigh"] = "cannyHigh"
    value: int = Field(default=150, ge=1, le=255)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Canny High Threshold"
        json_schema_extra = {
            "shortDescription": "Strong-edge threshold"
        }


class HoughThreshold(Config):
    """
    Sets the minimum Hough accumulator votes required to accept a line.

    Higher values return fewer, stronger candidates.
    Lower values can recover faint or short lines but may increase false detections.
    """

    name: Literal["houghThreshold"] = "houghThreshold"
    value: int = Field(default=50, ge=1, le=1000)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Hough Vote Threshold"
        json_schema_extra = {
            "shortDescription": "Minimum accumulator votes"
        }


class MinLineLength(Config):
    """
    Sets the minimum accepted Hough line segment length in pixels.

    Increase this value to suppress short texture edges.
    Decrease it to retain smaller line segments.
    """

    name: Literal["minLineLength"] = "minLineLength"
    value: int = Field(default=30, ge=1, le=10000)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Minimum Line Length"
        json_schema_extra = {
            "shortDescription": "Shortest accepted segment"
        }


class MaxLineGap(Config):
    """
    Sets the maximum gap in pixels that can be connected
    between points of a line during Hough detection.

    Larger values help recover interrupted lines but may
    connect unrelated segments.
    """

    name: Literal["maxLineGap"] = "maxLineGap"
    value: int = Field(default=20, ge=0, le=10000)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Maximum Line Gap"
        json_schema_extra = {
            "shortDescription": "Broken-line linking"
        }


class MergeDistance(Config):
    """
    Sets the maximum perpendicular distance used to merge
    nearby and similarly oriented line segments.

    A value of zero disables capsule-level segment merging.
    """

    name: Literal["mergeDistance"] = "mergeDistance"
    value: float = Field(default=4.0, ge=0.0, le=1000.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Segment Merge Distance"
        json_schema_extra = {
            "shortDescription": "Collinear consolidation"
        }


class MaxOutputLines(Config):
    """
    Limits the maximum number of detected line segments returned.

    This prevents dense images from producing unnecessarily
    large detection responses.
    """

    name: Literal["maxOutputLines"] = "maxOutputLines"
    value: int = Field(default=100, ge=1, le=1000)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Maximum Output Lines"
        json_schema_extra = {
            "shortDescription": "Response-size limit"
        }


# ---------------------------------------------------------
# LINE DETECTION PROFILES
# ---------------------------------------------------------

class GeneralLineProfile(Config):
    """
    Detects straight lines across the image using balanced
    Canny and probabilistic Hough settings.

    Suitable for documents, diagrams, architecture,
    and general industrial images.
    """

    name: Literal["GeneralLines"] = "GeneralLines"
    value: Literal["GeneralLines"] = "GeneralLines"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "General Lines"
        json_schema_extra = {
            "shortDescription": "Balanced straight-line detection"
        }


class BrokenLineProfile(Config):
    """
    Uses more sensitive detection settings to recover short,
    interrupted, and dashed line segments.

    Suitable for fragmented markings and partially visible lines.
    Sensitive settings may also increase false positives.
    """

    name: Literal["BrokenLines"] = "BrokenLines"
    value: Literal["BrokenLines"] = "BrokenLines"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Broken and Dashed Lines"
        json_schema_extra = {
            "shortDescription": "Detection of interrupted lines"
        }


class StrictLineProfile(Config):
    """
    Prioritizes long and prominent line segments using
    stricter detection settings.

    Suitable for noisy or textured images, although
    faint and short lines may be missed.
    """

    name: Literal["StrictLines"] = "StrictLines"
    value: Literal["StrictLines"] = "StrictLines"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Long and Strong Lines"
        json_schema_extra = {
            "shortDescription": "Detection of prominent lines"
        }


class RoadLaneProfile(Config):
    """
    Focuses on road lane markings in forward-facing road images.

    Uses road-region, marking-colour, and perspective-related
    filtering to identify lane boundaries and markings.
    Results may vary with lighting, faded paint, shadows, and camera angle.
    """

    name: Literal["RoadLaneMarkings"] = "RoadLaneMarkings"
    value: Literal["RoadLaneMarkings"] = "RoadLaneMarkings"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Road Lane Markings"
        json_schema_extra = {
            "shortDescription": "Road lane and boundary detection"
        }


class CustomLineProfile(Config):
    """
    Provides manual control of Canny thresholds, Hough votes,
    minimum line length, maximum line gap, segment merging,
    and maximum output lines.

    Use this profile when the predefined settings do not
    match the characteristics of the input image.
    """

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
        json_schema_extra = {
            "shortDescription": "Manual line detection settings"
        }


class LineProfile(Config):
    """
    Select the line detection profile that best matches the input image.

    - General Lines: Balanced, general-purpose straight-line detection.
    - Broken and Dashed Lines: Detection of interrupted line segments.
    - Long and Strong Lines: Prioritizes prominent and longer lines.
    - Road Lane Markings: Focuses on road lanes and boundaries.
    - Custom: Provides manual control of detection parameters.
    """

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
        json_schema_extra = {
            "shortDescription": "Select a line detection profile"
        }


# ---------------------------------------------------------
# CIRCLE DETECTION PARAMETERS
# ---------------------------------------------------------

class CircleDp(Config):
    """
    Controls the ratio between input image resolution
    and Hough circle accumulator resolution.

    Values closer to one provide finer detection
    at a higher computational cost.
    """

    name: Literal["dp"] = "dp"
    value: float = Field(default=1.2, gt=0.0, le=10.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Accumulator Resolution (dp)"
        json_schema_extra = {
            "shortDescription": "Circle accumulator scale"
        }


class MinCircleDistance(Config):
    """
    Sets the minimum distance in pixels between detected circle centres.

    Larger values prevent nearby circles from being detected
    too close together. A value of zero uses the capsule's
    automatic image-size-aware default.
    """

    name: Literal["minDist"] = "minDist"
    value: float = Field(default=0.0, ge=0.0, le=100000.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Minimum Circle Distance"
        json_schema_extra = {
            "shortDescription": "Minimum centre separation"
        }


class CircleEdgeThreshold(Config):
    """
    Sets the upper edge threshold used internally by HoughCircles
    for Canny edge detection.

    Lower values can recover weak circular edges but may
    increase false candidates.
    """

    name: Literal["param1"] = "param1"
    value: float = Field(default=100.0, gt=0.0, le=10000.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Circle Edge Threshold"
        json_schema_extra = {
            "shortDescription": "Internal Canny high threshold"
        }


class CircleAccumulatorThreshold(Config):
    """
    Sets the accumulator threshold for accepting circle candidates.

    Lower values detect more candidates but may increase
    false positives. Higher values require stronger circular evidence.
    """

    name: Literal["param2"] = "param2"
    value: float = Field(default=30.0, gt=0.0, le=10000.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Circle Accumulator Threshold"
        json_schema_extra = {
            "shortDescription": "Circle acceptance threshold"
        }


class MinRadius(Config):
    """
    Sets the minimum accepted circle radius in pixels.

    Use zero when no lower radius limit is required.
    """

    name: Literal["minRadius"] = "minRadius"
    value: int = Field(default=0, ge=0, le=100000)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Minimum Circle Radius"
        json_schema_extra = {
            "shortDescription": "Smallest accepted circle"
        }


class MaxRadius(Config):
    """
    Sets the maximum accepted circle radius in pixels.

    Use zero to leave the upper radius unrestricted.
    """

    name: Literal["maxRadius"] = "maxRadius"
    value: int = Field(default=0, ge=0, le=100000)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Maximum Circle Radius"
        json_schema_extra = {
            "shortDescription": "Largest accepted circle"
        }


class MaxOutputCircles(Config):
    """
    Limits the maximum number of accepted circles returned.

    This prevents dense images from producing unnecessarily
    large detection responses.
    """

    name: Literal["maxOutputCircles"] = "maxOutputCircles"
    value: int = Field(default=100, ge=1, le=1000)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Maximum Output Circles"
        json_schema_extra = {
            "shortDescription": "Response-size limit"
        }


# ---------------------------------------------------------
# CIRCLE DETECTION PROFILES
# ---------------------------------------------------------

class GeneralCircleProfile(Config):
    """
    Detects circular shapes with balanced HoughCircles settings.

    Suitable when circle sizes and spacing are not known
    in advance.
    """

    name: Literal["GeneralCircles"] = "GeneralCircles"
    value: Literal["GeneralCircles"] = "GeneralCircles"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "General Circles"
        json_schema_extra = {
            "shortDescription": "General-purpose circle detection"
        }


class CoinCircleProfile(Config):
    """
    Targets clearly separated circular objects such as coins
    and round industrial parts.

    Touching or heavily overlapping circles may be harder to distinguish.
    """

    name: Literal["CoinCircles"] = "CoinCircles"
    value: Literal["CoinCircles"] = "CoinCircles"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Coins and Circular Objects"
        json_schema_extra = {
            "shortDescription": "Detection of separate circular objects"
        }


class CustomCircleProfile(Config):
    """
    Provides manual control of Hough circle detection parameters.

    Configure accumulator resolution, centre distance, edge threshold,
    circle acceptance threshold, radius limits, and maximum output count.
    """

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
        json_schema_extra = {
            "shortDescription": "Manual circle detection settings"
        }


class CircleProfile(Config):
    """
    Select the circle detection profile that best matches the input image.

    - General Circles: Balanced, general-purpose circle detection.
    - Coins and Circular Objects: Focuses on separate round objects.
    - Custom: Provides manual control of Hough circle parameters.
    """

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
        json_schema_extra = {
            "shortDescription": "Select a circle detection profile"
        }


# ---------------------------------------------------------
# TASK CONFIGURATION
# ---------------------------------------------------------

class LineDetectionConfigs(Configs):
    lineProfile: LineProfile = Field(default_factory=LineProfile)


class CircleDetectionConfigs(Configs):
    circleProfile: CircleProfile = Field(default_factory=CircleProfile)


# ---------------------------------------------------------
# OUTPUTS
# ---------------------------------------------------------

class LineDetectionOutputs(Outputs):
    outputImage: OutputImage
    outputDetections: OutputDetections


class CircleDetectionOutputs(Outputs):
    outputImage: OutputImage
    outputDetections: OutputDetections


# ---------------------------------------------------------
# REQUESTS
# ---------------------------------------------------------

class LineDetectionRequest(Request):
    inputs: Optional[PackageInputs] = None
    configs: LineDetectionConfigs

    @validator("configs", pre=True)
    def normalize_empty_configs(cls, value):
        # NovaVision serializes a task with no visible tuning fields as [].
        return {} if isinstance(value, list) and not value else value

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class CircleDetectionRequest(Request):
    inputs: Optional[PackageInputs] = None
    configs: CircleDetectionConfigs

    @validator("configs", pre=True)
    def normalize_empty_configs(cls, value):
        return {} if isinstance(value, list) and not value else value

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


# ---------------------------------------------------------
# RESPONSES
# ---------------------------------------------------------

class LineDetectionResponse(Response):
    outputs: LineDetectionOutputs


class CircleDetectionResponse(Response):
    outputs: CircleDetectionOutputs


# ---------------------------------------------------------
# EXECUTORS
# ---------------------------------------------------------

class LineDetectionExecutor(Config):
    """
    Runs the line detection algorithm.

    Uses the selected Line Detection Profile and returns
    the annotated image and detected line segments.
    """

    name: Literal["LineDetection"] = "LineDetection"
    value: Union[LineDetectionRequest, LineDetectionResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Line Detection"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


class CircleDetectionExecutor(Config):
    """
    Runs the circle detection algorithm.

    Uses the selected Circle Detection Profile and returns
    the annotated image and detected circular objects.
    """

    name: Literal["CircleDetection"] = "CircleDetection"
    value: Union[CircleDetectionRequest, CircleDetectionResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Circle Detection"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


class ConfigExecutor(Config):
    """
    Select the geometric feature detection task.

    - Line Detection: Finds straight segments using probabilistic Hough transform.
    - Circle Detection: Finds circular shapes using gradient Hough transform.
    """

    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[LineDetectionExecutor, CircleDetectionExecutor]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Task"
        json_schema_extra = {
            "shortDescription": "Select the detection algorithm"
        }


# ---------------------------------------------------------
# PACKAGE MODEL
# ---------------------------------------------------------

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