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
# SETTINGS
# ---------------------------------------------------------
# Each executor opens with these fields pre-filled with its defaults
# (the default values below). The user may change any of them.

class EdgeSensitivity(Config):
    """
    Scales the automatically calculated edge threshold. Above 100 keeps fainter edges, below 100 keeps only strong edges. 100 is the automatic value.
    """

    name: Literal["edgeSensitivityPct"] = "edgeSensitivityPct"
    value: float = Field(default=100.0, ge=10, le=500)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Edge Sensitivity (%)"
        json_schema_extra = {
            "shortDescription": "Edge detection sensitivity"
        }


class DenoiseStrength(Config):
    """
    Scales the automatic noise removal. 100 is the automatic value, 0 disables strong denoising.
    """

    name: Literal["denoiseStrengthPct"] = "denoiseStrengthPct"
    value: float = Field(default=100.0, ge=0, le=300)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Denoise Strength (%)"
        json_schema_extra = {
            "shortDescription": "Noise removal strength"
        }


class MinLineLengthPct(Config):
    """
    Shortest accepted line, as a percentage of the shorter image side.
    """

    name: Literal["minLineLengthPct"] = "minLineLengthPct"
    value: float = Field(default=8.0, ge=1, le=100)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Min Line Length (%)"
        json_schema_extra = {
            "shortDescription": "Shortest accepted line"
        }


class MaxGapPct(Config):
    """
    Largest gap that may be bridged inside one line, as a percentage of the shorter image side.
    """

    name: Literal["maxGapPct"] = "maxGapPct"
    value: float = Field(default=1.5, ge=0.1, le=100)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Max Gap (%)"
        json_schema_extra = {
            "shortDescription": "Largest gap inside one line"
        }


class MinDashFillPct(Config):
    """
    Minimum share of a broken line that must actually be visible.
    """

    name: Literal["minDashFillPct"] = "minDashFillPct"
    value: float = Field(default=22.0, ge=5, le=100)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Min Dash Fill (%)"
        json_schema_extra = {
            "shortDescription": "Visible part of a broken line"
        }


class MaxLines(Config):
    """
    Maximum number of lines returned.
    """

    name: Literal["maxLines"] = "maxLines"
    value: int = Field(default=200, ge=1, le=1000)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Max Lines"
        json_schema_extra = {
            "shortDescription": "Maximum number of lines"
        }


class MinRadiusPct(Config):
    """
    Smallest circle radius, as a percentage of the shorter image side.
    """

    name: Literal["minRadiusPct"] = "minRadiusPct"
    value: float = Field(default=1.5, ge=0.5, le=100)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Min Radius (%)"
        json_schema_extra = {
            "shortDescription": "Smallest circle radius"
        }


class MaxRadiusPct(Config):
    """
    Largest circle radius, as a percentage of the shorter image side.
    """

    name: Literal["maxRadiusPct"] = "maxRadiusPct"
    value: float = Field(default=50.0, ge=1, le=200)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Max Radius (%)"
        json_schema_extra = {
            "shortDescription": "Largest circle radius"
        }


class MinSupportPct(Config):
    """
    Share of the outline that must lie on real edges. Higher is stricter and removes false detections, lower finds fainter shapes.
    """

    name: Literal["minSupportPct"] = "minSupportPct"
    value: float = Field(default=60.0, ge=5, le=100)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Min Edge Support (%)"
        json_schema_extra = {
            "shortDescription": "Edge evidence required"
        }


class MinArcCoveragePct(Config):
    """
    Minimum share of the circle outline that must be visible.
    """

    name: Literal["minArcCoveragePct"] = "minArcCoveragePct"
    value: float = Field(default=25.0, ge=5, le=100)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Min Arc Coverage (%)"
        json_schema_extra = {
            "shortDescription": "Visible part of a circle"
        }


class MaxCircles(Config):
    """
    Maximum number of circles returned.
    """

    name: Literal["maxCircles"] = "maxCircles"
    value: int = Field(default=60, ge=1, le=500)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Max Circles"
        json_schema_extra = {
            "shortDescription": "Maximum number of circles"
        }


class MinAxisPct(Config):
    """
    Smallest ellipse half-axis, as a percentage of the shorter image side.
    """

    name: Literal["minAxisPct"] = "minAxisPct"
    value: float = Field(default=2.0, ge=0.5, le=50)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Min Axis (%)"
        json_schema_extra = {
            "shortDescription": "Smallest ellipse half-axis"
        }


class MaxEllipses(Config):
    """
    Maximum number of ellipses returned.
    """

    name: Literal["maxEllipses"] = "maxEllipses"
    value: int = Field(default=40, ge=1, le=500)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Max Ellipses"
        json_schema_extra = {
            "shortDescription": "Maximum number of ellipses"
        }


class MinAreaPct(Config):
    """
    Smallest shape size, as a percentage of the shorter image side.
    """

    name: Literal["minAreaPct"] = "minAreaPct"
    value: float = Field(default=3.0, ge=0.5, le=100)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Min Shape Size (%)"
        json_schema_extra = {
            "shortDescription": "Smallest shape side"
        }


class ApproxTolerancePct(Config):
    """
    How much the outline may be simplified when counting corners. Higher merges small corners.
    """

    name: Literal["approxTolerancePct"] = "approxTolerancePct"
    value: float = Field(default=2.0, ge=0.5, le=10)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Corner Tolerance (%)"
        json_schema_extra = {
            "shortDescription": "Outline simplification"
        }


class MaxShapes(Config):
    """
    Maximum number of shapes returned.
    """

    name: Literal["maxShapes"] = "maxShapes"
    value: int = Field(default=60, ge=1, le=500)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Max Shapes"
        json_schema_extra = {
            "shortDescription": "Maximum number of shapes"
        }


# ---------------------------------------------------------
# CONFIGS
# ---------------------------------------------------------

class LineDetectionConfigs(Configs):
    edgeSensitivityPct: EdgeSensitivity = Field(
        default_factory=lambda: EdgeSensitivity(value=100.0)
    )
    minLineLengthPct: MinLineLengthPct = Field(
        default_factory=lambda: MinLineLengthPct(value=8.0)
    )
    maxGapPct: MaxGapPct = Field(
        default_factory=lambda: MaxGapPct(value=1.5)
    )
    maxLines: MaxLines = Field(
        default_factory=lambda: MaxLines(value=200)
    )


class CircleDetectionConfigs(Configs):
    edgeSensitivityPct: EdgeSensitivity = Field(
        default_factory=lambda: EdgeSensitivity(value=100.0)
    )
    minRadiusPct: MinRadiusPct = Field(
        default_factory=lambda: MinRadiusPct(value=1.5)
    )
    maxRadiusPct: MaxRadiusPct = Field(
        default_factory=lambda: MaxRadiusPct(value=50.0)
    )
    minSupportPct: MinSupportPct = Field(
        default_factory=lambda: MinSupportPct(value=60.0)
    )
    maxCircles: MaxCircles = Field(
        default_factory=lambda: MaxCircles(value=60)
    )


class NoiseDetectionConfigs(Configs):
    denoiseStrengthPct: DenoiseStrength = Field(
        default_factory=lambda: DenoiseStrength(value=100.0)
    )
    edgeSensitivityPct: EdgeSensitivity = Field(
        default_factory=lambda: EdgeSensitivity(value=100.0)
    )
    minLineLengthPct: MinLineLengthPct = Field(
        default_factory=lambda: MinLineLengthPct(value=8.0)
    )
    minSupportPct: MinSupportPct = Field(
        default_factory=lambda: MinSupportPct(value=60.0)
    )


class BrokenDetectionConfigs(Configs):
    edgeSensitivityPct: EdgeSensitivity = Field(
        default_factory=lambda: EdgeSensitivity(value=100.0)
    )
    minLineLengthPct: MinLineLengthPct = Field(
        default_factory=lambda: MinLineLengthPct(value=10.0)
    )
    maxGapPct: MaxGapPct = Field(
        default_factory=lambda: MaxGapPct(value=30.0)
    )
    minDashFillPct: MinDashFillPct = Field(
        default_factory=lambda: MinDashFillPct(value=22.0)
    )
    minSupportPct: MinSupportPct = Field(
        default_factory=lambda: MinSupportPct(value=40.0)
    )


class PartialEdgeDetectionConfigs(Configs):
    edgeSensitivityPct: EdgeSensitivity = Field(
        default_factory=lambda: EdgeSensitivity(value=100.0)
    )
    minLineLengthPct: MinLineLengthPct = Field(
        default_factory=lambda: MinLineLengthPct(value=10.0)
    )
    minArcCoveragePct: MinArcCoveragePct = Field(
        default_factory=lambda: MinArcCoveragePct(value=25.0)
    )
    minSupportPct: MinSupportPct = Field(
        default_factory=lambda: MinSupportPct(value=15.0)
    )


class EllipseDetectionConfigs(Configs):
    edgeSensitivityPct: EdgeSensitivity = Field(
        default_factory=lambda: EdgeSensitivity(value=100.0)
    )
    minSupportPct: MinSupportPct = Field(
        default_factory=lambda: MinSupportPct(value=60.0)
    )
    minAxisPct: MinAxisPct = Field(
        default_factory=lambda: MinAxisPct(value=2.0)
    )
    maxEllipses: MaxEllipses = Field(
        default_factory=lambda: MaxEllipses(value=40)
    )


class ShapeDetectionConfigs(Configs):
    edgeSensitivityPct: EdgeSensitivity = Field(
        default_factory=lambda: EdgeSensitivity(value=100.0)
    )
    minAreaPct: MinAreaPct = Field(
        default_factory=lambda: MinAreaPct(value=3.0)
    )
    approxTolerancePct: ApproxTolerancePct = Field(
        default_factory=lambda: ApproxTolerancePct(value=2.0)
    )
    maxShapes: MaxShapes = Field(
        default_factory=lambda: MaxShapes(value=60)
    )


# ---------------------------------------------------------
# OUTPUTS
# ---------------------------------------------------------

class DetectionOutputs(Outputs):
    outputImage: OutputImage
    outputDetections: OutputDetections


# ---------------------------------------------------------
# REQUESTS
# ---------------------------------------------------------

class LineDetectionRequest(Request):
    inputs: Optional[PackageInputs] = None
    configs: LineDetectionConfigs = Field(default_factory=LineDetectionConfigs)

    @validator("configs", pre=True)
    def normalize_empty_configs(cls, value):
        # NovaVision serializes a task with no visible fields as [].
        return {} if isinstance(value, list) and not value else value

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class CircleDetectionRequest(Request):
    inputs: Optional[PackageInputs] = None
    configs: CircleDetectionConfigs = Field(default_factory=CircleDetectionConfigs)

    @validator("configs", pre=True)
    def normalize_empty_configs(cls, value):
        # NovaVision serializes a task with no visible fields as [].
        return {} if isinstance(value, list) and not value else value

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class NoiseDetectionRequest(Request):
    inputs: Optional[PackageInputs] = None
    configs: NoiseDetectionConfigs = Field(default_factory=NoiseDetectionConfigs)

    @validator("configs", pre=True)
    def normalize_empty_configs(cls, value):
        # NovaVision serializes a task with no visible fields as [].
        return {} if isinstance(value, list) and not value else value

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class BrokenDetectionRequest(Request):
    inputs: Optional[PackageInputs] = None
    configs: BrokenDetectionConfigs = Field(default_factory=BrokenDetectionConfigs)

    @validator("configs", pre=True)
    def normalize_empty_configs(cls, value):
        # NovaVision serializes a task with no visible fields as [].
        return {} if isinstance(value, list) and not value else value

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class PartialEdgeDetectionRequest(Request):
    inputs: Optional[PackageInputs] = None
    configs: PartialEdgeDetectionConfigs = Field(default_factory=PartialEdgeDetectionConfigs)

    @validator("configs", pre=True)
    def normalize_empty_configs(cls, value):
        # NovaVision serializes a task with no visible fields as [].
        return {} if isinstance(value, list) and not value else value

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class EllipseDetectionRequest(Request):
    inputs: Optional[PackageInputs] = None
    configs: EllipseDetectionConfigs = Field(default_factory=EllipseDetectionConfigs)

    @validator("configs", pre=True)
    def normalize_empty_configs(cls, value):
        # NovaVision serializes a task with no visible fields as [].
        return {} if isinstance(value, list) and not value else value

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class ShapeDetectionRequest(Request):
    inputs: Optional[PackageInputs] = None
    configs: ShapeDetectionConfigs = Field(default_factory=ShapeDetectionConfigs)

    @validator("configs", pre=True)
    def normalize_empty_configs(cls, value):
        # NovaVision serializes a task with no visible fields as [].
        return {} if isinstance(value, list) and not value else value

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


# ---------------------------------------------------------
# RESPONSES
# ---------------------------------------------------------

class DetectionResponse(Response):
    outputs: DetectionOutputs


# ---------------------------------------------------------
# EXECUTORS
# ---------------------------------------------------------


class LineDetectionExecutor(Config):
    """
    Finds straight lines with a standard Hough transform. Every edge pixel votes, then each peak is verified against the edge map and an LSD (a-contrario) check, and given its true start and end point.
    """

    name: Literal["LineDetection"] = "LineDetection"
    value: Union[LineDetectionRequest, DetectionResponse]
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
    Finds circles with gradient Hough voting, then verifies each one against the edge map and refines centre and radius by least squares.
    """

    name: Literal["CircleDetection"] = "CircleDetection"
    value: Union[CircleDetectionRequest, DetectionResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Circle Detection"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }



class NoiseDetectionExecutor(Config):
    """
    Finds lines and circles in noisy images. Estimates the noise level, denoises with non-local means and sets edge thresholds from the image itself.
    """

    name: Literal["NoiseDetection"] = "NoiseDetection"
    value: Union[NoiseDetectionRequest, DetectionResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Noise-Robust Detection"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }



class BrokenDetectionExecutor(Config):
    """
    Finds dashed, interrupted or fragmented lines and broken circle rims by bridging gaps along the voted line or circle.
    """

    name: Literal["BrokenDetection"] = "BrokenDetection"
    value: Union[BrokenDetectionRequest, DetectionResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Broken Line Detection"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }



class PartialEdgeDetectionExecutor(Config):
    """
    Finds lines and circular arcs that are only partly visible (occluded, cut by the frame or faded).
    """

    name: Literal["PartialEdgeDetection"] = "PartialEdgeDetection"
    value: Union[PartialEdgeDetectionRequest, DetectionResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Partial Edge Detection"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }



class EllipseDetectionExecutor(Config):
    """
    Finds ellipses by robust least-squares fitting of edge groups, verified by gradient support around the whole outline.
    """

    name: Literal["EllipseDetection"] = "EllipseDetection"
    value: Union[EllipseDetectionRequest, DetectionResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Ellipse Detection"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }



class ShapeDetectionExecutor(Config):
    """
    Finds closed shapes and names them: triangle, square, rectangle, pentagon, hexagon, octagon, circle or ellipse.
    """

    name: Literal["ShapeDetection"] = "ShapeDetection"
    value: Union[ShapeDetectionRequest, DetectionResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Shape Detection"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }



class ConfigExecutor(Config):
    """
    Select the detection task.

    Line, Circle, Noise, Broken, Partial edge, Ellipse and Shape detection.
    """

    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[LineDetectionExecutor, CircleDetectionExecutor, NoiseDetectionExecutor, BrokenDetectionExecutor, PartialEdgeDetectionExecutor, EllipseDetectionExecutor, ShapeDetectionExecutor]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Task"
        json_schema_extra = {
            "shortDescription": "Select the detection task"
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