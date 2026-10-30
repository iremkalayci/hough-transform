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




