from pydantic import BaseModel
from typing import Any, Dict, Optional, Literal


class Parameters(BaseModel):
    """Represents the definition of a single function parameter.

    Attributes:
        type: The data type allowed for the parameter.
        description: A brief explanation of the parameter.
    """

    type: Literal["number", "string", "boolean", "integer"]
    description: Optional[str] = None


class Returns(BaseModel):
    """Defines the return value of a function.

    Attributes:
        type: The expected return data type.
        description: A brief explanation of the returned value.
    """

    type: Literal["number", "string", "boolean", "integer"]
    description: Optional[str] = None


class Function(BaseModel):
    """A complete definition of a callable tool/function.

    Attributes:
        name: The unique identifier of the function.
        description: What the function does.
        parameters: Mapping of parameter names to their definitions.
        returns: The definition of the return value.
    """

    name: str
    description: str
    parameters: Dict[str, Parameters]
    returns: Returns


class Calling(BaseModel):
    """A user query to be processed by the function calling engine.

    Attributes:
        prompt: The natural language query.
    """

    prompt: str


class FunctionCallResult(BaseModel):
    """Result of a function call prediction.

    Attributes:
        prompt: The original natural-language request.
        name: The name of the function to call.
        parameters: All required arguments with correct types.
    """

    prompt: str
    name: str
    parameters: Dict[str, Any]
