from pydantic import BaseModel
from typing import Dict, Optional, Literal


class Parameters(BaseModel):
    """
    Represents the definition of a single function parameter.
    Attributes:
        type: The data type allowed (number, string, boolean, or integer).
        description: A brief explanation of what the parameter represents.
    """
    type: Literal["number", "string", "boolean", "integer"]
    description: Optional[str] = None


class Returns(BaseModel):
    """
    Defines the structure and type of the value returned by a function.
    Attributes:
        type: The expected return data type.
        description: A brief explanation of the returned value.
    """
    type: Literal["number", "string", "boolean", "integer"]
    description: Optional[str] = None


class Function(BaseModel):
    """
    A complete definition of a tool/function available for the LLM.
    Attributes:
        name: The unique identifier of the function.
        description: Detailed information on what the function does.
        parameters: A mapping of parameter names to their definitions.
        returns: The definition of the return value.
    """
    name: str
    description: str
    parameters: Dict[str, Parameters]
    returns: Returns


class Calling(BaseModel):
    """
    Represents a test case or a user query to be processed.
    Attributes:
        prompt: The natural language string describing the task for the LLM.
    """
    prompt: str
