from pydantic import BaseModel
from typing import Dict, Optional, Literal


class Parameters(BaseModel):
    type: Literal["number", "string", "boolean"]
    description: Optional[str] = None


class Returns(BaseModel):
    type: Literal["number", "string", "boolean"]
    description: Optional[str] = None


class Function(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Parameters]
    returns: Returns


class Calling(BaseModel):
    prompt: str
