from pydantic import BaseModel, Field
from typing import List, Any

class WorkflowField(BaseModel):
    name: str = Field(description="Exact parameter name from Composio schema")
    type: str = Field(description="string, boolean, integer, or number")
    description: str = Field(default="", description="Friendly description of what this parameter does")
    value: Any = Field(default="", description="Pre-filled value, or placeholder like {{step_N.key}}")

class WorkflowStep(BaseModel):
    tool_name: str = Field(description="Exact tool action slug (e.g. GMAIL_SEND_EMAIL)")
    step_description: str = Field(description="Short sentence explaining what this step does")
    fields: List[WorkflowField] = Field(default_factory=list)

class WorkflowStructure(BaseModel):
    name: str
    description: str
    steps: List[WorkflowStep] = Field(default_factory=list)
