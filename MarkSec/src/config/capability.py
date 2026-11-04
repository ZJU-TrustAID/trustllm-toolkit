
from enum import Enum, auto
from typing import List, Optional
from pydantic import BaseModel

class ModelRole(Enum):
    VICTIM = "victim"
    ATTACKER = "attacker"
    ORACLE = "oracle"

class ComponentCapability(BaseModel):
    """
    Metadata defining what a component (Attack/Watermark) can do and what it needs.
    """
    name: str
    type: str  # "watermark", "attack", "metric"
    supported_models: List[str] = [] # e.g. ["llama", "gpt2"] or ["any"]
    requires_gpu: bool = True
    supports_streaming: bool = False
    is_evaluate_only: bool = False # If true, cannot be used in generation phase
    
    # For attacks specifically
    required_roles: List[ModelRole] = []
    
    class Config:
        extra = "allow"

class CapabilityRegistry:
    _registry = {}

    @classmethod
    def register(cls, capability: ComponentCapability):
        cls._registry[(capability.type, capability.name)] = capability

    @classmethod
    def get(cls, type_: str, name: str) -> Optional[ComponentCapability]:
        return cls._registry.get((type_, name))

