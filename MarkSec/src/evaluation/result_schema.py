from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class DetectionResult(BaseModel):
    """Result of a watermark detection."""
    prediction: Optional[bool] = None
    z_score: Optional[float] = None

class SampleRecord(BaseModel):
    """
    Record for a single sample in the experiment.
    Contains text generations, detection results, and quality metrics.
    """
    sample_index: int
    prompt: str
    status: str = "success"  # "success" or "failed"

    # Text Content
    natural_text: str = ""
    full_natural_text: str = ""
    source_text: str = ""
    reference: str = ""
    watermarked_text: str = ""
    full_watermarked_text: str = ""
    
    # Optional fields depending on experiment configuration
    attacked_text: Optional[str] = None
    full_attacked_text: Optional[str] = None
    spoof_text: Optional[str] = None
    full_spoof_text: Optional[str] = None
    error_msg: Optional[str] = None

    # Detection Results
    detection_natural: Optional[DetectionResult] = None
    detection_watermark: Optional[DetectionResult] = None
    detection_attack: Optional[DetectionResult] = None
    detection_spoof: Optional[DetectionResult] = None

    # Quality Metrics (e.g., PPL, BLEU)
    quality_metrics: Dict[str, Any] = Field(default_factory=dict)

    # Extensible fields for future use or custom data
    extras: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SampleRecord":
        """
        Create a SampleRecord from a dictionary, handling backward compatibility.
        """
        # Separate known fields from extras
        known_fields = cls.model_fields.keys()
        extras = {}
        init_data = {}
        
        for k, v in data.items():
            if k in known_fields:
                if v is None and k in ["full_natural_text", "natural_text", "source_text", "reference", "watermarked_text", "full_watermarked_text"]:
                    init_data[k] = ""
                else:
                    init_data[k] = v
            else:
                extras[k] = v
        
        if "extras" in init_data:
            extras.update(init_data.pop("extras"))
            
        return cls(**init_data, extras=extras)

class ExperimentReport(BaseModel):
    """
    Complete report of an experiment containing multiple sample records.
    """
    schema_version: str = "1.0.0"
    records: List[SampleRecord] = Field(default_factory=list)
