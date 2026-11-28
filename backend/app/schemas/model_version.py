from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, ConfigDict


class ModelVersionResponse(BaseModel):
    id: str
    name: str
    version_tag: str
    architecture: str
    test_accuracy: Optional[float] = None
    test_balanced_acc: Optional[float] = None
    test_macro_f1: Optional[float] = None
    metrics_summary: Optional[Dict[str, Any]] = None
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ModelSetActiveRequest(BaseModel):
    model_version_id: str
