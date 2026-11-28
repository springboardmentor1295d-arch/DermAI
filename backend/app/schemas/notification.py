from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict
from backend.app.models.notification import AlertSeverity


class NotificationResponse(BaseModel):
    id: str
    user_id: str
    title: str
    message: str
    severity: AlertSeverity
    is_read: bool
    link: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
