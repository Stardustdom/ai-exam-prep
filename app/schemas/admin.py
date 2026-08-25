from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID

class AdminLoginRequest(BaseModel):
    username: str
    password: str

class AdminTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int

class ExamCreate(BaseModel):
    name: str
    short_name: str
    description: Optional[str] = None
    subjects: Optional[List[str]] = None
    is_active: bool = True

class ExamUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    subjects: Optional[List[str]] = None
    is_active: Optional[bool] = None

class ExamResponse(BaseModel):
    id: UUID
    name: str
    short_name: str
    description: Optional[str]
    is_active: bool
    subjects: Optional[List[str]]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class ResourceResponse(BaseModel):
    id: UUID
    filename: str
    file_size: int
    mime_type: str
    status: str
    status_message: Optional[str]
    processed_at: Optional[datetime]
    created_at: datetime

    class Config:
        from_attributes = True

class ResourceUploadResponse(BaseModel):
    id: str
    filename: str
    status: str

class BlueprintResponse(BaseModel):
    id: str
    exam_id: str
    version: int
    blueprint_data: Dict[str, Any]
    generated_at: datetime

class CurriculumResponse(BaseModel):
    exam_id: str
    subjects: List[Dict[str, Any]]

class DashboardStats(BaseModel):
    total_exams: int
    total_resources: int
    total_sample_papers: int
    processing_resources: int
    completed_resources: int
    failed_resources: int
    extracted_subjects: int
    extracted_chapters: int
    extracted_topics: int
    total_quizzes: int
    total_users: int
    quiz_attempts: int