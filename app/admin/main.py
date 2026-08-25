# app/admin/main.py
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession
from pathlib import Path
from typing import List, Optional
import uuid
from datetime import datetime

from app.database import get_db
from app.database.models import ProcessingStatus
from app.database.repositories import (
    ExamRepository,
    ResourceRepository,
    SamplePaperRepository,
    SubjectRepository,
    ChapterRepository,
    TopicRepository,
    QuizRepository,
    BlueprintRepository,
    UserRepository
)
from app.services.storage import StorageService
from app.services.workers import WorkerService
from app.services.embeddings import EmbeddingService
from app.services.auth import authenticate_admin, create_access_token, decode_access_token
from app.schemas.admin import (
    AdminLoginRequest,
    AdminTokenResponse,
    ExamCreate,
    ExamUpdate,
    ExamResponse,
    ResourceResponse,
    ResourceUploadResponse,
    BlueprintResponse,
    CurriculumResponse,
    DashboardStats
)
from app.config.settings import settings

app = FastAPI(title="AI Exam Preparation Platform - Admin API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

_ADMIN_UI_PATH = Path(__file__).parent / "static" / "index.html"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def admin_ui():
    """Serves the admin UI (login + exam/resource/blueprint management) at /admin/.
    It's a static page that authenticates against /admin/api/auth/login and then
    calls the JSON endpoints below directly from the browser."""
    return _ADMIN_UI_PATH.read_text(encoding="utf-8")


async def verify_admin(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify the admin JWT issued by /api/auth/login. Returns the admin username."""
    username = decode_access_token(credentials.credentials)
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication credentials",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return username


async def _compute_and_store_exam_embedding(exam_repo: ExamRepository, exam) -> None:
    """Refresh the exam's embedding from its name/short_name/description/aliases (used for semantic exam resolution).
    Fetches aliases via an explicit query rather than exam.aliases — that relationship
    lazy-loads and would raise MissingGreenlet under the async ORM if touched here."""
    embedding_service = EmbeddingService()
    aliases = await exam_repo.get_aliases_for_exam(str(exam.id))
    alias_names = " ".join(a.alias for a in aliases)
    text_repr = f"{exam.name} {exam.short_name} {exam.description or ''} {alias_names}".strip()
    embedding = await embedding_service.embed_text(text_repr)
    if embedding:
        await exam_repo.update_embedding(str(exam.id), embedding)


@app.post("/api/auth/login", response_model=AdminTokenResponse)
async def admin_login(credentials: AdminLoginRequest):
    """Admin login. Issues a JWT bearer token for use on all other /api/* endpoints."""
    if not authenticate_admin(credentials.username, credentials.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )
    token = create_access_token(subject=credentials.username)
    return AdminTokenResponse(access_token=token, expires_in_minutes=settings.jwt_expire_minutes)


@app.get("/api/dashboard/stats", response_model=DashboardStats)
async def get_dashboard_stats(
    db: AsyncSession = Depends(get_db),
    admin: bool = Depends(verify_admin)
):
    """Get dashboard statistics"""
    exam_repo = ExamRepository(db)
    resource_repo = ResourceRepository(db)
    sample_paper_repo = SamplePaperRepository(db)
    subject_repo = SubjectRepository(db)
    chapter_repo = ChapterRepository(db)
    topic_repo = TopicRepository(db)
    quiz_repo = QuizRepository(db)
    user_repo = UserRepository(db)

    exams = await exam_repo.get_all()
    resources = await resource_repo.get_all()
    sample_papers = await sample_paper_repo.get_all()
    subjects = await subject_repo.get_all()
    chapters = await chapter_repo.get_all()
    topics = await topic_repo.get_all()
    quizzes = await quiz_repo.get_all()
    users = await user_repo.get_all()

    # Calculate processing stats
    processing_count = len([r for r in resources if r.status in ["processing", "extracting_text", "chunking", "embedding", "extracting_structure"]])
    completed_count = len([r for r in resources if r.status == "completed"])
    failed_count = len([r for r in resources if r.status == "failed"])
    quiz_attempts = len([q for q in quizzes if q.status in ("submitted", "evaluated")])

    return DashboardStats(
        total_exams=len(exams),
        total_resources=len(resources),
        total_sample_papers=len(sample_papers),
        processing_resources=processing_count,
        completed_resources=completed_count,
        failed_resources=failed_count,
        extracted_subjects=len(subjects),
        extracted_chapters=len(chapters),
        extracted_topics=len(topics),
        total_quizzes=len(quizzes),
        total_users=len(users),
        quiz_attempts=quiz_attempts
    )


@app.post("/api/exams", response_model=ExamResponse)
async def create_exam(
    exam_data: ExamCreate,
    db: AsyncSession = Depends(get_db),
    admin: bool = Depends(verify_admin)
):
    """Create a new exam"""
    exam_repo = ExamRepository(db)
    
    # Check if exam already exists
    existing = await exam_repo.get_by_short_name(exam_data.short_name.upper())
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Exam with short name {exam_data.short_name} already exists"
        )
    
    exam = await exam_repo.create(exam_data.model_dump())
    await _compute_and_store_exam_embedding(exam_repo, exam)
    return ExamResponse.model_validate(exam)


@app.get("/api/exams", response_model=List[ExamResponse])
async def get_exams(
    db: AsyncSession = Depends(get_db),
    admin: bool = Depends(verify_admin)
):
    """Get all exams"""
    exam_repo = ExamRepository(db)
    exams = await exam_repo.get_all()
    return [ExamResponse.model_validate(exam) for exam in exams]


@app.patch("/api/exams/{exam_id}", response_model=ExamResponse)
async def update_exam(
    exam_id: str,
    exam_data: ExamUpdate,
    db: AsyncSession = Depends(get_db),
    admin: bool = Depends(verify_admin)
):
    """Update an exam. Recomputes its semantic-matching embedding and invalidates any
    cached exam-resolution entries pointing at it, since the name/description/active
    state that those entries were resolved against may have changed."""
    from app.services.semantic_cache import SemanticCacheService
    from app.database.repositories import SemanticCacheRepository

    exam_repo = ExamRepository(db)
    exam = await exam_repo.update(exam_id, exam_data.model_dump(exclude_unset=True))
    if not exam:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Exam not found")

    await _compute_and_store_exam_embedding(exam_repo, exam)
    cache_service = SemanticCacheService(SemanticCacheRepository(db), EmbeddingService())
    await cache_service.invalidate_entity("exam", exam_id)

    return ExamResponse.model_validate(exam)


_ALLOWED_RESOURCE_EXTENSIONS = {".pdf", ".doc", ".docx", ".txt"}


def _validate_upload_filename(filename: str) -> None:
    """Rejects missing/unsupported extensions and path-traversal attempts (spec 28)."""
    import os
    if not filename or os.path.basename(filename) != filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in _ALLOWED_RESOURCE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(_ALLOWED_RESOURCE_EXTENSIONS))}"
        )


@app.post("/api/exams/{exam_id}/resources")
async def upload_resource(
    exam_id: str,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    storage: StorageService = Depends(),
    worker: WorkerService = Depends(),
    admin: bool = Depends(verify_admin)
):
    """Upload a resource for an exam"""
    _validate_upload_filename(file.filename)

    # Check file size
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)

    if file_size > settings.max_file_size:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds maximum of {settings.max_file_size / 1024 / 1024}MB"
        )

    # Save file and compute its checksum (also used to detect re-uploads of identical content)
    import hashlib
    content = await file.read()
    checksum = hashlib.sha256(content).hexdigest()
    await file.seek(0)

    file_path = await storage.save_file(
        file=file,
        subdirectory=f"exams/{exam_id}/resources"
    )

    # Create resource record
    resource_repo = ResourceRepository(db)
    resource = await resource_repo.create({
        "exam_id": uuid.UUID(exam_id),
        "filename": file.filename,
        "file_path": file_path,
        "file_size": file_size,
        "mime_type": file.content_type or "application/octet-stream",
        "checksum": checksum,
        "status": ProcessingStatus.UPLOADED
    })

    # Queue processing job
    await worker.queue_resource_processing(str(resource.id))

    return ResourceUploadResponse(
        id=str(resource.id),
        filename=file.filename,
        status=resource.status
    )


@app.get("/api/exams/{exam_id}/resources", response_model=List[ResourceResponse])
async def get_resources(
    exam_id: str,
    db: AsyncSession = Depends(get_db),
    admin: bool = Depends(verify_admin)
):
    """Get all resources for an exam"""
    resource_repo = ResourceRepository(db)
    resources = await resource_repo.get_by_exam(exam_id)
    return [ResourceResponse.model_validate(r) for r in resources]


@app.delete("/api/exams/{exam_id}/resources/{resource_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_resource(
    exam_id: str,
    resource_id: str,
    db: AsyncSession = Depends(get_db),
    storage: StorageService = Depends(),
    admin: bool = Depends(verify_admin)
):
    """Delete a resource, its stored file, and its derived chunks/curriculum references"""
    resource_repo = ResourceRepository(db)
    resource = await resource_repo.get_by_id(resource_id)
    if not resource or str(resource.exam_id) != exam_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Resource not found")

    await storage.delete_file(resource.file_path)
    await resource_repo.delete(resource_id)  # cascades to documents/chunks via FK ondelete


@app.post("/api/exams/{exam_id}/resources/{resource_id}/reprocess")
async def reprocess_resource(
    exam_id: str,
    resource_id: str,
    db: AsyncSession = Depends(get_db),
    worker: WorkerService = Depends(),
    admin: bool = Depends(verify_admin)
):
    """Re-run the document intelligence pipeline for a resource (e.g. after a parsing fix)"""
    resource_repo = ResourceRepository(db)
    resource = await resource_repo.get_by_id(resource_id)
    if not resource or str(resource.exam_id) != exam_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Resource not found")

    await resource_repo.update_status(resource_id, ProcessingStatus.QUEUED)
    await worker.queue_resource_processing(resource_id)
    return {"status": "queued"}


@app.post("/api/exams/{exam_id}/sample-papers")
async def upload_sample_paper(
    exam_id: str,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    storage: StorageService = Depends(),
    worker: WorkerService = Depends(),
    admin: bool = Depends(verify_admin)
):
    """Upload a sample paper for an exam"""
    _validate_upload_filename(file.filename)

    import hashlib
    content = await file.read()
    checksum = hashlib.sha256(content).hexdigest()
    file_size = len(content)
    await file.seek(0)

    if file_size > settings.max_file_size:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds maximum of {settings.max_file_size / 1024 / 1024}MB"
        )

    file_path = await storage.save_file(
        file=file,
        subdirectory=f"exams/{exam_id}/sample-papers"
    )

    sample_paper_repo = SamplePaperRepository(db)
    sample_paper = await sample_paper_repo.create({
        "exam_id": uuid.UUID(exam_id),
        "filename": file.filename,
        "file_path": file_path,
        "file_size": file_size,
        "mime_type": file.content_type or "application/octet-stream",
        "checksum": checksum,
        "status": ProcessingStatus.UPLOADED
    })

    # Queue question extraction + blueprint (re)generation for this exam
    await worker.queue_blueprint_analysis(str(sample_paper.id))

    return {"id": str(sample_paper.id), "filename": file.filename, "status": sample_paper.status}


@app.get("/api/exams/{exam_id}/sample-papers")
async def get_sample_papers(
    exam_id: str,
    db: AsyncSession = Depends(get_db),
    admin: bool = Depends(verify_admin)
):
    """List sample papers for an exam"""
    sample_paper_repo = SamplePaperRepository(db)
    papers = await sample_paper_repo.get_by_exam(exam_id)
    return [
        {
            "id": str(p.id),
            "filename": p.filename,
            "status": p.status,
            "year": p.year,
            "created_at": p.created_at
        }
        for p in papers
    ]


@app.delete("/api/exams/{exam_id}/sample-papers/{paper_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_sample_paper(
    exam_id: str,
    paper_id: str,
    db: AsyncSession = Depends(get_db),
    storage: StorageService = Depends(),
    admin: bool = Depends(verify_admin)
):
    """Delete a sample paper and its extracted questions"""
    sample_paper_repo = SamplePaperRepository(db)
    paper = await sample_paper_repo.get_by_id(paper_id)
    if not paper or str(paper.exam_id) != exam_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sample paper not found")

    await storage.delete_file(paper.file_path)
    await sample_paper_repo.delete(paper_id)  # cascades to sample_questions via FK ondelete


@app.get("/api/exams/{exam_id}/blueprint", response_model=BlueprintResponse)
async def get_exam_blueprint(
    exam_id: str,
    db: AsyncSession = Depends(get_db),
    admin: bool = Depends(verify_admin)
):
    """Get the exam blueprint"""
    blueprint_repo = BlueprintRepository(db)
    blueprint = await blueprint_repo.get_active_by_exam(exam_id)
    
    if not blueprint:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No blueprint found for this exam"
        )
    
    return BlueprintResponse(
        id=str(blueprint.id),
        exam_id=str(blueprint.exam_id),
        version=blueprint.version,
        blueprint_data=blueprint.blueprint_data,
        generated_at=blueprint.generated_at
    )


@app.get("/api/exams/{exam_id}/curriculum", response_model=CurriculumResponse)
async def get_curriculum(
    exam_id: str,
    db: AsyncSession = Depends(get_db),
    admin: bool = Depends(verify_admin)
):
    """Get the curriculum hierarchy"""
    subject_repo = SubjectRepository(db)
    chapter_repo = ChapterRepository(db)
    topic_repo = TopicRepository(db)
    
    subjects = await subject_repo.get_by_exam(exam_id)
    
    curriculum = []
    for subject in subjects:
        subject_data = {
            "id": str(subject.id),
            "name": subject.name,
            "chapters": []
        }
        
        chapters = await chapter_repo.get_by_subject(str(subject.id))
        for chapter in chapters:
            chapter_data = {
                "id": str(chapter.id),
                "name": chapter.name,
                "topics": []
            }
            
            topics = await topic_repo.get_by_chapter(str(chapter.id))
            chapter_data["topics"] = [
                {"id": str(t.id), "name": t.name}
                for t in topics
            ]
            
            subject_data["chapters"].append(chapter_data)
        
        curriculum.append(subject_data)
    
    return CurriculumResponse(
        exam_id=exam_id,
        subjects=curriculum
    )


@app.post("/api/exams/{exam_id}/blueprint/regenerate")
async def regenerate_blueprint(
    exam_id: str,
    db: AsyncSession = Depends(get_db),
    worker: WorkerService = Depends(),
    admin: bool = Depends(verify_admin)
):
    """Regenerate the exam blueprint"""
    await worker.queue_blueprint_generation(exam_id)
    return {"status": "queued"}


@app.get("/api/exams/{exam_id}/processing-status")
async def get_processing_status(
    exam_id: str,
    db: AsyncSession = Depends(get_db),
    admin: bool = Depends(verify_admin)
):
    """Get processing status of all resources"""
    resource_repo = ResourceRepository(db)
    resources = await resource_repo.get_by_exam(exam_id)
    
    return {
        "total": len(resources),
        "statuses": [
            {"id": str(r.id), "filename": r.filename, "status": r.status}
            for r in resources
        ]
    }