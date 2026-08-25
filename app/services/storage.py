import os
import shutil
import hashlib
from pathlib import Path
from typing import Optional
from fastapi import UploadFile
from app.config.settings import settings
import aiofiles
import logging

logger = logging.getLogger(__name__)

class StorageService:
    def __init__(self):
        self.storage_path = Path(settings.storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    async def save_file(
        self,
        file: UploadFile,
        subdirectory: str = "uploads",
        filename: Optional[str] = None
    ) -> str:
        """Save a file to storage"""
        try:
            # Create full path
            full_path = self.storage_path / subdirectory
            full_path.mkdir(parents=True, exist_ok=True)
            
            # Generate filename if not provided
            if not filename:
                # Generate hash of content
                content = await file.read()
                file_hash = hashlib.sha256(content).hexdigest()[:16]
                filename = f"{file_hash}_{file.filename}"
                # Reset file position
                await file.seek(0)
            
            # Save file
            file_path = full_path / filename
            async with aiofiles.open(file_path, "wb") as f:
                content = await file.read()
                await f.write(content)
                # Reset file position
                await file.seek(0)
            
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to save file: {e}")
            raise

    async def delete_file(self, file_path: str) -> bool:
        """Delete a file from storage"""
        try:
            path = Path(file_path)
            if path.exists():
                path.unlink()
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete file: {e}")
            return False

    async def get_file_size(self, file_path: str) -> int:
        """Get file size"""
        try:
            path = Path(file_path)
            return path.stat().st_size if path.exists() else 0
        except Exception:
            return 0