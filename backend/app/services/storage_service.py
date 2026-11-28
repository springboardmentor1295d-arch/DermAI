import io
import uuid
from pathlib import Path
from typing import Optional, Tuple, Union
from PIL import Image
from fastapi import HTTPException, status

from backend.app.core.config import settings
from backend.app.core.logging import logger

ALLOWED_IMAGE_MIME_TYPES = ["image/jpeg", "image/png", "image/webp", "image/jpg"]


class StorageService:
    """Service handling validation, persistence, and cleanup of case images."""

    @classmethod
    def validate_image(
        cls,
        content: bytes,
        filename: str,
        content_type: Optional[str] = None
    ) -> Tuple[int, int, str]:
        """
        Validates:
        1. Non-empty file content
        2. Extension matches allowed list (.png, .jpg, .jpeg, .webp)
        3. Content-Type (if provided) matches allowed image MIME types
        4. Maximum file size (15MB)
        5. Actual image decoding integrity via PIL (catches corrupted or malformed files)
        Returns: (width, height, resolved_mime_type)
        """
        if not content or len(content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty"
            )

        if len(content) > settings.MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File size exceeds {settings.MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)}MB limit"
            )

        ext = Path(filename or "").suffix.lower()
        if ext not in settings.ALLOWED_IMAGE_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported image format '{ext}'. Allowed: {settings.ALLOWED_IMAGE_EXTENSIONS}"
            )

        if content_type:
            ct = content_type.lower().split(";")[0].strip()
            if ct not in ALLOWED_IMAGE_MIME_TYPES:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported image MIME type '{content_type}'"
                )

        # Actual image decoding check
        try:
            with Image.open(io.BytesIO(content)) as pil_img:
                pil_img.verify()

            # verify() leaves the stream pointer in an indeterminate state; reopen to read dimensions and format
            with Image.open(io.BytesIO(content)) as pil_img:
                width, height = pil_img.size
                img_format = pil_img.format
                if img_format not in ["PNG", "JPEG", "WEBP"]:
                    raise ValueError(f"Image format '{img_format}' is not supported")
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"Image decoding failed for '{filename}': {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Corrupted or malformed image file"
            )

        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }
        resolved_mime = mime_map.get(ext, content_type or "image/jpeg")

        return width, height, resolved_mime

    @classmethod
    def save_image_file(cls, content: bytes, original_filename: str) -> Tuple[Path, str]:
        """
        Persists image bytes to configured settings.UPLOAD_DIR using a UUID-based safe filename.
        Returns: (dest_path, safe_filename)
        """
        ext = Path(original_filename or "").suffix.lower()
        if ext not in settings.ALLOWED_IMAGE_EXTENSIONS:
            ext = ".png"
        safe_filename = f"{uuid.uuid4().hex}{ext}"
        dest_path = (settings.UPLOAD_DIR / safe_filename).resolve()

        settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with open(dest_path, "wb") as f:
                f.write(content)
        except Exception as e:
            logger.error(f"Failed to write file to storage: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to persist uploaded image to storage"
            )

        return dest_path, safe_filename

    @classmethod
    def delete_file(cls, file_path: Union[Path, str]) -> None:
        """Cleans up a stored file safely upon transaction failure."""
        try:
            p = Path(file_path).resolve()
            if p.exists() and p.is_file():
                p.unlink()
                logger.info(f"Cleaned up file {p}")
        except Exception as e:
            logger.warning(f"Failed to delete file {file_path}: {e}")

    @classmethod
    def resolve_safe_file_path(cls, filename: str) -> Optional[Path]:
        """
        Resolves a filename within UPLOAD_DIR, strictly preventing directory traversal.
        """
        safe_name = Path(filename).name
        resolved = (settings.UPLOAD_DIR / safe_name).resolve()
        upload_dir_resolved = settings.UPLOAD_DIR.resolve()
        try:
            resolved.relative_to(upload_dir_resolved)
        except ValueError:
            return None
        if not resolved.exists() or not resolved.is_file():
            return None
        return resolved
