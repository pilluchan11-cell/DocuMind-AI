"""
Document Processing Module for DocuMind AI.
Extracts clean, structured text from PDF, DOCX, and TXT documents.
Preserves original page numbers for PDFs, handles tables and sections in DOCX,
gracefully processes multi-encoding TXT files, and prevents upload errors.
"""

import io
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Any, Union, BinaryIO, Optional, Tuple


class DocumentProcessingError(Exception):
    """Custom exception raised when document extraction or validation fails."""
    pass


class DocumentProcessor:
    """
    Robust multi-format document parser supporting PDF, DOCX, and TXT files.
    Ensures safe stream/byte-based processing for Streamlit UploadedFile objects.
    """

    SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt"]

    @classmethod
    def process_file(
        cls,
        file_source: Union[str, Path, BinaryIO, bytes, Any],
        filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validates, inspects, and extracts structured text from an uploaded file or path.

        Args:
            file_source: Streamlit UploadedFile, raw bytes, BytesIO, or file path.
            filename: Optional filename override.

        Returns:
            Dictionary containing metadata, pages/blocks, word/character counts, and preview.
        """
        if file_source is None:
            raise DocumentProcessingError("No file input was provided.")

        # 1. Safely extract raw bytes from the file source
        bytes_data = cls._read_bytes(file_source)
        if not bytes_data or len(bytes_data.strip()) == 0:
            name_label = filename or getattr(file_source, "name", "document")
            raise DocumentProcessingError(
                f"Document '{name_label}' is completely empty (0 bytes). Please upload a valid document."
            )

        # 2. Resolve filename and extension
        if filename is None:
            if hasattr(file_source, "name") and file_source.name:
                filename = str(file_source.name)
            elif isinstance(file_source, (str, Path)):
                filename = Path(file_source).name
            else:
                filename = "uploaded_document"

        suffix = Path(filename).suffix.lower()

        # If extension missing, detect from magic bytes
        if not suffix:
            if bytes_data.startswith(b"%PDF"):
                suffix = ".pdf"
                filename = f"{filename}.pdf"
            elif bytes_data.startswith(b"PK\x03\x04"):
                suffix = ".docx"
                filename = f"{filename}.docx"
            else:
                suffix = ".txt"
                filename = f"{filename}.txt"

        if suffix not in cls.SUPPORTED_EXTENSIONS:
            raise DocumentProcessingError(
                f"Unsupported file format '{suffix}'. Supported formats: {', '.join(cls.SUPPORTED_EXTENSIONS)}"
            )

        file_hash = hashlib.sha256(bytes_data).hexdigest()
        file_size = len(bytes_data)

        # 3. Format-specific extraction
        if suffix == ".pdf":
            blocks, total_pages = cls._extract_pdf(bytes_data, filename)
        elif suffix == ".docx":
            blocks, total_pages = cls._extract_docx(bytes_data, filename)
        elif suffix == ".txt":
            blocks, total_pages = cls._extract_txt(bytes_data, filename)
        else:
            raise DocumentProcessingError(f"Unsupported format: {suffix}")

        # 4. Content validation & metrics
        total_chars = sum(b.get("char_count", len(b["text"])) for b in blocks)
        all_text = " ".join(b["text"] for b in blocks).strip()
        words = re.findall(r'\b\w+\b', all_text)
        total_words = len(words)

        if total_chars == 0 or total_words == 0:
            raise DocumentProcessingError(
                f"Document '{filename}' contains no readable text. "
                "If this is a scanned PDF or image-only file, please note that Optical Character Recognition (OCR) is not supported."
            )

        doc_id = f"doc_{file_hash[:8]}"
        preview = all_text[:500] + ("..." if len(all_text) > 500 else "")

        return {
            "document_name": filename,
            "doc_id": doc_id,
            "file_type": suffix.upper().lstrip("."),
            "file_size_bytes": file_size,
            "file_hash": file_hash,
            "pages": blocks,
            "total_pages": total_pages,  # Integer for PDF, None for DOCX and TXT
            "total_characters": total_chars,
            "total_chars": total_chars,
            "total_words": total_words,
            "preview_text": preview,
        }

    @staticmethod
    def _read_bytes(file_source: Any) -> bytes:
        """
        Safely extracts raw bytes from any file source:
        - Streamlit UploadedFile
        - raw bytes / bytearray / memoryview
        - io.BytesIO / io.BufferedReader
        - str or pathlib.Path
        """
        if file_source is None:
            raise DocumentProcessingError("No file provided. File source is None.")

        # 1. Direct byte buffers
        if isinstance(file_source, bytes):
            return file_source
        if isinstance(file_source, (bytearray, memoryview)):
            return bytes(file_source)

        # 2. Objects with getvalue() (Streamlit UploadedFile, io.BytesIO)
        if hasattr(file_source, "getvalue") and callable(getattr(file_source, "getvalue")):
            try:
                val = file_source.getvalue()
                if hasattr(file_source, "seek") and callable(getattr(file_source, "seek")):
                    try:
                        file_source.seek(0)
                    except Exception:
                        pass
                if isinstance(val, bytes):
                    return val
                elif isinstance(val, str):
                    return val.encode("utf-8", errors="replace")
            except Exception:
                pass

        # 3. Pathlib Path or string filepath on disk
        if isinstance(file_source, Path):
            if not file_source.exists():
                raise DocumentProcessingError(f"File not found on disk: {file_source}")
            return file_source.read_bytes()

        if isinstance(file_source, str):
            try:
                p = Path(file_source)
                if p.is_file():
                    return p.read_bytes()
            except Exception:
                pass
            return file_source.encode("utf-8", errors="replace")

        # 4. Stream or file-like object with read()
        if hasattr(file_source, "read") and callable(getattr(file_source, "read")):
            if hasattr(file_source, "seek") and callable(getattr(file_source, "seek")):
                try:
                    file_source.seek(0)
                except Exception:
                    pass

            data = file_source.read()

            if hasattr(file_source, "seek") and callable(getattr(file_source, "seek")):
                try:
                    file_source.seek(0)
                except Exception:
                    pass

            if isinstance(data, bytes):
                return data
            elif isinstance(data, str):
                return data.encode("utf-8", errors="replace")

        # 5. Buffer wrapper fallback
        if hasattr(file_source, "buffer"):
            return DocumentProcessor._read_bytes(file_source.buffer)

        raise DocumentProcessingError(
            f"Invalid file input object provided: {type(file_source).__name__}. "
            "Expected a file path, raw bytes, or Streamlit UploadedFile."
        )

    # --------------------------------------------------------------------------
    # FORMAT EXTRACTORS
    # --------------------------------------------------------------------------
    @classmethod
    def _extract_pdf(cls, bytes_data: bytes, filename: str) -> Tuple[List[Dict[str, Any]], int]:
        """
        Extracts text page-by-page from PDF using PyMuPDF (fitz) with safe stream API.
        Preserves original page numbers (1-indexed).
        """
        pages = []

        # Attempt 1: PyMuPDF / fitz
        try:
            import pymupdf
            doc = pymupdf.open(stream=bytes_data, filetype="pdf")
            try:
                if getattr(doc, "is_encrypted", False):
                    raise DocumentProcessingError(
                        f"PDF '{filename}' is password protected. Please upload an unencrypted document."
                    )
                total_p = len(doc)
                if total_p == 0:
                    raise DocumentProcessingError(f"PDF '{filename}' contains 0 pages.")

                for page_idx in range(total_p):
                    page = doc[page_idx]
                    text = page.get_text("text").strip()
                    pages.append({
                        "page_number": page_idx + 1,
                        "text": text,
                        "char_count": len(text),
                        "document_name": filename
                    })
                return pages, total_p
            finally:
                doc.close()

        except DocumentProcessingError:
            raise
        except ImportError:
            pass
        except Exception:
            # Fall through to pypdf fallback
            pass

        # Attempt 2: pypdf fallback
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(bytes_data))
            if getattr(reader, "is_encrypted", False):
                raise DocumentProcessingError(
                    f"PDF '{filename}' is password protected. Please upload an unencrypted document."
                )
            total_p = len(reader.pages)
            if total_p == 0:
                raise DocumentProcessingError(f"PDF '{filename}' contains 0 pages.")

            for idx, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                pages.append({
                    "page_number": idx,
                    "text": text,
                    "char_count": len(text),
                    "document_name": filename
                })
            return pages, total_p

        except DocumentProcessingError:
            raise
        except Exception as e:
            raise DocumentProcessingError(
                f"Failed to parse PDF '{filename}'. File may be corrupted or unreadable: {str(e)}"
            )

    @classmethod
    def _extract_docx(cls, bytes_data: bytes, filename: str) -> Tuple[List[Dict[str, Any]], Optional[int]]:
        """
        Extracts paragraphs and tables from Microsoft Word (.docx).
        Does NOT invent page numbers because DOCX extraction does not preserve pagination.
        """
        try:
            import docx
            doc = docx.Document(io.BytesIO(bytes_data))
        except Exception as e:
            raise DocumentProcessingError(
                f"Failed to read DOCX file '{filename}'. File may be corrupted. Error: {str(e)}"
            )

        content_blocks = []

        # 1. Extract paragraphs
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            content_blocks.append(text)

        # 2. Extract tables and convert to readable Markdown
        for table in doc.tables:
            table_rows = []
            for row in table.rows:
                cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                if any(cells):
                    table_rows.append("| " + " | ".join(cells) + " |")

            if table_rows:
                if len(table_rows) > 1:
                    header_sep = "| " + " | ".join(["---"] * len(table.columns)) + " |"
                    table_rows.insert(1, header_sep)
                content_blocks.append("\n" + "\n".join(table_rows) + "\n")

        full_doc_text = "\n\n".join(content_blocks).strip()
        if not full_doc_text:
            raise DocumentProcessingError(f"Word document '{filename}' contains no readable text.")

        # Return single content block with page_number = None (pagination not reliably preserved)
        return [{
            "page_number": None,
            "text": full_doc_text,
            "char_count": len(full_doc_text),
            "document_name": filename
        }], None

    @classmethod
    def _extract_txt(cls, bytes_data: bytes, filename: str) -> Tuple[List[Dict[str, Any]], Optional[int]]:
        """
        Extracts text from plain text files, handling UTF-8, UTF-8 BOM, and other encodings.
        Sets page_number = None as plain text does not have physical pages.
        """
        text = ""
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        for enc in encodings:
            try:
                text = bytes_data.decode(enc)
                break
            except UnicodeDecodeError:
                continue

        if not text:
            raise DocumentProcessingError(f"Unable to decode text file '{filename}' with standard encodings.")

        text = text.strip()
        if not text:
            raise DocumentProcessingError(f"Text file '{filename}' is empty.")

        return [{
            "page_number": None,
            "text": text,
            "char_count": len(text),
            "document_name": filename
        }], None
