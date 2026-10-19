import codecs

import pytest

from app.core.errors import (
    DocumentProcessingError,
    DocumentValidationError,
    ImageOnlyPdfError,
)
from app.services.document_processor import (
    DocumentProcessor,
    decode_text_bytes,
    sanitize_filename,
    validate_upload_metadata,
)
from tests.conftest import make_pdf_bytes


def test_sanitize_filename_blocks_path_traversal():
    assert sanitize_filename("../../etc/passwd.txt") == "passwd.txt"
    assert sanitize_filename("C:\\temp\\report.pdf") == "report.pdf"
    assert sanitize_filename("normal.txt") == "normal.txt"


def test_sanitize_filename_rejects_bad_names():
    with pytest.raises(DocumentValidationError):
        sanitize_filename("")
    with pytest.raises(DocumentValidationError):
        sanitize_filename("..")


def test_validate_rejects_unsupported_extension():
    with pytest.raises(DocumentValidationError):
        validate_upload_metadata("data.docx", "application/octet-stream")


def test_validate_rejects_mismatched_content_type():
    with pytest.raises(DocumentValidationError):
        validate_upload_metadata("scan.pdf", "text/plain")
    with pytest.raises(DocumentValidationError):
        validate_upload_metadata("notes.txt", "image/png")


def test_extract_txt_utf8(processor: DocumentProcessor):
    result = processor.extract("Hello multilingual world".encode("utf-8"), "a.txt", "text/plain")
    assert result.extension == ".txt"
    assert result.page_count == 1
    assert result.pages[0].page_number is None
    assert "multilingual" in result.pages[0].text


def test_extract_txt_utf16_with_bom(processor: DocumentProcessor):
    data = "Héllo UTF16".encode("utf-16")
    result = processor.extract(data, "b.txt", "text/plain")
    assert "Héllo" in result.pages[0].text


def test_extract_txt_cp1252(processor: DocumentProcessor):
    data = "café — naïve".encode("cp1252")
    result = processor.extract(data, "c.txt", "text/plain")
    assert "café" in result.pages[0].text
    assert "naïve" in result.pages[0].text


def test_extract_txt_cp1251(processor: DocumentProcessor):
    text = "Привет мир, это русский текст для проверки кодировки cp1251 и он достаточно длинный."
    result = processor.extract(text.encode("cp1251"), "ru.txt", "text/plain")
    assert "Привет" in result.pages[0].text


def test_extract_txt_rejects_binary(processor: DocumentProcessor):
    with pytest.raises(DocumentValidationError):
        processor.extract(b"\x00\x01\x02binary", "d.txt", "text/plain")


def test_extract_rejects_empty(processor: DocumentProcessor):
    with pytest.raises(DocumentValidationError):
        processor.extract(b"", "e.txt", "text/plain")


def test_decode_text_bytes_reports_undecodable():
    with pytest.raises(DocumentValidationError):
        decode_text_bytes(b"")


def test_extract_pdf_preserves_pages(processor: DocumentProcessor):
    data = make_pdf_bytes(["First page content", "Second page content"])
    result = processor.extract(data, "doc.pdf", "application/pdf")
    assert result.page_count == 2
    assert [page.page_number for page in result.pages] == [1, 2]
    assert "First page" in result.pages[0].text


def test_extract_pdf_blank_pages_reports_ocr(processor: DocumentProcessor):
    data = make_pdf_bytes([None, None])
    with pytest.raises(ImageOnlyPdfError) as exc_info:
        processor.extract(data, "scan.pdf", "application/pdf")
    assert "OCR" in str(exc_info.value)


def test_extract_corrupt_pdf(processor: DocumentProcessor):
    with pytest.raises(DocumentProcessingError):
        processor.extract(b"%PDF-1.4 this is not really a pdf", "bad.pdf", "application/pdf")


def test_utf32_bom_is_supported():
    data = codecs.BOM_UTF32_LE + "wide text".encode("utf-32-le")
    text, encoding = decode_text_bytes(data)
    assert text == "wide text"
    assert encoding == "utf-32"
