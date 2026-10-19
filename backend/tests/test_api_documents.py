from fastapi.testclient import TestClient

from tests.conftest import make_pdf_bytes

TXT_TEXT = (
    "The Eiffel Tower is located in Paris. "
    "It was completed in 1889. "
    "The tower is 330 metres tall."
)


def upload(client: TestClient, name: str, data: bytes, content_type: str):
    return client.post("/api/documents/upload", files={"files": (name, data, content_type)})


def test_health_endpoint(api_client: TestClient):
    response = api_client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"]
    assert "vector_store" in body["components"]


def test_upload_list_detail_delete_flow(api_client: TestClient):
    response = upload(api_client, "tower.txt", TXT_TEXT.encode("utf-8"), "text/plain")
    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["status"] == "completed"
    assert result["chunk_count"] > 0
    assert result["chunks_stored"] == result["chunk_count"]
    document_id = result["document_id"]

    listing = api_client.get("/api/documents").json()
    assert listing["total"] == 1
    assert listing["documents"][0]["document_id"] == document_id
    assert listing["documents"][0]["status"] == "completed"

    detail = api_client.get(f"/api/documents/{document_id}")
    assert detail.status_code == 200
    assert detail.json()["filename"] == "tower.txt"

    deleted = api_client.delete(f"/api/documents/{document_id}")
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True
    assert api_client.get(f"/api/documents/{document_id}").status_code == 404


def test_upload_rejects_unsupported_extension(api_client: TestClient):
    response = upload(api_client, "notes.docx", b"hello", "application/octet-stream")
    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["status"] == "rejected"
    assert "Unsupported file type" in result["error"]
    assert result["document_id"] is None


def test_upload_reports_corrupt_pdf_as_failed(api_client: TestClient):
    response = upload(api_client, "broken.pdf", b"%PDF-1.4 nope", "application/pdf")
    result = response.json()["results"][0]
    assert result["status"] == "failed"
    assert result["error"]
    assert result["document_id"]


def test_upload_reports_image_only_pdf(api_client: TestClient):
    response = upload(api_client, "scan.pdf", make_pdf_bytes([None]), "application/pdf")
    result = response.json()["results"][0]
    assert result["status"] == "failed"
    assert "OCR" in result["error"]


def test_upload_rejects_oversized_file(api_client: TestClient):
    big = ("a" * 200) * 8000  # ~1.6 MB, above the 1 MB test limit
    response = upload(api_client, "big.txt", big.encode("utf-8"), "text/plain")
    result = response.json()["results"][0]
    assert result["status"] == "rejected"
    assert "maximum upload size" in result["error"]


def test_upload_rejects_too_many_files(api_client: TestClient):
    files = [
        ("files", (f"f{index}.txt", b"hello", "text/plain")) for index in range(4)
    ]
    response = api_client.post("/api/documents/upload", files=files)
    assert response.status_code == 400


def test_upload_requires_a_file(api_client: TestClient):
    assert api_client.post("/api/documents/upload").status_code == 422


def test_duplicate_upload_reuses_document(api_client: TestClient):
    first = upload(api_client, "a.txt", TXT_TEXT.encode("utf-8"), "text/plain").json()["results"][0]
    second = upload(api_client, "b.txt", TXT_TEXT.encode("utf-8"), "text/plain").json()["results"][0]
    assert second["duplicate"] is True
    assert second["document_id"] == first["document_id"]
    assert api_client.get("/api/documents").json()["total"] == 1


def test_get_unknown_document_returns_404(api_client: TestClient):
    assert api_client.get("/api/documents/does-not-exist").status_code == 404


def test_delete_unknown_document_returns_404(api_client: TestClient):
    assert api_client.delete("/api/documents/does-not-exist").status_code == 404


def test_multiple_files_are_processed_independently(api_client: TestClient):
    files = [
        ("files", ("ok.txt", TXT_TEXT.encode("utf-8"), "text/plain")),
        ("files", ("bad.pdf", b"%PDF-1.4 nope", "application/pdf")),
        ("files", ("skip.docx", b"x", "application/octet-stream")),
    ]
    response = api_client.post("/api/documents/upload", files=files)
    results = response.json()["results"]
    assert [item["status"] for item in results] == ["completed", "failed", "rejected"]
