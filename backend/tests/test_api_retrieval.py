from fastapi.testclient import TestClient


def upload(client: TestClient, text: str, name: str = "doc.txt"):
    return client.post(
        "/api/documents/upload",
        files={"files": (name, text.encode("utf-8"), "text/plain")},
    ).json()["results"][0]


def test_search_returns_matching_chunk(api_client: TestClient):
    text = "Quantum computing uses qubits. Superposition is a key property of qubits."
    document = upload(api_client, text)
    response = api_client.post(
        "/api/retrieval/search",
        json={"query": "Quantum computing uses qubits.", "top_k": 3},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["metric"] == "cosine_distance"
    assert body["total"] >= 1
    top = body["results"][0]
    assert top["document_id"] == document["document_id"]
    assert top["chunk_id"]
    assert top["text"]
    assert isinstance(top["distance"], float)
    assert isinstance(top["similarity"], float)
    assert top["distance"] <= body["results"][-1]["distance"] + 1e-6


def test_search_on_empty_store_returns_no_results(api_client: TestClient):
    response = api_client.post("/api/retrieval/search", json={"query": "anything"})
    assert response.status_code == 200
    assert response.json()["results"] == []


def test_search_respects_top_k(api_client: TestClient):
    upload(api_client, " ".join(f"token{i}" for i in range(200)), "long.txt")
    response = api_client.post(
        "/api/retrieval/search", json={"query": "token5", "top_k": 2}
    )
    body = response.json()
    assert body["top_k"] == 2
    assert len(body["results"]) <= 2


def test_search_validates_blank_query(api_client: TestClient):
    assert api_client.post("/api/retrieval/search", json={"query": "   "}).status_code == 422
    assert api_client.post("/api/retrieval/search", json={"query": ""}).status_code == 422


def test_search_validates_top_k_bounds(api_client: TestClient):
    response = api_client.post("/api/retrieval/search", json={"query": "x", "top_k": 999})
    assert response.status_code == 422


def test_search_can_filter_by_document(api_client: TestClient):
    first = upload(api_client, "Apples are red fruits.", "apples.txt")
    upload(api_client, "Cars have four wheels.", "cars.txt")
    response = api_client.post(
        "/api/retrieval/search",
        json={"query": "Apples are red fruits.", "document_ids": [first["document_id"]]},
    )
    body = response.json()
    assert body["results"]
    assert all(item["document_id"] == first["document_id"] for item in body["results"])
