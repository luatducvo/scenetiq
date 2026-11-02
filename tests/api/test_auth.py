from fastapi.testclient import TestClient

from .conftest import AUTH


def test_request_without_token_is_rejected(client: TestClient):
    assert client.get("/scenes").status_code == 401


def test_request_with_wrong_token_is_rejected(client: TestClient):
    assert client.get("/scenes", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_request_with_correct_token_is_accepted(client: TestClient):
    assert client.get("/scenes", headers=AUTH).status_code == 200
