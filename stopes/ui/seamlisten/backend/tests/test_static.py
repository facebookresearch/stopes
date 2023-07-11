import pytest
from fastapi.testclient import TestClient
from app.static import router


@pytest.fixture
def client():
    return TestClient(router)


def test_read_root(client):
    response = client.get("/")
    assert response.status_code == 200
    # Add more assertions to validate the response as needed


def test_get_logo(client):
    response = client.get("/logo.png")
    assert response.status_code == 200
    # Add more assertions to validate the response as needed
