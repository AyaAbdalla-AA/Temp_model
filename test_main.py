from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200


def test_predict():
    response = client.post(
        "/predict",
        json={"X_fahrenheit": 100}
    )

    assert response.status_code == 200
    assert "prediction" in response.json()
