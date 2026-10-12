import io
import pickle

import pytest
pytest.importorskip("fastapi")
pytest.importorskip("multipart")
from fastapi.testclient import TestClient
from api import main


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "MODEL_CACHE_PATH", str(tmp_path / "model.pkl"))
    monkeypatch.setattr(main, "_global_model", None)
    with TestClient(main.app) as client:
        yield client


def test_json_train_predict_and_persistence(client):
    X = [[i, "a" if i % 2 else "b"] for i in range(12)]
    response = client.post("/train", json={"X": X, "y": ["yes", "no"] * 6,
                                         "models": ["linear"], "cv_folds": 2})
    assert response.status_code == 200, response.text
    assert response.json()["weights"] == {"linear": 1.0}
    response = client.post("/predict", json={"data": X[:2]})
    assert response.status_code == 200, response.text
    assert set(response.json()["predictions"]) <= {"yes", "no"}
    with open(main.MODEL_CACHE_PATH, "rb") as stream:
        assert pickle.load(stream)._fitted
    assert client.get("/ping").json()["model_loaded"]


def test_csv_train_preserves_schema(client):
    csv = "number,category,target\n" + "".join(f"{i},{'a' if i % 2 else 'b'},{i % 2}\n" for i in range(12))
    response = client.post("/train_upload", files={"file": ("data.csv", io.BytesIO(csv.encode()), "text/csv")},
                           data={"target_col": "target"})
    assert response.status_code == 200, response.text
    assert response.json()["weights"]
    response = client.post("/predict", json={"data": [[0, "b"], [1, "a"]]})
    assert response.status_code == 200, response.text


def test_invalid_requests(client):
    assert client.post("/predict", json={"data": [[1]]}).status_code == 400
    assert client.post("/train", json={"X": [[1]], "y": [1], "task": "bad"}).status_code == 422
    response = client.post("/train_upload", files={"file": ("data.csv", b"x,y\n1,2\n")},
                           data={"target_col": "missing"})
    assert response.status_code == 400
    assert "not found" in response.json()["detail"]
