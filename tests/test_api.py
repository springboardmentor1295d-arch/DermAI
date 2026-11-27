import os
import io

import pytest

pytest.importorskip("fastapi"); pytest.importorskip("jose"); pytest.importorskip("passlib")
os.environ["DATABASE_URL"] = "sqlite:///./test_dermai.db"
os.environ["STORAGE_DIR"] = "./test_storage"
os.environ["MODEL_DIR"] = "./nonexistent"

from fastapi.testclient import TestClient  # noqa: E402

from backend.app.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def hdr(c, email, pw):
    t = c.post("/auth/login", data={"username": email, "password": pw}).json()["access_token"]
    return {"Authorization": f"Bearer {t}"}


def test_register_login_rbac(client):
    r = client.post("/auth/register", json={"email": "p@x.com", "password": "pass12345"})
    assert r.status_code in (201, 409)
    assert client.post("/auth/register", json={"email": "a@x.com", "password": "pass12345", "role": "admin"}).status_code == 403
    h = hdr(client, "p@x.com", "pass12345")
    assert client.get("/stats/admin", headers=h).status_code == 403      # patient cannot see admin stats
    assert client.get("/cases").status_code == 401                        # unauthenticated


def test_case_and_upload_flow(client):
    import cv2, numpy as np
    h = hdr(client, "p@x.com", "pass12345")
    c = client.post("/cases", json={"patient_code": "DEMO-1", "age": 50}, headers=h).json()
    rng = np.random.default_rng(1)
    img = cv2.GaussianBlur(rng.integers(60, 200, (500, 500, 3), dtype=np.uint8), (0, 0), 1)
    cv2.circle(img, (250, 250), 80, (60, 30, 30), -1)
    ok, buf = cv2.imencode(".jpg", img)
    r = client.post(f"/cases/{c['id']}/images", files={"file": ("a.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")}, headers=h)
    assert r.status_code == 201, r.text
    bad = client.post(f"/cases/{c['id']}/images", files={"file": ("a.txt", io.BytesIO(b"hi"), "text/plain")}, headers=h)
    assert bad.status_code == 415
    # no model deployed in tests -> analysis must fail clearly, not crash
    assert client.post(f"/images/{r.json()['id']}/analyze", headers=h).status_code == 503
