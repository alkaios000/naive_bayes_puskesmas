import pytest

def test_viewer_forbidden_on_admin_actions(client, viewer_token):
    """
    Kriteria Penerimaan 11:
    Viewer tidak bisa mengunggah, melatih, atau menghapus (403 Forbidden).
    """
    headers = {"Authorization": f"Bearer {viewer_token}"}

    # 1. Upload
    res_upload = client.post(
        "/api/datasets/upload",
        files={"file": ("test.csv", b"tahun,bulan\n2024,1", "text/csv")},
        headers=headers
    )
    assert res_upload.status_code == 403

    # 2. Train
    res_train = client.post(
        "/api/models/train",
        json={"dataset_id": 1, "threshold_pct": 0.10, "min_abs_change": 3, "alpha": 1.0, "test_ratio": 0.20},
        headers=headers
    )
    assert res_train.status_code == 403

    # 3. Delete
    res_delete = client.delete("/api/datasets/1", headers=headers)
    assert res_delete.status_code == 403

def test_viewer_allowed_on_read_actions(client, viewer_token):
    """
    Viewer diizinkan melihat daftar dataset, metrik, dan prediksi (200 OK).
    """
    headers = {"Authorization": f"Bearer {viewer_token}"}
    res = client.get("/api/datasets", headers=headers)
    assert res.status_code == 200
