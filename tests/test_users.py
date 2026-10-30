import pytest
from app.models import User, UserRole
from app.security import get_password_hash

def test_viewer_forbidden_on_user_management(client, viewer_token):
    """
    Viewer tidak boleh mengakses endpoint manajemen pengguna (403 Forbidden).
    """
    headers = {"Authorization": f"Bearer {viewer_token}"}

    # 1. List users
    res_list = client.get("/api/users", headers=headers)
    assert res_list.status_code == 403

    # 2. Create user
    res_create = client.post(
        "/api/users",
        json={"username": "new_user", "password": "password123", "role": "viewer"},
        headers=headers
    )
    assert res_create.status_code == 403

    # 3. Update user
    res_update = client.put(
        "/api/users/1",
        json={"role": "admin"},
        headers=headers
    )
    assert res_update.status_code == 403

    # 4. Delete user
    res_delete = client.delete("/api/users/1", headers=headers)
    assert res_delete.status_code == 403

def test_admin_crud_user_management(client, admin_token, db_session):
    """
    Admin dapat melihat, membuat, mengubah, dan menghapus akun pengguna.
    """
    headers = {"Authorization": f"Bearer {admin_token}"}

    # 1. List users
    res_list = client.get("/api/users", headers=headers)
    assert res_list.status_code == 200
    users = res_list.json()
    assert len(users) >= 1

    # 2. Create user baru
    username = "petugas_sarang_test"
    # Pastikan bersih sebelum test
    db_session.query(User).filter(User.username == username).delete()
    db_session.commit()

    res_create = client.post(
        "/api/users",
        json={"username": username, "password": "password123", "role": "viewer"},
        headers=headers
    )
    assert res_create.status_code == 201
    created_user = res_create.json()
    user_id = created_user["id"]
    assert created_user["username"] == username
    assert created_user["role"] == "viewer"

    # 3. Create duplikat ditolak
    res_dup = client.post(
        "/api/users",
        json={"username": username, "password": "password123", "role": "viewer"},
        headers=headers
    )
    assert res_dup.status_code == 400

    # 4. Update role & password
    res_update = client.put(
        f"/api/users/{user_id}",
        json={"role": "admin", "password": "newpassword123"},
        headers=headers
    )
    assert res_update.status_code == 200
    updated = res_update.json()
    assert updated["role"] == "admin"

    # 5. Admin tidak boleh menghapus akun sendiri
    admin_user = db_session.query(User).filter(User.username == "test_admin").first()
    res_del_self = client.delete(f"/api/users/{admin_user.id}", headers=headers)
    assert res_del_self.status_code == 400

    # 6. Hapus user target
    res_del = client.delete(f"/api/users/{user_id}", headers=headers)
    assert res_del.status_code == 200
    assert "berhasil dihapus" in res_del.json()["message"]

    # 7. Cek user sudah terhapus
    res_check = client.get(f"/api/users", headers=headers)
    assert not any(u["id"] == user_id for u in res_check.json())
