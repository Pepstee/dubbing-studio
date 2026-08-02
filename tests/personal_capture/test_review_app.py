from unittest.mock import patch

from dubbing.apps.personal_capture.review import create_capture_app
from tests.personal_capture.test_config import deployment_config


class _Service:
    def records(self):
        return ()


def test_review_templates_and_static_assets_are_served(tmp_path):
    config = deployment_config(tmp_path)
    workspace = tmp_path / "life-logging" / "audio-processing"
    (workspace / "recordings" / "inbox").mkdir(parents=True)
    token = tmp_path / "capture.token"
    token.write_text("a" * 64, encoding="utf-8")
    config_path = tmp_path / "config.json"
    import json

    config_path.write_text(json.dumps(config), encoding="utf-8")
    with patch(
        "dubbing.apps.personal_capture.review.build_control_service",
        return_value=_Service(),
    ):
        app = create_capture_app(config_path)
    client = app.test_client()

    assert client.get("/").status_code == 401
    assert client.get("/?token=" + "a" * 64).status_code == 401
    login = client.get("/login")
    response = client.post("/login", data={"token": "a" * 64}, follow_redirects=True)
    css = client.get("/static/styles.css")

    assert login.status_code == 200
    assert b"never placed in the URL" in login.data
    assert response.status_code == 200
    assert b"Review queue" in response.data
    assert b"styles.css" in response.data
    assert css.status_code == 200
    assert response.headers["Content-Security-Policy"].startswith(
        "default-src 'none'; style-src 'self'"
    )
