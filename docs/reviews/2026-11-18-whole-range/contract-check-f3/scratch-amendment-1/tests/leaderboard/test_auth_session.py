from __future__ import annotations

import bts.leaderboard.auth as auth


def test_fetch_login_session_returns_xsid_and_user(monkeypatch):
    class Response:
        status_code = 200

        def json(self):
            return {
                "success": {
                    "xSid": "abc_123",
                    "user": {"id": 50311, "username": "stonehengee"},
                }
            }

    monkeypatch.setattr(auth.httpx, "post", lambda *args, **kwargs: Response())

    session = auth.fetch_login_session(uid="okta-uid", cookies={"oktaid": "okta-uid"})

    assert session.xsid == "abc_123"
    assert session.user_id == 50311
    assert session.username == "stonehengee"


def test_fetch_xsid_still_works(monkeypatch):
    class Response:
        status_code = 200

        def json(self):
            return {
                "success": {
                    "xSid": "z_9",
                    "user": {"id": 1, "username": "x"},
                }
            }

    monkeypatch.setattr(auth.httpx, "post", lambda *args, **kwargs: Response())

    assert auth.fetch_xsid("uid", {"oktaid": "uid"}) == "z_9"
