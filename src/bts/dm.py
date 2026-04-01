"""Bluesky DM notifications for BTS orchestrator failures."""

import json
import os
import subprocess
from urllib.error import HTTPError
from urllib.request import Request

from bts.util import retry_urlopen

BSKY_HOST = "https://bsky.social/xrpc"
CHAT_PROXY = "did:web:api.bsky.chat#bsky_chat"
BOT_HANDLE = "beatthestreakbot.bsky.social"


def get_dm_password() -> str:
    """Get Bluesky DM app password from keychain or environment.

    Uses the DM-scoped password (separate from posting password).
    """
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-a", "claude-cli",
             "-s", "bluesky-bts-app-password-dm", "-w"],
            capture_output=True, text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except FileNotFoundError:
        pass

    password = os.environ.get("BTS_BLUESKY_DM_PASSWORD")
    if password:
        return password

    raise RuntimeError(
        "DM password not found. Set BTS_BLUESKY_DM_PASSWORD or add "
        "bluesky-bts-app-password-dm to keychain."
    )


def send_dm(recipient_handle: str, text: str) -> str:
    """Send a Bluesky DM to recipient_handle. Returns message ID.

    Steps: authenticate -> resolve handle -> get/create convo -> send.
    Uses the AT Protocol chat API with the atproto-proxy header for
    routing to the chat service (did:web:api.bsky.chat#bsky_chat).
    """
    password = get_dm_password()

    # Step 1: Authenticate
    auth_data = json.dumps({
        "identifier": BOT_HANDLE,
        "password": password,
    }).encode()
    req = Request(
        f"{BSKY_HOST}/com.atproto.server.createSession",
        data=auth_data,
        headers={"Content-Type": "application/json"},
    )
    try:
        session = json.loads(retry_urlopen(req, timeout=15).read())
    except HTTPError as e:
        if e.code == 401:
            raise RuntimeError("DM auth failed — check bluesky-bts-app-password-dm") from e
        raise RuntimeError(f"DM auth error (HTTP {e.code})") from e

    jwt = session["accessJwt"]

    # Step 2: Resolve recipient handle to DID
    req = Request(
        f"{BSKY_HOST}/com.atproto.identity.resolveHandle?handle={recipient_handle}",
        headers={"Authorization": f"Bearer {jwt}"},
    )
    target_did = json.loads(retry_urlopen(req, timeout=15).read())["did"]

    # Step 3: Get or create conversation
    req = Request(
        f"{BSKY_HOST}/chat.bsky.convo.getConvoForMembers?members={target_did}",
        headers={
            "Authorization": f"Bearer {jwt}",
            "atproto-proxy": CHAT_PROXY,
        },
    )
    convo_id = json.loads(retry_urlopen(req, timeout=15).read())["convo"]["id"]

    # Step 4: Send message
    msg_data = json.dumps({
        "convoId": convo_id,
        "message": {"text": text},
    }).encode()
    req = Request(
        f"{BSKY_HOST}/chat.bsky.convo.sendMessage",
        data=msg_data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {jwt}",
            "atproto-proxy": CHAT_PROXY,
        },
    )
    result = json.loads(retry_urlopen(req, timeout=15).read())
    return result["id"]
