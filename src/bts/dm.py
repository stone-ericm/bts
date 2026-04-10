"""Bluesky DM notifications for BTS orchestrator failures."""

import json
from urllib.error import HTTPError
from urllib.request import Request

from bts.util import retry_urlopen

BSKY_HOST = "https://bsky.social/xrpc"
CHAT_HOST = "https://api.bsky.chat/xrpc"
BOT_HANDLE = "beatthestreakbot.bsky.social"


def get_bluesky_dm_password() -> str:
    """Alias for posting.get_bluesky_password -- same password, different context.

    Kept as a separate function name for call-site clarity. DM and posting
    use the same app password since consolidation.
    """
    from bts.posting import get_bluesky_password
    return get_bluesky_password()


# Backward compat alias
get_dm_password = get_bluesky_dm_password


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

    # Step 3: Get or create conversation (via chat service directly)
    req = Request(
        f"{CHAT_HOST}/chat.bsky.convo.getConvoForMembers?members={target_did}",
        headers={"Authorization": f"Bearer {jwt}"},
    )
    convo_id = json.loads(retry_urlopen(req, timeout=15).read())["convo"]["id"]

    # Step 4: Send message
    msg_data = json.dumps({
        "convoId": convo_id,
        "message": {"text": text},
    }).encode()
    req = Request(
        f"{CHAT_HOST}/chat.bsky.convo.sendMessage",
        data=msg_data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {jwt}",
        },
    )
    result = json.loads(retry_urlopen(req, timeout=15).read())
    return result["id"]
