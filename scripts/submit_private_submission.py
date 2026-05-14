#!/usr/bin/env python3
"""Submit agent.py to the Subnet 66 private submission API."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any


DEFAULT_API_URL = "https://ninja66.ai/api/submissions"
MAX_AGENT_BYTES = 5_000_000
PRIVATE_SUBMISSION_RE = re.compile(r"^private-submission:[A-Za-z0-9_.-]{1,128}:[0-9a-f]{64}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit a private Subnet 66 ninja agent.py.")
    parser.add_argument("--agent", type=Path, default=Path("agent.py"), help="Path to submitted agent.py.")
    parser.add_argument("--api-url", default=os.getenv("NINJA_SUBMISSION_API", DEFAULT_API_URL))
    parser.add_argument("--submission-id", help="Optional stable submission id. Defaults to hotkey/hash derived id.")
    parser.add_argument("--hotkey", help="Expected miner hotkey SS58 address. Defaults to loaded wallet hotkey.")
    parser.add_argument("--wallet-name", default=os.getenv("BT_WALLET_NAME", "default"))
    parser.add_argument("--wallet-hotkey", default=os.getenv("BT_WALLET_HOTKEY", "default"))
    parser.add_argument("--wallet-path", default=os.getenv("BT_WALLET_PATH"))
    parser.add_argument("--dry-run", action="store_true", help="Build and print the request without sending it.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        agent_path = args.agent.expanduser().resolve()
        agent_py = agent_path.read_bytes()
        if len(agent_py) > MAX_AGENT_BYTES:
            raise ValueError(f"agent.py is {len(agent_py)} bytes; maximum is {MAX_AGENT_BYTES} bytes")
        agent_sha256 = hashlib.sha256(agent_py).hexdigest()
        wallet = load_wallet(args)
        hotkey = wallet.hotkey.ss58_address
        if args.hotkey and args.hotkey != hotkey:
            raise ValueError(f"loaded wallet hotkey {hotkey} does not match --hotkey {args.hotkey}")
        submission_id = args.submission_id or derive_submission_id(hotkey=hotkey, agent_sha256=agent_sha256)
        signature_payload = private_submission_signature_payload(
            hotkey=hotkey,
            submission_id=submission_id,
            agent_sha256=agent_sha256,
        )
        signature = sign_payload(wallet, signature_payload)
        print_request_summary(
            agent_path=agent_path,
            hotkey=hotkey,
            submission_id=submission_id,
            agent_sha256=agent_sha256,
            signature_payload=signature_payload,
        )
        if args.dry_run:
            print("dry_run: true")
            return 0

        response = post_submission(
            api_url=args.api_url,
            hotkey=hotkey,
            submission_id=submission_id,
            signature=signature,
            agent_filename=agent_path.name,
            agent_py=agent_py,
        )
        print(json.dumps(response, indent=2, sort_keys=True))
        if not bool(response.get("accepted")):
            return 1
        commitment = str(response.get("commitment") or "")
        validate_private_commitment(commitment)
        return 0
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def load_wallet(args: argparse.Namespace) -> Any:
    try:
        bt = importlib.import_module("bittensor")
    except ImportError as exc:
        raise RuntimeError("bittensor is not installed in this Python environment") from exc
    wallet_kwargs = {"name": args.wallet_name, "hotkey": args.wallet_hotkey}
    if args.wallet_path:
        wallet_kwargs["path"] = args.wallet_path
    return bt.Wallet(**wallet_kwargs)


def derive_submission_id(*, hotkey: str, agent_sha256: str) -> str:
    safe_hotkey = re.sub(r"[^A-Za-z0-9_.-]", "-", hotkey)[:16] or "hotkey"
    return f"{safe_hotkey}-{agent_sha256[:16]}"


def private_submission_signature_payload(*, hotkey: str, submission_id: str, agent_sha256: str) -> bytes:
    return f"tau-private-submission-v1:{hotkey}:{submission_id}:{agent_sha256.lower()}".encode("utf-8")


def sign_payload(wallet: Any, payload: bytes) -> str:
    signature = wallet.hotkey.sign(payload)
    if isinstance(signature, str):
        return signature.removeprefix("0x")
    if isinstance(signature, bytes):
        return signature.hex()
    if hasattr(signature, "hex"):
        return signature.hex()
    raise TypeError(f"unsupported signature type: {type(signature).__name__}")


def print_request_summary(
    *,
    agent_path: Path,
    hotkey: str,
    submission_id: str,
    agent_sha256: str,
    signature_payload: bytes,
) -> None:
    print(f"agent: {agent_path}")
    print(f"hotkey: {hotkey}")
    print(f"submission_id: {submission_id}")
    print(f"agent_sha256: {agent_sha256}")
    print(f"signature_payload: {signature_payload.decode('utf-8')}")


def post_submission(
    *,
    api_url: str,
    hotkey: str,
    submission_id: str,
    signature: str,
    agent_filename: str,
    agent_py: bytes,
) -> dict[str, Any]:
    fields = {
        "hotkey": hotkey,
        "submission_id": submission_id,
        "signature": signature,
    }
    files = {"agent": (agent_filename, agent_py, "text/x-python")}
    body, content_type = encode_multipart_form(fields=fields, files=files)
    request = urllib.request.Request(
        api_url,
        data=body,
        headers={"Content-Type": content_type, "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return decode_json_response(response.read())
    except urllib.error.HTTPError as exc:
        payload = decode_json_response(exc.read())
        if payload:
            return payload
        raise RuntimeError(f"submission API returned HTTP {exc.code}") from exc


def encode_multipart_form(
    *,
    fields: dict[str, str],
    files: dict[str, tuple[str, bytes, str]],
) -> tuple[bytes, str]:
    boundary = f"----ninja66-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode(),
                b"\r\n",
            ]
        )
    for name, (filename, content, content_type) in files.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                (
                    f'Content-Disposition: form-data; name="{name}"; '
                    f'filename="{filename}"\r\n'
                ).encode(),
                f"Content-Type: {content_type}\r\n\r\n".encode(),
                content,
                b"\r\n",
            ]
        )
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def decode_json_response(body: bytes) -> dict[str, Any]:
    if not body:
        return {}
    payload = json.loads(body.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("submission API returned non-object JSON")
    return payload


def validate_private_commitment(commitment: str) -> None:
    if not PRIVATE_SUBMISSION_RE.fullmatch(commitment):
        raise ValueError("accepted API response did not include a valid private-submission commitment")


if __name__ == "__main__":
    sys.exit(main())
