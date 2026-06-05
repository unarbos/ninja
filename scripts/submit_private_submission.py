#!/usr/bin/env python3
"""Submit agent.py or a multi-file harness to the Subnet 66 private submission API."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import os
import re
import sys
import tarfile
import urllib.error
import urllib.request
import uuid
import zipfile
from pathlib import Path
from typing import Any


DEFAULT_API_URL = "https://ninja66.ai/api/submissions"
USER_AGENT = "ninja66-private-submission/2.0"
MAX_AGENT_BYTES = 5_000_000
MAX_BUNDLE_BYTES = 5_000_000
PRIVATE_SUBMISSION_RE = re.compile(r"^private-submission:[A-Za-z0-9_.-]{1,128}:[0-9a-f]{64}$")
DEFAULT_ENTRYPOINT = "agent.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit a private Subnet 66 ninja harness.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--agent", type=Path, help="Path to submitted agent.py (v1).")
    source.add_argument("--bundle", type=Path, help="Path to a harness directory containing agent.py (v2).")
    source.add_argument("--archive", type=Path, help="Path to a .tar.gz or .zip harness archive (v2).")
    parser.add_argument("--api-url", default=os.getenv("NINJA_SUBMISSION_API", DEFAULT_API_URL))
    parser.add_argument("--submission-id", help="Optional stable submission id. Defaults to hotkey/hash derived id.")
    parser.add_argument("--hotkey", help="Expected miner hotkey SS58 address. Defaults to loaded wallet hotkey.")
    parser.add_argument("--wallet-name", default=os.getenv("BT_WALLET_NAME", "default"))
    parser.add_argument("--wallet-hotkey", default=os.getenv("BT_WALLET_HOTKEY", "default"))
    parser.add_argument("--wallet-path", default=os.getenv("BT_WALLET_PATH"))
    parser.add_argument("--agent-username", help="Optional public display username for this miner agent.")
    parser.add_argument("--coldkey", help="Coldkey SS58 that owns the submitting hotkey. Defaults to the loaded wallet coldkey.")
    parser.add_argument(
        "--coldkey-signature",
        help="Coldkey signature over tau-agent-submission-username:<agent-username>. Defaults to signing with the loaded wallet coldkey.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build and print the request without sending it.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        bundle_files, archive_bytes, archive_name = load_submission_payload(args)
        wallet = load_wallet(args)
        hotkey = wallet.hotkey.ss58_address
        if args.hotkey and args.hotkey != hotkey:
            raise ValueError(f"loaded wallet hotkey {hotkey} does not match --hotkey {args.hotkey}")

        if bundle_files is not None:
            agent_sha256 = hashlib.sha256(bundle_files[DEFAULT_ENTRYPOINT]).hexdigest()
            bundle_sha256 = canonical_bundle_sha256(bundle_files)
            content_sha256 = bundle_sha256
            submission_id = args.submission_id or derive_submission_id(hotkey=hotkey, content_sha256=content_sha256)
            signature_payload = bundle_signature_payload(
                hotkey=hotkey,
                submission_id=submission_id,
                bundle_sha256=bundle_sha256,
            )
            source_label = str(args.bundle or args.archive)
        else:
            agent_path = args.agent.expanduser().resolve()
            agent_py = agent_path.read_bytes()
            if len(agent_py) > MAX_AGENT_BYTES:
                raise ValueError(f"{agent_path} is {len(agent_py)} bytes; max is {MAX_AGENT_BYTES} bytes")
            agent_sha256 = hashlib.sha256(agent_py).hexdigest()
            bundle_sha256 = None
            content_sha256 = agent_sha256
            submission_id = args.submission_id or derive_submission_id(hotkey=hotkey, content_sha256=content_sha256)
            signature_payload = private_submission_signature_payload(
                hotkey=hotkey,
                submission_id=submission_id,
                agent_sha256=agent_sha256,
            )
            source_label = str(agent_path)

        signature = sign_payload(wallet, signature_payload)
        identity = build_username_identity(args=args, wallet=wallet)
        print_request_summary(
            source_label=source_label,
            hotkey=hotkey,
            submission_id=submission_id,
            agent_sha256=agent_sha256,
            bundle_sha256=bundle_sha256,
            signature_payload=signature_payload,
            identity=identity,
        )
        if args.dry_run:
            print("dry_run: true")
            return 0

        if archive_bytes is not None:
            response = post_bundle_submission(
                api_url=args.api_url,
                hotkey=hotkey,
                submission_id=submission_id,
                signature=signature,
                identity=identity,
                archive_name=archive_name or "bundle.tar.gz",
                archive_bytes=archive_bytes,
            )
        elif bundle_files is not None:
            response = post_bundle_submission(
                api_url=args.api_url,
                hotkey=hotkey,
                submission_id=submission_id,
                signature=signature,
                identity=identity,
                archive_name="bundle.tar.gz",
                archive_bytes=build_tar_gz(bundle_files),
            )
        else:
            response = post_agent_submission(
                api_url=args.api_url,
                hotkey=hotkey,
                submission_id=submission_id,
                signature=signature,
                identity=identity,
                agent_filename=args.agent.name,
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


def load_submission_payload(args: argparse.Namespace) -> tuple[dict[str, bytes] | None, bytes | None, str | None]:
    if args.archive is not None:
        archive_path = args.archive.expanduser().resolve()
        archive_bytes = archive_path.read_bytes()
        if len(archive_bytes) > MAX_BUNDLE_BYTES:
            raise ValueError(f"{archive_path} is {len(archive_bytes)} bytes; max is {MAX_BUNDLE_BYTES} bytes")
        bundle_files = bundle_files_from_archive(archive_bytes, archive_name=archive_path.name)
        enforce_bundle_size_limit(bundle_files)
        return bundle_files, archive_bytes, archive_path.name
    if args.bundle is not None:
        bundle_files = collect_harness_from_directory(args.bundle)
        enforce_bundle_size_limit(bundle_files)
        return bundle_files, None, None
    return None, None, None


def collect_harness_from_directory(path: Path) -> dict[str, bytes]:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(f"--bundle must be a directory: {resolved}")
    files: dict[str, bytes] = {}
    for file_path in sorted(resolved.rglob("*.py")):
        relative = file_path.relative_to(resolved)
        if any(part.startswith(".") for part in relative.parts):
            continue
        if "scripts" in relative.parts:
            continue
        files[relative.as_posix()] = file_path.read_bytes()
    if DEFAULT_ENTRYPOINT not in files:
        raise ValueError(f"harness directory must include `{DEFAULT_ENTRYPOINT}`")
    return files


def bundle_files_from_archive(data: bytes, *, archive_name: str) -> dict[str, bytes]:
    lowered = archive_name.lower()
    if lowered.endswith((".tar.gz", ".tgz")) or data[:2] != b"PK":
        return files_from_tar_gz(data)
    return files_from_zip(data)


def files_from_tar_gz(data: bytes) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile() or member.issym() or member.islnk():
                continue
            path = normalize_relative_path(member.name)
            files[path] = archive.extractfile(member).read()  # type: ignore[union-attr]
    return files


def files_from_zip(data: bytes) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            path = normalize_relative_path(info.filename)
            files[path] = archive.read(info)
    return files


def normalize_relative_path(raw_path: str) -> str:
    path = raw_path.replace("\\", "/").lstrip("./")
    parts = [part for part in path.split("/") if part and part not in {".", ".."}]
    if not parts:
        raise ValueError(f"invalid archive path `{raw_path}`")
    return "/".join(parts)


def canonical_bundle_sha256(files: dict[str, bytes]) -> str:
    lines = [
        f"{path}:{hashlib.sha256(content).hexdigest()}"
        for path, content in sorted(files.items())
    ]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def enforce_bundle_size_limit(files: dict[str, bytes]) -> None:
    total = sum(len(content) for content in files.values())
    if total > MAX_BUNDLE_BYTES:
        raise ValueError(f"bundle is {total} bytes; maximum is {MAX_BUNDLE_BYTES} bytes")


def build_tar_gz(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path, content in sorted(files.items()):
            info = tarfile.TarInfo(name=path)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def load_wallet(args: argparse.Namespace) -> Any:
    try:
        bt = importlib.import_module("bittensor")
    except ImportError as exc:
        raise RuntimeError("bittensor is not installed in this Python environment") from exc
    wallet_kwargs = {"name": args.wallet_name, "hotkey": args.wallet_hotkey}
    if args.wallet_path:
        wallet_kwargs["path"] = args.wallet_path
    return bt.Wallet(**wallet_kwargs)


def derive_submission_id(*, hotkey: str, content_sha256: str) -> str:
    safe_hotkey = re.sub(r"[^A-Za-z0-9_.-]", "-", hotkey)[:16] or "hotkey"
    return f"{safe_hotkey}-{content_sha256[:16]}"


def private_submission_signature_payload(*, hotkey: str, submission_id: str, agent_sha256: str) -> bytes:
    return f"tau-private-submission-v1:{hotkey}:{submission_id}:{agent_sha256.lower()}".encode("utf-8")


def bundle_signature_payload(*, hotkey: str, submission_id: str, bundle_sha256: str) -> bytes:
    return f"tau-private-submission-v2:{hotkey}:{submission_id}:{bundle_sha256.lower()}".encode("utf-8")


def sign_payload(wallet: Any, payload: bytes) -> str:
    signature = wallet.hotkey.sign(payload)
    if isinstance(signature, str):
        return signature.removeprefix("0x")
    if isinstance(signature, bytes):
        return signature.hex()
    if hasattr(signature, "hex"):
        return signature.hex()
    raise TypeError(f"unsupported signature type: {type(signature).__name__}")


def username_signature_payload(username: str) -> bytes:
    return f"tau-agent-submission-username:{username}".encode("utf-8")


def wallet_coldkey_address(wallet: Any) -> str | None:
    for attr in ("coldkeypub", "coldkey"):
        value = getattr(wallet, attr, None)
        ss58 = getattr(value, "ss58_address", None)
        if ss58:
            return str(ss58)
    return None


def sign_with_coldkey(wallet: Any, payload: bytes) -> str:
    coldkey = getattr(wallet, "coldkey", None)
    if coldkey is None or not hasattr(coldkey, "sign"):
        raise RuntimeError("loaded wallet coldkey is not available; pass --coldkey-signature")
    signature = coldkey.sign(payload)
    if isinstance(signature, str):
        return signature.removeprefix("0x")
    if isinstance(signature, bytes):
        return signature.hex()
    if hasattr(signature, "hex"):
        return signature.hex()
    raise TypeError(f"unsupported coldkey signature type: {type(signature).__name__}")


def build_username_identity(*, args: argparse.Namespace, wallet: Any) -> dict[str, str]:
    username = (args.agent_username or "").strip()
    coldkey = (args.coldkey or "").strip()
    signature = (args.coldkey_signature or "").strip()
    if not any((username, coldkey, signature)):
        return {}
    if not username:
        raise ValueError("--agent-username is required when passing username identity fields")
    if not coldkey:
        coldkey = wallet_coldkey_address(wallet) or ""
    if not coldkey:
        raise ValueError("--coldkey is required because the loaded wallet coldkey address was not available")
    if not signature:
        signature = sign_with_coldkey(wallet, username_signature_payload(username))
    return {
        "agent_username": username,
        "coldkey": coldkey,
        "coldkey_signature": signature,
    }


def print_request_summary(
    *,
    source_label: str,
    hotkey: str,
    submission_id: str,
    agent_sha256: str,
    bundle_sha256: str | None,
    signature_payload: bytes,
    identity: dict[str, str],
) -> None:
    print(f"source: {source_label}")
    print(f"hotkey: {hotkey}")
    print(f"submission_id: {submission_id}")
    print(f"agent_sha256: {agent_sha256}")
    if bundle_sha256:
        print(f"bundle_sha256: {bundle_sha256}")
    print(f"signature_payload: {signature_payload.decode('utf-8')}")
    if identity:
        print(f"agent_username: {identity['agent_username']}")
        print(f"coldkey: {identity['coldkey']}")
        print(f"username_signature_payload: {username_signature_payload(identity['agent_username']).decode('utf-8')}")


def post_agent_submission(
    *,
    api_url: str,
    hotkey: str,
    submission_id: str,
    signature: str,
    identity: dict[str, str],
    agent_filename: str,
    agent_py: bytes,
) -> dict[str, Any]:
    fields = {
        "hotkey": hotkey,
        "submission_id": submission_id,
        "signature": signature,
    }
    fields.update(identity)
    files = {"agent": (agent_filename, agent_py, "text/x-python")}
    body, content_type = encode_multipart_form(fields=fields, files=files)
    return post_multipart(api_url=api_url, body=body, content_type=content_type)


def post_bundle_submission(
    *,
    api_url: str,
    hotkey: str,
    submission_id: str,
    signature: str,
    identity: dict[str, str],
    archive_name: str,
    archive_bytes: bytes,
) -> dict[str, Any]:
    fields = {
        "hotkey": hotkey,
        "submission_id": submission_id,
        "signature": signature,
    }
    fields.update(identity)
    content_type = "application/gzip" if archive_name.lower().endswith((".tar.gz", ".tgz")) else "application/zip"
    files = {"bundle": (archive_name, archive_bytes, content_type)}
    body, multipart_type = encode_multipart_form(fields=fields, files=files)
    return post_multipart(api_url=api_url, body=body, content_type=multipart_type)


def post_multipart(*, api_url: str, body: bytes, content_type: str) -> dict[str, Any]:
    request = urllib.request.Request(
        api_url,
        data=body,
        headers={
            "Accept": "application/json",
            "Content-Type": content_type,
            "User-Agent": USER_AGENT,
        },
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
    try:
        payload = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("submission API returned non-JSON response") from exc
    if not isinstance(payload, dict):
        raise ValueError("submission API returned non-object JSON")
    return payload


def validate_private_commitment(commitment: str) -> None:
    if not PRIVATE_SUBMISSION_RE.fullmatch(commitment):
        raise ValueError("accepted API response did not include a valid private-submission commitment")


if __name__ == "__main__":
    sys.exit(main())
