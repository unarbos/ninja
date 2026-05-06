#!/usr/bin/env python3
"""Submit a Subnet 66 miner commitment with Bittensor.

Example:
    ./scripts/commit_on_chain.py \
        --wallet-name my-wallet \
        --wallet-hotkey default \
        --hotkey 5... \
        --commit "github-pr:unarbos/ninja#12@0123456789abcdef0123456789abcdef01234567"
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
import sys
from typing import Any


DEFAULT_NETUID = 66
MAX_RAW_COMMITMENT_BYTES = 128
HOTKEY_SPENT_SINCE_BLOCK = 8_104_340
NINJA_PR_COMMITMENT_RE = re.compile(
    r"^github-pr:unarbos/ninja#(?P<number>\d+)@(?P<sha>[0-9a-fA-F]{7,64})$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit a Bittensor on-chain commitment for a ninja PR."
    )
    parser.add_argument(
        "commitment",
        nargs="?",
        help='Commitment string, usually "github-pr:unarbos/ninja#<pr>@<head-sha>".',
    )
    parser.add_argument(
        "--commit",
        dest="commit",
        help='Commitment string, usually "github-pr:unarbos/ninja#<pr>@<head-sha>".',
    )
    parser.add_argument(
        "--wallet-name",
        default=os.getenv("BT_WALLET_NAME", "default"),
        help="Bittensor wallet name. Defaults to BT_WALLET_NAME or 'default'.",
    )
    parser.add_argument(
        "--wallet-hotkey",
        default=os.getenv("BT_WALLET_HOTKEY", "default"),
        help="Bittensor wallet hotkey name. Defaults to BT_WALLET_HOTKEY or 'default'.",
    )
    parser.add_argument(
        "--wallet-path",
        default=os.getenv("BT_WALLET_PATH"),
        help="Optional Bittensor wallet path. Defaults to BT_WALLET_PATH.",
    )
    parser.add_argument(
        "--hotkey",
        help="Expected miner hotkey SS58 address. When set, the loaded wallet hotkey must match.",
    )
    parser.add_argument(
        "--netuid",
        type=int,
        default=int(os.getenv("BT_NETUID", DEFAULT_NETUID)),
        help=f"Subnet netuid. Defaults to BT_NETUID or {DEFAULT_NETUID}.",
    )
    parser.add_argument(
        "--network",
        default=os.getenv("BT_SUBTENSOR_NETWORK"),
        help="Optional Bittensor network or websocket endpoint.",
    )
    parser.add_argument(
        "--period",
        type=int,
        default=128,
        help="Extrinsic validity period in blocks.",
    )
    parser.add_argument(
        "--wait-finalization",
        action="store_true",
        help="Wait for finalization instead of only inclusion.",
    )
    parser.add_argument(
        "--no-wait-inclusion",
        action="store_true",
        help="Submit without waiting for block inclusion.",
    )
    parser.add_argument(
        "--mev-protection",
        action="store_true",
        help="Submit through Bittensor MEV protection.",
    )
    parser.add_argument(
        "--skip-registration-check",
        action="store_true",
        help="Skip checking that the hotkey is registered on the subnet before submit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print what would be submitted without sending an extrinsic.",
    )
    return parser.parse_args()


def resolve_commitment(args: argparse.Namespace) -> str:
    if args.commit and args.commitment and args.commit != args.commitment:
        raise ValueError("received different commitment strings from --commit and positional argument")
    commitment = args.commit or args.commitment
    if commitment is None:
        raise ValueError("commitment is required; pass --commit or a positional commitment string")
    return commitment.strip()


def validate_commitment(commitment: str) -> None:
    if not commitment:
        raise ValueError("commitment string is empty")

    try:
        encoded = commitment.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("commitment must be ASCII") from exc

    if len(encoded) > MAX_RAW_COMMITMENT_BYTES:
        raise ValueError(
            f"commitment is {len(encoded)} bytes; Bittensor Raw metadata supports at most "
            f"{MAX_RAW_COMMITMENT_BYTES} bytes"
        )

    if not NINJA_PR_COMMITMENT_RE.fullmatch(commitment):
        raise ValueError(
            "commitment must match github-pr:unarbos/ninja#<pr-number>@<head-sha>; "
            "raw owner/repo@sha commitments are not accepted by the validator"
        )


def response_field(response: Any, field: str) -> Any:
    return getattr(response, field, None)


def print_response(response: Any) -> None:
    print(f"success: {bool(response_field(response, 'success'))}")
    message = response_field(response, "message")
    if message:
        print(f"message: {message}")

    receipt = response_field(response, "extrinsic_receipt")
    if receipt is not None:
        extrinsic_hash = getattr(receipt, "extrinsic_hash", None)
        block_hash = getattr(receipt, "block_hash", None)
        if extrinsic_hash:
            print(f"extrinsic_hash: {extrinsic_hash}")
        if block_hash:
            print(f"block_hash: {block_hash}")


def load_bittensor():
    try:
        return importlib.import_module("bittensor")
    except ImportError as exc:  # pragma: no cover - depends on the caller environment
        raise RuntimeError("bittensor is not installed in this Python environment") from exc


def load_wallet(bt: Any, args: argparse.Namespace):
    wallet_kwargs = {"name": args.wallet_name, "hotkey": args.wallet_hotkey}
    if args.wallet_path:
        wallet_kwargs["path"] = args.wallet_path
    return bt.Wallet(**wallet_kwargs)


def main() -> int:
    args = parse_args()

    try:
        commitment = resolve_commitment(args)
        validate_commitment(commitment)
        bt = load_bittensor()
        wallet = load_wallet(bt, args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    wallet_hotkey = wallet.hotkey.ss58_address
    if args.hotkey and wallet_hotkey != args.hotkey:
        print(
            f"error: loaded wallet hotkey {wallet_hotkey} does not match --hotkey {args.hotkey}",
            file=sys.stderr,
        )
        return 2

    print(f"wallet_name: {args.wallet_name}")
    print(f"wallet_hotkey_name: {args.wallet_hotkey}")
    print(f"wallet_hotkey: {wallet_hotkey}")
    print(f"netuid: {args.netuid}")
    print(f"commitment: {commitment}")
    print(f"hotkey_spent_since_block: {HOTKEY_SPENT_SINCE_BLOCK}")

    if args.dry_run:
        print("dry_run: true")
        return 0

    try:
        with bt.SubtensorApi(network=args.network, websocket_shutdown_timer=0) as subtensor:
            print(f"chain: {subtensor}")
            print(f"block: {subtensor.block}")

            if not args.skip_registration_check:
                uid = subtensor.subnets.get_uid_for_hotkey_on_subnet(wallet_hotkey, args.netuid)
                if uid is None:
                    print(
                        f"error: hotkey {wallet_hotkey} is not registered on subnet {args.netuid}",
                        file=sys.stderr,
                    )
                    return 1
                print(f"uid: {uid}")

            response = subtensor.commitments.set_commitment(
                wallet=wallet,
                netuid=args.netuid,
                data=commitment,
                period=args.period,
                wait_for_inclusion=not args.no_wait_inclusion,
                wait_for_finalization=args.wait_finalization,
                mev_protection=args.mev_protection,
                raise_error=False,
            )
            print_response(response)
            if not response.success:
                return 1

            if not args.no_wait_inclusion:
                current = subtensor.commitments.get_all_commitments(args.netuid).get(wallet_hotkey)
                if current is not None:
                    print(f"readback: {current}")
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
