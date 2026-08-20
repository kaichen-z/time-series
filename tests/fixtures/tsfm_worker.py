#!/usr/bin/env python3
"""Standalone JSON-lines worker used by broker integration tests."""

from __future__ import annotations

import json
import os
import sys
import time


def _response(request: dict[str, object], mode: str) -> dict[str, object]:
    request_id = str(request["request_id"])
    if mode == "request_id_mismatch":
        request_id = f"wrong-{request_id}"
    if mode == "non_finite":
        values = [11.0, float("nan")]
    else:
        values = [11.0, 12.0]
    return {
        "protocol_version": 1,
        "request_id": request_id,
        "status": "success",
        "values": values,
        "metadata": {"pid": os.getpid()},
    }


def main() -> int:
    mode = sys.argv[1]
    state_path = sys.argv[2] if len(sys.argv) > 2 else None
    if mode == "never_read":
        if state_path is not None:
            with open(state_path, "w", encoding="utf-8") as handle:
                handle.write(str(os.getpid()))
        time.sleep(60)
        return 0
    if mode == "record_pid" and state_path is not None:
        with open(state_path, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
            handle.flush()

    for line in sys.stdin:
        request = json.loads(line)
        if mode == "record_request" and state_path is not None:
            with open(state_path, "w", encoding="utf-8") as handle:
                json.dump(request, handle, sort_keys=True)
        if mode == "timeout":
            if state_path is not None:
                with open(state_path, "w", encoding="utf-8") as handle:
                    handle.write(str(os.getpid()))
            time.sleep(60)
            continue
        if mode == "crash_once" and state_path is not None and not os.path.exists(state_path):
            with open(state_path, "w", encoding="utf-8") as handle:
                handle.write(str(os.getpid()))
            return 23
        if mode == "malformed":
            sys.stdout.write("not-json\n")
        elif mode == "malformed_secret_status":
            sys.stdout.write(
                '{"protocol_version":1,"request_id":'
                + json.dumps(request["request_id"])
                + ',"status":'
                + json.dumps(os.environ["HF_TOKEN"])
                + ',"reason_code":"broken","message":"broken"}\n'
            )
        elif mode == "environment_snapshot":
            hf_token = os.environ.get("HF_TOKEN", "")
            legacy_hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
            tabpfn_token = os.environ.get("TABPFN_TOKEN", "")
            sys.stdout.write(
                json.dumps(
                    {
                        "protocol_version": 1,
                        "request_id": request["request_id"],
                        "status": "success",
                        "values": [11.0, 12.0],
                        "metadata": {
                            "credentials_present": bool(
                                hf_token and legacy_hf_token and tabpfn_token
                            ),
                            "excluded_present": any(
                                name in os.environ
                                for name in (
                                    "GITHUB_PAT",
                                    "SSH_PRIVATE_KEY",
                                    "AWS_ACCESS_KEY_ID",
                                )
                            ),
                            "safe_environment": {
                                name: os.environ.get(name)
                                for name in ("HOME", "PATH", "HF_HOME")
                            },
                            hf_token: {
                                "tokens": [hf_token, legacy_hf_token, tabpfn_token],
                                "request_frequency": request["frequency"],
                            },
                        },
                    }
                )
                + "\n"
            )
        elif mode == "inherited_token_failure":
            sys.stdout.write(
                json.dumps(
                    {
                        "protocol_version": 1,
                        "request_id": request["request_id"],
                        "status": "unavailable",
                        "reason_code": "checkpoint_unavailable",
                        "message": f"upstream included {os.environ['HF_TOKEN']}",
                    }
                )
                + "\n"
            )
        elif mode == "partial_line":
            sys.stdout.write(json.dumps(_response(request, "success"), allow_nan=True))
            sys.stdout.flush()
            time.sleep(2)
            sys.stdout.write("\n")
        elif (
            mode == "extreme_integer_once"
            and state_path is not None
            and not os.path.exists(state_path)
        ):
            with open(state_path, "w", encoding="utf-8") as handle:
                handle.write(str(os.getpid()))
            sys.stdout.write(
                '{"protocol_version":1,"request_id":'
                + json.dumps(request["request_id"])
                + ',"status":"success","values":['
                + "1"
                + ("0" * 400)
                + '],"metadata":{}}\n'
            )
        else:
            sys.stdout.write(json.dumps(_response(request, mode), allow_nan=True) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
