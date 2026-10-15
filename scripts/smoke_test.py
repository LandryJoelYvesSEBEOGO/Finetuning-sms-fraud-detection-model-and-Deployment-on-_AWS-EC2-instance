"""Smoke-test a running API instance.

Usage: python scripts/smoke_test.py [BASE_URL]   (default: http://localhost:8080)
Exits non-zero if any check fails. Uses only the standard library.
"""

import json
import sys
import urllib.request

SAMPLES = [
    "Hey, are we still on for lunch tomorrow at 12?",
    "URGENT: Your bank account has been suspended. Verify your details at http://secure-bank-verify.co",
]


def call(base_url: str, path: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data else "GET",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def main() -> int:
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"
    print("health:", call(base_url, "/health"))
    print("ready: ", call(base_url, "/ready"))
    for message in SAMPLES:
        result = call(base_url, "/v1/predict", {"message": message})
        print(f"{result['label']:>10}  score={result['fraud_score']:.3f}  {message[:60]}")
    batch = call(base_url, "/v1/predict/batch", {"messages": SAMPLES})
    assert len(batch["predictions"]) == len(SAMPLES)
    print("batch:  ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
