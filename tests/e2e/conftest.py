"""Browser tests: the real page in public/, served by the API with a fake model."""

import json
import os
import re
import socket
import threading
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
import uvicorn

from sms_fraud.config import Settings
from sms_fraud.main import create_app

playwright_api = pytest.importorskip("playwright.sync_api")

FRAUD_WORDS = ("won", "win", "urgent")


class KeywordPredictor:
    version = "e2e@0000000+fake"

    def __init__(self, settings: Settings) -> None:
        pass

    def predict_proba(self, messages: Sequence[str]) -> list[float]:
        return [0.97 if any(w in m.lower() for w in FRAUD_WORDS) else 0.04 for m in messages]


def _vercel_headers(path: str) -> dict[str, str]:
    """Headers vercel.json applies to path (its sources are regexes in path-to-regexp form)."""
    config = json.loads((Path(__file__).resolve().parents[2] / "vercel.json").read_text())
    headers: dict[str, str] = {}
    for rule in config.get("headers", []):
        if re.fullmatch(rule["source"], path):
            headers.update({h["key"]: h["value"] for h in rule["headers"]})
    return headers


def _with_production_headers(app):
    """Serve every response with the headers Vercel adds, CSP included."""

    @app.middleware("http")
    async def add_headers(request, call_next):
        response = await call_next(request)
        response.headers.update(_vercel_headers(request.url.path))
        return response

    return app


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="session")
def base_url() -> Iterator[str]:
    settings = Settings(_env_file=None, frontend_dir="public", rate_limit_per_minute=0)
    app = _with_production_headers(create_app(settings, predictor_factory=KeywordPredictor))
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("test server did not start")
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(scope="session")
def browser() -> Iterator["playwright_api.Browser"]:
    with playwright_api.sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch()
        except Exception as exc:  # browser binaries not installed
            if os.environ.get("E2E_REQUIRED"):
                raise
            pytest.skip(f"Chromium unavailable: {exc}")
        yield browser
        browser.close()


@pytest.fixture
def page(browser, base_url: str) -> Iterator["playwright_api.Page"]:
    context = browser.new_context(locale="en-US", viewport={"width": 1200, "height": 800})
    page = context.new_page()
    errors: list[str] = []
    page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
    page.on("pageerror", lambda exc: errors.append(str(exc)))
    page.goto(base_url)
    yield page
    context.close()
    assert not [e for e in errors if "Failed to load resource" not in e], errors
