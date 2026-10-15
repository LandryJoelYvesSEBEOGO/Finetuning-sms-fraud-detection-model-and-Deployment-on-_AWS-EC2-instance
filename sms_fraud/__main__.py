"""Run the API locally: ``python -m sms_fraud``."""

import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "sms_fraud.main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8080")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
    )
