"""
app.py — Hugging Face Spaces entry point
-----------------------------------------
HF Spaces expects an app.py. This simply imports and starts the FastAPI server.
The server is defined in server.py.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from config import settings
from server import app  # noqa: F401

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=settings.port, reload=False)
