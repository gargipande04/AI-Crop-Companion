"""
Root entrypoint kept for backwards compatibility.

The FastAPI app lives in app/server.py, the application is
started:

    python main.py
"""

from app.server import app, run


if __name__ == "__main__":
    run()
