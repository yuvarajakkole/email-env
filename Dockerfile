# ─────────────────────────────────────────────────────────────
# Email Triage RL Environment — Dockerfile
# ─────────────────────────────────────────────────────────────
# Build:
#   docker build -t email-triage-env .
#
# Run (pass .env at runtime — NEVER bake secrets into the image):
#   docker run -p 8000:8000 --env-file .env email-triage-env
#
# Or with individual env vars:
#   docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... email-triage-env
#
# The API key is loaded by config.py from the .env file or environment.
# It is NEVER stored in the image, NEVER logged, NEVER in source code.
# ─────────────────────────────────────────────────────────────

FROM python:3.11-slim

LABEL maintainer="OpenEnv Competition"
LABEL description="Email Triage RL Environment v3 — OpenEnv Spec"
LABEL version="3.0.0"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy source — .env is in .dockerignore so it is NEVER copied
COPY . .


# Create package __init__ files
RUN touch env/__init__.py tasks/__init__.py graders/__init__.py \
         rewards/__init__.py data/__init__.py agent/__init__.py 2>/dev/null || true

ENV PYTHONPATH=/app
# PORT and secrets come from --env-file .env at runtime, not here
ENV PORT=7860

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["python", "server.py"]
