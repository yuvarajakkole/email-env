---
title: Email Triage RL Environment
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Email Triage RL Environment

See [README.md](README.md) for full documentation.

**API Docs:** After deployment, visit `/docs` for the interactive Swagger UI.

**Quick test:**
```bash
curl https://<your-space>.hf.space/health
curl -X POST https://<your-space>.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "hard", "seed": 42}'
```
