---
title: AI Assignment Grader
emoji: 🎓
colorFrom: slate
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# 🎓 AI Assignment Grader

Grade student Jupyter notebooks using Qwen2.5-Coder:7b via Ollama.

Upload assignment PDF + rubric TXT + student IPYNB and get:
- Total score out of 25
- Strengths with code references
- Detailed rubric violations with fixes and code snippets

**First load takes 3–5 minutes** (model download ~4.5 GB). Subsequent starts are fast.