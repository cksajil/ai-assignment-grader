"""
AI Assignment Grader — app.py
HuggingFace Spaces / local entry point.

For Google Colab usage, open AI_Grader_Complete_v2.ipynb instead.
"""

import json
import os
import re
import subprocess
import tempfile
import textwrap
import time
import traceback
import threading

import gradio as gr
import fitz  # PyMuPDF
import nbformat
import requests
from tenacity import retry, stop_after_attempt, wait_fixed

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
# MODEL_NAME   = os.getenv("MODEL_NAME", "qwen2.5-coder:7b")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5-coder:3b")
OLLAMA_URL = "http://localhost:11434/api/chat"
# MAX_CODE_LEN = 3500
MAX_CODE_LEN = 2000
TIMEOUT = 300

# ─────────────────────────────────────────────────────────────────────────────
# File Parsers
# ─────────────────────────────────────────────────────────────────────────────


def extract_pdf_text(pdf_path: str) -> str:
    """Extract all text from a PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    pages = [page.get_text("text").strip() for page in doc]
    doc.close()
    full = "\n\n".join(p for p in pages if p)
    if not full.strip():
        raise ValueError("PDF appears empty or image-only (no extractable text).")
    # return full[:1500] if len(full) > 1500 else full
    return full[:800] if len(full) > 800 else full


def extract_notebook_code(ipynb_path: str) -> str:
    """Extract code + markdown cells from .ipynb with cell separators."""
    nb = nbformat.read(ipynb_path, as_version=4)
    parts = []
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            src = cell.source.strip()
            if src:
                parts.append(f"# ── Code Cell {i+1} ──\n{src}")
        elif cell.cell_type == "markdown":
            src = cell.source.strip()
            if src:
                parts.append(
                    f"# ── Markdown Cell {i+1} ──\n# " + "\n# ".join(src.splitlines())
                )
    if not parts:
        raise ValueError("Notebook has no code cells.")
    code = "\n\n".join(parts)
    if len(code) > MAX_CODE_LEN:
        code = code[:MAX_CODE_LEN] + f"\n\n# ... [truncated to {MAX_CODE_LEN} chars]"
    return code


def run_code_sandbox(ipynb_path: str, timeout: int = 30) -> dict:
    """Execute notebook code in a subprocess sandbox."""
    try:
        nb = nbformat.read(ipynb_path, as_version=4)
        code_lines = []
        for cell in nb.cells:
            if cell.cell_type == "code":
                lines = [
                    l
                    for l in cell.source.splitlines()
                    if not l.strip().startswith(("!", "%"))
                ]
                if lines:
                    code_lines.extend(lines)
                    code_lines.append("")
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            f.write("\n".join(code_lines))
            tmp_py = f.name
        result = subprocess.run(
            ["python", tmp_py], capture_output=True, text=True, timeout=timeout
        )
        os.unlink(tmp_py)
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout[:2000],
            "stderr": result.stderr[:1000],
            "error": None,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": "",
            "error": f"Timed out after {timeout}s",
        }
    except Exception as e:
        return {"success": False, "stdout": "", "stderr": "", "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a strict programming instructor grading a Jupyter notebook.

    SCORING: Evaluate each rubric criterion. Score cannot exceed criterion max. Total cannot exceed 25. Verify sum before responding.

    RUBRIC CHECK: For every penalty rule violated, add one areas_to_improve entry with rubric_requirement set to that exact rule. One violation = one entry.

    FEEDBACK: Strengths must cite actual code. Issues must name exact variable/function. No vague feedback. No invented mistakes.

    Respond ONLY with this JSON:
    {
      "criterion_scores": [{"name": str, "score": int, "max": int}],
      "total_score": int,
      "strengths": [str],
      "areas_to_improve": [{
        "category": "Bug|Code Quality|Data Preprocessing|Modeling|Missing Requirement",
        "rubric_requirement": str,
        "issue": str,
        "why_it_matters": str,
        "fix": str
      }]
    }
"""
).strip()
# SYSTEM_PROMPT = textwrap.dedent(
#     """
#     You are a strict but fair programming instructor grading a student Jupyter notebook.

#     You will receive:
#     - An assignment question
#     - A grading rubric with criteria, point values, and explicit penalty rules
#     - The student code extracted from their notebook
#     - Optionally: sandbox execution output

#     SCORING RULES — follow exactly:
#     1. Read each criterion in the rubric. Note its maximum points.
#     2. Apply every penalty listed that applies to the student code.
#     3. A criterion score CANNOT exceed its stated maximum.
#     4. Sum all criterion scores. The total CANNOT exceed 25.
#     5. If no rubric criterion mentions a topic, do not award or deduct for it.
#     6. When in doubt, deduct — do not give benefit of the doubt.

#     RUBRIC CROSS-CHECK — mandatory:
#     Before writing areas_to_improve, scan every single rubric penalty rule.
#     For each rule, ask: "Did the student violate this?"
#     - YES → create one areas_to_improve entry with rubric_requirement set to that exact rule.
#     - NO  → skip it.
#     Each violation = one separate entry. Do not merge separate issues.

#     FEEDBACK RULES:
#     - Strengths: cite actual code, function names, or techniques the student used.
#     - areas_to_improve entries must be specific and forensic.
#     - Do NOT be vague. "Improve variable names" is rejected.
#     - Do NOT invent mistakes not visible in the code.
#     - The sum of criterion_scores must equal total_score. Verify before responding.
#     - Respond ONLY with valid JSON. No markdown, no text outside the JSON.

#     Return exactly this schema:
#     {
#       "criterion_scores": [
#         {"name": "<criterion name>", "score": <integer>, "max": <integer>}
#       ],
#       "total_score": <integer — must equal sum of criterion_scores, max 25>,
#       "strengths": ["Specific strength with code reference"],
#       "areas_to_improve": [
#         {
#           "category": "<Bug | Code Quality | Data Preprocessing | Modeling | Missing Requirement>",
#           "rubric_requirement": "The exact rubric penalty rule that was violated",
#           "issue": "What the student did wrong or completely missed",
#           "why_it_matters": "The consequence or reason this is wrong",
#           "fix": "Concrete suggestion or corrected code snippet"
#         }
#       ]
#     }
# """
# ).strip()


def build_user_prompt(question, rubric, code, execution):
    parts = [
        f"=== ASSIGNMENT QUESTION ===\n{question.strip()}",
        f"=== GRADING RUBRIC ===\n{rubric.strip()}",
        f"=== STUDENT CODE ===\n{code.strip()}",
    ]
    if execution:
        status = "✓ ran successfully" if execution["success"] else "✗ failed"
        block = f"Status: {status}\n"
        if execution["stdout"]:
            block += f"Output:\n{execution['stdout']}\n"
        if execution["stderr"]:
            block += f"Errors:\n{execution['stderr']}\n"
        if execution["error"]:
            block += f"Exception: {execution['error']}\n"
        parts.append(f"=== EXECUTION RESULTS ===\n{block}")
    parts.append(
        "=== YOUR TASK ===\n"
        "Go through every rubric penalty rule line by line.\n"
        "For each one violated, add an entry to areas_to_improve.\n"
        "Return ONLY the JSON schema specified."
    )
    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Ollama call
# ─────────────────────────────────────────────────────────────────────────────


@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def call_ollama(question, rubric, code, execution):
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_prompt(question, rubric, code, execution),
            },
        ],
        "stream": False,
        "format": "json",
        # "options": {
        #     "temperature": 0.0,
        #     "num_predict": 1500,
        #     "num_ctx": 4096,
        #     "num_thread": 4,
        #     "num_batch": 512,
        # },
        "options": {
            "temperature": 0.0,
            "num_predict": 800,  # was 1500 — your JSON output is ~400 tokens max
            "num_ctx": 2048,  # was 4096 — rubric + code fits in 2048 easily
            "num_thread": 2,  # match exactly to free tier vCPU count
            "num_batch": 128,  # smaller batch = less memory pressure on 2 vCPU
        },
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    raw = resp.json()["message"]["content"]
    raw = re.sub(r"^```json\s*", "", raw.strip())
    raw = re.sub(r"```$", "", raw.strip())
    result = json.loads(raw)

    # Python-side score guard — LLM cannot inflate scores
    criteria = result.get("criterion_scores", [])
    if criteria:
        for c in criteria:
            c["score"] = min(c.get("score", 0), c.get("max", 0))
        computed = sum(c["score"] for c in criteria)
        result["total_score"] = min(computed, 25)
    else:
        result["total_score"] = min(result.get("total_score", 0), 25)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# HTML renderer
# ─────────────────────────────────────────────────────────────────────────────

CATEGORY_COLORS = {
    "Bug": ("#fee2e2", "#dc2626", "#fca5a5"),
    "Code Quality": ("#fef9c3", "#ca8a04", "#fde047"),
    "Data Preprocessing": ("#ede9fe", "#7c3aed", "#c4b5fd"),
    "Modeling": ("#fff7ed", "#ea580c", "#fdba74"),
    "Missing Requirement": ("#f0f9ff", "#0284c7", "#7dd3fc"),
}


def grade_to_color(pct):
    if pct >= 0.85:
        return "#22c55e"
    if pct >= 0.70:
        return "#84cc16"
    if pct >= 0.55:
        return "#f59e0b"
    if pct >= 0.40:
        return "#f97316"
    return "#ef4444"


def render_html_report(result, llm_elapsed=0.0, total_elapsed=0.0):
    total = result.get("total_score", 0)
    pct = total / 25
    color = grade_to_color(pct)
    grade = (
        "A+"
        if pct >= 0.90
        else (
            "A"
            if pct >= 0.80
            else (
                "B"
                if pct >= 0.70
                else "C" if pct >= 0.60 else "D" if pct >= 0.50 else "F"
            )
        )
    )

    # Criterion breakdown
    criteria = result.get("criterion_scores", [])
    crit_html = ""
    if criteria:
        rows = ""
        for c in criteria:
            c_pct = c["score"] / c["max"] if c.get("max") else 0
            c_col = grade_to_color(c_pct)
            rows += f"""
            <tr style="border-bottom:1px solid #f1f5f9">
              <td style="padding:8px 12px;font-size:13px;color:#374151;font-weight:500">{c['name']}</td>
              <td style="padding:8px 12px;text-align:center;font-weight:700;color:{c_col};font-size:14px">{c['score']}/{c['max']}</td>
              <td style="padding:8px 12px;width:140px">
                <div style="background:#e5e7eb;border-radius:99px;height:7px">
                  <div style="background:{c_col};width:{int(c_pct*100)}%;height:7px;border-radius:99px"></div>
                </div>
              </td>
            </tr>"""
        crit_html = f"""
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:16px;margin-bottom:16px">
          <h3 style="margin:0 0 12px;font-size:14px;color:#475569;font-weight:600">Criterion Breakdown</h3>
          <table style="width:100%;border-collapse:collapse">
            <thead><tr style="border-bottom:2px solid #e5e7eb">
              <th style="padding:6px 12px;text-align:left;font-size:12px;color:#6b7280">Criterion</th>
              <th style="padding:6px 12px;text-align:center;font-size:12px;color:#6b7280">Score</th>
              <th style="padding:6px 12px;font-size:12px;color:#6b7280">Progress</th>
            </tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </div>"""

    # Strengths
    strengths_li = "".join(
        f"<li style='margin-bottom:8px;line-height:1.6'>{s}</li>"
        for s in result.get("strengths", [])
    )

    # Improvement cards
    improve_cards = ""
    for item in result.get("areas_to_improve", []):
        cat = item.get("category", "Code Quality")
        rubric_req = item.get("rubric_requirement", "")
        issue = item.get("issue", "")
        why = item.get("why_it_matters", "")
        fix = item.get("fix", "")
        bg, text, border = CATEGORY_COLORS.get(cat, ("#f8fafc", "#475569", "#cbd5e1"))

        rubric_badge = ""
        if rubric_req:
            rubric_badge = f"""
            <div style="background:#1e293b;border-radius:6px;padding:7px 12px;margin-bottom:10px;display:flex;align-items:flex-start;gap:8px">
              <span style="color:#f59e0b;font-size:11px;font-weight:700;white-space:nowrap;margin-top:1px">RUBRIC</span>
              <span style="color:#e2e8f0;font-size:12px;line-height:1.5;font-style:italic">"{rubric_req}"</span>
            </div>"""

        is_code = any(
            tok in fix
            for tok in ["\n", "def ", "df.", "import ", " = ", "()", "[]", ":"]
        )
        fix_html = (
            f"<pre style='background:#1e293b;color:#e2e8f0;padding:10px 14px;border-radius:6px;"
            f"font-size:12px;overflow-x:auto;margin:8px 0 0;white-space:pre-wrap'>{fix}</pre>"
            if is_code
            else f"<p style='margin:6px 0 0;font-size:13px;color:#374151'><b>Fix:</b> {fix}</p>"
        )

        improve_cards += f"""
        <div style="border:1px solid {border};background:{bg};border-radius:10px;padding:16px;margin-bottom:14px">
          <div style="display:flex;align-items:flex-start;gap:8px;margin-bottom:10px">
            <span style="background:{text};color:white;font-size:11px;font-weight:700;padding:3px 9px;border-radius:99px;white-space:nowrap;margin-top:1px">{cat}</span>
            <span style="font-weight:600;font-size:14px;color:#111827;line-height:1.4">{issue}</span>
          </div>
          {rubric_badge}
          <p style="margin:0 0 4px;font-size:13px;color:#4b5563;line-height:1.5"><b>Why it matters:</b> {why}</p>
          {fix_html}
        </div>"""

    if not improve_cards:
        improve_cards = "<div style='text-align:center;padding:24px;color:#6b7280;font-size:13px'>No rubric violations found.</div>"

    n_issues = len(result.get("areas_to_improve", []))

    return f"""
    <div style="font-family:'Segoe UI',system-ui,sans-serif;max-width:780px;margin:0 auto">
      <div style="background:linear-gradient(135deg,#1e293b,#0f172a);border-radius:16px;padding:28px 32px;margin-bottom:20px;display:flex;justify-content:space-between;align-items:center">
        <div>
          <div style="color:#94a3b8;font-size:12px;text-transform:uppercase;letter-spacing:.1em;margin-bottom:4px">Total Score</div>
          <div style="color:#f1f5f9;font-size:42px;font-weight:900;line-height:1">
            {total}<span style="font-size:20px;font-weight:400;color:#94a3b8"> / 25</span>
          </div>
          <div style="margin-top:14px;background:#1e3a5f;border-radius:99px;height:10px;width:260px">
            <div style="background:{color};width:{int(pct*100)}%;height:10px;border-radius:99px"></div>
          </div>
          <div style="color:#94a3b8;font-size:13px;margin-top:6px">{int(pct*100)}% · Qwen2.5-Coder via Ollama</div>
        </div>
        <div style="text-align:center">
          <div style="width:80px;height:80px;border-radius:50%;background:{color};display:flex;align-items:center;justify-content:center;font-size:32px;font-weight:900;color:white">{grade}</div>
          <div style="color:#94a3b8;font-size:11px;margin-top:6px">Grade</div>
        </div>
      </div>

      {crit_html}

      <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:12px;padding:20px;margin-bottom:16px">
        <h3 style="margin:0 0 12px;color:#15803d;font-size:15px;font-weight:600">Strengths</h3>
        <ul style="margin:0;padding-left:20px;color:#166534;font-size:13px;line-height:1.7">
          {strengths_li or "<li>No specific strengths identified.</li>"}
        </ul>
      </div>

      <div style="background:white;border:1px solid #e5e7eb;border-radius:12px;padding:20px;box-shadow:0 1px 3px rgba(0,0,0,.06)">
        <h3 style="margin:0 0 6px;font-size:15px;color:#111827;font-weight:600">
          Areas to Improve
          <span style="font-size:12px;font-weight:400;color:#6b7280;margin-left:6px">({n_issues} rubric violation{'s' if n_issues!=1 else ''} found)</span>
        </h3>
        <p style="margin:0 0 14px;font-size:12px;color:#9ca3af">
          Each card shows the exact rubric rule violated (dark banner), what went wrong, why it matters, and how to fix it.
        </p>
        {improve_cards}
      </div>

      <div style="margin-top:16px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:14px 20px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">
        <div style="display:flex;gap:20px;flex-wrap:wrap">
          <div style="text-align:center">
            <div style="font-size:10px;color:#9ca3af;text-transform:uppercase;letter-spacing:.05em;margin-bottom:2px">LLM Inference</div>
            <div style="font-size:18px;font-weight:800;color:#1e293b">{llm_elapsed:.1f}s</div>
          </div>
          <div style="width:1px;background:#e2e8f0"></div>
          <div style="text-align:center">
            <div style="font-size:10px;color:#9ca3af;text-transform:uppercase;letter-spacing:.05em;margin-bottom:2px">Total Time</div>
            <div style="font-size:18px;font-weight:800;color:#1e293b">{total_elapsed:.1f}s</div>
          </div>
          <div style="width:1px;background:#e2e8f0"></div>
          <div style="text-align:center">
            <div style="font-size:10px;color:#9ca3af;text-transform:uppercase;letter-spacing:.05em;margin-bottom:2px">Model</div>
            <div style="font-size:13px;font-weight:600;color:#475569">Qwen2.5-Coder via Ollama</div>
          </div>
        </div>
        <div style="font-size:11px;color:#9ca3af">AI Assignment Grader</div>
      </div>
    </div>"""


# ─────────────────────────────────────────────────────────────────────────────
# Status helpers
# ─────────────────────────────────────────────────────────────────────────────


def _step(icon, label, msg, elapsed_str=""):
    badge = (
        f"<span style='float:right;color:#94a3b8;font-size:11px'>⏱ {elapsed_str}</span>"
        if elapsed_str
        else ""
    )
    return (
        f"<div style='padding:8px 12px;margin-bottom:6px;background:#f8fafc;"
        f"border-radius:8px;border-left:3px solid #94a3b8;font-size:13px;overflow:hidden'>"
        f"{badge}<b>{icon} {label}:</b> {msg}</div>"
    )


def _timer_html(seconds):
    mins, secs = int(seconds) // 60, int(seconds) % 60
    time_str = f"{mins}m {secs:02d}s" if mins > 0 else f"{secs}s"
    pulse_w = int((seconds % 10) / 10 * 100)
    return f"""
    <div style="background:#0f172a;border-radius:10px;padding:14px 18px;margin-top:4px">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
        <span style="color:#94a3b8;font-size:12px;font-weight:600">⚙️ Evaluating with Qwen2.5-Coder</span>
        <span style="color:#f59e0b;font-size:16px;font-weight:800;font-variant-numeric:tabular-nums">⏱ {time_str}</span>
      </div>
      <div style="background:#1e3a5f;border-radius:99px;height:5px">
        <div style="background:#f59e0b;width:{pulse_w}%;height:5px;border-radius:99px;transition:width 0.9s ease"></div>
      </div>
      <div style="color:#475569;font-size:11px;margin-top:6px">LLM inference in progress — typically 30–90 seconds on CPU</div>
    </div>"""


def _done_status(llm_elapsed, total_elapsed):
    return f"""
    <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;padding:12px 16px;margin-top:4px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">
      <span style="color:#15803d;font-size:13px;font-weight:600">✅ Grading complete</span>
      <div style="display:flex;gap:16px">
        <div style="text-align:center">
          <div style="font-size:10px;color:#6b7280;text-transform:uppercase">LLM inference</div>
          <div style="font-size:15px;font-weight:800;color:#1e293b">{llm_elapsed:.1f}s</div>
        </div>
        <div style="width:1px;background:#d1fae5"></div>
        <div style="text-align:center">
          <div style="font-size:10px;color:#6b7280;text-transform:uppercase">Total time</div>
          <div style="font-size:15px;font-weight:800;color:#1e293b">{total_elapsed:.1f}s</div>
        </div>
      </div>
    </div>"""


# ─────────────────────────────────────────────────────────────────────────────
# Main grading function
# ─────────────────────────────────────────────────────────────────────────────


def grade_assignment(question_pdf, rubric_txt, notebook_ipynb, run_code):
    overall_start = time.time()
    step_log = ""

    def _elapsed():
        return f"{time.time() - overall_start:.1f}s"

    try:
        if question_pdf is None:
            yield None, "<p style='color:red'>❌ Please upload the assignment PDF.</p>", ""
            return
        if rubric_txt is None:
            yield None, "<p style='color:red'>❌ Please upload the rubric TXT file.</p>", ""
            return
        if notebook_ipynb is None:
            yield None, "<p style='color:red'>❌ Please upload the student notebook (.ipynb).</p>", ""
            return

        pdf_path = question_pdf if isinstance(question_pdf, str) else question_pdf.name
        rubric_path = rubric_txt if isinstance(rubric_txt, str) else rubric_txt.name
        nb_path = (
            notebook_ipynb if isinstance(notebook_ipynb, str) else notebook_ipynb.name
        )

        # Step 1 — Parse
        step_log = _step("⏳", "Step 1/3", "Parsing files...")
        yield None, None, step_log

        question = extract_pdf_text(pdf_path)
        with open(rubric_path, "r", encoding="utf-8") as f:
            rubric = f.read()
        if not rubric.strip():
            yield None, "<p style='color:red'>❌ Rubric file is empty.</p>", ""
            return
        code = extract_notebook_code(nb_path)

        step_log = _step(
            "✅",
            "Step 1/3 complete",
            f"PDF: {len(question):,} chars · Rubric: {len(rubric):,} chars · Code: {len(code):,} chars",
            _elapsed(),
        )
        yield None, None, step_log

        # Step 2 — Sandbox
        execution = None
        if run_code:
            step_log += _step("⏳", "Step 2/3", "Running notebook in sandbox...")
            yield None, None, step_log
            t0 = time.time()
            execution = run_code_sandbox(nb_path, timeout=30)
            icon = "✅" if execution["success"] else "⚠️"
            msg = (
                "Ran successfully"
                if execution["success"]
                else (execution.get("error") or execution.get("stderr", ""))[:80]
            )
            step_log += _step(
                icon, "Step 2/3 complete", f"{msg} ({time.time()-t0:.1f}s)", _elapsed()
            )
        else:
            step_log += _step("⏭️", "Step 2/3", "Code execution skipped.", _elapsed())
        yield None, None, step_log

        # Step 3 — LLM in background thread
        step_log += _step(
            "🧠", "Step 3/3", "Dispatching to Qwen2.5-Coder...", _elapsed()
        )
        yield None, None, step_log

        llm_result, llm_error = {}, {}
        llm_start = time.time()

        def _run():
            try:
                llm_result["data"] = call_ollama(question, rubric, code, execution)
            except Exception as e:
                llm_error["err"] = e

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        while thread.is_alive():
            yield None, None, step_log + _timer_html(time.time() - llm_start)
            time.sleep(1)

        thread.join()

        if "err" in llm_error:
            raise llm_error["err"]

        llm_elapsed = time.time() - llm_start
        total_elapsed = time.time() - overall_start

        result = llm_result["data"]
        html = render_html_report(result, llm_elapsed, total_elapsed)
        json_out = json.dumps(result, indent=2)

        yield json_out, html, _done_status(llm_elapsed, total_elapsed)

    except Exception as e:
        tb = traceback.format_exc()
        yield None, f"""
        <div style='background:#fee2e2;padding:16px;border-radius:10px;border:1px solid #fca5a5'>
          <b style='color:#991b1b'>❌ Error: {e}</b>
          <pre style='font-size:11px;margin-top:8px;color:#7f1d1d;overflow:auto'>{tb}</pre>
        </div>""", ""


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="AI Assignment Grader",
    theme=gr.themes.Soft(primary_hue="slate"),
    css="""
        .upload-box { border: 2px dashed #cbd5e1 !important; border-radius: 10px !important; }
        .grade-btn  { background: linear-gradient(135deg,#1e293b,#334155) !important;
                      color: white !important; font-weight: 700 !important;
                      font-size: 16px !important; height: 52px !important; }
        .status-box { min-height: 44px; }
    """,
) as demo:

    gr.HTML(
        """
    <div style="text-align:center;padding:24px 0 8px">
      <h1 style="font-size:28px;font-weight:800;margin:0;color:#0f172a">🎓 AI Assignment Grader</h1>
      <p style="color:#64748b;margin:6px 0 0;font-size:14px">
        Powered by Qwen2.5-Coder via Ollama · Score · Strengths · Rubric-aware Feedback
      </p>
    </div>
    """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📂 Upload Files")
            question_pdf = gr.File(
                label="📄 Assignment Question (PDF)",
                file_types=[".pdf"],
                elem_classes=["upload-box"],
            )
            rubric_txt = gr.File(
                label="📋 Grading Rubric (TXT)",
                file_types=[".txt"],
                elem_classes=["upload-box"],
            )
            notebook_ipynb = gr.File(
                label="📓 Student Notebook (IPYNB)",
                file_types=[".ipynb"],
                elem_classes=["upload-box"],
            )
            run_code = gr.Checkbox(
                label="⚙️ Run code in sandbox (30s timeout)",
                value=False,
                info="Executes the notebook and feeds output to the LLM.",
            )
            grade_btn = gr.Button(
                "🚀 Grade Assignment", variant="primary", elem_classes=["grade-btn"]
            )
            status_box = gr.HTML(value="", elem_classes=["status-box"])

            gr.Markdown(
                """
            ---
            **Output:** Score / 25 · Per-criterion breakdown · Strengths · Rubric violations with fixes

            **Tips**
            - PDF must have selectable text
            - Use explicit penalty rules in rubric for best feedback
            - First load: model download (~5 min) · Grading: 30–90 sec
            """
            )

        with gr.Column(scale=2):
            gr.Markdown("### 📊 Results")
            with gr.Tabs():
                with gr.TabItem("📋 Feedback Report"):
                    report_html = gr.HTML(
                        value="""<div style='color:#94a3b8;padding:60px;text-align:center;
                                 font-size:15px;border:2px dashed #e2e8f0;border-radius:12px;margin:8px'>
                                 Upload files and click <b>🚀 Grade Assignment</b> to begin.</div>"""
                    )
                with gr.TabItem("🔧 Raw JSON"):
                    json_output = gr.Code(
                        language="json", label="Raw grading output", lines=30
                    )

    grade_btn.click(
        fn=grade_assignment,
        inputs=[question_pdf, rubric_txt, notebook_ipynb, run_code],
        outputs=[json_output, report_html, status_box],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
