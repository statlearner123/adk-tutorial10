"""
conftest.py — pytest configuration for support_agent evaluation tests.

- Adds workspace root to sys.path so `support_agent` is importable.
- Forces Gemini API mode (GOOGLE_GENAI_USE_VERTEXAI=FALSE).
- GOOGLE_API_KEY is loaded from environment (set via GitHub Actions secret,
  or export it in your shell before running locally).
"""
import os
import sys
from pathlib import Path

# ── sys.path so `import support_agent` works ─────────────────────────────────
WORKSPACE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

# ── Gemini API (not Vertex) ───────────────────────────────────────────────────
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"

# ── Local dev: load from key.txt if GOOGLE_API_KEY not already set ───────────
# (key.txt is in .gitignore and never committed)
key_file = WORKSPACE_ROOT / "key.txt"
if key_file.exists() and not os.environ.get("GOOGLE_API_KEY"):
    lines = key_file.read_text().splitlines()
    if lines:
        os.environ["GOOGLE_API_KEY"] = lines[0].strip()
