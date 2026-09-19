"""Read-only final checks. Report secret matches by filename, never by value."""
import ast
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import nbformat

ROOT = Path(__file__).resolve().parents[1]
paths = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, text=True).splitlines()
secrets = [os.environ.get(name, "") for name in ("AGNESAI_API_KEY", "HF_TOKEN")]
patterns = [re.compile(r"\bhf_[A-Za-z0-9]{20,}"), re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"), re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")]
hits = []
imports = set()
code_count = 0
for relative in paths:
    path = ROOT / relative
    if not path.is_file():
        continue
    text = path.read_text(encoding="utf-8", errors="replace")
    if any(secret and len(secret) > 12 and secret in text for secret in secrets) or any(pattern.search(text) for pattern in patterns):
        hits.append(relative)
    sources = [text] if path.suffix == ".py" else []
    if path.suffix == ".ipynb":
        notebook = nbformat.read(path, as_version=4)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                code_count += 1
                sources.append(cell.source)
                assert not any(output.output_type == "error" for output in cell.get("outputs", [])), relative
    for source in sources:
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imports.update(item.name.split(".")[0] for item in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
requirements = {re.split(r"[<>=!~\[]", line.strip())[0].lower().replace("_", "-") for line in (ROOT / "requirements.txt").read_text().splitlines() if line.strip() and not line.startswith("#")}
aliases = {"sklearn":"scikit-learn", "IPython":"ipython"}
third_party = imports - sys.stdlib_module_names - {"src"}
missing = sorted(module for module in third_party if aliases.get(module, module).lower().replace("_", "-") not in requirements)
assert not hits, f"Potential secrets in files: {hits}"
assert not missing, f"Missing direct requirements: {missing}"
assert (ROOT / ".env.example").read_text().splitlines() == ["AGNESAI_API_KEY", "HF_TOKEN", "AGNES_BASE_URL"]
ignored = [".env", ".venv/test", "data/cache/test", "data/qdrant/test", "__pycache__/test", "notebooks/.ipynb_checkpoints/test"]
for name in ignored:
    assert subprocess.run(["git", "check-ignore", "-q", name], cwd=ROOT).returncode == 0, name
print(json.dumps({"candidate_files_scanned":len(paths), "secret_matches":len(hits), "notebook_code_cells":code_count, "direct_imports_covered":sorted(third_party), "ignore_checks":len(ignored)}, indent=2))
print("Limit: local configured-key/pattern scan only; no external secret scanner or remote-history audit.")
