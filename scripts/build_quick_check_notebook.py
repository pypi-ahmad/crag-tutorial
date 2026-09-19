"""
Generates notebooks/00_quick_check.ipynb.
Verifies:
1. agnes-3.0-flash ping (API key protected, never printed)
2. Load one HotpotQA row
3. Open Qdrant path and print collection count
"""

import nbformat as nbf
from pathlib import Path

def build_quick_check():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python (CRAG Tutorial)",
            "language": "python",
            "name": "crag-tutorial",
        },
        "language_info": {
            "name": "python",
            "version": "3.12",
        },
    }

    cells = []

    # Cell 1: Intro
    cells.append(nbf.v4.new_markdown_cell(r"""# 00: Environment and service check

Run this short check before the course notebooks:
1. Ping `agnes-3.0-flash` without exposing credentials.
2. Load one record from the 200-question HotpotQA slice.
3. Open `data/qdrant` and report the collection state.
"""))

    # Cell 2: Imports & Environment Check
    cells.append(nbf.v4.new_code_cell(r"""import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import check_environment

# Verify environment variables securely
env = check_environment()
print("=== Security & Environment Check ===")
for k, v in env.items():
    status = "SET (Ready)" if v else "MISSING (Required)"
    print(f"  • {k:16}: {status}")

assert env["AGNESAI_API_KEY"], "AGNESAI_API_KEY is missing from environment."
assert env["HF_TOKEN"], "HF_TOKEN is missing from environment."
print("\n[OK] All credentials verified securely in memory.")
"""))

    # Cell 3: Ping agnes-3.0-flash
    cells.append(nbf.v4.new_markdown_cell(r"""### 1. Ping `agnes-3.0-flash`
Send a small completion request to check network access and model availability. The client reads the API key from the environment and does not log it.
"""))

    cells.append(nbf.v4.new_code_cell(r"""from src.agnes_client import chat
from src.config import MODEL_NAME

print(f"Pinging {MODEL_NAME} via Agnes AI Hub...")
response = chat(
    messages=[
        {"role": "user", "content": "Respond strictly with: PONG (Agnes 3.0 Flash Ready)"}
    ],
    model=MODEL_NAME,
    max_tokens=20,
    temperature=0.0,
)

print(f"\nResponse from Agnes AI: {response}")
assert "PONG" in response.upper(), f"Unexpected ping response: {response}"
print("[OK] LLM connectivity verified successfully.")
"""))

    # Cell 4: Load one Hotpot row
    cells.append(nbf.v4.new_markdown_cell(r"""### 2. Load one HotpotQA row
Check that the Hugging Face slice cache can be read.
"""))

    cells.append(nbf.v4.new_code_cell(r"""from src.data_hotpot import build_slice

records, smoke_ids = build_slice(n=200, seed=42)
sample = records[0]

print("=== HotpotQA Record Verification ===")
print(f"Slice Total Records: {len(records)}")
print(f"Sample Question ID : {sample['id']}")
print(f"Question           : {sample['question']}")
print(f"Gold Answer        : {sample['gold_answer']}")
print(f"Supporting Titles  : {sample['gold_titles']}")
print(f"Total Paragraphs   : {len(sample['context_paragraphs'])} (Gold + Distractors)")
print("\n[OK] Dataset slice loaded and validated successfully.")
"""))

    # Cell 5: Open Qdrant path & print count
    cells.append(nbf.v4.new_markdown_cell(r"""### 3. Open Qdrant and count the collection
Open embedded Qdrant at `data/qdrant`, report the indexed point count, then close the client.
"""))

    cells.append(nbf.v4.new_code_cell(r"""from src.qdrant_store import get_qdrant_client, close_qdrant_client, DEFAULT_COLLECTION

client = get_qdrant_client()
exists = client.collection_exists(DEFAULT_COLLECTION)

print("=== Embedded Qdrant Vector Store Check ===")
print(f"Collection Name: '{DEFAULT_COLLECTION}'")
print(f"Exists on Disk : {exists}")

if exists:
    info = client.get_collection(DEFAULT_COLLECTION)
    print(f"Indexed Points : {info.points_count}")
else:
    print("Collection not indexed yet. It will be indexed during tutorial execution.")

close_qdrant_client()
print("\n[OK] Qdrant disk store opened and closed cleanly.")
"""))

    teaching = [
        ("Confirm prerequisites without exposing credentials.", "This is course infrastructure, not a paper algorithm.", "Resolve imports and report whether each credential exists.", "Missing flags or imports point to the launcher or kernel environment.", "Both required names should show as SET; their values never appear."),
        ("Separate a working local setup from a working provider connection.", "Checks the tutorial model used for evaluation and generation.", "Send one live Chat Completions ping with agnes-3.0-flash.", "Authentication, rate limits, or an unavailable service stop the request. Never paste a credential into the notebook.", "PONG confirms connectivity. It says nothing about reasoning quality."),
        ("Use real course data.", "HotpotQA replaces the paper datasets for this tutorial.", "Load the cached or newly built seed-42 slice and display one row.", "For HF errors, check HF_TOKEN, the network, and the distractor/validation selection.", "The row includes a question, answer, and annotated supporting titles."),
        ("Release the embedded database between notebooks.", "This is retrieval infrastructure, separate from the paper evaluator.", "Open data/qdrant, report the collection count, and close it.", "A lock means another process owns this path. Close that kernel instead of deleting the lock.", "No collection is expected before indexing. An existing collection reports its point count."),
    ]
    code_index = 0
    for i, cell in enumerate(cells):
        if cell.cell_type == "code":
            parts = teaching[code_index]
            cells[i - 1].source += "\n\n" + "\n\n".join(f"**{label}.** {text}" for label, text in zip(("Motivation", "Paper mapping", "Next cell", "Failure signals", "Read the output"), parts))
            code_index += 1
    nb.cells = cells
    out_path = Path("notebooks/00_quick_check.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"Successfully generated {out_path} ({len(cells)} cells).")

if __name__ == "__main__":
    build_quick_check()
