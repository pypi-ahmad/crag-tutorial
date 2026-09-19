"""Exercise the unmodified Windows launcher in an isolated, ignored fixture."""
from pathlib import Path
import argparse
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--fixture", type=Path, help="Reuse an existing ignored launcher-test fixture to check the existing-venv path.")
args = parser.parse_args()
fixture = args.fixture.resolve() if args.fixture else Path(tempfile.mkdtemp(prefix="launcher-test-", dir=ROOT / "data/cache"))
assert fixture.parent == (ROOT / "data/cache").resolve() and fixture.name.startswith("launcher-test-"), "Fixture must be inside this repo's ignored cache."
for name in ("run.cmd", "requirements.txt"):
    shutil.copy2(ROOT / name, fixture / name)
(fixture / "notebooks").mkdir(exist_ok=True)
shutil.copy2(ROOT / "notebooks/01_crag_tutorial.ipynb", fixture / "notebooks/01_crag_tutorial.ipynb")
config = fixture / "jupyter-config"
config.mkdir(exist_ok=True)
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
(config / "jupyter_server_config.json").write_text(json.dumps({"ServerApp": {"open_browser": False, "ip": "127.0.0.1", "port": port, "port_retries": 0}}), encoding="utf-8")
env = os.environ.copy()
env.update(JUPYTER_CONFIG_DIR=str(config), JUPYTER_DATA_DIR=str(fixture / "jupyter-data"), JUPYTER_RUNTIME_DIR=str(fixture / "jupyter-runtime"))
for name in ("AGNESAI_API_KEY", "HF_TOKEN", "VIRTUAL_ENV"):
    env.pop(name, None)
log_path = fixture / "launcher.log"
mode = "existing-venv" if (fixture / ".venv/Scripts/python.exe").exists() else "new-venv"
print(f"Isolated launcher fixture: {fixture.name}; mode={mode}.", flush=True)
with log_path.open("w", encoding="utf-8") as log:
    process = subprocess.Popen(["cmd.exe", "/d", "/c", "run.cmd"], cwd=fixture, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        deadline = time.monotonic() + 900
        last_report = 0
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Launcher exited {process.returncode}; inspect ignored fixture log.")
            try:
                with urlopen(f"http://127.0.0.1:{port}/tree/notebooks/01_crag_tutorial.ipynb", timeout=2) as response:
                    if response.status == 200:
                        print(f"PASS ({mode}): dependencies installed, isolated kernel registered and notebook route served HTTP 200.", flush=True)
                        break
            except (OSError, TimeoutError):
                pass
            if time.monotonic() - last_report > 30:
                print("Launcher setup still running; waiting for local notebook server.", flush=True)
                last_report = time.monotonic()
            time.sleep(2)
        else:
            raise TimeoutError("Launcher did not start within 15 minutes.")
        kernel = fixture / "jupyter-data/kernels/crag-tutorial/kernel.json"
        assert kernel.exists(), "Launcher did not register the isolated kernel."
        subprocess.run([str(fixture / ".venv/Scripts/python.exe"), "--version"], check=True)
        (fixture / f"verified-{mode}.json").write_text(json.dumps({"http_status": 200, "kernel_registered": True, "launcher_unmodified": True, "mode": mode}), encoding="utf-8")
    finally:
        # Exact PID belongs to this fixture; never terminate unrelated Jupyter processes.
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
        process.wait(timeout=30)
