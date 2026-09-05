"""Run one package entrypoint; keep stdout, stderr and exit status observable."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    package_dir = Path(__file__).resolve().parent
    logs = run_dir / "logs"
    try:
        package = json.loads((run_dir / "package.json").read_text(encoding="utf-8-sig"))
        command = [
            sys.executable, "-u", str(package_dir / package["entrypoint"]),
            "--config", str(run_dir / "config.json"),
            "--run-dir", str(run_dir), *package["args"],
        ]
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        # All runtime files belong to this run; imports belong to the package.
        process = subprocess.Popen(
            command, cwd=run_dir, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
        )
        lock = threading.Lock()
        with (logs / "console.log").open("w", encoding="utf-8") as combined:
            def capture(pipe, filename, stream, label):
                with (logs / filename).open("w", encoding="utf-8") as output:
                    for line in pipe:
                        output.write(line)
                        output.flush()
                        with lock:
                            combined.write(f"[{label}] {line}")
                            combined.flush()
                            stream.write(line)
                            stream.flush()
                pipe.close()

            threads = [
                threading.Thread(target=capture, args=(process.stdout, "stdout.log", sys.stdout, "stdout")),
                threading.Thread(target=capture, args=(process.stderr, "errors.log", sys.stderr, "stderr")),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            return process.wait()
    except Exception:
        detail = traceback.format_exc()
        with (logs / "errors.log").open("a", encoding="utf-8") as output:
            output.write(detail)
        print(detail, file=sys.stderr, end="")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
