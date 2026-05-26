"""
Run logging utility.

Call start_run() at the top of each pipeline step to:
  - Create logs/<step>/<YYYYMMDD_HHMMSS>/
  - Tee all stdout (and optionally stderr) to run.log in that directory
  - Copy listed artifact paths into the run directory on exit

Usage:
    from src.run_log import start_run
    from pathlib import Path

    PARAMS_PATH = Path("models/svm_best_params.json")

    RUN_DIR = start_run("step3_optimize_svm", artifacts=[PARAMS_PATH])
"""

import atexit
import shutil
import sys
from datetime import datetime
from pathlib import Path

LOGS_ROOT = Path("logs")


class _Tee:
    """Mirrors writes to a primary stream and a timestamped log file."""

    def __init__(self, primary, log_file):
        self._primary = primary
        self._log = log_file
        self._at_line_start = True

    def write(self, data):
        self._primary.write(data)
        if not data:
            return
        # Prepend a timestamp at the start of each line in the log file only.
        parts = data.split("\n")
        for i, part in enumerate(parts):
            if i > 0:
                self._log.write("\n")
                self._at_line_start = True
            if part:
                if self._at_line_start:
                    self._log.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ")
                    self._at_line_start = False
                self._log.write(part)

    def flush(self):
        self._primary.flush()
        self._log.flush()

    def fileno(self):
        return self._primary.fileno()

    # Forward attribute lookups to the primary stream (needed by some libs).
    def __getattr__(self, name):
        return getattr(self._primary, name)


def start_run(
    step_name: str,
    artifacts: list[Path] | None = None,
    tee_stderr: bool = True,
) -> Path:
    """
    Set up a timestamped run directory and start logging.

    Args:
        step_name:  Used as the subdirectory name, e.g. "step3_optimize_svm".
        artifacts:  Paths of files to copy into the run dir on process exit.
                    Files that do not exist yet at call time are copied at exit,
                    so pass the paths you *intend* to write.
        tee_stderr: Also tee stderr to the log file (default True).

    Returns:
        Path to the run directory (e.g. logs/step3_optimize_svm/20260524_143012/).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = LOGS_ROOT / step_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "run.log"
    log_file = open(log_path, "w", encoding="utf-8", buffering=1)

    sys.stdout = _Tee(sys.__stdout__, log_file)
    if tee_stderr:
        sys.stderr = _Tee(sys.__stderr__, log_file)

    print(f"Run directory: {run_dir}")
    print(f"Log: {log_path}")
    print()

    if artifacts:
        def _copy_artifacts():
            for src in artifacts:
                src = Path(src)
                if src.exists():
                    shutil.copy2(src, run_dir / src.name)
                    print(f"[run_log] Copied artifact → {run_dir / src.name}", file=sys.__stdout__)
                else:
                    print(f"[run_log] Artifact not found, skipping: {src}", file=sys.__stdout__)
        atexit.register(_copy_artifacts)

    return run_dir