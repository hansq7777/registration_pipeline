from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from bootstrap_qt_env import bootstrap_qt_dll_path

    bootstrap_qt_dll_path()

    try:
        from PySide6 import QtWidgets  # noqa: F401
    except Exception as exc:
        print(f"QT_IMPORT_FAILED: {exc}")
        return 1

    try:
        from hitl_gui.windows.workflow_window import WorkflowWindow  # noqa: F401
    except Exception as exc:
        print(f"GUI_IMPORT_FAILED: {exc}")
        return 1

    print("SELF_CHECK_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
