from __future__ import annotations

import os
import sys
from pathlib import Path


def bootstrap_qt_dll_path() -> None:
    root = Path(__file__).resolve().parent
    pyside_dir = root / ".venv" / "Lib" / "site-packages" / "PySide6"
    if pyside_dir.exists():
        try:
            os.add_dll_directory(str(pyside_dir))
        except (AttributeError, FileNotFoundError):
            pass
        plugins_dir = pyside_dir / "plugins" / "platforms"
        os.environ.setdefault("QT_PLUGIN_PATH", str(pyside_dir / "plugins"))
        os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(plugins_dir))
        os.environ["PATH"] = f"{pyside_dir};{plugins_dir};" + os.environ.get("PATH", "")


def main() -> int:
    bootstrap_qt_dll_path()
    from launch_gui import main as launch_main

    return launch_main()


if __name__ == "__main__":
    raise SystemExit(main())
