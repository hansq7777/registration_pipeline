# Histology GUI MVP

This folder contains the first implementation-oriented scaffold for the
human-in-the-loop histology GUI.

Contents:

- `schema/histology_project_v1.sql`
  - SQLite schema for the first project database
- `DIRECTORY_AND_NAMING_CONVENTIONS_v1.md`
  - workspace layout and file naming rules
- `requirements_windows.txt`
  - Windows dependency list
- `setup_windows_env.bat`
  - create `.venv` and install Windows dependencies
- `start_gui.bat`
  - Windows launcher
- `run_timing_harness.bat`
  - Windows-side timing harness launcher that uses the same Python environment selection logic as the GUI
- `launch_gui.py`
  - explicit Python entry point used by the Windows launcher
- `timing_harness_windows.py`
  - benchmark script for cold start / warm start / cache-hit timing and small quality-speed checks
- `bootstrap_qt_env.py`
  - Windows Qt bootstrap helper that adds PySide6 DLL and plugin paths before launching the GUI
- `hitl_gui/models.py`
  - Python data models and enums
- `hitl_gui/state.py`
  - section/review state enums and event names
- `hitl_gui/db.py`
  - SQLite bootstrap and simple helpers
- `hitl_gui/windows/`
  - GUI MVP window skeletons
- `hitl_gui/app.py`
  - Qt application entry point

Status:

- schema and code are skeletons for implementation
- no full editing behavior is implemented yet
- the goal is to lock contracts before building the real UI

Windows quick start:

1. Open `cmd.exe`
2. `cd /d C:\work\registration_pipeline\histology\gui_mvp`
3. Run `setup_windows_env.bat`
4. Run `start_gui.bat`

Windows timing harness:

1. Open `cmd.exe`
2. `cd /d C:\work\registration_pipeline\histology\gui_mvp`
3. Run `run_timing_harness.bat --output-dir "C:\Users\Siqi\Desktop\REVIEW\windows_timing_run"`
