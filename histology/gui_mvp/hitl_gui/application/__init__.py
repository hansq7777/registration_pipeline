from __future__ import annotations

__all__ = ["ExportWorker", "WorkflowService"]


def __getattr__(name: str):
    if name == "ExportWorker":
        from .export_service import ExportWorker

        return ExportWorker
    if name == "WorkflowService":
        from .workflow_service import WorkflowService

        return WorkflowService
    raise AttributeError(name)
