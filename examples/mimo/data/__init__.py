"""MIMO data providers and task encoders."""

__all__ = ["VisionAudioQASample"]


def __getattr__(name):
    if name == "VisionAudioQASample":
        from .energon_avlm_task_encoder import VisionAudioQASample

        return VisionAudioQASample
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
