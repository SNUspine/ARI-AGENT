import os
import sys

MODEL_FILENAME = "c2c7.onnx"


def get_model_path():
    # PyInstaller frozen exe: look in _internal/resource/
    if getattr(sys, 'frozen', False):
        base = os.path.dirname(sys.executable)
        path = os.path.join(base, "_internal", "resource", MODEL_FILENAME)
        if os.path.exists(path):
            return path
    # Dev mode: look in parent directory (shared model)
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", MODEL_FILENAME)
    if os.path.exists(path):
        return os.path.abspath(path)
    # Fallback: resource/ subfolder
    path = os.path.join(here, "resource", MODEL_FILENAME)
    if os.path.exists(path):
        return path
    # Fallback: same directory
    path = os.path.join(here, MODEL_FILENAME)
    if os.path.exists(path):
        return path
    raise FileNotFoundError(
        f"Model file not found. Place {MODEL_FILENAME} in the parent directory."
    )
