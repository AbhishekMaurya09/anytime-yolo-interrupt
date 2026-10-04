from .detection_result import AnytimeDetectionResult
from .result_manager import ResultManager
from .interrupt_manager import InterruptManager, InterruptResponse, NO_DETECTION_AVAILABLE
from .sensor_interface import SensorInterface, SensorEvent
from .anytime_detector import (
    AnytimeDetector,
    AnytimeModelAdapter,
    MockAnytimeAdapter,
    HeadSpec,
)

# GelanAnytimeAdapter (the REAL, torch-based adapter) is intentionally
# NOT imported here -- it needs torch, and this package must stay
# importable without torch (for the framework/tests/MockAnytimeAdapter
# path). Import it directly when you have torch + a checkpoint:
#     from anytime_yolo_interrupt.inference.gelan_adapter import GelanAnytimeAdapter

__all__ = [
    "AnytimeDetectionResult",
    "ResultManager",
    "InterruptManager",
    "InterruptResponse",
    "NO_DETECTION_AVAILABLE",
    "SensorInterface",
    "SensorEvent",
    "AnytimeDetector",
    "AnytimeModelAdapter",
    "MockAnytimeAdapter",
    "HeadSpec",
]
