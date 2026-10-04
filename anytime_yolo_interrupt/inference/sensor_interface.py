"""
sensor_interface.py
---------------------
Generic external-sensor-to-interrupt bridge (brief section 9B).

Deliberately does not hard-code any physical sensor. It is a thin
callback: whoever owns the real LiDAR/radar/ultrasonic driver calls
`sensor_callback(value)` on every new reading; this module decides
whether that reading crosses the configured threshold and, if so,
triggers the InterruptManager.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .interrupt_manager import InterruptManager


@dataclass
class SensorEvent:
    sensor_value: float
    threshold: float
    triggered: bool


class SensorInterface:
    def __init__(
        self,
        interrupt_manager: InterruptManager,
        threshold: float,
        comparator: Callable[[float, float], bool] = lambda value, threshold: value < threshold,
    ):
        """
        comparator(value, threshold) -> bool
            Default: interrupt when value < threshold (e.g. an obstacle
            distance in meters dropping below a safe minimum). Pass a
            different comparator for sensors where "interrupt" means
            exceeding a threshold (e.g. a G-force spike).
        """
        self.interrupt_manager = interrupt_manager
        self.threshold = threshold
        self._comparator = comparator
        self.last_event: Optional[SensorEvent] = None

    def sensor_callback(self, sensor_value: float) -> SensorEvent:
        triggered = self._comparator(sensor_value, self.threshold)
        event = SensorEvent(sensor_value=sensor_value, threshold=self.threshold, triggered=triggered)
        self.last_event = event
        if triggered:
            self.interrupt_manager.trigger_interrupt()
        return event
