"""Métricas de rendimiento y estadísticas de sesión."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional


@dataclass
class EventoHistorial:
    """Un evento registrado en el historial de la sesión."""

    timestamp: float
    tipo_alerta: str
    nivel_riesgo: str
    riesgo_valor: float

    @property
    def hora_legible(self) -> str:
        """Devuelve la hora del evento en formato HH:MM:SS."""
        return time.strftime("%H:%M:%S", time.localtime(self.timestamp))


class PerformanceMetrics:
    """Mide FPS y tiempo de procesamiento por frame con una ventana móvil."""

    def __init__(self, window_size: int = 30) -> None:
        """Inicializa el medidor con el tamaño de ventana móvil dado."""
        self._frame_times: Deque[float] = deque(maxlen=window_size)
        self._last_tick: Optional[float] = None
        self.total_frames: int = 0

    def tick(self) -> float:
        """Marca el procesamiento de un frame y retorna el tiempo transcurrido (ms).

        Debe llamarse una vez por cada frame procesado.
        """
        now = time.perf_counter()
        elapsed_ms = 0.0
        if self._last_tick is not None:
            elapsed_ms = (now - self._last_tick) * 1000.0
            self._frame_times.append(elapsed_ms)
        self._last_tick = now
        self.total_frames += 1
        return elapsed_ms

    @property
    def fps(self) -> float:
        """FPS estimado a partir de la ventana móvil de tiempos de frame."""
        if not self._frame_times:
            return 0.0
        avg_ms = sum(self._frame_times) / len(self._frame_times)
        return 1000.0 / avg_ms if avg_ms > 0 else 0.0

    @property
    def ms_per_frame(self) -> float:
        """Tiempo promedio de procesamiento por frame en milisegundos."""
        if not self._frame_times:
            return 0.0
        return sum(self._frame_times) / len(self._frame_times)


class SessionStats:
    """Acumula estadísticas de una sesión de monitoreo (parpadeos, bostezos, alertas)."""

    def __init__(self, history_max_events: int = 200) -> None:
        """Inicializa contadores en cero y el historial de eventos vacío."""
        self.session_start: float = time.time()
        self.blink_count: int = 0
        self.yawn_count: int = 0
        self.alert_counts: Dict[str, int] = {}
        self.risk_time_seconds: Dict[str, float] = {
            "BAJO": 0.0,
            "MEDIO": 0.0,
            "ALTO": 0.0,
            "CRITICO": 0.0,
        }
        self.history: Deque[EventoHistorial] = deque(maxlen=history_max_events)
        self._last_update: Optional[float] = None

    def registrar_parpadeo(self) -> None:
        """Incrementa el contador de parpadeos."""
        self.blink_count += 1

    def registrar_bostezo(self) -> None:
        """Incrementa el contador de bostezos."""
        self.yawn_count += 1

    def acumular_tiempo_riesgo(self, nivel_riesgo: str) -> None:
        """Suma el tiempo transcurrido desde la última actualización al nivel dado."""
        now = time.time()
        if self._last_update is not None:
            delta = now - self._last_update
            if nivel_riesgo in self.risk_time_seconds:
                self.risk_time_seconds[nivel_riesgo] += delta
        self._last_update = now

    def registrar_alerta(self, tipo_alerta: str, nivel_riesgo: str, riesgo_valor: float) -> None:
        """Agrega un evento al historial y actualiza el conteo por tipo de alerta."""
        self.alert_counts[tipo_alerta] = self.alert_counts.get(tipo_alerta, 0) + 1
        self.history.appendleft(
            EventoHistorial(
                timestamp=time.time(),
                tipo_alerta=tipo_alerta,
                nivel_riesgo=nivel_riesgo,
                riesgo_valor=riesgo_valor,
            )
        )

    @property
    def duracion_sesion_segundos(self) -> float:
        """Segundos transcurridos desde el inicio de la sesión."""
        return time.time() - self.session_start

    def historial_reciente(self, n: int = 20) -> List[EventoHistorial]:
        """Retorna los últimos ``n`` eventos registrados (más reciente primero)."""
        return list(self.history)[:n]
