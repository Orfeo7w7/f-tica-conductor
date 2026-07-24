"""Gestión de alertas activas con control de repetición (cooldown)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional

_COLOR_POR_NIVEL = {
    "BAJO": "#00e676",
    "MEDIO": "#ffb300",
    "ALTO": "#ff6d00",
    "CRITICO": "#ff1744",
}


@dataclass
class AlertaActiva:
    """Representa la alerta vigente en el instante actual."""

    tipo_alerta: str
    nivel_riesgo: str
    riesgo_valor: float
    timestamp: float

    @property
    def color(self) -> str:
        """Color hexadecimal asociado al nivel de riesgo de la alerta."""
        return _COLOR_POR_NIVEL.get(self.nivel_riesgo, "#00e676")


class AlertManager:
    """Controla qué alertas se consideran "nuevas" para evitar spam por frame."""

    def __init__(self, cooldown_seconds: float = 3.0) -> None:
        """Configura el tiempo mínimo entre registros repetidos de una misma alerta.

        Args:
            cooldown_seconds: Segundos que deben pasar antes de volver a
                registrar en el historial la misma combinación de alerta.
        """
        self.cooldown_seconds = cooldown_seconds
        self._ultimo_registro: Dict[str, float] = {}
        self.alerta_actual: Optional[AlertaActiva] = None

    def actualizar(self, tipo_alerta: str, nivel_riesgo: str, riesgo_valor: float) -> bool:
        """Actualiza la alerta activa y determina si debe registrarse en el historial.

        Args:
            tipo_alerta: Etiqueta categórica de la alerta detectada en el frame.
            nivel_riesgo: Nivel de riesgo asociado.
            riesgo_valor: Valor numérico de riesgo (0-100).

        Returns:
            ``True`` si esta alerta debe agregarse al historial (nueva o fuera
            del período de cooldown), ``False`` en caso contrario.
        """
        ahora = time.time()
        self.alerta_actual = AlertaActiva(
            tipo_alerta=tipo_alerta, nivel_riesgo=nivel_riesgo,
            riesgo_valor=riesgo_valor, timestamp=ahora,
        )

        if tipo_alerta == "NINGUNA":
            return False

        ultimo = self._ultimo_registro.get(tipo_alerta, 0.0)
        if ahora - ultimo >= self.cooldown_seconds:
            self._ultimo_registro[tipo_alerta] = ahora
            return True
        return False
