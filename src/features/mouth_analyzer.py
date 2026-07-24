"""Cálculo del Mouth Aspect Ratio (MAR) y detección de bostezos."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Landmarks de MediaPipe Face Mesh: labio superior/inferior interno y comisuras.
LABIO_SUPERIOR = 13
LABIO_INFERIOR = 14
COMISURA_IZQUIERDA = 78
COMISURA_DERECHA = 308


def _mar(landmarks_px: np.ndarray) -> float:
    """Calcula el Mouth Aspect Ratio: apertura vertical / ancho horizontal."""
    superior = landmarks_px[LABIO_SUPERIOR]
    inferior = landmarks_px[LABIO_INFERIOR]
    izquierda = landmarks_px[COMISURA_IZQUIERDA]
    derecha = landmarks_px[COMISURA_DERECHA]

    vertical = np.linalg.norm(superior - inferior)
    horizontal = np.linalg.norm(izquierda - derecha)
    if horizontal == 0:
        return 0.0
    return float(vertical / horizontal)


class MouthAnalyzer:
    """Analiza el estado de la boca: MAR y detección de bostezos sostenidos."""

    def __init__(self, mar_threshold: float = 0.60, yawn_min_seconds: float = 1.2) -> None:
        """Configura los umbrales de detección de bostezo.

        Args:
            mar_threshold: MAR por encima del cual se considera boca abierta.
            yawn_min_seconds: Segundos sostenidos de boca abierta para contar bostezo.
        """
        self.mar_threshold = mar_threshold
        self.yawn_min_seconds = yawn_min_seconds

        self._inicio_apertura: Optional[float] = None
        self._bostezo_confirmado_en_curso = False
        self.yawn_count = 0

    def actualizar_umbral(self, nuevo_umbral: float) -> None:
        """Reemplaza el umbral MAR de bostezo vigente (usado por
        ``CalibradorIndividual`` para pasar de umbrales fijos a
        personalizados). No altera el estado de apertura/bostezo ya
        acumulado; solo cambia qué MAR se considera "boca abierta" desde el
        próximo llamado a ``actualizar()``.
        """
        self.mar_threshold = nuevo_umbral

    def calcular_mar(self, landmarks_px: np.ndarray) -> float:
        """Calcula el MAR a partir de los landmarks faciales completos."""
        return _mar(landmarks_px)

    def actualizar(self, mar: float) -> dict:
        """Actualiza el estado interno con el MAR del frame actual.

        Args:
            mar: Valor de MAR calculado para el frame actual.

        Returns:
            Diccionario con ``boca_abierta``, ``duracion_apertura_seg`` y
            ``bostezo_detectado`` (True solo en el frame donde se confirma).
        """
        boca_abierta = mar > self.mar_threshold
        bostezo_detectado = False

        if boca_abierta:
            if self._inicio_apertura is None:
                self._inicio_apertura = time.time()
            duracion = time.time() - self._inicio_apertura
            if duracion >= self.yawn_min_seconds and not self._bostezo_confirmado_en_curso:
                self.yawn_count += 1
                bostezo_detectado = True
                self._bostezo_confirmado_en_curso = True
        else:
            self._inicio_apertura = None
            self._bostezo_confirmado_en_curso = False

        duracion_apertura = 0.0
        if self._inicio_apertura is not None:
            duracion_apertura = time.time() - self._inicio_apertura

        return {
            "boca_abierta": boca_abierta,
            "duracion_apertura_seg": duracion_apertura,
            "bostezo_detectado": bostezo_detectado,
        }
