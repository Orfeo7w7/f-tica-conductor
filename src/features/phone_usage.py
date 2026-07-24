"""Heurística de detección de uso de celular a partir de manos y rostro."""

from __future__ import annotations

import time
from typing import List, Optional

import numpy as np

from src.detection.hand_detector import HandDetector, ManoDetectada
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PhoneUsageDetector:
    """Estima un score de "uso de celular" combinando proximidad y forma de mano.

    No existe un detector de objetos en el stack (solo MediaPipe Hands), por
    lo que esta es una heurística geométrica: una mano cerrada y sostenida
    cerca del rostro durante cierto tiempo se interpreta como sospecha de uso
    de celular, no como reconocimiento real del objeto.
    """

    def __init__(self, distancia_umbral: float = 0.35, persistencia_min_seg: float = 1.5) -> None:
        """Configura los umbrales de la heurística.

        Args:
            distancia_umbral: Distancia normalizada mano-rostro por debajo de
                la cual se considera "cerca".
            persistencia_min_seg: Segundos sostenidos de cercanía+mano cerrada
                necesarios para confirmar el uso de celular.
        """
        self.distancia_umbral = distancia_umbral
        self.persistencia_min_seg = persistencia_min_seg
        self._inicio_sospecha: Optional[float] = None

    def calcular_score(
        self, manos: List[ManoDetectada], centro_rostro_norm: np.ndarray
    ) -> dict:
        """Calcula el score 0-100 de uso de celular para el frame actual.

        Args:
            manos: Manos detectadas en el frame.
            centro_rostro_norm: Centro del rostro en coordenadas normalizadas.

        Returns:
            Diccionario con ``score`` (0-100), ``mano_cerca`` (bool) y
            ``confirmado`` (persistencia sostenida superada).
        """
        if not manos:
            self._inicio_sospecha = None
            return {"score": 0.0, "mano_cerca": False, "confirmado": False, "duracion_seg": 0.0}

        mejor_distancia = min(
            HandDetector.distancia_mano_cara(m, centro_rostro_norm) for m in manos
        )
        mano_mas_cercana = min(
            manos, key=lambda m: HandDetector.distancia_mano_cara(m, centro_rostro_norm)
        )
        score_empunada = HandDetector.mano_empunada(mano_mas_cercana)

        mano_cerca = mejor_distancia < self.distancia_umbral

        if mano_cerca and score_empunada > 0.45:
            if self._inicio_sospecha is None:
                self._inicio_sospecha = time.time()
        else:
            self._inicio_sospecha = None

        duracion = 0.0
        if self._inicio_sospecha is not None:
            duracion = time.time() - self._inicio_sospecha

        proximidad_score = np.clip(
            1.0 - (mejor_distancia / self.distancia_umbral), 0.0, 1.0
        )
        score_base = 0.5 * proximidad_score + 0.5 * score_empunada if mano_cerca else 0.0

        confirmado = duracion >= self.persistencia_min_seg
        factor_persistencia = np.clip(duracion / self.persistencia_min_seg, 0.0, 1.0) if mano_cerca else 0.0

        score_final = float(score_base * (0.5 + 0.5 * factor_persistencia) * 100.0)

        return {
            "score": score_final,
            "mano_cerca": mano_cerca,
            "confirmado": confirmado,
            "duracion_seg": duracion,
        }
