"""Detección de manos y heurísticas geométricas con MediaPipe Hands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import mediapipe as mp
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

PUNTA_DEDOS = [4, 8, 12, 16, 20]
BASE_PALMA = 0
MUNECA = 0


@dataclass
class ManoDetectada:
    """Landmarks y metadatos de una mano detectada."""

    landmarks_px: np.ndarray  # (21, 2) en píxeles
    landmarks_norm: np.ndarray  # (21, 2) normalizados [0,1]
    lado: str  # "Left" o "Right" (según MediaPipe, especular a la imagen)

    def punto(self, indice: int) -> np.ndarray:
        """Retorna la coordenada en píxel de un landmark de la mano (0-20)."""
        return self.landmarks_px[indice]


class HandDetector:
    """Detecta manos y calcula heurísticas útiles para inferir uso de celular.

    Usa MediaPipe Hands (21 puntos por mano). No reconoce el objeto "celular"
    en sí (no hay detector de objetos en el stack), sino que aproxima su uso
    mediante la proximidad mano-rostro y la forma de la mano (empuñada).
    """

    def __init__(self, max_num_hands: int = 2, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5) -> None:
        """Inicializa el modelo de MediaPipe Hands.

        Args:
            max_num_hands: Número máximo de manos a detectar simultáneamente.
            min_detection_confidence: Confianza mínima para detección inicial.
            min_tracking_confidence: Confianza mínima para seguimiento entre frames.
        """
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def procesar(self, frame_bgr: np.ndarray) -> List[ManoDetectada]:
        """Procesa un frame BGR y retorna la lista de manos detectadas.

        Args:
            frame_bgr: Frame de video en formato BGR.

        Returns:
            Lista de ``ManoDetectada`` (vacía si no se detectó ninguna mano).
        """
        try:
            h, w = frame_bgr.shape[:2]
            frame_rgb = frame_bgr[:, :, ::-1]
            resultado = self._hands.process(frame_rgb)

            manos: List[ManoDetectada] = []
            if not resultado.multi_hand_landmarks:
                return manos

            handedness_list = resultado.multi_handedness or []
            for i, hand_landmarks in enumerate(resultado.multi_hand_landmarks):
                landmarks_norm = np.array(
                    [[p.x, p.y] for p in hand_landmarks.landmark], dtype=np.float32
                )
                landmarks_px = landmarks_norm * np.array([w, h], dtype=np.float32)
                lado = "Desconocido"
                if i < len(handedness_list):
                    lado = handedness_list[i].classification[0].label
                manos.append(
                    ManoDetectada(
                        landmarks_px=landmarks_px, landmarks_norm=landmarks_norm, lado=lado
                    )
                )
            return manos
        except Exception:
            logger.exception("Error al procesar el frame en HandDetector")
            return []

    @staticmethod
    def distancia_mano_cara(mano: ManoDetectada, centro_rostro_norm: np.ndarray) -> float:
        """Distancia euclidiana normalizada entre la muñeca y el centro del rostro.

        Args:
            mano: Mano detectada.
            centro_rostro_norm: Centro del rostro en coordenadas normalizadas [0,1].

        Returns:
            Distancia normalizada (0 = superpuestos, ~1.4 = esquinas opuestas).
        """
        muneca_norm = mano.landmarks_norm[MUNECA]
        return float(np.linalg.norm(muneca_norm - centro_rostro_norm))

    @staticmethod
    def mano_empunada(mano: ManoDetectada) -> float:
        """Estima qué tan "empuñada" (cerrada) está la mano, sugiriendo sujetar un objeto.

        Calcula la distancia promedio entre la punta de los dedos y la base de
        la palma, normalizada por el tamaño de la mano; valores bajos indican
        una mano cerrada (como al sujetar un celular).

        Returns:
            Score 0-1 donde valores altos = mano cerrada/empuñada.
        """
        palma = mano.landmarks_norm[BASE_PALMA]
        puntas = mano.landmarks_norm[PUNTA_DEDOS]
        distancias = np.linalg.norm(puntas - palma, axis=1)
        dist_promedio = float(distancias.mean())

        muneca = mano.landmarks_norm[0]
        dedo_medio_base = mano.landmarks_norm[9]
        tamano_mano = float(np.linalg.norm(dedo_medio_base - muneca)) + 1e-6

        apertura_relativa = dist_promedio / tamano_mano
        # apertura_relativa alto (~2.5+) = mano abierta; bajo (~1.0) = mano cerrada
        score_empunada = np.clip(1.0 - (apertura_relativa - 1.0) / 1.5, 0.0, 1.0)
        return float(score_empunada)

    def cerrar(self) -> None:
        """Libera los recursos del modelo de MediaPipe."""
        self._hands.close()
