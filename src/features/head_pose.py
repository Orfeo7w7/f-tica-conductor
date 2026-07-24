"""Estimación de la orientación de la cabeza (yaw/pitch/roll) vía solvePnP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from src.detection.face_detector import PUNTOS_POSE_CABEZA
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Modelo 3D genérico de rostro (mm), correspondiente a PUNTOS_POSE_CABEZA:
# [nariz, mentón, ojo_izq_ext, ojo_der_ext, boca_izq, boca_der]
_MODELO_3D = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ],
    dtype=np.float64,
)


@dataclass
class PoseCabeza:
    """Ángulos de orientación de la cabeza en grados."""

    yaw: float
    pitch: float
    roll: float
    cabeza_desviada: bool


class HeadPoseEstimator:
    """Estima la pose de la cabeza a partir de landmarks faciales 2D."""

    def __init__(self, yaw_deviation_deg: float = 20.0, pitch_deviation_deg: float = 18.0) -> None:
        """Configura los umbrales de desviación que se consideran distracción.

        Args:
            yaw_deviation_deg: Grados de giro horizontal para marcar desvío.
            pitch_deviation_deg: Grados de inclinación vertical para marcar desvío.
        """
        self.yaw_deviation_deg = yaw_deviation_deg
        self.pitch_deviation_deg = pitch_deviation_deg

    def estimar(self, landmarks_px: np.ndarray, frame_shape: Tuple[int, int]) -> PoseCabeza:
        """Calcula yaw/pitch/roll con ``cv2.solvePnP``.

        Args:
            landmarks_px: Landmarks faciales completos en coordenadas de píxel.
            frame_shape: ``(alto, ancho)`` del frame de video.

        Returns:
            ``PoseCabeza`` con los ángulos estimados y si la cabeza está desviada.
        """
        h, w = frame_shape[:2]
        puntos_2d = landmarks_px[PUNTOS_POSE_CABEZA].astype(np.float64)

        focal_length = float(w)
        centro = (w / 2.0, h / 2.0)
        matriz_camara = np.array(
            [[focal_length, 0, centro[0]], [0, focal_length, centro[1]], [0, 0, 1]],
            dtype=np.float64,
        )
        coef_distorsion = np.zeros((4, 1))

        try:
            exito, rotacion_vec, _traslacion_vec = cv2.solvePnP(
                _MODELO_3D, puntos_2d, matriz_camara, coef_distorsion,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not exito:
                return PoseCabeza(0.0, 0.0, 0.0, False)

            matriz_rotacion, _ = cv2.Rodrigues(rotacion_vec)
            sy = np.sqrt(matriz_rotacion[0, 0] ** 2 + matriz_rotacion[1, 0] ** 2)
            singular = sy < 1e-6

            if not singular:
                pitch = np.arctan2(matriz_rotacion[2, 1], matriz_rotacion[2, 2])
                yaw = np.arctan2(-matriz_rotacion[2, 0], sy)
                roll = np.arctan2(matriz_rotacion[1, 0], matriz_rotacion[0, 0])
            else:
                pitch = np.arctan2(-matriz_rotacion[1, 2], matriz_rotacion[1, 1])
                yaw = np.arctan2(-matriz_rotacion[2, 0], sy)
                roll = 0.0

            yaw_deg = float(np.degrees(yaw))
            pitch_deg = float(np.degrees(pitch))
            roll_deg = float(np.degrees(roll))

            desviada = (
                abs(yaw_deg) > self.yaw_deviation_deg
                or abs(pitch_deg) > self.pitch_deviation_deg
            )

            return PoseCabeza(yaw=yaw_deg, pitch=pitch_deg, roll=roll_deg, cabeza_desviada=desviada)
        except Exception:
            logger.exception("Error al estimar la pose de cabeza")
            return PoseCabeza(0.0, 0.0, 0.0, False)
