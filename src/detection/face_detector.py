"""Detección facial y extracción de landmarks con MediaPipe Face Mesh."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import mediapipe as mp
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Índices de landmarks de MediaPipe Face Mesh (468 puntos) usados por el sistema.
OJO_IZQUIERDO = [33, 160, 158, 133, 153, 144]
OJO_DERECHO = [362, 385, 387, 263, 373, 380]
BOCA_VERTICAL_HORIZONTAL = [13, 14, 78, 308]  # (labio_sup, labio_inf, comisura_izq, comisura_der)
PUNTOS_POSE_CABEZA = [1, 152, 33, 263, 61, 291]  # nariz, mentón, ojo_izq, ojo_der, boca_izq, boca_der


@dataclass
class RostroDetectado:
    """Resultado de la detección facial en un frame."""

    landmarks_px: np.ndarray  # (468, 2) en coordenadas de píxel
    landmarks_norm: np.ndarray  # (468, 3) normalizados [0,1] + profundidad relativa
    bbox: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)

    def puntos(self, indices: List[int]) -> np.ndarray:
        """Retorna las coordenadas en píxel de un subconjunto de landmarks."""
        return self.landmarks_px[indices]


class FaceDetector:
    """Detecta un rostro y expone landmarks de ojos, boca y pose de cabeza.

    Usa MediaPipe Face Mesh (468 puntos, refinado con iris) y está pensado
    para operar en tiempo real sobre frames de webcam.
    """

    def __init__(self, max_num_faces: int = 1, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5) -> None:
        """Inicializa el modelo de Face Mesh de MediaPipe.

        Args:
            max_num_faces: Número máximo de rostros a detectar (1 = conductor).
            min_detection_confidence: Confianza mínima para detección inicial.
            min_tracking_confidence: Confianza mínima para seguimiento entre frames.
        """
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def procesar(self, frame_bgr: np.ndarray) -> Optional[RostroDetectado]:
        """Procesa un frame BGR y retorna el rostro detectado (o ``None``).

        Args:
            frame_bgr: Frame de video en formato BGR (como lo entrega OpenCV).

        Returns:
            ``RostroDetectado`` con landmarks en píxeles y normalizados, o
            ``None`` si no se detectó ningún rostro en el frame.
        """
        try:
            h, w = frame_bgr.shape[:2]
            frame_rgb = frame_bgr[:, :, ::-1]
            resultado = self._face_mesh.process(frame_rgb)

            if not resultado.multi_face_landmarks:
                return None

            landmarks = resultado.multi_face_landmarks[0].landmark
            landmarks_norm = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
            landmarks_px = np.array(
                [[p.x * w, p.y * h] for p in landmarks], dtype=np.float32
            )

            x_min, y_min = landmarks_px.min(axis=0)
            x_max, y_max = landmarks_px.max(axis=0)
            bbox = (int(max(x_min, 0)), int(max(y_min, 0)), int(min(x_max, w)), int(min(y_max, h)))

            return RostroDetectado(
                landmarks_px=landmarks_px, landmarks_norm=landmarks_norm, bbox=bbox
            )
        except Exception:
            logger.exception("Error al procesar el frame en FaceDetector")
            return None

    def cerrar(self) -> None:
        """Libera los recursos del modelo de MediaPipe."""
        self._face_mesh.close()
