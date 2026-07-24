"""Cálculo del Eye Aspect Ratio (EAR) y detección de parpadeos/somnolencia."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from src.detection.face_detector import OJO_DERECHO, OJO_IZQUIERDO
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Landmarks de iris (requieren refine_landmarks=True en Face Mesh).
IRIS_IZQUIERDO_CENTRO = 468
IRIS_DERECHO_CENTRO = 473


def _ear_de_ojo(puntos: np.ndarray) -> float:
    """Calcula el EAR de un ojo a partir de sus 6 puntos característicos.

    Fórmula de Soukupová & Čech: (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    """
    p1, p2, p3, p4, p5, p6 = puntos
    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal == 0:
        return 0.0
    return float((vertical_1 + vertical_2) / (2.0 * horizontal))


class EyeAnalyzer:
    """Analiza el estado de los ojos: EAR, parpadeos y duración de cierre."""

    def __init__(self, ear_threshold: float = 0.21, blink_min_frames: int = 2,
                 drowsy_closed_seconds: float = 1.0) -> None:
        """Configura los umbrales de detección de parpadeo y somnolencia.

        Args:
            ear_threshold: EAR por debajo del cual se considera el ojo cerrado.
            blink_min_frames: Frames consecutivos mínimos para contar un parpadeo.
            drowsy_closed_seconds: Segundos de ojo cerrado que indican somnolencia.
        """
        self.ear_threshold = ear_threshold
        self.blink_min_frames = blink_min_frames
        self.drowsy_closed_seconds = drowsy_closed_seconds

        self._frames_cerrado = 0
        self._inicio_cierre: Optional[float] = None
        self.blink_count = 0

    def actualizar_umbral(self, nuevo_umbral: float) -> None:
        """Reemplaza el umbral EAR de cierre vigente (usado por
        ``CalibradorIndividual`` para pasar de umbrales fijos a
        personalizados). No altera el estado de cierre/parpadeo ya
        acumulado; solo cambia qué EAR se considera "cerrado" desde el
        próximo llamado a ``actualizar()``.
        """
        self.ear_threshold = nuevo_umbral

    def calcular_ear(self, landmarks_px: np.ndarray) -> tuple[float, float, float]:
        """Calcula el EAR de ambos ojos y su promedio.

        Args:
            landmarks_px: Landmarks faciales completos en coordenadas de píxel.

        Returns:
            Tupla ``(ear_izquierdo, ear_derecho, ear_promedio)``.
        """
        ear_izq = _ear_de_ojo(landmarks_px[OJO_IZQUIERDO])
        ear_der = _ear_de_ojo(landmarks_px[OJO_DERECHO])
        return ear_izq, ear_der, (ear_izq + ear_der) / 2.0

    def actualizar(self, ear_promedio: float) -> dict:
        """Actualiza el estado interno con el EAR del frame actual.

        Args:
            ear_promedio: EAR promedio calculado para el frame actual.

        Returns:
            Diccionario con ``ojo_cerrado``, ``duracion_cierre_seg`` y
            ``parpadeo_detectado`` (True solo en el frame donde se confirma).
        """
        ojo_cerrado = ear_promedio < self.ear_threshold
        parpadeo_detectado = False

        if ojo_cerrado:
            if self._inicio_cierre is None:
                self._inicio_cierre = time.time()
            self._frames_cerrado += 1
        else:
            if self._frames_cerrado >= self.blink_min_frames:
                self.blink_count += 1
                parpadeo_detectado = True
            self._frames_cerrado = 0
            self._inicio_cierre = None

        duracion_cierre = 0.0
        if self._inicio_cierre is not None:
            duracion_cierre = time.time() - self._inicio_cierre

        return {
            "ojo_cerrado": ojo_cerrado,
            "duracion_cierre_seg": duracion_cierre,
            "parpadeo_detectado": parpadeo_detectado,
            "es_somnolencia_prolongada": duracion_cierre >= self.drowsy_closed_seconds,
        }

    @staticmethod
    def calcular_mirada_fuera(landmarks_px: np.ndarray, umbral: float = 0.32) -> bool:
        """Estima si la mirada está desviada usando la posición del iris.

        Compara la posición horizontal del centro del iris respecto a las
        comisuras de cada ojo; si el iris se aleja del centro más allá del
        umbral en ambos ojos, se considera que la mirada está fuera de eje.

        Args:
            landmarks_px: Landmarks faciales completos (requiere iris, 478 pts).
            umbral: Desviación máxima del centro (0.5 = centrado) tolerada.

        Returns:
            ``True`` si la mirada se estima desviada del centro.
        """
        if landmarks_px.shape[0] < 478:
            return False

        def _ratio_horizontal(iris_idx: int, corner_a: int, corner_b: int) -> float:
            iris_x = landmarks_px[iris_idx][0]
            x_a, x_b = landmarks_px[corner_a][0], landmarks_px[corner_b][0]
            x_min, x_max = min(x_a, x_b), max(x_a, x_b)
            ancho = x_max - x_min
            if ancho == 0:
                return 0.5
            return float((iris_x - x_min) / ancho)

        ratio_izq = _ratio_horizontal(IRIS_IZQUIERDO_CENTRO, OJO_IZQUIERDO[0], OJO_IZQUIERDO[3])
        ratio_der = _ratio_horizontal(IRIS_DERECHO_CENTRO, OJO_DERECHO[0], OJO_DERECHO[3])
        ratio_promedio = (ratio_izq + ratio_der) / 2.0

        return abs(ratio_promedio - 0.5) > umbral
