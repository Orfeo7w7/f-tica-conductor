"""Gestión de la captura de video desde la webcam."""

from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class VideoCapture:
    """Envoltorio sobre ``cv2.VideoCapture`` con reintentos y liberación segura."""

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        max_reintentos: int = 3,
    ) -> None:
        """Abre la cámara indicada, reintentando si falla la apertura inicial.

        Args:
            camera_index: Índice del dispositivo de cámara (0 = webcam por defecto).
            width: Ancho deseado del frame capturado.
            height: Alto deseado del frame capturado.
            max_reintentos: Número de intentos de apertura antes de fallar.
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self._cap: Optional[cv2.VideoCapture] = None
        self._abrir(max_reintentos)

    def _abrir(self, max_reintentos: int) -> None:
        for intento in range(1, max_reintentos + 1):
            cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self._cap = cap
                logger.info(
                    "Cámara %s abierta correctamente (intento %d/%d)",
                    self.camera_index,
                    intento,
                    max_reintentos,
                )
                return
            logger.warning(
                "No se pudo abrir la cámara %s (intento %d/%d)",
                self.camera_index,
                intento,
                max_reintentos,
            )
            cap.release()
            time.sleep(0.3)
        raise RuntimeError(
            f"No fue posible abrir la cámara con índice {self.camera_index} "
            f"tras {max_reintentos} intentos."
        )

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """Lee un frame de la cámara.

        Returns:
            Tupla ``(exito, frame_bgr, timestamp)``. Si falla, ``frame_bgr`` es ``None``.
        """
        if self._cap is None:
            return False, None, time.time()
        ok, frame = self._cap.read()
        if not ok:
            logger.warning("Fallo al leer frame de la cámara %s", self.camera_index)
            return False, None, time.time()
        return True, frame, time.time()

    def is_opened(self) -> bool:
        """Indica si la cámara sigue abierta y lista para capturar."""
        return self._cap is not None and self._cap.isOpened()

    def release(self) -> None:
        """Libera el recurso de la cámara."""
        if self._cap is not None:
            self._cap.release()
            logger.info("Cámara %s liberada", self.camera_index)
            self._cap = None

    def __enter__(self) -> "VideoCapture":
        return self

    def __exit__(self, *_exc) -> None:
        self.release()
