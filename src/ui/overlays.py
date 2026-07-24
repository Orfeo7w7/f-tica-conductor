"""Anotaciones dibujadas sobre el video en tiempo real (estilo HUD/cyberpunk)."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

# Paleta de colores en BGR (OpenCV), colores planos, sin degradados.
COLOR_CYAN = (255, 229, 0)
COLOR_CYAN_TENUE = (140, 110, 0)
COLOR_AMBAR = (0, 179, 255)
COLOR_NARANJA = (0, 109, 255)
COLOR_ROJO = (68, 23, 255)
COLOR_VERDE = (118, 230, 0)
COLOR_TEXTO = (236, 227, 215)
COLOR_FONDO_PANEL = (20, 17, 11)

_COLOR_POR_NIVEL = {
    "BAJO": COLOR_VERDE,
    "MEDIO": COLOR_AMBAR,
    "ALTO": COLOR_NARANJA,
    "CRITICO": COLOR_ROJO,
}

_MP_FACEMESH = mp.solutions.face_mesh


class VideoOverlay:
    """Dibuja anotaciones en el video: malla facial, estados, alertas y métricas."""

    def draw_face_mesh(self, frame: np.ndarray, landmarks_px: np.ndarray) -> np.ndarray:
        """Dibuja la malla facial en estilo wireframe cian sobre el frame.

        Args:
            frame: Frame BGR sobre el que dibujar (se modifica en su lugar).
            landmarks_px: Landmarks faciales completos en coordenadas de píxel.

        Returns:
            El mismo frame con la malla dibujada.
        """
        for conexion in _MP_FACEMESH.FACEMESH_TESSELATION:
            i, j = conexion
            p1 = tuple(landmarks_px[i].astype(int))
            p2 = tuple(landmarks_px[j].astype(int))
            cv2.line(frame, p1, p2, COLOR_CYAN_TENUE, 1, cv2.LINE_AA)

        for indice in (33, 133, 362, 263, 61, 291, 1, 152):
            punto = tuple(landmarks_px[indice].astype(int))
            cv2.circle(frame, punto, 2, COLOR_CYAN, -1, cv2.LINE_AA)

        return frame

    def draw_eye_status(
        self, frame: np.ndarray, ear_left: float, ear_right: float, ojo_cerrado: bool
    ) -> np.ndarray:
        """Muestra el estado de los ojos (EAR y abierto/cerrado) en la esquina.

        Args:
            frame: Frame BGR sobre el que dibujar.
            ear_left: EAR del ojo izquierdo.
            ear_right: EAR del ojo derecho.
            ojo_cerrado: Si el sistema considera los ojos cerrados.
        """
        color = COLOR_ROJO if ojo_cerrado else COLOR_VERDE
        estado = "CERRADO" if ojo_cerrado else "ABIERTO"
        texto = f"OJOS [{estado}]  EAR-I:{ear_left:.2f}  EAR-D:{ear_right:.2f}"
        self._texto_con_fondo(frame, texto, (16, 30), color)
        return frame

    def draw_alert_box(
        self, frame: np.ndarray, tipo_alerta: str, nivel_riesgo: str
    ) -> np.ndarray:
        """Dibuja un recuadro de alerta angular en la parte superior del frame.

        Args:
            frame: Frame BGR sobre el que dibujar.
            tipo_alerta: Etiqueta categórica de la alerta activa.
            nivel_riesgo: Nivel de riesgo (BAJO/MEDIO/ALTO/CRITICO).
        """
        h, w = frame.shape[:2]
        color = _COLOR_POR_NIVEL.get(nivel_riesgo, COLOR_VERDE)

        x0, y0, x1, y1 = 0, 0, w, 54
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), COLOR_FONDO_PANEL, -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, dst=frame)
        cv2.rectangle(frame, (x0, y0), (x1, y1 - 1), color, 2)
        cv2.line(frame, (0, y1), (w, y1), color, 3)

        etiqueta = tipo_alerta.replace("_", " ")
        cv2.putText(
            frame, f"RIESGO {nivel_riesgo} :: {etiqueta}", (18, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA,
        )
        return frame

    def draw_calibracion_overlay(
        self, frame: np.ndarray, progreso_seg: float, progreso_total_seg: float
    ) -> np.ndarray:
        """Dibuja el banner de calibración inicial con barra de progreso.

        Mismo patrón visual que ``draw_alert_box`` (banner translúcido en
        la parte superior), pero en cian en vez de un color de riesgo — no
        es una alerta, es información neutral sobre el estado del sistema.

        Args:
            frame: Frame BGR sobre el que dibujar.
            progreso_seg: Segundos transcurridos de la fase de calibración.
            progreso_total_seg: Duración total configurada de la fase.
        """
        h, w = frame.shape[:2]

        x0, y0, x1, y1 = 0, 0, w, 54
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), COLOR_FONDO_PANEL, -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, dst=frame)
        cv2.rectangle(frame, (x0, y0), (x1, y1 - 1), COLOR_CYAN, 2)
        cv2.line(frame, (0, y1), (w, y1), COLOR_CYAN, 3)

        texto = (
            f"CALIBRANDO PERFIL :: MANTENGA LOS OJOS ABIERTOS CON NORMALIDAD "
            f"({progreso_seg:.0f}/{progreso_total_seg:.0f}S)"
        )
        cv2.putText(
            frame, texto, (18, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.62, COLOR_CYAN, 2, cv2.LINE_AA,
        )

        barra_x0, barra_y0 = 18, y1 + 10
        barra_w, barra_h = w - 36, 8
        cv2.rectangle(frame, (barra_x0, barra_y0), (barra_x0 + barra_w, barra_y0 + barra_h), (40, 34, 24), -1)
        proporcion = np.clip(progreso_seg / progreso_total_seg, 0.0, 1.0) if progreso_total_seg > 0 else 0.0
        relleno = int(barra_w * proporcion)
        cv2.rectangle(frame, (barra_x0, barra_y0), (barra_x0 + relleno, barra_y0 + barra_h), COLOR_CYAN, -1)
        cv2.rectangle(frame, (barra_x0, barra_y0), (barra_x0 + barra_w, barra_y0 + barra_h), COLOR_TEXTO, 1)

        return frame

    def draw_metrics_overlay(self, frame: np.ndarray, metrics: Dict[str, float]) -> np.ndarray:
        """Muestra métricas numéricas superpuestas (FPS, ms/frame, riesgo, etc.).

        Args:
            frame: Frame BGR sobre el que dibujar.
            metrics: Diccionario de métricas con valores numéricos o texto.
        """
        h, _w = frame.shape[:2]
        y = h - 16
        for clave, valor in metrics.items():
            texto = f"{clave.upper()}: {valor}"
            self._texto_con_fondo(frame, texto, (16, y), COLOR_CYAN, alto_fila=22)
            y -= 24
        return frame

    def draw_phone_detection(self, frame: np.ndarray, phone_risk: float, confirmado: bool) -> np.ndarray:
        """Dibuja un indicador de sospecha de uso de celular.

        Args:
            frame: Frame BGR sobre el que dibujar.
            phone_risk: Score 0-100 de uso de celular.
            confirmado: Si la persistencia temporal confirmó el uso.
        """
        h, w = frame.shape[:2]
        color = COLOR_ROJO if confirmado else (COLOR_AMBAR if phone_risk > 30 else COLOR_CYAN_TENUE)
        barra_x0, barra_y0 = w - 220, 20
        barra_w, barra_h = 200, 14

        cv2.rectangle(frame, (barra_x0, barra_y0), (barra_x0 + barra_w, barra_y0 + barra_h), (40, 34, 24), -1)
        relleno = int(barra_w * np.clip(phone_risk / 100.0, 0, 1))
        cv2.rectangle(frame, (barra_x0, barra_y0), (barra_x0 + relleno, barra_y0 + barra_h), color, -1)
        cv2.rectangle(frame, (barra_x0, barra_y0), (barra_x0 + barra_w, barra_y0 + barra_h), COLOR_TEXTO, 1)

        etiqueta = "CELULAR CONFIRMADO" if confirmado else "MONITOR CELULAR"
        cv2.putText(
            frame, f"{etiqueta} {phone_risk:0.0f}%", (barra_x0, barra_y0 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )
        return frame

    @staticmethod
    def _texto_con_fondo(
        frame: np.ndarray, texto: str, posicion: Tuple[int, int], color: Tuple[int, int, int],
        alto_fila: int = 26,
    ) -> None:
        """Dibuja texto con una franja de fondo sólida detrás para legibilidad."""
        x, y = posicion
        (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x - 6, y - th - 8), (x + tw + 6, y + 6), COLOR_FONDO_PANEL, -1)
        cv2.putText(frame, texto, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
