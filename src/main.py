"""Punto de entrada y ensamblado del pipeline completo de análisis por frame."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from src.detection.face_detector import FaceDetector
from src.detection.hand_detector import HandDetector
from src.expert_system.fuzzy_engine import FuzzyEngine
from src.expert_system.rules import EntradasReglas, determinar_alerta
from src.features.calibracion import ESTADO_CALIBRANDO, CalibradorIndividual
from src.features.eye_analyzer import EyeAnalyzer
from src.features.head_pose import HeadPoseEstimator
from src.features.monitor_perclos import MonitorPERCLOS
from src.features.mouth_analyzer import MouthAnalyzer
from src.features.phone_usage import PhoneUsageDetector
from src.ui.alerts import AlertManager
from src.ui.overlays import VideoOverlay
from src.utils.config_loader import cargar_config
from src.utils.logger import get_logger, setup_logger
from src.utils.metrics import PerformanceMetrics, SessionStats
from src.utils.sonido import reproducir_pitido_alarma

logger = get_logger(__name__)


class Pipeline:
    """Orquesta captura → detección → features → inferencia difusa → overlay.

    Se instancia una vez por sesión de monitoreo y se invoca ``procesar_frame``
    por cada frame capturado de la webcam.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Construye todos los componentes del pipeline a partir de la config.

        Args:
            config: Diccionario de configuración (ver ``config/config.yaml``).
                Si es ``None``, se carga la configuración por defecto del proyecto.
        """
        self.config = config or cargar_config()
        umbrales = self.config["thresholds"]

        setup_logger(
            "src",
            log_file=self.config["logging"]["file"],
            level=self.config["logging"]["level"],
            max_bytes=self.config["logging"]["max_bytes"],
            backup_count=self.config["logging"]["backup_count"],
        )

        self.face_detector = FaceDetector()
        self.hand_detector = HandDetector()
        self.eye_analyzer = EyeAnalyzer(
            ear_threshold=umbrales["ear_closed"],
            blink_min_frames=umbrales["ear_blink_min_frames"],
            drowsy_closed_seconds=umbrales["drowsy_eye_closed_seconds"],
        )
        self.mouth_analyzer = MouthAnalyzer(
            mar_threshold=umbrales["mar_yawn"], yawn_min_seconds=umbrales["yawn_min_seconds"]
        )
        self.calibrador: Optional[CalibradorIndividual] = None
        calibracion_cfg = self.config.get("calibracion", {})
        if calibracion_cfg.get("habilitado", False):
            try:
                self.calibrador = CalibradorIndividual(
                    ear_threshold_fallback=umbrales["ear_closed"],
                    mar_threshold_fallback=umbrales["mar_yawn"],
                    duracion_calibracion_seg=calibracion_cfg["duracion_seg"],
                    k_desvios=calibracion_cfg["k_desvios"],
                    alpha_ema=calibracion_cfg["alpha_ema"],
                    muestras_minimas=calibracion_cfg["muestras_minimas"],
                    factor_espera_maxima=calibracion_cfg["factor_espera_maxima"],
                    ear_limites=(calibracion_cfg["ear_umbral_min"], calibracion_cfg["ear_umbral_max"]),
                    mar_limites=(calibracion_cfg["mar_umbral_min"], calibracion_cfg["mar_umbral_max"]),
                )
            except Exception:
                logger.exception("No se pudo inicializar CalibradorIndividual; se usaran umbrales fijos")
                self.calibrador = None

        self.monitor_perclos: Optional[MonitorPERCLOS] = None
        alarma_cfg = self.config.get("alarma_sonora", {})
        if alarma_cfg.get("habilitado", False):
            try:
                self.monitor_perclos = MonitorPERCLOS(
                    ventana_seg=alarma_cfg["ventana_perclos_seg"],
                    umbral_perclos=alarma_cfg["umbral_perclos"],
                    cooldown_pitido_seg=alarma_cfg["cooldown_pitido_seg"],
                    gap_maximo_seg=alarma_cfg["gap_maximo_seg"],
                )
            except Exception:
                logger.exception("No se pudo inicializar MonitorPERCLOS; la alarma sonora quedara deshabilitada")
                self.monitor_perclos = None

        self.head_pose_estimator = HeadPoseEstimator(
            yaw_deviation_deg=umbrales["head_yaw_deviation_deg"],
            pitch_deviation_deg=umbrales["head_pitch_deviation_deg"],
        )
        self.phone_detector = PhoneUsageDetector(
            distancia_umbral=umbrales["phone_hand_face_distance"],
            persistencia_min_seg=umbrales["phone_min_seconds"],
        )
        self.fuzzy_engine = FuzzyEngine()
        self.overlay = VideoOverlay()
        self.alert_manager = AlertManager(
            cooldown_seconds=self.config["session"]["alert_cooldown_seconds"]
        )

        self.metrics = PerformanceMetrics()
        self.session_stats = SessionStats(
            history_max_events=self.config["session"]["history_max_events"]
        )

        logger.info("Pipeline inicializado correctamente")

    def procesar_frame(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """Procesa un frame completo y retorna el estado + el frame anotado.

        Args:
            frame_bgr: Frame de video en formato BGR (salida de OpenCV).

        Returns:
            Diccionario con el frame anotado y todas las métricas/estado
            calculadas para ese frame (ver claves en el cuerpo del método).
        """
        ms_frame = self.metrics.tick()
        frame_salida = frame_bgr.copy()
        rostro = self.face_detector.procesar(frame_bgr)
        manos = self.hand_detector.procesar(frame_bgr)

        if rostro is None:
            self.overlay.draw_metrics_overlay(
                frame_salida,
                {"fps": f"{self.metrics.fps:.1f}", "estado": "ROSTRO NO DETECTADO"},
            )
            return {
                "frame": frame_salida,
                "rostro_detectado": False,
                "riesgo": 0.0,
                "nivel_riesgo": "BAJO",
                "tipo_alerta": "SIN_ROSTRO",
                "calibrando": False,
                "calibracion_estado": self.calibrador.estado if self.calibrador is not None else "deshabilitado",
                "perclos": None,
                "fps": self.metrics.fps,
                "ms_frame": ms_frame,
            }

        ear_izq, ear_der, ear_prom = self.eye_analyzer.calcular_ear(rostro.landmarks_px)
        mar = self.mouth_analyzer.calcular_mar(rostro.landmarks_px)

        calib_estado = None
        if self.calibrador is not None:
            try:
                calib_estado = self.calibrador.actualizar(ear_prom, mar)
                self.eye_analyzer.actualizar_umbral(calib_estado["umbral_ear"])
                self.mouth_analyzer.actualizar_umbral(calib_estado["umbral_mar"])
            except Exception:
                logger.exception("Error en CalibradorIndividual; se usaran umbrales fijos de config.yaml")
                calib_estado = None

        eye_state = self.eye_analyzer.actualizar(ear_prom)
        mouth_state = self.mouth_analyzer.actualizar(mar)

        if calib_estado is not None and calib_estado["estado"] == ESTADO_CALIBRANDO:
            self.overlay.draw_face_mesh(frame_salida, rostro.landmarks_px)
            self.overlay.draw_calibracion_overlay(
                frame_salida, calib_estado["progreso_seg"], calib_estado["progreso_total_seg"]
            )
            self.overlay.draw_metrics_overlay(
                frame_salida, {"fps": f"{self.metrics.fps:.1f}", "estado": "CALIBRANDO"}
            )
            return {
                "frame": frame_salida,
                "rostro_detectado": True,
                "riesgo": 0.0,
                "nivel_riesgo": "BAJO",
                "tipo_alerta": "CALIBRANDO",
                "calibrando": True,
                "calibracion_estado": calib_estado["estado"],
                "calibracion_progreso_seg": calib_estado["progreso_seg"],
                "calibracion_progreso_total_seg": calib_estado["progreso_total_seg"],
                "perclos": None,
                "ear_promedio": ear_prom,
                "mar": mar,
                "fps": self.metrics.fps,
                "ms_frame": ms_frame,
            }

        perclos_valor = None
        if self.monitor_perclos is not None:
            try:
                perclos_estado = self.monitor_perclos.actualizar(eye_state["ojo_cerrado"])
                perclos_valor = perclos_estado["perclos"]
                if perclos_estado["debe_sonar_pitido"]:
                    reproducir_pitido_alarma(self.config["alarma_sonora"])
                    logger.info("Pitido PERCLOS disparado: perclos=%.3f", perclos_valor)
            except Exception:
                logger.exception("Error en MonitorPERCLOS; se omite esta lectura")

        pose = self.head_pose_estimator.estimar(rostro.landmarks_px, frame_bgr.shape)
        mirada_fuera = EyeAnalyzer.calcular_mirada_fuera(rostro.landmarks_px)

        centro_rostro_norm = rostro.landmarks_norm[:, :2].mean(axis=0)
        phone_state = self.phone_detector.calcular_score(manos, centro_rostro_norm)

        somnolencia = self._calcular_somnolencia(eye_state, mouth_state)
        distraccion = self._calcular_distraccion(pose, mirada_fuera)
        celular = phone_state["score"]
        duracion_somnolencia = eye_state["duracion_cierre_seg"]
        duracion_celular = phone_state["duracion_seg"]

        riesgo = self.fuzzy_engine.inferir(somnolencia, distraccion, celular, duracion_somnolencia)

        bostezos_nivel = self._nivel_bostezos(mouth_state)
        parpadeos_nivel = self._nivel_parpadeos(eye_state)

        entradas = EntradasReglas(
            somnolencia=somnolencia,
            distraccion=distraccion,
            celular=celular,
            duracion_somnolencia=duracion_somnolencia,
            duracion_celular=duracion_celular,
            cabeza_desviada=pose.cabeza_desviada,
            mirada_fuera=mirada_fuera,
            mano_cerca_rostro=phone_state["mano_cerca"],
            bostezos_nivel=bostezos_nivel,
            parpadeos_nivel=parpadeos_nivel,
        )
        nivel_riesgo, tipo_alerta, numero_regla = determinar_alerta(entradas, riesgo)

        es_evento_nuevo = self.alert_manager.actualizar(tipo_alerta, nivel_riesgo, riesgo)
        if es_evento_nuevo:
            self.session_stats.registrar_alerta(tipo_alerta, nivel_riesgo, riesgo)
            logger.info("Alerta: %s (nivel=%s, riesgo=%.1f, regla=%d)", tipo_alerta, nivel_riesgo, riesgo, numero_regla)
        self.session_stats.acumular_tiempo_riesgo(nivel_riesgo)

        if eye_state["parpadeo_detectado"]:
            self.session_stats.registrar_parpadeo()
        if mouth_state["bostezo_detectado"]:
            self.session_stats.registrar_bostezo()

        self.overlay.draw_face_mesh(frame_salida, rostro.landmarks_px)
        self.overlay.draw_eye_status(frame_salida, ear_izq, ear_der, eye_state["ojo_cerrado"])
        self.overlay.draw_phone_detection(frame_salida, celular, phone_state["confirmado"])
        self.overlay.draw_alert_box(frame_salida, tipo_alerta, nivel_riesgo)
        self.overlay.draw_metrics_overlay(
            frame_salida,
            {
                "fps": f"{self.metrics.fps:.1f}",
                "ms/frame": f"{ms_frame:.1f}",
                "yaw/pitch": f"{pose.yaw:.0f} / {pose.pitch:.0f}",
            },
        )

        return {
            "frame": frame_salida,
            "rostro_detectado": True,
            "riesgo": riesgo,
            "nivel_riesgo": nivel_riesgo,
            "tipo_alerta": tipo_alerta,
            "calibrando": False,
            "calibracion_estado": calib_estado["estado"] if calib_estado is not None else "deshabilitado",
            "perclos": perclos_valor,
            "numero_regla": numero_regla,
            "somnolencia": somnolencia,
            "distraccion": distraccion,
            "celular": celular,
            "ear_promedio": ear_prom,
            "mar": mar,
            "yaw": pose.yaw,
            "pitch": pose.pitch,
            "cabeza_desviada": pose.cabeza_desviada,
            "mirada_fuera": mirada_fuera,
            "duracion_somnolencia": duracion_somnolencia,
            "fps": self.metrics.fps,
            "ms_frame": ms_frame,
        }

    @staticmethod
    def _calcular_somnolencia(eye_state: Dict[str, Any], mouth_state: Dict[str, Any]) -> float:
        """Combina cierre de ojos sostenido y bostezos en un score 0-100."""
        score_ojos = 0.0
        if eye_state["ojo_cerrado"]:
            score_ojos = min(100.0, 40.0 + eye_state["duracion_cierre_seg"] * 30.0)

        score_boca = 0.0
        if mouth_state["boca_abierta"]:
            score_boca = min(100.0, 30.0 + mouth_state["duracion_apertura_seg"] * 25.0)

        return float(max(score_ojos, score_boca))

    @staticmethod
    def _calcular_distraccion(pose, mirada_fuera: bool) -> float:
        """Combina la desviación de cabeza y la mirada fuera de eje en un score 0-100."""
        score_yaw = min(70.0, (abs(pose.yaw) / 45.0) * 70.0)
        score_pitch = min(30.0, (abs(pose.pitch) / 45.0) * 30.0)
        score = score_yaw + score_pitch
        if mirada_fuera:
            score = max(score, 55.0)
        return float(min(100.0, score))

    @staticmethod
    def _nivel_bostezos(mouth_state: Dict[str, Any]) -> str:
        """Clasifica el nivel de bostezo actual en baja/media/alta."""
        if mouth_state["boca_abierta"] and mouth_state["duracion_apertura_seg"] >= 1.2:
            return "alta"
        if mouth_state["boca_abierta"]:
            return "media"
        return "baja"

    @staticmethod
    def _nivel_parpadeos(eye_state: Dict[str, Any]) -> str:
        """Clasifica la frecuencia/estado de parpadeo actual en baja/media/alta."""
        if eye_state["parpadeo_detectado"]:
            return "alta"
        if eye_state["ojo_cerrado"]:
            return "media"
        return "baja"

    def cerrar(self) -> None:
        """Libera los recursos de los modelos de MediaPipe usados por el pipeline."""
        self.face_detector.cerrar()
        self.hand_detector.cerrar()
        logger.info("Pipeline cerrado y recursos liberados")


def main() -> None:
    """Punto de entrada por CLI: indica cómo lanzar el dashboard Streamlit."""
    print("Este proyecto se ejecuta con Streamlit. Use:\n")
    print("    streamlit run src/ui/dashboard.py\n")


if __name__ == "__main__":
    main()
