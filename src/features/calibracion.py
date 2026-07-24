"""Calibración individual adaptativa de los umbrales EAR/MAR por conductor.

Reemplaza los umbrales fijos globales de ``config.yaml`` (``ear_closed``,
``mar_yawn``) por umbrales aprendidos en tiempo real para la persona que está
frente a la cámara: forma de los ojos, uso de lentes e iluminación varían lo
suficiente entre conductores como para que un único valor fijo genere falsos
positivos/negativos. Ver la sección "Por qué el calibrador es un módulo
separado" en ``CLAUDE.md`` para el razonamiento completo de este diseño
(en particular, por qué el EMA se actualiza solo con muestras no anómalas).
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

ESTADO_CALIBRANDO = "calibrando"
ESTADO_CALIBRADO = "calibrado"
ESTADO_RECALIBRANDO = "recalibrando"


def _filtrar_outliers_iqr(muestras: np.ndarray) -> np.ndarray:
    """Descarta valores fuera de ``[Q1 - 1.5*IQR, Q3 + 1.5*IQR]``.

    Con menos de 4 muestras, o si el IQR es 0 (todas las muestras casi
    idénticas), o si el filtro dejaría menos de 2 supervivientes, se
    retornan las muestras originales sin filtrar: es preferible un umbral
    un poco ruidoso a calcular media/desvío sobre 0-1 puntos.
    """
    if muestras.size < 4:
        return muestras
    q1, q3 = np.percentile(muestras, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        return muestras
    limite_inf = q1 - 1.5 * iqr
    limite_sup = q3 + 1.5 * iqr
    filtradas = muestras[(muestras >= limite_inf) & (muestras <= limite_sup)]
    return filtradas if filtradas.size >= 2 else muestras


def _umbral_desde_media_desvio(
    media: float, desvio: float, k: float, hacia_arriba: bool, limites: Tuple[float, float]
) -> float:
    """Calcula el umbral y lo recorta a ``limites`` (cota de seguridad).

    La dirección del umbral no depende de si la variable es "EAR" o "MAR"
    por nombre, sino de qué comparación usa el analizador aguas abajo:

    - ``hacia_arriba=False`` -> ``umbral = media - k*desvio``. Para EAR:
      ``EyeAnalyzer`` marca cierre con ``ear < umbral``, así que el umbral
      debe quedar POR DEBAJO de la línea base de ojo abierto.
    - ``hacia_arriba=True`` -> ``umbral = media + k*desvio``. Para MAR:
      ``MouthAnalyzer`` marca bostezo con ``mar > umbral``, así que el
      umbral debe quedar POR ENCIMA de la línea base de boca cerrada.

    Aplicar "media - k*desvio" a MAR (como en una lectura literal de
    "usar la misma fórmula para ambas variables") pondría el umbral por
    debajo del reposo y ``mar > umbral`` sería casi siempre verdadero:
    bostezo "detectado" de forma permanente. Por eso ambas direcciones se
    resuelven acá con un único parámetro booleano en vez de duplicar esta
    función para cada variable.
    """
    umbral = media + k * desvio if hacia_arriba else media - k * desvio
    return float(np.clip(umbral, limites[0], limites[1]))


def _umbral_por_desvios(
    muestras: np.ndarray, k: float, hacia_arriba: bool, limites: Tuple[float, float]
) -> Tuple[float, float, float]:
    """Media, desvío estándar y umbral (recortado) de un lote de muestras."""
    media = float(np.mean(muestras))
    desvio = float(np.std(muestras))
    umbral = _umbral_desde_media_desvio(media, desvio, k, hacia_arriba, limites)
    return media, desvio, umbral


def _actualizar_media_desvio_ema(
    media: float, desvio: float, nuevo_valor: float, alpha: float
) -> Tuple[float, float]:
    """Un paso de media/varianza móvil ponderada (EMA) para un valor nuevo.

    La varianza se actualiza usando la desviación del nuevo valor respecto
    a la media ANTERIOR (forma estándar de EMA de varianza online), y luego
    se actualiza la media. No requiere guardar el historial de muestras.
    """
    varianza_anterior = desvio ** 2
    nueva_varianza = alpha * (nuevo_valor - media) ** 2 + (1 - alpha) * varianza_anterior
    nueva_media = alpha * nuevo_valor + (1 - alpha) * media
    return nueva_media, float(np.sqrt(nueva_varianza))


class CalibradorIndividual:
    """Aprende y mantiene umbrales EAR/MAR personalizados para un conductor.

    Ciclo de vida:

    1. ``calibrando`` (inicial, bloqueante): se acumulan muestras crudas de
       EAR/MAR sin generar alertas, durante ``duracion_calibracion_seg``
       segundos (con mínimo de ``muestras_minimas`` para poder cerrar, y un
       escape hatch a ``duracion_calibracion_seg * factor_espera_maxima``
       para no quedar atascado si la cámara detecta el rostro de forma
       intermitente). Al cerrar: outliers descartados vía IQR, umbral
       inicial = media ± k·desvío (dirección según variable, ver
       ``_umbral_desde_media_desvio``), recortado a los límites de
       seguridad configurados.
    2. ``calibrado`` (estable): el umbral se sigue afinando cuadro a cuadro
       vía media/desvío móvil ponderado (EMA), pero SOLO con muestras que
       el umbral vigente ya clasifica como normales (ojo abierto / boca
       cerrada). Esto evita que un episodio real de somnolencia sostenida
       arrastre el umbral hacia sí mismo y el sistema "aprenda" el sueño
       como la nueva normalidad — ver la nota en ``CLAUDE.md``.
    3. ``recalibrando`` (ráfaga opcional, no bloqueante): disparada
       manualmente vía ``forzar_recalibracion()``. Repite el procedimiento
       de la fase 1 sobre un buffer nuevo, pero a diferencia de
       ``calibrando`` NO bloquea alertas: el umbral vigente sigue siendo el
       último calibrado+EMA hasta que la ráfaga complete.

    Si en cualquier momento la calibración no está disponible o fallara,
    quien la usa (``Pipeline``) simplemente no debe invocarla y los
    umbrales fijos de ``config.yaml`` (pasados como fallback al construir
    ``EyeAnalyzer``/``MouthAnalyzer``) siguen vigentes sin cambios.
    """

    def __init__(
        self,
        ear_threshold_fallback: float,
        mar_threshold_fallback: float,
        duracion_calibracion_seg: float = 30.0,
        k_desvios: float = 2.0,
        alpha_ema: float = 0.02,
        muestras_minimas: int = 50,
        factor_espera_maxima: float = 3.0,
        ear_limites: Tuple[float, float] = (0.12, 0.28),
        mar_limites: Tuple[float, float] = (0.35, 0.85),
    ) -> None:
        """Configura la calibración. Los ``*_fallback`` son los umbrales
        fijos de ``config.yaml``: se usan como umbral vigente mientras no
        haya calibración completa (estado inicial ``calibrando``)."""
        self._ear_fallback = ear_threshold_fallback
        self._mar_fallback = mar_threshold_fallback
        self._duracion_calibracion_seg = duracion_calibracion_seg
        self._k = k_desvios
        self._alpha_ema = alpha_ema
        self._muestras_minimas = muestras_minimas
        self._factor_espera_maxima = factor_espera_maxima
        self._ear_limites = ear_limites
        self._mar_limites = mar_limites

        self._estado = ESTADO_CALIBRANDO
        self._duracion_fase_actual = duracion_calibracion_seg
        self._inicio_fase: Optional[float] = None
        self._buffer_ear: List[float] = []
        self._buffer_mar: List[float] = []

        self._umbral_ear = ear_threshold_fallback
        self._umbral_mar = mar_threshold_fallback
        # Sembrados recién al cerrar la primera fase de acumulación.
        self._media_ear_ema: Optional[float] = None
        self._desvio_ear_ema: Optional[float] = None
        self._media_mar_ema: Optional[float] = None
        self._desvio_mar_ema: Optional[float] = None

    def actualizar(self, ear_promedio: float, mar: float) -> dict:
        """Procesa un cuadro (llamar una vez por frame con rostro detectado).

        Returns:
            Diccionario con ``estado`` (``calibrando``/``calibrado``/
            ``recalibrando``), ``calibrado`` (bool, ``False`` solo durante
            la fase inicial), ``progreso_seg``/``progreso_total_seg``
            (segundos acumulados de la fase de acumulación en curso, 0 si
            no aplica) y los umbrales vigentes ``umbral_ear``/``umbral_mar``.
        """
        if self._estado in (ESTADO_CALIBRANDO, ESTADO_RECALIBRANDO):
            return self._actualizar_fase_acumulacion(ear_promedio, mar)
        return self._actualizar_fase_estable(ear_promedio, mar)

    def forzar_recalibracion(self, duracion_seg: Optional[float] = None) -> None:
        """Reinicia el buffer y repite la calibración inicial sin bloquear
        alertas: el umbral vigente sigue siendo el último calibrado+EMA
        hasta que la ráfaga complete. Idempotente si ya está en curso
        (reinicia el temporizador y el buffer). ``duracion_seg`` permite
        una ráfaga más corta/larga que la configurada; por defecto usa
        ``duracion_calibracion_seg``.
        """
        self._duracion_fase_actual = (
            duracion_seg if duracion_seg is not None else self._duracion_calibracion_seg
        )
        self._buffer_ear.clear()
        self._buffer_mar.clear()
        self._inicio_fase = None
        self._estado = ESTADO_RECALIBRANDO
        logger.info("Recalibracion solicitada (duracion=%.1fs)", self._duracion_fase_actual)

    @property
    def estado(self) -> str:
        return self._estado

    @property
    def calibrado(self) -> bool:
        return self._estado != ESTADO_CALIBRANDO

    @property
    def umbral_ear(self) -> float:
        return self._umbral_ear

    @property
    def umbral_mar(self) -> float:
        return self._umbral_mar

    def _actualizar_fase_acumulacion(self, ear_promedio: float, mar: float) -> dict:
        ahora = time.time()
        if self._inicio_fase is None:
            self._inicio_fase = ahora
        self._buffer_ear.append(ear_promedio)
        self._buffer_mar.append(mar)

        transcurrido = ahora - self._inicio_fase
        n = len(self._buffer_ear)
        duracion_objetivo = self._duracion_fase_actual

        completar_normal = transcurrido >= duracion_objetivo and n >= self._muestras_minimas
        completar_forzado = transcurrido >= duracion_objetivo * self._factor_espera_maxima
        if completar_normal or completar_forzado:
            if completar_forzado and not completar_normal:
                logger.warning(
                    "Calibracion forzada tras agotar el tiempo de espera maximo "
                    "con solo %d muestras (< %d requeridas)",
                    n, self._muestras_minimas,
                )
            self._cerrar_fase_acumulacion()

        return self._estado_dict(
            progreso_seg=min(transcurrido, duracion_objetivo),
            progreso_total_seg=duracion_objetivo,
        )

    def _cerrar_fase_acumulacion(self) -> None:
        muestras_ear = np.array(self._buffer_ear, dtype=float)
        muestras_mar = np.array(self._buffer_mar, dtype=float)

        if muestras_ear.size < 2:
            logger.warning(
                "Calibracion sin muestras suficientes (%d); se mantienen los umbrales de respaldo",
                muestras_ear.size,
            )
            media_ear, desvio_ear, umbral_ear = self._ear_fallback, 0.0, self._ear_fallback
            media_mar, desvio_mar, umbral_mar = self._mar_fallback, 0.0, self._mar_fallback
        else:
            ear_filtradas = _filtrar_outliers_iqr(muestras_ear)
            mar_filtradas = _filtrar_outliers_iqr(muestras_mar)
            media_ear, desvio_ear, umbral_ear = _umbral_por_desvios(
                ear_filtradas, self._k, hacia_arriba=False, limites=self._ear_limites
            )
            media_mar, desvio_mar, umbral_mar = _umbral_por_desvios(
                mar_filtradas, self._k, hacia_arriba=True, limites=self._mar_limites
            )

        self._umbral_ear = umbral_ear
        self._umbral_mar = umbral_mar
        self._media_ear_ema = media_ear
        self._desvio_ear_ema = desvio_ear
        self._media_mar_ema = media_mar
        self._desvio_mar_ema = desvio_mar

        self._buffer_ear.clear()
        self._buffer_mar.clear()
        self._inicio_fase = None
        self._duracion_fase_actual = self._duracion_calibracion_seg
        self._estado = ESTADO_CALIBRADO
        logger.info(
            "Calibracion completada: umbral_ear=%.4f umbral_mar=%.4f (muestras=%d)",
            umbral_ear, umbral_mar, muestras_ear.size,
        )

    def _actualizar_fase_estable(self, ear_promedio: float, mar: float) -> dict:
        # Salvaguarda anti-retroalimentacion: una muestra ya anomala segun
        # el umbral vigente (ojo cerrado / boca abierta) nunca actualiza la
        # linea base, para que un episodio real de somnolencia sostenida no
        # se "aprenda" como normal.
        if ear_promedio >= self._umbral_ear:
            self._media_ear_ema, self._desvio_ear_ema = _actualizar_media_desvio_ema(
                self._media_ear_ema, self._desvio_ear_ema, ear_promedio, self._alpha_ema
            )
            self._umbral_ear = _umbral_desde_media_desvio(
                self._media_ear_ema, self._desvio_ear_ema, self._k,
                hacia_arriba=False, limites=self._ear_limites,
            )
        if mar <= self._umbral_mar:
            self._media_mar_ema, self._desvio_mar_ema = _actualizar_media_desvio_ema(
                self._media_mar_ema, self._desvio_mar_ema, mar, self._alpha_ema
            )
            self._umbral_mar = _umbral_desde_media_desvio(
                self._media_mar_ema, self._desvio_mar_ema, self._k,
                hacia_arriba=True, limites=self._mar_limites,
            )
        return self._estado_dict(progreso_seg=0.0, progreso_total_seg=self._duracion_calibracion_seg)

    def _estado_dict(self, progreso_seg: float, progreso_total_seg: float) -> dict:
        return {
            "estado": self._estado,
            "calibrado": self.calibrado,
            "progreso_seg": progreso_seg,
            "progreso_total_seg": progreso_total_seg,
            "umbral_ear": self._umbral_ear,
            "umbral_mar": self._umbral_mar,
        }
