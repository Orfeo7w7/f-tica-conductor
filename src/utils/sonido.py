"""Reproducción de la alarma sonora (pitido) de somnolencia sostenida.

Windows-only (como el resto del proyecto, ver CLAUDE.md), vía el módulo
estándar ``winsound``. El import se protege explícitamente: si por algún
motivo no está disponible, el módulo se degrada a un no-op con una
advertencia en el log en vez de romper el import de quien lo use.
"""

from __future__ import annotations

import threading

from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import winsound
    _WINSOUND_DISPONIBLE = True
except ImportError:
    winsound = None  # type: ignore[assignment]
    _WINSOUND_DISPONIBLE = False
    logger.warning("winsound no esta disponible; la alarma sonora quedara deshabilitada")


def reproducir_pitido_alarma(config_alarma: dict) -> None:
    """Dispara el patrón de pitidos de alarma en un hilo en segundo plano.

    No bloquea al llamador (``winsound.Beep`` es sincrónico y, con la
    repetición configurada, tomaría varios cientos de milisegundos — un
    costo inaceptable dentro del loop de procesamiento de frames). No debe
    lanzar excepciones hacia el llamador bajo ninguna circunstancia.

    Args:
        config_alarma: Sección ``alarma_sonora`` de ``config.yaml``, con
            las claves ``frecuencia_hz``, ``duracion_pitido_ms`` y
            ``repeticiones``.
    """
    if not _WINSOUND_DISPONIBLE:
        return

    frecuencia_hz = int(config_alarma.get("frecuencia_hz", 1000))
    duracion_ms = int(config_alarma.get("duracion_pitido_ms", 200))
    repeticiones = int(config_alarma.get("repeticiones", 3))

    hilo = threading.Thread(
        target=_reproducir_patron,
        args=(frecuencia_hz, duracion_ms, repeticiones),
        daemon=True,
    )
    hilo.start()


def _reproducir_patron(frecuencia_hz: int, duracion_ms: int, repeticiones: int) -> None:
    """Target del hilo en segundo plano; nunca debe propagar excepciones
    (una excepción dentro de un hilo daemon no crashea el proceso, pero
    tampoco se ve en ningún lado si no se loguea explícitamente acá)."""
    try:
        for _ in range(repeticiones):
            winsound.Beep(frecuencia_hz, duracion_ms)
    except Exception:
        logger.exception("Error al reproducir el pitido de alarma")
