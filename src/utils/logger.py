"""Configuración centralizada de logging para todo el sistema."""

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional

_CONFIGURED_LOGGERS: dict[str, logging.Logger] = {}


def setup_logger(
    name: str,
    log_file: str = "logs/app.log",
    level: str = "INFO",
    max_bytes: int = 1_048_576,
    backup_count: int = 3,
) -> logging.Logger:
    """Crea (o reutiliza) un logger con salida a archivo rotativo y a consola.

    Args:
        name: Nombre del logger, normalmente ``__name__`` del módulo.
        log_file: Ruta del archivo de log.
        level: Nivel mínimo de severidad (DEBUG, INFO, WARNING, ERROR).
        max_bytes: Tamaño máximo del archivo antes de rotar.
        backup_count: Número de archivos rotados a conservar.

    Returns:
        Instancia de ``logging.Logger`` configurada.
    """
    if name in _CONFIGURED_LOGGERS:
        return _CONFIGURED_LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    try:
        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        # Si el archivo no puede crearse (permisos, disco), seguimos solo con consola.
        pass

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    _CONFIGURED_LOGGERS[name] = logger
    return logger


def get_logger(name: str) -> logging.Logger:
    """Obtiene un logger ya configurado o uno por defecto si aún no se configuró."""
    if name in _CONFIGURED_LOGGERS:
        return _CONFIGURED_LOGGERS[name]
    return setup_logger(name)
