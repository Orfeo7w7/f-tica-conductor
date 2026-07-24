"""Carga de la configuración del sistema desde config/config.yaml."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict

import yaml

_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "config", "config.yaml"
)


@lru_cache(maxsize=1)
def cargar_config(ruta: str = _DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Lee y cachea el archivo de configuración YAML del proyecto.

    Args:
        ruta: Ruta al archivo config.yaml.

    Returns:
        Diccionario anidado con toda la configuración.
    """
    ruta_absoluta = os.path.abspath(ruta)
    with open(ruta_absoluta, "r", encoding="utf-8") as archivo:
        return yaml.safe_load(archivo)
