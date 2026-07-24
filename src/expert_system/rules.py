"""Reglas explícitas del sistema experto: clasifican el tipo de alerta.

El motor difuso (``fuzzy_engine.py``) resuelve el riesgo numérico continuo.
Esta capa complementaria aplica las 12 reglas expertas tal como fueron
enunciadas para determinar la etiqueta categórica de alerta, ya que una
salida difusa no puede representar directamente una categoría textual como
"USO_CELULAR" o "FATIGA_CRÍTICA".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

NIVELES_RIESGO = ("BAJO", "MEDIO", "ALTO", "CRITICO")
_ORDEN_SEVERIDAD = {nivel: i for i, nivel in enumerate(NIVELES_RIESGO)}


def _nivel_cualitativo(valor: float, bajo: float = 30, alto: float = 60) -> str:
    """Clasifica un valor 0-100 en 'baja' / 'media' / 'alta'."""
    if valor < bajo:
        return "baja"
    if valor > alto:
        return "alta"
    return "media"


def nivel_riesgo_desde_valor(riesgo: float) -> str:
    """Convierte el riesgo numérico difuso (0-100) en su nivel categórico.

    BAJO 0-30, MEDIO 30-70, ALTO 70-90, CRÍTICO 90-100.
    """
    if riesgo >= 90:
        return "CRITICO"
    if riesgo >= 70:
        return "ALTO"
    if riesgo >= 30:
        return "MEDIO"
    return "BAJO"


@dataclass
class EntradasReglas:
    """Entradas crudas que consumen las 12 reglas expertas."""

    somnolencia: float          # 0-100
    distraccion: float          # 0-100
    celular: float               # 0-100
    duracion_somnolencia: float  # segundos, 0-10
    duracion_celular: float      # segundos de sospecha de celular sostenida
    cabeza_desviada: bool
    mirada_fuera: bool
    mano_cerca_rostro: bool
    bostezos_nivel: str          # "baja" | "media" | "alta"
    parpadeos_nivel: str         # "baja" | "media" | "alta"


@dataclass
class _ReglaExperta:
    """Una de las 12 reglas del sistema experto."""

    numero: int
    descripcion: str
    condicion: Callable[[EntradasReglas], bool]
    nivel_riesgo: str
    tipo_alerta: str


def _construir_reglas() -> List[_ReglaExperta]:
    """Define las 12 reglas en orden de prioridad (más severas primero)."""
    return [
        _ReglaExperta(
            numero=1, descripcion="Fatiga extrema",
            condicion=lambda e: _nivel_cualitativo(e.somnolencia) == "alta" and e.duracion_somnolencia >= 4,
            nivel_riesgo="CRITICO", tipo_alerta="FATIGA_CRITICA",
        ),
        _ReglaExperta(
            numero=7, descripcion="Fatiga con distracción",
            condicion=lambda e: _nivel_cualitativo(e.somnolencia) == "alta" and e.cabeza_desviada,
            nivel_riesgo="CRITICO", tipo_alerta="PELIGRO_INMINENTE",
        ),
        _ReglaExperta(
            numero=12, descripcion="Múltiples alertas críticas",
            condicion=lambda e: (
                (_nivel_cualitativo(e.celular) == "alta" or _nivel_cualitativo(e.somnolencia) == "alta")
                and _nivel_cualitativo(e.distraccion) == "alta"
            ),
            nivel_riesgo="CRITICO", tipo_alerta="MULTIPLES_FACTORES",
        ),
        _ReglaExperta(
            numero=8, descripcion="Uso prolongado de celular",
            condicion=lambda e: _nivel_cualitativo(e.celular) == "alta" and e.duracion_celular >= 4,
            nivel_riesgo="ALTO", tipo_alerta="CELULAR_PROLONGADO",
        ),
        _ReglaExperta(
            numero=3, descripcion="Uso de celular",
            condicion=lambda e: _nivel_cualitativo(e.celular) == "alta" and e.mano_cerca_rostro,
            nivel_riesgo="ALTO", tipo_alerta="USO_CELULAR",
        ),
        _ReglaExperta(
            numero=5, descripcion="Combinación peligrosa",
            condicion=lambda e: _nivel_cualitativo(e.somnolencia) == "media" and _nivel_cualitativo(e.celular) == "media",
            nivel_riesgo="ALTO", tipo_alerta="MULTIPLES_FACTORES",
        ),
        _ReglaExperta(
            numero=11, descripcion="Fatiga con bostezos",
            condicion=lambda e: e.bostezos_nivel == "alta" and _nivel_cualitativo(e.somnolencia) == "alta",
            nivel_riesgo="ALTO", tipo_alerta="FATIGA_CON_BOSTEZOS",
        ),
        _ReglaExperta(
            numero=4, descripcion="Distracción visual",
            condicion=lambda e: e.cabeza_desviada and e.mirada_fuera,
            nivel_riesgo="MEDIO", tipo_alerta="DISTRACCION_VISUAL",
        ),
        _ReglaExperta(
            numero=2, descripcion="Fatiga moderada",
            condicion=lambda e: _nivel_cualitativo(e.somnolencia) == "media" and e.parpadeos_nivel == "alta",
            nivel_riesgo="MEDIO", tipo_alerta="FATIGA_MODERADA",
        ),
        _ReglaExperta(
            numero=6, descripcion="Bostezos frecuentes",
            condicion=lambda e: e.bostezos_nivel == "alta" and _nivel_cualitativo(e.somnolencia) == "media",
            nivel_riesgo="MEDIO", tipo_alerta="FATIGA_PROGRESIVA",
        ),
        _ReglaExperta(
            numero=9, descripcion="Somnolencia inicial",
            condicion=lambda e: _nivel_cualitativo(e.somnolencia) == "baja" and e.parpadeos_nivel == "media",
            nivel_riesgo="BAJO", tipo_alerta="PRIMEROS_SIGNOS",
        ),
        _ReglaExperta(
            numero=10, descripcion="Estado normal",
            condicion=lambda e: (
                _nivel_cualitativo(e.somnolencia) == "baja"
                and _nivel_cualitativo(e.celular) == "baja"
                and not e.cabeza_desviada
            ),
            nivel_riesgo="BAJO", tipo_alerta="NINGUNA",
        ),
    ]


_REGLAS = _construir_reglas()


def determinar_alerta(entradas: EntradasReglas, riesgo_difuso: float) -> Tuple[str, str, int]:
    """Determina el nivel de riesgo final y el tipo de alerta.

    Combina el riesgo numérico calculado por el motor difuso con la primera
    regla experta que coincida (evaluadas en orden de severidad), quedándose
    con el nivel de riesgo más severo entre ambos como medida conservadora.

    Args:
        entradas: Variables crudas necesarias para evaluar las 12 reglas.
        riesgo_difuso: Riesgo numérico (0-100) calculado por ``FuzzyEngine``.

    Returns:
        Tupla ``(nivel_riesgo, tipo_alerta, numero_regla)``. ``numero_regla``
        es 0 si ninguna regla coincidió (estado por defecto: sin alerta).
    """
    nivel_por_fuzzy = nivel_riesgo_desde_valor(riesgo_difuso)

    for regla in _REGLAS:
        if regla.condicion(entradas):
            nivel_final = max(
                nivel_por_fuzzy, regla.nivel_riesgo, key=lambda n: _ORDEN_SEVERIDAD[n]
            )
            return nivel_final, regla.tipo_alerta, regla.numero

    return nivel_por_fuzzy, "NINGUNA", 0
