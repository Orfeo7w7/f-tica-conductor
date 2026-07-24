"""Funciones de pertenencia difusas para las variables del sistema experto."""

from __future__ import annotations

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


def construir_variables() -> dict:
    """Construye los antecedentes y el consecuente difusos del sistema.

    Rangos y particiones según especificación:
        - somnolencia (0-100): baja 0-30, media 20-60, alta 50-100
        - distraccion (0-100): baja 0-30, media 20-60, alta 50-100
        - celular (0-100): bajo 0-30, medio 20-60, alto 50-100
        - duracion (0-10s): corta 0-2, media 1-5, larga 4-10
        - riesgo (0-100): bajo 0-30, medio 30-70, alto 70-100 (crítico 90-100)

    Returns:
        Diccionario con las claves ``somnolencia``, ``distraccion``, ``celular``,
        ``duracion`` (antecedentes) y ``riesgo`` (consecuente).
    """
    somnolencia = ctrl.Antecedent(np.arange(0, 101, 1), "somnolencia")
    distraccion = ctrl.Antecedent(np.arange(0, 101, 1), "distraccion")
    celular = ctrl.Antecedent(np.arange(0, 101, 1), "celular")
    duracion = ctrl.Antecedent(np.arange(0, 10.1, 0.1), "duracion")
    riesgo = ctrl.Consequent(np.arange(0, 101, 1), "riesgo")

    somnolencia["baja"] = fuzz.trimf(somnolencia.universe, [0, 0, 30])
    somnolencia["media"] = fuzz.trimf(somnolencia.universe, [20, 40, 60])
    somnolencia["alta"] = fuzz.trimf(somnolencia.universe, [50, 100, 100])

    distraccion["baja"] = fuzz.trimf(distraccion.universe, [0, 0, 30])
    distraccion["media"] = fuzz.trimf(distraccion.universe, [20, 40, 60])
    distraccion["alta"] = fuzz.trimf(distraccion.universe, [50, 100, 100])

    celular["bajo"] = fuzz.trimf(celular.universe, [0, 0, 30])
    celular["medio"] = fuzz.trimf(celular.universe, [20, 40, 60])
    celular["alto"] = fuzz.trimf(celular.universe, [50, 100, 100])

    duracion["corta"] = fuzz.trimf(duracion.universe, [0, 0, 2])
    duracion["media"] = fuzz.trimf(duracion.universe, [1, 3, 5])
    duracion["larga"] = fuzz.trimf(duracion.universe, [4, 10, 10])

    riesgo["bajo"] = fuzz.trimf(riesgo.universe, [0, 0, 30])
    riesgo["medio"] = fuzz.trimf(riesgo.universe, [30, 50, 70])
    riesgo["alto"] = fuzz.trimf(riesgo.universe, [60, 80, 100])
    riesgo["critico"] = fuzz.trimf(riesgo.universe, [90, 100, 100])

    return {
        "somnolencia": somnolencia,
        "distraccion": distraccion,
        "celular": celular,
        "duracion": duracion,
        "riesgo": riesgo,
    }
