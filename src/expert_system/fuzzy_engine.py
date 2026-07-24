"""Motor de inferencia difuso: calcula el riesgo numérico (0-100)."""

from __future__ import annotations

from typing import Optional

from skfuzzy import control as ctrl

from src.expert_system.membership import construir_variables
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FuzzyEngine:
    """Motor difuso Mamdani que combina 4 variables en un riesgo numérico 0-100.

    Las reglas numéricas siguen el mismo espíritu que las 12 reglas expertas
    textuales del sistema (ver ``expert_system/rules.py``), pero producen una
    salida continua de riesgo; la clasificación categórica de alerta la
    resuelve la capa de reglas explícitas en Python.
    """

    def __init__(self) -> None:
        """Construye las variables difusas, las reglas y el sistema de control."""
        variables = construir_variables()
        self.somnolencia = variables["somnolencia"]
        self.distraccion = variables["distraccion"]
        self.celular = variables["celular"]
        self.duracion = variables["duracion"]
        self.riesgo = variables["riesgo"]

        self._reglas = self._construir_reglas()
        self._sistema_control = ctrl.ControlSystem(self._reglas)
        # Nota: no se reutiliza una única instancia de ControlSystemSimulation
        # a través de frames porque skfuzzy no soporta bien resetear sus
        # entradas de forma segura; se crea una simulación liviana por frame.

    def _construir_reglas(self) -> list[ctrl.Rule]:
        s, d, c, dur, r = (
            self.somnolencia, self.distraccion, self.celular, self.duracion, self.riesgo
        )
        return [
            ctrl.Rule(s["alta"] & dur["larga"], r["critico"]),               # R1: fatiga extrema
            ctrl.Rule(s["media"], r["medio"]),                                # R2: fatiga moderada
            ctrl.Rule(c["alto"], r["alto"]),                                  # R3: uso de celular
            ctrl.Rule(d["alta"], r["medio"]),                                 # R4: distracción visual
            ctrl.Rule(s["media"] & c["medio"], r["alto"]),                    # R5: combinación peligrosa
            ctrl.Rule(s["media"] & d["media"], r["medio"]),                   # R6: fatiga progresiva
            ctrl.Rule(s["alta"] & d["alta"], r["critico"]),                   # R7: peligro inminente
            ctrl.Rule(c["alto"] & dur["larga"], r["alto"]),                   # R8: celular prolongado
            ctrl.Rule(s["baja"] & d["baja"], r["bajo"]),                      # R9: primeros signos / normal
            ctrl.Rule(s["baja"] & c["bajo"] & d["baja"], r["bajo"]),          # R10: estado normal
            ctrl.Rule(s["alta"], r["alto"]),                                  # R11: fatiga alta sostenida
            ctrl.Rule((c["alto"] | s["alta"]) & d["alta"], r["critico"]),     # R12: múltiples alertas
        ]

    def inferir(
        self, somnolencia: float, distraccion: float, celular: float, duracion: float
    ) -> float:
        """Ejecuta la inferencia difusa y retorna el riesgo estimado (0-100).

        Args:
            somnolencia: Nivel de somnolencia (0-100).
            distraccion: Nivel de distracción (0-100).
            celular: Nivel de uso de celular (0-100).
            duracion: Duración de somnolencia sostenida en segundos (0-10).

        Returns:
            Riesgo estimado en el rango 0-100. Si la inferencia falla
            (entradas fuera de rango o sin reglas activadas), retorna el
            máximo simple de las entradas como respaldo conservador.
        """
        try:
            simulacion = ctrl.ControlSystemSimulation(self._sistema_control)
            simulacion.input["somnolencia"] = float(min(max(somnolencia, 0), 100))
            simulacion.input["distraccion"] = float(min(max(distraccion, 0), 100))
            simulacion.input["celular"] = float(min(max(celular, 0), 100))
            simulacion.input["duracion"] = float(min(max(duracion, 0), 10))
            simulacion.compute()
        except Exception:
            logger.exception("Fallo en la inferencia difusa; usando respaldo por máximo")
            return float(max(somnolencia, distraccion, celular))

        if "riesgo" not in simulacion.output:
            # Ninguna de las 12 reglas se activó para esta combinación de
            # entradas (zona sin cobertura entre particiones difusas); esto
            # es esperado ocasionalmente, no un fallo del motor.
            logger.debug(
                "Ninguna regla difusa se activó (somnolencia=%.1f, distraccion=%.1f, "
                "celular=%.1f, duracion=%.1f); usando respaldo por máximo",
                somnolencia, distraccion, celular, duracion,
            )
            return float(max(somnolencia, distraccion, celular))

        return float(simulacion.output["riesgo"])
