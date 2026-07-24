"""Monitor de PERCLOS (PERcentage of eyelid CLOSure) para la alarma sonora.

Complementa, sin reemplazar, las alertas instantáneas de ``rules.py`` (por
ejemplo ``FATIGA_CRITICA`` a los 4s de cierre sostenido): esas reaccionan a
un evento puntual y requieren que el conductor esté mirando la pantalla;
este módulo reacciona a una tendencia sostenida de varios minutos y dispara
un pitido audible, pensado para funcionar incluso si el conductor ya no está
atento a la interfaz. Ver la sección "Por qué PERCLOS es un módulo separado"
en ``CLAUDE.md`` para el razonamiento completo de este diseño.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)


class MonitorPERCLOS:
    """Rastrea la fracción de tiempo con ojos cerrados en una ventana móvil.

    Deliberadamente NO dispara por un cierre aislado: un cierre de pocos
    segundos representa una fracción mínima de una ventana de varios
    minutos y no puede, por sí solo, empujar el PERCLOS por encima de un
    umbral razonable. Solo un patrón sostenido/frecuente de cierres a lo
    largo de la ventana lo hace.
    """

    def __init__(
        self,
        ventana_seg: float = 180.0,
        umbral_perclos: float = 0.15,
        cooldown_pitido_seg: float = 30.0,
        gap_maximo_seg: float = 2.0,
    ) -> None:
        """Configura la ventana y los umbrales de la alarma.

        Args:
            ventana_seg: Duración de la ventana móvil sobre la que se
                calcula el PERCLOS.
            umbral_perclos: Fracción (0-1) de tiempo con ojos cerrados en
                la ventana a partir de la cual corresponde sonar la alarma.
            cooldown_pitido_seg: Mínimo entre pitidos consecutivos. Si la
                condición persiste, el pitido se repite cada este intervalo
                (no es un disparo único por episodio).
            gap_maximo_seg: Tope al tiempo transcurrido entre llamadas
                consecutivas que se acumula al cálculo. Un hueco más largo
                (rostro no detectado, fase de calibración inicial,
                monitoreo pausado y reanudado) se recorta a este valor en
                vez de atribuirse íntegramente a "cerrado" o "abierto"
                según el estado del frame de reanudación.
        """
        self._ventana_seg = ventana_seg
        self._umbral_perclos = umbral_perclos
        self._cooldown_pitido_seg = cooldown_pitido_seg
        self._gap_maximo_seg = gap_maximo_seg

        self._muestras: Deque[Tuple[float, float, bool]] = deque()
        self._tiempo_cerrado = 0.0
        self._tiempo_total = 0.0
        self._ultimo_ts: Optional[float] = None
        self._ultimo_pitido_ts = float("-inf")

    def actualizar(self, ojo_cerrado: bool) -> dict:
        """Procesa una lectura de estado de ojo (llamar una vez por frame
        con rostro detectado, fuera de la fase de calibración inicial).

        Returns:
            ``{"perclos": float (0.0-1.0), "debe_sonar_pitido": bool}``.
            ``debe_sonar_pitido`` es ``True`` cuando el PERCLOS vigente
            supera ``umbral_perclos`` y pasó al menos ``cooldown_pitido_seg``
            desde el último pitido disparado.
        """
        ahora = time.time()
        delta_t = 0.0 if self._ultimo_ts is None else min(
            ahora - self._ultimo_ts, self._gap_maximo_seg
        )
        self._ultimo_ts = ahora

        self._muestras.append((ahora, delta_t, ojo_cerrado))
        self._tiempo_total += delta_t
        if ojo_cerrado:
            self._tiempo_cerrado += delta_t

        limite = ahora - self._ventana_seg
        while self._muestras and self._muestras[0][0] < limite:
            _, delta_evictado, cerrado_evictado = self._muestras.popleft()
            self._tiempo_total -= delta_evictado
            if cerrado_evictado:
                self._tiempo_cerrado -= delta_evictado

        perclos = self._tiempo_cerrado / self._tiempo_total if self._tiempo_total > 0 else 0.0

        debe_sonar_pitido = (
            perclos >= self._umbral_perclos
            and (ahora - self._ultimo_pitido_ts) >= self._cooldown_pitido_seg
        )
        if debe_sonar_pitido:
            self._ultimo_pitido_ts = ahora

        return {"perclos": perclos, "debe_sonar_pitido": debe_sonar_pitido}
