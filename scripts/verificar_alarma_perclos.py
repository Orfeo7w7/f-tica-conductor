"""Verificación manual de la alarma sonora por PERCLOS (sin webcam, sin sonido real).

No existe suite de pruebas automatizada en este proyecto (ver CLAUDE.md);
este script sigue el mismo patrón ad-hoc de ``scripts/verificar_calibracion.py``.

Los escenarios 1-4 prueban la lógica de ``MonitorPERCLOS`` directamente
(no a través de ``Pipeline``) usando un reloj falso en vez de
``time.sleep()`` real: la ventana de producción es de 180 segundos, y
esperar minutos reales por cada escenario sería impracticable, además de
que dormidas reales de milisegundos son demasiado imprecisas en Windows
para probar con confianza una lógica sensible a proporciones de tiempo.
El reloj falso solo reemplaza el nombre ``time`` dentro del módulo
``src.features.monitor_perclos`` (no el módulo ``time`` real ni ningún
otro import de ``time`` en el resto del proceso), así que es un parche
acotado y se restaura al terminar cada escenario.

El escenario 5 sí corre a través de ``Pipeline`` completo (fabricando
landmarks y monkeypencheando los detectores, igual que
``verificar_calibracion.py``), porque lo que prueba es el cableado de la
integración (deshabilitado / excepción -> nunca crashea, nunca dispara el
pitido), no la matemática de PERCLOS en sí.

Ejecutar:
    venv\\Scripts\\python.exe scripts\\verificar_alarma_perclos.py
"""

from __future__ import annotations

import contextlib
import copy
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.features.monitor_perclos as monitor_perclos_module
from src.features.monitor_perclos import MonitorPERCLOS


class _RelojFalso:
    """Sustituto de ``time`` que solo expone ``.time()``, controlable a mano."""

    def __init__(self, inicio: float = 1_000_000.0) -> None:
        self._ahora = inicio

    def avanzar(self, segundos: float) -> None:
        self._ahora += segundos

    def time(self) -> float:
        return self._ahora


@contextlib.contextmanager
def _reloj_falso(inicio: float = 1_000_000.0):
    time_real = monitor_perclos_module.time
    reloj = _RelojFalso(inicio)
    monitor_perclos_module.time = reloj
    try:
        yield reloj
    finally:
        monitor_perclos_module.time = time_real


def escenario_1_cierre_aislado_no_dispara() -> bool:
    print("\n[1] Un cierre aislado de 5s (el ejemplo del propio usuario) NO debe disparar el pitido")
    with _reloj_falso():
        monitor = MonitorPERCLOS(
            ventana_seg=180.0, umbral_perclos=0.15, cooldown_pitido_seg=30.0, gap_maximo_seg=2.0
        )
        reloj = monitor_perclos_module.time
        resultado = None
        for _ in range(175):
            reloj.avanzar(1.0)
            resultado = monitor.actualizar(False)
        for _ in range(5):
            reloj.avanzar(1.0)
            resultado = monitor.actualizar(True)

    ok = not resultado["debe_sonar_pitido"] and resultado["perclos"] < 0.15
    print(f"    perclos final={resultado['perclos']:.3f} debe_sonar_pitido={resultado['debe_sonar_pitido']}")
    print("    OK" if ok else "    FALLO")
    return ok


def escenario_2_cierre_sostenido_dispara() -> bool:
    print("\n[2] Cierres sostenidos que empujan el PERCLOS por encima del umbral SI deben disparar")
    with _reloj_falso():
        monitor = MonitorPERCLOS(
            ventana_seg=180.0, umbral_perclos=0.15, cooldown_pitido_seg=30.0, gap_maximo_seg=2.0
        )
        reloj = monitor_perclos_module.time
        for _ in range(30):
            reloj.avanzar(1.0)
            monitor.actualizar(False)
        disparo = False
        resultado = None
        for _ in range(40):
            reloj.avanzar(1.0)
            resultado = monitor.actualizar(True)
            disparo = disparo or resultado["debe_sonar_pitido"]

    ok = disparo and resultado["perclos"] >= 0.15
    print(f"    perclos final={resultado['perclos']:.3f} disparo={disparo}")
    print("    OK" if ok else "    FALLO")
    return ok


def escenario_3_se_repite_mientras_persiste() -> bool:
    print("\n[3] Mientras el PERCLOS se mantenga alto, el pitido se repite cada cooldown_pitido_seg")
    with _reloj_falso():
        monitor = MonitorPERCLOS(
            ventana_seg=180.0, umbral_perclos=0.15, cooldown_pitido_seg=30.0, gap_maximo_seg=2.0
        )
        reloj = monitor_perclos_module.time
        for _ in range(30):
            reloj.avanzar(1.0)
            monitor.actualizar(False)
        disparos = []
        for i in range(120):  # 2 minutos sostenidos con ojos cerrados
            reloj.avanzar(1.0)
            r = monitor.actualizar(True)
            if r["debe_sonar_pitido"]:
                disparos.append(i)

    ok = len(disparos) >= 2
    if len(disparos) >= 2:
        separaciones = [b - a for a, b in zip(disparos, disparos[1:])]
        ok = ok and all(s >= 25 for s in separaciones)  # ~cooldown_pitido_seg, con margen
    print(f"    disparos en los indices={disparos} (deberian repetirse, no ser uno solo)")
    print("    OK" if ok else "    FALLO")
    return ok


def escenario_4_hueco_no_infla_perclos() -> bool:
    print("\n[4] Un hueco largo (sin rostro / calibrando / pausado) no debe inflar el PERCLOS")
    with _reloj_falso():
        monitor = MonitorPERCLOS(
            ventana_seg=180.0, umbral_perclos=0.15, cooldown_pitido_seg=30.0, gap_maximo_seg=2.0
        )
        reloj = monitor_perclos_module.time
        for _ in range(60):
            reloj.avanzar(1.0)
            monitor.actualizar(False)
        reloj.avanzar(90.0)  # hueco largo; sin el recorte aportaria 90s de "cerrado"
        resultado = monitor.actualizar(True)  # el frame de reanudacion "cae" en ojo cerrado

    ok = resultado["perclos"] < 0.15 and not resultado["debe_sonar_pitido"]
    print(f"    perclos tras el hueco={resultado['perclos']:.3f} (deberia seguir bajo, no ~60%)")
    print("    OK" if ok else "    FALLO")
    return ok


# --- Escenario 5: a traves del Pipeline completo -----------------------

N_LANDMARKS = 478
_RNG_BASE = np.random.default_rng(42)
_BASE_PX = _RNG_BASE.uniform(100, 500, size=(N_LANDMARKS, 2)).astype(np.float32)
_BASE_NORM = _RNG_BASE.uniform(0.2, 0.8, size=(N_LANDMARKS, 3)).astype(np.float32)

_PITIDOS_DISPARADOS: list = []


def _spy_pitido(config_alarma: dict) -> None:
    _PITIDOS_DISPARADOS.append(config_alarma)


def _instalar_spy() -> None:
    import src.main as main_module

    main_module.reproducir_pitido_alarma = _spy_pitido


def _construir_rostro(ear: float, mar: float):
    from src.detection.face_detector import OJO_DERECHO, OJO_IZQUIERDO, RostroDetectado
    from src.features.mouth_analyzer import (
        COMISURA_DERECHA, COMISURA_IZQUIERDA, LABIO_INFERIOR, LABIO_SUPERIOR,
    )

    landmarks_px = _BASE_PX.copy()

    def _fijar_ojo(indices, ear_valor: float) -> None:
        h = 40.0
        v = ear_valor * h
        p1, p2, p3, p4, p5, p6 = indices
        landmarks_px[p1] = [0.0, 0.0]
        landmarks_px[p4] = [h, 0.0]
        landmarks_px[p2] = [h * 0.3, -v / 2.0]
        landmarks_px[p6] = [h * 0.3, v / 2.0]
        landmarks_px[p3] = [h * 0.7, -v / 2.0]
        landmarks_px[p5] = [h * 0.7, v / 2.0]

    _fijar_ojo(OJO_IZQUIERDO, ear)
    _fijar_ojo(OJO_DERECHO, ear)

    w = 60.0
    vert = mar * w
    landmarks_px[COMISURA_IZQUIERDA] = [0.0, 0.0]
    landmarks_px[COMISURA_DERECHA] = [w, 0.0]
    landmarks_px[LABIO_SUPERIOR] = [w / 2.0, -vert / 2.0]
    landmarks_px[LABIO_INFERIOR] = [w / 2.0, vert / 2.0]

    return RostroDetectado(
        landmarks_px=landmarks_px.astype(np.float32),
        landmarks_norm=_BASE_NORM.copy(),
        bbox=(0, 0, 640, 480),
    )


def _pipeline_de_prueba(overrides_calibracion=None, overrides_alarma=None):
    from src.main import Pipeline
    from src.utils.config_loader import cargar_config

    config = copy.deepcopy(cargar_config())
    # Calibracion rapida para llegar enseguida al flujo normal (post-calibrando).
    config["calibracion"].update({"duracion_seg": 0.05, "muestras_minimas": 5})
    if overrides_calibracion:
        config["calibracion"].update(overrides_calibracion)
    if overrides_alarma:
        config["alarma_sonora"].update(overrides_alarma)
    return Pipeline(config=config)


def _procesar_frame(pipeline, ear: float, mar: float) -> dict:
    import numpy as _np

    rostro = _construir_rostro(ear, mar)
    pipeline.face_detector.procesar = lambda frame: rostro
    pipeline.hand_detector.procesar = lambda frame: []
    frame = _np.zeros((480, 640, 3), dtype=_np.uint8)
    return pipeline.procesar_frame(frame)


def escenario_5_fallback_nunca_falla() -> bool:
    print("\n[5] Fallback: alarma deshabilitada, y monitor que lanza excepcion -> nunca crashea, nunca dispara el pitido")
    import time as time_real

    _PITIDOS_DISPARADOS.clear()

    # 5a: alarma deshabilitada por configuracion
    pipeline = _pipeline_de_prueba(overrides_alarma={"habilitado": False})
    resultado = None
    for _ in range(10):
        resultado = _procesar_frame(pipeline, 0.05, 0.15)  # ojo "cerrado"
        time_real.sleep(0.01)
    ok_deshabilitada = (
        pipeline.monitor_perclos is None
        and resultado["perclos"] is None
        and len(_PITIDOS_DISPARADOS) == 0
    )
    print(f"    deshabilitada: monitor_perclos is None={pipeline.monitor_perclos is None}, "
          f"perclos en resultado={resultado['perclos']}, pitidos disparados={len(_PITIDOS_DISPARADOS)}")
    print("    OK" if ok_deshabilitada else "    FALLO")
    pipeline.cerrar()

    # 5b: monitor presente pero que lanza excepcion en cada actualizar()
    pipeline2 = _pipeline_de_prueba()

    def _actualizar_roto(*_args, **_kwargs):
        raise RuntimeError("fallo simulado en MonitorPERCLOS.actualizar")

    pipeline2.monitor_perclos.actualizar = _actualizar_roto
    try:
        resultado2 = _procesar_frame(pipeline2, 0.05, 0.15)
        no_crasheo = True
    except Exception as exc:  # noqa: BLE001 - justamente lo que queremos detectar
        print(f"    EXCEPCION NO MANEJADA (esto es un fallo real): {exc}")
        no_crasheo = False
        resultado2 = None

    ok_excepcion = (
        no_crasheo
        and resultado2 is not None
        and resultado2["perclos"] is None
        and len(_PITIDOS_DISPARADOS) == 0
    )
    print(f"    procesar_frame no lanzo excepcion: {no_crasheo}")
    if resultado2 is not None:
        print(f"    perclos en resultado={resultado2['perclos']}, pitidos disparados={len(_PITIDOS_DISPARADOS)}")
    print("    OK" if ok_excepcion else "    FALLO")
    pipeline2.cerrar()

    return ok_deshabilitada and ok_excepcion


def main() -> None:
    _instalar_spy()
    escenarios = [
        escenario_1_cierre_aislado_no_dispara,
        escenario_2_cierre_sostenido_dispara,
        escenario_3_se_repite_mientras_persiste,
        escenario_4_hueco_no_infla_perclos,
        escenario_5_fallback_nunca_falla,
    ]
    resultados = [(f.__name__, f()) for f in escenarios]

    print("\n" + "=" * 70)
    print("RESUMEN")
    for nombre, ok in resultados:
        print(f"  {'OK   ' if ok else 'FALLO'} - {nombre}")
    total_ok = sum(1 for _, ok in resultados if ok)
    print(f"\n{total_ok}/{len(resultados)} escenarios OK")

    if total_ok != len(resultados):
        sys.exit(1)


if __name__ == "__main__":
    main()
