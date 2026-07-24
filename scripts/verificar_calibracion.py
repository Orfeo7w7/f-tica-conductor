"""Verificación manual del módulo de calibración individual (sin webcam).

No existe suite de pruebas automatizada en este proyecto (ver CLAUDE.md);
este script sigue el mismo patrón ad-hoc ya usado para el resto del
pipeline (fabricar landmarks falsos con numpy y monkeypatchear
``face_detector.procesar``/``hand_detector.procesar``), pero cubre
explícitamente las transiciones de estado del ``CalibradorIndividual`` que
no caben en un one-liner.

Ejecutar:
    venv\\Scripts\\python.exe scripts\\verificar_calibracion.py
"""

from __future__ import annotations

import copy
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.detection.face_detector import OJO_DERECHO, OJO_IZQUIERDO, RostroDetectado
from src.features.mouth_analyzer import (
    COMISURA_DERECHA, COMISURA_IZQUIERDA, LABIO_INFERIOR, LABIO_SUPERIOR,
)
from src.main import Pipeline
from src.utils.config_loader import cargar_config

N_LANDMARKS = 478
DURACION_TEST = 0.2
MUESTRAS_MIN_TEST = 20

_RNG_BASE = np.random.default_rng(42)
_BASE_PX = _RNG_BASE.uniform(100, 500, size=(N_LANDMARKS, 2)).astype(np.float32)
_BASE_NORM = _RNG_BASE.uniform(0.2, 0.8, size=(N_LANDMARKS, 3)).astype(np.float32)


def _construir_rostro(ear: float, mar: float) -> RostroDetectado:
    """Fabrica un ``RostroDetectado`` con EAR/MAR exactos en los puntos que
    consumen ``eye_analyzer``/``mouth_analyzer``. El resto de landmarks son
    genéricos: solo necesitan existir para que ``overlays.py``/``head_pose.py``
    no fallen al indexar, su geometría real no importa para estas pruebas.
    """
    landmarks_px = _BASE_PX.copy()

    def _fijar_ojo(indices: list[int], ear_valor: float) -> None:
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


def _pipeline_de_prueba(**overrides_calibracion) -> Pipeline:
    """Pipeline con una copia independiente de la config, para poder bajar
    ``duracion_seg``/``muestras_minimas`` (u otros parámetros) sin afectar
    otros escenarios ni el ``config.yaml`` real."""
    config = copy.deepcopy(cargar_config())
    config["calibracion"].update(overrides_calibracion)
    return Pipeline(config=config)


def _procesar_frame(pipeline: Pipeline, ear: float, mar: float) -> dict:
    rostro = _construir_rostro(ear, mar)
    pipeline.face_detector.procesar = lambda frame: rostro
    pipeline.hand_detector.procesar = lambda frame: []
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    return pipeline.procesar_frame(frame)


def escenario_1_calibracion_completa() -> bool:
    print("\n[1] Calibracion inicial completa")
    pipeline = _pipeline_de_prueba(duracion_seg=DURACION_TEST, muestras_minimas=MUESTRAS_MIN_TEST)
    rng = np.random.default_rng(1)
    resultado = None
    for _ in range(50):
        ear = 0.30 + rng.normal(0, 0.01)
        mar = 0.15 + rng.normal(0, 0.01)
        resultado = _procesar_frame(pipeline, ear, mar)
        time.sleep(0.01)

    cfg_calib = pipeline.config["calibracion"]
    umbral_ear, umbral_mar = pipeline.calibrador.umbral_ear, pipeline.calibrador.umbral_mar
    ok = (
        resultado["calibracion_estado"] == "calibrado"
        and umbral_ear != pipeline.config["thresholds"]["ear_closed"]
        and cfg_calib["ear_umbral_min"] <= umbral_ear <= cfg_calib["ear_umbral_max"]
        and cfg_calib["mar_umbral_min"] <= umbral_mar <= cfg_calib["mar_umbral_max"]
    )
    print(f"    estado final={resultado['calibracion_estado']} umbral_ear={umbral_ear:.4f} umbral_mar={umbral_mar:.4f}")
    print("    OK" if ok else "    FALLO")
    pipeline.cerrar()
    return ok


def escenario_2_rechazo_outliers() -> bool:
    print("\n[2] Rechazo de outliers via IQR (parpadeos mal ubicados durante la calibracion)")
    pipeline = _pipeline_de_prueba(duracion_seg=DURACION_TEST, muestras_minimas=MUESTRAS_MIN_TEST)
    rng = np.random.default_rng(2)
    resultado = None
    for i in range(50):
        ear = 0.05 if i % 10 == 0 else 0.30 + rng.normal(0, 0.005)
        resultado = _procesar_frame(pipeline, ear, 0.15)
        time.sleep(0.01)

    umbral = pipeline.calibrador.umbral_ear
    # Sin filtrar, los outliers en 0.05 arrastrarian la media muy por debajo
    # de ~0.30 - k*0.005; con el filtro, el umbral debe quedar cerca de la
    # linea base "limpia".
    ok = resultado["calibracion_estado"] == "calibrado" and umbral > 0.20
    print(f"    umbral_ear={umbral:.4f} (deberia estar cerca de la linea base limpia, no arrastrado por los outliers)")
    print("    OK" if ok else "    FALLO")
    pipeline.cerrar()
    return ok


def escenario_3_deriva_ema() -> bool:
    print("\n[3] Deriva gradual por EMA tras calibrar (cambio lento y sostenido, ej. glare/lentes)")
    # Media de calibracion deliberadamente lejos de los limites de
    # seguridad (0.12-0.28), para que el umbral inicial tenga margen real
    # y no quede pegado al clamp desde el arranque.
    pipeline = _pipeline_de_prueba(duracion_seg=DURACION_TEST, muestras_minimas=MUESTRAS_MIN_TEST, alpha_ema=0.05)
    rng = np.random.default_rng(3)
    for _ in range(50):
        ear = 0.22 + rng.normal(0, 0.01)
        _procesar_frame(pipeline, ear, 0.15)
        time.sleep(0.01)

    umbral_inicial = pipeline.calibrador.umbral_ear
    ear_actual = 0.22
    cruzo_umbral = False
    for _ in range(300):
        # Decremento pequeño frente al alpha configurado (ventana efectiva
        # ~1/alpha = 20 muestras): el EMA deberia poder seguirle el paso
        # sin que la muestra caiga nunca por debajo del umbral vigente
        # (que es justamente lo que mantiene activa la actualizacion).
        ear_actual -= 0.0003
        if ear_actual < pipeline.calibrador.umbral_ear:
            cruzo_umbral = True
        _procesar_frame(pipeline, ear_actual, 0.15)
    umbral_final = pipeline.calibrador.umbral_ear

    ok = umbral_final < umbral_inicial - 0.01
    print(f"    umbral inicial={umbral_inicial:.4f} umbral final={umbral_final:.4f} (deberia haber bajado)")
    if cruzo_umbral:
        print("    nota: la muestra sintetica llego a cruzar el umbral vigente en algun punto "
              "(el EMA se congela ahi, es el comportamiento esperado, no un fallo)")
    print("    OK" if ok else "    FALLO")
    pipeline.cerrar()
    return ok


def escenario_4_no_retroalimentacion() -> bool:
    print("\n[4] El EMA NO debe aprender un episodio de somnolencia sostenida como normal")
    pipeline = _pipeline_de_prueba(duracion_seg=DURACION_TEST, muestras_minimas=MUESTRAS_MIN_TEST, alpha_ema=0.05)
    rng = np.random.default_rng(4)
    for _ in range(50):
        ear = 0.30 + rng.normal(0, 0.01)
        _procesar_frame(pipeline, ear, 0.15)
        time.sleep(0.01)

    umbral_antes = pipeline.calibrador.umbral_ear
    ear_somnoliento = umbral_antes - 0.05
    for _ in range(200):
        _procesar_frame(pipeline, ear_somnoliento, 0.15)
    umbral_despues = pipeline.calibrador.umbral_ear

    ok = abs(umbral_despues - umbral_antes) < 0.002
    print(f"    umbral antes={umbral_antes:.4f} umbral despues={umbral_despues:.4f} (deberian ser casi iguales)")
    print("    OK" if ok else "    FALLO")
    pipeline.cerrar()
    return ok


def escenario_5_recalibracion() -> bool:
    print("\n[5] forzar_recalibracion(): repite la calibracion sin bloquear alertas")
    pipeline = _pipeline_de_prueba(duracion_seg=DURACION_TEST, muestras_minimas=MUESTRAS_MIN_TEST)
    rng = np.random.default_rng(5)
    for _ in range(50):
        ear = 0.30 + rng.normal(0, 0.01)
        _procesar_frame(pipeline, ear, 0.15)
        time.sleep(0.01)
    calibrado_inicial = pipeline.calibrador.estado == "calibrado"

    pipeline.calibrador.forzar_recalibracion(duracion_seg=DURACION_TEST)
    estado_inmediato = pipeline.calibrador.estado

    vistos_calibrando = False
    resultado = None
    for _ in range(40):
        ear = 0.28 + rng.normal(0, 0.01)
        resultado = _procesar_frame(pipeline, ear, 0.15)
        if resultado["tipo_alerta"] == "CALIBRANDO":
            vistos_calibrando = True
        time.sleep(0.01)

    ok = (
        calibrado_inicial
        and estado_inmediato == "recalibrando"
        and not vistos_calibrando
        and resultado["calibracion_estado"] == "calibrado"
    )
    print(f"    calibrado antes de recalibrar: {calibrado_inicial}")
    print(f"    estado justo tras forzar_recalibracion(): {estado_inmediato}")
    print(f"    tipo_alerta nunca fue CALIBRANDO durante la rafaga: {not vistos_calibrando}")
    print(f"    estado final: {resultado['calibracion_estado']}")
    print("    OK" if ok else "    FALLO")
    pipeline.cerrar()
    return ok


def escenario_6_fallback() -> bool:
    print("\n[6] Fallback a umbrales fijos: calibrador deshabilitado, y calibrador que lanza excepcion")

    # 6a: deshabilitado por configuracion
    pipeline = _pipeline_de_prueba(habilitado=False)
    umbral_fijo = pipeline.config["thresholds"]["ear_closed"]
    resultado = _procesar_frame(pipeline, 0.05, 0.15)
    ok_deshabilitado = (
        pipeline.calibrador is None
        and resultado["calibracion_estado"] == "deshabilitado"
        and pipeline.eye_analyzer.ear_threshold == umbral_fijo
    )
    print(f"    deshabilitado: calibrador is None={pipeline.calibrador is None}, "
          f"umbral se mantiene fijo={pipeline.eye_analyzer.ear_threshold == umbral_fijo}")
    print("    OK" if ok_deshabilitado else "    FALLO")
    pipeline.cerrar()

    # 6b: calibrador presente pero que lanza excepcion en cada actualizar()
    pipeline2 = _pipeline_de_prueba()
    umbral_fijo_2 = pipeline2.config["thresholds"]["ear_closed"]

    def _actualizar_roto(*_args, **_kwargs):
        raise RuntimeError("fallo simulado en CalibradorIndividual.actualizar")

    pipeline2.calibrador.actualizar = _actualizar_roto
    try:
        resultado2 = _procesar_frame(pipeline2, 0.30, 0.15)
        no_crasheo = True
    except Exception as exc:  # noqa: BLE001 - justamente lo que queremos detectar
        print(f"    EXCEPCION NO MANEJADA (esto es un fallo real): {exc}")
        no_crasheo = False
        resultado2 = None

    ok_excepcion = (
        no_crasheo
        and resultado2 is not None
        and resultado2["calibracion_estado"] == "deshabilitado"
        and pipeline2.eye_analyzer.ear_threshold == umbral_fijo_2
    )
    print(f"    procesar_frame no lanzo excepcion: {no_crasheo}")
    if resultado2 is not None:
        print(f"    umbral se mantiene fijo tras el error: {pipeline2.eye_analyzer.ear_threshold == umbral_fijo_2}")
    print("    OK" if ok_excepcion else "    FALLO")
    pipeline2.cerrar()

    return ok_deshabilitado and ok_excepcion


def escenario_7_sin_rostro() -> bool:
    print("\n[7] Sin rostro detectado: no debe tocar ni corromper el estado del calibrador")
    pipeline = _pipeline_de_prueba(duracion_seg=DURACION_TEST, muestras_minimas=MUESTRAS_MIN_TEST)
    pipeline.hand_detector.procesar = lambda frame: []
    pipeline.face_detector.procesar = lambda frame: None
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    estado_antes = pipeline.calibrador.estado
    resultado = None
    for _ in range(20):
        resultado = pipeline.procesar_frame(frame)
    estado_despues = pipeline.calibrador.estado

    ok = (
        resultado["tipo_alerta"] == "SIN_ROSTRO"
        and resultado["calibrando"] is False
        and estado_despues == estado_antes
    )
    print(f"    estado antes={estado_antes} estado despues={estado_despues} (deben ser iguales)")
    print("    OK" if ok else "    FALLO")
    pipeline.cerrar()
    return ok


def main() -> None:
    escenarios = [
        escenario_1_calibracion_completa,
        escenario_2_rechazo_outliers,
        escenario_3_deriva_ema,
        escenario_4_no_retroalimentacion,
        escenario_5_recalibracion,
        escenario_6_fallback,
        escenario_7_sin_rostro,
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
