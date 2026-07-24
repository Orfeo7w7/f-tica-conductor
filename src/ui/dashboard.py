"""Dashboard Streamlit — Sistema Experto de Seguridad Vial.

Interfaz principal: video en vivo anotado, velocímetro de riesgo, métricas de
las variables difusas, panel de alertas, historial de eventos y estadísticas
de sesión. Tema oscuro tipo HUD automotriz: colores planos, sin degradados,
sin emojis, tipografía monoespaciada.
"""

from __future__ import annotations

import os
import sys
import threading

import av
import plotly.graph_objects as go
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.main import Pipeline
from src.utils.config_loader import cargar_config

CONFIG = cargar_config()
TEMA = CONFIG["theme"]

_COLOR_NIVEL = {
    "BAJO": TEMA["accent_green"],
    "MEDIO": TEMA["accent_amber"],
    "ALTO": TEMA["accent_orange"],
    "CRITICO": TEMA["accent_red"],
}


class ProcesadorWebcamNavegador(VideoProcessorBase):
    """Procesa la cámara del navegador y devuelve cada frame anotado.

    La cámara se captura en el navegador del visitante mediante WebRTC; nunca
    se intenta abrir un dispositivo de video en el servidor de Streamlit.
    """

    def __init__(self) -> None:
        self.pipeline = Pipeline(CONFIG)
        self._ultimo_resultado: dict = {}
        self._lock = threading.Lock()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        frame_bgr = frame.to_ndarray(format="bgr24")
        resultado = self.pipeline.procesar_frame(frame_bgr)
        resumen = {clave: valor for clave, valor in resultado.items() if clave != "frame"}
        with self._lock:
            self._ultimo_resultado = resumen
        return av.VideoFrame.from_ndarray(resultado["frame"], format="bgr24")

    def ultimo_resultado(self) -> dict:
        """Retorna una copia del último estado disponible para la interfaz."""
        with self._lock:
            return dict(self._ultimo_resultado)

    def on_ended(self) -> None:
        self.pipeline.cerrar()


def _inyectar_css() -> None:
    """Inyecta el CSS del tema oscuro automotriz: planos, angulares, sin degradados."""
    st.markdown(
        f"""
        <style>
        html, body, [class*="css"] {{
            font-family: Consolas, "Courier New", monospace !important;
        }}
        .stApp {{
            background-color: {TEMA['background']};
            color: {TEMA['text_primary']};
        }}
        #MainMenu, header, footer {{ visibility: hidden; }}

        .hud-titulo {{
            font-size: 1.7rem;
            font-weight: 700;
            letter-spacing: 0.35rem;
            color: {TEMA['accent_cyan']};
            border-bottom: 2px solid {TEMA['accent_cyan']};
            padding-bottom: 10px;
            margin-bottom: 4px;
            text-transform: uppercase;
        }}
        .hud-subtitulo {{
            color: {TEMA['text_dim']};
            letter-spacing: 0.2rem;
            font-size: 0.8rem;
            text-transform: uppercase;
            margin-bottom: 18px;
        }}
        .hud-panel {{
            background-color: {TEMA['panel_bg']};
            border: 1px solid {TEMA['border']};
            border-left: 3px solid {TEMA['accent_cyan']};
            padding: 14px 16px;
            margin-bottom: 14px;
        }}
        .hud-panel-header {{
            color: {TEMA['accent_cyan']};
            letter-spacing: 0.15rem;
            font-size: 0.78rem;
            text-transform: uppercase;
            border-bottom: 1px solid {TEMA['border']};
            padding-bottom: 6px;
            margin-bottom: 10px;
        }}
        .hud-metric-row {{
            display: flex;
            justify-content: space-between;
            font-size: 0.95rem;
            padding: 3px 0;
            border-bottom: 1px dotted {TEMA['border']};
        }}
        .hud-metric-label {{ color: {TEMA['text_dim']}; letter-spacing: 0.05rem; }}
        .hud-metric-value {{ color: {TEMA['text_primary']}; font-weight: 700; }}

        .hud-alert-banner {{
            border: 1px solid var(--c);
            border-left: 6px solid var(--c);
            padding: 12px 16px;
            font-size: 1.05rem;
            letter-spacing: 0.08rem;
            color: var(--c);
            background-color: {TEMA['panel_bg']};
            text-transform: uppercase;
            margin-bottom: 14px;
        }}

        .hud-history-item {{
            display: flex;
            justify-content: space-between;
            font-size: 0.82rem;
            padding: 5px 8px;
            border-left: 3px solid var(--c);
            background-color: {TEMA['background']};
            margin-bottom: 4px;
        }}
        .hud-history-time {{ color: {TEMA['text_dim']}; }}
        .hud-history-tipo {{ color: var(--c); font-weight: 700; }}

        div[data-testid="stToggle"] label p {{
            color: {TEMA['accent_cyan']} !important;
            letter-spacing: 0.15rem;
            text-transform: uppercase;
            font-size: 0.85rem !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _crear_gauge_riesgo(valor: float, nivel: str) -> go.Figure:
    """Construye el velocímetro (gauge) de riesgo con colores planos por zona."""
    color_aguja = _COLOR_NIVEL.get(nivel, TEMA["accent_green"])
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=valor,
            number={"suffix": "", "font": {"color": TEMA["text_primary"], "size": 42}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": TEMA["text_dim"], "tickwidth": 1},
                "bar": {"color": color_aguja, "thickness": 0.28},
                "bgcolor": TEMA["panel_bg"],
                "bordercolor": TEMA["border"],
                "borderwidth": 1,
                "steps": [
                    {"range": [0, 30], "color": "#0f2318"},
                    {"range": [30, 70], "color": "#2b230b"},
                    {"range": [70, 90], "color": "#2e1608"},
                    {"range": [90, 100], "color": "#2e0a12"},
                ],
                "threshold": {
                    "line": {"color": TEMA["accent_red"], "width": 3},
                    "thickness": 0.9,
                    "value": 90,
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor=TEMA["panel_bg"],
        font={"color": TEMA["text_primary"]},
        margin=dict(l=20, r=20, t=10, b=10),
        height=230,
    )
    return fig


def _crear_barras_variables(somnolencia: float, distraccion: float, celular: float) -> go.Figure:
    """Construye un gráfico de barras horizontal con las 3 variables difusas de entrada."""
    etiquetas = ["SOMNOLENCIA", "DISTRACCION", "CELULAR"]
    valores = [somnolencia, distraccion, celular]
    colores = [TEMA["accent_cyan"], TEMA["accent_amber"], TEMA["accent_orange"]]

    fig = go.Figure(
        go.Bar(
            x=valores, y=etiquetas, orientation="h",
            marker=dict(color=colores, line=dict(color=TEMA["border"], width=1)),
            text=[f"{v:.0f}%" for v in valores], textposition="outside",
            textfont=dict(color=TEMA["text_primary"]),
        )
    )
    fig.update_layout(
        paper_bgcolor=TEMA["panel_bg"], plot_bgcolor=TEMA["panel_bg"],
        font={"color": TEMA["text_primary"], "size": 12},
        xaxis=dict(range=[0, 115], showgrid=False, visible=False),
        yaxis=dict(showgrid=False),
        margin=dict(l=10, r=10, t=10, b=10),
        height=170,
    )
    return fig


def _panel_metricas_placeholder(contenedor, resultado: dict) -> None:
    """Renderiza el panel de métricas numéricas (EAR, MAR, pose, FPS)."""
    if not resultado.get("rostro_detectado", False):
        contenedor.markdown(
            f"""
            <div class="hud-panel">
                <div class="hud-panel-header">TELEMETRIA</div>
                <div class="hud-metric-row"><span class="hud-metric-label">ESTADO</span>
                <span class="hud-metric-value">ROSTRO NO DETECTADO</span></div>
                <div class="hud-metric-row"><span class="hud-metric-label">FPS</span>
                <span class="hud-metric-value">{resultado.get('fps', 0):.1f}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    filas = [
        ("EAR PROMEDIO", f"{resultado['ear_promedio']:.3f}"),
        ("MAR", f"{resultado['mar']:.3f}"),
        ("YAW / PITCH", f"{resultado['yaw']:.1f} / {resultado['pitch']:.1f}"),
        ("CABEZA DESVIADA", "SI" if resultado["cabeza_desviada"] else "NO"),
        ("MIRADA FUERA DE EJE", "SI" if resultado["mirada_fuera"] else "NO"),
        ("DURACION SOMNOLENCIA", f"{resultado['duracion_somnolencia']:.2f} s"),
        ("FPS", f"{resultado['fps']:.1f}"),
        ("MS/FRAME", f"{resultado['ms_frame']:.1f}"),
    ]
    if resultado.get("calibracion_estado") == "recalibrando":
        filas.append(("PERFIL", "RECALIBRANDO (alertas activas)"))
    if resultado.get("perclos") is not None:
        filas.append(("PERCLOS (3 MIN)", f"{resultado['perclos'] * 100:.1f} %"))
    filas_html = "".join(
        f'<div class="hud-metric-row"><span class="hud-metric-label">{k}</span>'
        f'<span class="hud-metric-value">{v}</span></div>'
        for k, v in filas
    )
    contenedor.markdown(
        f'<div class="hud-panel"><div class="hud-panel-header">TELEMETRIA</div>{filas_html}</div>',
        unsafe_allow_html=True,
    )


def _panel_calibracion(contenedor, resultado: dict) -> None:
    """Renderiza el panel de progreso durante la fase inicial de calibración.

    Se muestra en el lugar del panel de telemetría (``metricas_slot``)
    mientras ``resultado["calibrando"]`` es ``True`` — en esa fase no hay
    alertas ni riesgo que mostrar, solo el progreso de aprendizaje del
    perfil del conductor.
    """
    progreso = resultado.get("calibracion_progreso_seg", 0.0)
    total = resultado.get("calibracion_progreso_total_seg", 0.0)
    porcentaje = min(100.0, (progreso / total * 100.0)) if total > 0 else 0.0
    contenedor.markdown(
        f"""
        <div class="hud-panel">
            <div class="hud-panel-header">CALIBRACION DE PERFIL</div>
            <div class="hud-metric-row"><span class="hud-metric-label">ESTADO</span>
            <span class="hud-metric-value">APRENDIENDO EAR/MAR DEL CONDUCTOR</span></div>
            <div class="hud-metric-row"><span class="hud-metric-label">PROGRESO</span>
            <span class="hud-metric-value">{progreso:.0f} / {total:.0f} S</span></div>
            <div style="background-color:{TEMA['background']}; border:1px solid {TEMA['border']};
                        height:10px; margin-top:10px;">
                <div style="background-color:{TEMA['accent_cyan']}; width:{porcentaje:.0f}%; height:100%;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _panel_historial(contenedor, session_stats) -> None:
    """Renderiza el historial de eventos recientes con color según severidad."""
    eventos = session_stats.historial_reciente(12)
    if not eventos:
        items_html = (
            f'<div class="hud-metric-label">SIN EVENTOS REGISTRADOS AUN</div>'
        )
    else:
        items_html = "".join(
            f'<div class="hud-history-item" style="--c:{_COLOR_NIVEL.get(e.nivel_riesgo, TEMA["accent_green"])}">'
            f'<span class="hud-history-time">{e.hora_legible}</span>'
            f'<span class="hud-history-tipo">{e.tipo_alerta.replace("_", " ")}</span>'
            f'<span>{e.nivel_riesgo}</span>'
            f"</div>"
            for e in eventos
        )
    contenedor.markdown(
        f'<div class="hud-panel"><div class="hud-panel-header">HISTORIAL DE EVENTOS</div>{items_html}</div>',
        unsafe_allow_html=True,
    )


def _panel_estadisticas(contenedor, session_stats) -> None:
    """Renderiza las estadísticas acumuladas de la sesión actual."""
    duracion_min = session_stats.duracion_sesion_segundos / 60.0
    conteo_alertas = sum(session_stats.alert_counts.values())
    filas = [
        ("DURACION SESION", f"{duracion_min:.1f} min"),
        ("PARPADEOS", f"{session_stats.blink_count}"),
        ("BOSTEZOS", f"{session_stats.yawn_count}"),
        ("ALERTAS TOTALES", f"{conteo_alertas}"),
    ]
    filas_html = "".join(
        f'<div class="hud-metric-row"><span class="hud-metric-label">{k}</span>'
        f'<span class="hud-metric-value">{v}</span></div>'
        for k, v in filas
    )
    contenedor.markdown(
        f'<div class="hud-panel"><div class="hud-panel-header">ESTADISTICAS DE SESION</div>{filas_html}</div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    """Función principal del dashboard Streamlit."""
    st.set_page_config(
        page_title="Sistema Experto de Seguridad Vial",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _inyectar_css()

    st.markdown(
        '<div class="hud-titulo">SISTEMA EXPERTO DE SEGURIDAD VIAL</div>'
        '<div class="hud-subtitulo">MONITOREO DE DISTRACCION Y FATIGA DEL CONDUCTOR &mdash; '
        'VISION ARTIFICIAL + LOGICA DIFUSA</div>',
        unsafe_allow_html=True,
    )

    col_control, _ = st.columns([1, 5])
    activo = col_control.toggle("ACTIVAR MONITOREO", key="activo")

    col_video, col_panel = st.columns([2, 1])
    video_slot = col_video.empty()
    alerta_slot = col_video.empty()
    gauge_slot = col_panel.empty()
    barras_slot = col_panel.empty()
    metricas_slot = col_panel.empty()
    historial_slot = col_video.empty()
    stats_slot = col_panel.empty()

    if not activo:
        video_slot.markdown(
            f'<div class="hud-panel" style="height:480px; display:flex; '
            f'align-items:center; justify-content:center; color:{TEMA["text_dim"]}; '
            f'letter-spacing:0.2rem; text-transform:uppercase;">'
            f"CAMARA INACTIVA &mdash; ACTIVE EL MONITOREO PARA INICIAR</div>",
            unsafe_allow_html=True,
        )
        gauge_slot.plotly_chart(
            _crear_gauge_riesgo(0, "BAJO"), use_container_width=True, key="gauge_riesgo_idle"
        )
        metricas_slot.info("Active el monitoreo y permita el acceso a la cámara cuando el navegador lo solicite.")
        return

    with col_video:
        st.caption("Permita el acceso a la cámara y presione START para iniciar el análisis en tiempo real.")
        contexto = webrtc_streamer(
            key="camara-navegador",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=ProcesadorWebcamNavegador,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": CONFIG["camera"]["width"]},
                    "height": {"ideal": CONFIG["camera"]["height"]},
                    "facingMode": "user",
                },
                "audio": False,
            },
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            async_processing=True,
        )

    resultado = {}
    if contexto.video_processor:
        resultado = contexto.video_processor.ultimo_resultado()

    if not resultado:
        alerta_slot.markdown(
            f'<div class="hud-alert-banner" style="--c:{TEMA["accent_cyan"]}">'
            "CAMARA DEL NAVEGADOR LISTA :: PRESIONE START PARA COMENZAR</div>",
            unsafe_allow_html=True,
        )
        gauge_slot.plotly_chart(_crear_gauge_riesgo(0, "BAJO"), use_container_width=True)
        return

    nivel = resultado["nivel_riesgo"]
    tipo_alerta = resultado["tipo_alerta"]
    if resultado.get("calibrando"):
        alerta_slot.markdown(
            f'<div class="hud-alert-banner" style="--c:{TEMA["accent_cyan"]}">'
            "CALIBRANDO :: MANTENGA LOS OJOS ABIERTOS CON NORMALIDAD</div>",
            unsafe_allow_html=True,
        )
        _panel_calibracion(metricas_slot, resultado)
    else:
        color = _COLOR_NIVEL.get(nivel, TEMA["accent_green"])
        alerta_slot.markdown(
            f'<div class="hud-alert-banner" style="--c:{color}">'
            f"RIESGO {nivel} :: {tipo_alerta.replace('_', ' ')}</div>",
            unsafe_allow_html=True,
        )
        _panel_metricas_placeholder(metricas_slot, resultado)

    gauge_slot.plotly_chart(_crear_gauge_riesgo(resultado["riesgo"], nivel), use_container_width=True)
    barras_slot.plotly_chart(
        _crear_barras_variables(
            resultado.get("somnolencia", 0.0),
            resultado.get("distraccion", 0.0),
            resultado.get("celular", 0.0),
        ),
        use_container_width=True,
    )
    _panel_estadisticas(stats_slot, contexto.video_processor.pipeline.session_stats)
    _panel_historial(historial_slot, contexto.video_processor.pipeline.session_stats)


if __name__ == "__main__":
    main()
