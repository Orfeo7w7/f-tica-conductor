# Sistema Experto de Seguridad Vial

Sistema de visión artificial en tiempo real para detectar distracción y
fatiga del conductor mediante webcam, combinando **MediaPipe** (rostro y
manos), extracción de características clásicas de visión (EAR, MAR, pose de
cabeza) y un **sistema experto difuso** (scikit-fuzzy) que determina el nivel
de riesgo y el tipo de alerta. La interfaz es un dashboard **Streamlit** con
tema oscuro estilo HUD automotriz.

Proyecto desarrollado para el curso de **Sistemas Expertos e Inteligencia
Artificial**.

## Requisitos de entorno

MediaPipe todavía no publica wheels para Python 3.14. Este proyecto requiere
**Python 3.12** (o inferior compatible con MediaPipe 0.10.x).

En Windows, si tiene varias versiones de Python instaladas, use el lanzador
`py` para seleccionar la 3.12 explícitamente.

## Instalación

```bash
cd driver-safety-system

# Crear entorno virtual con Python 3.12
py -3.12 -m venv venv

# Activar el entorno virtual (PowerShell)
venv\Scripts\Activate.ps1

# Instalar dependencias
pip install -r requirements.txt
```

## Ejecución

```bash
venv\Scripts\streamlit run src\ui\dashboard.py
```

Se abrirá el dashboard en el navegador (por defecto `http://localhost:8501`).
Active el interruptor **ACTIVAR MONITOREO** para iniciar la captura de la
webcam y el análisis en tiempo real.

## Arquitectura

```
src/
├── camera/video_capture.py     Captura de video (OpenCV)
├── detection/                  MediaPipe Face Mesh y Hands
│   ├── face_detector.py        468 landmarks faciales
│   └── hand_detector.py        21 landmarks por mano
├── features/                   Extracción de características
│   ├── eye_analyzer.py         EAR, parpadeos, mirada (iris)
│   ├── mouth_analyzer.py       MAR, bostezos
│   ├── head_pose.py            Pose de cabeza (solvePnP)
│   └── phone_usage.py          Heurística de uso de celular
├── expert_system/               Sistema experto
│   ├── membership.py           Funciones de pertenencia difusas
│   ├── fuzzy_engine.py         Motor de inferencia (riesgo numérico)
│   └── rules.py                12 reglas expertas (tipo de alerta)
├── ui/                          Interfaz
│   ├── dashboard.py             App Streamlit principal
│   ├── overlays.py              Anotaciones sobre el video
│   └── alerts.py                Gestión de alertas activas (cooldown)
├── utils/                       Logging y métricas
└── main.py                      Pipeline que integra todo el flujo
```

### Sistema experto difuso

El motor difuso (`fuzzy_engine.py`) recibe 4 variables de entrada:

| Variable | Rango | Particiones |
|---|---|---|
| Somnolencia | 0-100% | baja (0-30) · media (20-60) · alta (50-100) |
| Distracción | 0-100% | baja (0-30) · media (20-60) · alta (50-100) |
| Uso de celular | 0-100% | bajo (0-30) · medio (20-60) · alto (50-100) |
| Duración de somnolencia | 0-10 s | corta (0-2) · media (1-5) · larga (4-10) |

Y produce un **riesgo numérico continuo (0-100)** mediante un sistema de
control Mamdani (`skfuzzy.control`). Como una salida difusa no puede
representar directamente una categoría textual ("USO_CELULAR",
"FATIGA_CRÍTICA", etc.), la capa `rules.py` evalúa las **12 reglas expertas**
explícitas del dominio (en orden de severidad) sobre las mismas variables de
entrada para determinar el **tipo de alerta**, y combina ese resultado con el
nivel de riesgo difuso quedándose con el más severo de los dos.

Niveles de riesgo: **BAJO** (0-30) · **MEDIO** (30-70) · **ALTO** (70-90) ·
**CRÍTICO** (90-100).

### Notas sobre las heurísticas

- **Detección de celular**: no hay un detector de objetos en el stack (solo
  MediaPipe Hands), por lo que el uso de celular se **aproxima** mediante la
  proximidad mano-rostro, la forma de la mano (empuñada) y la persistencia en
  el tiempo. No es reconocimiento real del objeto.
- **Mirada fuera de eje**: se estima con la posición horizontal del iris
  (landmarks 468-477 de Face Mesh con `refine_landmarks=True`) relativa a las
  comisuras de cada ojo.

## Configuración

Todos los umbrales (EAR, MAR, ángulos de cabeza, distancias, tiempos de
persistencia) y los colores del tema visual están centralizados en
`config/config.yaml`.

## Rendimiento

- Objetivo: ≥ 20 FPS, < 50 ms de procesamiento por frame, latencia de alerta
  < 1 segundo. El valor real depende del hardware de la máquina donde se
  ejecute; el propio dashboard muestra FPS y ms/frame en vivo.
