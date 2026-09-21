# High Garden Coffee · Inteligencia de mercado

Prueba técnica de Machine Learning. Pipeline reproducible que responde una sola pregunta de negocio:
**¿dónde crece la demanda de café, con qué confianza y qué hacer con ello?**

A partir de 30 años de consumo doméstico (1990/91 – 2019/20) en 55 países, el proyecto hace un análisis exploratorio,
pronostica 5 temporadas, segmenta los mercados, construye un ranking de oportunidad y agrega una capa de IA generativa
(Claude) cuyas cifras se verifican automáticamente. Incluye una app para preguntar en lenguaje natural.

## Resultados principales

| Etapa | Hallazgo |
|---|---|
| EDA | El consumo se multiplicó por 2.56 (1,171 M → 2,999 M), pero el crecimiento pasó de 3.3% anual a 1.5% en las últimas 5 temporadas. Brasil pesa 44% del consumo; 55% de las variaciones anuales son exactamente cero. |
| Pronóstico | Modelo híbrido: error de volumen (WAPE) de 4.7% frente a 7.7% del baseline ingenuo (−39%). Proyección de +12.4% a 2024/25, con un sesgo optimista de 5.8% a 5 temporadas: leer como techo razonable. |
| Segmentos | 6 segmentos con K-Means. Los 8 «Grandes en crecimiento sostenido» concentran 71% del consumo; los «Emergentes» pasaron de 8.8% anual (15 temporadas) a 1.8% (últimas 5). |
| Ranking | 7 mercados prioritarios (82% del consumo) explican +362 M de los +372 M de crecimiento neto proyectado. Vietnam y Tailandia llevan alerta de optimismo. |
| Datos externos | En dólares reales el precio internacional bajó 13% entre 1990/91 y 2017/18, aunque el nominal subió 62%. |
| Precios | Rangos del 80% para Arábica y Robusta a 1–6 años. El modelo no anticipó el alza de 2021–2024, aunque los 12 precios realizados de 2019–2024 cayeron dentro de la banda (muy ancha). |
| LLM | El LLM redacta; los números salen del pipeline y un verificador decide qué se publica. 24/24 análisis finales válidos, 36/36 respuestas del agente con cifras verificadas. |

## Estructura

```
.
├── 01_eda.ipynb              Calidad de datos, unidades, ceros, concentración   → outputs/coffee_long.parquet
├── 02_forecast.ipynb         Backtesting y pronóstico a 5 temporadas            → forecast, backtest, metrics
├── 03_segmentation.ipynb     Segmentación de mercados (K-Means)                 → segments.parquet
├── 04_opportunity.ipynb      Score de oportunidad y sensibilidad de pesos       → ranking.parquet / .csv
├── 05_llm_insights.ipynb     Informe ejecutivo y agente verificados             → insights.json, informe_ejecutivo.md
├── 06_datos_externos.ipynb   Oferta, precios e IPC de EE. UU.                   → coffee_extra, prices_*.parquet
├── 07_price_forecast.ipynb   Rangos de precio Arábica y Robusta                 → price_*.parquet / .csv
├── app.py                    App de Streamlit para preguntar en lenguaje natural
├── utils.py                  Utilidades compartidas (rutas, series, métricas)
├── llm_utils.py              Capa LLM: fichas, prompts, verificador, agente, caché
├── app_utils.py              Lógica de la app (fichas, gráficos, reconocimiento de países)
├── external_data.py          ETL de datos externos
├── price_forecast.py         Modelos y backtest de precios
├── test_*.py                 158 pruebas automáticas (no requieren API key ni red)
├── coffee_db.parquet         Dataset base: consumo doméstico por país y temporada
├── datos_externos/           CSV de la ICO, IPC de EE. UU. (BLS) y precios realizados 2019–2025
└── outputs/                  Resultados de cada etapa (parquet/CSV/JSON) y gráficos en figs/
```

Cada notebook guarda su resultado en `outputs/` y el siguiente lo lee, así que puede re-ejecutarse cualquier parte por separado.
Los resultados ya vienen incluidos: no hace falta ejecutar nada para explorarlos.

Orden de ejecución: `01` → `02` y `03` (independientes entre sí) → `04` → `05`. Los notebooks `06` → `07` son una extensión
(precios) y `06` usa `outputs/ranking.parquet` solo para una tabla de lectura.

## Cómo ejecutarlo

Requiere Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate          # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install pytest                 # opcional, para correr las pruebas con pytest
```

Notebooks: abre la carpeta en VS Code o Jupyter y ejecuta los `.ipynb` en el orden indicado.

App interactiva:

```bash
streamlit run app.py
```

Pruebas (158, sin API key ni red):

```bash
python -m pytest -q
```

## Configurar la API de Claude (opcional)

Sin API key todo funciona en **modo plantilla**: el texto sale de plantillas determinísticas que pasan las mismas reglas
de verificación (o de la caché `outputs/llm_cache.json`, si existe). Para usar el LLM:

```bash
cp .env.example .env
# edita .env y pon tu ANTHROPIC_API_KEY
```

Variables (todas en `.env` o como variables de entorno):

| Variable | Uso |
|---|---|
| `ANTHROPIC_API_KEY` | Clave de la API. Sin ella se usa el modo plantilla. |
| `CLAUDE_MODEL` | Opcional. Por defecto `claude-sonnet-5`. |
| `ANTHROPIC_BASE_URL` | Opcional, solo si usas un endpoint distinto al oficial. |

> **Nunca subas `.env` a GitHub.** El `.gitignore` de este repositorio ya lo excluye. Si una clave llegó a publicarse, revócala y crea otra.

## Cómo se usa el LLM

Principio de diseño: **el LLM no analiza datos, redacta sobre resultados ya calculados**.

1. **Ficha de hechos:** los resultados de los notebooks 01–04 se resumen en un JSON cerrado por país y para el portafolio.
2. **Claude** redacta con salida estructurada (herramienta con esquema).
3. **Verificador:** cada cifra del texto debe existir en la ficha, la recomendación debe ser coherente con el nivel del ranking
   y se exigen las cautelas (alerta de optimismo, series planas, nichos).
4. Si no es válido, reintenta con feedback (hasta 3 veces); si sigue fallando o no hay API, usa una plantilla determinística verificada.
5. **Agente de preguntas** con 10 herramientas (function calling) sobre ranking, perfiles, comparaciones, pronóstico, histórico,
   tipos de café, catálogo de datos y precios; sus cifras también se verifican contra lo que devuelven las herramientas.

Lo medido con la API real está en el notebook 05. Límites conocidos: el verificador valida cifras, coherencia y cautelas, no cada
afirmación cualitativa (conviene leer el informe antes de presentarlo); el agente no es determinista y la batería de evaluación es
pequeña (12 preguntas); las herramientas de precios aún no se han medido con la API.

## Datos y fuentes

- `coffee_db.parquet`: dataset base del caso (consumo doméstico anual por país y tipo de café). La unidad no está declarada;
  el análisis asume que probablemente son kg (Brasil 2019/20 ≈ 22 M de sacos de 60 kg) y las conclusiones son relativas.
- `datos_externos/`: archivos históricos de la **ICO** (Organización Internacional del Café; citar la fuente), IPC-U de EE. UU. del
  **BLS** y precios realizados 2019–2025 (Banco Mundial y FMI). La base completa WCSD de la ICO es de pago y no se usa.

## Limitaciones

- Los 51 países modelados son productores: el dato no incluye demanda de importación ni a los grandes consumidores (EE. UU., Alemania).
- Los datos de consumo terminan en 2019/20 y no capturan shocks posteriores; 21 países tienen series casi planas que parecen estimaciones.
- El pronóstico de consumo es un escenario con sesgo optimista; no usa precios (en una prueba exploratoria no explicaron el crecimiento).
- Los precios son de referencia global, no los precios de venta de High Garden, y sus bandas son muy anchas.
- Para pasar de demanda a margen faltan importaciones, precios propios y logística.
