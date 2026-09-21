"""Funciones de datos y gráficos del asistente de mercados (sin Streamlit, para poder probarlas).

- detectar_paises: encuentra países en el texto de una pregunta (español o inglés, con o sin tildes)
- ficha_pais: segmento (cluster), ranking, nivel, score y alertas de un país
- serie_pais / serie_total: histórico + proyección con intervalo
- ranking_grafico / segmentos_grafico: datos para los gráficos
- grafico_*: gráficos Altair
- resumen_sin_llm: resumen determinístico (plantilla verificada) cuando no hay API key
- precios: es_pregunta_de_precio, serie_precio_global, ranking_precio_productor, serie_precio_productor y sus gráficos
  (solo se muestran en preguntas de precios; los datos salen de los notebooks 06-07)
"""
from __future__ import annotations

import re

import altair as alt
import numpy as np
import pandas as pd

import llm_utils as L
from utils import get_series

# Nombres en español (sin tildes, en minúscula) -> nombre en el dataset
NOMBRES_ES = {
    "brasil": "Brazil", "camerun": "Cameroon", "republica centroafricana": "Central African Republic",
    "costa de marfil": "Côte d'Ivoire", "republica dominicana": "Dominican Republic", "etiopia": "Ethiopia",
    "kenia": "Kenya", "mexico": "Mexico", "papua nueva guinea": "Papua New Guinea", "filipinas": "Philippines",
    "republica democratica del congo": "RD Congo", "republica del congo": "Congo", "ruanda": "Rwanda",
    "sierra leona": "Sierra Leone", "tailandia": "Thailand", "viet nam": "Vietnam", "zimbabue": "Zimbabwe",
    "haiti": "Haiti", "panama": "Panama", "peru": "Peru", "laos": "Laos",
    # presentes en el dataset original pero no modelados (series vacías o muy cortas)
    "guinea ecuatorial": "Equatorial Guinea", "timor oriental": "Timor-Leste", "timor leste": "Timor-Leste",
}

EJEMPLOS = [
    "¿Cómo va Vietnam y qué riesgo tiene?",
    "Compara Colombia y México",
    "¿Cuáles son los 3 mercados con mayor score de oportunidad?",
    "¿Por qué Bolivia es un nicho y no prioritario?",
    "¿Qué variedades de café hay y cuánto pesa cada una?",
    "Dame un resumen del portafolio",
    "¿En qué país subió más el precio al productor?",
    "¿Entre qué valores puede estar el precio del café Arábica en 2023?",
    "¿En qué país aumentará más el precio en 2023 respecto a 2018?",
    "¿Cómo se han comportado los precios al productor en Tailandia?",
]

COLORES_SEGMENTO = {
    "Grandes en crecimiento sostenido": "#2e7d32",
    "Emergentes de alto crecimiento": "#1565c0",
    "Maduros estancados": "#ef6c00",
    "En retroceso": "#c62828",
    "Pequeños volátiles": "#8e24aa",
    "Nicho plano": "#757575",
}
COLOR_PAIS = "#c0392b"
COLOR_BASE = "#6f4e37"


# ---------------------------------------------------------------- detección de países
def detectar_paises(texto: str, T: dict):
    """Devuelve (modelados, no_modelados): países mencionados, en orden de aparición.

    'No modelados' son países presentes en el dataset original pero sin serie suficiente (p. ej. Nepal).
    Se busca de los nombres más largos a los más cortos y se 'consume' el texto encontrado, para que
    'República Democrática del Congo' no cuente además como 'Congo', ni 'Papúa Nueva Guinea' como 'Guinea'.
    """
    modelados = set(T["rk"].index)
    todos = set(T["long"]["country"])
    idx = {L._norm(p): p for p in todos}
    for k, v in {**L._ALIAS, **NOMBRES_ES}.items():
        if v in todos:
            idx[k] = v
    t = " " + re.sub(r"[^a-z0-9' \-]", " ", L._norm(texto)) + " "
    hallados = []
    for k in sorted(idx, key=len, reverse=True):
        patron = re.compile(rf"(?<![a-z0-9]){re.escape(k)}(?![a-z0-9])")
        while True:
            m = patron.search(t)
            if not m:
                break
            hallados.append((m.start(), idx[k]))
            t = t[:m.start()] + " " * (m.end() - m.start()) + t[m.end():]
    orden = []
    for _, p in sorted(hallados):
        if p not in orden:
            orden.append(p)
    return [p for p in orden if p in modelados], [p for p in orden if p not in modelados]


# ---------------------------------------------------------------- fichas y datos de gráficos
def ficha_pais(pais: str, T: dict) -> dict:
    fs = L.country_fact_sheet(pais, T)
    fs["definicion_nivel"] = L.DEFINICIONES_NIVELES[fs["nivel"]]
    fs["definicion_segmento"] = L.DEFINICIONES_SEGMENTOS.get(fs["segmento"], "")
    return fs


def serie_pais(pais: str, T: dict) -> pd.DataFrame:
    """Histórico + proyección a 5 temporadas con intervalo del 80% (la proyección arranca en el último dato real)."""
    s = get_series(T["long"], pais)
    hist = pd.DataFrame({"anio": s.index.astype(int), "valor": s.values, "tipo": "Histórico"})
    f = T["fc"][T["fc"]["country"] == pais].sort_values("h")
    ult = hist.iloc[-1]
    proy = pd.DataFrame({
        "anio": [int(ult.anio)] + f["year"].astype(int).tolist(),
        "valor": [ult.valor] + f["yhat"].tolist(),
        "bajo": [ult.valor] + f["lo"].tolist(),
        "alto": [ult.valor] + f["hi"].tolist(),
        "tipo": "Proyección",
    })
    out = pd.concat([hist, proy], ignore_index=True)
    out["pais"] = pais
    return out


def serie_total(T: dict) -> pd.DataFrame:
    """Consumo total de los países modelados: histórico y proyección (sin intervalo)."""
    paises = list(T["rk"].index)
    h = T["long"][T["long"]["country"].isin(paises)].groupby("year")["consumption_m"].sum()
    f = T["fc"].groupby("year")["yhat"].sum()
    hist = pd.DataFrame({"anio": h.index.astype(int), "valor": h.values, "tipo": "Histórico"})
    proy = pd.DataFrame({"anio": [int(h.index[-1])] + f.index.astype(int).tolist(),
                         "valor": [h.iloc[-1]] + f.tolist(), "tipo": "Proyección"})
    out = pd.concat([hist, proy], ignore_index=True)
    out["pais"] = "Portafolio (51 países)"
    return out


def ranking_grafico(paises, T: dict, top: int = 15) -> pd.DataFrame:
    """Top N del ranking más los países mencionados (aunque queden fuera del top)."""
    rk = T["rk"].sort_values("posicion")
    d = pd.concat([rk.head(top), rk.loc[rk.index.intersection(list(paises))]])
    d = d[~d.index.duplicated()].sort_values("posicion")
    return pd.DataFrame({
        "pais": d.index, "posicion": d["posicion"].astype(int).values, "score": d["score"].values,
        "nivel": d["nivel"].values, "segmento": d["segmento"].values,
        "seleccionado": d.index.isin(list(paises)),
        "etiqueta": [f"{int(p)}. {c}" for c, p in zip(d.index, d["posicion"])],
    })


def segmentos_grafico(T: dict) -> pd.DataFrame:
    seg, rk = T["seg"], T["rk"]
    return pd.DataFrame({
        "pais": seg.index, "segmento": seg["segmento"].values,
        "consumo_2019": seg["consumo_2019"].values,
        "crecimiento_pct": (np.exp(seg["trend_15"].values) - 1) * 100,
        "posicion": rk.loc[seg.index, "posicion"].astype(int).values,
    })


# ---------------------------------------------------------------- gráficos (Altair)
def grafico_serie(df: pd.DataFrame, titulo: str = "") -> alt.Chart:
    base = alt.Chart(df).encode(x=alt.X("anio:Q", title="Temporada (año de inicio)", axis=alt.Axis(format="d")))
    proy = alt.datum.tipo == "Proyección"
    capas = []
    if "bajo" in df.columns:
        capas.append(base.transform_filter(proy).mark_area(opacity=0.18, color="#2e86c1").encode(
            y=alt.Y("bajo:Q", title="Consumo (millones)"), y2="alto:Q"))
    capas.append(base.transform_filter(alt.datum.tipo == "Histórico").mark_line(color="#222", strokeWidth=2).encode(
        y=alt.Y("valor:Q", title="Consumo (millones)"),
        tooltip=[alt.Tooltip("anio:Q", title="Año", format="d"), alt.Tooltip("valor:Q", title="Consumo", format=",.1f")]))
    capas.append(base.transform_filter(proy).mark_line(color="#2e86c1", strokeWidth=2, strokeDash=[6, 3]).encode(
        y="valor:Q",
        tooltip=[alt.Tooltip("anio:Q", title="Año", format="d"), alt.Tooltip("valor:Q", title="Proyección", format=",.1f")]))
    return alt.layer(*capas).properties(title=titulo, height=280)


def grafico_ranking(df: pd.DataFrame) -> alt.Chart:
    return alt.Chart(df).mark_bar().encode(
        y=alt.Y("etiqueta:N", sort=alt.EncodingSortField(field="posicion", order="ascending"), title=None),
        x=alt.X("score:Q", title="Score de oportunidad (0-100)", scale=alt.Scale(domain=[0, 100])),
        color=alt.condition(alt.datum.seleccionado, alt.value(COLOR_PAIS), alt.value(COLOR_BASE)),
        tooltip=["pais", "posicion", alt.Tooltip("score:Q", format=".1f"), "nivel", "segmento"],
    ).properties(height=max(220, 24 * len(df)))


def grafico_segmentos(df: pd.DataFrame, paises=()) -> alt.Chart:
    paises = list(paises)
    dominio = list(COLORES_SEGMENTO)
    color = alt.Color("segmento:N", scale=alt.Scale(domain=dominio, range=[COLORES_SEGMENTO[s] for s in dominio]),
                      legend=alt.Legend(title="Segmento (cluster)", orient="bottom", columns=3, labelLimit=320))
    dom_x = [float(df["consumo_2019"].min()) * 0.6, float(df["consumo_2019"].max()) * 1.6]     # ajustado a los datos
    base = alt.Chart(df).encode(
        x=alt.X("consumo_2019:Q", scale=alt.Scale(type="log", domain=dom_x),
                title="Consumo 2019/20 (millones, escala log)", axis=alt.Axis(grid=False)),
        y=alt.Y("crecimiento_pct:Q", title="Crecimiento anual de largo plazo (%)"),
        tooltip=["pais", "segmento", alt.Tooltip("consumo_2019:Q", format=",.1f"),
                 alt.Tooltip("crecimiento_pct:Q", format=".1f"), "posicion"])
    puntos = base.mark_circle(size=70, opacity=0.75).encode(color=color)
    capas = [puntos, alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#999", strokeDash=[3, 3]).encode(y="y:Q")]
    if paises:
        sel = df[df["pais"].isin(paises)]
        capas.append(alt.Chart(sel).mark_circle(size=260, color="none", stroke=COLOR_PAIS, strokeWidth=3).encode(
            x="consumo_2019:Q", y="crecimiento_pct:Q"))
        capas.append(alt.Chart(sel).mark_text(dy=-16, fontWeight="bold", color=COLOR_PAIS).encode(
            x="consumo_2019:Q", y="crecimiento_pct:Q", text="pais:N"))
    return alt.layer(*capas).properties(height=380)


# ---------------------------------------------------------------- resumen sin LLM
def tarjeta_md(a: dict, titulo: str) -> str:
    """Formatea un análisis (plantilla o LLM) como Markdown."""
    return (f"**{titulo} · {a['recomendacion']}**\n\n{a['titular']}\n\n{a['resumen']}\n\n"
            f"- *A favor:* {' '.join(a['fortalezas'])}\n- *Riesgos:* {' '.join(a['riesgos'])}\n"
            f"- *Por qué:* {a['justificacion']}")


def resumen_sin_llm(paises, T: dict, max_paises: int = 3) -> str:
    """Resumen determinístico (plantillas verificadas) para cuando no hay LLM."""
    if paises:
        return "\n\n---\n\n".join(tarjeta_md(L.template_mercado(L.country_fact_sheet(p, T)), p)
                                  for p in list(paises)[:max_paises])
    a = L.template_portafolio(L.portfolio_fact_sheet(T))
    return (f"**{a['titular']}**\n\n{a['resumen_ejecutivo']}\n\n**Hallazgos**\n\n" +
            "\n".join(f"- {x}" for x in a["hallazgos"]) + "\n\n**Riesgos y límites**\n\n" +
            "\n".join(f"- {x}" for x in a["riesgos_y_limites"]))


# ---------------------------------------------------------------- precios (notebooks 06-07)
HERRAMIENTAS_PRECIO = {"precios_globales", "precio_productor"}
COLOR_PRONOSTICO, COLOR_REALIZADO = "#2e86c1", "#2e7d32"
NOMBRE_SERIE_PRECIO = {"other_milds": "Arábica (Other Milds)", "robustas": "Robusta"}


def es_pregunta_de_precio(pregunta: str, herramientas=()) -> bool:
    """¿La pregunta trata de precios? (por las herramientas que usó el agente o por la palabra 'precio')."""
    if set(herramientas) & HERRAMIENTAS_PRECIO:
        return True
    return bool(re.search(r"\bprecios?\b|\bcotizacion", L._norm(pregunta)))


def hay_precios_globales(T: dict) -> bool:
    return L._hay_precios_globales(T)


def serie_precio_global(T: dict) -> pd.DataFrame:
    """Precio real anual (US$ de 2019/kg): histórico 1990-2018, pronóstico con banda del 80% y precio realizado.

    La proyección arranca en el último dato real (2018), igual que en los gráficos de consumo.
    """
    filas = []
    for col, nombre in NOMBRE_SERIE_PRECIO.items():
        real = L._anual_precio(T, col)
        for y, v in real.items():
            filas.append((nombre, int(y), float(v), np.nan, np.nan, "Histórico"))
        ult_y, ult_v = int(real.index.max()), float(real.iloc[-1])
        filas.append((nombre, ult_y, ult_v, ult_v, ult_v, "Pronóstico"))
        fc = T["price_fc"][T["price_fc"]["serie"] == col].sort_values("anio")
        for r in fc.itertuples():
            filas.append((nombre, int(r.anio), float(r.central), float(r.lo), float(r.hi), "Pronóstico"))
        va = T["price_val"][T["price_val"]["serie"] == col].sort_values("anio")
        for r in va.itertuples():
            filas.append((nombre, int(r.anio), float(r.real), np.nan, np.nan, "Realizado"))
    return pd.DataFrame(filas, columns=["serie", "anio", "valor", "bajo", "alto", "tipo"])


def ranking_precio_productor(T: dict, desde: int | None = None, hasta: int | None = None, n: int = 5,
                             tipo: str | None = None) -> pd.DataFrame | None:
    """Mayores aumentos y caídas del precio real al productor, POR TIPO de café (mismos números que ve el agente)."""
    r = L.tool_precio_productor(T, desde=desde, hasta=hasta, n=n, tipo=tipo)
    if "por_tipo" not in r:
        return None
    filas = []
    for e in r["por_tipo"]:
        if e.get("sin_datos_suficientes"):
            continue
        filas += [dict(x, tipo=e["tipo"], grupo="Mayores aumentos") for x in e["mayores_aumentos"]]
        filas += [dict(x, tipo=e["tipo"], grupo="Mayores caídas") for x in e["mayores_caidas"]]
    if not filas:
        return None
    d = pd.DataFrame(filas).drop_duplicates(["tipo", "pais"])
    d["etiqueta"] = np.where(d["productor_pequeno"], d["pais"] + " (productor pequeño)", d["pais"])
    claves = ("n_paises_con_dato", "n_paises_con_dato_en_desde", "n_paises_con_dato_en_hasta", "n_paises_con_algun_precio")
    d.attrs.update(periodo=r["periodo"], por_defecto=r["ventana_por_defecto"], n_modelados=r["n_paises_modelados"],
                   n_clave=r["n_mercados_clave"], fuera=r["mercados_clave_fuera_del_ranking"],
                   por_tipo={e["tipo"]: {k: e[k] for k in claves} for e in r["por_tipo"]})
    return d


def serie_precio_productor(paises, T: dict) -> pd.DataFrame:
    """Precio real al productor por año y por tipo de café, solo para los años con dato de los países mencionados."""
    ex = T.get("extra")
    vacio = pd.DataFrame(columns=["pais", "tipo", "serie", "anio", "precio"])
    if ex is None or "price_grower_arabica_real_usd_kg" not in ex.columns:
        return vacio
    partes = []
    for clave, nombre in (("arabica", "Arábica"), ("robusta", "Robusta")):
        col = f"price_grower_{clave}_real_usd_kg"
        d = ex[ex["country"].isin(list(paises)) & ex[col].notna()]
        partes.append(pd.DataFrame({"pais": d["country"].values, "tipo": nombre, "anio": d["year"].astype(int).values,
                                    "precio": d[col].values}))
    out = pd.concat(partes, ignore_index=True) if partes else vacio
    if len(out) == 0:
        return vacio
    out["serie"] = out["pais"] + " · " + out["tipo"]
    return out.sort_values(["pais", "tipo", "anio"]).reset_index(drop=True)


def resumen_precio_pais(pais: str, T: dict) -> str:
    """Una línea con los años que tienen dato por tipo de café (para la ficha y los pies de gráfico)."""
    r = L.tool_precio_productor(T, pais=pais)
    if "tipos" not in r:
        return "sin dato de precio al productor"
    return " · ".join(f"{t['tipo']} {t['primer_anio']}–{t['ultimo_anio']} ({t['n_anios_con_dato']} años con dato)"
                      for t in r["tipos"])


def grafico_precio_serie(df: pd.DataFrame, titulo: str = "") -> alt.Chart:
    base = alt.Chart(df).encode(x=alt.X("anio:Q", title="Año", axis=alt.Axis(format="d")))
    pron = alt.datum.tipo == "Pronóstico"
    tip = [alt.Tooltip("anio:Q", title="Año", format="d"), alt.Tooltip("valor:Q", title="US$ de 2019/kg", format=".2f")]
    capas = [
        base.transform_filter(pron).mark_area(opacity=0.18, color=COLOR_PRONOSTICO).encode(
            y=alt.Y("bajo:Q", title="US$ de 2019 por kg"), y2="alto:Q"),
        base.transform_filter(alt.datum.tipo == "Histórico").mark_line(color="#222", strokeWidth=2).encode(
            y=alt.Y("valor:Q", title="US$ de 2019 por kg"), tooltip=tip),
        base.transform_filter(pron).mark_line(color=COLOR_PRONOSTICO, strokeWidth=2, strokeDash=[6, 3]).encode(
            y="valor:Q", tooltip=tip),
        base.transform_filter(alt.datum.tipo == "Realizado").mark_point(
            color=COLOR_REALIZADO, filled=True, size=70, opacity=1).encode(y="valor:Q", tooltip=tip),
    ]
    return alt.layer(*capas).properties(title=titulo, height=260)


def grafico_variacion_precio(df: pd.DataFrame) -> alt.Chart:
    """Barras de un solo tipo de café (Arábica y Robusta no son comparables: se dibujan por separado)."""
    orden = df.sort_values("variacion_real_pct", ascending=False)["etiqueta"].tolist()
    return alt.Chart(df).mark_bar().encode(
        y=alt.Y("etiqueta:N", sort=orden, title=None),
        x=alt.X("variacion_real_pct:Q", title="Variación del precio real al productor (%)"),
        color=alt.condition(alt.datum.productor_pequeno, alt.value("#9e9e9e"), alt.value(COLOR_BASE)),
        tooltip=["pais", alt.Tooltip("precio_real_desde:Q", title="US$ de 2019/kg (inicio)", format=".2f"),
                 alt.Tooltip("precio_real_hasta:Q", title="US$ de 2019/kg (fin)", format=".2f"),
                 alt.Tooltip("variacion_real_pct:Q", format="+.1f"), alt.Tooltip("produccion_hasta_M_sacos:Q", format=".1f")],
    ).properties(height=max(200, 26 * len(df)))


def grafico_precio_productor(df: pd.DataFrame) -> alt.Chart:
    return alt.Chart(df).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X("anio:Q", title="Año (solo los años con dato)", axis=alt.Axis(format="d")),
        y=alt.Y("precio:Q", title="US$ de 2019 por kg"), color=alt.Color("serie:N", title=None),
        tooltip=["serie", alt.Tooltip("anio:Q", format="d"), alt.Tooltip("precio:Q", format=".2f")]).properties(height=280)
