"""Asistente de mercados de café · High Garden Coffee.

Ejecutar (desde la carpeta de los notebooks):   streamlit run app.py

Haces una pregunta; la app muestra:
  1. un resumen (agente con LLM si hay API key; si no, plantilla determinística verificada),
  2. el segmento (cluster) y la posición en el ranking de cada país que menciones,
  3. gráficos: serie con pronóstico, ranking y mapa de segmentos,
  4. si la pregunta es de precios: precio internacional por tipo de café (pronóstico frente a lo realizado) y
     precio al productor por país.

Los datos salen de los notebooks 01-04 y, para precios, 06-07 (carpeta outputs/; los precios son opcionales). Las fichas y los gráficos se calculan directo de
los datos; solo el texto del resumen lo escribe el LLM, y sus cifras se verifican contra las herramientas.
"""
from __future__ import annotations

import streamlit as st

import app_utils as A
import llm_utils as L

st.set_page_config(page_title="Asistente de mercados de café", page_icon="☕", layout="wide")


# ------------------------------------------------------------------ compatibilidad entre versiones de Streamlit
def mostrar_chart(chart) -> None:
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:                       # versiones anteriores
        st.altair_chart(chart, use_container_width=True)


def boton_ancho(etiqueta: str, **kw) -> bool:
    try:
        return st.button(etiqueta, width="stretch", **kw)
    except TypeError:
        return st.button(etiqueta, use_container_width=True, **kw)


# ------------------------------------------------------------------ datos y cliente (una sola vez por sesión del servidor)
@st.cache_resource(show_spinner="Cargando resultados…")
def cargar_tablas():
    return L.load_tables()


@st.cache_resource
def obtener_cliente():
    return L.get_client()


try:
    T = cargar_tablas()
except Exception as e:  # noqa: BLE001
    st.error("No encuentro los resultados del análisis (carpeta `outputs/`). "
             "Ejecuta primero los notebooks 01 a 04 y vuelve a abrir la app.")
    st.caption(f"Detalle: {type(e).__name__}: {e}")
    st.stop()

client = obtener_cliente()
MODEL = L.get_model()


# ------------------------------------------------------------------ lógica
def responder(pregunta: str) -> dict:
    paises, excluidos = A.detectar_paises(pregunta, T)
    texto, modo, verificada, problemas, herramientas, aviso, llamadas = None, "plantilla", None, [], [], "", []
    if client is not None:
        try:
            with st.spinner("Consultando al agente…"):
                r = L.preguntar(client, pregunta, T, MODEL)
            if r["respuesta"]:
                texto, modo = r["respuesta"], "llm"
                verificada, problemas = r["verificada"], r["problemas"]
                llamadas = r["herramientas"]
                herramientas = [c["herramienta"] for c in llamadas]
        except Exception as e:  # noqa: BLE001 - la app no debe caerse por un fallo de la API
            aviso = f"No pude usar el LLM ({type(e).__name__}). Muestro el resumen determinístico."
    if texto is None:
        texto = A.resumen_sin_llm(paises, T)
    return {"pregunta": pregunta, "paises": paises, "excluidos": excluidos, "texto": texto, "modo": modo,
            "verificada": verificada, "problemas": problemas, "herramientas": herramientas, "aviso": aviso,
            "llamadas": llamadas}


def elegir_ejemplo(ejemplo: str) -> None:
    st.session_state["q"] = ejemplo
    st.session_state["auto"] = True


def ficha(pais: str) -> None:
    f = A.ficha_pais(pais, T)
    st.markdown(f"#### {pais}")
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    c1.caption("Segmento (cluster)")
    c1.markdown(f"**{f['segmento']}**")
    c2.metric("Ranking", f"{f['posicion_ranking']} de {f['n_paises_ranking']}")
    c3.metric("Nivel", f["nivel"])
    c4.metric("Score", f"{f['score_0_100']:.1f}")
    k1, k2, k3 = st.columns(3)
    k1.metric("Consumo 2019/20 (M)", f"{f['consumo_2019_20_M']:,.1f}")
    k2.metric("Proyección 2024/25 (M)", f"{f['proyeccion_2024_25_M']:,.1f}", f"{f['crecimiento_proyectado_M']:+,.1f} M")
    k3.metric("Crecimiento anual proyectado", f"{f['cagr_proyectado_pct']:.1f}%",
              f"reciente {f['crecimiento_reciente_pct']:.1f}%", delta_color="off")
    if f["alerta_optimismo"]:
        st.warning(f"**Alerta de optimismo:** la proyección ({f['cagr_proyectado_pct']:.1f}% anual) supera en "
                   f"{f['brecha_proyeccion_vs_reciente_pp']:.1f} puntos porcentuales su ritmo reciente "
                   f"({f['crecimiento_reciente_pct']:.1f}%). Conviene validarla antes de comprometer inversión.")
    if f["tipo_serie"] == "plana":
        st.info(f"**Serie plana:** {f['pct_anios_planos']:.0f}% de los años sin cambio; probablemente son "
                f"estimaciones y el pronóstico aporta poca información.")
    if "extra" in T:
        st.caption(f"Precio al productor: {A.resumen_precio_pais(pais, T)}.")
    with st.expander("¿Qué significan el segmento y el nivel?"):
        st.markdown(f"**{f['segmento']}:** {f['definicion_segmento']}")
        st.markdown(f"**{f['nivel']}:** {f['definicion_nivel']}")
        st.caption(f"Robustez del ranking: top {L.CONSTANTES['top_n_ranking']} en {f['pct_escenarios_top10']:.0f}% de los "
                   f"escenarios de pesos simulados · cuota del consumo total: {f['cuota_consumo_total_pct']:.1f}%")


def graficos(paises: list) -> None:
    t1, t2, t3 = st.tabs(["📈 Serie y pronóstico", "🏆 Ranking", "🧩 Segmentos (clusters)"])
    with t1:
        if paises:
            mostrados = paises[:3]
            cols = st.columns(len(mostrados))
            for col, p in zip(cols, mostrados):
                with col:
                    mostrar_chart(A.grafico_serie(A.serie_pais(p, T), p))
            st.caption("Línea negra: histórico · línea azul discontinua: proyección · banda: intervalo del 80% "
                       "calibrado con el backtest.")
            if len(paises) > 3:
                st.caption(f"Se muestran los 3 primeros países de {len(paises)} mencionados.")
        else:
            mostrar_chart(A.grafico_serie(A.serie_total(T), "Consumo total de los 51 países modelados"))
            st.caption("Menciona un país en la pregunta para ver su serie y su intervalo.")
    with t2:
        mostrar_chart(A.grafico_ranking(A.ranking_grafico(paises, T)))
        st.caption("Top 15 por score de oportunidad" + ("; en rojo, los países que mencionaste." if paises else "."))
    with t3:
        mostrar_chart(A.grafico_segmentos(A.segmentos_grafico(T), paises))
        st.caption("Cada punto es un país: tamaño del mercado (eje X, escala log) contra su crecimiento de largo plazo. "
                   "El color es el segmento." + (" Círculo rojo: los países que mencionaste." if paises else ""))


def precios(res: dict) -> None:
    """Panel de precios: solo aparece cuando la pregunta es de precios. Usa los mismos parámetros que el agente."""
    st.subheader("Precios")
    globales, productor = A.hay_precios_globales(T), "extra" in T
    if not (globales or productor):
        st.info("Los resultados de precios no están cargados: ejecuta los notebooks 06 y 07 y vuelve a abrir la app.")
        return
    llamadas = res.get("llamadas", [])
    sin_pais = [c["argumentos"] for c in llamadas if c["herramienta"] == "precio_productor" and not c["argumentos"].get("pais")]
    if globales:
        df = A.serie_precio_global(T)
        cols = st.columns(len(A.NOMBRE_SERIE_PRECIO))
        for col, nombre in zip(cols, A.NOMBRE_SERIE_PRECIO.values()):
            with col:
                mostrar_chart(A.grafico_precio_serie(df[df["serie"] == nombre], nombre))
        st.caption("Línea negra: precio real histórico · azul discontinua: pronóstico hecho con datos hasta 2018 · banda: "
                   "intervalo del 80% (se queda corta desde el cuarto año) · puntos verdes: lo que realmente ocurrió. "
                   "Dólares de 2019 por kg. Es un precio internacional por tipo de café: no existe pronóstico por país.")
    if productor and (sin_pais or (res["modo"] != "llm" and not res["paises"] and A.es_pregunta_de_precio(res["pregunta"], []))):
        args = sin_pais[-1] if sin_pais else {}
        try:
            desde, hasta, n = args.get("desde"), args.get("hasta"), int(args.get("n", 5))
        except (TypeError, ValueError):
            desde, hasta, n = None, None, 5
        rk = A.ranking_precio_productor(T, desde, hasta, n, args.get("tipo"))
        if rk is not None:
            at, p = rk.attrs, rk.attrs["periodo"]
            st.markdown(f"**Precio real al productor por país, {p['desde']} → {p['hasta']}**"
                        + (" (período por defecto: el reciente con más países)" if at["por_defecto"] else ""))
            tipos = list(dict.fromkeys(rk["tipo"]))
            for col, tp in zip(st.columns(len(tipos)), tipos):
                with col:
                    st.markdown(f"*{tp}*")
                    mostrar_chart(A.grafico_variacion_precio(rk[rk["tipo"] == tp]))
            cob = " · ".join(f"{tp}: {v['n_paises_con_dato']} países en ambos años ({v['n_paises_con_dato_en_desde']} en {p['desde']}, "
                             f"{v['n_paises_con_dato_en_hasta']} en {p['hasta']}; {v['n_paises_con_algun_precio']} en algún año)"
                             for tp, v in at["por_tipo"].items())
            fuera = (f" Mercados clave que no se pueden comparar en este período: {', '.join(at['fuera'])}." if at["fuera"] else "")
            st.caption(f"{cob}.{fuera} Arábica y Robusta no son comparables entre sí. Muchos países dejan de reportar con el "
                       f"tiempo. Gris: productor pequeño (menos de {L.UMBRAL_PRODUCTOR_PEQUENO_M_SACOS} M de sacos): series ruidosas. "
                       "Es histórico: no hay pronóstico por país.")
        else:
            st.info("Muy pocos países reportan precio al productor en ese período.")
    if productor and res["paises"]:
        mencionados = res["paises"][:3]
        sp = A.serie_precio_productor(mencionados, T)
        if len(sp):
            st.markdown("**Precio real al productor de los países mencionados (años con dato)**")
            mostrar_chart(A.grafico_precio_productor(sp))
        for p in mencionados:
            st.caption(f"**{p}:** {A.resumen_precio_pais(p, T)}.")
        if len(sp):
            st.caption("Solo se muestran los años que tienen dato: el precio de cada país termina en su último año con dato "
                       "y no se extrapola. Dólares de 2019 por kg. Arábica y Robusta no son comparables entre sí.")


def mostrar(res: dict) -> None:
    if res["aviso"]:
        st.warning(res["aviso"])
    st.subheader("Resumen")
    st.markdown(res["texto"])
    if res["modo"] == "llm":
        if res["verificada"]:
            usadas = ", ".join(dict.fromkeys(res["herramientas"])) or "ninguna"
            st.caption(f"✅ Cifras verificadas contra los datos · herramientas consultadas: {usadas} · modelo {MODEL}")
        else:
            st.warning("⚠️ Algunas cifras de este texto no pudieron verificarse contra los datos; "
                       "contrástalas con las fichas de abajo. " +
                       " ".join(p["detalle"] for p in res["problemas"]))
    else:
        st.caption("Resumen determinístico (plantilla verificada, sin LLM). " +
                   ("" if res["paises"] else "Sin LLM no interpreto la pregunta: muestro el resumen del portafolio."))
    if res["modo"] != "llm" and A.es_pregunta_de_precio(res["pregunta"], res["herramientas"]):
        st.info("Sin LLM no puedo responder preguntas de precios en texto; abajo están los gráficos de precios.")
    for p in res["excluidos"]:
        st.warning(f"**{p}** aparece en el dataset original pero no se modeló (serie sin datos suficientes).")
    if res["paises"]:
        st.divider()
        st.subheader("Cluster y ranking")
        for p in res["paises"][:3]:
            ficha(p)
        if len(res["paises"]) > 3:
            st.caption(f"Mencionaste {len(res['paises'])} países; se muestran los 3 primeros.")
    if A.es_pregunta_de_precio(res["pregunta"], res["herramientas"]):
        st.divider()
        precios(res)
    st.divider()
    st.subheader("Gráficos")
    graficos(res["paises"])


# ------------------------------------------------------------------ interfaz
with st.sidebar:
    st.header("Estado")
    if client is not None:
        st.success(f"LLM activo · {MODEL}")
    else:
        st.info("Sin API key: resumen determinístico (sin LLM). Define `ANTHROPIC_API_KEY` en el `.env` para activar el agente.")
    st.caption(f"llm_utils {L.CODE_VERSION}")
    st.header("Ejemplos")
    for i, ej in enumerate(A.EJEMPLOS):
        boton_ancho(ej, key=f"ej{i}", on_click=elegir_ejemplo, args=(ej,))

st.title("☕ Asistente de mercados de café")
st.caption("Pregunta por un país o por el portafolio: verás el resumen, su cluster (segmento), su posición en el "
           "ranking y los gráficos. Escribe los países en español o en inglés.")

st.session_state.setdefault("q", "")
with st.form("formulario"):
    st.text_input("Tu pregunta", key="q", placeholder="Ej.: ¿Cómo va Vietnam y qué riesgo tiene?")
    enviar = st.form_submit_button("Preguntar", type="primary")

disparar = st.session_state.pop("auto", False)
if enviar or disparar:
    pregunta = st.session_state.get("q", "").strip()
    if pregunta:
        st.session_state["resultado"] = responder(pregunta)
    else:
        st.warning("Escribe una pregunta o elige un ejemplo de la barra lateral.")

if st.session_state.get("resultado"):
    mostrar(st.session_state["resultado"])
else:
    st.info("Escribe una pregunta arriba o elige un ejemplo de la barra lateral.")

st.caption("Las cifras salen de los notebooks 01-04 (consumo) y 06-07 (precios). Las fichas y los gráficos se calculan directamente de los datos; "
           "solo el texto del resumen lo redacta el LLM, y sus cifras se verifican contra los datos.")
