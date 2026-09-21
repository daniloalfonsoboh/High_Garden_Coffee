"""Pruebas de la app (funciones de datos y flujo completo con streamlit.testing).

No requieren API key ni red. La ruta con LLM usa el SDK real contra un servidor simulado (como test_llm_utils).
Requieren haber corrido los notebooks 01-04. Ejecutar: python test_app.py   (o pytest test_app.py)
"""
from __future__ import annotations

import traceback

import pandas as pd

import app_utils as A
import llm_utils as L

_T = None


def T():
    global _T
    if _T is None:
        _T = L.load_tables()
    return _T


# ------------------------------------------------------------------ detección de países
def test_detects_spanish_and_english_names_with_or_without_accents():
    assert A.detectar_paises("¿Cómo va Vietnam?", T()) == (["Vietnam"], [])
    assert A.detectar_paises("Compara Brasil, México y Etiopía", T())[0] == ["Brazil", "Mexico", "Ethiopia"]
    assert A.detectar_paises("compara COLOMBIA y mexico", T())[0] == ["Colombia", "Mexico"]
    assert A.detectar_paises("Filipinas y Tailandia", T())[0] == ["Philippines", "Thailand"]
    assert A.detectar_paises("Costa de Marfil", T())[0] == ["Côte d'Ivoire"]


def test_detection_does_not_confuse_similar_names():
    assert A.detectar_paises("India o Indonesia", T())[0] == ["India", "Indonesia"]
    assert A.detectar_paises("República Democrática del Congo", T())[0] == ["RD Congo"]
    assert A.detectar_paises("República del Congo", T())[0] == ["Congo"]
    assert A.detectar_paises("Papúa Nueva Guinea", T())[0] == ["Papua New Guinea"]
    assert A.detectar_paises("Guinea", T())[0] == ["Guinea"]
    assert A.detectar_paises("Guinea Ecuatorial y Guinea", T()) == (["Guinea"], ["Equatorial Guinea"])


def test_detection_flags_unmodeled_and_ignores_no_country():
    assert A.detectar_paises("¿Y Nepal?", T()) == ([], ["Nepal"])
    assert A.detectar_paises("Zambia", T()) == ([], ["Zambia"])
    assert A.detectar_paises("Dame un resumen del portafolio", T()) == ([], [])
    assert A.detectar_paises("¿Cuánto importa Alemania?", T()) == ([], [])       # no está en el dataset


def test_detection_keeps_order_of_appearance_without_duplicates():
    assert A.detectar_paises("Vietnam, Bolivia, otra vez Vietnam y Peru", T())[0] == ["Vietnam", "Bolivia", "Peru"]


def test_every_modeled_country_is_detectable_by_its_own_name():
    for p in T()["rk"].index:
        assert p in A.detectar_paises(f"háblame de {p}", T())[0], p


# ------------------------------------------------------------------ datos de fichas y gráficos
def test_country_sheet_has_cluster_ranking_and_definitions():
    f = A.ficha_pais("Vietnam", T())
    assert f["segmento"] and f["posicion_ranking"] >= 1 and f["n_paises_ranking"] == len(T()["rk"])
    assert f["definicion_nivel"] and f["definicion_segmento"]


def test_series_are_continuous_between_history_and_projection():
    df = A.serie_pais("Vietnam", T())
    hist, proy = df[df.tipo == "Histórico"], df[df.tipo == "Proyección"]
    assert proy.iloc[0].valor == hist.iloc[-1].valor and proy.iloc[0].anio == hist.iloc[-1].anio
    assert len(proy) == 6                                                     # último dato real + 5 temporadas
    assert (proy.bajo <= proy.valor + 1e-9).all() and (proy.alto >= proy.valor - 1e-9).all()


def test_total_series_matches_portfolio_totals():
    df = A.serie_total(T())
    assert abs(df[df.tipo == "Proyección"].valor.iloc[-1] - T()["rk"]["y24"].sum()) < 1e-6


def test_ranking_chart_includes_mentioned_country_outside_top():
    d = A.ranking_grafico(["Kenya"], T(), top=15)
    assert "Kenya" in d.pais.tolist() and d[d.pais == "Kenya"].seleccionado.iloc[0]
    assert d.seleccionado.sum() == 1 and d.posicion.is_monotonic_increasing
    assert not A.ranking_grafico([], T()).seleccionado.any()


def test_segments_chart_covers_all_modeled_countries():
    d = A.segmentos_grafico(T())
    assert len(d) == len(T()["rk"]) and d.segmento.notna().all() and (d.consumo_2019 > 0).all()


def test_charts_build_without_errors():
    A.grafico_serie(A.serie_pais("Bolivia", T()), "Bolivia").to_dict()
    A.grafico_serie(A.serie_total(T()), "Total").to_dict()
    A.grafico_ranking(A.ranking_grafico(["Vietnam"], T())).to_dict()
    A.grafico_segmentos(A.segmentos_grafico(T()), ["Vietnam"]).to_dict()
    A.grafico_segmentos(A.segmentos_grafico(T())).to_dict()


def test_segments_chart_legend_and_axis_are_readable():
    """Leyenda recortada y eje X hasta 10,000 (el mayor mercado es ~1,300) en la primera versión."""
    import json
    d = A.segmentos_grafico(T())
    spec = json.dumps(A.grafico_segmentos(d).to_dict())
    assert '"orient": "bottom"' in spec and '"labelLimit": 320' in spec
    dom = [c for c in A.grafico_segmentos(d).to_dict()["layer"][0]["encoding"]["x"]["scale"]["domain"]]
    assert dom[0] < d.consumo_2019.min() and d.consumo_2019.max() < dom[1] < 10 * d.consumo_2019.max()


def test_offline_summary_for_country_and_portfolio():
    assert "Vietnam" in A.resumen_sin_llm(["Vietnam"], T())
    txt = A.resumen_sin_llm(["Vietnam", "Bolivia", "Peru", "Kenya"], T(), max_paises=3)
    assert txt.count("·") >= 3 and "Kenya" not in txt                        # respeta el máximo
    assert "prioritarios" in A.resumen_sin_llm([], T())


# ------------------------------------------------------------------ flujo completo de la app
def _app(cliente=None):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    st.cache_resource.clear()
    L.get_client = lambda: cliente
    return AppTest.from_file("app.py", default_timeout=90).run()


def _preguntar(at, q):
    at.text_input(key="q").set_value(q)
    [b for b in at.button if b.label == "Preguntar"][0].click().run()
    return at


def _charts(at):
    return len(at.get("arrow_vega_lite_chart")) + len(at.get("vega_lite_chart"))


def test_app_offline_country_question_shows_cluster_ranking_and_charts():
    original = L.get_client
    try:
        at = _preguntar(_app(None), "¿Cómo va Vietnam y qué riesgo tiene?")
        assert not at.exception, [e.value for e in at.exception]
        etiquetas = {m.label: m.value for m in at.metric}
        assert etiquetas["Ranking"].endswith(f"de {len(T()['rk'])}") and etiquetas["Nivel"] == "Prioritario"
        assert any("Emergentes de alto crecimiento" in m.value for m in at.markdown)      # segmento (cluster)
        assert _charts(at) == 3 and len(at.tabs) == 3
        assert any("Alerta de optimismo" in w.value for w in at.warning)
        assert any("sin LLM" in c.value for c in at.caption)
    finally:
        L.get_client = original


def test_app_offline_portfolio_question_and_unmodeled_country():
    original = L.get_client
    try:
        at = _preguntar(_app(None), "Dame un resumen del portafolio")
        assert not at.exception and "Cluster y ranking" not in [s.value for s in at.subheader]
        assert _charts(at) == 3
        at = _preguntar(at, "¿Qué pasa con Nepal?")
        assert not at.exception and any("Nepal" in w.value and "no se modeló" in w.value for w in at.warning)
    finally:
        L.get_client = original


def test_app_example_button_and_empty_question():
    original = L.get_client
    try:
        at = _app(None)
        [b for b in at.button if b.label.startswith("Compara")][0].click().run()
        assert not at.exception and at.text_input(key="q").value == "Compara Colombia y México"
        assert [m.value for m in at.markdown if m.value.startswith("####")] == ["#### Colombia", "#### Mexico"]
        at = _preguntar(_app(None), "   ")
        assert not at.exception and any("Escribe una pregunta" in w.value for w in at.warning)
    finally:
        L.get_client = original


def _cliente_falso(*respuestas):
    import test_llm_utils as TL
    return TL._client(TL.FakeAPI(list(respuestas)))


def test_app_with_llm_shows_verified_answer_and_tools():
    import test_llm_utils as TL
    original = L.get_client
    try:
        f = L.country_fact_sheet("Vietnam", T())
        texto = (f"Vietnam proyecta {f['cagr_proyectado_pct']}% anual frente a {f['crecimiento_reciente_pct']}% "
                 f"reciente: la proyección puede ser optimista.")
        cli = _cliente_falso(TL._msg([TL._tool_use("perfil_pais", {"pais": "Vietnam"}, "t1")], "tool_use"),
                             TL._msg([TL._text(texto)]))
        at = _preguntar(_app(cli), "¿Por qué Vietnam tiene alerta de optimismo?")
        assert not at.exception, [e.value for e in at.exception]
        assert any(texto in m.value for m in at.markdown)                       # el texto del agente
        assert any("Cifras verificadas" in c.value and "perfil_pais" in c.value for c in at.caption)
        assert [m.label for m in at.metric][0] == "Ranking" and _charts(at) == 3   # ficha y gráficos siguen saliendo
    finally:
        L.get_client = original


def test_app_with_llm_flags_unverified_numbers():
    import test_llm_utils as TL
    original = L.get_client
    try:
        cli = _cliente_falso(TL._msg([TL._tool_use("perfil_pais", {"pais": "Vietnam"}, "t1")], "tool_use"),
                             TL._msg([TL._text("Vietnam crecerá 12.3% anual.")]))
        at = _preguntar(_app(cli), "¿Cuánto crece Vietnam?")
        assert not at.exception
        assert any("no pudieron verificarse" in w.value for w in at.warning)
    finally:
        L.get_client = original


def test_app_falls_back_when_api_fails():
    import test_llm_utils as TL
    original = L.get_client
    try:
        cli = _cliente_falso((500, {"type": "error", "error": {"type": "api_error", "message": "boom"}}))
        at = _preguntar(_app(cli), "¿Cómo va Vietnam?")
        assert not at.exception, [e.value for e in at.exception]
        assert any("No pude usar el LLM" in w.value for w in at.warning)
        assert any("Vietnam consumió" in m.value for m in at.markdown)          # resumen determinístico
        assert _charts(at) == 3
    finally:
        L.get_client = original


# ------------------------------------------------------------------ precios (notebooks 06-07)
def _T_precios():
    t = T()
    assert A.hay_precios_globales(t) and "extra" in t, "Ejecuta antes los notebooks 06 y 07"
    return t


def test_price_question_detection():
    assert A.es_pregunta_de_precio("¿En qué país subió más el precio al productor?")
    assert A.es_pregunta_de_precio("¿Cuál es la cotización del café?") and A.es_pregunta_de_precio("Precios del Arábica")
    assert A.es_pregunta_de_precio("hola", herramientas=["precios_globales"])
    assert not A.es_pregunta_de_precio("¿Cómo va Vietnam y qué riesgo tiene?")
    assert not A.es_pregunta_de_precio("Dame un resumen del portafolio", herramientas=["top_ranking"])
    assert not A.es_pregunta_de_precio("¿Qué precisión tiene el modelo?")                         # 'precisión' no es 'precio'


def test_examples_include_price_questions():
    assert any("subió más el precio al productor" in e for e in A.EJEMPLOS)
    assert any("aumentará más el precio" in e for e in A.EJEMPLOS)                                # la pregunta que debe declinarse


def test_global_price_series_are_continuous_and_match_tables():
    t = _T_precios()
    d = A.serie_precio_global(t)
    assert set(d.serie) == {"Arábica (Other Milds)", "Robusta"} and set(d.tipo) == {"Histórico", "Pronóstico", "Realizado"}
    for nombre, g in d.groupby("serie"):
        h, p, r = (g[g.tipo == x].sort_values("anio") for x in ("Histórico", "Pronóstico", "Realizado"))
        assert h.anio.max() == 2018 and p.anio.min() == 2018 and list(p.anio) == list(range(2018, 2025))
        assert p.iloc[0].valor == h.iloc[-1].valor and p.iloc[0].bajo == p.iloc[0].alto == p.iloc[0].valor      # continuidad
        assert (p.iloc[1:].bajo < p.iloc[1:].valor).all() and (p.iloc[1:].valor < p.iloc[1:].alto).all()
        assert list(r.anio) == list(range(2019, 2025))
    col = "other_milds"
    fc = t["price_fc"][(t["price_fc"].serie == col) & (t["price_fc"].anio == 2022)].iloc[0]
    fila = d[(d.serie == "Arábica (Other Milds)") & (d.tipo == "Pronóstico") & (d.anio == 2022)].iloc[0]
    assert fila.valor == fc.central and fila.bajo == fc.lo and fila.alto == fc.hi


def test_price_ranking_data_matches_what_the_agent_sees():
    t = _T_precios()
    d = A.ranking_precio_productor(t, 2013, 2018, 3)
    r = L.tool_precio_productor(t, desde=2013, hasta=2018, n=3)
    for e in r["por_tipo"]:
        s = d[d.tipo == e["tipo"]]
        assert list(s[s.grupo == "Mayores aumentos"].pais) == [x["pais"] for x in e["mayores_aumentos"]]
        assert list(s[s.grupo == "Mayores caídas"].pais) == [x["pais"] for x in e["mayores_caidas"]]
        assert d.attrs["por_tipo"][e["tipo"]]["n_paises_con_dato"] == e["n_paises_con_dato"]
        assert d.attrs["por_tipo"][e["tipo"]]["n_paises_con_algun_precio"] >= e["n_paises_con_dato"]
    assert d.attrs["n_modelados"] == len(t["rk"]) and d.attrs["fuera"] == r["mercados_clave_fuera_del_ranking"]
    assert d[d.productor_pequeno].etiqueta.str.contains("productor pequeño").all() and set(d.tipo) <= {"Arábica", "Robusta"}
    assert A.ranking_precio_productor(t, 2018, 2013) is None                                       # años inválidos: sin datos
    por_defecto = A.ranking_precio_productor(t)                                                    # sin años: ventana por defecto
    assert por_defecto.attrs["periodo"] == {"desde": L.PRODUCTOR_VENTANA_DEFECTO[0], "hasta": L.PRODUCTOR_VENTANA_DEFECTO[1]}
    cuenta = lambda x: sum(v["n_paises_con_dato"] for v in x.attrs["por_tipo"].values())
    assert por_defecto.attrs["por_defecto"] and cuenta(por_defecto) > cuenta(d)                    # 2008→2013 conserva más países
    solo = A.ranking_precio_productor(t, tipo="Robusta")
    assert set(solo.tipo) == {"Robusta"} and A.ranking_precio_productor(t, tipo="cacao") is None


def test_grower_price_series_by_type_only_for_years_with_data():
    t = _T_precios()
    d = A.serie_precio_productor(["Colombia", "Thailand", "Nepal", "Narnia"], t)
    assert set(d.pais) == {"Colombia", "Thailand"} and set(d.serie) == {"Colombia · Arábica", "Thailand · Arábica", "Thailand · Robusta"}
    assert d.precio.gt(0).all() and d.anio.between(1990, 2019).all()
    rob = d[d.serie == "Thailand · Robusta"]
    ult = next(x["ultimo_anio"] for x in L.tool_precio_productor(t, pais="Thailand")["tipos"] if x["tipo"] == "Robusta")
    assert rob.anio.is_monotonic_increasing and rob.anio.max() == ult
    assert len(A.serie_precio_productor([], t)) == 0
    viejo = dict(t); viejo["extra"] = t["extra"].drop(columns=["price_grower_arabica_real_usd_kg", "price_grower_robusta_real_usd_kg"])
    assert len(A.serie_precio_productor(["Thailand"], viejo)) == 0                                   # parquet antiguo: sin datos, sin romper


def test_country_price_summary_line_states_years_with_data():
    t = _T_precios()
    linea = A.resumen_precio_pais("Thailand", t)
    r = L.tool_precio_productor(t, pais="Thailand")
    for x in r["tipos"]:
        assert f"{x['tipo']} {x['primer_anio']}–{x['ultimo_anio']} ({x['n_anios_con_dato']} años con dato)" in linea
    assert A.resumen_precio_pais("Paraguay", t) == "sin dato de precio al productor"


def test_price_charts_build_valid_specs():
    t = _T_precios()
    d = A.serie_precio_global(t)
    A.grafico_precio_serie(d[d.serie == "Robusta"], "Robusta").to_dict()
    rk = A.ranking_precio_productor(t)
    A.grafico_variacion_precio(rk[rk.tipo == "Robusta"]).to_dict()
    A.grafico_precio_productor(A.serie_precio_productor(["Colombia", "Thailand"], t)).to_dict()


def test_app_offline_price_question_shows_price_panel_without_changing_other_sections():
    original = L.get_client
    try:
        at = _preguntar(_app(None), "¿En qué país subió más el precio al productor?")
        assert not at.exception, [e.value for e in at.exception]
        assert "Precios" in [x.value for x in at.subheader]
        assert _charts(at) == 3 + 2 + 2 and len(at.tabs) == 3                                     # 3 pestañas + 2 globales + 1 ranking por tipo
        assert any("Sin LLM no puedo responder preguntas de precios" in i.value for i in at.info)
        assert any("no existe pronóstico por país" in c.value for c in at.caption)
        assert any("países en ambos años" in c.value and "en algún año" in c.value and "no son comparables entre sí" in c.value
                   for c in at.caption)                                                              # cobertura por tipo
        assert any("período por defecto" in m.value and "→" in m.value for m in at.markdown)          # dice qué ventana usó
        assert any("Mercados clave que no se pueden comparar" in c.value for c in at.caption)      # y a quién deja fuera
        sin = _preguntar(at, "Dame un resumen del portafolio")                                     # una pregunta normal no muestra precios
        assert "Precios" not in [x.value for x in sin.subheader] and _charts(sin) == 3
    finally:
        L.get_client = original


def test_app_price_panel_follows_the_agents_tool_arguments():
    import test_llm_utils as TL
    original = L.get_client
    try:
        r = L.tool_precio_productor(T(), desde=2010, hasta=2018, n=3)
        e = next(x for x in r["por_tipo"] if x["tipo"] == "Robusta")
        texto = (f"Entre 2010 y 2018, en Robusta, el mayor aumento real fue de {e['mayores_aumentos'][0]['pais']}: "
                 f"{e['mayores_aumentos'][0]['variacion_real_pct']}%. {e['n_paises_con_dato']} de {r['n_paises_modelados']} "
                 "países tienen precio en ambos años.")
        cli = _cliente_falso(TL._msg([TL._tool_use("precio_productor", {"desde": 2010, "hasta": 2018, "n": 3}, "t1")], "tool_use"),
                             TL._msg([TL._text(texto)]))
        at = _preguntar(_app(cli), "¿En qué país subió más el precio entre 2010 y 2018?")
        assert not at.exception, [e.value for e in at.exception]
        assert any("2010 → 2018" in m.value for m in at.markdown)                                   # mismo período que pidió el agente
        assert any("Cifras verificadas" in c.value and "precio_productor" in c.value for c in at.caption)
        assert not any("Sin LLM" in i.value for i in at.info)
    finally:
        L.get_client = original


def test_app_country_price_question_shows_grower_chart():
    original = L.get_client
    try:
        at = _preguntar(_app(None), "¿Cómo va el precio en Colombia?")
        assert not at.exception, [e.value for e in at.exception]
        assert any("Precio real al productor de los países mencionados" in m.value for m in at.markdown)
        assert "Cluster y ranking" in [x.value for x in at.subheader]                               # la ficha del país sigue saliendo
    finally:
        L.get_client = original


def test_app_country_price_question_describes_years_with_data_by_type():
    """El caso pedido: preguntar por Tailandia y ver los precios de los años que tienen dato, por tipo de café."""
    original = L.get_client
    try:
        at = _preguntar(_app(None), "¿Cómo se han comportado los precios en Tailandia?")
        assert not at.exception, [e.value for e in at.exception]
        assert any("(años con dato)" in m.value for m in at.markdown)
        r = L.tool_precio_productor(T(), pais="Thailand")
        linea = A.resumen_precio_pais("Thailand", T())
        assert any(c.value.startswith("**Thailand:**") and linea in c.value for c in at.caption)     # años con dato por tipo
        assert any("no se extrapola" in c.value for c in at.caption)
        assert _charts(at) == 3 + 2 + 1                                                             # pestañas + globales + serie del país
        assert str(r["ultimo_anio_con_dato"]) in linea and r["ultimo_anio_con_dato"] < 2019        # el dato termina antes del presente
    finally:
        L.get_client = original


def test_app_country_card_shows_price_availability_even_without_a_price_question():
    original = L.get_client
    try:
        at = _preguntar(_app(None), "¿Cómo va Vietnam y qué riesgo tiene?")
        assert not at.exception, [e.value for e in at.exception]
        assert any(c.value == f"Precio al productor: {A.resumen_precio_pais('Vietnam', T())}." for c in at.caption)
        assert "Precios" not in [x.value for x in at.subheader] and _charts(at) == 3                # sin panel: no era pregunta de precios
    finally:
        L.get_client = original


def test_app_with_llm_country_price_answer_is_verified_and_shows_the_series():
    import test_llm_utils as TL
    original = L.get_client
    try:
        rob = next(x for x in L.tool_precio_productor(T(), pais="Thailand")["tipos"] if x["tipo"] == "Robusta")
        texto = (f"Con los datos disponibles ({rob['primer_anio']}–{rob['ultimo_anio']}), el Robusta de Tailandia pasó de "
                 f"{rob['precio_real_primer_anio']} a {rob['precio_real_ultimo_anio']} US$/kg ({rob['variacion_real_total_pct']}%).")
        cli = _cliente_falso(TL._msg([TL._tool_use("precio_productor", {"pais": "Tailandia"}, "t1")], "tool_use"),
                             TL._msg([TL._text(texto)]))
        at = _preguntar(_app(cli), "¿Cómo van los precios en Tailandia?")
        assert not at.exception, [e.value for e in at.exception]
        assert any(texto in m.value for m in at.markdown)
        assert any("Cifras verificadas" in c.value and "precio_productor" in c.value for c in at.caption)
        assert any("(años con dato)" in m.value for m in at.markdown)
    finally:
        L.get_client = original


def test_app_without_price_tables_shows_friendly_message():
    original_client, original_tables = L.get_client, L.load_tables
    try:
        L.load_tables = lambda: {k: v for k, v in original_tables().items() if not k.startswith("price") and k != "extra"}
        at = _preguntar(_app(None), "¿Cuál es el precio del café?")
        assert not at.exception, [e.value for e in at.exception]
        assert any("no están cargados" in i.value for i in at.info)
        assert _charts(at) == 3                                                                     # sigue mostrando los gráficos de siempre
    finally:
        L.get_client, L.load_tables = original_client, original_tables


def run_all(verbose: bool = True) -> pd.DataFrame:
    filas = []
    for name, fn in sorted((k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)):
        try:
            fn()
            filas.append({"prueba": name, "resultado": "OK", "detalle": ""})
        except Exception as e:  # noqa: BLE001
            filas.append({"prueba": name, "resultado": "FALLA", "detalle": f"{type(e).__name__}: {str(e)[:160]}"})
            if verbose:
                traceback.print_exc()
    df = pd.DataFrame(filas)
    if verbose:
        print(f"{(df.resultado == 'OK').sum()} OK · {(df.resultado == 'FALLA').sum()} fallas")
    return df


if __name__ == "__main__":
    r = run_all()
    print(r.to_string(index=False))
    raise SystemExit(int((r.resultado == "FALLA").any()))
