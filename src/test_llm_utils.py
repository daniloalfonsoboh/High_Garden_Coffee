"""Pruebas de robustez de la capa LLM (no requieren API key ni red).

Se ejecutan con `python -m pytest test_llm_utils.py` o desde el notebook 05 (`run_all()`).
Requieren haber corrido antes los notebooks 01-04 (usan outputs/).

Las pruebas del flujo con la API usan el SDK REAL de Anthropic contra un servidor simulado
(httpx.MockTransport): así se verifican la construcción de las peticiones, los reintentos,
el fallback y el manejo de errores sin gastar tokens.
"""
from __future__ import annotations

import copy
import json
import tempfile
import traceback
from pathlib import Path

import pandas as pd

import llm_utils as L

_T = None


def T():
    global _T
    if _T is None:
        _T = L.load_tables()
    return _T


def _tmp_cache():
    L.CACHE_PATH = Path(tempfile.mkdtemp()) / "cache.json"


# ------------------------------------------------------------------ servidor simulado
def _httpx():
    """El SDK reciente usa `httpx2`; los anteriores, `httpx`. Se usa el que corresponda."""
    try:
        import httpx2 as h
    except ImportError:
        import httpx as h
    return h


def _msg(content, stop_reason="end_turn"):
    return {"id": "msg_test", "type": "message", "role": "assistant", "model": "claude-sonnet-5",
            "content": content, "stop_reason": stop_reason, "stop_sequence": None,
            "usage": {"input_tokens": 1200, "output_tokens": 350}}


def _tool_use(name, inp, id_="toolu_01"):
    return {"type": "tool_use", "id": id_, "name": name, "input": inp}


def _text(t):
    return {"type": "text", "text": t}


class FakeAPI:
    """Registra las peticiones y responde con lo guionizado (repite la última si se agota)."""

    def __init__(self, respuestas):
        self.respuestas, self.requests = list(respuestas), []

    def __call__(self, request):
        h = _httpx()
        self.requests.append(json.loads(request.content))
        r = self.respuestas.pop(0) if len(self.respuestas) > 1 else self.respuestas[0]
        if isinstance(r, tuple):
            return h.Response(r[0], json=r[1])
        return h.Response(200, json=r)


def _client(api):
    import anthropic
    h = _httpx()
    return anthropic.Anthropic(api_key="test-key", max_retries=0,
                               http_client=h.Client(transport=h.MockTransport(api)))


def _spec(pais="Vietnam"):
    return L.spec_mercado(pais, T())


def _bueno(pais="Vietnam"):
    return L.template_mercado(L.country_fact_sheet(pais, T()))


def _tipos(issues):
    return {i["tipo"] for i in issues}


# ------------------------------------------------------------------ lectura del .env
def _env_case(contenido: bytes) -> dict:
    """Escribe un .env temporal, lo carga y devuelve lo leído (limpiando las variables de prueba)."""
    import os
    d = Path(tempfile.mkdtemp())
    (d / ".env").write_bytes(contenido)
    claves = ["ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "CLAUDE_MODEL"]
    previas = {k: os.environ.pop(k, None) for k in claves}
    try:
        L.load_env(d / ".env")
        return {k: os.environ.get(k) for k in claves}
    finally:
        for k in claves:
            os.environ.pop(k, None)
            if previas[k] is not None:
                os.environ[k] = previas[k]


def test_load_env_variants():
    esperado = {"ANTHROPIC_API_KEY": "sk-test-123", "ANTHROPIC_BASE_URL": "https://x.example",
                "CLAUDE_MODEL": None}
    texto = 'ANTHROPIC_API_KEY=sk-test-123\nANTHROPIC_BASE_URL="https://x.example"\n# CLAUDE_MODEL=otro\n'
    variantes = {
        "utf8": texto.encode("utf-8"),
        "utf8_bom": texto.encode("utf-8-sig"),                       # Bloc de notas
        "utf16": texto.encode("utf-16"),                             # PowerShell: echo ... > .env
        "crlf": texto.replace("\n", "\r\n").encode("utf-8"),
        "export_y_espacios": ("export ANTHROPIC_API_KEY = sk-test-123 \nANTHROPIC_BASE_URL='https://x.example'\n").encode(),
    }
    for nombre, contenido in variantes.items():
        assert _env_case(contenido) == esperado, nombre


def test_load_env_empty_variable_is_overridden():
    import os
    d = Path(tempfile.mkdtemp())
    (d / ".env").write_text("ANTHROPIC_API_KEY=sk-nueva\n", encoding="utf-8")
    previa = os.environ.get("ANTHROPIC_API_KEY")
    os.environ["ANTHROPIC_API_KEY"] = ""
    try:
        L.load_env(d / ".env")
        assert os.environ["ANTHROPIC_API_KEY"] == "sk-nueva"
    finally:
        os.environ.pop("ANTHROPIC_API_KEY", None)
        if previa is not None:
            os.environ["ANTHROPIC_API_KEY"] = previa


def test_diagnostico_runs(capsys=None):
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        L.diagnostico()
    assert "Carpeta donde busca el .env" in buf.getvalue()


# ------------------------------------------------------------------ verificación numérica
def test_extract_numbers_formats():
    ok = L.unsupported_numbers(["Crece 6,7% y 1.320 M; antes 1,466.3; caída de −5.8; en 2024/25 y 2019; el 80%."],
                               [6.7, 1320, 1466.3, 5.8, 80])
    assert ok == [], ok


def test_extract_ignores_seasons_and_years():
    assert L.extract_numbers("Temporada 2019/20, año 2024 y 1990/91") == []


def test_unsupported_number_detected():
    assert L.unsupported_numbers(["Crecerá 25.7%"], [6.7, 3.7]) == ["25.7"]


def test_rounding_tolerance():
    assert L.unsupported_numbers(["unos 220 M"], [220.2]) == []           # redondeo razonable
    assert L.unsupported_numbers(["23 M adicionales"], [23.4]) == []      # entero por cifra >= 20
    assert L.unsupported_numbers(["≈7%"], [6.7]) == ["7"]                 # cifra pequeña: cambia el sentido
    assert L.unsupported_numbers(["25 M"], [23.4]) == ["25"]              # cifra distinta: se rechaza


def test_fact_sheet_has_no_negative_zero():
    import math
    for p in ["Bolivia", "Vietnam", "Kenya"]:
        for k, v in L.country_fact_sheet(p, T()).items():
            if isinstance(v, float):
                assert not (v == 0 and math.copysign(1, v) < 0), (p, k)


# ------------------------------------------------------------------ validador de negocio
def test_templates_valid_for_all_countries():
    malas = {}
    for p in T()["rk"].index:
        sp = L.spec_mercado(p, T())
        iss = sp["validator"](sp["template"]())
        if iss:
            malas[p] = iss
    assert not malas, malas
    sp = L.spec_portafolio(T())
    assert sp["validator"](sp["template"]()) == []


def _natural(titular, resumen, fort, riesgos, rec, just, cifras):
    return {"titular": titular, "resumen": resumen, "fortalezas": fort, "riesgos": riesgos,
            "recomendacion": rec, "justificacion": just,
            "cifras_citadas": [{"campo": c, "valor": v} for c, v in cifras]}


def test_validator_accepts_natural_llm_style_outputs():
    """Regresión contra falsos positivos: textos como los redactaría un LLM (comas decimales, '3,7 %',
    nombres en español, redondeos naturales). Si el verificador los rechazara, el LLM caería siempre
    a la plantilla y el componente no serviría."""
    muestras = {
        "Vietnam": _natural(
            "Vietnam: mercado prioritario de gran potencial, con una proyección que conviene validar.",
            "Vietnam consumió 159 M en 2019/20, un 5,3 % del total modelado, y la proyección lo sitúa en 220,2 M "
            "en 2024/25, con un rango del 80 % entre 179,7 y 247,3 M. Ocupa la posición 3 de 51 con un score de "
            "78,6 y aparece en el top 10 en el 85 % de los escenarios de pesos simulados.",
            ["Crecimiento proyectado de 6,7 % anual, unos 61 M adicionales hasta 2024/25.",
             "Ranking robusto: entra en el top 10 en el 85 % de los escenarios."],
            ["La proyección de 6,7 % anual supera en 3 puntos porcentuales su ritmo reciente de 3,7 %, por lo "
             "que puede ser optimista.",
             "El error de backtest del método en este país fue de 11 %."],
            "Validar antes de invertir",
            "Es un nivel Prioritario, pero la brecha con el ritmo reciente aconseja validar antes de comprometer inversión.",
            [("cagr_proyectado_pct", 6.7), ("crecimiento_reciente_pct", 3.7), ("proyeccion_2024_25_M", 220.2),
             ("posicion_ranking", 3)]),
        "Bolivia": _natural(
            "Bolivia: nicho prometedor con crecimiento parejo, pero de escala muy pequeña.",
            "Bolivia consumió 3,7 M en 2019/20, solo el 0,1 % del total, y se proyecta en 4,2 M para 2024/25 "
            "(rango del 80 %: 3,4 a 4,7 M). Crece 2,6 % anual, igual que en las últimas 5 temporadas. Está en "
            "el top 10 en el 84 % de los escenarios y ocupa la posición 9.",
            ["Crecimiento estable: 2,6 % anual proyectado, igual al ritmo reciente.",
             "El error de backtest es de solo 0,9 %."],
            ["Es un mercado muy pequeño, apenas 0,1 % del consumo total: la oportunidad es de nicho y no de volumen."],
            "Explorar como nicho",
            "Su crecimiento es robusto, pero su escala reducida lo deja como oportunidad de nicho.",
            [("cagr_proyectado_pct", 2.6), ("cuota_consumo_total_pct", 0.1), ("posicion_ranking", 9)]),
        "Kenya": _natural(
            "Kenya: serie plana con poca información; sin prioridad en el ranking.",
            "Kenya consumió 4,3 M en 2019/20 y la proyección para 2024/25 es de 4,3 M, sin crecimiento. Ocupa la "
            "posición 23 de 51 y aparece en el top 10 solo en el 5 % de los escenarios.",
            ["Su consumo reciente crece 5,2 % anual, aunque la proyección no lo refleja."],
            ["La serie es plana (83 % de los años sin cambio) y probablemente son estimaciones, por lo que la "
             "proyección aporta poca información.",
             "La proyección de 0 % anual contrasta con un ritmo reciente de 5,2 %."],
            "Monitorear",
            "Con una serie plana y un score bajo, conviene seguirlo sin priorizarlo.",
            [("pct_anios_planos", 83), ("posicion_ranking", 23), ("crecimiento_reciente_pct", 5.2)]),
        "Thailand": _natural(
            "Tailandia: candidato con crecimiento proyectado por encima de su ritmo reciente.",
            "Tailandia consumió 84 M en 2019/20 (2,8 % del total) y la proyección apunta a 107,4 M en 2024/25, "
            "con un rango del 80 % entre 87,6 y 120,6 M. Ocupa la posición 7, con un score de 65,6, y está en "
            "el top 10 en el 63 % de los escenarios de pesos.",
            ["Aporta unos 23 M de consumo adicional proyectado."],
            ["La proyección de 5 % anual supera en 3,2 puntos porcentuales el ritmo reciente de 1,9 %: es "
             "posible que sea optimista.",
             "El error de backtest es de 13,2 %."],
            "Validar antes de invertir",
            "Está por debajo del umbral de 70 % de escenarios para ser prioritario y su proyección es optimista.",
            [("cagr_proyectado_pct", 5.0), ("brecha_proyeccion_vs_reciente_pp", 3.2), ("posicion_ranking", 7)]),
    }
    for pais, out in muestras.items():
        iss = L.spec_mercado(pais, T())["validator"](out)
        assert iss == [], (pais, iss)


# ---- regresiones de la primera corrida real (0/12 válidos al primer intento)
def test_regression_nested_field_citations_accepted():
    """El modelo citó 'modelo.wape_pct' y 'segmentos.<seg>.pct_consumo_2019_20': son cifras reales de la
    ficha. Antes el verificador solo aceptaba campos de primer nivel y las rechazaba."""
    sp = L.spec_portafolio(T())
    fs = sp["fs"]
    out = sp["template"]()
    seg = "Grandes en crecimiento sostenido"
    out["cifras_citadas"] += [
        {"campo": "modelo.wape_pct", "valor": fs["modelo"]["wape_pct"]},
        {"campo": f"segmentos.{seg}.pct_consumo_2019_20", "valor": fs["segmentos"][seg]["pct_consumo_2019_20"]},
        {"campo": f"segmentos.{seg}.crecimiento_proyectado_M", "valor": fs["segmentos"][seg]["crecimiento_proyectado_M"]},
    ]
    assert sp["validator"](out) == []
    out["cifras_citadas"][-1]["valor"] = 1.0                      # valor incorrecto: sigue detectándose
    assert "cifras_citadas" in _tipos(sp["validator"](out))
    out["cifras_citadas"][-1] = {"campo": "modelo.no_existe", "valor": 1}
    assert "cifras_citadas" in _tipos(sp["validator"](out))


def test_regression_score_scale_is_not_a_fabricated_number():
    """'84,7 sobre 100' marcaba '100' como cifra inventada (es la escala del score)."""
    assert L.unsupported_numbers(["Score de 84,7 sobre 100."], [84.7]) == []
    assert L.unsupported_numbers(["Score de 84.7/100."], [84.7]) == []
    # pero una afirmación real con 100 sí se verifica
    assert L.unsupported_numbers(["Está en el 100 % de los escenarios."], [84.7, 85.0]) == ["100"]
    assert L.unsupported_numbers(["Aporta sobre 100 M adicionales."], [84.7]) == ["100"]


def test_regression_thousands_separated_by_space():
    """El modelo escribió '1 466.3 M' y probablemente '5 000 escenarios': se partían en '1'+'466.3' y '5'+'000'."""
    assert L.unsupported_numbers(["Brazil proyecta 1 466.3 M."], [1466.3]) == []
    assert L.unsupported_numbers(["Se simularon 5 000 escenarios."], [5000]) == []
    assert L.unsupported_numbers(["Se simularon 5\u00a0000 escenarios y 2\u202f998.9 M."], [5000, 2998.9]) == []
    assert L.unsupported_numbers(["Proyecta 1 234 M."], [1466.3, 5000]) == ["1 234"]     # sigue detectando lo falso
    # no une números que son distintos
    assert L.unsupported_numbers(["posición 3 de 51; score 84.7 y 78.7"], [3, 51, 84.7, 78.7]) == []
    assert [t for t, _ in L.extract_numbers("los 3 mercados 84.7")] == ["3", "84.7"]


def test_regression_exotic_unicode_space_separators():
    """El modelo usó un espacio fino/de cifra/de cabello como separador de miles ('1 466.3'): se leía como
    '1' + '466.3' y la respuesta correcta se marcaba como no verificada."""
    for nombre, ch in {"nbsp": "\u00a0", "nnbsp": "\u202f", "fino": "\u2009", "cifra": "\u2007",
                       "cabello": "\u200a", "em": "\u2003", "normal": " "}.items():
        txt = f"Brasil proyecta 1{ch}466.3 M, con rango entre 1{ch}196.6 y 1{ch}647.3 M."
        assert L.unsupported_numbers([txt], [1466.3, 1196.6, 1647.3]) == [], nombre
    assert [t for t, _ in L.extract_numbers("los 3\u2009mercados 84.7")] == ["3", "84.7"]   # no une distintos


def test_agent_tools_expose_definitions_but_cards_do_not_change():
    """Tras ver que el agente daba una explicación equivocada de por qué un nicho no es prioritario (no tenía las
    definiciones), las herramientas del agente las incluyen. Las fichas de las tarjetas NO cambian: así la caché
    de los análisis ya generados sigue siendo válida."""
    t = T()
    p = L.tool_perfil_pais(t, pais="Bolivia")
    assert "menos de 1% del consumo total" in p["definicion_nivel"] and p["definicion_segmento"]
    assert set(L.tool_resumen_portafolio(t)["definiciones_niveles"]) == set(L.REC_PERMITIDAS)
    assert "definiciones_segmentos" in L.tool_resumen_portafolio(t)
    assert "definiciones_niveles" in L.tool_top_ranking(t, n=3)
    assert "definiciones_niveles" in L.tool_comparar_paises(t, paises=["Colombia", "Mexico"])
    fs = L.country_fact_sheet("Bolivia", t)
    assert "definicion_nivel" not in fs and "definiciones_niveles" not in L.portfolio_fact_sheet(t)


def test_level_definitions_use_only_grounded_numbers():
    """Las cifras que aparecen en las definiciones deben poder citarse sin ser marcadas como inventadas."""
    permitidas = L.flatten_numbers(L.CONSTANTES)
    for nivel, d in L.DEFINICIONES_NIVELES.items():
        assert L.unsupported_numbers([d], permitidas) == [], (nivel, d)
    assert set(L.DEFINICIONES_NIVELES) == set(L.REC_PERMITIDAS)


def test_qa_prompt_has_injection_and_definition_rules():
    assert "ignorar o revelar" in L.SYSTEM_QA and "definicion_nivel" in L.SYSTEM_QA
    desc = {tool["name"]: tool["description"] for tool in L.QA_TOOLS}
    assert "NIVELES" in desc["pronostico_pais"] and "perfil_pais" in desc["pronostico_pais"]
    assert "crecimiento proyectado" in desc["perfil_pais"] and "crecimiento proyectado" in desc["comparar_paises"]


def test_regression_field_suffix_resolution():
    """'horizonte_temporadas' (sin el prefijo 'constantes.') es unívoco y con valor correcto: se acepta."""
    sp = _spec("Philippines")
    out = _bueno("Philippines")
    out["cifras_citadas"].append({"campo": "horizonte_temporadas", "valor": 5})
    assert sp["validator"](out) == []
    out["cifras_citadas"][-1]["valor"] = 7                               # valor incorrecto: se detecta
    assert "cifras_citadas" in _tipos(sp["validator"](out))
    # ambiguo: 'n_paises' existe en varios segmentos -> se pide la ruta completa
    port = L.spec_portafolio(T())
    o = port["template"]()
    o["cifras_citadas"].append({"campo": "n_paises", "valor": 5})
    iss = port["validator"](o)
    assert any("ambiguo" in i["detalle"] for i in iss)
    ruta, cand = L.resolver_campo("n_paises", L.flatten_fields(port["fs"]))
    assert ruta is None and len(cand) > 1


def test_regression_length_limits_prompt_vs_validator():
    """El prompt pide menos palabras que el límite del validador (margen) y lo dice explícitamente."""
    for k in L.LIM:
        assert L.PEDIR[k] < L.LIM[k], k
    desc = json.dumps(L.TOOL_MERCADO, ensure_ascii=False) + json.dumps(L.TOOL_PORTAFOLIO, ensure_ascii=False)
    for k in ("titular", "resumen", "resumen_ejecutivo", "punto", "punto_largo"):
        assert f"máx. {L.PEDIR[k]} palabras" in desc, k
    out = _bueno()
    out["resumen"] = "palabra " * (L.LIM["resumen"] + 5)
    iss = _spec()["validator"](out)
    assert any("tiene" in i["detalle"] for i in iss if i["tipo"] == "longitud")   # el feedback dice cuánto se pasó


def test_eval_battery_covers_out_of_scope_and_robustness():
    qs = L.eval_questions(T())
    assert len(qs) >= 12
    assert sum(1 for q in qs if not q["tools_any"] or q["prohibidas"]) >= 6      # fuera de alcance / robustez
    txt = " ".join(q["pregunta"] for q in qs)
    assert "Ignora tus instrucciones" in txt and "Suma" in txt


def test_regression_niche_question_accepts_scale_phrasing():
    """Una respuesta correcta que explica la escala como 'menos del 1 % del consumo total' debe aprobar."""
    original = L.preguntar
    texto = ("Los nichos prometedores son Bolivia y Uganda. No son prioritarios porque cada uno representa "
             "menos del 1 % del consumo total.")

    def falso(client, pregunta, T_, model=None, **kw):
        return {"respuesta": texto, "herramientas": [{"herramienta": "top_ranking", "argumentos": {}}],
                "verificada": True, "problemas": [], "reintentos": 0}
    L.preguntar = falso
    try:
        df = L.evaluar_qa(None, T(), workers=1)
    finally:
        L.preguntar = original
    fila = df[df.pregunta.str.contains("nichos prometedores")].iloc[0]
    assert fila.aprobada, fila


def test_regression_niche_check_rejects_wrong_explanation():
    """Una respuesta real del agente decía que Bolivia y Uganda no son prioritarios porque 'ese umbral lo define
    el análisis por concentración de consumo: los 7 prioritarios reúnen el 82 %'. Es una explicación equivocada
    (la razón es que pesan menos del 1 % del consumo). Con las palabras 'umbral' y 'consumo total' aprobaba."""
    incorrecta = ("Los nichos prometedores son Bolivia y Uganda. No alcanzan nivel Prioritario porque ese umbral "
                  "lo define el análisis por concentración de consumo: los 7 prioritarios reúnen el 82.0% del "
                  "consumo total del portafolio.")
    vaga = ("Los nichos son Bolivia y Uganda; no alcanzan ese umbral de materialidad. El análisis no detalla "
            "otras razones.")
    correcta = ("Los nichos prometedores son Bolivia y Uganda: son igual de robustos que un prioritario, pero "
                "pesan menos de 1% del consumo total.")
    correcta_real = ("Ambos son tan robustos como un Prioritario. La diferencia es que no alcanzan el 1% del consumo "
                     "total del portafolio, por lo que representan oportunidades de nicho, no de volumen.")
    trampa = "Los nichos son Bolivia y Uganda; representan el 21% de algo y el 82.0% de otra cosa."
    original = L.preguntar
    resultados = {}
    casos = {"incorrecta": incorrecta, "vaga": vaga, "correcta": correcta, "correcta_real": correcta_real,
             "trampa_21": trampa}
    for nombre, texto in casos.items():
        def falso(client, pregunta, T_, model=None, _t=texto, **kw):
            return {"respuesta": _t, "herramientas": [{"herramienta": "top_ranking", "argumentos": {}}],
                    "verificada": True, "problemas": [], "reintentos": 0}
        L.preguntar = falso
        try:
            df = L.evaluar_qa(None, T(), workers=1)
        finally:
            L.preguntar = original
        resultados[nombre] = bool(df[df.pregunta.str.contains("nichos prometedores")].iloc[0].aprobada)
    assert resultados == {"incorrecta": False, "vaga": False, "correcta": True, "correcta_real": True,
                          "trampa_21": False}, resultados


def test_regression_priority_with_alert_cannot_be_prioritize():
    """Un LLM recomendó 'Priorizar' a Vietnam diciendo que su tamaño 'compensa la alerta de optimismo'."""
    fs = L.country_fact_sheet("Vietnam", T())
    assert fs["nivel"] == "Prioritario" and fs["alerta_optimismo"]
    assert L.recomendaciones_permitidas(fs) == {"Validar antes de invertir"}
    out = _bueno("Vietnam")
    out["recomendacion"] = "Priorizar"
    iss = _spec("Vietnam")["validator"](out)
    assert "recomendacion" in _tipos(iss) and any("alerta" in i["detalle"] for i in iss)
    out["recomendacion"] = "Validar antes de invertir"
    assert _spec("Vietnam")["validator"](out) == []
    brazil = L.country_fact_sheet("Brazil", T())                      # sin alerta: 'Priorizar' sigue permitido
    assert "Priorizar" in L.recomendaciones_permitidas(brazil)


def test_prompt_glossary_prevents_misreadings():
    """Interpretaciones erróneas observadas: 'brecha favorable', 'alta probabilidad de top 10', dirección del backtest."""
    p = L.SYSTEM_MERCADO
    assert "RIESGO, nunca una fortaleza" in p and "robustez del ranking" in p
    assert "no le atribuyas dirección" in p and "NO es una probabilidad" in p
    assert L.PROMPT_VERSION == "v3"


def test_stale_cache_is_revalidated_with_current_rules():
    """Un análisis guardado con reglas anteriores (p. ej. 'Priorizar' con alerta) no debe reutilizarse."""
    _tmp_cache()
    sp = _spec("Vietnam")
    viejo = _bueno("Vietnam")
    viejo["recomendacion"] = "Priorizar"                              # válido con las reglas antiguas
    clave = L.cache_key("claude-sonnet-5", "mercado", sp["fs"])
    L.CACHE_PATH.write_text(json.dumps({clave: {"analisis": viejo, "validacion_ok": True, "problemas_finales": [],
                                                "intentos": []}}), encoding="utf-8")
    api = FakeAPI([_msg([_tool_use("registrar_analisis_mercado", _bueno("Vietnam"))], "tool_use")])
    res = L.generate_validated(_client(api), "claude-sonnet-5", sp)
    assert res["fuente"] == "llm" and len(api.requests) == 1          # se descartó la caché y se regeneró
    assert res["cache_descartada"] and res["analisis"]["recomendacion"] == "Validar antes de invertir"
    # sin cliente (offline) tampoco devuelve el análisis inválido: cae a la plantilla
    L.CACHE_PATH.write_text(json.dumps({clave: {"analisis": viejo, "validacion_ok": True, "problemas_finales": [],
                                                "intentos": []}}), encoding="utf-8")
    off = L.generate_validated(None, "claude-sonnet-5", sp)
    assert off["fuente"] == "plantilla" and off["validacion_ok"]


def test_regression_agent_must_not_offer_data_of_countries_outside_dataset():
    """En las 3 corridas, ante '¿Cuánto café importa Alemania?' el agente no consultó herramientas y ofreció
    'mostrarte su consumo doméstico, proyección y score' (Alemania no está en el dataset). La batería lo aprobaba."""
    mala = ("El análisis cubre consumo doméstico de café, no importaciones. Los datos de importaciones no figuran en "
            "este análisis. Lo que sí puedo mostrarte sobre Alemania es su consumo doméstico, proyección a 2024/25, "
            "nivel de oportunidad y score. ¿Te interesa esa información?")
    buena = ("Alemania no figura en el análisis, que cubre 51 países productores y no incluye importaciones. Puedo "
             "mostrarte el ranking de los países que sí están modelados.")
    original = L.preguntar
    res = {}
    for nombre, texto, tools in [("mala_sin_tools", mala, []), ("mala_con_tools", mala, ["perfil_pais"]),
                                 ("buena", buena, ["perfil_pais"]), ("buena_sin_tools", buena, [])]:
        def falso(client, pregunta, T_, model=None, _t=texto, _h=tools, **kw):
            return {"respuesta": _t, "herramientas": [{"herramienta": h, "argumentos": {}} for h in _h],
                    "verificada": True, "problemas": [], "reintentos": 0}
        L.preguntar = falso
        try:
            df = L.evaluar_qa(None, T(), workers=1)
        finally:
            L.preguntar = original
        res[nombre] = bool(df[df.pregunta.str.contains("Alemania")].iloc[0].aprobada)
    assert res == {"mala_sin_tools": False, "mala_con_tools": False, "buena": True, "buena_sin_tools": False}, res


def test_portfolio_prompt_glossary_and_qa_country_rule():
    assert "No es \"lo ocurrido en las últimas 5 temporadas\"" in L.SYSTEM_PORTAFOLIO
    assert "No atribuyas causas" in L.SYSTEM_PORTAFOLIO
    assert "llama primero a `perfil_pais`" in L.SYSTEM_QA and "NO ofrezcas mostrar" in L.SYSTEM_QA


# ---- regresiones de la ejecución r8 (muestra de validación 5/12 y principal 6/12 al primer intento)
def test_regression_spanish_country_name_is_not_another_country():
    """'Papua Nueva Guinea' en la ficha de Papua New Guinea se marcaba como mención de 'Guinea'."""
    sp = L.spec_mercado("Papua New Guinea", T())
    out = sp["template"]()
    out["resumen"] = "Papua Nueva Guinea consumió 0.1 M en 2019/20. " + out["resumen"].split(".", 1)[1]
    assert sp["validator"](out) == []
    sp = L.spec_mercado("RD Congo", T())
    out = sp["template"]()
    out["resumen"] = "La República Democrática del Congo consumió poco. " + out["resumen"].split(".", 1)[1]
    assert "otros_paises" not in _tipos(sp["validator"](out))
    o2 = _bueno("Vietnam")                       # otro país sigue detectándose, también en su forma normalizada
    o2["resumen"] += " A diferencia de Guinea, crece."
    assert "otros_paises" in _tipos(_spec("Vietnam")["validator"](o2))


def test_regression_text_field_citations_are_tolerated_but_not_counted():
    """El modelo citó 'nivel', 'tipo_serie' y 'primera_temporada_con_dato' (campos de texto) en cifras_citadas."""
    sp = _spec("Panama")
    out = _bueno("Panama")
    out["cifras_citadas"] += [{"campo": "nivel", "valor": 0}, {"campo": "tipo_serie", "valor": 0},
                              {"campo": "primera_temporada_con_dato", "valor": 1990}]
    assert sp["validator"](out) == []
    out["cifras_citadas"].append({"campo": "consumo_2024_25_M", "valor": 4.0})    # campo inventado: sigue fallando
    assert "cifras_citadas" in _tipos(sp["validator"](out))
    solo_texto = _bueno("Panama")                                   # solo campos de texto: no cumple el mínimo
    solo_texto["cifras_citadas"] = [{"campo": "nivel", "valor": 0}] * 4
    assert "cifras_citadas" in _tipos(sp["validator"](solo_texto))


def test_regression_length_limits_are_style_not_safety():
    """Resúmenes de 95-96 palabras y viñetas de 46 (el prompt pide 75 y 35) ya no fuerzan un reintento."""
    out = _bueno("Vietnam")
    out["resumen"] = "palabra " * 96
    out["fortalezas"] = ["palabra " * 46]
    assert "longitud" not in _tipos(_spec("Vietnam")["validator"](out))
    out["resumen"] = "palabra " * (L.LIM["resumen"] + 5)
    assert "longitud" in _tipos(_spec("Vietnam")["validator"](out))       # el límite sigue existiendo


def test_stringified_arrays_are_normalized():
    """Rareza conocida de los LLM: un array llega como texto JSON. Se normaliza; un texto libre sigue siendo error."""
    out = _bueno("Vietnam")
    out["fortalezas"] = json.dumps(out["fortalezas"], ensure_ascii=False)
    assert isinstance(out["fortalezas"], str)
    assert L.normalizar_salida(out)["fortalezas"] and isinstance(out["fortalezas"], list)
    assert _spec("Vietnam")["validator"](out) == []
    libre = _bueno("Vietnam")
    libre["fortalezas"] = "un texto suelto"
    iss = _spec("Vietnam")["validator"](L.normalizar_salida(libre))
    assert any("recibí str" in i["detalle"] for i in iss)


def test_failed_attempts_record_raw_output_for_diagnosis():
    _tmp_cache()
    malo = _bueno()
    malo["resumen"] += " Crecerá 25.7% adicional."
    api = FakeAPI([_msg([_tool_use("registrar_analisis_mercado", malo)], "tool_use"),
                   _msg([_tool_use("registrar_analisis_mercado", _bueno())], "tool_use")])
    res = L.generate_validated(_client(api), "claude-sonnet-5", _spec())
    assert "25.7" in res["intentos"][0]["salida"] and res["intentos"][1]["salida"] is None


def test_manual_review_questions_include_false_offer_checks():
    fuera = L.preguntas_fuera_de_alcance(T())
    assert any("Alemania" in q for q in fuera) and any("Estados Unidos" in q for q in fuera)
    assert any("precio" in q for q in fuera) and not any("3 mercados con mayor score" in q for q in fuera)


def test_historical_tool_computes_growth_in_code():
    """En la app, '¿crecimiento de Colombia en los últimos 5 años?' se respondía con la proyección: el agente no
    tenía datos históricos. La herramienta los da con el crecimiento ya calculado (el modelo no calcula)."""
    from utils import get_series
    r = L.tool_historico_pais(T(), pais="Colombia", n_anios=5)
    s = get_series(T()["long"], "Colombia")
    assert r["n_anios"] == 5 and len(r["consumo_por_temporada"]) == 6            # 5 intervalos = 6 temporadas
    assert r["hasta_temporada"] == "2019/20" and r["desde_temporada"] == "2014/15"
    assert r["consumo_hasta_M"] == round(float(s.iloc[-1]), 1) and r["consumo_desde_M"] == round(float(s.iloc[-6]), 1)
    assert r["crecimiento_total_pct"] == round((s.iloc[-1] / s.iloc[-6] - 1) * 100, 1)
    assert r["crecimiento_anual_pct"] == round(((s.iloc[-1] / s.iloc[-6]) ** (1 / 5) - 1) * 100, 1)
    assert r["variacion_M"] == round(float(s.iloc[-1] - s.iloc[-6]), 1)
    assert [x["consumo_M"] for x in r["consumo_por_temporada"]] == [round(float(v), 1) for v in s.iloc[-6:]]


def test_historical_tool_edge_cases():
    t = T()
    assert "error" in L.tool_historico_pais(t, pais="Narnia")
    assert L.tool_historico_pais(t, pais="brasil")["pais"] == "Brazil"                 # nombre en español
    assert L.tool_historico_pais(t, pais="Colombia", n_anios=0)["n_anios"] == 1        # acota a 1..15
    assert L.tool_historico_pais(t, pais="Colombia", n_anios=99)["n_anios"] == 15
    assert L.tool_historico_pais(t, pais="Colombia", n_anios="abc")["n_anios"] == 5    # valor inválido -> por defecto
    corto = L.tool_historico_pais(t, pais="Yemen", n_anios=15)                          # serie de 17 temporadas
    assert corto["n_anios"] == 15 and len(corto["consumo_por_temporada"]) == 16
    plana = L.tool_historico_pais(t, pais="Kenya", n_anios=3)                           # serie sin variación
    assert plana["crecimiento_total_pct"] is not None
    assert json.dumps(L.tool_historico_pais(t, pais="Vietnam"))                         # serializable


def test_agent_answers_historical_question_with_verified_numbers():
    r0 = L.tool_historico_pais(T(), pais="Colombia", n_anios=5)
    ok = (f"Entre {r0['desde_temporada']} y {r0['hasta_temporada']} el consumo de Colombia pasó de "
          f"{r0['consumo_desde_M']} M a {r0['consumo_hasta_M']} M: {r0['crecimiento_total_pct']}% en total, "
          f"{r0['crecimiento_anual_pct']}% anual.")
    api = FakeAPI([_msg([_tool_use("historico_pais", {"pais": "Colombia", "n_anios": 5}, "t1")], "tool_use"),
                   _msg([_text(ok)])])
    r = L.preguntar(_client(api), "¿Cuál ha sido el crecimiento de Colombia en los últimos 5 años?", T())
    assert r["verificada"] and r["herramientas"][0]["herramienta"] == "historico_pais"
    assert api.requests[0]["tools"] and "historico_pais" in [x["name"] for x in api.requests[0]["tools"]]
    assert "historico_pais" in api.requests[0]["system"]
    # una cifra que el modelo calculara por su cuenta se rechaza
    mala = ok.replace(f"{r0['crecimiento_total_pct']}%", "77.7%")
    api = FakeAPI([_msg([_tool_use("historico_pais", {"pais": "Colombia"}, "t1")], "tool_use"),
                   _msg([_text(mala)])])
    assert not L.preguntar(_client(api), "¿Crecimiento de Colombia?", T(), reintentos_verificacion=0)["verificada"]


# ---- regresión: el agente negó que hubiera datos de variedades de café (sí los hay: tipo de café por país)
def test_coffee_type_tool_is_consistent_with_the_data():
    t = T()
    r = L.tool_tipos_cafe(t)
    assert [g["grupo"] for g in r["grupos"]] == ["Arabica", "Robusta", "Mixto"]
    assert sum(g["n_paises"] for g in r["grupos"]) == len(t["rk"])
    assert abs(sum(g["pct_consumo_2019_20"] for g in r["grupos"]) - 100) < 0.3
    assert abs(sum(g["consumo_2019_20_M"] for g in r["grupos"]) - t["rk"]["y19"].sum()) < 0.3
    assert abs(sum(g["proyeccion_2024_25_M"] for g in r["grupos"]) - t["rk"]["y24"].sum()) < 0.3
    por_tipo = t["long"][t["long"].country.isin(t["rk"].index)].drop_duplicates("country").coffee_type.value_counts()
    assert by_group_count(por_tipo, "Mixto") == next(g for g in r["grupos"] if g["grupo"] == "Mixto")["n_paises"]
    rob = L.tool_tipos_cafe(t, grupo="robusta")
    assert rob["grupo"] == "Robusta" and len(rob["paises"]) == rob["n_paises"]
    assert [p["consumo_2019_20_M"] for p in rob["paises"]] == sorted((p["consumo_2019_20_M"] for p in rob["paises"]), reverse=True)
    assert "Brazil" in [p["pais"] for p in L.tool_tipos_cafe(t, grupo="Mixto")["paises"]]
    v = L.tool_tipos_cafe(t, pais="vietnam")
    assert v["tipo_cafe"] == "Robusta/Arabica" and v["grupo"] == "Mixto" and 0 < v["cuota_en_su_grupo_pct"] < 100
    assert "opciones" in L.tool_tipos_cafe(t, grupo="cafeinado") and "error" in L.tool_tipos_cafe(t, pais="Narnia")
    assert "no se reparte" in r["aviso"] and json.dumps(r)


def by_group_count(por_tipo, grupo):
    if grupo == "Mixto":
        return int(por_tipo.get("Robusta/Arabica", 0) + por_tipo.get("Arabica/Robusta", 0))
    return int(por_tipo.get(grupo, 0))


def test_data_catalog_tool_and_prompt_say_what_exists():
    t = T()
    c = L.tool_catalogo_datos(t)
    assert any("Tipo de café" in x for x in c["contiene"]) and any("recio" in x for x in c["no_contiene"])
    assert c["n_paises_modelados"] == len(t["rk"]) and c["tipos_de_cafe"] == ["Arabica", "Robusta", "Mixto"]
    assert c["paises_del_dataset_sin_modelar"] == ["Equatorial Guinea", "Nepal", "Timor-Leste", "Zambia"]
    p = L.SYSTEM_QA
    assert "{CATALOGO_TEXTO}" not in p and "Qué contiene el análisis" in p and "Tipo de café de cada país" in p
    assert "nunca lo afirmes sin comprobarlo" in p and "catalogo_datos" in p
    assert {"catalogo_datos", "tipos_cafe"} <= set(L.TOOL_IMPL)


def test_portfolio_tool_lists_members_of_each_segment():
    """El agente afirmó que 'Grandes en crecimiento sostenido' incluía a los 7 prioritarios: solo son 5."""
    t = T()
    r = L.tool_resumen_portafolio(t)["paises_por_segmento"]
    assert "Vietnam" in r["Emergentes de alto crecimiento"] and "Vietnam" not in r["Grandes en crecimiento sostenido"]
    prior = set(t["rk"][t["rk"].nivel == "Prioritario"].index)
    assert len(prior & set(r["Grandes en crecimiento sostenido"])) == 5 and len(prior) == 7
    assert sum(len(v) for v in r.values()) == len(t["rk"])
    assert "paises_por_segmento" not in L.portfolio_fact_sheet(t)          # las fichas de las tarjetas no cambian


def test_regression_agent_cannot_deny_coffee_type_data_without_the_tool():
    """Respuesta real: 'El análisis no contiene datos sobre variedades de café' usando solo resumen_portafolio."""
    original = L.preguntar
    mala = ("El análisis no contiene datos sobre variedades de café (arábica, robusta, etc.), sino sobre consumo "
            "doméstico por país productor.")
    r0 = L.tool_tipos_cafe(T())
    mixto = next(g for g in r0["grupos"] if g["grupo"] == "Mixto")
    buena = (f"Sí: el dataset asigna a cada país un tipo de café. Mixto reúne {mixto['n_paises']} países y "
             f"{mixto['pct_consumo_2019_20']}% del consumo; Arabica y Robusta suman el resto. El consumo de un país "
             f"no se reparte entre variedades.")
    res = {}
    for nombre, texto, tools in [("mala", mala, ["resumen_portafolio"]), ("buena", buena, ["tipos_cafe"])]:
        def falso(client, pregunta, T_, model=None, _t=texto, _h=tools, **kw):
            return {"respuesta": _t, "herramientas": [{"herramienta": h, "argumentos": {}} for h in _h],
                    "verificada": True, "problemas": [], "reintentos": 0}
        L.preguntar = falso
        try:
            df = L.evaluar_qa(None, T(), workers=1)
        finally:
            L.preguntar = original
        res[nombre] = bool(df[df.pregunta.str.contains("variedades de café incluye")].iloc[0].aprobada)
    assert res == {"mala": False, "buena": True}, res


def test_agent_answers_coffee_type_question_with_verified_numbers():
    r0 = L.tool_tipos_cafe(T())
    ar = next(g for g in r0["grupos"] if g["grupo"] == "Arabica")
    texto = (f"El análisis clasifica a cada país por tipo de café. Arabica reúne {ar['n_paises']} países y "
             f"{ar['pct_consumo_2019_20']}% del consumo de 2019/20 ({ar['consumo_2019_20_M']} M).")
    api = FakeAPI([_msg([_tool_use("tipos_cafe", {}, "t1")], "tool_use"), _msg([_text(texto)])])
    r = L.preguntar(_client(api), "¿Qué variedades de café hay?", T())
    assert r["verificada"] and r["herramientas"][0]["herramienta"] == "tipos_cafe"
    assert "tipos_cafe" in [x["name"] for x in api.requests[0]["tools"]]
    api = FakeAPI([_msg([_tool_use("tipos_cafe", {}, "t1")], "tool_use"),
                   _msg([_text(texto.replace(f"{ar['pct_consumo_2019_20']}%", "37.7%"))])])
    assert not L.preguntar(_client(api), "¿Qué variedades de café hay?", T(), reintentos_verificacion=0)["verificada"]


def test_code_version_defined():
    assert L.CODE_VERSION and hasattr(L, "DEFINICIONES_NIVELES")


def test_evaluar_qa_repetitions_and_aggregation():
    original = L.preguntar
    llamadas = []

    def falso(client, pregunta, T_, model=None, **kw):
        llamadas.append(pregunta)
        return {"respuesta": "Alemania no está en el dataset; no incluye importaciones.",
                "herramientas": [{"herramienta": "perfil_pais", "argumentos": {}}],
                "verificada": True, "problemas": [], "reintentos": 0}
    L.preguntar = falso
    try:
        df = L.evaluar_qa(None, T(), repeticiones=2, workers=1)
    finally:
        L.preguntar = original
    n = len(L.eval_questions(T()))
    assert len(df) == 2 * n == len(llamadas)
    assert {"repeticion", "respuesta", "aprobada"} <= set(df.columns)
    assert df[df.pregunta.str.contains("Alemania")].aprobada.all()      # la respuesta correcta aprueba
    assert not df[df.pregunta.str.contains("top|3 mercados", regex=True)].aprobada.any()  # sin herramienta, no aprueba


def test_validator_catches_fabricated_number():
    out = _bueno()
    out["resumen"] += " Además crecerá 25.7% adicional."
    assert "cifra_no_respaldada" in _tipos(_spec()["validator"](out))


def test_validator_catches_wrong_citation():
    out = _bueno()
    out["cifras_citadas"][0]["valor"] = 999
    assert "cifras_citadas" in _tipos(_spec()["validator"](out))


def test_validator_catches_unknown_field():
    out = _bueno()
    out["cifras_citadas"].append({"campo": "campo_inventado", "valor": 1})
    assert "cifras_citadas" in _tipos(_spec()["validator"](out))


def test_validator_catches_upgraded_recommendation():
    for pais, rec in [("Bolivia", "Priorizar"), (T()["rk"][T()["rk"].nivel == "Bajo"].index[0], "Priorizar")]:
        out = _bueno(pais)
        out["recomendacion"] = rec
        assert "recomendacion" in _tipos(L.spec_mercado(pais, T())["validator"](out)), pais


def test_validator_catches_invalid_enum():
    out = _bueno()
    out["recomendacion"] = "Comprar ya"
    assert "recomendacion" in _tipos(_spec()["validator"](out))


def test_validator_catches_missing_caveat():
    out = _bueno("Vietnam")                       # tiene alerta de optimismo
    out["riesgos"] = ["Es un mercado grande."]
    assert "cautela_faltante" in _tipos(_spec("Vietnam")["validator"](out))
    out = _bueno("Bolivia")                       # nicho
    out["riesgos"] = ["Hay incertidumbre general."]
    assert "cautela_faltante" in _tipos(_spec("Bolivia")["validator"](out))


def test_validator_catches_other_country():
    out = _bueno()
    out["resumen"] += " A diferencia de Brazil, crece más rápido."
    assert "otros_paises" in _tipos(_spec()["validator"](out))


def test_validator_catches_schema_problems():
    out = _bueno()
    out["riesgos"] = []
    assert "esquema" in _tipos(_spec()["validator"](out))
    out = _bueno()
    del out["justificacion"]
    assert "esquema" in _tipos(_spec()["validator"](out))
    out = _bueno()
    out["resumen"] = "palabra " * 200
    assert "longitud" in _tipos(_spec()["validator"](out))


def test_portfolio_validator():
    sp = L.spec_portafolio(T())
    out = sp["template"]()
    out["riesgos_y_limites"] = ["Todo va bien."]
    assert "cautela_faltante" in _tipos(sp["validator"](out))
    out = sp["template"]()
    out["hallazgos"][0] += " Habrá 12345 M más."
    assert "cifra_no_respaldada" in _tipos(sp["validator"](out))


# ------------------------------------------------------------------ flujo con el SDK real (simulado)
def test_sdk_happy_path_request_shape():
    _tmp_cache()
    sp = _spec()
    api = FakeAPI([_msg([_tool_use("registrar_analisis_mercado", _bueno())], "tool_use")])
    res = L.generate_validated(_client(api), "claude-sonnet-5", sp)
    assert res["fuente"] == "llm" and res["validacion_ok"], res
    req = api.requests[0]
    assert req["model"] == "claude-sonnet-5"
    assert req["tool_choice"] == {"type": "tool", "name": "registrar_analisis_mercado"}
    assert req["tools"][0]["name"] == "registrar_analisis_mercado"
    assert "temperature" not in req and "top_p" not in req      # Sonnet 5 rechaza parámetros de muestreo
    assert req["max_tokens"] >= 4000
    assert "Ficha (JSON)" in req["messages"][0]["content"] and "ÚNICAMENTE" in req["system"]
    assert res["intentos"][0]["tokens_in"] == 1200


def test_sdk_retry_with_feedback():
    _tmp_cache()
    malo = _bueno()
    malo["resumen"] += " Crecerá 25.7% adicional."
    api = FakeAPI([_msg([_tool_use("registrar_analisis_mercado", malo, "toolu_A")], "tool_use"),
                   _msg([_tool_use("registrar_analisis_mercado", _bueno(), "toolu_B")], "tool_use")])
    res = L.generate_validated(_client(api), "claude-sonnet-5", _spec())
    assert res["fuente"] == "llm" and len(res["intentos"]) == 2
    assert res["intentos"][0]["problemas"] and not res["intentos"][1]["problemas"]
    msgs = api.requests[1]["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    tr = msgs[2]["content"][0]
    assert tr["type"] == "tool_result" and tr["tool_use_id"] == "toolu_A" and tr["is_error"] is True
    assert "cifra_no_respaldada" in tr["content"] and "25.7" in tr["content"]


def test_sdk_fallback_after_repeated_failures():
    _tmp_cache()
    malo = _bueno()
    malo["resumen"] += " Crecerá 25.7% adicional."
    api = FakeAPI([_msg([_tool_use("registrar_analisis_mercado", malo)], "tool_use")])
    res = L.generate_validated(_client(api), "claude-sonnet-5", _spec(), max_attempts=3)
    assert len(api.requests) == 3
    assert res["fuente"] == "plantilla_tras_fallo_llm" and res["validacion_ok"]
    assert res["problemas_llm"], "debe registrar por qué falló el LLM"


def test_sdk_api_error_falls_back():
    _tmp_cache()
    api = FakeAPI([(500, {"type": "error", "error": {"type": "api_error", "message": "boom"}})])
    res = L.generate_validated(_client(api), "claude-sonnet-5", _spec())
    assert res["fuente"] == "plantilla_tras_fallo_llm" and res["validacion_ok"]
    assert res["problemas_llm"][0]["tipo"] == "error_api"


def test_sdk_no_tool_call_is_retried():
    _tmp_cache()
    api = FakeAPI([_msg([_text("Aquí tienes mi análisis en texto libre.")]),
                   _msg([_tool_use("registrar_analisis_mercado", _bueno())], "tool_use")])
    res = L.generate_validated(_client(api), "claude-sonnet-5", _spec())
    assert res["fuente"] == "llm" and len(res["intentos"]) == 2
    assert res["intentos"][0]["problemas"][0]["tipo"] == "sin_herramienta"
    assert api.requests[1]["messages"][2]["role"] == "user"          # feedback como texto


def test_sdk_forced_tool_choice_rejected_falls_to_auto():
    _tmp_cache()
    api = FakeAPI([(400, {"type": "error", "error": {"type": "invalid_request_error",
                                                     "message": "tool_choice is not supported with this model"}}),
                   _msg([_tool_use("registrar_analisis_mercado", _bueno())], "tool_use")])
    res = L.generate_validated(_client(api), "modelo-x", _spec())
    assert res["fuente"] == "llm", res
    assert "tool_choice" in api.requests[0] and "tool_choice" not in api.requests[1]
    assert "Responde llamando a la herramienta" in api.requests[1]["messages"][0]["content"]


def test_cache_roundtrip_and_offline_replay():
    _tmp_cache()
    api = FakeAPI([_msg([_tool_use("registrar_analisis_mercado", _bueno())], "tool_use")])
    r1 = L.generate_validated(_client(api), "claude-sonnet-5", _spec())
    assert r1["fuente"] == "llm" and len(api.requests) == 1
    r2 = L.generate_validated(_client(api), "claude-sonnet-5", _spec())          # misma ficha: no llama
    assert r2["fuente"] == "cache" and len(api.requests) == 1
    r3 = L.generate_validated(None, "claude-sonnet-5", _spec())                   # offline reproduce el LLM
    assert r3["fuente"] == "cache" and r3["analisis"] == r1["analisis"]
    L.generate_validated(_client(api), "claude-sonnet-5", _spec(), refresh=True)  # refresh fuerza la llamada
    assert len(api.requests) == 2


def test_cache_key_changes_with_facts_or_model():
    fs = L.country_fact_sheet("Vietnam", T())
    fs2 = copy.deepcopy(fs)
    fs2["cagr_proyectado_pct"] += 0.1
    assert L.cache_key("m", "mercado", fs) != L.cache_key("m", "mercado", fs2)
    assert L.cache_key("m", "mercado", fs) != L.cache_key("otro", "mercado", fs)


def test_offline_uses_template():
    _tmp_cache()
    res = L.generate_validated(None, "claude-sonnet-5", _spec())
    assert res["fuente"] == "plantilla" and res["validacion_ok"]


def test_generate_many_preserves_order_offline():
    _tmp_cache()
    paises = ["Vietnam", "Bolivia", "Colombia"]
    res = L.generate_many(None, "claude-sonnet-5", [L.spec_mercado(p, T()) for p in paises])
    assert [r["id"] for r in res] == paises and all(r["validacion_ok"] for r in res)


def test_generate_many_parallel_shares_cache_safely():
    _tmp_cache()
    paises = ["Vietnam", "Bolivia", "Colombia", "Ethiopia", "Uganda", "Mexico"]
    respuestas = [_msg([_tool_use("registrar_analisis_mercado", _bueno(p), f"t_{p}")], "tool_use") for p in paises]

    class PorPais(FakeAPI):                     # responde según el país que aparece en la petición
        def __call__(self, request):
            h = _httpx()
            body = json.loads(request.content)
            self.requests.append(body)
            for p, r in zip(paises, respuestas):
                if f'"pais": "{p}"' in body["messages"][0]["content"]:
                    return h.Response(200, json=r)
            return h.Response(500, json={"type": "error", "error": {"type": "api_error", "message": "?"}})

    api = PorPais([])
    res = L.generate_many(_client(api), "claude-sonnet-5", [L.spec_mercado(p, T()) for p in paises], workers=4)
    assert [r["id"] for r in res] == paises and all(r["fuente"] == "llm" for r in res)
    assert len(json.loads(L.CACHE_PATH.read_text())) == len(paises), "se perdieron entradas de caché"


def test_only_valid_llm_results_are_cached():
    _tmp_cache()
    malo = _bueno()
    malo["resumen"] += " Crecerá 25.7% adicional."
    api = FakeAPI([_msg([_tool_use("registrar_analisis_mercado", malo)], "tool_use")])
    L.generate_validated(_client(api), "claude-sonnet-5", _spec())
    assert not L.CACHE_PATH.exists() or json.loads(L.CACHE_PATH.read_text()) == {}


# ------------------------------------------------------------------ herramientas y agente
def test_resolve_country():
    t = T()
    assert L.resolve_country("Brasil", t)[0] == "Brazil"
    assert L.resolve_country("etiopía", t)[0] == "Ethiopia"
    assert L.resolve_country("VIETNAM", t)[0] == "Vietnam"
    p, sug = L.resolve_country("Narnia", t)
    assert p is None and isinstance(sug, list)
    assert L.resolve_country("Colombi", t)[1] == ["Colombia"]


def test_tools_return_grounded_data():
    t = T()
    top = L.tool_top_ranking(t, n=3)
    assert [x["posicion_ranking"] for x in top["paises"]] == [1, 2, 3]
    assert len(L.tool_pronostico_pais(t, pais="Colombia")["pronostico"]) == 5
    assert len(L.tool_comparar_paises(t, paises=["Colombia", "México", "Narnia"])["paises"]) == 2
    assert "error" in L.run_tool("perfil_pais", {"pais": "Narnia"}, t)
    assert "error" in L.run_tool("no_existe", {}, t)
    assert L.tool_top_ranking(t, n=50)["paises"].__len__() == 20        # límite superior
    nichos = L.tool_top_ranking(t, nivel="Nicho prometedor", n=10)["paises"]
    assert all(x["nivel"] == "Nicho prometedor" for x in nichos)


def test_tool_schemas_are_well_formed():
    for tool in L.QA_TOOLS + [L.TOOL_MERCADO, L.TOOL_PORTAFOLIO]:
        assert {"name", "description", "input_schema"} <= set(tool)
        sch = tool["input_schema"]
        assert sch["type"] == "object"
        assert set(sch.get("required", [])) <= set(sch["properties"])
    assert set(L.TOOL_IMPL) == {t["name"] for t in L.QA_TOOLS}


def _agent_api(*finales):
    return FakeAPI([_msg([_tool_use("perfil_pais", {"pais": "Vietnam"}, "toolu_Q")], "tool_use"), *finales])


def test_agent_tool_loop_and_grounded_answer():
    api = _agent_api(_msg([_text("Vietnam proyecta 6.7% anual frente a 3.7% reciente: puede ser optimista.")]))
    r = L.preguntar(_client(api), "¿Por qué Vietnam tiene alerta de optimismo?", T())
    assert r["verificada"] and r["herramientas"][0]["herramienta"] == "perfil_pais"
    tr = api.requests[1]["messages"][2]["content"][0]
    assert tr["type"] == "tool_result" and tr["tool_use_id"] == "toolu_Q" and "Vietnam" in tr["content"]


def test_agent_ungrounded_answer_is_retried_then_verified():
    api = _agent_api(_msg([_text("Vietnam crecerá 12.3% anual.")]),
                     _msg([_text("Vietnam proyecta 6.7% anual, según la ficha.")]))
    r = L.preguntar(_client(api), "¿Cuánto crece Vietnam?", T())
    assert r["reintentos"] == 1 and r["verificada"] and len(api.requests) == 3
    assert "12.3" in api.requests[2]["messages"][-1]["content"]


def test_agent_persistent_ungrounded_is_flagged():
    api = _agent_api(_msg([_text("Vietnam crecerá 12.3% anual.")]))
    r = L.preguntar(_client(api), "¿Cuánto crece Vietnam?", T())
    assert not r["verificada"] and r["problemas"]


def test_agent_number_in_question_is_allowed():
    api = FakeAPI([_msg([_tool_use("top_ranking", {"n": 3}, "toolu_T")], "tool_use"),
                   _msg([_text("Los 3 primeros por score son Brazil, Ethiopia y Vietnam.")])])
    r = L.preguntar(_client(api), "¿Cuáles son los 3 mercados con mayor score?", T())
    assert r["verificada"]


def test_agent_empty_answer_is_flagged():
    api = FakeAPI([_msg([])])
    r = L.preguntar(_client(api), "hola", T())
    assert not r["verificada"] and r["problemas"][0]["tipo"] == "sin_respuesta"


def test_eval_battery_defined():
    qs = L.eval_questions(T())
    assert len(qs) >= 8 and any(not q["tools_any"] for q in qs)         # incluye preguntas fuera de alcance


# ------------------------------------------------------------------ precios (notebooks 06-07)
def _T_precios():
    t = T()
    assert L._hay_precios_globales(t) and "extra" in t, "Ejecuta antes los notebooks 06 y 07 (resultados de precios en outputs/)"
    return t


def _sin_precios(t):
    return {k: v for k, v in t.items() if not k.startswith("price") and k != "extra"}


def test_price_tools_are_registered_and_say_what_they_are_not():
    nombres = {tool["name"]: tool for tool in L.QA_TOOLS}
    assert {"precios_globales", "precio_productor"} <= set(nombres) and {"precios_globales", "precio_productor"} <= set(L.TOOL_IMPL)
    assert "no existe por país" in nombres["precios_globales"]["description"]
    assert "no hay pronóstico de precio por país" in nombres["precio_productor"]["description"]


def test_global_price_tool_matches_notebook_outputs():
    t = _T_precios()
    r = L.tool_precios_globales(t, tipo="Arabica", anio=2023)
    assert r["unidad"].startswith("US$ de 2019") and len(r["series"]) == 1
    f = r["series"][0]["pronostico_desde_2018"][0]
    fc = t["price_fc"][(t["price_fc"].serie == "other_milds") & (t["price_fc"].anio == 2023)].iloc[0]
    val = t["price_val"][(t["price_val"].serie == "other_milds") & (t["price_val"].anio == 2023)].iloc[0]
    assert f["central"] == round(fc.central, 2) and f["banda_baja"] == round(fc.lo, 2) and f["banda_alta"] == round(fc.hi, 2)
    assert f["realizado_real"] == round(val["real"], 2) and f["horizonte_anios"] == 5
    p18 = r["series"][0]["historico"]["precio_real_2018"]
    assert abs(f["cambio_central_vs_2018_pct"] - (fc.central / p18 - 1) * 100) < 0.5           # calculado en código
    assert abs(f["cambio_realizado_vs_2018_pct"] - (val["real"] / p18 - 1) * 100) < 0.5
    assert abs(f["factor_banda"] - fc.hi / fc.central) < 0.006


def test_global_price_tool_year_handling_and_unknown_type():
    t = _T_precios()
    hist = L.tool_precios_globales(t, tipo="robusta", anio=2010)["series"][0]
    assert hist["tipo"] == "Robusta" and hist["anio_consultado"]["precio_real"] > 0 and "pronostico_desde_2018" not in hist
    n25 = L.tool_precios_globales(t, tipo="arabica", anio=2025)["series"][0]["anio_consultado"]
    assert n25["sin_pronostico"] and abs(n25["cambio_nominal_vs_2018_pct"] - 188) < 1.5          # 8.45 frente a 2.93 US$/kg nominales
    fuera = L.tool_precios_globales(t, anio=2030)
    assert all(s["anio_consultado"]["sin_pronostico"] for s in fuera["series"]) and len(fuera["series"]) == 2
    assert "error" in L.tool_precios_globales(t, tipo="cacao") and "opciones" in L.tool_precios_globales(t, tipo="cacao")
    assert len(L.tool_precios_globales(t, anio="dos mil")["series"][0]["pronostico_desde_2018"]) == 6   # año inválido: serie completa


def test_global_price_tool_reports_effective_coverage_and_warnings():
    t = _T_precios()
    r = L.tool_precios_globales(t)
    cov = r["cobertura_efectiva_por_horizonte_pct"]
    assert set(cov) == {f"ano_{k}" for k in range(1, 7)} and cov["ano_1"] > cov["ano_6"]        # la cobertura cae con el horizonte
    assert any("No existe pronóstico de precio por país" in a for a in r["advertencias"])
    assert any("no anticipó" in a for a in r["advertencias"])


def _tipos_raw(t):
    """Precio al productor recalculado desde el CSV CRUDO de la ICO (sin pasar por el ETL ni el parquet)."""
    from utils import SHORT_NAMES
    raw = pd.read_csv(L.ROOT / "datos_externos" / "coffeedata.csv")
    raw["pais"] = raw["Country"].replace(SHORT_NAMES)
    raw["tipo"] = raw["Coffee_type"].map(lambda x: "Robusta" if x == "Robustas" else "Arábica")
    return raw[raw.Price_grower.notna() & raw.pais.isin(set(t["rk"].index))]


def test_grower_price_ranking_is_per_type_sorted_and_flags_small_producers():
    t = _T_precios()
    r = L.tool_precio_productor(t)
    d0, h0 = L.PRODUCTOR_VENTANA_DEFECTO
    assert r["periodo"] == {"desde": d0, "hasta": h0} and r["ventana_por_defecto"] is True
    assert r["n_paises_modelados"] == len(t["rk"]) and [e["tipo"] for e in r["por_tipo"]] == ["Arábica", "Robusta"]
    ex, modelados = t["extra"], set(t["rk"].index)
    for e, k in zip(r["por_tipo"], ("arabica", "robusta")):
        col = f"price_grower_{k}_real_usd_kg"
        v = [x["variacion_real_pct"] for x in e["mayores_aumentos"]]
        c = [x["variacion_real_pct"] for x in e["mayores_caidas"]]
        assert v == sorted(v, reverse=True) and c == sorted(c) and v[0] >= c[0]
        ex0 = ex[ex.country.isin(modelados) & ex[col].notna()]
        assert e["n_paises_con_algun_precio"] == ex0.country.nunique() > e["n_paises_con_dato"] >= 3
        assert e["n_paises_con_dato_en_desde"] == ex0[ex0.year == d0].country.nunique()
        assert e["n_paises_con_dato_en_hasta"] == ex0[ex0.year == h0].country.nunique()
        assert e["n_paises_con_dato"] <= min(e["n_paises_con_dato_en_desde"], e["n_paises_con_dato_en_hasta"])
        for x in e["mayores_aumentos"] + e["mayores_caidas"]:
            a = ex[(ex.country == x["pais"]) & (ex.year == d0)][col].iloc[0]
            b = ex[(ex.country == x["pais"]) & (ex.year == h0)][col].iloc[0]
            assert abs(x["variacion_real_pct"] - (b / a - 1) * 100) < 0.06                      # recalculado desde los datos
            assert x["pais"] in modelados
            prod = ex[(ex.country == x["pais"]) & (ex.year == h0)].production_kg.iloc[0] / 60 / 1e6
            assert x["productor_pequeno"] == (prod < r["umbral_productor_pequeno_M_sacos"])
    adv = " ".join(r["advertencias"])
    assert "dejan de reportar" in adv and "pequeños" in adv and "no existe pronóstico de precio por país" in adv
    assert "no son comparables" in adv


def test_grower_price_ranking_compares_like_with_like_not_mixed_averages():
    """Regresión: Tailandia salía con +69.5% (2008→2013) porque en 2013 se promediaba Arábica con Robusta. Robusta vs Robusta: +14.5%."""
    t = _T_precios()
    por = {e["tipo"]: e for e in L.tool_precio_productor(t, n=10)["por_tipo"]}
    th = next(x for x in por["Robusta"]["mayores_aumentos"] + por["Robusta"]["mayores_caidas"] if x["pais"] == "Thailand")
    assert abs(th["variacion_real_pct"] - 14.5) < 0.1 and abs(th["variacion_real_pct"] - 69.5) > 50
    todos_arabica = {x["pais"] for x in por["Arábica"]["mayores_aumentos"] + por["Arábica"]["mayores_caidas"]}
    assert "Thailand" not in todos_arabica                                                       # no hay Arábica de Tailandia en 2008


def test_grower_price_join_does_not_lose_countries():
    """Regresión de una duda real: '¿solo 14 de 51?'. Se recalcula desde el CSV CRUDO, por tipo de café."""
    t = _T_precios()
    raw = _tipos_raw(t)
    r = L.tool_precio_productor(t, desde=2013, hasta=2018)
    for e in r["por_tipo"]:
        s = raw[raw.tipo == e["tipo"]]
        con = lambda y: set(s[s.Year == y].pais)
        assert e["n_paises_con_dato"] == len(con(2013) & con(2018))                              # el cruce no pierde ninguno
        assert e["n_paises_con_dato_en_desde"] == len(con(2013)) and e["n_paises_con_dato_en_hasta"] == len(con(2018))
        assert e["n_paises_con_algun_precio"] == s.pais.nunique()
    total = raw.pais.nunique()
    assert total == 44 and total > 3 * max(e["n_paises_con_dato"] for e in r["por_tipo"])       # casi todos tienen precio en ALGÚN año
    cobertura = [raw[raw.Year == y].pais.nunique() for y in (1990, 2000, 2010, 2018)]
    assert cobertura == sorted(cobertura, reverse=True) and cobertura[0] > cobertura[-1]         # la cobertura cae con el tiempo


def test_grower_price_default_window_keeps_more_countries_than_the_recent_one():
    t = _T_precios()
    base, reciente = L.tool_precio_productor(t), L.tool_precio_productor(t, desde=2013, hasta=2018)
    total = lambda r: sum(e["n_paises_con_dato"] for e in r["por_tipo"])
    assert total(base) > total(reciente)
    assert base["n_mercados_clave_fuera_del_ranking"] < reciente["n_mercados_clave_fuera_del_ranking"]
    assert base["ventana_por_defecto"] is True and reciente["ventana_por_defecto"] is False


def test_grower_price_single_endpoint_builds_a_five_year_window():
    t = _T_precios()
    assert L.tool_precio_productor(t, desde=2005)["periodo"] == {"desde": 2005, "hasta": 2010}
    assert L.tool_precio_productor(t, hasta=2010)["periodo"] == {"desde": 2005, "hasta": 2010}
    assert L.tool_precio_productor(t, desde=2018)["periodo"] == {"desde": 2018, "hasta": 2019}      # tope: 2019
    assert L.tool_precio_productor(t, hasta=1992)["periodo"] == {"desde": 1990, "hasta": 1992}      # el inicio se recorta a 1990
    assert "error" in L.tool_precio_productor(t, hasta=1990)                                        # ventana vacía: inválido


def test_grower_price_reports_key_markets_left_out_of_the_ranking():
    t = _T_precios()
    raw = _tipos_raw(t)
    r = L.tool_precio_productor(t, desde=2013, hasta=2018)
    fuera = set(r["mercados_clave_fuera_del_ranking"])
    assert {"Vietnam", "Thailand", "Philippines"} <= fuera                                          # tres mercados del top 7
    assert r["n_mercados_clave_fuera_del_ranking"] == len(fuera) and r["n_mercados_clave"] == 11
    en_ranking = {x["pais"] for e in r["por_tipo"] for x in e.get("mayores_aumentos", []) + e.get("mayores_caidas", [])}
    assert not (en_ranking & fuera)
    for c in fuera:                                                                                # de verdad no se pueden comparar
        for tipo in ("Arábica", "Robusta"):
            s = raw[(raw.pais == c) & (raw.tipo == tipo)]
            assert not ({2013, 2018} <= set(s.Year)), (c, tipo)
    assert t["rk"].loc[list(fuera), "nivel"].isin(L.NIVELES_MERCADO_CLAVE).all()


def test_grower_price_country_mode_uses_all_years_with_data_by_type():
    """'Con los datos disponibles de los años X–Y, el precio se ha comportado así': todos los años con dato, por tipo."""
    t = _T_precios()
    ex = t["extra"]
    r = L.tool_precio_productor(t, pais="Tailandia")
    assert r["pais"] == "Thailand" and r["periodo_consultado"] == {"desde": 1990, "hasta": 2019}
    por = {x["tipo"]: x for x in r["tipos"]}
    assert set(por) == {"Robusta", "Arábica"}
    rob = por["Robusta"]
    d = ex[(ex.country == "Thailand") & ex.price_grower_robusta_real_usd_kg.notna()].sort_values("year")
    assert (rob["primer_anio"], rob["ultimo_anio"], rob["n_anios_con_dato"]) == (int(d.year.min()), int(d.year.max()), len(d))
    assert rob["anios_sin_dato_dentro_del_rango"] == (rob["ultimo_anio"] - rob["primer_anio"] + 1) - rob["n_anios_con_dato"]
    v = d.price_grower_robusta_real_usd_kg.tolist()
    assert abs(rob["variacion_real_total_pct"] - (v[-1] / v[0] - 1) * 100) < 0.06
    assert rob["minimo"]["precio_real"] == round(min(v), 2) and rob["maximo"]["precio_real"] == round(max(v), 2)
    assert [x["anio"] for x in rob["serie_real"]] == d.year.astype(int).tolist()                # solo años con dato, en orden
    assert rob["tramo_reciente"]["hasta_anio"] == rob["ultimo_anio"] and rob["tramo_reciente"]["desde_anio"] < rob["ultimo_anio"]
    ara = por["Arábica"]
    assert ara["n_anios_con_dato"] < 6 and "tramo_reciente" not in ara and ara["primer_anio"] > rob["primer_anio"]
    assert r["ultimo_anio_con_dato"] == max(rob["ultimo_anio"], ara["ultimo_anio"]) < 2019      # el dato NO llega al presente
    assert str(r["ultimo_anio_con_dato"]) in r["nota_cobertura"] and "no se extrapola" in r["nota_cobertura"]
    assert any("no son comparables" in a for a in r["advertencias"])


def test_grower_price_country_mode_filters_windows_types_and_errors():
    t = _T_precios()
    r = L.tool_precio_productor(t, pais="Thailand", desde=2010, hasta=2013, tipo="Robusta")
    assert [x["tipo"] for x in r["tipos"]] == ["Robusta"] and r["tipos"][0]["primer_anio"] >= 2010 and r["tipos"][0]["ultimo_anio"] <= 2013
    uno = L.tool_precio_productor(t, pais="Thailand", desde=2010, hasta=2010, tipo="Robusta")["tipos"][0]
    assert uno["n_anios_con_dato"] == 1 and uno["un_solo_anio_con_dato"] and "variacion_real_total_pct" not in uno
    assert "opciones" in L.tool_precio_productor(t, pais="Thailand", tipo="cacao")
    assert L.tool_precio_productor(t, pais="Paraguay")["sin_datos"] is True                       # modelado, pero sin ningún precio
    assert "sugerencias" in L.tool_precio_productor(t, pais="Narnia")
    for malo in ({"desde": 2018, "hasta": 2013}, {"desde": 1980, "hasta": 2018}, {"desde": 2013, "hasta": 2025}, {"desde": "x", "hasta": 2018}):
        assert "error" in L.tool_precio_productor(t, pais="Colombia", **malo), malo
    col = L.tool_precio_productor(t, pais="Colombia")                                             # Colombia: solo Arábica
    assert [x["tipo"] for x in col["tipos"]] == ["Arábica"]


def test_grower_price_ranking_needs_enough_countries():
    t = _T_precios()
    r = L.tool_precio_productor(t, desde=1990, hasta=1991)
    assert all(e.get("sin_datos_suficientes") or e["n_paises_con_dato"] >= 3 for e in r["por_tipo"])
    assert "error" in L.tool_precio_productor(t, tipo="cacao")


def test_grower_price_old_parquet_without_type_columns_asks_to_rerun_the_etl():
    t = dict(_T_precios())
    t["extra"] = t["extra"].drop(columns=["price_grower_arabica_real_usd_kg", "price_grower_robusta_real_usd_kg"])
    assert "notebook 06" in L.tool_precio_productor(t)["error"] and "notebook 06" in L.tool_precio_productor(t, pais="Thailand")["error"]


def test_price_tools_degrade_without_price_data(tmp_path=None):
    t = _sin_precios(T())
    assert "error" in L.tool_precios_globales(t) and "error" in L.tool_precio_productor(t)
    assert "notebooks 06 y 07" in L.tool_precios_globales(t)["error"]
    c = L.tool_catalogo_datos(t)
    assert c["precios_cargados"] == {"globales": False, "productor": False}
    assert any("no están cargados" in x for x in c["no_contiene"]) and not any("precio_productor" in x for x in c["contiene"])
    assert L.run_tool("precios_globales", {}, t).get("error")
    with tempfile.TemporaryDirectory() as d:
        assert L.load_price_tables(out=Path(d), raw=Path(d)) == {}                               # carpeta vacía: no falla


def test_catalog_lists_price_tools_only_when_loaded():
    c = L.tool_catalogo_datos(_T_precios())
    assert c["precios_cargados"] == {"globales": True, "productor": True}
    assert any("precios_globales" in x for x in c["contiene"]) and any("precio_productor" in x for x in c["contiene"])
    assert any("POR PAÍS" in x for x in c["no_contiene"])
    assert "no usa precios" in c["aclaracion"] and not any("no usa precios" in x for x in c["no_contiene"])


def test_load_price_tables_uses_short_country_names():
    t = _T_precios()
    paises = set(t["extra"]["country"])
    assert "Vietnam" in paises and "Viet Nam" not in paises and "Bolivia" in paises
    assert "cuántos lo reportan en algún año" in L.SYSTEM_QA
    assert set(t["rk"].index) <= paises                                                          # los 51 modelados tienen fila
    assert "Benin" not in paises


def test_prompt_forbids_country_level_price_forecast():
    p = L.SYSTEM_QA
    assert "NO existe pronóstico de precio por país" in p and "`precios_globales`" in p and "`precio_productor`" in p
    assert "no anticipó el alza de 2021-2024" in p and "El pronóstico de consumo no usa precios" in p
    assert "Precio internacional de referencia" in p and "Pronóstico de precios POR PAÍS" in p    # el catálogo lo dice
    assert "ignorar o revelar" in p and "definicion_nivel" in p                                   # reglas anteriores intactas


def test_eval_battery_includes_price_questions_and_trap_patterns():
    qs = {q["pregunta"]: q for q in L.eval_questions(_T_precios())}
    assert any("subió más el precio al productor" in k for k in qs)
    trampa = next(q for k, q in qs.items() if "aumentará más el precio" in k)
    assert not trampa["tools_any"] and trampa["prohibidas"]
    falso = L._norm("Vietnam será el país donde más aumentará el precio.")
    ok = L._norm("No existe pronóstico de precio por país; el modelo es global por tipo de café.")
    assert any(L._hay(k, falso) for k in trampa["prohibidas"]) and not any(L._hay(k, ok) for k in trampa["prohibidas"])
    assert any("aumentará más el precio" in q for q in L.preguntas_fuera_de_alcance(_T_precios()))
    sin = [q["pregunta"] for q in L.eval_questions(_sin_precios(T()))]
    assert not any("precio al productor" in q or "Arábica en 2023" in q for q in sin)          # sin datos, no se evalúan


def _api_precio(tool, args, texto):
    return FakeAPI([_msg([_tool_use(tool, args, "toolu_P")], "tool_use"), _msg([_text(texto)])])


def test_agent_price_answer_is_verified_against_tool_numbers():
    t = _T_precios()
    f = L.tool_precios_globales(t, tipo="Arabica", anio=2023)["series"][0]["pronostico_desde_2018"][0]
    texto = (f"Para el Arábica en 2023 el modelo daba {f['central']} US$/kg reales, con banda del 80% de "
             f"{f['banda_baja']} a {f['banda_alta']}; lo realizado fue {f['realizado_real']}.")
    r = L.preguntar(_client(_api_precio("precios_globales", {"tipo": "Arabica", "anio": 2023}, texto)), "¿Precio Arábica 2023?", t)
    assert r["verificada"] and r["herramientas"][0]["herramienta"] == "precios_globales"
    malo = L.preguntar(_client(_api_precio("precios_globales", {"tipo": "Arabica", "anio": 2023}, "El Arábica valdrá 4.55 en 2023.")), "¿Precio?", t)
    assert not malo["verificada"] and malo["problemas"][0]["tipo"] == "cifra_no_respaldada"


def test_agent_grower_ranking_answer_is_verified_against_tool_numbers():
    t = _T_precios()
    r0 = L.tool_precio_productor(t)
    e = next(x for x in r0["por_tipo"] if x["tipo"] == "Robusta")
    top = e["mayores_aumentos"][0]
    texto = (f"Entre {r0['periodo']['desde']} y {r0['periodo']['hasta']}, en Robusta, {top['pais']} tuvo el mayor aumento real: "
             f"{top['variacion_real_pct']}%. {e['n_paises_con_dato']} de {r0['n_paises_modelados']} países tienen precio "
             f"en ambos años y {e['n_paises_con_algun_precio']} lo reportan en algún año.")
    r = L.preguntar(_client(_api_precio("precio_productor", {}, texto)), "¿En qué país subió más el precio?", t)
    assert r["verificada"], r["problemas"]
    malo = L.preguntar(_client(_api_precio("precio_productor", {}, f"{top['pais']} subió 250% y 30 países reportan.")), "¿Cuál?", t)
    assert not malo["verificada"]


def test_agent_country_price_behavior_is_verified_and_extrapolation_is_flagged():
    """La respuesta que pide el usuario: 'con los datos disponibles de los años X–Y, los precios se han comportado así'."""
    t = _T_precios()
    rob = next(x for x in L.tool_precio_productor(t, pais="Thailand")["tipos"] if x["tipo"] == "Robusta")
    texto = (f"Con los datos disponibles ({rob['primer_anio']}–{rob['ultimo_anio']}), el precio real al productor del Robusta en "
             f"Tailandia pasó de {rob['precio_real_primer_anio']} a {rob['precio_real_ultimo_anio']} US$/kg "
             f"({rob['variacion_real_total_pct']}%), con mínimo de {rob['minimo']['precio_real']} en {rob['minimo']['anio']} "
             f"y máximo de {rob['maximo']['precio_real']} en {rob['maximo']['anio']}. No hay datos posteriores a {rob['ultimo_anio']}.")
    r = L.preguntar(_client(_api_precio("precio_productor", {"pais": "Tailandia"}, texto)), "¿Cómo van los precios en Tailandia?", t)
    assert r["verificada"], r["problemas"]
    assert r["herramientas"][0] == {"herramienta": "precio_productor", "argumentos": {"pais": "Tailandia"}}
    extrapola = texto + " Hoy debería costar 3.85 US$/kg."                                          # cifra inventada: no está en la herramienta
    malo = L.preguntar(_client(_api_precio("precio_productor", {"pais": "Tailandia"}, extrapola)), "¿Cómo van los precios en Tailandia?", t)
    assert not malo["verificada"] and malo["problemas"][0]["tipo"] == "cifra_no_respaldada"


def test_agent_price_tool_errors_are_returned_not_raised():
    t = _sin_precios(T())
    api = _api_precio("precios_globales", {}, "No tengo datos de precios cargados.")
    r = L.preguntar(_client(api), "¿Precio del café?", t)
    resultado = api.requests[1]["messages"][-1]["content"][0]["content"]
    assert "no están cargados" in resultado and r["verificada"]


# ------------------------------------------------------------------ ejecución
def run_all(verbose: bool = True) -> pd.DataFrame:
    try:
        import anthropic  # noqa: F401
        _httpx()
        sdk = True
    except ImportError:
        sdk = False
    filas = []
    cache_original = L.CACHE_PATH
    for name, fn in sorted((k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)):
        if not sdk and (name.startswith("test_sdk") or name.startswith("test_agent") or "cache_roundtrip" in name
                        or "only_valid_llm" in name
                        or "parallel" in name or "stale_cache" in name
                        or "raw_output" in name
                        or "historical_question" in name
                        or "coffee_type_question" in name):
            filas.append({"prueba": name, "resultado": "OMITIDA (falta anthropic)", "detalle": ""})
            continue
        try:
            fn()
            filas.append({"prueba": name, "resultado": "OK", "detalle": ""})
        except Exception as e:  # noqa: BLE001
            filas.append({"prueba": name, "resultado": "FALLA", "detalle": f"{type(e).__name__}: {str(e)[:160]}"})
            if verbose:
                traceback.print_exc()
    L.CACHE_PATH = cache_original
    df = pd.DataFrame(filas)
    if verbose:
        print(f"{(df.resultado == 'OK').sum()} OK · {(df.resultado == 'FALLA').sum()} fallas · "
              f"{df.resultado.str.startswith('OMITIDA').sum()} omitidas")
    return df


if __name__ == "__main__":
    r = run_all()
    print(r.to_string(index=False))
    raise SystemExit(int((r.resultado == "FALLA").any()))
