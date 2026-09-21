"""Pruebas del pronóstico de precios (sin red). Ejecutar:  python test_price_forecast.py   (o pytest)."""
from __future__ import annotations

import traceback
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

import external_data as E
import price_forecast as P

warnings.filterwarnings("ignore")
_c = {}


def datos():
    if "s" not in _c:
        _c["s"] = P.preparar_series()
        _c["bt"] = P.backtest(_c["s"])
        _c["real"] = P.cargar_realizado()
    return _c["s"], _c["bt"], _c["real"]


# ------------------------------------------------------------------ datos realizados
def test_realized_data_conversions():
    _, _, real = datos()
    r = real.set_index("year")
    assert abs(r.loc[2021, "other_milds_nominal"] - 204.69403 * E.CENTAVOS_LB_A_USD_KG) < 1e-3     # centavos/lb -> US$/kg
    assert abs(r.loc[2021, "other_milds_nominal"] - 4.5127) < 5e-4
    assert r.loc[2020, "cpi_u_annual_avg"] == 258.811
    # 2019 es el año base: real ≈ nominal (el IPC anual publicado, 255.657, y el promedio mensual, 255.6574, difieren en 2e-6)
    assert abs(r.loc[2019, "other_milds_real"] / r.loc[2019, "other_milds_nominal"] - 1) < 1e-5
    assert r.loc[2022, "other_milds_real"] < r.loc[2022, "other_milds_nominal"]                    # deflactar baja el valor
    assert pd.isna(r.loc[2025, "other_milds_real"]) and pd.notna(r.loc[2025, "other_milds_nominal"])   # sin IPC 2025


def test_ico_file_joins_world_bank_at_2018():
    """Las dos fuentes deben empalmar: promedio 2018 del CSV de la ICO = Pink Sheet (2.93 y 1.87 US$/kg)."""
    ip = pd.read_csv(E.RAW / "indicator-prices.csv")
    ip = ip[ip["months"].str.endswith("2018")]
    assert abs(ip["Other Milds"].mean() - 2.93) < 0.005 and abs(ip["Robustas"].mean() - 1.87) < 0.005


# ------------------------------------------------------------------ modelos
def test_no_look_ahead():
    s, _, _ = datos()
    for serie in s.values():
        for modelo in P.MODELOS:
            completo = P.pronostico_anual(serie, 2008, modelo)
            truncado = P.pronostico_anual(serie.loc[:"2008-12-31"], 2008, modelo)
            assert np.allclose(completo, truncado), modelo


def test_models_behave_as_specified():
    y = np.log(np.array([3.0] * 100 + [1.0] * 40))                    # último valor muy por debajo de la media
    naive = P.pronostico_mensual(y, "naive", 24)
    mr = P.pronostico_mensual(y, "meanrev", 24)
    blend = P.pronostico_mensual(y, "blend", 24)
    assert np.allclose(naive, y[-1])
    assert np.all(np.diff(mr) > 0) and mr[-1] > y[-1]                 # revierte hacia arriba
    assert np.allclose(blend, 0.5 * (naive + mr))
    try:
        P.pronostico_mensual(y, "otro", 3)
        raise AssertionError("debió fallar")
    except ValueError:
        pass
    try:
        P.pronostico_anual(pd.Series(np.ones(60), index=pd.date_range("2000-01-01", periods=60, freq="MS")), 2004, "naive")
        raise AssertionError("debió fallar por historia corta")
    except ValueError:
        pass


def test_annual_aggregation_matches_manual():
    s, _, _ = datos()
    serie = s["robustas"]
    y = np.log(serie.loc[:"2005-12-31"].values)
    manual = np.exp(P.pronostico_mensual(y, "meanrev", 72)).reshape(6, 12).mean(axis=1)
    assert np.allclose(P.pronostico_anual(serie, 2005, "meanrev"), manual)


# ------------------------------------------------------------------ backtest
def test_backtest_alignment_and_counts():
    s, bt, _ = datos()
    esperadas = sum(min(P.K_MAX, P.ULTIMO_ANIO_DATOS - o) for o in range(P.PRIMER_ORIGEN, P.ULTIMO_ANIO_DATOS))
    assert len(bt) == esperadas * len(P.SERIES) * len(P.MODELOS)
    assert (bt["origen"] + bt["k"]).max() <= P.ULTIMO_ANIO_DATOS                # nunca evalúa contra años sin dato
    r = bt[(bt.serie == "other_milds") & (bt.modelo == "naive") & (bt.origen == 2004) & (bt.k == 3)].iloc[0]
    assert abs(r["real"] - P.promedio_anual(s["other_milds"], 2007)) < 1e-12
    assert abs(r["log_err"] - np.log(r["real"] / r["pronostico"])) < 1e-12


def test_metrics_shape_and_sanity():
    _, bt, _ = datos()
    m = P.metricas(bt)
    assert set(m["modelo"]) == set(P.MODELOS) and set(m["k"]) == set(range(1, P.K_MAX + 1))
    assert (m["mdape"] >= 0).all()
    mr = m[m.modelo == "meanrev"].groupby("k")["mdape"].mean()
    assert mr.loc[1] < mr.loc[6]                                               # el error crece con el horizonte


# ------------------------------------------------------------------ intervalos
def test_semiancho_formula():
    assert abs(P._semiancho([0.1, -0.1], 0.8) - norm.ppf(0.9) * 0.1) < 1e-12
    assert P._semiancho([0.2, 0.2], 0.8) > P._semiancho([0.1, 0.1], 0.8)


def test_bands_are_symmetric_pooled_and_grow_with_horizon():
    _, bt, _ = datos()
    cal = P.calibrar_intervalos(bt)
    assert np.allclose(cal["q_lo"], -cal["q_hi"])
    for (modelo, k), g in cal.groupby(["modelo", "k"]):
        assert g["q_hi"].nunique() == 1                                        # ambas series comparten banda por horizonte
    sem = cal[cal.modelo == "meanrev"].groupby("k")["q_hi"].first()
    assert (sem.diff().dropna() > 0).all()


def test_forecast_band_contains_central_and_is_positive():
    s, bt, _ = datos()
    rg = P.pronostico_con_rangos(s, bt, 2018, "meanrev")
    assert (rg["lo"] > 0).all() and (rg["lo"] < rg["central"]).all() and (rg["central"] < rg["hi"]).all()
    assert list(rg["anio"].unique()) == list(range(2019, 2025))


def test_loo_coverage_is_reasonable_and_reported():
    _, bt, _ = datos()
    cov = P.cobertura_loo(bt)
    mr = cov[cov.modelo == "meanrev"]
    por_k = mr.groupby("k").apply(lambda d: np.average(d["cobertura"], weights=d["n_eval"]))
    assert por_k.loc[[1, 2, 3]].mean() >= 0.70            # años 1-3: cobertura efectiva cercana al 80% nominal
    assert por_k.loc[1] <= 0.95                            # y no es trivialmente 100%
    assert mr["n_eval"].min() >= 5


# ------------------------------------------------------------------ validación fuera de muestra
def test_out_of_sample_validation_is_real_and_uses_only_history_to_2018():
    s, bt, real = datos()
    rg = P.pronostico_con_rangos(s, bt, 2018, "meanrev")
    v = P.validar_fuera_de_muestra(rg, real)
    assert len(v) == 12 and v["anio"].min() == 2019 and v["anio"].max() == 2024   # 2025 no tiene IPC anual
    fila = v[(v.serie == "other_milds") & (v.anio == 2022)].iloc[0]
    assert abs(fila["real"] - 5.6321 * 255.657 / 292.655) < 5e-3                  # nominal -> real con el IPC de 2022
    assert v["dentro"].sum() == (v["posicion"] == "dentro").sum()
    # las series usadas para pronosticar terminan en dic-2018
    assert max(x.index.max() for x in s.values()) == pd.Timestamp("2018-12-01")


def test_pipeline_is_reproducible():
    a = P.backtest(P.preparar_series())
    b = P.backtest(P.preparar_series())
    pd.testing.assert_frame_equal(a, b)


def run_all(verbose: bool = True) -> pd.DataFrame:
    filas = []
    for name, fn in sorted((k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)):
        try:
            fn()
            filas.append({"prueba": name, "resultado": "OK", "detalle": ""})
        except Exception as e:  # noqa: BLE001
            filas.append({"prueba": name, "resultado": "FALLA", "detalle": f"{type(e).__name__}: {str(e)[:150]}"})
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
