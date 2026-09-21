"""Pruebas del ETL de datos externos (sin red). Ejecutar:  python test_external_data.py   (o pytest)."""
from __future__ import annotations

import traceback

import numpy as np
import pandas as pd

import external_data as E

_cache = {}


def extra():
    if "extra" not in _cache:
        _cache["extra"], _cache["rep"] = E.construir_extra()
    return _cache["extra"], _cache["rep"]


# ------------------------------------------------------------------ IPC
def test_cpi_matches_published_bls_annual_averages():
    """Promedios anuales publicados por el BLS en la misma tabla de donde se transcribió la serie mensual."""
    publicados = {1990: 130.7, 1996: 156.9, 2000: 172.2, 2008: 215.303, 2010: 218.056, 2016: 240.007, 2019: 255.657}
    ipc = E.ipc_anual(E.leer_ipc())
    for y, v in publicados.items():
        assert abs(ipc[y] - v) <= 0.051, (y, ipc[y], v)
    assert len(E.leer_ipc()) == 360                                   # 30 años x 12 meses


def test_cpi_is_complete_and_positive():
    ipc = E.leer_ipc()
    assert ipc["cpi_u_nsa"].notna().all() and (ipc["cpi_u_nsa"] > 100).all()
    assert ipc["fecha"].min() == pd.Timestamp("1990-01-01") and ipc["fecha"].max() == pd.Timestamp("2019-12-01")


# ------------------------------------------------------------------ unidades
def test_cents_per_lb_conversion():
    assert abs(100 * E.CENTAVOS_LB_A_USD_KG - 2.2046226) < 1e-6
    # Colombia 1990: 69.523 c/lb en coffeedata vs 1.535 US$/kg en prices-paid-to-growers de la ICO
    assert abs(69.523 * E.CENTAVOS_LB_A_USD_KG - 1.535) / 1.535 < 0.005


def test_thousand_bags_to_kg():
    assert E.MILES_SACOS_A_KG == 60000
    ex, _ = extra()
    br = ex[(ex.country == "Brazil") & (ex.year == 1990)].iloc[0]
    assert br["domestic_consumption_kg"] == 492_000_000          # 8 200 miles de sacos x 60 000


# ------------------------------------------------------------------ diseño: inner join y sin duplicados por tipo de café
def test_extra_is_unique_per_country_year_and_inner_joined():
    ex, rep = extra()
    assert not ex.duplicated(["country", "year"]).any()
    paises, cons = E.leer_parquet_principal()
    assert set(ex["country"]) <= set(paises)                       # solo países del parquet principal
    assert "Benin" not in set(ex["country"]) and "Benin" in rep["descartados_por_inner_join"]
    assert rep["paises_parquet_sin_datos_extra"] == []
    assert ex["year"].between(1990, 2019).all()


def test_coffee_type_duplication_is_collapsed():
    raw = pd.read_csv(E.RAW / "coffeedata.csv")
    dup = raw.groupby(["Country", "Year"]).size()
    assert (dup == 2).sum() == 480                                  # 480 país-año venían duplicados por grupo de café
    cd = E.cargar_coffeedata()
    assert not cd.duplicated(["country", "year"]).any()
    # producción idéntica entre grupos -> no se altera; el precio se promedia entre los grupos
    b = raw[(raw.Country == "Brazil") & (raw.Year == 1990)]
    assert b.Production.nunique() == 1 and len(b) == 2
    esperado = b.Price_grower.mean() * E.CENTAVOS_LB_A_USD_KG
    fila = cd[(cd.country == "Brazil") & (cd.year == 1990)].iloc[0]
    assert abs(fila["price_grower_usd_kg"] - esperado) < 1e-9 and fila["n_price_groups"] == 2


def test_production_is_identical_across_coffee_type_rows():
    raw = pd.read_csv(E.RAW / "coffeedata.csv")
    g = raw.groupby(["Country", "Year"])[["Production", "Exports", "Gross_opening_stocks", "Domestic_consumption"]].nunique()
    assert (g.max().max()) == 1, "si difiere entre grupos, 'first' sería incorrecto"


def test_consumption_column_equals_main_parquet():
    ex, _ = extra()
    _, cons = E.leer_parquet_principal()
    m = ex.merge(cons, on=["country", "year"], suffixes=("", "_p"))
    assert len(m) == len(ex) and (m["domestic_consumption_kg"] == m["domestic_consumption_kg_p"]).all()


# ------------------------------------------------------------------ calidad
def test_stock_balance_closes_with_crop_year_exports():
    ex, _ = extra()
    b = ex[ex["production_kg"] / 60 / 1000 > 500]["balance_gap_pct"].abs().dropna()
    assert len(b) > 600 and b.median() < 0.005


def test_no_negative_values_and_coverage():
    ex, _ = extra()
    for c in ["production_kg", "exports_crop_kg", "exports_cal_kg", "opening_stocks_kg", "price_grower_usd_kg"]:
        assert (ex[c].dropna() >= 0).all(), c
    assert ex.groupby("year").country.nunique().loc[2018] == 55
    assert ex.groupby("year").country.nunique().loc[2019] == 48    # coffeedata.csv no trae 7 países en 2019


def test_source_switch_in_2019_has_no_artificial_jump():
    ex, _ = extra()
    p = ex.pivot(index="country", columns="year", values="production_kg")
    r = (p[2019] / p[2018]).replace([np.inf, -np.inf], np.nan).dropna()[p[2018] > 0.5 * 60e6]
    assert 0.85 < r.median() < 1.05 and len(r) >= 15
    assert set(ex[ex.year == 2019].source_supply) == {"coffeedata"} and set(ex[ex.year <= 2018].source_supply) == {"ico_csv"}


# ------------------------------------------------------------------ precios
def test_real_prices_formula_and_deflation_direction():
    df, base = E.construir_precios()
    assert abs(base - 255.657) < 0.01
    for c in E.GRUPOS_PRECIO.values():
        assert np.allclose(df[f"{c}_real"] * df["cpi_u_nsa"] / base, df[c])
    assert (df["ico_composite_real"] > df["ico_composite"]).all()   # todo el período es anterior a 2019: real > nominal
    sep01 = df[df.fecha == "2001-09-01"].iloc[0]
    assert abs(sep01["ico_composite"] - 0.909) < 0.001              # mínimo histórico (41.2 c/lb = 0.908 US$/kg)


def test_crop_year_assignment_and_completeness():
    df, _ = E.construir_precios()
    ass = df.set_index("fecha")["crop_year"]
    assert ass["1990-10-01"] == 1990 and ass["1991-09-01"] == 1990 and ass["1991-01-01"] == 1990 and ass["1991-10-01"] == 1991
    an = E.precios_por_anio_cafetero(df)
    completos = an[an.completo]
    assert completos.crop_year.min() == 1990 and completos.crop_year.max() == 2017 and len(completos) == 28


def test_real_price_declined_while_nominal_rose():
    df, _ = E.construir_precios()
    a = E.precios_por_anio_cafetero(df).set_index("crop_year")
    assert a.loc[2017, "ico_composite"] > a.loc[1990, "ico_composite"]            # nominal sube
    assert a.loc[2017, "ico_composite_real"] < a.loc[1990, "ico_composite_real"]  # real baja


def test_grower_price_real_uses_annual_cpi():
    ex, rep = extra()
    fila = ex[(ex.country == "Colombia") & (ex.year == 1990)].iloc[0]
    esperado = fila["price_grower_usd_kg"] * rep["ipc_base"] / E.ipc_anual(E.leer_ipc())[1990]
    assert abs(fila["price_grower_real_usd_kg"] - esperado) < 1e-9


def test_build_is_reproducible():
    a, _ = E.construir_extra()
    b, _ = E.construir_extra()
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
