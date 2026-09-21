"""Datos externos como parquets ADICIONALES (ICO + IPC del BLS).

Diseño
------
* `coffee_extra.parquet`   nivel país-año, SIN duplicar por tipo de café, y solo con los países del parquet
                           principal (inner join). Oferta, existencias, exportaciones y precio al productor.
* `prices_global.parquet`  serie mensual de precios internacionales (nominal y real) : no es de un país.
* `prices_crop_year.parquet` mismo precio promediado por año cafetero (oct-sep), alineado con las temporadas.

Por qué el precio internacional va aparte: repetirlo en las 55 filas de cada año lo duplicaría 55 veces
sin aportar información; se une por `crop_year` cuando se necesita.

Insumos (carpeta `datos_externos/`):
    total-production.csv, exports-crop-year.csv, exports-calendar-year.csv, gross-opening-stocks.csv,
    indicator-prices.csv (ICO), coffeedata.csv (ICO, hasta 2019) y cpi_us_bls.csv (BLS).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils import OUT, ROOT

RAW = ROOT / "datos_externos"
PARQUET = ROOT / "coffee_db.parquet"

BAG_KG = 60.0                                  # un saco = 60 kg
MILES_SACOS_A_KG = 1000 * BAG_KG               # 1 (miles de sacos) = 60 000 kg
LB_A_KG = 0.45359237
CENTAVOS_LB_A_USD_KG = 1 / LB_A_KG / 100       # 100 c/lb = 2.2046 US$/kg
BASE_YEAR = 2019                               # precios reales en dólares de 2019 (último año del parquet)
ANIOS = list(range(1990, 2020))                # 1990/91 ... 2019/20

GRUPOS_PRECIO = {"ICO composite indicator": "ico_composite", "Colombian Milds": "colombian_milds",
                 "Other Milds": "other_milds", "Brazilian Naturals": "brazilian_naturals", "Robustas": "robustas"}


# ------------------------------------------------------------------ lectura
def leer_ico_ancho(path) -> pd.DataFrame:
    """CSV ancho de la ICO (una fila por país, una columna por año) -> largo: country, year, valor."""
    d = pd.read_csv(path)
    k = d.columns[0]
    d[k] = d[k].astype(str).str.strip()
    largo = d.melt(id_vars=k, var_name="year", value_name="valor").rename(columns={k: "country"})
    largo["year"] = largo["year"].astype(int)
    return largo


def leer_parquet_principal(path=PARQUET) -> tuple[list, pd.DataFrame]:
    """Países del parquet principal y su consumo doméstico (kg) en formato largo."""
    w = pd.read_parquet(path)
    temporadas = [c for c in w.columns if "/" in c]
    largo = w.melt(id_vars="Country", value_vars=temporadas, var_name="season", value_name="domestic_consumption_kg")
    largo = largo.rename(columns={"Country": "country"})
    largo["year"] = largo["season"].str[:4].astype(int)
    return sorted(w["Country"].unique()), largo[["country", "year", "season", "domestic_consumption_kg"]]


# ------------------------------------------------------------------ IPC y precios internacionales
def leer_ipc(raw=RAW) -> pd.DataFrame:
    return pd.read_csv(raw / "cpi_us_bls.csv", parse_dates=["fecha"])[["fecha", "cpi_u_nsa"]]


def ipc_anual(ipc: pd.DataFrame) -> pd.Series:
    """Promedio anual del IPC-U (año calendario)."""
    return ipc.groupby(ipc["fecha"].dt.year)["cpi_u_nsa"].mean()


def construir_precios(raw=RAW, base_year: int = BASE_YEAR):
    """Precios internacionales mensuales, nominales (US$/kg) y reales (US$ de `base_year`), con año cafetero."""
    ip = pd.read_csv(raw / "indicator-prices.csv")
    ip["fecha"] = pd.to_datetime(ip["months"], format="%m/%Y")
    ip = ip.rename(columns=GRUPOS_PRECIO)[["fecha"] + list(GRUPOS_PRECIO.values())]
    ipc = leer_ipc(raw)
    base = float(ipc.loc[ipc["fecha"].dt.year == base_year, "cpi_u_nsa"].mean())
    df = ip.merge(ipc, on="fecha", how="left")
    for c in GRUPOS_PRECIO.values():
        df[f"{c}_real"] = df[c] * base / df["cpi_u_nsa"]
    df["crop_year"] = np.where(df["fecha"].dt.month >= 10, df["fecha"].dt.year, df["fecha"].dt.year - 1)
    return df, base


def precios_por_anio_cafetero(df: pd.DataFrame) -> pd.DataFrame:
    """Promedio por año cafetero (oct-sep). `completo` = 12 meses disponibles."""
    cols = [c for c in df.columns if c in GRUPOS_PRECIO.values() or c.endswith("_real")]
    g = df.groupby("crop_year")
    out = g[cols].mean()
    out["n_meses"] = g["fecha"].count()
    out["completo"] = out["n_meses"] == 12
    return out.reset_index()


# ------------------------------------------------------------------ oferta (nivel país-año, sin duplicar por tipo de café)
def cargar_oferta_ico(raw=RAW) -> pd.DataFrame:
    """Producción, exportaciones (año cafetero y calendario) y existencias iniciales, 1990-2018, en kg."""
    archivos = {"production_kg": "total-production.csv", "exports_crop_kg": "exports-crop-year.csv",
                "exports_cal_kg": "exports-calendar-year.csv", "opening_stocks_kg": "gross-opening-stocks.csv"}
    partes = []
    for col, fn in archivos.items():
        d = leer_ico_ancho(raw / fn)
        d[col] = d.pop("valor") * MILES_SACOS_A_KG
        partes.append(d.set_index(["country", "year"]))
    return pd.concat(partes, axis=1).reset_index()


def cargar_coffeedata(raw=RAW) -> pd.DataFrame:
    """coffeedata.csv trae una fila por grupo de café de la ICO: se COLAPSA a país-año.

    Producción, exportaciones y existencias se repiten idénticas entre grupos (se toma la primera).
    El precio al productor sí cambia por grupo, y se entrega de DOS formas:
      * por tipo de café (`price_grower_arabica_usd_kg`, `price_grower_robusta_usd_kg`): USAR ESTA. Los grupos
        Colombian Milds, Other Milds y Brazilian Naturals son Arábica; Robustas es Robusta.
      * `price_grower_usd_kg`: promedio de los grupos que reportan. Se conserva por compatibilidad, pero SALTA cuando
        cambia cuántos grupos reportan (Tailandia: solo Robusta hasta 2012 y Robusta+Arábica desde 2013 -> el promedio
        mezcla dos mercados; su +69.5% 2008->2013 es +14.5% si se compara Robusta con Robusta).
    Precio en centavos/lb -> US$/kg.
    """
    cd = pd.read_csv(raw / "coffeedata.csv")
    cd["tipo_precio"] = np.where(cd["Coffee_type"] == "Robustas", "robusta", "arabica")
    por_tipo = cd.pivot_table(index=["Country", "Year"], columns="tipo_precio", values="Price_grower", aggfunc="mean")
    por_tipo.columns = [f"price_grower_{c}_usd_kg" for c in por_tipo.columns]
    por_tipo = (por_tipo * CENTAVOS_LB_A_USD_KG).reset_index().rename(columns={"Country": "country", "Year": "year"})
    g = cd.groupby(["Country", "Year"], as_index=False).agg(
        production_kg=("Production", "first"), exports_cal_kg=("Exports", "first"),
        opening_stocks_kg=("Gross_opening_stocks", "first"), price_c_lb=("Price_grower", "mean"),
        n_price_groups=("Price_grower", "count"))
    for c in ("production_kg", "exports_cal_kg", "opening_stocks_kg"):
        g[c] = g[c] * MILES_SACOS_A_KG
    g["price_grower_usd_kg"] = g.pop("price_c_lb") * CENTAVOS_LB_A_USD_KG
    g = g.rename(columns={"Country": "country", "Year": "year"})
    return g.merge(por_tipo, on=["country", "year"], how="left")


def construir_extra(parquet=PARQUET, raw=RAW):
    """Parquet adicional país-año, con inner join a los países del parquet principal.

    Devuelve (extra, reporte). Regla de fuentes: ICO (hasta 2018) y, donde falta (2019), coffeedata.csv.
    """
    paises, cons = leer_parquet_principal(parquet)
    ico, cd = cargar_oferta_ico(raw), cargar_coffeedata(raw)

    ico["source_supply"] = "ico_csv"
    columnas_precio = ["price_grower_usd_kg", "n_price_groups", "price_grower_arabica_usd_kg", "price_grower_robusta_usd_kg"]
    cd_s = cd.drop(columns=columnas_precio).assign(source_supply="coffeedata")
    llaves = ["country", "year"]
    # combinar: se prefiere ICO; coffeedata solo rellena lo que ICO no tiene
    oferta = ico.set_index(llaves).combine_first(cd_s.set_index(llaves)).reset_index()
    oferta["source_supply"] = np.where(oferta["year"] <= 2018, "ico_csv", "coffeedata")
    oferta = oferta.merge(cd[llaves + columnas_precio], on=llaves, how="left")

    # ---- INNER JOIN con el parquet principal: solo sus países y su ventana de años
    antes = set(oferta["country"])
    oferta = oferta[oferta["country"].isin(paises) & oferta["year"].isin(ANIOS)]
    extra = oferta.merge(cons, on=llaves, how="inner")

    # ---- precio al productor en términos reales (US$ de 2019), con el IPC anual de EE. UU.
    ipc_a = ipc_anual(leer_ipc(raw))
    base = float(ipc_a.loc[BASE_YEAR])
    deflactor = base / extra["year"].map(ipc_a)
    for c in ("price_grower_usd_kg", "price_grower_arabica_usd_kg", "price_grower_robusta_usd_kg"):
        extra[c.replace("_usd_kg", "_real_usd_kg")] = extra[c] * deflactor

    # ---- indicadores derivados (año cafetero: exportaciones por año cafetero, que sí cuadran con existencias)
    extra = extra.sort_values(llaves).reset_index(drop=True)
    prod = extra["production_kg"].where(extra["production_kg"] > 0)
    extra["consumption_share_prod"] = extra["domestic_consumption_kg"] / prod
    extra["export_crop_share_prod"] = extra["exports_crop_kg"] / prod
    extra["implied_closing_stocks_kg"] = (extra["production_kg"] + extra["opening_stocks_kg"]
                                          - extra["exports_crop_kg"] - extra["domestic_consumption_kg"])
    siguiente = extra.groupby("country")["opening_stocks_kg"].shift(-1)
    extra["balance_gap_pct"] = (extra["implied_closing_stocks_kg"] - siguiente) / prod

    cols = ["country", "year", "season", "production_kg", "exports_crop_kg", "exports_cal_kg", "opening_stocks_kg",
            "domestic_consumption_kg", "consumption_share_prod", "export_crop_share_prod", "implied_closing_stocks_kg",
            "balance_gap_pct", "price_grower_arabica_usd_kg", "price_grower_arabica_real_usd_kg",
            "price_grower_robusta_usd_kg", "price_grower_robusta_real_usd_kg",
            "price_grower_usd_kg", "price_grower_real_usd_kg", "n_price_groups", "source_supply"]
    extra = extra[cols]
    reporte = {
        "paises_parquet": len(paises), "paises_extra": extra["country"].nunique(),
        "paises_parquet_sin_datos_extra": sorted(set(paises) - set(extra["country"])),
        "descartados_por_inner_join": sorted(antes - set(paises)),
        "filas": len(extra), "anio_base_real": BASE_YEAR, "ipc_base": base,
    }
    return extra, reporte


def guardar(extra: pd.DataFrame, precios: pd.DataFrame, por_anio: pd.DataFrame, out=OUT) -> None:
    out.mkdir(exist_ok=True)
    extra.to_parquet(out / "coffee_extra.parquet", index=False)
    precios.to_parquet(out / "prices_global.parquet", index=False)
    por_anio.to_parquet(out / "prices_crop_year.parquet", index=False)
