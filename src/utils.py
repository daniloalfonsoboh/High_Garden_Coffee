"""Utilidades compartidas por los notebooks del caso High Garden Coffee.

Vive junto a los notebooks y al archivo `coffee_db.parquet`.
Las rutas se resuelven respecto a este archivo, así que funcionan
sin importar desde qué carpeta abras VS Code.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "coffee_db.parquet"
OUT = ROOT / "outputs"
FIGS = OUT / "figs"
OUT.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

# Mínimo de temporadas positivas consecutivas (hasta 2019/20) para modelar un país
MIN_POSITIVE_SEASONS = 15

SHORT_NAMES = {
    "Bolivia (Plurinational State of)": "Bolivia",
    "Lao People's Democratic Republic": "Laos",
    "Democratic Republic of Congo": "RD Congo",
    "Trinidad & Tobago": "Trinidad y Tobago",
    "Viet Nam": "Vietnam",
}

# Robusta/Arabica y Arabica/Robusta se agrupan como "Mixto"
GROUP_MAP = {
    "Arabica": "Arabica",
    "Robusta": "Robusta",
    "Robusta/Arabica": "Mixto",
    "Arabica/Robusta": "Mixto",
}


# ----------------------------------------------------------------- carga
def load_wide() -> pd.DataFrame:
    """Lee el parquet original (formato ancho: una columna por temporada)."""
    return pd.read_parquet(DATA_PATH)


def season_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if "/" in c]


def to_long(wide: pd.DataFrame) -> pd.DataFrame:
    """Pasa a formato largo: country, coffee_type, season, year, consumption."""
    long = wide.melt(
        id_vars=["Country", "Coffee type"],
        value_vars=season_cols(wide),
        var_name="season",
        value_name="consumption",
    ).rename(columns={"Country": "country", "Coffee type": "coffee_type"})
    long["country"] = long["country"].replace(SHORT_NAMES)
    long["year"] = long["season"].str[:4].astype(int)  # 1990/91 -> 1990
    long["coffee_group"] = long["coffee_type"].map(GROUP_MAP)
    long["consumption_m"] = long["consumption"] / 1e6  # millones de unidades originales
    return long.sort_values(["country", "year"]).reset_index(drop=True)


def load_long() -> pd.DataFrame:
    """Lee el formato largo generado por 01_eda (con la marca in_universe)."""
    return pd.read_parquet(OUT / "coffee_long.parquet")


def _positive_tail(values) -> int:
    n = 0
    for v in values[::-1]:
        if v > 0:
            n += 1
        else:
            break
    return n


def modelable_countries(long: pd.DataFrame, min_seasons: int = MIN_POSITIVE_SEASONS) -> list:
    """Países con al menos `min_seasons` temporadas positivas consecutivas hasta el final."""
    tail = long.sort_values("year").groupby("country")["consumption"].apply(
        lambda s: _positive_tail(s.values)
    )
    return sorted(tail[tail >= min_seasons].index)


def get_series(long: pd.DataFrame, country: str) -> pd.Series:
    """Serie anual (índice = año de inicio de temporada) sin los ceros iniciales."""
    s = long[long["country"] == country].set_index("year")["consumption_m"].sort_index()
    return s[s.index >= s[s > 0].index.min()]


# ------------------------------------------------------------ crecimiento
def cagr_smooth(y, smooth: int = 3) -> float:
    """CAGR suavizado: compara el promedio de las primeras vs últimas `smooth` observaciones."""
    y = np.asarray(y, dtype=float)
    start, end = y[:smooth].mean(), y[-smooth:].mean()
    periods = len(y) - smooth
    return (end / start) ** (1 / periods) - 1


def log_trend(y) -> float:
    """Pendiente de log(y) vs tiempo ≈ crecimiento anual continuo."""
    y = np.asarray(y, dtype=float)
    return float(np.polyfit(np.arange(len(y)), np.log(y), 1)[0])


# ---------------------------------------------------------------- métricas
def wape(actual, pred) -> float:
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    return float(np.abs(actual - pred).sum() / np.abs(actual).sum())


def smape(actual, pred) -> float:
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    return float(np.mean(2 * np.abs(actual - pred) / (np.abs(actual) + np.abs(pred))))


def mase(actual, pred, train) -> float:
    """Error absoluto medio escalado por el error del naive dentro de la muestra de entrenamiento."""
    scale = np.mean(np.abs(np.diff(np.asarray(train, float))))
    if scale == 0:
        return np.nan
    return float(np.mean(np.abs(np.asarray(actual, float) - np.asarray(pred, float))) / scale)


# ------------------------------------------------------------------ gráficos
def save_fig(name: str, fig=None):
    """Guarda la figura en outputs/figs (se reutilizan en la presentación)."""
    import matplotlib.pyplot as plt

    fig = fig or plt.gcf()
    fig.savefig(FIGS / f"{name}.png", dpi=150, bbox_inches="tight")
