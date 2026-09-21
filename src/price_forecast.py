"""Pronóstico de RANGOS de precio del café (Arábica y Robusta), en términos reales.

Principios
----------
* Se modela el log del precio REAL (US$ de 2019 por kg), mensual, 1990-2018 (indicadores ICO deflactados con el IPC).
* Se pronostica el PROMEDIO ANUAL (año calendario) a k = 1..6 años desde un origen (diciembre del año `origen`).
* Solo 3 modelos, sin hiperparámetros elegidos a mano:  ingenuo · reversión a la media AR(1) · combinación 50/50.
* Los rangos salen de los errores reales del backtest: banda simétrica en logs con semiancho z·RMS del error por
  horizonte, AGRUPANDO ambas series (más datos por horizonte). Se probaron cuantiles empíricos y bandas gaussianas por
  serie: subestimaban la incertidumbre (cobertura fuera de muestra 58-72%). La cobertura efectiva se mide y se reporta.
* Ningún cálculo con fecha posterior al origen: ver `test_price_forecast.py::test_no_look_ahead`.
* La validación final usa precios REALIZADOS 2019-2025 que NO participaron en la elección del modelo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

import external_data as E

SERIES = {"other_milds": "Arábica (Other Milds)", "robustas": "Robusta"}
MODELOS = ("naive", "meanrev", "blend")
K_MAX = 6
PRIMER_ORIGEN = 1999                     # 10 años de historia mínima (120 meses)
ULTIMO_ANIO_DATOS = 2018
NIVEL = 0.80


# ------------------------------------------------------------------ datos
def preparar_series(raw=E.RAW) -> dict[str, pd.Series]:
    """Precio real mensual (US$ de 2019/kg) por serie, indexado por fecha."""
    df, _ = E.construir_precios(raw)
    df = df.set_index("fecha")
    return {c: df[f"{c}_real"].astype(float) for c in SERIES}


def cargar_realizado(raw=E.RAW) -> pd.DataFrame:
    """Precios realizados 2019-2025 (nominal) y su versión real en US$ de 2019 cuando hay IPC anual."""
    d = pd.read_csv(raw / "precios_realizados_2019_2025.csv")
    base = float(E.ipc_anual(E.leer_ipc(raw)).loc[E.BASE_YEAR])
    for nom, col in (("other_milds", "arabica_usd_kg"), ("robustas", "robusta_usd_kg")):
        d[f"{nom}_nominal"] = d[col]
        d[f"{nom}_real"] = d[col] * base / d["cpi_u_annual_avg"]          # NaN en 2025 (sin IPC anual)
    return d[["year"] + [f"{n}_{t}" for n in SERIES for t in ("nominal", "real")] + ["cpi_u_annual_avg"]]


# ------------------------------------------------------------------ modelos (sobre log precio real mensual)
def _ar1(y: np.ndarray) -> tuple[float, float]:
    """AR(1) por MCO: y_t = c + phi*y_{t-1}. Devuelve (phi, media de largo plazo)."""
    phi, c = np.polyfit(y[:-1], y[1:], 1)
    phi = float(min(phi, 0.9995))
    return phi, float(c / (1 - phi))


def pronostico_mensual(y: np.ndarray, modelo: str, horizonte: int) -> np.ndarray:
    """Log-precio pronosticado para h = 1..horizonte meses."""
    if modelo not in MODELOS:
        raise ValueError(f"modelo desconocido: {modelo}")
    ultimo = y[-1]
    naive = np.repeat(ultimo, horizonte)
    if modelo == "naive":
        return naive
    phi, mu = _ar1(y)
    h = np.arange(1, horizonte + 1)
    meanrev = mu + phi ** h * (ultimo - mu)
    return meanrev if modelo == "meanrev" else 0.5 * (naive + meanrev)


def pronostico_anual(serie: pd.Series, origen: int, modelo: str, k_max: int = K_MAX) -> np.ndarray:
    """Promedio anual pronosticado para los años origen+1 .. origen+k_max, usando SOLO datos hasta dic del origen."""
    hist = serie.loc[: f"{origen}-12-31"]
    if len(hist) < 120:
        raise ValueError("historia insuficiente (<120 meses)")
    f = np.exp(pronostico_mensual(np.log(hist.values), modelo, 12 * k_max))
    return f.reshape(k_max, 12).mean(axis=1)


def promedio_anual(serie: pd.Series, anio: int) -> float:
    return float(serie[serie.index.year == anio].mean())


# ------------------------------------------------------------------ backtest
def backtest(series: dict[str, pd.Series], k_max: int = K_MAX) -> pd.DataFrame:
    """Origen móvil (cada diciembre desde 1999). Error en logs: log(real / pronóstico)."""
    filas = []
    for nombre, s in series.items():
        for origen in range(PRIMER_ORIGEN, ULTIMO_ANIO_DATOS):
            for modelo in MODELOS:
                f = pronostico_anual(s, origen, modelo, k_max)
                for k in range(1, k_max + 1):
                    if origen + k > ULTIMO_ANIO_DATOS:
                        break
                    real = promedio_anual(s, origen + k)
                    filas.append((nombre, modelo, origen, k, f[k - 1], real, np.log(real / f[k - 1])))
    return pd.DataFrame(filas, columns=["serie", "modelo", "origen", "k", "pronostico", "real", "log_err"])


def metricas(bt: pd.DataFrame) -> pd.DataFrame:
    g = bt.groupby(["serie", "modelo", "k"])["log_err"]
    m = g.agg(n="size", sesgo_log="median").reset_index()
    m["mdape"] = g.apply(lambda e: float(np.median(np.abs(np.exp(e) - 1)))).values
    return m


def _semiancho(errores, nivel: float = NIVEL) -> float:
    """Semiancho (en logs) de la banda centrada en el pronóstico: z * RMS del error."""
    return float(norm.ppf(0.5 + nivel / 2) * np.sqrt(np.mean(np.square(np.asarray(errores, dtype=float)))))


def calibrar_intervalos(bt: pd.DataFrame, nivel: float = NIVEL) -> pd.DataFrame:
    """Banda por (serie, modelo, k) = pronóstico * exp(±semiancho), con RMS agrupado entre series."""
    filas = []
    for (modelo, k), g in bt.groupby(["modelo", "k"]):
        semi = _semiancho(g["log_err"], nivel)
        for serie in sorted(bt["serie"].unique()):
            filas.append((serie, modelo, k, -semi, semi, len(g)))
    return pd.DataFrame(filas, columns=["serie", "modelo", "k", "q_lo", "q_hi", "n"])


def cobertura_loo(bt: pd.DataFrame, nivel: float = NIVEL) -> pd.DataFrame:
    """Cobertura honesta: cada origen se evalúa con una banda calibrada SIN él ni los orígenes que se solapan (|Δ| < k)."""
    filas = []
    for (modelo, k), g in bt.groupby(["modelo", "k"]):
        for serie, gs in g.groupby("serie"):
            dentro = []
            for _, r in gs.iterrows():
                lejos = g[(g["origen"] - r["origen"]).abs() >= k]
                if (lejos["serie"] == serie).sum() < 5:
                    continue
                dentro.append(abs(r["log_err"]) <= _semiancho(lejos["log_err"], nivel))
            if dentro:
                filas.append((serie, modelo, k, len(dentro), float(np.mean(dentro))))
    return pd.DataFrame(filas, columns=["serie", "modelo", "k", "n_eval", "cobertura"])


# ------------------------------------------------------------------ pronóstico con rangos y validación fuera de muestra
def pronostico_con_rangos(series: dict[str, pd.Series], bt: pd.DataFrame, origen: int, modelo: str,
                          k_max: int = K_MAX, nivel: float = NIVEL) -> pd.DataFrame:
    cal = calibrar_intervalos(bt, nivel).set_index(["serie", "modelo", "k"])
    filas = []
    for nombre, s in series.items():
        f = pronostico_anual(s, origen, modelo, k_max)
        for k in range(1, k_max + 1):
            c = cal.loc[(nombre, modelo, k)]
            filas.append((nombre, origen + k, k, f[k - 1], f[k - 1] * np.exp(c.q_lo), f[k - 1] * np.exp(c.q_hi), int(c.n)))
    return pd.DataFrame(filas, columns=["serie", "anio", "k", "central", "lo", "hi", "n_calibracion"])


def validar_fuera_de_muestra(rangos: pd.DataFrame, realizado: pd.DataFrame) -> pd.DataFrame:
    """Compara el pronóstico hecho con datos hasta 2018 contra los precios REALES posteriores (US$ de 2019)."""
    r = realizado.set_index("year")
    filas = []
    for _, f in rangos.iterrows():
        col = f"{f.serie}_real"
        if f.anio not in r.index or pd.isna(r.loc[f.anio, col]):
            continue
        real = float(r.loc[f.anio, col])
        filas.append((f.serie, int(f.anio), int(f.k), f.central, f.lo, f.hi, real, real / f.central - 1,
                      bool(f.lo <= real <= f.hi), "arriba" if real > f.hi else ("abajo" if real < f.lo else "dentro")))
    return pd.DataFrame(filas, columns=["serie", "anio", "k", "central", "lo", "hi", "real", "desvio_vs_central", "dentro", "posicion"])
