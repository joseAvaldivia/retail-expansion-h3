"""De la predicción a la decisión: venta incremental de abrir una tienda, canibalización,
lista corta de aperturas y ranking de municipios."""
from __future__ import annotations

from dataclasses import dataclass

import h3
import numpy as np
import pandas as pd

from .modelo import predecir_por_tienda


@dataclass(frozen=True)
class Reglas:
    canibal_anillo1: float = 0.15   # cada tienda propia a 1 anillo resta 15 % a la nueva
    canibal_anillo2: float = 0.05
    canibal_max: float = 0.60
    poblacion_minima: int = 500
    distancia_minima: int = 3       # anillos H3 entre aperturas de la lista corta
    n_aperturas: int = 20


def puntuar(df: pd.DataFrame, modelo, reglas: Reglas = Reglas()) -> pd.DataFrame:
    """Agrega venta predicha, venta incremental y venta neta de canibalización por hexágono."""
    df = df.copy()
    n = df["farmacias_propias"].to_numpy(dtype=float)
    actual = predecir_por_tienda(modelo, df, n) * n                 # venta del hexágono hoy
    con_una_mas = predecir_por_tienda(modelo, df, n + 1) * (n + 1)  # con una tienda más
    df["venta_predicha_tienda_nueva"] = (con_una_mas / (n + 1)).round(0)
    df["venta_incremental"] = (con_una_mas - actual).round(0)
    canib = np.minimum(reglas.canibal_max,
                       reglas.canibal_anillo1 * df["propias_anillo1"] + reglas.canibal_anillo2 * df["propias_anillo2"])
    df["canibalizacion_vecinos"] = canib.round(3)
    df["venta_neta"] = (df["venta_incremental"] * (1 - canib)).round(0)
    df.loc[df["poblacion"] < reglas.poblacion_minima, "venta_neta"] = 0.0
    df["venta_predicha_hex"] = np.where(n > 0, actual, df["venta_predicha_tienda_nueva"]).round(0)
    return df


def lista_corta(df: pd.DataFrame, reglas: Reglas = Reglas()) -> pd.DataFrame:
    """Selección voraz: el hexágono con mayor venta neta, sin repetir zonas a < distancia_minima."""
    candidatos = df[df["venta_neta"] > 0].sort_values("venta_neta", ascending=False)
    elegidos: list[str] = []
    for celda in candidatos["h3"]:
        if all(h3.grid_distance(celda, e) >= reglas.distancia_minima
               for e in elegidos if h3.get_resolution(e) == h3.get_resolution(celda)):
            elegidos.append(celda)
        if len(elegidos) == reglas.n_aperturas:
            break
    out = df.set_index("h3").loc[elegidos].reset_index()
    out.insert(0, "prioridad", np.arange(1, len(out) + 1))
    return out[["prioridad", "h3", "ciudad", "municipio", "poblacion", "ingreso_pc", "competidores",
                "farmacias_propias", "venta_predicha_tienda_nueva", "venta_incremental",
                "canibalizacion_vecinos", "venta_neta", "lat", "lng"]]


def ranking_municipios(df: pd.DataFrame, aperturas: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["ciudad", "municipio"]).agg(
        hexagonos=("h3", "size"), poblacion=("poblacion", "sum"), ingreso_pc=("ingreso_pc", "mean"),
        farmacias_propias=("farmacias_propias", "sum"), competidores=("competidores", "sum"),
        venta_actual=("ventas_mes", "sum"), potencial_neto=("venta_neta", lambda s: s[s > 0].sum()),
    )
    g["habitantes_por_farmacia"] = (g["poblacion"] / (g["farmacias_propias"] + g["competidores"]).replace(0, np.nan)).round(0)
    g["propias_por_10k_hab"] = (g["farmacias_propias"] / g["poblacion"].replace(0, np.nan) * 1e4).round(2)
    g["aperturas_en_lista"] = aperturas.groupby("municipio").size().reindex(g.index.get_level_values(1)).fillna(0).astype(int).to_numpy()
    g["venta_neta_lista"] = aperturas.groupby("municipio")["venta_neta"].sum().reindex(g.index.get_level_values(1)).fillna(0).to_numpy()
    return g.sort_values("venta_neta_lista", ascending=False).reset_index().round(1)
