"""País ficticio y su red de farmacias, a nivel de hexágono H3.

"País Demo" se ubica a propósito en mar abierto del Atlántico Sur: ninguna celda H3 coincide con
un lugar habitado real. Ciudades, municipios, población, ingreso, puntos de interés, farmacias
propias ("Cadena A"), competidores y ventas se generan con una semilla y parámetros escritos a
mano. Las columnas `verdad_*` guardan la demanda real con la que se simularon las ventas, para
medir después si el modelo la recupera.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import h3
import numpy as np
import pandas as pd

RESOLUCION = 8                       # ~0,74 km² por hexágono
CAJA = (-34.0, -29.0, -22.0, -14.0)  # lat_min, lat_max, lng_min, lng_max (océano abierto)
SUFIJOS = ["Centro", "Norte", "Sur", "Este", "Oeste", "Noreste", "Suroeste"]

# nombre, municipios, radio en anillos H3, población máxima por hexágono, ingreso base (UM/año)
CIUDADES = [
    ("Alfa", 7, 10, 10500, 6200), ("Beta", 6, 9, 9000, 5600), ("Gamma", 5, 7, 7000, 5200),
    ("Delta", 4, 6, 6000, 4800), ("Épsilon", 4, 6, 5500, 4500), ("Zeta", 3, 5, 4500, 4300),
    ("Eta", 3, 5, 4200, 4100), ("Theta", 2, 4, 3500, 3900), ("Iota", 2, 4, 3300, 3800),
    ("Kappa", 2, 3, 3000, 3700), ("Lambda", 1, 3, 2600, 3600), ("Mu", 1, 2, 2400, 3500),
    ("Nu", 1, 2, 2200, 3400), ("Ómicron", 1, 2, 2000, 3400), ("Pi", 1, 2, 1900, 3300),
    ("Rho", 1, 2, 1800, 3300),
]


@dataclass(frozen=True)
class Verdad:
    """Parámetros con los que se simula la demanda (lo que el modelo debería recuperar)."""
    base: float = 120_000          # UM/mes de potencial para 3.000 hab. e ingreso 5.000
    exp_poblacion: float = 0.55
    peso_vecinos: float = 0.35     # población del primer anillo que también compra aquí
    exp_ingreso: float = 0.90
    exp_salud: float = 0.15
    exp_comercio: float = 0.20
    peso_competidor: float = 0.60  # un competidor "vale" 0,6 farmacias propias
    resto_mercado: float = 0.80    # demanda que se va a otros canales
    ruido: float = 0.25            # sigma lognormal de la venta mensual por tienda


def _km(lat1, lng1, lat2, lng2):
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1) * math.cos(math.radians((lat1 + lat2) / 2))
    return 6371 * math.hypot(dlat, dlng)


def _centros_ciudades(rng: np.random.Generator) -> list[tuple[float, float]]:
    lat0, lat1, lng0, lng1 = CAJA
    centros: list[tuple[float, float]] = []
    while len(centros) < len(CIUDADES):
        lat, lng = rng.uniform(lat0, lat1), rng.uniform(lng0, lng1)
        if all(_km(lat, lng, a, b) > 60 for a, b in centros):
            centros.append((lat, lng))
    return centros


def generar_pais(semilla: int = 42, verdad: Verdad = Verdad()) -> pd.DataFrame:
    """Una fila por hexágono H3 con atributos, red de farmacias, ventas observadas y verdad."""
    rng = np.random.default_rng(semilla)
    filas = []
    for (ciudad, n_mun, radio, pob_max, ingreso), (clat, clng) in zip(CIUDADES, _centros_ciudades(rng)):
        centro = h3.latlng_to_cell(clat, clng, RESOLUCION)
        decaimiento = 1.2 + radio * 0.35   # km
        # Centros de municipio: el primero en el centro, el resto alrededor.
        munis = [(f"{ciudad} {SUFIJOS[0]}", clat, clng)]
        for k in range(1, n_mun):
            ang = 2 * math.pi * k / max(n_mun - 1, 1) + rng.uniform(-0.3, 0.3)
            dist_km = rng.uniform(0.45, 0.8) * radio * 0.46 * 1.7
            munis.append((f"{ciudad} {SUFIJOS[k]}",
                          clat + dist_km / 111 * math.sin(ang),
                          clng + dist_km / (111 * math.cos(math.radians(clat))) * math.cos(ang)))
        ingreso_mun = {m[0]: ingreso * (1.12 if i == 0 else rng.normal(1, 0.08)) for i, m in enumerate(munis)}

        for celda in h3.grid_disk(centro, radio):
            lat, lng = h3.cell_to_latlng(celda)
            d = _km(lat, lng, clat, clng)
            municipio = min(munis, key=lambda m: _km(lat, lng, m[1], m[2]))[0]
            centralidad = math.exp(-d / decaimiento)
            pob = pob_max * centralidad * rng.lognormal(0, 0.35)
            if pob < 250 and rng.random() < 0.5:
                pob = 0.0
            filas.append({
                "h3": celda, "ciudad": ciudad, "municipio": municipio, "lat": lat, "lng": lng,
                "dist_centro_km": round(d, 2), "poblacion": int(round(min(pob, 12000))),
                "ingreso_pc": round(ingreso_mun[municipio] * rng.normal(1, 0.03), 1),
                "_centralidad": centralidad,
            })

    df = pd.DataFrame(filas).drop_duplicates("h3").reset_index(drop=True)
    pob = df["poblacion"].to_numpy(dtype=float)
    cen = df["_centralidad"].to_numpy()
    df["poi_salud"] = rng.poisson(pob / 2500 * (0.5 + cen))
    df["poi_comercio"] = rng.poisson(pob / 350 * (0.4 + cen) ** 1.2)
    df["poi_educacion"] = rng.poisson(pob / 1500)
    df["poi_total"] = df[["poi_salud", "poi_comercio", "poi_educacion"]].sum(axis=1)

    # Población del primer anillo (vecinos que también compran en este hexágono)
    idx = {c: i for i, c in enumerate(df["h3"])}
    vecinos = np.array([sum(pob[idx[v]] for v in h3.grid_ring(c, 1) if v in idx) for c in df["h3"]])
    pob_efectiva = pob + verdad.peso_vecinos * vecinos

    v = verdad
    demanda = (v.base * (pob_efectiva / 3000) ** v.exp_poblacion
               * (df["ingreso_pc"] / 5000) ** v.exp_ingreso
               * (1 + df["poi_salud"]) ** v.exp_salud
               * (1 + df["poi_comercio"] / 10) ** v.exp_comercio)
    demanda = np.where(pob_efectiva > 0, demanda, 0.0)

    # Competidores y farmacias propias: ambos se ubican donde hay más demanda (sesgo de selección)
    dz = np.log1p(demanda / 50_000)
    df["comp_cadena_b"] = rng.poisson(0.10 * dz ** 1.6)
    df["comp_cadena_c"] = rng.poisson(0.07 * dz ** 1.6)
    df["comp_independientes"] = rng.poisson(0.12 * dz ** 1.4)
    df["competidores"] = df[["comp_cadena_b", "comp_cadena_c", "comp_independientes"]].sum(axis=1)
    p_tienda = 1 / (1 + np.exp(-(dz - 2.4) * 2.2))
    tiene = rng.random(len(df)) < p_tienda
    df["farmacias_propias"] = np.where(tiene, 1 + rng.poisson(0.35 * dz / 2, len(df)), 0)

    cuota = 1 / (df["farmacias_propias"] + v.peso_competidor * df["competidores"] + v.resto_mercado)
    por_tienda = demanda * cuota
    ruido = rng.lognormal(-v.ruido ** 2 / 2, v.ruido, len(df))
    df["ventas_por_tienda"] = np.where(df["farmacias_propias"] > 0, (por_tienda * ruido).round(0), np.nan)
    df["ventas_mes"] = (df["ventas_por_tienda"] * df["farmacias_propias"]).fillna(0)

    df["verdad_demanda"] = demanda.round(0)
    # Venta esperada de UNA tienda nueva si se abriera aquí (con la competencia actual)
    df["verdad_tienda_nueva"] = (demanda / (1 + df["farmacias_propias"] + v.peso_competidor
                                            * df["competidores"] + v.resto_mercado)).round(0)
    return df.drop(columns="_centralidad")
