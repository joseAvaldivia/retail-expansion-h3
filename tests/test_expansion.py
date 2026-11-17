import h3
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import r2_score

from expansion_h3 import (Reglas, agregar_rasgos, entrenar, generar_pais, lista_corta, puntuar,
                          ranking_municipios)
from expansion_h3.geografia import CAJA, RESOLUCION
from expansion_h3.mapa import construir_mapa


@pytest.fixture(scope="module")
def pais():
    return agregar_rasgos(generar_pais(semilla=42))


@pytest.fixture(scope="module")
def puntuado(pais):
    res = entrenar(pais)
    return res, puntuar(pais, res.modelo)


def test_reproducible():
    a, b = generar_pais(semilla=3), generar_pais(semilla=3)
    pd.testing.assert_frame_equal(a, b)


def test_geografia_ficticia(pais):
    assert pais["h3"].is_unique
    assert all(h3.get_resolution(c) == RESOLUCION for c in pais["h3"])
    lat0, lat1, lng0, lng1 = CAJA
    # Todo cae en la caja de mar abierto (con margen por el radio de las ciudades)
    assert pais["lat"].between(lat0 - 0.5, lat1 + 0.5).all()
    assert pais["lng"].between(lng0 - 0.5, lng1 + 0.5).all()
    assert pais["municipio"].nunique() >= 40


def test_ventas_solo_donde_hay_tienda(pais):
    con = pais["farmacias_propias"] > 0
    assert pais.loc[con, "ventas_por_tienda"].gt(0).all()
    assert pais.loc[~con, "ventas_por_tienda"].isna().all()


def test_rasgos_de_anillo(pais):
    fila = pais.iloc[0]
    esperado = pais.set_index("h3").reindex(list(h3.grid_ring(fila["h3"], 1)))["poblacion"].sum()
    assert fila["poblacion_anillo1"] == pytest.approx(esperado)


def test_modelo_recupera_la_verdad(puntuado):
    res, p = puntuado
    assert res.cv["r2_log"].max() > 0.6
    m = (p["poblacion"] >= 500) & (p["farmacias_propias"] == 0)
    r2 = r2_score(np.log(p.loc[m, "verdad_tienda_nueva"]), np.log(p.loc[m, "venta_predicha_tienda_nueva"]))
    assert r2 > 0.85


def test_mas_tiendas_en_el_hexagono_rinden_menos_por_tienda(puntuado):
    _, p = puntuado
    # La venta incremental de abrir una tienda nunca supera a la de una tienda nueva sola
    assert (p["venta_incremental"] <= p["venta_predicha_tienda_nueva"] + 1).all()


def test_lista_corta_respeta_distancia(puntuado):
    _, p = puntuado
    reglas = Reglas(n_aperturas=15, distancia_minima=4)
    lista = lista_corta(p, reglas)
    assert len(lista) == 15
    assert lista["venta_neta"].is_monotonic_decreasing
    celdas = lista["h3"].tolist()
    for i, a in enumerate(celdas):
        for b in celdas[i + 1:]:
            assert h3.grid_distance(a, b) >= 4


def test_ranking_y_mapa(puntuado, tmp_path):
    _, p = puntuado
    lista = lista_corta(p)
    mun = ranking_municipios(p, lista)
    assert mun["aperturas_en_lista"].sum() == len(lista)
    html = construir_mapa(p, lista, mun, tmp_path / "mapa.html").read_text(encoding="utf-8")
    assert "leaflet@1.9.4" in html
    assert html.count('"h3":') == len(p)
    assert "verdad_" not in html
