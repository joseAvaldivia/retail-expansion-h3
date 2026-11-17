"""Corre todo el flujo: datos sintéticos → rasgos → modelo → oportunidades → mapa.

    python -m expansion_h3 [--semilla 42] [--salida salidas]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score

from .geografia import generar_pais
from .mapa import construir_mapa
from .modelo import agregar_rasgos, entrenar
from .oportunidad import Reglas, lista_corta, puntuar, ranking_municipios


def correr(semilla: int = 42, salida: Path = Path("salidas"), mapa: Path | None = None) -> dict:
    salida.mkdir(parents=True, exist_ok=True)
    df = agregar_rasgos(generar_pais(semilla))
    res = entrenar(df)
    reglas = Reglas()
    p = puntuar(df, res.modelo, reglas)
    aperturas = lista_corta(p, reglas)
    municipios = ranking_municipios(p, aperturas)

    publicas = [c for c in p.columns if not c.startswith("verdad_")]
    p[publicas].to_csv(salida / "hexagonos.csv", index=False)
    aperturas.to_csv(salida / "lista_corta.csv", index=False)
    municipios.to_csv(salida / "ranking_municipios.csv", index=False)
    construir_mapa(p, aperturas, municipios, mapa or salida / "mapa_expansion.html")

    # Validación contra la verdad sintética (solo posible porque los datos son simulados)
    m = (p["poblacion"] >= reglas.poblacion_minima) & (p["farmacias_propias"] == 0)
    metricas = {
        "semilla": semilla,
        "hexagonos": int(len(p)), "municipios": int(p["municipio"].nunique()),
        "hexagonos_con_tienda": int((p["farmacias_propias"] > 0).sum()),
        "modelo_elegido": res.nombre,
        "validacion_cruzada_por_municipio": res.cv.to_dict(orient="records"),
        "r2_log_vs_verdad_hex_sin_tienda": round(float(r2_score(
            np.log(p.loc[m, "verdad_tienda_nueva"]), np.log(p.loc[m, "venta_predicha_tienda_nueva"]))), 4),
        "importancia_rasgos": res.importancias.round(4).to_dict(),
        "venta_neta_lista_corta": float(aperturas["venta_neta"].sum()),
    }
    (salida / "metricas.json").write_text(json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8")
    return metricas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--salida", type=Path, default=Path("salidas"))
    ap.add_argument("--mapa", type=Path, default=None, help="ruta del HTML (por defecto, dentro de --salida)")
    a = ap.parse_args()
    m = correr(a.semilla, a.salida, a.mapa)
    print(json.dumps({k: v for k, v in m.items() if k != "importancia_rasgos"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
