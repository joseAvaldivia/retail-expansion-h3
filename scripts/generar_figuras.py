"""Figuras del README a partir de las salidas de `python -m expansion_h3`.

    python -m expansion_h3 --mapa docs/mapa_expansion.html
    python scripts/generar_figuras.py
"""
from __future__ import annotations

import json
from pathlib import Path

import h3
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
SAL = RAIZ / "salidas"
IMG = RAIZ / "docs" / "img"
IMG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

hexes = pd.read_csv(SAL / "hexagonos.csv")
lista = pd.read_csv(SAL / "lista_corta.csv")

# 1. Ciudad principal: venta predicha por hexágono + lista corta
ciudad = hexes.groupby("ciudad")["poblacion"].sum().idxmax()
h = hexes[hexes["ciudad"] == ciudad]
polys = [[(lng, lat) for lat, lng in h3.cell_to_boundary(c)] for c in h["h3"]]
fig, ax = plt.subplots(figsize=(7.5, 6))
col = PolyCollection(polys, array=np.log10(h["venta_predicha_hex"].clip(lower=1)), cmap="Blues",
                     edgecolors="white", linewidths=0.3)
ax.add_collection(col)
propias = h[h["farmacias_propias"] > 0]
ax.scatter(propias["lng"], propias["lat"], s=8, c="#ffd166", edgecolors="black", linewidths=0.4,
           label="Farmacia propia actual", zorder=3)
l = lista[lista["ciudad"] == ciudad]
for _, r in l.iterrows():
    ax.scatter(r["lng"], r["lat"], s=160, c="#c0392b", edgecolors="white", zorder=4)
    ax.text(r["lng"], r["lat"], str(r["prioridad"]), color="white", ha="center", va="center",
            fontsize=7, fontweight="bold", zorder=5)
ax.autoscale_view()
ax.set_aspect(1 / np.cos(np.radians(h["lat"].mean())))
ax.set_xticks([]), ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(col, ax=ax, shrink=0.7)
cb.set_label("log10 venta predicha (UM/mes)")
ax.legend(loc="lower left", frameon=False)
ax.set_title(f"Ciudad {ciudad}: venta predicha por hexágono y aperturas priorizadas (rojo)", loc="left")
fig.tight_layout()
fig.savefig(IMG / "mapa_ciudad.png", dpi=150)

# 2. Validación: predicción vs. verdad sintética
from expansion_h3 import agregar_rasgos, entrenar, generar_pais, puntuar  # noqa: E402

df = agregar_rasgos(generar_pais(json.loads((SAL / "metricas.json").read_text(encoding="utf-8"))["semilla"]))
p = puntuar(df, entrenar(df).modelo)
m = (p["poblacion"] >= 500) & (p["farmacias_propias"] == 0)
fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.scatter(p.loc[m, "verdad_tienda_nueva"] / 1000, p.loc[m, "venta_predicha_tienda_nueva"] / 1000,
           s=6, alpha=0.5, color="#1f5aa6")
lim = [0, max(p.loc[m, "verdad_tienda_nueva"].max(), p.loc[m, "venta_predicha_tienda_nueva"].max()) / 1000]
ax.plot(lim, lim, color="#c0392b", lw=1, ls="--")
ax.set_xlabel("Venta real simulada de una tienda nueva (miles UM/mes)")
ax.set_ylabel("Venta predicha (miles UM/mes)")
ax.set_title("Hexágonos SIN tienda: el modelo nunca vio su venta", loc="left")
fig.tight_layout()
fig.savefig(IMG / "validacion.png", dpi=150)
print("Figuras en", IMG)
