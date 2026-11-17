"""Mapa interactivo autocontenido (Leaflet) para revisar la expansión por hexágono.

- Capas: venta predicha por hexágono, venta neta de una apertura y potencial por municipio.
- Superposiciones: lista corta de aperturas y farmacias propias actuales.
- Con zoom ≥ 13 cada hexágono muestra la venta predicha (miles) arriba y las farmacias abajo.
- Clic en un hexágono para seleccionarlo; el panel suma población, farmacias y ventas, y la
  selección se puede exportar a CSV.

No usa mapa base: "País Demo" es ficticio y está en mar abierto.
"""
from __future__ import annotations

import json
from pathlib import Path

import h3
import numpy as np
import pandas as pd

LEAFLET = "https://unpkg.com/leaflet@1.9.4/dist/leaflet"

PLANTILLA = r"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Expansión retail por hexágono</title>
<link rel="stylesheet" href="__LEAFLET__.css">
<script src="__LEAFLET__.js"></script>
<style>
  html, body { margin: 0; height: 100%; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
  #mapa { position: absolute; inset: 0; background: #eef1f4; }
  .panel { position: absolute; z-index: 1000; background: #fff; border-radius: 8px; padding: 10px 12px;
           box-shadow: 0 2px 10px rgba(0,0,0,.15); font-size: 12.5px; color: #222; }
  #titulo { top: 10px; left: 50px; max-width: 360px; }
  #titulo h1 { font-size: 15px; margin: 0 0 4px; }
  #titulo p { margin: 0; color: #555; line-height: 1.4; }
  #resumen { top: 10px; right: 10px; min-width: 230px; }
  #resumen h2, #leyenda h2 { font-size: 13px; margin: 0 0 6px; }
  #resumen table { border-collapse: collapse; width: 100%; }
  #resumen td { padding: 2px 0; } #resumen td:last-child { text-align: right; font-variant-numeric: tabular-nums; }
  #resumen button { margin-top: 8px; margin-right: 4px; font: inherit; padding: 3px 8px; cursor: pointer; }
  #leyenda { bottom: 20px; left: 10px; }
  .fila { display: flex; align-items: center; gap: 6px; margin: 2px 0; }
  .caja { width: 14px; height: 10px; border: 1px solid #999; }
  .etiqueta { background: transparent; border: none; box-shadow: none; text-align: center;
              font-size: 10px; line-height: 1.05; color: #111; font-weight: 600; }
  .etiqueta::before { display: none; }
  .apertura { background: #c0392b; color: #fff; border-radius: 50%; width: 20px; height: 20px;
              line-height: 20px; text-align: center; font-size: 11px; font-weight: 700; border: 2px solid #fff; }
</style>
</head>
<body>
<div id="mapa"></div>
<div id="titulo" class="panel">
  <h1>Expansión de farmacias · País Demo</h1>
  <p>Datos 100 % sintéticos. Clic en los hexágonos para sumarlos al resumen.
     Con más zoom: venta predicha (miles de UM) arriba y farmacias propias abajo.</p>
</div>
<div id="resumen" class="panel">
  <h2>Selección</h2>
  <table>
    <tr><td>Hexágonos</td><td id="r_hex">0</td></tr>
    <tr><td>Municipios</td><td id="r_mun">0</td></tr>
    <tr><td>Población</td><td id="r_pob">0</td></tr>
    <tr><td>Farmacias propias</td><td id="r_pro">0</td></tr>
    <tr><td>Competidores</td><td id="r_com">0</td></tr>
    <tr><td>Venta predicha (UM/mes)</td><td id="r_ven">0</td></tr>
    <tr><td>Venta neta de abrir (UM/mes)</td><td id="r_net">0</td></tr>
  </table>
  <button id="limpiar">Limpiar</button><button id="exportar">Exportar CSV</button>
</div>
<div id="leyenda" class="panel"></div>
<script>
const HEX = __HEX__;
const MUNI = __MUNI__;
const APERTURAS = __APERTURAS__;
const CORTES = __CORTES__;
const COLORES = ["#f1eef6", "#bdc9e1", "#74a9cf", "#2b8cbe", "#045a8d"];
const fmt = n => Math.round(n).toLocaleString("es");

const mapa = L.map("mapa", { zoomSnap: 0.5, preferCanvas: false, attributionControl: false });
mapa.fitBounds(L.latLngBounds(HEX.map(h => h.centro)), { padding: [20, 20] });
const seleccion = new Set();
const capas = {};

function color(v, cortes) { let i = 0; while (i < cortes.length && v > cortes[i]) i++; return COLORES[i]; }
function estilo(h, campo) {
  const v = campo === "potencial_municipio" ? MUNI[h.municipio] : h[campo];
  return { color: seleccion.has(h.h3) ? "#c0392b" : "#ffffff", weight: seleccion.has(h.h3) ? 2.5 : 0.6,
           fillColor: color(v, CORTES[campo]), fillOpacity: 0.85 };
}
function capa(campo) {
  const g = L.featureGroup();
  HEX.forEach(h => {
    const p = L.polygon(h.borde, estilo(h, campo));
    p.hex = h; p.campo = campo;
    p.bindTooltip(`<b>${h.municipio}</b><br>Población: ${fmt(h.poblacion)}<br>` +
      `Farmacias propias: ${h.propias} · competidores: ${h.competidores}<br>` +
      `Venta predicha: ${fmt(h.venta_predicha_hex)} UM/mes<br>Venta neta de abrir: ${fmt(h.venta_neta)} UM/mes`,
      { sticky: true });
    p.on("click", () => { seleccion.has(h.h3) ? seleccion.delete(h.h3) : seleccion.add(h.h3); refrescar(); });
    g.addLayer(p);
  });
  return g;
}
capas["Venta predicha por hexágono"] = capa("venta_predicha_hex");
capas["Venta neta de una apertura"] = capa("venta_neta");
capas["Potencial por municipio"] = capa("potencial_municipio");

const aperturas = L.layerGroup(APERTURAS.map(a => L.marker([a.lat, a.lng], {
  icon: L.divIcon({ className: "", html: `<div class="apertura">${a.prioridad}</div>`, iconSize: [24, 24] })
}).bindTooltip(`#${a.prioridad} · ${a.municipio}<br>Venta neta: ${fmt(a.venta_neta)} UM/mes`)));
const propias = L.layerGroup(HEX.filter(h => h.propias > 0).map(h =>
  L.circleMarker(h.centro, { radius: 3, color: "#111", weight: 1, fillColor: "#ffd166", fillOpacity: 1 })));

capas["Venta predicha por hexágono"].addTo(mapa);
aperturas.addTo(mapa);
L.control.layers(capas, { "Lista corta de aperturas": aperturas, "Farmacias propias actuales": propias },
                 { collapsed: false, position: "bottomright" }).addTo(mapa);

function capaActiva() { return Object.values(capas).find(c => mapa.hasLayer(c)); }
function refrescar() {
  const activa = capaActiva();
  if (activa) activa.eachLayer(p => p.setStyle(estilo(p.hex, p.campo)));
  const sel = HEX.filter(h => seleccion.has(h.h3));
  const suma = k => sel.reduce((s, h) => s + h[k], 0);
  document.getElementById("r_hex").textContent = sel.length;
  document.getElementById("r_mun").textContent = new Set(sel.map(h => h.municipio)).size;
  document.getElementById("r_pob").textContent = fmt(suma("poblacion"));
  document.getElementById("r_pro").textContent = fmt(suma("propias"));
  document.getElementById("r_com").textContent = fmt(suma("competidores"));
  document.getElementById("r_ven").textContent = fmt(suma("venta_predicha_hex"));
  document.getElementById("r_net").textContent = fmt(suma("venta_neta"));
}
function leyenda() {
  const activa = capaActiva(); if (!activa) return;
  let campo; activa.eachLayer(p => campo = p.campo);
  const c = CORTES[campo], t = { venta_predicha_hex: "Venta predicha (UM/mes)",
    venta_neta: "Venta neta de abrir (UM/mes)", potencial_municipio: "Potencial neto del municipio" }[campo];
  let html = `<h2>${t}</h2>`;
  for (let i = 0; i < COLORES.length; i++) {
    const desde = i === 0 ? 0 : c[i - 1], hasta = c[i];
    html += `<div class="fila"><span class="caja" style="background:${COLORES[i]}"></span>` +
            (hasta === undefined ? `más de ${fmt(desde)}` : `${fmt(desde)} – ${fmt(hasta)}`) + `</div>`;
  }
  document.getElementById("leyenda").innerHTML = html;
}
mapa.on("baselayerchange", () => { refrescar(); leyenda(); etiquetas(); });
leyenda();

// Etiquetas solo con zoom alto y solo para lo visible
const capaEtiquetas = L.layerGroup().addTo(mapa);
function etiquetas() {
  capaEtiquetas.clearLayers();
  if (mapa.getZoom() < 13) return;
  const b = mapa.getBounds();
  HEX.forEach(h => {
    if (!b.contains(h.centro)) return;
    capaEtiquetas.addLayer(L.tooltip({ permanent: true, direction: "center", className: "etiqueta",
      interactive: false }).setLatLng(h.centro).setContent(`${Math.round(h.venta_predicha_hex / 1000)}<br>${h.propias}`));
  });
}
mapa.on("zoomend moveend", etiquetas);

document.getElementById("limpiar").onclick = () => { seleccion.clear(); refrescar(); };
document.getElementById("exportar").onclick = () => {
  const cols = ["h3", "ciudad", "municipio", "poblacion", "propias", "competidores", "venta_predicha_hex", "venta_neta"];
  const filas = HEX.filter(h => seleccion.has(h.h3)).map(h => cols.map(c => h[c]).join(","));
  const blob = new Blob([[cols.join(","), ...filas].join("\n")], { type: "text/csv" });
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "seleccion_hexagonos.csv"; a.click();
};
</script>
</body>
</html>
"""


def _cortes(valores: pd.Series) -> list[float]:
    v = valores[valores > 0]
    return [float(round(x, 0)) for x in np.quantile(v, [0.2, 0.4, 0.6, 0.8])] if len(v) else [0, 0, 0, 0]


def construir_mapa(df: pd.DataFrame, aperturas: pd.DataFrame, municipios: pd.DataFrame, destino: Path) -> Path:
    hexes = []
    for r in df.itertuples(index=False):
        borde = [[round(a, 5), round(b, 5)] for a, b in h3.cell_to_boundary(r.h3)]
        hexes.append({
            "h3": r.h3, "ciudad": r.ciudad, "municipio": r.municipio, "poblacion": int(r.poblacion),
            "propias": int(r.farmacias_propias), "competidores": int(r.competidores),
            "venta_predicha_hex": float(r.venta_predicha_hex), "venta_neta": float(r.venta_neta),
            "borde": borde, "centro": [round(r.lat, 5), round(r.lng, 5)],
        })
    muni = municipios.set_index("municipio")["potencial_neto"].astype(float).to_dict()
    cortes = {
        "venta_predicha_hex": _cortes(df["venta_predicha_hex"]),
        "venta_neta": _cortes(df["venta_neta"]),
        "potencial_municipio": _cortes(pd.Series(list(muni.values()))),
    }
    html = (PLANTILLA.replace("__LEAFLET__", LEAFLET)
            .replace("__HEX__", json.dumps(hexes, ensure_ascii=False, separators=(",", ":")))
            .replace("__MUNI__", json.dumps(muni, ensure_ascii=False))
            .replace("__APERTURAS__", aperturas[["prioridad", "municipio", "lat", "lng", "venta_neta"]]
                     .to_json(orient="records", force_ascii=False))
            .replace("__CORTES__", json.dumps(cortes)))
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(html, encoding="utf-8")
    return destino
