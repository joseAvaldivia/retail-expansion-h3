"""Rasgos espaciales y modelo de venta por tienda.

El modelo aprende de los hexágonos que YA tienen farmacia propia (venta mensual por tienda) y
predice cuánto vendería una tienda en cualquier otro hexágono. La validación cruzada agrupa por
municipio: si se mezclaran hexágonos vecinos entre entrenamiento y prueba, el error saldría
optimista por autocorrelación espacial.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import h3
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RASGOS = [
    "log_poblacion", "log_poblacion_anillo1", "log_poblacion_anillo2", "log_ingreso",
    "poi_salud", "poi_comercio", "poi_educacion", "competidores", "competidores_anillo1",
    "farmacias_propias", "propias_anillo1", "dist_centro_km",
]


def agregar_rasgos(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega sumas por anillo H3 (k = 1 y k = 2) y transformaciones logarítmicas."""
    df = df.copy()
    idx = {c: i for i, c in enumerate(df["h3"])}
    pob = df["poblacion"].to_numpy(dtype=float)
    comp = df["competidores"].to_numpy(dtype=float)
    prop = df["farmacias_propias"].to_numpy(dtype=float)

    def suma_anillo(valores: np.ndarray, k: int) -> np.ndarray:
        return np.array([sum(valores[idx[v]] for v in h3.grid_ring(c, k) if v in idx) for c in df["h3"]])

    df["poblacion_anillo1"] = suma_anillo(pob, 1)
    df["poblacion_anillo2"] = suma_anillo(pob, 2)
    df["competidores_anillo1"] = suma_anillo(comp, 1)
    df["propias_anillo1"] = suma_anillo(prop, 1)
    df["propias_anillo2"] = suma_anillo(prop, 2)
    df["log_poblacion"] = np.log1p(df["poblacion"])
    df["log_poblacion_anillo1"] = np.log1p(df["poblacion_anillo1"])
    df["log_poblacion_anillo2"] = np.log1p(df["poblacion_anillo2"])
    df["log_ingreso"] = np.log(df["ingreso_pc"])
    return df


def _modelos() -> dict:
    return {
        "Regresión log-lineal (Ridge)": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))),
        # Restricciones monótonas: más población o ingreso nunca bajan la venta; más
        # competidores o más tiendas propias en el hexágono nunca la suben.
        "Gradient boosting monótono": HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=12,
            l2_regularization=1.0, random_state=0,
            monotonic_cst=[1, 1, 1, 1, 0, 0, 0, -1, -1, -1, 0, 0]),
    }


@dataclass
class Resultado:
    nombre: str
    modelo: object
    cv: pd.DataFrame                       # métricas por modelo
    importancias: pd.Series = field(default_factory=pd.Series)


def entrenar(df: pd.DataFrame, n_folds: int = 5) -> Resultado:
    """Compara modelos con validación cruzada agrupada por municipio y reentrena el mejor."""
    ent = df[df["farmacias_propias"] > 0].dropna(subset=["ventas_por_tienda"])
    X, y, grupos = ent[RASGOS], np.log(ent["ventas_por_tienda"]), ent["municipio"]
    gkf = GroupKFold(n_splits=min(n_folds, grupos.nunique()))
    filas = []
    for nombre, modelo in _modelos().items():
        pred = np.zeros(len(y))
        for tr, te in gkf.split(X, y, grupos):
            modelo.fit(X.iloc[tr], y.iloc[tr])
            pred[te] = modelo.predict(X.iloc[te])
        filas.append({
            "modelo": nombre,
            "r2_log": r2_score(y, pred),
            "mape": mean_absolute_percentage_error(np.exp(y), np.exp(pred)),
            "n_tiendas_hex": len(y),
        })
    cv = pd.DataFrame(filas).sort_values("r2_log", ascending=False).reset_index(drop=True)
    mejor = cv.loc[0, "modelo"]
    modelo = _modelos()[mejor].fit(X, y)

    from sklearn.inspection import permutation_importance
    imp = permutation_importance(modelo, X, y, n_repeats=5, random_state=0)
    importancias = pd.Series(imp.importances_mean, index=RASGOS).sort_values(ascending=False)
    return Resultado(mejor, modelo, cv.round(4), importancias)


def predecir_por_tienda(modelo, df: pd.DataFrame, farmacias_propias: pd.Series | np.ndarray) -> np.ndarray:
    """Venta mensual esperada por tienda si el hexágono tuviera `farmacias_propias` tiendas."""
    X = df[RASGOS].copy()
    X["farmacias_propias"] = np.asarray(farmacias_propias, dtype=float)
    return np.exp(modelo.predict(X))
