"""Modelo de expansión para retail farmacéutico por hexágonos H3 (datos sintéticos)."""
from .geografia import Verdad, generar_pais
from .modelo import agregar_rasgos, entrenar, predecir_por_tienda
from .oportunidad import Reglas, lista_corta, puntuar, ranking_municipios

__all__ = ["Verdad", "generar_pais", "agregar_rasgos", "entrenar", "predecir_por_tienda",
           "Reglas", "puntuar", "lista_corta", "ranking_municipios"]
__version__ = "1.0.0"
