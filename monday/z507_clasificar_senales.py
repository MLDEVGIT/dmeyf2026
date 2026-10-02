#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
z507_clasificar_senales.py

DMEyF 2026
Clasificación e interpretación de señales pre-BAJA+2.

Objetivo
--------
Consolidar los resultados generados por z506_prebaja_features.py
para distinguir:

1. fuerza global de la señal;
2. robustez temporal;
3. robustez entre clusters;
4. naturaleza temporal de la transformación;
5. señales generales;
6. señales heterogéneas;
7. señales específicas de determinados perfiles.

IMPORTANTE
----------
Este script NO:

- vuelve a leer competencia_01.csv;
- reconstruye features;
- entrena modelos;
- recalcula clusters;
- utiliza BAJA+2 como predictor;
- interpreta SMD como causalidad;
- interpreta la muestra balanceada como prevalencia poblacional.

El análisis es descriptivo/exploratorio.

Entradas
--------
/data/dmeyf/datasets/evaluacion_clusters/prebaja_features/

    comparacion_global.csv
    comparacion_por_foto.csv
    comparacion_por_cluster.csv
    consistencia_por_foto.csv
    ranking_variables.csv

Salidas
-------
/data/dmeyf/datasets/evaluacion_clusters/clasificacion_senales/

    clasificacion_features.csv
    senales_generales_robustas.csv
    senales_generales_heterogeneas.csv
    senales_especificas_cluster.csv
    ranking_variables.csv
    resumen_clusters_features.csv
    resumen_transformaciones.csv
    resumen_z507.txt

Criterios de magnitud
---------------------
Los cortes de |SMD| son reglas interpretativas explícitas:

    < 0.20       débil
    0.20 - 0.50  moderada
    0.50 - 0.80  fuerte
    >= 0.80      muy fuerte

No constituyen tests de significancia.

Robustez temporal
-----------------
Se considera especialmente informativa una señal cuando:

- fue evaluada en >= 2 fotos;
- mantiene el mismo signo;
- y tiene magnitud global relevante.

Una señal observada en una sola foto NO se denomina
"temporalmente robusta".

Robustez entre clusters
-----------------------
Se distingue:

- mismo signo en los 5 clusters;
- mismo signo sólo en algunos clusters;
- cambio de signo entre clusters;
- evidencia insuficiente por tamaño muestral.

Una señal global fuerte puede ser heterogénea entre perfiles.

Autor:
    DMEyF 2026
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# =====================================================================
# CONFIGURACIÓN
# =====================================================================

IN_DIR = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/prebaja_features"
)

OUT_DIR = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/clasificacion_senales"
)

ARCHIVO_GLOBAL = IN_DIR / "comparacion_global.csv"
ARCHIVO_FOTO = IN_DIR / "comparacion_por_foto.csv"
ARCHIVO_CLUSTER = IN_DIR / "comparacion_por_cluster.csv"
ARCHIVO_CONSISTENCIA = IN_DIR / "consistencia_por_foto.csv"
ARCHIVO_RANKING_Z506 = IN_DIR / "ranking_variables.csv"


# Cortes interpretativos de magnitud SMD.
SMD_MODERADA = 0.20
SMD_FUERTE = 0.50
SMD_MUY_FUERTE = 0.80

# Para hablar de replicación temporal necesitamos
# al menos dos fotos.
MIN_FOTOS_REPLICACION = 2

# Número esperado de clusters de z504/z506.
N_CLUSTERS_ESPERADOS = 5


# =====================================================================
# UTILIDADES
# =====================================================================

def titulo(txt: str) -> None:
    print()
    print("=" * 78)
    print(txt)
    print("=" * 78)


def subtitulo(txt: str) -> None:
    print()
    print("-" * 78)
    print(txt)
    print("-" * 78)


def safe_bool(x) -> bool:
    """
    Conversión robusta a bool para columnas leídas desde CSV.
    """

    if pd.isna(x):
        return False

    if isinstance(x, (bool, np.bool_)):
        return bool(x)

    if isinstance(x, (int, np.integer)):
        return bool(x)

    if isinstance(x, (float, np.floating)):
        if np.isnan(x):
            return False
        return bool(x)

    s = str(x).strip().lower()

    return s in {
        "true",
        "1",
        "yes",
        "si",
        "sí",
        "t",
    }


def clasificar_magnitud(abs_smd: float) -> str:
    """
    Clasifica la magnitud de un standardized mean difference.
    """

    if pd.isna(abs_smd):
        return "sin_dato"

    x = abs(float(abs_smd))

    if not np.isfinite(x):
        return "sin_dato"

    if x < SMD_MODERADA:
        return "debil"

    if x < SMD_FUERTE:
        return "moderada"

    if x < SMD_MUY_FUERTE:
        return "fuerte"

    return "muy_fuerte"


def direccion_smd(smd: float) -> str:
    """
    Describe la dirección de la diferencia BAJA+2 - FIEL.
    """

    if pd.isna(smd):
        return "sin_dato"

    x = float(smd)

    if not np.isfinite(x):
        return "sin_dato"

    if x < 0:
        return "menor_en_baja2"

    if x > 0:
        return "mayor_en_baja2"

    return "sin_diferencia"


def naturaleza_transformacion(
    transformacion: str,
) -> str:
    """
    Agrupa las transformaciones de z506 en familias interpretables.
    """

    if pd.isna(transformacion):
        return "desconocida"

    t = str(transformacion).strip().lower()

    mapa = {
        "t0": "nivel_actual",

        "delta1": "cambio",
        "delta2": "cambio",

        "slope2": "tendencia",
        "slope3": "tendencia",

        "media2": "nivel_reciente",
        "media3": "nivel_reciente",

        "min3": "extremo_reciente",
        "max3": "extremo_reciente",

        "std3": "variabilidad_reciente",
    }

    return mapa.get(
        t,
        "otra",
    )


def evidencia_temporal(
    n_fotos: float,
    mismo_signo: bool,
) -> str:
    """
    Clasificación descriptiva de la evidencia temporal.
    """

    if pd.isna(n_fotos):
        return "sin_evaluacion"

    n = int(n_fotos)

    if n <= 0:
        return "sin_evaluacion"

    if n == 1:
        return "puntual_1_foto"

    if mismo_signo:
        return f"consistente_{n}_fotos"

    return f"inestable_{n}_fotos"


def evidencia_clusters(
    n_clusters: float,
    mismo_signo: bool,
) -> str:
    """
    Clasificación descriptiva de la evidencia entre clusters.
    """

    if pd.isna(n_clusters):
        return "sin_evaluacion"

    n = int(n_clusters)

    if n <= 0:
        return "sin_evaluacion"

    if n < N_CLUSTERS_ESPERADOS:
        if mismo_signo:
            return f"parcial_consistente_{n}_clusters"

        return f"parcial_heterogenea_{n}_clusters"

    if mismo_signo:
        return "consistente_5_clusters"

    return "heterogenea_5_clusters"


# =====================================================================
# VALIDACIÓN DE ARCHIVOS
# =====================================================================

def validar_archivos() -> None:

    titulo("VALIDACIÓN DE ENTRADAS")

    archivos = [
        ARCHIVO_GLOBAL,
        ARCHIVO_FOTO,
        ARCHIVO_CLUSTER,
        ARCHIVO_CONSISTENCIA,
        ARCHIVO_RANKING_Z506,
    ]

    faltantes = []

    for p in archivos:

        if p.exists():
            print(
                f"OK  {p.name:<32} "
                f"{p.stat().st_size:>12,} bytes"
            )
        else:
            print(
                f"FALTA  {p}"
            )
            faltantes.append(p)

    if faltantes:
        raise FileNotFoundError(
            "Faltan archivos necesarios de z506."
        )


# =====================================================================
# CARGA
# =====================================================================

def cargar_datos():

    titulo("CARGA DE RESULTADOS Z506")

    global_df = pd.read_csv(
        ARCHIVO_GLOBAL
    )

    foto_df = pd.read_csv(
        ARCHIVO_FOTO
    )

    cluster_df = pd.read_csv(
        ARCHIVO_CLUSTER
    )

    consistencia_df = pd.read_csv(
        ARCHIVO_CONSISTENCIA
    )

    ranking_z506 = pd.read_csv(
        ARCHIVO_RANKING_Z506
    )

    print(
        f"Comparación global       : "
        f"{global_df.shape}"
    )

    print(
        f"Comparación por foto     : "
        f"{foto_df.shape}"
    )

    print(
        f"Comparación por cluster  : "
        f"{cluster_df.shape}"
    )

    print(
        f"Consistencia temporal    : "
        f"{consistencia_df.shape}"
    )

    print(
        f"Ranking variables z506   : "
        f"{ranking_z506.shape}"
    )

    return (
        global_df,
        foto_df,
        cluster_df,
        consistencia_df,
        ranking_z506,
    )


# =====================================================================
# VALIDACIÓN DE COLUMNAS
# =====================================================================

def exigir_columnas(
    df: pd.DataFrame,
    columnas: list[str],
    nombre: str,
) -> None:

    faltantes = [
        c
        for c in columnas
        if c not in df.columns
    ]

    if faltantes:
        raise ValueError(
            f"{nombre}: faltan columnas: "
            f"{faltantes}"
        )


def validar_columnas(
    global_df: pd.DataFrame,
    foto_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    consistencia_df: pd.DataFrame,
) -> None:

    titulo("VALIDACIÓN DE ESTRUCTURA")

    exigir_columnas(
        global_df,
        [
            "feature",
            "variable",
            "transformacion",
            "n_baja2",
            "n_fiel",
            "smd",
            "abs_smd",
            "cumple_min_n",
        ],
        "comparacion_global",
    )

    exigir_columnas(
        foto_df,
        [
            "feature",
            "foto_ancla",
            "smd",
            "abs_smd",
            "cumple_min_n",
        ],
        "comparacion_por_foto",
    )

    exigir_columnas(
        cluster_df,
        [
            "feature",
            "cluster",
            "smd",
            "abs_smd",
            "cumple_min_n",
        ],
        "comparacion_por_cluster",
    )

    exigir_columnas(
        consistencia_df,
        [
            "feature",
            "n_fotos",
            "smd_mean",
            "abs_smd_mean",
            "abs_smd_min",
            "abs_smd_max",
            "mismo_signo",
        ],
        "consistencia_por_foto",
    )

    print("Estructura de archivos: OK")


# =====================================================================
# RESUMEN ENTRE CLUSTERS
# =====================================================================

def construir_resumen_clusters(
    cluster_df: pd.DataFrame,
) -> pd.DataFrame:

    titulo("CONSISTENCIA ENTRE CLUSTERS")

    x = cluster_df.copy()

    x["cumple_min_n"] = (
        x["cumple_min_n"]
        .map(safe_bool)
    )

    # Sólo utilizamos comparaciones que z506 considera
    # suficientemente respaldadas por tamaño muestral.
    x = x[
        x["cumple_min_n"]
    ].copy()

    x["smd"] = pd.to_numeric(
        x["smd"],
        errors="coerce",
    )

    x["abs_smd"] = pd.to_numeric(
        x["abs_smd"],
        errors="coerce",
    )

    # Excluir infinitos además de NaN.
    x = x[
        x["smd"].notna()
        & np.isfinite(x["smd"])
    ].copy()

    if x.empty:
        raise ValueError(
            "No quedaron comparaciones por cluster "
            "con tamaño muestral suficiente."
        )

    # -------------------------------------------------------------
    # Resumen numérico
    # -------------------------------------------------------------

    resumen = (
        x.groupby(
            "feature",
            dropna=False,
        )
        .agg(
            n_clusters=(
                "cluster",
                "nunique",
            ),
            smd_cluster_mean=(
                "smd",
                "mean",
            ),
            smd_cluster_min=(
                "smd",
                "min",
            ),
            smd_cluster_max=(
                "smd",
                "max",
            ),
            abs_smd_cluster_mean=(
                "abs_smd",
                "mean",
            ),
            abs_smd_cluster_min=(
                "abs_smd",
                "min",
            ),
            abs_smd_cluster_max=(
                "abs_smd",
                "max",
            ),
        )
        .reset_index()
    )

    # -------------------------------------------------------------
    # Signo
    # -------------------------------------------------------------

    signos = x[
        [
            "feature",
            "cluster",
            "smd",
        ]
    ].copy()

    signos["signo"] = np.sign(
        signos["smd"]
    )

    def mismo_signo_serie(
        s: pd.Series,
    ) -> bool:

        vals = set(
            s[
                s != 0
            ]
            .dropna()
            .astype(int)
            .tolist()
        )

        return len(vals) <= 1

    signos_resumen = (
        signos
        .groupby("feature")["signo"]
        .agg(mismo_signo_serie)
        .rename(
            "mismo_signo_clusters"
        )
        .reset_index()
    )

    resumen = resumen.merge(
        signos_resumen,
        on="feature",
        how="left",
    )

    resumen[
        "evidencia_clusters"
    ] = resumen.apply(
        lambda r: evidencia_clusters(
            r["n_clusters"],
            safe_bool(
                r["mismo_signo_clusters"]
            ),
        ),
        axis=1,
    )

    print(
        f"Features con evidencia por cluster: "
        f"{len(resumen):,}"
    )

    print()
    print("Número de clusters evaluables:")

    print(
        resumen[
            "n_clusters"
        ]
        .value_counts()
        .sort_index()
    )

    print()
    print("Consistencia de signo:")

    print(
        pd.crosstab(
            resumen["n_clusters"],
            resumen[
                "mismo_signo_clusters"
            ],
            margins=True,
        )
    )

    return resumen


# =====================================================================
# CONSOLIDACIÓN
# =====================================================================

def consolidar_features(
    global_df: pd.DataFrame,
    consistencia_df: pd.DataFrame,
    resumen_clusters: pd.DataFrame,
) -> pd.DataFrame:

    titulo("CONSOLIDACIÓN DE FEATURES")

    g = global_df.copy()

    g["cumple_min_n"] = (
        g["cumple_min_n"]
        .map(safe_bool)
    )

    g["smd"] = pd.to_numeric(
        g["smd"],
        errors="coerce",
    )

    g["abs_smd"] = pd.to_numeric(
        g["abs_smd"],
        errors="coerce",
    )

    # -------------------------------------------------------------
    # Consistencia temporal
    # -------------------------------------------------------------

    c = consistencia_df.copy()

    c["mismo_signo"] = (
        c["mismo_signo"]
        .map(safe_bool)
    )

    columnas_temporales = [
        "feature",
        "n_fotos",
        "smd_mean",
        "abs_smd_mean",
        "abs_smd_min",
        "abs_smd_max",
        "mismo_signo",
    ]

    c = c[
        columnas_temporales
    ].copy()

    c = c.rename(
        columns={
            "mismo_signo":
                "mismo_signo_fotos",

            "smd_mean":
                "smd_fotos_mean",

            "abs_smd_mean":
                "abs_smd_fotos_mean",

            "abs_smd_min":
                "abs_smd_fotos_min",

            "abs_smd_max":
                "abs_smd_fotos_max",
        }
    )

    # -------------------------------------------------------------
    # Merge
    # -------------------------------------------------------------

    out = g.merge(
        c,
        on="feature",
        how="left",
    )

    out = out.merge(
        resumen_clusters,
        on="feature",
        how="left",
    )

    # -------------------------------------------------------------
    # Clasificaciones descriptivas
    # -------------------------------------------------------------

    out["magnitud_global"] = (
        out["abs_smd"]
        .apply(clasificar_magnitud)
    )

    out["direccion_global"] = (
        out["smd"]
        .apply(direccion_smd)
    )

    out["naturaleza"] = (
        out["transformacion"]
        .apply(
            naturaleza_transformacion
        )
    )

    out["evidencia_temporal"] = (
        out.apply(
            lambda r: evidencia_temporal(
                r["n_fotos"],
                safe_bool(
                    r[
                        "mismo_signo_fotos"
                    ]
                ),
            ),
            axis=1,
        )
    )

    # -------------------------------------------------------------
    # Flags explícitos
    # -------------------------------------------------------------

    out["global_relevante"] = (
        out["cumple_min_n"]
        & (
            out["abs_smd"]
            >= SMD_FUERTE
        )
    )

    out["global_muy_fuerte"] = (
        out["cumple_min_n"]
        & (
            out["abs_smd"]
            >= SMD_MUY_FUERTE
        )
    )

    out["temporal_replicada"] = (
        out["n_fotos"]
        .fillna(0)
        .ge(
            MIN_FOTOS_REPLICACION
        )
        & out[
            "mismo_signo_fotos"
        ].fillna(False)
    )

    out["clusters_completos"] = (
        out["n_clusters"]
        .fillna(0)
        .eq(
            N_CLUSTERS_ESPERADOS
        )
    )

    out["clusters_consistentes"] = (
        out["clusters_completos"]
        & out[
            "mismo_signo_clusters"
        ].fillna(False)
    )

    out["clusters_heterogeneos"] = (
        out["clusters_completos"]
        & ~out[
            "mismo_signo_clusters"
        ].fillna(False)
    )

    # -------------------------------------------------------------
    # Clasificación interpretativa principal
    # -------------------------------------------------------------

    def categoria(
        r: pd.Series,
    ) -> str:

        if not safe_bool(
            r["cumple_min_n"]
        ):
            return (
                "evidencia_limitada_muestra"
            )

        abs_smd = r["abs_smd"]

        if (
            pd.isna(abs_smd)
            or not np.isfinite(abs_smd)
        ):
            return "sin_evaluacion"

        fuerte = (
            abs_smd >= SMD_FUERTE
        )

        moderada = (
            abs_smd >= SMD_MODERADA
        )

        temporal = safe_bool(
            r["temporal_replicada"]
        )

        clusters_ok = safe_bool(
            r["clusters_consistentes"]
        )

        clusters_het = safe_bool(
            r["clusters_heterogeneos"]
        )

        # Señal fuerte global, replicada en el tiempo
        # y homogénea en los cinco perfiles.
        if (
            fuerte
            and temporal
            and clusters_ok
        ):
            return "general_robusta"

        # Señal fuerte global y temporal, pero
        # cambia de comportamiento entre clusters.
        if (
            fuerte
            and temporal
            and clusters_het
        ):
            return "general_heterogenea"

        # Señal global fuerte pero sin suficiente
        # replicación temporal.
        if fuerte and not temporal:
            return (
                "fuerte_evidencia_temporal_limitada"
            )

        # No domina globalmente, pero existen
        # diferencias relevantes dentro de perfiles.
        max_cluster = r.get(
            "abs_smd_cluster_max",
            np.nan,
        )

        if (
            moderada
            and pd.notna(max_cluster)
            and np.isfinite(max_cluster)
            and max_cluster >= SMD_FUERTE
        ):
            return "especifica_de_perfil"

        if (
            pd.notna(max_cluster)
            and np.isfinite(max_cluster)
            and max_cluster >= SMD_FUERTE
        ):
            return "especifica_de_perfil"

        if moderada:
            return "moderada"

        return "debil"

    out["categoria"] = (
        out.apply(
            categoria,
            axis=1,
        )
    )

    # -------------------------------------------------------------
    # Orden
    # -------------------------------------------------------------

    prioridad = {
        "general_robusta": 1,
        "general_heterogenea": 2,
        "especifica_de_perfil": 3,
        "fuerte_evidencia_temporal_limitada": 4,
        "moderada": 5,
        "debil": 6,
        "evidencia_limitada_muestra": 7,
        "sin_evaluacion": 8,
    }

    out["orden_categoria"] = (
        out["categoria"]
        .map(prioridad)
        .fillna(99)
        .astype(int)
    )

    out = out.sort_values(
        [
            "orden_categoria",
            "abs_smd",
        ],
        ascending=[
            True,
            False,
        ],
        na_position="last",
    ).reset_index(drop=True)

    print(
        f"Features consolidadas: "
        f"{len(out):,}"
    )

    print()
    print("Categorías:")

    print(
        out[
            "categoria"
        ]
        .value_counts()
    )

    return out


# =====================================================================
# SEÑALES ESPECÍFICAS DE CLUSTER
# =====================================================================

def construir_senales_especificas(
    cluster_df: pd.DataFrame,
    global_df: pd.DataFrame,
) -> pd.DataFrame:

    titulo("SEÑALES ESPECÍFICAS DE PERFIL")

    x = cluster_df.copy()

    x["cumple_min_n"] = (
        x["cumple_min_n"]
        .map(safe_bool)
    )

    x["smd"] = pd.to_numeric(
        x["smd"],
        errors="coerce",
    )

    x["abs_smd"] = pd.to_numeric(
        x["abs_smd"],
        errors="coerce",
    )

    x = x[
        x["cumple_min_n"]
        & x["abs_smd"].notna()
        & np.isfinite(x["abs_smd"])
        & x["abs_smd"].ge(
            SMD_FUERTE
        )
    ].copy()

    g = global_df[
        [
            "feature",
            "smd",
            "abs_smd",
        ]
    ].copy()

    g["smd"] = pd.to_numeric(
        g["smd"],
        errors="coerce",
    )

    g["abs_smd"] = pd.to_numeric(
        g["abs_smd"],
        errors="coerce",
    )

    g = g.rename(
        columns={
            "smd": "smd_global",
            "abs_smd":
                "abs_smd_global",
        }
    )

    x = x.merge(
        g,
        on="feature",
        how="left",
    )

    x["direccion_cluster"] = (
        x["smd"]
        .apply(direccion_smd)
    )

    x["magnitud_cluster"] = (
        x["abs_smd"]
        .apply(clasificar_magnitud)
    )

    x["signo_distinto_global"] = (
        np.sign(x["smd"])
        != np.sign(x["smd_global"])
    )

    # Priorizamos aquellas que son fuertes dentro
    # del cluster pero no necesariamente fuertes
    # globalmente.
    x["es_especifica"] = (
        x["abs_smd_global"]
        .fillna(0)
        .lt(SMD_FUERTE)
        | x["signo_distinto_global"]
    )

    x = x.sort_values(
        [
            "es_especifica",
            "abs_smd",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(drop=True)

    print(
        f"Comparaciones cluster con |SMD| >= "
        f"{SMD_FUERTE:.2f}: {len(x):,}"
    )

    print(
        f"Marcadas como específicas: "
        f"{x['es_especifica'].sum():,}"
    )

    return x


# =====================================================================
# RANKING DE VARIABLES ORIGINALES
# =====================================================================

def construir_ranking_variables(
    clasificacion: pd.DataFrame,
) -> pd.DataFrame:

    titulo("RANKING DE VARIABLES ORIGINALES")

    x = clasificacion[
        clasificacion[
            "cumple_min_n"
        ]
    ].copy()

    if x.empty:
        print(
            "No hay features con tamaño muestral suficiente."
        )
        return pd.DataFrame()

    x["abs_smd"] = pd.to_numeric(
        x["abs_smd"],
        errors="coerce",
    )

    # -------------------------------------------------------------
    # Resumen por variable original.
    #
    # No construimos un score artificial.
    # Conservamos métricas separadas.
    #
    # Es importante mantener aquí incluso variables cuya totalidad
    # de SMD globales sea NaN. Esas variables forman parte del
    # universo evaluado, aunque no tengan una "mejor feature"
    # seleccionable.
    # -------------------------------------------------------------

    ranking = (
        x.groupby(
            "variable",
            dropna=False,
        )
        .agg(
            n_features=(
                "feature",
                "size",
            ),

            n_features_smd_valido=(
                "abs_smd",
                lambda s: (
                    pd.to_numeric(
                        s,
                        errors="coerce",
                    )
                    .replace(
                        [np.inf, -np.inf],
                        np.nan,
                    )
                    .notna()
                    .sum()
                ),
            ),

            max_abs_smd_global=(
                "abs_smd",
                "max",
            ),

            mean_abs_smd_global=(
                "abs_smd",
                "mean",
            ),

            n_global_fuerte=(
                "global_relevante",
                "sum",
            ),

            n_global_muy_fuerte=(
                "global_muy_fuerte",
                "sum",
            ),

            n_temporal_replicada=(
                "temporal_replicada",
                "sum",
            ),

            n_clusters_consistentes=(
                "clusters_consistentes",
                "sum",
            ),

            n_clusters_heterogeneos=(
                "clusters_heterogeneos",
                "sum",
            ),

            n_general_robusta=(
                "categoria",
                lambda s: (
                    s == "general_robusta"
                ).sum(),
            ),

            n_general_heterogenea=(
                "categoria",
                lambda s: (
                    s
                    == "general_heterogenea"
                ).sum(),
            ),

            n_especifica_perfil=(
                "categoria",
                lambda s: (
                    s
                    == "especifica_de_perfil"
                ).sum(),
            ),
        )
        .reset_index()
    )

    # -------------------------------------------------------------
    # CORRECCIÓN:
    #
    # En la versión anterior se hacía:
    #
    #     x.groupby("variable")["abs_smd"].idxmax()
    #
    # Eso falla si una variable tiene todas sus features con
    # abs_smd = NaN.
    #
    # Para elegir la mejor feature usamos únicamente filas cuyo
    # abs_smd sea finito. Esto NO elimina la variable del ranking:
    # simplemente deja sus columnas mejor_* como NaN cuando no
    # existe una comparación global válida.
    # -------------------------------------------------------------

    validas_mejor = x[
        x["abs_smd"].notna()
        & np.isfinite(x["abs_smd"])
    ].copy()

    if validas_mejor.empty:

        mejores = pd.DataFrame(
            columns=[
                "variable",
                "mejor_feature",
                "mejor_transformacion",
                "mejor_smd",
                "mejor_abs_smd",
                "mejor_categoria",
                "mejor_direccion",
                "mejor_naturaleza",
            ]
        )

    else:

        idx = (
            validas_mejor
            .groupby(
                "variable",
                dropna=False,
            )[
                "abs_smd"
            ]
            .idxmax()
        )

        mejores = validas_mejor.loc[
            idx,
            [
                "variable",
                "feature",
                "transformacion",
                "smd",
                "abs_smd",
                "categoria",
                "direccion_global",
                "naturaleza",
            ],
        ].copy()

        mejores = mejores.rename(
            columns={
                "feature":
                    "mejor_feature",

                "transformacion":
                    "mejor_transformacion",

                "smd":
                    "mejor_smd",

                "abs_smd":
                    "mejor_abs_smd",

                "categoria":
                    "mejor_categoria",

                "direccion_global":
                    "mejor_direccion",

                "naturaleza":
                    "mejor_naturaleza",
            }
        )

    ranking = ranking.merge(
        mejores,
        on="variable",
        how="left",
    )

    # Orden lexicográfico, no score:
    # primero robustas, luego heterogéneas,
    # luego fuerza global.
    ranking = ranking.sort_values(
        [
            "n_general_robusta",
            "n_general_heterogenea",
            "n_global_fuerte",
            "max_abs_smd_global",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
        na_position="last",
    ).reset_index(drop=True)

    ranking.insert(
        0,
        "ranking",
        np.arange(
            1,
            len(ranking) + 1,
        ),
    )

    print(
        f"Variables originales: "
        f"{len(ranking):,}"
    )

    n_sin_smd = int(
        (
            ranking[
                "n_features_smd_valido"
            ]
            == 0
        ).sum()
    )

    print(
        f"Variables sin SMD global válido: "
        f"{n_sin_smd:,}"
    )

    return ranking


# =====================================================================
# RESUMEN DE TRANSFORMACIONES
# =====================================================================

def construir_resumen_transformaciones(
    clasificacion: pd.DataFrame,
) -> pd.DataFrame:

    x = clasificacion[
        clasificacion[
            "cumple_min_n"
        ]
    ].copy()

    resumen = (
        x.groupby(
            [
                "transformacion",
                "naturaleza",
            ],
            dropna=False,
        )
        .agg(
            n_features=(
                "feature",
                "size",
            ),
            mediana_abs_smd=(
                "abs_smd",
                "median",
            ),
            media_abs_smd=(
                "abs_smd",
                "mean",
            ),
            max_abs_smd=(
                "abs_smd",
                "max",
            ),
            n_moderadas_o_mas=(
                "abs_smd",
                lambda s: (
                    s >= SMD_MODERADA
                ).sum(),
            ),
            n_fuertes_o_mas=(
                "abs_smd",
                lambda s: (
                    s >= SMD_FUERTE
                ).sum(),
            ),
            n_temporal_replicada=(
                "temporal_replicada",
                "sum",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "n_fuertes_o_mas",
                "max_abs_smd",
            ],
            ascending=[
                False,
                False,
            ],
            na_position="last",
        )
    )

    return resumen


# =====================================================================
# IMPRESIÓN DE RESULTADOS
# =====================================================================

def imprimir_top(
    df: pd.DataFrame,
    categoria: str,
    n: int = 20,
) -> None:

    x = df[
        df["categoria"]
        == categoria
    ].copy()

    subtitulo(
        categoria.upper()
    )

    if x.empty:
        print(
            "No se encontraron features "
            "en esta categoría."
        )
        return

    columnas = [
        "feature",
        "smd",
        "abs_smd",
        "magnitud_global",
        "direccion_global",
        "n_fotos",
        "mismo_signo_fotos",
        "n_clusters",
        "mismo_signo_clusters",
    ]

    columnas = [
        c
        for c in columnas
        if c in x.columns
    ]

    print(
        x[
            columnas
        ]
        .head(n)
        .to_string(
            index=False
        )
    )


# =====================================================================
# RESUMEN TXT
# =====================================================================

def generar_resumen_txt(
    clasificacion: pd.DataFrame,
    especificas: pd.DataFrame,
    ranking: pd.DataFrame,
) -> str:

    lineas = []

    lineas.append(
        "Z507 — CLASIFICACION DE SENALES PRE-BAJA+2"
    )

    lineas.append(
        "=" * 70
    )

    lineas.append("")

    lineas.append(
        "Criterios de magnitud SMD:"
    )

    lineas.append(
        "  |SMD| < 0.20       : debil"
    )

    lineas.append(
        "  0.20 <= |SMD| < .50: moderada"
    )

    lineas.append(
        "  0.50 <= |SMD| < .80: fuerte"
    )

    lineas.append(
        "  |SMD| >= 0.80      : muy fuerte"
    )

    lineas.append("")

    lineas.append(
        "IMPORTANTE:"
    )

    lineas.append(
        "Los cortes son reglas interpretativas."
    )

    lineas.append(
        "No constituyen tests de significancia."
    )

    lineas.append(
        "SMD describe asociacion, no causalidad."
    )

    lineas.append("")

    lineas.append(
        f"Features analizadas: "
        f"{len(clasificacion):,}"
    )

    lineas.append("")

    lineas.append(
        "Categorias:"
    )

    vc = (
        clasificacion[
            "categoria"
        ]
        .value_counts()
    )

    for categoria, n in vc.items():

        lineas.append(
            f"  {categoria:<40} "
            f"{n:>6,}"
        )

    lineas.append("")

    lineas.append(
        "Definiciones:"
    )

    lineas.append("")

    lineas.append(
        "general_robusta:"
    )

    lineas.append(
        "  señal global fuerte, replicada temporalmente "
        "y con signo consistente en los cinco clusters."
    )

    lineas.append("")

    lineas.append(
        "general_heterogenea:"
    )

    lineas.append(
        "  señal global fuerte y replicada temporalmente, "
        "pero con cambio de signo entre clusters."
    )

    lineas.append("")

    lineas.append(
        "especifica_de_perfil:"
    )

    lineas.append(
        "  señal que alcanza magnitud fuerte dentro de "
        "al menos un cluster aunque no domine globalmente."
    )

    lineas.append("")

    lineas.append(
        "fuerte_evidencia_temporal_limitada:"
    )

    lineas.append(
        "  señal global fuerte cuya cobertura temporal "
        "no permite afirmar replicacion."
    )

    lineas.append("")

    # -------------------------------------------------------------
    # Top robustas
    # -------------------------------------------------------------

    robustas = clasificacion[
        clasificacion[
            "categoria"
        ]
        == "general_robusta"
    ].head(20)

    lineas.append(
        "TOP SENALES GENERALES ROBUSTAS"
    )

    lineas.append(
        "-" * 70
    )

    if robustas.empty:

        lineas.append(
            "No se identificaron señales "
            "con este criterio."
        )

    else:

        for _, r in robustas.iterrows():

            n_fotos = (
                int(r["n_fotos"])
                if pd.notna(r["n_fotos"])
                else 0
            )

            n_clusters = (
                int(r["n_clusters"])
                if pd.notna(r["n_clusters"])
                else 0
            )

            lineas.append(
                f"{r['feature']}: "
                f"SMD={r['smd']:.4f}; "
                f"|SMD|={r['abs_smd']:.4f}; "
                f"fotos={n_fotos}; "
                f"clusters={n_clusters}"
            )

    lineas.append("")

    # -------------------------------------------------------------
    # Top heterogéneas
    # -------------------------------------------------------------

    hetero = clasificacion[
        clasificacion[
            "categoria"
        ]
        == "general_heterogenea"
    ].head(20)

    lineas.append(
        "TOP SENALES GENERALES HETEROGENEAS"
    )

    lineas.append(
        "-" * 70
    )

    if hetero.empty:

        lineas.append(
            "No se identificaron señales "
            "con este criterio."
        )

    else:

        for _, r in hetero.iterrows():

            lineas.append(
                f"{r['feature']}: "
                f"SMD global={r['smd']:.4f}; "
                f"SMD clusters="
                f"[{r['smd_cluster_min']:.4f}, "
                f"{r['smd_cluster_max']:.4f}]"
            )

    lineas.append("")

    # -------------------------------------------------------------
    # Variables
    # -------------------------------------------------------------

    lineas.append(
        "TOP VARIABLES ORIGINALES"
    )

    lineas.append(
        "-" * 70
    )

    if not ranking.empty:

        for _, r in (
            ranking
            .head(20)
            .iterrows()
        ):

            max_smd = r[
                "max_abs_smd_global"
            ]

            max_txt = (
                f"{max_smd:.4f}"
                if pd.notna(max_smd)
                and np.isfinite(max_smd)
                else "NA"
            )

            mejor = (
                r["mejor_feature"]
                if pd.notna(
                    r["mejor_feature"]
                )
                else "sin_feature_valida"
            )

            lineas.append(
                f"{int(r['ranking']):>2}. "
                f"{r['variable']}: "
                f"max |SMD|="
                f"{max_txt}; "
                f"mejor={mejor}"
            )

    lineas.append("")

    lineas.append(
        "NOTA METODOLOGICA"
    )

    lineas.append(
        "-" * 70
    )

    lineas.append(
        "La muestra BAJA+2 / FIEL utilizada en los análisis "
        "previos fue construida con fines comparativos."
    )

    lineas.append(
        "Las proporciones observadas no representan "
        "probabilidades poblacionales de baja."
    )

    lineas.append("")

    lineas.append(
        "Una señal fuerte globalmente puede desaparecer "
        "o cambiar de signo al condicionar por cluster."
    )

    lineas.append(
        "Esto se interpreta como heterogeneidad entre "
        "perfiles, no como contradiccion del análisis global."
    )

    lineas.append("")

    lineas.append(
        "Las transformaciones con una sola foto disponible "
        "se consideran evidencia temporal limitada."
    )

    lineas.append("")

    lineas.append(
        f"Comparaciones fuertes por cluster: "
        f"{len(especificas):,}"
    )

    return "\n".join(
        lineas
    )


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:

    titulo(
        "Z507 — CLASIFICACIÓN DE SEÑALES PRE-BAJA+2"
    )

    print(
        """
Este script consolida la evidencia generada por Z506.

Se analizan tres dimensiones por separado:

  1. magnitud global;
  2. consistencia temporal;
  3. consistencia entre clusters.

No se construye un score único arbitrario.
"""
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Entradas
    # -------------------------------------------------------------

    validar_archivos()

    (
        global_df,
        foto_df,
        cluster_df,
        consistencia_df,
        ranking_z506,
    ) = cargar_datos()

    validar_columnas(
        global_df,
        foto_df,
        cluster_df,
        consistencia_df,
    )

    # -------------------------------------------------------------
    # Clusters
    # -------------------------------------------------------------

    resumen_clusters = (
        construir_resumen_clusters(
            cluster_df
        )
    )

    # -------------------------------------------------------------
    # Consolidación
    # -------------------------------------------------------------

    clasificacion = (
        consolidar_features(
            global_df,
            consistencia_df,
            resumen_clusters,
        )
    )

    # -------------------------------------------------------------
    # Señales específicas
    # -------------------------------------------------------------

    especificas = (
        construir_senales_especificas(
            cluster_df,
            global_df,
        )
    )

    # -------------------------------------------------------------
    # Ranking variables
    # -------------------------------------------------------------

    ranking = (
        construir_ranking_variables(
            clasificacion
        )
    )

    # -------------------------------------------------------------
    # Transformaciones
    # -------------------------------------------------------------

    resumen_transformaciones = (
        construir_resumen_transformaciones(
            clasificacion
        )
    )

    # -------------------------------------------------------------
    # Subconjuntos interpretativos
    # -------------------------------------------------------------

    robustas = clasificacion[
        clasificacion[
            "categoria"
        ]
        == "general_robusta"
    ].copy()

    heterogeneas = clasificacion[
        clasificacion[
            "categoria"
        ]
        == "general_heterogenea"
    ].copy()

    # -------------------------------------------------------------
    # Mostrar resultados
    # -------------------------------------------------------------

    titulo(
        "RESULTADOS PRINCIPALES"
    )

    imprimir_top(
        clasificacion,
        "general_robusta",
    )

    imprimir_top(
        clasificacion,
        "general_heterogenea",
    )

    imprimir_top(
        clasificacion,
        "especifica_de_perfil",
    )

    subtitulo(
        "TOP VARIABLES ORIGINALES"
    )

    if ranking.empty:

        print(
            "No se pudo construir ranking."
        )

    else:

        columnas = [
            "ranking",
            "variable",
            "max_abs_smd_global",
            "n_global_fuerte",
            "n_general_robusta",
            "n_general_heterogenea",
            "mejor_feature",
            "mejor_categoria",
        ]

        print(
            ranking[
                columnas
            ]
            .head(30)
            .to_string(
                index=False
            )
        )

    subtitulo(
        "RESUMEN POR TRANSFORMACIÓN"
    )

    print(
        resumen_transformaciones
        .to_string(
            index=False
        )
    )

    # -------------------------------------------------------------
    # Guardado
    # -------------------------------------------------------------

    titulo(
        "GUARDADO DE RESULTADOS"
    )

    archivos_salida = {
        "clasificacion_features.csv":
            clasificacion,

        "senales_generales_robustas.csv":
            robustas,

        "senales_generales_heterogeneas.csv":
            heterogeneas,

        "senales_especificas_cluster.csv":
            especificas,

        "ranking_variables.csv":
            ranking,

        "resumen_clusters_features.csv":
            resumen_clusters,

        "resumen_transformaciones.csv":
            resumen_transformaciones,
    }

    for nombre, df in (
        archivos_salida.items()
    ):

        destino = (
            OUT_DIR / nombre
        )

        df.to_csv(
            destino,
            index=False,
        )

        print(
            f"{nombre:<38} "
            f"{df.shape}"
        )

    # -------------------------------------------------------------
    # TXT
    # -------------------------------------------------------------

    resumen_txt = generar_resumen_txt(
        clasificacion,
        especificas,
        ranking,
    )

    ruta_txt = (
        OUT_DIR
        / "resumen_z507.txt"
    )

    ruta_txt.write_text(
        resumen_txt,
        encoding="utf-8",
    )

    print(
        f"{'resumen_z507.txt':<38} "
        "OK"
    )

    # -------------------------------------------------------------
    # Validaciones finales
    # -------------------------------------------------------------

    titulo(
        "VALIDACIONES FINALES"
    )

    if (
        clasificacion[
            "feature"
        ].duplicated().any()
    ):
        raise ValueError(
            "Hay features duplicadas "
            "en la clasificación final."
        )

    if (
        clasificacion[
            "categoria"
        ].isna().any()
    ):
        raise ValueError(
            "Hay features sin categoría."
        )

    if not ranking.empty:

        if (
            ranking[
                "variable"
            ].duplicated().any()
        ):
            raise ValueError(
                "Hay variables duplicadas "
                "en el ranking final."
            )

    print(
        "Features únicas           : OK"
    )

    print(
        "Categorías completas      : OK"
    )

    print(
        "Variables ranking únicas  : OK"
    )

    print(
        "Z506 no fue recalculado   : OK"
    )

    print(
        "Clusters no recalculados  : OK"
    )

    # -------------------------------------------------------------
    # Final
    # -------------------------------------------------------------

    titulo(
        "Z507 FINALIZADO"
    )

    print(
        f"Directorio de salida:\n"
        f"{OUT_DIR}"
    )

    print()

    print(
        "Interpretación:"
    )

    print(
        """
- GENERAL ROBUSTA:
  señal fuerte globalmente, replicada en el tiempo y
  consistente entre los cinco perfiles.

- GENERAL HETEROGÉNEA:
  señal fuerte globalmente y replicada temporalmente,
  pero cuyo comportamiento cambia entre clusters.

- ESPECÍFICA DE PERFIL:
  señal que adquiere fuerza dentro de determinados
  clusters aunque no necesariamente a nivel global.

- EVIDENCIA TEMPORAL LIMITADA:
  señal fuerte cuya transformación sólo puede
  evaluarse en una cobertura temporal insuficiente.

Estas categorías son descriptivas.
No constituyen causalidad ni selección definitiva
de variables para el modelo de competencia.
"""
    )


# =====================================================================
# EJECUCIÓN
# =====================================================================

if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:

        print(
            "\nProceso interrumpido por el usuario."
        )

        sys.exit(130)

    except Exception as exc:

        print()
        print(
            "=" * 78
        )

        print(
            "ERROR EN Z507"
        )

        print(
            "=" * 78
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise