"""
z506_prebaja_features.py

DMEyF 2026
EDA predictiva pre-BAJA+2.

Objetivo
--------
Estudiar qué señales observables ANTES de BAJA+2 diferencian a los
clientes que posteriormente tendrán BAJA+2 de los clientes fieles.

Este script NO:
- reentrena Random Forest;
- recalcula clusters;
- utiliza la observación BAJA+2 como predictor;
- genera todavía el dataset definitivo de competencia.

Diseño temporal
---------------
Para un cliente BAJA+2:

    ... -> t=-2 -> t=-1 -> t=0 -> BAJA+2
                         ^
                         |
                    foto_ancla

t=0 es la última observación estrictamente anterior al evento.

Para los fieles se utiliza exactamente la misma foto calendario que
para los futuros BAJA+2.

Esto evita comparar, por ejemplo, futuros BAJA+2 observados en 202103
contra fieles observados en 202108.

Features descriptivas
---------------------
Para cada variable numérica:

    valor_t0
    delta1      = t0 - t-1
    delta2      = t0 - t-2
    media_2
    media_3
    min_3
    max_3
    std_3
    slope_2
    slope_3

Las features se calculan únicamente cuando existe historia suficiente.

Comparación
-----------
Se calcula Standardized Mean Difference (SMD):

    SMD = (media_BAJA2 - media_FIEL) / SD_pooled

Se generan resultados:
- globales;
- por foto_ancla;
- por cluster k5.

Los rankings aplican cobertura mínima para evitar que resultados
basados en muy pocos clientes dominen el análisis.

IMPORTANTE
----------
Este script es EDA predictiva.

Una señal encontrada aquí todavía NO implica que deba incorporarse
automáticamente al modelo final.

La validación temporal de competencia deberá diseñarse por separado.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import duckdb
import numpy as np
import pandas as pd


# =====================================================================
# CONFIGURACIÓN
# =====================================================================

DATASET = Path(
    "/data/dmeyf/datasets/competencia_01.csv"
)

ASIGNACIONES = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/"
    "perfiles/asignaciones_k2_k5.csv"
)

DICCIONARIO = Path(
    "/data/dmeyf/datasets/DiccionarioDatos_2026.ods"
)

OUT_DIR = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/"
    "prebaja_features"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Cobertura mínima para considerar una comparación
MIN_N_GLOBAL = 100
MIN_N_FOTO = 50
MIN_N_CLUSTER = 50

# Evita rankings enormes en pantalla
TOP_N_PRINT = 30

# Columnas que nunca deben transformarse como features
EXCLUIR = {
    "numero_de_cliente",
    "foto_mes",
    "clase_ternaria",
    "k5",
    "grupo",
    "foto_evento",
    "foto_ancla",
    "t",
}


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


def safe_float(x):
    """
    Convierte valores escalares a float.

    Devuelve NaN para pd.NA, None o valores no convertibles.
    """
    if x is None:
        return np.nan

    try:
        if pd.isna(x):
            return np.nan
    except Exception:
        pass

    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def slope(values: np.ndarray) -> float:
    """
    Pendiente lineal de una secuencia ordenada temporalmente.

    x = 0, 1, ..., n-1

    Requiere al menos dos observaciones válidas.
    """
    y = np.asarray(values, dtype="float64")

    mask = np.isfinite(y)
    y = y[mask]

    if len(y) < 2:
        return np.nan

    x = np.arange(len(y), dtype="float64")

    if np.all(y == y[0]):
        return 0.0

    try:
        return float(
            np.polyfit(
                x,
                y,
                1
            )[0]
        )
    except Exception:
        return np.nan


def pooled_sd(
    n1: int,
    sd1: float,
    n0: int,
    sd0: float,
) -> float:
    """
    Desvío estándar pooled.
    """
    if n1 < 2 or n0 < 2:
        return np.nan

    if not np.isfinite(sd1) or not np.isfinite(sd0):
        return np.nan

    denom = n1 + n0 - 2

    if denom <= 0:
        return np.nan

    var = (
        (n1 - 1) * sd1**2
        + (n0 - 1) * sd0**2
    ) / denom

    if var <= 0 or not np.isfinite(var):
        return np.nan

    return float(np.sqrt(var))


# =====================================================================
# DICCIONARIO
# =====================================================================

def cargar_diccionario() -> pd.DataFrame:

    titulo("CARGA DEL DICCIONARIO")

    if not DICCIONARIO.exists():
        warnings.warn(
            f"No se encontró el diccionario: {DICCIONARIO}"
        )

        return pd.DataFrame(
            columns=[
                "campo",
                "unidad",
                "Significado",
            ]
        )

    dic = pd.read_excel(
        DICCIONARIO,
        sheet_name="Diccionario",
        engine="odf",
    )

    print(
        f"Variables en diccionario: "
        f"{len(dic):,}"
    )

    print(
        f"Campos únicos          : "
        f"{dic['campo'].nunique():,}"
    )

    return dic


# =====================================================================
# ASIGNACIONES DE CLUSTER
# =====================================================================

def cargar_asignaciones() -> pd.DataFrame:

    titulo("CARGA DE CLUSTERS k=5")

    a = pd.read_csv(
        ASIGNACIONES,
        usecols=[
            "numero_de_cliente",
            "k5",
        ],
    )

    a["numero_de_cliente"] = (
        a["numero_de_cliente"]
        .astype("int64")
    )

    a = a.rename(
        columns={
            "k5": "cluster"
        }
    )

    print(
        f"Clientes asignados: "
        f"{a['numero_de_cliente'].nunique():,}"
    )

    print()
    print("Clusters:")
    print(
        a["cluster"]
        .value_counts()
        .sort_index()
    )

    return a


# =====================================================================
# CARGA DEL DATASET
# =====================================================================

def cargar_dataset(
    asignaciones: pd.DataFrame,
) -> pd.DataFrame:

    titulo("CARGA DEL DATASET")

    ids = asignaciones[
        "numero_de_cliente"
    ].tolist()

    con = duckdb.connect()

    # Cargamos todos los campos.
    # DuckDB resulta más robusto para este CSV que pandas.
    df = con.execute(
        f"""
        SELECT *
        FROM read_csv(
            '{DATASET}',
            sample_size=-1
        )
        """
    ).df()

    con.close()

    # Normalización del ID
    df["numero_de_cliente"] = pd.to_numeric(
        df["numero_de_cliente"],
        errors="coerce",
    ).astype("Int64")

    df = df[
        df["numero_de_cliente"].isin(ids)
    ].copy()

    df["numero_de_cliente"] = (
        df["numero_de_cliente"]
        .astype("int64")
    )

    df["foto_mes"] = pd.to_numeric(
        df["foto_mes"],
        errors="coerce",
    ).astype("Int64")

    df = df.merge(
        asignaciones,
        on="numero_de_cliente",
        how="inner",
        validate="many_to_one",
    )

    df = df.sort_values(
        [
            "numero_de_cliente",
            "foto_mes",
        ]
    ).reset_index(drop=True)

    print(
        f"Filas seleccionadas : "
        f"{len(df):,}"
    )

    print(
        f"Clientes            : "
        f"{df['numero_de_cliente'].nunique():,}"
    )

    print(
        f"Columnas             : "
        f"{df.shape[1]:,}"
    )

    return df


# =====================================================================
# IDENTIFICAR VARIABLES NUMÉRICAS
# =====================================================================

def detectar_variables(
    df: pd.DataFrame,
) -> list[str]:

    titulo("DETECCIÓN DE VARIABLES")

    variables = []

    for col in df.columns:

        if col in EXCLUIR:
            continue

        if col == "cluster":
            continue

        if col == "clase_ternaria":
            continue

        # Intentamos convertir.
        x = pd.to_numeric(
            df[col],
            errors="coerce",
        )

        # Tiene que existir al menos algún dato numérico.
        if x.notna().sum() == 0:
            continue

        variables.append(col)

        df[col] = x

    print(
        f"Variables numéricas utilizables: "
        f"{len(variables):,}"
    )

    return variables


# =====================================================================
# CONSTRUCCIÓN DEL DISEÑO TEMPORAL
# =====================================================================

def construir_eventos(
    df: pd.DataFrame,
) -> pd.DataFrame:

    titulo("IDENTIFICACIÓN DE BAJA+2")

    eventos = (
        df[
            df["clase_ternaria"]
            == "BAJA+2"
        ][
            [
                "numero_de_cliente",
                "foto_mes",
            ]
        ]
        .rename(
            columns={
                "foto_mes": "foto_evento"
            }
        )
        .copy()
    )

    # Debe existir un solo BAJA+2 por cliente
    n_eventos = (
        eventos
        .groupby("numero_de_cliente")
        .size()
    )

    if (n_eventos > 1).any():
        raise ValueError(
            "Hay clientes con más de una "
            "observación BAJA+2."
        )

    print(
        f"Clientes BAJA+2: "
        f"{eventos['numero_de_cliente'].nunique():,}"
    )

    print()
    print("Foto del evento:")
    print(
        eventos["foto_evento"]
        .value_counts()
        .sort_index()
    )

    return eventos


def construir_casos(
    df: pd.DataFrame,
    eventos: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    titulo("CONSTRUCCIÓN DE CASOS PRE-BAJA+2")

    casos = df.merge(
        eventos,
        on="numero_de_cliente",
        how="inner",
    )

    # Sólo observaciones estrictamente anteriores
    # al evento.
    casos = casos[
        casos["foto_mes"]
        < casos["foto_evento"]
    ].copy()

    casos = casos.sort_values(
        [
            "numero_de_cliente",
            "foto_mes",
        ]
    )

    # Ancla = última observación pre-evento
    anclas = (
        casos
        .groupby(
            "numero_de_cliente",
            as_index=False,
            group_keys=False,
        )
        .tail(1)[
            [
                "numero_de_cliente",
                "foto_mes",
                "foto_evento",
                "cluster",
                "clase_ternaria",
            ]
        ]
        .rename(
            columns={
                "foto_mes": "foto_ancla"
            }
        )
        .copy()
    )

    print(
        "Clientes BAJA+2 originales : "
        f"{eventos['numero_de_cliente'].nunique():,}"
    )

    print(
        "Con historia pre-evento    : "
        f"{anclas['numero_de_cliente'].nunique():,}"
    )

    perdidos = (
        eventos["numero_de_cliente"].nunique()
        - anclas["numero_de_cliente"].nunique()
    )

    print(
        "Sin historia pre-evento    : "
        f"{perdidos:,}"
    )

    print()
    print("Foto ancla:")
    print(
        anclas["foto_ancla"]
        .value_counts()
        .sort_index()
    )

    print()
    print("Clase en el ancla:")
    print(
        anclas["clase_ternaria"]
        .value_counts(
            dropna=False
        )
    )

    # Validación crítica
    if not (
        anclas["clase_ternaria"]
        == "CONTINUA"
    ).all():
        raise ValueError(
            "No todos los casos tienen "
            "CONTINUA en la foto ancla."
        )

    # Incorporamos foto_ancla a todas las filas
    # históricas de cada caso.
    casos = casos.merge(
        anclas[
            [
                "numero_de_cliente",
                "foto_ancla",
            ]
        ],
        on="numero_de_cliente",
        how="inner",
    )

    # Posición temporal relativa.
    #
    # No usamos diferencia de YYYYMM porque puede haber
    # huecos calendario.
    casos["t"] = (
        casos
        .groupby("numero_de_cliente")
        .cumcount()
    )

    max_t = (
        casos
        .groupby("numero_de_cliente")["t"]
        .transform("max")
    )

    casos["t"] = (
        casos["t"] - max_t
    )

    # La última observación pre-evento debe ser t=0
    check = (
        casos[
            casos["t"] == 0
        ][
            [
                "numero_de_cliente",
                "foto_mes",
            ]
        ]
        .merge(
            anclas[
                [
                    "numero_de_cliente",
                    "foto_ancla",
                ]
            ],
            on="numero_de_cliente",
        )
    )

    if not (
        check["foto_mes"]
        == check["foto_ancla"]
    ).all():
        raise ValueError(
            "Error en alineación temporal "
            "de los casos."
        )

    casos["grupo"] = "BAJA+2"

    print()
    print("Historia disponible:")
    print(
        casos
        .groupby("numero_de_cliente")
        .size()
        .value_counts()
        .sort_index()
    )

    return casos, anclas


# =====================================================================
# CONTROLES FIELES
# =====================================================================

def identificar_fieles(
    df: pd.DataFrame,
    eventos: pd.DataFrame,
) -> pd.DataFrame:

    ids_evento = set(
        eventos["numero_de_cliente"]
    )

    fieles = df[
        ~df["numero_de_cliente"].isin(
            ids_evento
        )
    ].copy()

    print(
        f"Clientes fieles disponibles: "
        f"{fieles['numero_de_cliente'].nunique():,}"
    )

    return fieles


def construir_controles(
    fieles: pd.DataFrame,
    anclas: pd.DataFrame,
) -> pd.DataFrame:

    titulo("CONSTRUCCIÓN DE CONTROLES CALENDARIO")

    fotos = sorted(
        anclas["foto_ancla"]
        .dropna()
        .unique()
        .tolist()
    )

    print(
        "Fotos ancla necesarias:",
        fotos,
    )

    bloques = []

    for foto in fotos:

        # Fieles observables hasta esa foto.
        hist = fieles[
            fieles["foto_mes"]
            <= foto
        ].copy()

        # Sólo interesan clientes que realmente
        # tengan observación EN la foto ancla.
        ids_con_foto = set(
            fieles.loc[
                fieles["foto_mes"] == foto,
                "numero_de_cliente",
            ]
        )

        hist = hist[
            hist["numero_de_cliente"]
            .isin(ids_con_foto)
        ].copy()

        if hist.empty:
            continue

        hist = hist.sort_values(
            [
                "numero_de_cliente",
                "foto_mes",
            ]
        )

        hist["foto_ancla"] = foto

        # Posición relativa dentro de cada cliente.
        hist["t"] = (
            hist
            .groupby("numero_de_cliente")
            .cumcount()
        )

        max_t = (
            hist
            .groupby("numero_de_cliente")["t"]
            .transform("max")
        )

        hist["t"] = (
            hist["t"] - max_t
        )

        hist["grupo"] = "FIEL"

        bloques.append(hist)

        print(
            f"{foto}: "
            f"{hist['numero_de_cliente'].nunique():,} "
            "fieles"
        )

    if not bloques:
        raise ValueError(
            "No fue posible construir "
            "controles fieles."
        )

    controles = pd.concat(
        bloques,
        ignore_index=True,
    )

    # Validación:
    # t=0 tiene que coincidir con foto_ancla.
    c0 = controles[
        controles["t"] == 0
    ]

    if not (
        c0["foto_mes"]
        == c0["foto_ancla"]
    ).all():
        raise ValueError(
            "Error temporal en controles."
        )

    return controles


# =====================================================================
# COBERTURA
# =====================================================================

def calcular_cobertura(
    casos: pd.DataFrame,
    controles: pd.DataFrame,
) -> pd.DataFrame:

    base = pd.concat(
        [
            casos,
            controles,
        ],
        ignore_index=True,
    )

    cobertura = (
        base
        .groupby(
            [
                "grupo",
                "foto_ancla",
                "cluster",
                "t",
            ],
            dropna=False,
        )
        .agg(
            filas=(
                "numero_de_cliente",
                "size",
            ),
            clientes=(
                "numero_de_cliente",
                "nunique",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "grupo",
                "foto_ancla",
                "cluster",
                "t",
            ]
        )
    )

    return cobertura


# =====================================================================
# FEATURE ENGINEERING DESCRIPTIVO
# =====================================================================

def construir_features(
    casos: pd.DataFrame,
    controles: pd.DataFrame,
    variables: list[str],
) -> pd.DataFrame:
    """
    Construye las features pre-evento de forma vectorizada.

    Mantiene exactamente las definiciones de la versión original:

        t0      = x(t=0)
        delta1  = x(t=0) - x(t=-1)
        delta2  = x(t=0) - x(t=-2)

        media2  = media de t=-1,0
        slope2  = pendiente OLS de t=-1,0

        media3  = media de t=-2,-1,0
        min3    = mínimo de t=-2,-1,0
        max3    = máximo de t=-2,-1,0
        std3    = desvío estándar muestral (ddof=1)
        slope3  = pendiente OLS de t=-2,-1,0

    Para ventanas equiespaciadas:

        slope2 = x0 - xm1
        slope3 = (x0 - xm2) / 2

    Esto evita loops cliente × variable y llamadas repetidas a
    pandas / np.polyfit.
    """

    titulo("CONSTRUCCIÓN DE FEATURES PRE-EVENTO")

    # -------------------------------------------------------------
    # Unificamos casos y controles.
    #
    # La clave lógica es:
    #   numero_de_cliente + grupo + foto_ancla + cluster
    #
    # Un fiel puede aparecer en varias fotos ancla.
    # -------------------------------------------------------------

    print("Preparando matriz temporal...")

    base = pd.concat(
        [
            casos,
            controles,
        ],
        ignore_index=True,
    )

    keys = [
        "numero_de_cliente",
        "grupo",
        "foto_ancla",
        "cluster",
    ]

    # -------------------------------------------------------------
    # Validaciones estructurales
    # -------------------------------------------------------------

    n_t0 = (
        base.loc[base["t"] == 0]
        .groupby(keys, dropna=False)
        .size()
    )

    if not (n_t0 == 1).all():
        raise ValueError(
            "Cada cliente/ancla debe tener "
            "exactamente un t=0."
        )

    # Sólo necesitamos t=0,-1,-2 para las features actuales.
    b = base.loc[
        base["t"].isin([-2, -1, 0]),
        keys + ["t"] + variables,
    ].copy()

    # Las variables ya fueron convertidas a numéricas por
    # detectar_variables(), pero aseguramos float64 para operar
    # vectorizadamente.
    b[variables] = b[variables].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # -------------------------------------------------------------
    # Metadata por cliente/ancla
    # -------------------------------------------------------------

    print("Construyendo metadata...")

    meta = (
        base.groupby(
            keys,
            dropna=False,
            sort=False,
        )
        .size()
        .rename("n_historia")
        .reset_index()
    )

    # -------------------------------------------------------------
    # Matrices X(t)
    #
    # Índice:
    #   cliente/grupo/foto/cluster
    #
    # Columnas:
    #   variables originales
    # -------------------------------------------------------------

    print("Construyendo matrices t=0, t=-1 y t=-2...")

    def matriz_t(t):
        x = (
            b.loc[
                b["t"] == t,
                keys + variables,
            ]
            .set_index(keys)
            [variables]
        )

        # Por validación temporal debería haber una fila por clave/t.
        if x.index.has_duplicates:
            raise ValueError(
                f"Hay claves duplicadas para t={t}."
            )

        return x.astype("float64")

    x0 = matriz_t(0)
    xm1 = matriz_t(-1)
    xm2 = matriz_t(-2)

    # Todos los resultados se alinean con las claves de t=0.
    idx = x0.index

    xm1 = xm1.reindex(idx)
    xm2 = xm2.reindex(idx)

    # -------------------------------------------------------------
    # Máscaras de disponibilidad.
    #
    # La versión original sólo calcula las ventanas cuando TODOS
    # los valores requeridos están presentes.
    # -------------------------------------------------------------

    valid2 = (
        x0.notna()
        & xm1.notna()
    )

    valid3 = (
        x0.notna()
        & xm1.notna()
        & xm2.notna()
    )

    # -------------------------------------------------------------
    # Features
    # -------------------------------------------------------------

    print("Calculando features vectorizadas...")

    t0 = x0.copy()

    delta1 = (
        x0 - xm1
    ).where(valid2)

    delta2 = (
        x0 - xm2
    ).where(
        x0.notna()
        & xm2.notna()
    )

    media2 = (
        (x0 + xm1) / 2.0
    ).where(valid2)

    # Con dos puntos equiespaciados [-1,0],
    # la pendiente OLS es exactamente x0-xm1.
    slope2 = (
        x0 - xm1
    ).where(valid2)

    media3 = (
        (xm2 + xm1 + x0) / 3.0
    ).where(valid3)

    # Para min/max evitamos que skipna modifique la semántica:
    # primero calculamos y luego aplicamos valid3.
    stack3 = np.stack(
        [
            xm2.to_numpy(dtype="float64"),
            xm1.to_numpy(dtype="float64"),
            x0.to_numpy(dtype="float64"),
        ],
        axis=0,
    )

    min3 = pd.DataFrame(
        np.nanmin(stack3, axis=0),
        index=idx,
        columns=variables,
    ).where(valid3)

    max3 = pd.DataFrame(
        np.nanmax(stack3, axis=0),
        index=idx,
        columns=variables,
    ).where(valid3)

    # Desvío estándar muestral, equivalente a:
    # np.std(arr, ddof=1)
    std3 = pd.DataFrame(
        np.nanstd(
            stack3,
            axis=0,
            ddof=1,
        ),
        index=idx,
        columns=variables,
    ).where(valid3)

    # Para t=[-2,-1,0], la pendiente OLS es exactamente:
    # (x0-xm2)/2
    slope3 = (
        (x0 - xm2) / 2.0
    ).where(valid3)

    # -------------------------------------------------------------
    # Renombrado
    # -------------------------------------------------------------

    def renombrar(df, sufijo):
        out = df.copy()
        out.columns = [
            f"{c}__{sufijo}"
            for c in out.columns
        ]
        return out

    bloques = [
        renombrar(t0, "t0"),
        renombrar(delta1, "delta1"),
        renombrar(delta2, "delta2"),
        renombrar(media2, "media2"),
        renombrar(slope2, "slope2"),
        renombrar(media3, "media3"),
        renombrar(min3, "min3"),
        renombrar(max3, "max3"),
        renombrar(std3, "std3"),
        renombrar(slope3, "slope3"),
    ]

    # -------------------------------------------------------------
    # IMPORTANTE:
    # La versión original agregaba las columnas por VARIABLE:
    #
    # var1__t0, var1__delta1, ..., var1__slope3,
    # var2__t0, ...
    #
    # Reproducimos ese orden para mantener compatibilidad.
    # -------------------------------------------------------------

    calc = pd.concat(
        bloques,
        axis=1,
    )

    orden = []

    sufijos = [
        "t0",
        "delta1",
        "delta2",
        "media2",
        "slope2",
        "media3",
        "min3",
        "max3",
        "std3",
        "slope3",
    ]

    for var in variables:
        for sufijo in sufijos:
            orden.append(
                f"{var}__{sufijo}"
            )

    calc = calc[orden]

    calc = calc.reset_index()

    # -------------------------------------------------------------
    # Metadata + features
    # -------------------------------------------------------------

    features = meta.merge(
        calc,
        on=keys,
        how="left",
        validate="one_to_one",
    )

    # Mismo orden inicial que la versión original.
    primeras = [
        "numero_de_cliente",
        "grupo",
        "foto_ancla",
        "cluster",
        "n_historia",
    ]

    features = features[
        primeras
        + [
            c
            for c in features.columns
            if c not in primeras
        ]
    ]

    # -------------------------------------------------------------
    # Validaciones finales
    # -------------------------------------------------------------

    esperado = len(meta)

    if len(features) != esperado:
        raise ValueError(
            "Cantidad inesperada de filas "
            "en la matriz de features."
        )

    if features.duplicated(keys).any():
        raise ValueError(
            "Hay filas duplicadas en "
            "la matriz de features."
        )

    print()
    print(
        f"Filas feature matrix : "
        f"{len(features):,}"
    )

    print(
        f"Columnas             : "
        f"{features.shape[1]:,}"
    )

    print()
    print("Filas por grupo:")

    print(
        features["grupo"]
        .value_counts()
    )

    print()
    print("Filas por foto y grupo:")

    print(
        pd.crosstab(
            features["foto_ancla"],
            features["grupo"],
            margins=True,
        )
    )

    return features


# =====================================================================
# COMPARACIÓN SMD
# =====================================================================

def comparar_feature(
    df: pd.DataFrame,
    feature: str,
) -> dict | None:

    b = pd.to_numeric(
        df.loc[
            df["grupo"] == "BAJA+2",
            feature,
        ],
        errors="coerce",
    ).dropna()

    f = pd.to_numeric(
        df.loc[
            df["grupo"] == "FIEL",
            feature,
        ],
        errors="coerce",
    ).dropna()

    n_baja = len(b)
    n_fiel = len(f)

    if n_baja == 0 or n_fiel == 0:
        return None

    mean_b = safe_float(
        b.mean()
    )

    mean_f = safe_float(
        f.mean()
    )

    sd_b = safe_float(
        b.std(ddof=1)
    )

    sd_f = safe_float(
        f.std(ddof=1)
    )

    psd = pooled_sd(
        n_baja,
        sd_b,
        n_fiel,
        sd_f,
    )

    if (
        np.isfinite(psd)
        and psd > 0
    ):
        smd = (
            mean_b - mean_f
        ) / psd
    else:
        smd = np.nan

    return {
        "feature": feature,
        "n_baja2": n_baja,
        "n_fiel": n_fiel,
        "media_baja2": mean_b,
        "media_fiel": mean_f,
        "sd_baja2": sd_b,
        "sd_fiel": sd_f,
        "smd": smd,
        "abs_smd": (
            abs(smd)
            if np.isfinite(smd)
            else np.nan
        ),
    }


def comparar_dataset(
    df: pd.DataFrame,
    feature_cols: list[str],
    min_n: int,
    contexto: str,
    foto_ancla=None,
    cluster=None,
) -> pd.DataFrame:

    filas = []

    for feature in feature_cols:

        r = comparar_feature(
            df,
            feature,
        )

        if r is None:
            continue

        r["contexto"] = contexto
        r["foto_ancla"] = foto_ancla
        r["cluster"] = cluster

        r["cumple_min_n"] = (
            r["n_baja2"] >= min_n
            and r["n_fiel"] >= min_n
        )

        filas.append(r)

    if not filas:
        return pd.DataFrame()

    out = pd.DataFrame(filas)

    out = out.sort_values(
        [
            "cumple_min_n",
            "abs_smd",
        ],
        ascending=[
            False,
            False,
        ],
    )

    return out


def generar_comparaciones(
    features: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:

    titulo("COMPARACIÓN BAJA+2 vs FIELES")

    metadata = {
        "numero_de_cliente",
        "grupo",
        "foto_ancla",
        "cluster",
        "n_historia",
    }

    feature_cols = [
        c
        for c in features.columns
        if c not in metadata
    ]

    # -------------------------------------------------------------
    # GLOBAL
    # -------------------------------------------------------------

    global_df = comparar_dataset(
        features,
        feature_cols,
        MIN_N_GLOBAL,
        contexto="GLOBAL",
    )

    # -------------------------------------------------------------
    # POR FOTO
    # -------------------------------------------------------------

    por_foto = []

    for foto, g in features.groupby(
        "foto_ancla"
    ):

        r = comparar_dataset(
            g,
            feature_cols,
            MIN_N_FOTO,
            contexto="FOTO",
            foto_ancla=int(foto),
        )

        if not r.empty:
            por_foto.append(r)

    if por_foto:

        foto_df = pd.concat(
            por_foto,
            ignore_index=True,
        )

    else:

        foto_df = pd.DataFrame()

    # -------------------------------------------------------------
    # POR CLUSTER
    # -------------------------------------------------------------

    por_cluster = []

    for cluster, g in features.groupby(
        "cluster"
    ):

        r = comparar_dataset(
            g,
            feature_cols,
            MIN_N_CLUSTER,
            contexto="CLUSTER",
            cluster=int(cluster),
        )

        if not r.empty:
            por_cluster.append(r)

    if por_cluster:

        cluster_df = pd.concat(
            por_cluster,
            ignore_index=True,
        )

    else:

        cluster_df = pd.DataFrame()

    return (
        global_df,
        foto_df,
        cluster_df,
    )


# =====================================================================
# PARSEO DEL NOMBRE DE FEATURE
# =====================================================================

def separar_feature(
    resultados: pd.DataFrame,
) -> pd.DataFrame:

    if resultados.empty:
        return resultados

    x = resultados.copy()

    partes = x["feature"].str.rsplit(
        "__",
        n=1,
        expand=True,
    )

    x["variable"] = partes[0]
    x["transformacion"] = partes[1]

    return x


# =====================================================================
# INCORPORAR DICCIONARIO
# =====================================================================

def agregar_diccionario(
    resultados: pd.DataFrame,
    diccionario: pd.DataFrame,
) -> pd.DataFrame:

    if resultados.empty:
        return resultados

    if diccionario.empty:
        return resultados

    cols = [
        "campo",
        "unidad",
        "Significado",
    ]

    d = diccionario[
        cols
    ].copy()

    out = resultados.merge(
        d,
        left_on="variable",
        right_on="campo",
        how="left",
    )

    out = out.drop(
        columns=["campo"]
    )

    return out


# =====================================================================
# CONSISTENCIA ENTRE FOTOS
# =====================================================================

def consistencia_fotos(
    foto_df: pd.DataFrame,
) -> pd.DataFrame:

    """
    Resume si una feature mantiene dirección y magnitud
    a través de las distintas fotos calendario.
    """

    if foto_df.empty:
        return pd.DataFrame()

    x = foto_df[
        foto_df["cumple_min_n"]
    ].copy()

    x = x[
        x["smd"].notna()
    ]

    if x.empty:
        return pd.DataFrame()

    filas = []

    for feature, g in x.groupby(
        "feature"
    ):

        smd = g["smd"].to_numpy(
            dtype="float64"
        )

        signos = np.sign(smd)

        # Ignoramos exactamente cero para consistencia.
        signos_no_cero = signos[
            signos != 0
        ]

        if len(signos_no_cero) == 0:
            mismo_signo = True
        else:
            mismo_signo = (
                np.all(signos_no_cero > 0)
                or np.all(signos_no_cero < 0)
            )

        filas.append(
            {
                "feature": feature,
                "n_fotos": g[
                    "foto_ancla"
                ].nunique(),
                "smd_mean": float(
                    np.mean(smd)
                ),
                "smd_median": float(
                    np.median(smd)
                ),
                "abs_smd_mean": float(
                    np.mean(
                        np.abs(smd)
                    )
                ),
                "abs_smd_min": float(
                    np.min(
                        np.abs(smd)
                    )
                ),
                "abs_smd_max": float(
                    np.max(
                        np.abs(smd)
                    )
                ),
                "mismo_signo":
                    bool(mismo_signo),
            }
        )

    out = pd.DataFrame(filas)

    out = out.sort_values(
        [
            "mismo_signo",
            "n_fotos",
            "abs_smd_mean",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )

    return out


# =====================================================================
# RANKING DE VARIABLES ORIGINALES
# =====================================================================

def ranking_variables(
    global_df: pd.DataFrame,
) -> pd.DataFrame:

    """
    Evita que una misma variable aparezca muchas veces por
    sus transformaciones.

    Resume la mejor transformación de cada variable original.
    """

    if global_df.empty:
        return pd.DataFrame()

    x = global_df[
        global_df["cumple_min_n"]
    ].copy()

    x = separar_feature(x)

    x = x[
        x["abs_smd"].notna()
    ]

    if x.empty:
        return pd.DataFrame()

    idx = (
        x.groupby("variable")[
            "abs_smd"
        ]
        .idxmax()
    )

    out = x.loc[idx].copy()

    out = out.sort_values(
        "abs_smd",
        ascending=False,
    )

    return out


# =====================================================================
# IMPRESIÓN
# =====================================================================

def imprimir_ranking(
    df: pd.DataFrame,
    titulo_txt: str,
    n: int = TOP_N_PRINT,
) -> None:

    subtitulo(titulo_txt)

    if df.empty:
        print("Sin resultados.")
        return

    x = df.copy()

    if "cumple_min_n" in x.columns:
        x = x[
            x["cumple_min_n"]
        ]

    if x.empty:
        print(
            "No hay resultados que cumplan "
            "la cobertura mínima."
        )
        return

    cols = [
        c
        for c in [
            "feature",
            "variable",
            "transformacion",
            "foto_ancla",
            "cluster",
            "n_baja2",
            "n_fiel",
            "media_baja2",
            "media_fiel",
            "smd",
            "abs_smd",
        ]
        if c in x.columns
    ]

    print(
        x.head(n)[cols]
        .to_string(
            index=False,
            float_format=lambda z: (
                f"{z:.4f}"
            ),
        )
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    titulo(
        "Z506 — EDA PREDICTIVA PRE-BAJA+2"
    )

    print(
        """
Objetivo:

Comparar futuros BAJA+2 contra clientes fieles usando exclusivamente
información disponible ANTES del evento.

Para BAJA+2:
    t=0 = última observación estrictamente anterior a BAJA+2.

Para fieles:
    t=0 = la MISMA foto calendario utilizada como ancla para los casos.

Esto evita utilizar información contemporánea o posterior al evento
y reduce la confusión entre señal predictiva y efecto calendario.
"""
    )

    # -------------------------------------------------------------
    # CARGAS
    # -------------------------------------------------------------

    diccionario = cargar_diccionario()

    asignaciones = cargar_asignaciones()

    df = cargar_dataset(
        asignaciones
    )

    variables = detectar_variables(
        df
    )

    # -------------------------------------------------------------
    # DISEÑO TEMPORAL
    # -------------------------------------------------------------

    eventos = construir_eventos(
        df
    )

    casos, anclas = construir_casos(
        df,
        eventos,
    )

    fieles = identificar_fieles(
        df,
        eventos,
    )

    controles = construir_controles(
        fieles,
        anclas,
    )

    # -------------------------------------------------------------
    # VALIDACIONES
    # -------------------------------------------------------------

    titulo("VALIDACIÓN DEL DISEÑO")

    print(
        f"Casos con ancla pre-evento : "
        f"{anclas['numero_de_cliente'].nunique():,}"
    )

    print(
        f"Fieles únicos              : "
        f"{fieles['numero_de_cliente'].nunique():,}"
    )

    print()
    print("Casos por foto ancla:")

    print(
        anclas["foto_ancla"]
        .value_counts()
        .sort_index()
    )

    print()
    print("Controles por foto ancla:")

    print(
        controles[
            controles["t"] == 0
        ]
        .groupby("foto_ancla")[
            "numero_de_cliente"
        ]
        .nunique()
        .sort_index()
    )

    # -------------------------------------------------------------
    # COBERTURA
    # -------------------------------------------------------------

    cobertura = calcular_cobertura(
        casos,
        controles,
    )

    cobertura.to_csv(
        OUT_DIR / "cobertura_temporal.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # FEATURES
    # -------------------------------------------------------------

    features = construir_features(
        casos,
        controles,
        variables,
    )

    features.to_csv(
        OUT_DIR / "features_prebaja.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # COMPARACIONES
    # -------------------------------------------------------------

    (
        global_df,
        foto_df,
        cluster_df,
    ) = generar_comparaciones(
        features
    )

    global_df = separar_feature(
        global_df
    )

    foto_df = separar_feature(
        foto_df
    )

    cluster_df = separar_feature(
        cluster_df
    )

    global_df = agregar_diccionario(
        global_df,
        diccionario,
    )

    foto_df = agregar_diccionario(
        foto_df,
        diccionario,
    )

    cluster_df = agregar_diccionario(
        cluster_df,
        diccionario,
    )

    # -------------------------------------------------------------
    # CONSISTENCIA TEMPORAL
    # -------------------------------------------------------------

    consistencia = consistencia_fotos(
        foto_df
    )

    consistencia = separar_feature(
        consistencia
    )

    consistencia = agregar_diccionario(
        consistencia,
        diccionario,
    )

    # -------------------------------------------------------------
    # RANKING POR VARIABLE ORIGINAL
    # -------------------------------------------------------------

    ranking = ranking_variables(
        global_df
    )

    ranking = agregar_diccionario(
        ranking,
        diccionario,
    )

    # -------------------------------------------------------------
    # GUARDADO
    # -------------------------------------------------------------

    global_df.to_csv(
        OUT_DIR / "comparacion_global.csv",
        index=False,
    )

    foto_df.to_csv(
        OUT_DIR / "comparacion_por_foto.csv",
        index=False,
    )

    cluster_df.to_csv(
        OUT_DIR / "comparacion_por_cluster.csv",
        index=False,
    )

    consistencia.to_csv(
        OUT_DIR / "consistencia_por_foto.csv",
        index=False,
    )

    ranking.to_csv(
        OUT_DIR / "ranking_variables.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # RESULTADOS EN CONSOLA
    # -------------------------------------------------------------

    imprimir_ranking(
        global_df,
        "TOP FEATURES — COMPARACIÓN GLOBAL",
    )

    imprimir_ranking(
        ranking,
        "TOP VARIABLES ORIGINALES",
    )

    subtitulo(
        "FEATURES CONSISTENTES ENTRE FOTOS"
    )

    if consistencia.empty:

        print("Sin resultados.")

    else:

        x = consistencia[
            (
                consistencia["mismo_signo"]
            )
            & (
                consistencia["n_fotos"] >= 3
            )
        ].copy()

        cols = [
            "feature",
            "variable",
            "transformacion",
            "n_fotos",
            "smd_mean",
            "abs_smd_mean",
            "abs_smd_min",
            "abs_smd_max",
            "mismo_signo",
        ]

        cols = [
            c
            for c in cols
            if c in x.columns
        ]

        print(
            x.head(
                TOP_N_PRINT
            )[cols]
            .to_string(
                index=False,
                float_format=lambda z: (
                    f"{z:.4f}"
                ),
            )
        )

    # -------------------------------------------------------------
    # CTRX_QUARTER
    # -------------------------------------------------------------

    subtitulo(
        "SEGUIMIENTO ESPECÍFICO — ctrx_quarter"
    )

    if not global_df.empty:

        q = global_df[
            global_df["variable"]
            == "ctrx_quarter"
        ].copy()

        if q.empty:

            print(
                "ctrx_quarter no aparece "
                "en los resultados."
            )

        else:

            cols = [
                "feature",
                "transformacion",
                "n_baja2",
                "n_fiel",
                "media_baja2",
                "media_fiel",
                "smd",
                "abs_smd",
                "cumple_min_n",
            ]

            print(
                q[cols]
                .sort_values(
                    "abs_smd",
                    ascending=False,
                )
                .to_string(
                    index=False,
                    float_format=lambda z: (
                        f"{z:.4f}"
                    ),
                )
            )

    # -------------------------------------------------------------
    # FIN
    # -------------------------------------------------------------

    titulo("RESULTADOS GUARDADOS")

    for archivo in [
        "cobertura_temporal.csv",
        "features_prebaja.csv",
        "comparacion_global.csv",
        "comparacion_por_foto.csv",
        "comparacion_por_cluster.csv",
        "consistencia_por_foto.csv",
        "ranking_variables.csv",
    ]:

        print(
            OUT_DIR / archivo
        )

    print()
    print(
        "INTERPRETACIÓN:"
    )

    print(
        """
1. comparacion_global.csv
   Busca señales pre-evento agregadas.

2. comparacion_por_foto.csv
   Permite verificar que la señal no sea producto de una única foto.

3. consistencia_por_foto.csv
   Es especialmente importante:
   una feature que mantiene signo y magnitud en distintas fotos es
   más interesante que una feature extrema en un único período.

4. comparacion_por_cluster.csv
   Permite estudiar heterogeneidad entre perfiles de clientes.

5. ranking_variables.csv
   Resume la mejor transformación encontrada para cada variable
   original.

6. features_prebaja.csv
   Es una matriz experimental para EDA.
   NO debe considerarse todavía el dataset final de competencia.

Siguiente pregunta:

    ¿Qué señales observadas en z505 alrededor del evento sobreviven
    cuando sólo utilizamos información conocida ANTES de BAJA+2?
"""
    )


if __name__ == "__main__":
    main()