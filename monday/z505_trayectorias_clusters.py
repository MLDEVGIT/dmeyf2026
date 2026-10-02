"""
z505_trayectorias_clusters.py

DMEyF 2026
Trayectorias temporales de clientes BAJA+2 y fieles dentro de los
clusters k=5 obtenidos previamente.

Objetivo
--------
Estudiar la dinámica temporal de las variables originales distinguiendo:

1. tiempo relativo al evento BAJA+2;
2. tiempo relativo al final de la ventana para clientes fieles.

Los clusters NO se recalculan en este script.

Definiciones temporales
-----------------------
BAJA+2:
    t = 0  -> observación etiquetada BAJA+2
    t = -1 -> observación inmediatamente anterior
    t = -2 -> dos observaciones anteriores
    ...

    Toda observación posterior a BAJA+2 se descarta.

FIEL:
    t = 0  -> última observación disponible
    t = -1 -> observación inmediatamente anterior
    ...

IMPORTANTE
----------
Para BAJA+2, t=0 está alineado con el evento observado.

Para fieles, t=0 NO es un evento:
es solamente el final de la ventana disponible.

El eje t representa posición relativa entre observaciones y no
necesariamente distancia exacta en meses calendario.

Este script es EDA explicativa.
NO genera todavía features definitivas para la competencia.

Salidas
-------
/data/dmeyf/datasets/evaluacion_clusters/trayectorias_v2/

    cobertura_temporal.csv
    composicion_clusters.csv
    trayectorias_baja2.csv
    trayectorias_fieles.csv
    deltas_individuales.csv
    deltas_resumen.csv
    comparacion_deltas.csv
    pendientes_individuales.csv
    pendientes_resumen.csv
    comparacion_pendientes.csv
    ranking_candidatas.csv
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


# ======================================================================
# CONFIGURACIÓN
# ======================================================================

DATASET = Path(
    "/data/dmeyf/datasets/competencia_01.csv"
)

ASIGNACIONES = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/"
    "perfiles/asignaciones_k2_k5.csv"
)

DIR_SALIDA = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/trayectorias_v2"
)

DIR_SALIDA.mkdir(
    parents=True,
    exist_ok=True,
)


ID_COL = "numero_de_cliente"
MES_COL = "foto_mes"
TARGET_COL = "clase_ternaria"
CLUSTER_COL = "cluster"
GRUPO_COL = "grupo"

BAJA2 = "BAJA+2"
FIEL = "FIEL"


# Deltas temporales que vamos a analizar.
#
# delta1 = x(t=0) - x(t=-1)
# delta2 = x(t=0) - x(t=-2)
# delta3 = x(t=0) - x(t=-3)
#
# No todos los clientes BAJA+2 poseen suficiente historia para
# calcular los tres.
LAGS_DELTA = (1, 2, 3)


# ======================================================================
# VARIABLES CANDIDATAS
# ======================================================================

VARIABLES_CANDIDATAS = [

    # Actividad general
    "active_quarter",
    "ctrx_quarter",
    "cproductos",

    # Canales
    "internet",
    "thomebanking",
    "chomebanking_transacciones",
    "cmobile_app_trx",

    # Débito / ATM
    "ctarjeta_debito_transacciones",
    "mautoservicio",
    "catm_trx",
    "cextraccion_autoservicio",
    "mextraccion_autoservicio",

    # Transferencias / pagos
    "ctransferencias_emitidas",
    "cpagomiscuentas",
    "ccuenta_debitos_automaticos",
    "cpayroll_trx",
    "mpayroll",

    # Tarjetas
    "ctarjeta_visa",
    "ctarjeta_master",

    # Visa
    "ctarjeta_visa_transacciones",
    "Visa_cconsumos",
    "mtarjeta_visa_consumo",
    "ctarjeta_visa_debitos_automaticos",
    "Visa_mpagospesos",

    # Estados
    "Visa_status",
    "Master_status",

    # Fechas / mora
    "Visa_Finiciomora",
    "Master_Finiciomora",
    "Visa_fechaalta",
    "Master_fechaalta",

    # Comisiones
    "ccomisiones_otras",
    "mcomisiones_otras",
    "ccomisiones_mantenimiento",
    "mcomisiones_mantenimiento",
    "mcomisiones",
]


# ======================================================================
# UTILIDADES
# ======================================================================

def encabezado(texto: str) -> None:

    print()
    print("=" * 78)
    print(texto)
    print("=" * 78)


def safe_numeric(
    s: pd.Series,
) -> pd.Series:
    """
    Convierte una serie a numérica de forma robusta.
    """

    return (
        pd.to_numeric(
            s,
            errors="coerce",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )


def safe_float(
    valor,
) -> float:
    """
    Convierte a float preservando faltantes como np.nan.

    Evita errores del tipo:

        float(pd.NA)

    que puede aparecer, por ejemplo, al calcular std con una única
    observación válida.
    """

    if pd.isna(valor):
        return np.nan

    return float(valor)


def pooled_smd(
    media_a: float,
    sd_a: float,
    media_b: float,
    sd_b: float,
) -> float:
    """
    Standardized Mean Difference.

    Se usa como medida descriptiva de separación entre BAJA+2 y FIEL.

    NO representa causalidad.
    NO representa importancia predictiva.
    """

    valores = [
        media_a,
        sd_a,
        media_b,
        sd_b,
    ]

    if any(
        pd.isna(v)
        for v in valores
    ):
        return np.nan

    pooled = np.sqrt(
        (
            sd_a ** 2
            + sd_b ** 2
        )
        / 2.0
    )

    if pooled <= 1e-12:
        return np.nan

    return float(
        (
            media_a
            - media_b
        )
        / pooled
    )


# ======================================================================
# 1. CARGAR ASIGNACIONES DE CLUSTERS
# ======================================================================

def cargar_asignaciones() -> pd.DataFrame:

    if not ASIGNACIONES.exists():

        raise FileNotFoundError(
            f"No existe {ASIGNACIONES}"
        )

    a = pd.read_csv(
        ASIGNACIONES,
        usecols=[
            ID_COL,
            "k5",
        ],
    )

    a[ID_COL] = (
        pd.to_numeric(
            a[ID_COL],
            errors="raise",
        )
        .astype("int64")
    )

    a["k5"] = (
        pd.to_numeric(
            a["k5"],
            errors="raise",
        )
        .astype(int)
    )

    if a[ID_COL].duplicated().any():

        raise ValueError(
            "Hay clientes duplicados "
            "en asignaciones_k2_k5.csv."
        )

    a = a.rename(
        columns={
            "k5": CLUSTER_COL
        }
    )

    return a


# ======================================================================
# 2. CARGAR DATASET
# ======================================================================

def cargar_datos(
    asignaciones: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:

    if not DATASET.exists():

        raise FileNotFoundError(
            f"No existe {DATASET}"
        )

    encabezado(
        "CARGA DEL DATASET"
    )

    print(DATASET)

    con = duckdb.connect()

    df = con.execute(
        f"""
        SELECT *
        FROM read_csv(
            '{DATASET.as_posix()}',
            sample_size=-1
        )
        """
    ).df()

    con.close()

    df[ID_COL] = (
        pd.to_numeric(
            df[ID_COL],
            errors="raise",
        )
        .astype("int64")
    )

    ids = set(
        asignaciones[ID_COL]
    )

    df = df[
        df[ID_COL].isin(ids)
    ].copy()

    df = df.merge(
        asignaciones,
        on=ID_COL,
        how="inner",
        validate="many_to_one",
    )

    variables = [
        c
        for c in VARIABLES_CANDIDATAS
        if c in df.columns
    ]

    faltantes = [
        c
        for c in VARIABLES_CANDIDATAS
        if c not in df.columns
    ]

    print(
        f"Filas seleccionadas : "
        f"{len(df):,}"
    )

    print(
        f"Clientes            : "
        f"{df[ID_COL].nunique():,}"
    )

    print(
        f"Variables disponibles: "
        f"{len(variables)}"
    )

    if faltantes:

        print(
            "\nVariables candidatas "
            "no encontradas:"
        )

        for c in faltantes:
            print(
                f"  - {c}"
            )

    return (
        df,
        variables,
    )


# ======================================================================
# 3. CONSTRUIR EJE TEMPORAL
# ======================================================================

def preparar_tiempo(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye el eje temporal.

    BAJA+2
    ------
    t=0 es la observación cuya clase_ternaria == BAJA+2.

    Las observaciones posteriores al evento se eliminan.

    FIEL
    ----
    t=0 es la última observación disponible.

    En ambos casos t representa posición relativa entre observaciones,
    no necesariamente meses calendario consecutivos.
    """

    encabezado(
        "CONSTRUCCIÓN DEL EJE TEMPORAL"
    )

    df = df.copy()

    df[MES_COL] = (
        pd.to_numeric(
            df[MES_COL],
            errors="raise",
        )
        .astype(int)
    )

    df[TARGET_COL] = (
        df[TARGET_COL]
        .astype("string")
    )

    df = (
        df.sort_values(
            [
                ID_COL,
                MES_COL,
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # ------------------------------------------------------------------
    # Identificar clientes que alguna vez tienen BAJA+2
    # ------------------------------------------------------------------

    tiene_baja2 = (
        df.groupby(
            ID_COL
        )[TARGET_COL]
        .transform(
            lambda s:
                (s == BAJA2).any()
        )
    )

    df[GRUPO_COL] = np.where(
        tiene_baja2,
        BAJA2,
        FIEL,
    )

    # ------------------------------------------------------------------
    # Validar un único BAJA+2 por cliente
    # ------------------------------------------------------------------

    eventos_por_cliente = (
        df.loc[
            df[GRUPO_COL] == BAJA2
        ]
        .groupby(
            ID_COL
        )[TARGET_COL]
        .apply(
            lambda s:
                int(
                    (s == BAJA2).sum()
                )
        )
    )

    malos = eventos_por_cliente[
        eventos_por_cliente != 1
    ]

    if not malos.empty:

        raise ValueError(
            "Hay clientes BAJA+2 con una "
            "cantidad de eventos distinta de 1:\n"
            f"{malos.to_string()}"
        )

    # ==================================================================
    # BAJA+2
    # ==================================================================

    baja = df[
        df[GRUPO_COL] == BAJA2
    ].copy()

    mes_evento = (
        baja.loc[
            baja[TARGET_COL] == BAJA2,
            [
                ID_COL,
                MES_COL,
            ],
        ]
        .rename(
            columns={
                MES_COL:
                    "_mes_evento"
            }
        )
    )

    baja = baja.merge(
        mes_evento,
        on=ID_COL,
        how="left",
        validate="many_to_one",
    )

    n_antes = len(
        baja
    )

    # Eliminamos explícitamente todo lo posterior al BAJA+2.
    baja = baja[
        baja[MES_COL]
        <= baja["_mes_evento"]
    ].copy()

    n_despues = len(
        baja
    )

    print()
    print("BAJA+2:")

    print(
        f"  filas antes del corte          : "
        f"{n_antes:,}"
    )

    print(
        f"  filas hasta el evento          : "
        f"{n_despues:,}"
    )

    print(
        f"  filas posteriores descartadas  : "
        f"{n_antes - n_despues:,}"
    )

    baja = baja.sort_values(
        [
            ID_COL,
            MES_COL,
        ]
    )

    baja["_orden"] = (
        baja.groupby(
            ID_COL
        )
        .cumcount()
    )

    baja["_n_fotos"] = (
        baja.groupby(
            ID_COL
        )[ID_COL]
        .transform(
            "size"
        )
    )

    baja["t"] = (
        baja["_orden"]
        - baja["_n_fotos"]
        + 1
    ).astype(int)

    # ==================================================================
    # FIELES
    # ==================================================================

    fiel = df[
        df[GRUPO_COL] == FIEL
    ].copy()

    fiel = fiel.sort_values(
        [
            ID_COL,
            MES_COL,
        ]
    )

    fiel["_orden"] = (
        fiel.groupby(
            ID_COL
        )
        .cumcount()
    )

    fiel["_n_fotos"] = (
        fiel.groupby(
            ID_COL
        )[ID_COL]
        .transform(
            "size"
        )
    )

    fiel["t"] = (
        fiel["_orden"]
        - fiel["_n_fotos"]
        + 1
    ).astype(int)

    fiel["_mes_evento"] = pd.NA

    # ==================================================================
    # UNIR
    # ==================================================================

    out = pd.concat(
        [
            baja,
            fiel,
        ],
        ignore_index=True,
        sort=False,
    )

    out = (
        out.sort_values(
            [
                ID_COL,
                MES_COL,
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # ==================================================================
    # RESUMEN POR CLIENTE
    # ==================================================================

    clientes = (
        out.groupby(
            ID_COL
        )
        .agg(
            grupo=(
                GRUPO_COL,
                "first",
            ),
            n_fotos=(
                "_n_fotos",
                "first",
            ),
            cluster=(
                CLUSTER_COL,
                "first",
            ),
        )
        .reset_index()
    )

    print()
    print(
        "Cantidad de observaciones "
        "utilizadas × grupo:"
    )

    print(
        pd.crosstab(
            clientes["n_fotos"],
            clientes["grupo"],
            margins=True,
        ).to_string()
    )

    # ==================================================================
    # VALIDACIONES
    # ==================================================================

    n_clientes = (
        clientes[ID_COL]
        .nunique()
    )

    if n_clientes != 8134:

        raise ValueError(
            "Se esperaban 8.134 clientes "
            f"y quedaron {n_clientes:,}."
        )

    n_baja = int(
        (
            clientes[GRUPO_COL]
            == BAJA2
        ).sum()
    )

    n_fiel = int(
        (
            clientes[GRUPO_COL]
            == FIEL
        ).sum()
    )

    if n_baja != 4067:

        raise ValueError(
            "Se esperaban 4.067 clientes "
            f"BAJA+2 y quedaron {n_baja:,}."
        )

    if n_fiel != 4067:

        raise ValueError(
            "Se esperaban 4.067 clientes "
            f"fieles y quedaron {n_fiel:,}."
        )

    # ------------------------------------------------------------------
    # t=0 BAJA+2
    # ------------------------------------------------------------------

    ultimos_baja = out[
        (
            out[GRUPO_COL]
            == BAJA2
        )
        &
        (
            out["t"]
            == 0
        )
    ]

    if len(
        ultimos_baja
    ) != 4067:

        raise ValueError(
            "No existe exactamente una "
            "observación t=0 por cliente BAJA+2."
        )

    if not (
        ultimos_baja[TARGET_COL]
        == BAJA2
    ).all():

        raise ValueError(
            "Hay clientes BAJA+2 cuyo t=0 "
            "no coincide con BAJA+2."
        )

    # ------------------------------------------------------------------
    # Comprobar que no sobrevivió información posterior al evento
    # ------------------------------------------------------------------

    mes_evento_numeric = (
        pd.to_numeric(
            out["_mes_evento"],
            errors="coerce",
        )
    )

    posterior = (
        (
            out[GRUPO_COL]
            == BAJA2
        )
        &
        out["_mes_evento"].notna()
        &
        (
            out[MES_COL]
            > mes_evento_numeric
        )
    )

    if posterior.any():

        raise ValueError(
            "Sobrevivieron observaciones "
            "posteriores al evento BAJA+2."
        )

    # ------------------------------------------------------------------
    # t=0 fieles
    # ------------------------------------------------------------------

    ultimos_fieles = out[
        (
            out[GRUPO_COL]
            == FIEL
        )
        &
        (
            out["t"]
            == 0
        )
    ]

    if len(
        ultimos_fieles
    ) != 4067:

        raise ValueError(
            "No existe exactamente una "
            "observación t=0 por cliente fiel."
        )

    print()
    print(
        f"Clientes BAJA+2 : "
        f"{n_baja:,}"
    )

    print(
        f"Clientes fieles : "
        f"{n_fiel:,}"
    )

    print(
        "t=0 de los BAJA+2 coincide "
        "con BAJA+2: OK"
    )

    print(
        "Observaciones posteriores "
        "al evento eliminadas: OK"
    )

    print(
        "t=0 único para fieles: OK"
    )

    print()
    print(
        "VALIDACIÓN TEMPORAL: OK"
    )

    return out


# ======================================================================
# 4. COMPOSICIÓN DE CLUSTERS
# ======================================================================

def composicion_clusters(
    df: pd.DataFrame,
) -> pd.DataFrame:

    clientes = (
        df[
            [
                ID_COL,
                CLUSTER_COL,
                GRUPO_COL,
            ]
        ]
        .drop_duplicates()
    )

    out = (
        clientes
        .groupby(
            [
                CLUSTER_COL,
                GRUPO_COL,
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
        .reset_index()
    )

    for c in [
        BAJA2,
        FIEL,
    ]:

        if c not in out.columns:
            out[c] = 0

    out = out.rename(
        columns={
            BAJA2:
                "n_baja2",
            FIEL:
                "n_fieles",
        }
    )

    out["n_total"] = (
        out["n_baja2"]
        + out["n_fieles"]
    )

    out["share_baja2"] = (
        out["n_baja2"]
        / out["n_total"]
    )

    return out[
        [
            CLUSTER_COL,
            "n_total",
            "n_baja2",
            "n_fieles",
            "share_baja2",
        ]
    ]


# ======================================================================
# 5. COBERTURA TEMPORAL
# ======================================================================

def cobertura_temporal(
    df: pd.DataFrame,
) -> pd.DataFrame:

    out = (
        df.groupby(
            [
                GRUPO_COL,
                CLUSTER_COL,
                "t",
            ]
        )
        .agg(
            filas=(
                ID_COL,
                "size",
            ),
            clientes=(
                ID_COL,
                "nunique",
            ),
        )
        .reset_index()
        .sort_values(
            [
                GRUPO_COL,
                CLUSTER_COL,
                "t",
            ]
        )
    )

    return out


# ======================================================================
# 6. TRAYECTORIAS
# ======================================================================

def calcular_trayectorias(
    df: pd.DataFrame,
    variables: list[str],
    grupo: str,
) -> pd.DataFrame:
    """
    Resume cada variable por:

        grupo × cluster × t

    Si sólo existe una observación válida para una combinación,
    la desviación estándar muestral no está definida y se guarda
    como NaN.
    """

    d = df[
        df[GRUPO_COL]
        == grupo
    ]

    registros = []

    for (
        cluster,
        t,
    ), g in d.groupby(
        [
            CLUSTER_COL,
            "t",
        ]
    ):

        for variable in variables:

            x = (
                safe_numeric(
                    g[variable]
                )
                .dropna()
            )

            if x.empty:
                continue

            media = x.mean()
            mediana = x.median()
            std = x.std()
            q25 = x.quantile(
                0.25
            )
            q75 = x.quantile(
                0.75
            )

            registros.append(
                {
                    GRUPO_COL:
                        grupo,

                    CLUSTER_COL:
                        int(cluster),

                    "t":
                        int(t),

                    "variable":
                        variable,

                    "n":
                        int(len(x)),

                    "media":
                        safe_float(
                            media
                        ),

                    "mediana":
                        safe_float(
                            mediana
                        ),

                    "std":
                        safe_float(
                            std
                        ),

                    "q25":
                        safe_float(
                            q25
                        ),

                    "q75":
                        safe_float(
                            q75
                        ),
                }
            )

    return pd.DataFrame(
        registros
    )


# ======================================================================
# 7. DELTAS INDIVIDUALES
# ======================================================================

def calcular_deltas(
    df: pd.DataFrame,
    variables: list[str],
) -> pd.DataFrame:
    """
    Calcula cambios individuales respecto de t=0.

    delta1 = x(t0) - x(t-1)
    delta2 = x(t0) - x(t-2)
    delta3 = x(t0) - x(t-3)

    Si el cliente no posee suficiente historia, ese delta simplemente
    no se calcula.
    """

    registros = []

    columnas = (
        [
            ID_COL,
            CLUSTER_COL,
            GRUPO_COL,
            "t",
        ]
        + variables
    )

    d = df[
        columnas
    ].copy()

    for cliente, g in d.groupby(
        ID_COL,
        sort=False,
    ):

        g = g.set_index(
            "t"
        )

        if 0 not in g.index:
            continue

        actual = g.loc[
            0
        ]

        if isinstance(
            actual,
            pd.DataFrame,
        ):

            actual = (
                actual.iloc[
                    -1
                ]
            )

        cluster = int(
            actual[
                CLUSTER_COL
            ]
        )

        grupo = str(
            actual[
                GRUPO_COL
            ]
        )

        for lag in LAGS_DELTA:

            t_anterior = -lag

            if (
                t_anterior
                not in g.index
            ):
                continue

            anterior = g.loc[
                t_anterior
            ]

            if isinstance(
                anterior,
                pd.DataFrame,
            ):

                anterior = (
                    anterior.iloc[
                        -1
                    ]
                )

            for variable in variables:

                x0 = pd.to_numeric(
                    pd.Series(
                        [
                            actual[
                                variable
                            ]
                        ]
                    ),
                    errors="coerce",
                ).iloc[0]

                xl = pd.to_numeric(
                    pd.Series(
                        [
                            anterior[
                                variable
                            ]
                        ]
                    ),
                    errors="coerce",
                ).iloc[0]

                if (
                    pd.isna(x0)
                    or pd.isna(xl)
                ):
                    continue

                registros.append(
                    {
                        ID_COL:
                            int(cliente),

                        CLUSTER_COL:
                            cluster,

                        GRUPO_COL:
                            grupo,

                        "variable":
                            variable,

                        "lag":
                            lag,

                        "valor_t0":
                            float(x0),

                        "valor_tlag":
                            float(xl),

                        "delta":
                            float(
                                x0 - xl
                            ),
                    }
                )

    return pd.DataFrame(
        registros
    )


# ======================================================================
# 8. RESUMEN DE DELTAS
# ======================================================================

def resumir_deltas(
    detalle: pd.DataFrame,
) -> pd.DataFrame:

    if detalle.empty:
        return pd.DataFrame()

    out = (
        detalle
        .groupby(
            [
                GRUPO_COL,
                CLUSTER_COL,
                "variable",
                "lag",
            ],
            as_index=False,
        )
        .agg(
            n=(
                "delta",
                "size",
            ),
            media=(
                "delta",
                "mean",
            ),
            mediana=(
                "delta",
                "median",
            ),
            std=(
                "delta",
                "std",
            ),
            q25=(
                "delta",
                lambda x:
                    x.quantile(
                        0.25
                    ),
            ),
            q75=(
                "delta",
                lambda x:
                    x.quantile(
                        0.75
                    ),
            ),
        )
    )

    return out


# ======================================================================
# 9. COMPARACIÓN DELTAS BAJA+2 vs FIELES
# ======================================================================

def comparar_deltas(
    resumen: pd.DataFrame,
) -> pd.DataFrame:

    if resumen.empty:
        return pd.DataFrame()

    baja = (
        resumen[
            resumen[
                GRUPO_COL
            ] == BAJA2
        ]
        .drop(
            columns=[
                GRUPO_COL
            ]
        )
        .rename(
            columns={
                "n":
                    "n_baja2",
                "media":
                    "media_baja2",
                "mediana":
                    "mediana_baja2",
                "std":
                    "std_baja2",
                "q25":
                    "q25_baja2",
                "q75":
                    "q75_baja2",
            }
        )
    )

    fiel = (
        resumen[
            resumen[
                GRUPO_COL
            ] == FIEL
        ]
        .drop(
            columns=[
                GRUPO_COL
            ]
        )
        .rename(
            columns={
                "n":
                    "n_fiel",
                "media":
                    "media_fiel",
                "mediana":
                    "mediana_fiel",
                "std":
                    "std_fiel",
                "q25":
                    "q25_fiel",
                "q75":
                    "q75_fiel",
            }
        )
    )

    out = baja.merge(
        fiel,
        on=[
            CLUSTER_COL,
            "variable",
            "lag",
        ],
        how="inner",
    )

    out[
        "diferencia_delta"
    ] = (
        out[
            "media_baja2"
        ]
        - out[
            "media_fiel"
        ]
    )

    out[
        "smd_delta"
    ] = out.apply(
        lambda r:
            pooled_smd(
                r[
                    "media_baja2"
                ],
                r[
                    "std_baja2"
                ],
                r[
                    "media_fiel"
                ],
                r[
                    "std_fiel"
                ],
            ),
        axis=1,
    )

    out[
        "abs_smd_delta"
    ] = (
        out[
            "smd_delta"
        ]
        .abs()
    )

    out = out.sort_values(
        [
            "lag",
            "abs_smd_delta",
        ],
        ascending=[
            True,
            False,
        ],
    )

    return out


# ======================================================================
# 10. PENDIENTES INDIVIDUALES
# ======================================================================

def pendiente(
    t: np.ndarray,
    x: np.ndarray,
) -> float:
    """
    Pendiente lineal de x respecto de t.

    Exige al menos tres pares válidos.
    """

    t = np.asarray(
        t,
        dtype=float,
    )

    x = np.asarray(
        x,
        dtype=float,
    )

    mask = (
        np.isfinite(t)
        & np.isfinite(x)
    )

    t = t[
        mask
    ]

    x = x[
        mask
    ]

    if len(x) < 3:
        return np.nan

    if np.std(t) <= 1e-12:
        return np.nan

    try:

        b = np.polyfit(
            t,
            x,
            deg=1,
        )[0]

    except Exception:
        return np.nan

    return float(
        b
    )


def calcular_pendientes(
    df: pd.DataFrame,
    variables: list[str],
) -> pd.DataFrame:

    registros = []

    for cliente, g in df.groupby(
        ID_COL,
        sort=False,
    ):

        cluster = int(
            g[
                CLUSTER_COL
            ].iloc[0]
        )

        grupo = str(
            g[
                GRUPO_COL
            ].iloc[0]
        )

        t = (
            g["t"]
            .to_numpy(
                dtype=float
            )
        )

        for variable in variables:

            x = (
                safe_numeric(
                    g[variable]
                )
                .to_numpy(
                    dtype=float,
                    na_value=np.nan,
                )
            )

            b = pendiente(
                t,
                x,
            )

            if pd.isna(
                b
            ):
                continue

            registros.append(
                {
                    ID_COL:
                        int(cliente),

                    CLUSTER_COL:
                        cluster,

                    GRUPO_COL:
                        grupo,

                    "variable":
                        variable,

                    "pendiente":
                        float(b),

                    "n_fotos":
                        int(
                            len(g)
                        ),
                }
            )

    return pd.DataFrame(
        registros
    )


# ======================================================================
# 11. RESUMEN DE PENDIENTES
# ======================================================================

def resumir_pendientes(
    detalle: pd.DataFrame,
) -> pd.DataFrame:

    if detalle.empty:
        return pd.DataFrame()

    out = (
        detalle
        .groupby(
            [
                GRUPO_COL,
                CLUSTER_COL,
                "variable",
            ],
            as_index=False,
        )
        .agg(
            n=(
                "pendiente",
                "size",
            ),
            media=(
                "pendiente",
                "mean",
            ),
            mediana=(
                "pendiente",
                "median",
            ),
            std=(
                "pendiente",
                "std",
            ),
            q25=(
                "pendiente",
                lambda x:
                    x.quantile(
                        0.25
                    ),
            ),
            q75=(
                "pendiente",
                lambda x:
                    x.quantile(
                        0.75
                    ),
            ),
        )
    )

    return out


# ======================================================================
# 12. COMPARACIÓN DE PENDIENTES
# ======================================================================

def comparar_pendientes(
    resumen: pd.DataFrame,
) -> pd.DataFrame:

    if resumen.empty:
        return pd.DataFrame()

    baja = (
        resumen[
            resumen[
                GRUPO_COL
            ] == BAJA2
        ]
        .drop(
            columns=[
                GRUPO_COL
            ]
        )
        .rename(
            columns={
                "n":
                    "n_baja2",
                "media":
                    "media_baja2",
                "mediana":
                    "mediana_baja2",
                "std":
                    "std_baja2",
                "q25":
                    "q25_baja2",
                "q75":
                    "q75_baja2",
            }
        )
    )

    fiel = (
        resumen[
            resumen[
                GRUPO_COL
            ] == FIEL
        ]
        .drop(
            columns=[
                GRUPO_COL
            ]
        )
        .rename(
            columns={
                "n":
                    "n_fiel",
                "media":
                    "media_fiel",
                "mediana":
                    "mediana_fiel",
                "std":
                    "std_fiel",
                "q25":
                    "q25_fiel",
                "q75":
                    "q75_fiel",
            }
        )
    )

    out = baja.merge(
        fiel,
        on=[
            CLUSTER_COL,
            "variable",
        ],
        how="inner",
    )

    out[
        "diferencia_pendiente"
    ] = (
        out[
            "media_baja2"
        ]
        - out[
            "media_fiel"
        ]
    )

    out[
        "smd_pendiente"
    ] = out.apply(
        lambda r:
            pooled_smd(
                r[
                    "media_baja2"
                ],
                r[
                    "std_baja2"
                ],
                r[
                    "media_fiel"
                ],
                r[
                    "std_fiel"
                ],
            ),
        axis=1,
    )

    out[
        "abs_smd_pendiente"
    ] = (
        out[
            "smd_pendiente"
        ]
        .abs()
    )

    out = out.sort_values(
        "abs_smd_pendiente",
        ascending=False,
    )

    return out


# ======================================================================
# 13. RANKING EXPLORATORIO
# ======================================================================

def ranking_candidatas(
    comparacion_deltas: pd.DataFrame,
    comparacion_pendientes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ranking puramente exploratorio.

    Para cada cluster-variable se reúnen:

        SMD delta1
        SMD delta2
        SMD delta3
        SMD pendiente

    score_exploratorio =
        máximo valor absoluto de esas métricas.

    Este score NO es importancia predictiva.
    """

    base = None

    if not comparacion_deltas.empty:

        for lag in LAGS_DELTA:

            aux = comparacion_deltas[
                comparacion_deltas[
                    "lag"
                ] == lag
            ][
                [
                    CLUSTER_COL,
                    "variable",
                    "smd_delta",
                    "abs_smd_delta",
                    "n_baja2",
                    "n_fiel",
                ]
            ].copy()

            aux = aux.rename(
                columns={
                    "smd_delta":
                        f"smd_delta{lag}",

                    "abs_smd_delta":
                        f"abs_smd_delta{lag}",

                    "n_baja2":
                        f"n_baja2_delta{lag}",

                    "n_fiel":
                        f"n_fiel_delta{lag}",
                }
            )

            if base is None:

                base = aux

            else:

                base = base.merge(
                    aux,
                    on=[
                        CLUSTER_COL,
                        "variable",
                    ],
                    how="outer",
                )

    if base is None:

        base = pd.DataFrame(
            columns=[
                CLUSTER_COL,
                "variable",
            ]
        )

    if not comparacion_pendientes.empty:

        p = comparacion_pendientes[
            [
                CLUSTER_COL,
                "variable",
                "smd_pendiente",
                "abs_smd_pendiente",
                "n_baja2",
                "n_fiel",
            ]
        ].copy()

        p = p.rename(
            columns={
                "n_baja2":
                    "n_baja2_pendiente",

                "n_fiel":
                    "n_fiel_pendiente",
            }
        )

        base = base.merge(
            p,
            on=[
                CLUSTER_COL,
                "variable",
            ],
            how="outer",
        )

    metricas_abs = [
        c
        for c in base.columns
        if c.startswith(
            "abs_smd_"
        )
    ]

    if metricas_abs:

        base[
            "score_exploratorio"
        ] = (
            base[
                metricas_abs
            ]
            .max(
                axis=1,
                skipna=True,
            )
        )

    else:

        base[
            "score_exploratorio"
        ] = np.nan

    base = base.sort_values(
        [
            CLUSTER_COL,
            "score_exploratorio",
        ],
        ascending=[
            True,
            False,
        ],
    )

    return base


# ======================================================================
# 14. MOSTRAR DELTAS
# ======================================================================

def imprimir_top_deltas(
    comparacion: pd.DataFrame,
    top_n: int = 10,
) -> None:

    if comparacion.empty:
        return

    for lag in LAGS_DELTA:

        encabezado(
            f"TOP DELTA{lag}: "
            "BAJA+2 vs FIELES"
        )

        d = comparacion[
            comparacion[
                "lag"
            ] == lag
        ]

        for cluster in sorted(
            d[
                CLUSTER_COL
            ].unique()
        ):

            print()
            print(
                f"--- cluster_{cluster} ---"
            )

            aux = (
                d[
                    d[
                        CLUSTER_COL
                    ] == cluster
                ]
                .sort_values(
                    "abs_smd_delta",
                    ascending=False,
                )
                .head(
                    top_n
                )
            )

            columnas = [
                "variable",
                "n_baja2",
                "n_fiel",
                "media_baja2",
                "media_fiel",
                "diferencia_delta",
                "smd_delta",
            ]

            print(
                aux[
                    columnas
                ].to_string(
                    index=False,
                    float_format=
                        lambda x:
                            f"{x:,.3f}",
                )
            )


# ======================================================================
# 15. MOSTRAR PENDIENTES
# ======================================================================

def imprimir_top_pendientes(
    comparacion: pd.DataFrame,
    top_n: int = 10,
) -> None:

    if comparacion.empty:
        return

    encabezado(
        "TOP PENDIENTES: "
        "BAJA+2 vs FIELES"
    )

    for cluster in sorted(
        comparacion[
            CLUSTER_COL
        ].unique()
    ):

        print()
        print(
            f"--- cluster_{cluster} ---"
        )

        aux = (
            comparacion[
                comparacion[
                    CLUSTER_COL
                ] == cluster
            ]
            .sort_values(
                "abs_smd_pendiente",
                ascending=False,
            )
            .head(
                top_n
            )
        )

        columnas = [
            "variable",
            "n_baja2",
            "n_fiel",
            "media_baja2",
            "media_fiel",
            "diferencia_pendiente",
            "smd_pendiente",
        ]

        print(
            aux[
                columnas
            ].to_string(
                index=False,
                float_format=
                    lambda x:
                        f"{x:,.3f}",
            )
        )


# ======================================================================
# 16. MAIN
# ======================================================================

def main() -> None:

    encabezado(
        "Z505 V2 — TRAYECTORIAS PRE-BAJA+2"
    )

    print(
        """
Este análisis distingue explícitamente:

  BAJA+2:
      t=0 = observación etiquetada BAJA+2.
      Toda observación posterior se descarta.

  FIEL:
      t=0 = fin de la ventana disponible.
      NO representa un evento.

Los clusters k=5 vienen de z504 y NO se recalculan.

El eje t representa observaciones relativas:
t=-1 es la observación inmediatamente anterior,
aunque exista un salto de meses calendario.
"""
    )

    # ==================================================================
    # ASIGNACIONES
    # ==================================================================

    asignaciones = (
        cargar_asignaciones()
    )

    print(
        f"Clientes asignados: "
        f"{len(asignaciones):,}"
    )

    print()
    print(
        "Clusters k=5:"
    )

    print(
        asignaciones[
            CLUSTER_COL
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # ==================================================================
    # DATASET
    # ==================================================================

    df, variables = (
        cargar_datos(
            asignaciones
        )
    )

    # ==================================================================
    # EJE TEMPORAL
    # ==================================================================

    df = preparar_tiempo(
        df
    )

    # ==================================================================
    # COMPOSICIÓN
    # ==================================================================

    encabezado(
        "COMPOSICIÓN DE CLUSTERS"
    )

    composicion = (
        composicion_clusters(
            df
        )
    )

    mostrar = (
        composicion.copy()
    )

    mostrar[
        "share_baja2"
    ] = (
        mostrar[
            "share_baja2"
        ]
        * 100.0
    )

    print(
        mostrar.to_string(
            index=False,
            formatters={
                "share_baja2":
                    lambda x:
                        f"{x:.2f}%"
            },
        )
    )

    print()
    print(
        "NOTA: share_baja2 corresponde "
        "a la muestra balanceada."
    )

    print(
        "NO representa probabilidad "
        "poblacional de baja."
    )

    # ==================================================================
    # COBERTURA TEMPORAL
    # ==================================================================

    cobertura = (
        cobertura_temporal(
            df
        )
    )

    encabezado(
        "COBERTURA TEMPORAL — BAJA+2"
    )

    print(
        cobertura[
            cobertura[
                GRUPO_COL
            ] == BAJA2
        ].to_string(
            index=False
        )
    )

    encabezado(
        "COBERTURA TEMPORAL — FIELES"
    )

    print(
        cobertura[
            cobertura[
                GRUPO_COL
            ] == FIEL
        ].to_string(
            index=False
        )
    )

    # ==================================================================
    # TRAYECTORIAS
    # ==================================================================

    encabezado(
        "TRAYECTORIAS BAJA+2"
    )

    tray_baja = (
        calcular_trayectorias(
            df,
            variables,
            BAJA2,
        )
    )

    print(
        f"Registros: "
        f"{len(tray_baja):,}"
    )

    encabezado(
        "TRAYECTORIAS FIELES"
    )

    tray_fiel = (
        calcular_trayectorias(
            df,
            variables,
            FIEL,
        )
    )

    print(
        f"Registros: "
        f"{len(tray_fiel):,}"
    )

    # ==================================================================
    # DELTAS
    # ==================================================================

    encabezado(
        "DELTAS INDIVIDUALES"
    )

    deltas_detalle = (
        calcular_deltas(
            df,
            variables,
        )
    )

    print(
        f"Registros individuales: "
        f"{len(deltas_detalle):,}"
    )

    deltas_resumen = (
        resumir_deltas(
            deltas_detalle
        )
    )

    comparacion_d = (
        comparar_deltas(
            deltas_resumen
        )
    )

    print(
        f"Comparaciones BAJA+2/fiel: "
        f"{len(comparacion_d):,}"
    )

    # ==================================================================
    # PENDIENTES
    # ==================================================================

    encabezado(
        "PENDIENTES INDIVIDUALES"
    )

    pendientes_detalle = (
        calcular_pendientes(
            df,
            variables,
        )
    )

    print(
        f"Registros individuales: "
        f"{len(pendientes_detalle):,}"
    )

    pendientes_resumen = (
        resumir_pendientes(
            pendientes_detalle
        )
    )

    comparacion_p = (
        comparar_pendientes(
            pendientes_resumen
        )
    )

    print(
        f"Comparaciones BAJA+2/fiel: "
        f"{len(comparacion_p):,}"
    )

    # ==================================================================
    # RANKING EXPLORATORIO
    # ==================================================================

    ranking = (
        ranking_candidatas(
            comparacion_d,
            comparacion_p,
        )
    )

    # ==================================================================
    # MOSTRAR RESULTADOS
    # ==================================================================

    imprimir_top_deltas(
        comparacion_d,
        top_n=10,
    )

    imprimir_top_pendientes(
        comparacion_p,
        top_n=10,
    )

    encabezado(
        "TOP CANDIDATAS POR CLUSTER"
    )

    clusters_ranking = (
        ranking[
            CLUSTER_COL
        ]
        .dropna()
        .unique()
    )

    for cluster in sorted(
        clusters_ranking
    ):

        print()
        print(
            f"--- cluster_{int(cluster)} ---"
        )

        aux = (
            ranking[
                ranking[
                    CLUSTER_COL
                ] == cluster
            ]
            .head(
                15
            )
        )

        columnas = [
            "variable",
            "score_exploratorio",
            "smd_delta1",
            "smd_delta2",
            "smd_delta3",
            "smd_pendiente",
        ]

        columnas = [
            c
            for c in columnas
            if c in aux.columns
        ]

        print(
            aux[
                columnas
            ].to_string(
                index=False,
                float_format=
                    lambda x:
                        f"{x:,.3f}",
            )
        )

    # ==================================================================
    # GUARDAR RESULTADOS
    # ==================================================================

    encabezado(
        "GUARDANDO RESULTADOS"
    )

    archivos = {

        "cobertura_temporal.csv":
            cobertura,

        "composicion_clusters.csv":
            composicion,

        "trayectorias_baja2.csv":
            tray_baja,

        "trayectorias_fieles.csv":
            tray_fiel,

        "deltas_individuales.csv":
            deltas_detalle,

        "deltas_resumen.csv":
            deltas_resumen,

        "comparacion_deltas.csv":
            comparacion_d,

        "pendientes_individuales.csv":
            pendientes_detalle,

        "pendientes_resumen.csv":
            pendientes_resumen,

        "comparacion_pendientes.csv":
            comparacion_p,

        "ranking_candidatas.csv":
            ranking,
    }

    for nombre, tabla in archivos.items():

        path = (
            DIR_SALIDA
            / nombre
        )

        tabla.to_csv(
            path,
            index=False,
        )

        print(
            path
        )

    # ==================================================================
    # INTERPRETACIÓN
    # ==================================================================

    encabezado(
        "INTERPRETACIÓN METODOLÓGICA"
    )

    print(
        """
1. BAJA+2

   t=0 es exactamente la observación etiquetada BAJA+2.

2. INFORMACIÓN POSTERIOR

   Toda observación posterior al evento BAJA+2 fue eliminada
   antes de calcular trayectorias, deltas y pendientes.

3. FIELES

   t=0 es solamente el final de la ventana disponible.

4. EJE TEMPORAL

   t representa posición relativa entre observaciones.

   Por lo tanto:

       t=-1

   significa "observación inmediatamente anterior", pero no
   necesariamente "un mes calendario antes".

5. DELTA1

   x(t=0) - x(t=-1)

6. DELTA2

   x(t=0) - x(t=-2)

7. DELTA3

   x(t=0) - x(t=-3)

8. COBERTURA

   Los clientes BAJA+2 poseen distinta cantidad de historia.

   Por eso delta1, delta2, delta3 y pendiente no tienen
   necesariamente la misma cantidad de observaciones.

9. PENDIENTE

   Resume la tendencia lineal individual utilizando las
   observaciones disponibles hasta el ancla correspondiente.

   Se requieren al menos tres observaciones válidas.

10. SMD

    Standardized Mean Difference.

    Se utiliza solamente como medida descriptiva de separación
    entre las distribuciones BAJA+2 y FIEL.

11. SCORE EXPLORATORIO

    Es el máximo |SMD| observado entre:

        delta1
        delta2
        delta3
        pendiente

    Su objetivo es priorizar variables para inspección.

12. INTERPRETACIÓN

    El ranking NO representa:

        importancia predictiva,
        causalidad,
        ganancia de competencia,
        selección automática de variables.

13. LEAKAGE

    Este z505 es EDA explicativa.

    En BAJA+2 estamos observando el propio evento en t=0.

    Por lo tanto, una señal encontrada aquí NO puede trasladarse
    automáticamente a un modelo predictivo.

14. PRÓXIMO PASO

    z506 deberá traducir las señales interesantes a features
    construidas exclusivamente con información que realmente
    esté disponible en el momento de predicción definido por
    la competencia.

    Sólo después se evaluará si esas nuevas features mejoran
    la métrica económica fuera de muestra.
"""
    )


if __name__ == "__main__":
    main()