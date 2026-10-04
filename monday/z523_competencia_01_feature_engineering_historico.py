#!/usr/bin/env python3
"""
z523_competencia_01_feature_engineering_historico.py

Construcción y auditoría de Feature Engineering Histórico para
Competencia 01.

NO entrena modelos.
NO genera submits.

Objetivo
--------
Construir, para cada predictor original x:

    x
    x_lag1
    x_delta_lag1 = x - x_lag1

El lag se obtiene exclusivamente del MES CALENDARIO ANTERIOR
del mismo numero_de_cliente.

Ejemplo:
    fila 202106 -> lag1 exclusivamente desde 202105

No se utiliza simplemente "la observación anterior" porque un cliente
puede no aparecer en un mes y eso generaría un falso lag consecutivo.

Meses utilizados
-----------------
202103-202108

Esto permite construir:

    202104 <- lag 202103
    202105 <- lag 202104
    202106 <- lag 202105
    202107 <- lag 202106
    202108 <- lag 202107

202103 no tiene lag porque 202102 no está disponible.

Target
------
Se conserva clase_ternaria tal como viene en los datos.
No se construye ninguna feature usando el target.

Salida
------
Dataset histórico + auditorías de:
    - cobertura de lag
    - cantidad de NA
    - consistencia temporal
    - duplicados cliente/mes
    - cantidad de columnas generadas
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


# ============================================================
# Configuración
# ============================================================

DATASET = Path(
    "/data/dmeyf/datasets/competencia_01.csv"
)

OUTPUT_DIR = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/"
    "competencia_01/feature_engineering_z523"
)

OUTPUT_DATASET = (
    OUTPUT_DIR
    / "competencia_01_historico_lag1.parquet"
)

ID_COL = "numero_de_cliente"
MONTH_COL = "foto_mes"
TARGET_COL = "clase_ternaria"

MESES = [
    202103,
    202104,
    202105,
    202106,
    202107,
    202108,
]


# ============================================================
# Funciones de fecha
# ============================================================

def mes_anterior(
    foto_mes: int,
) -> int:
    """
    Devuelve YYYYMM del mes calendario anterior.
    """

    year = (
        foto_mes // 100
    )

    month = (
        foto_mes % 100
    )

    month -= 1

    if month == 0:
        year -= 1
        month = 12

    return (
        year * 100
        + month
    )


# ============================================================
# Lectura de esquema
# ============================================================

def obtener_esquema():

    con = duckdb.connect()

    descripcion = con.execute(
        f"""
        DESCRIBE
        SELECT *
        FROM read_csv_auto(
            '{DATASET}',
            header=true
        )
        """
    ).fetchdf()

    con.close()

    columnas = (
        descripcion[
            "column_name"
        ]
        .tolist()
    )

    tipos = dict(
        zip(
            descripcion[
                "column_name"
            ],
            descripcion[
                "column_type"
            ],
        )
    )

    excluir = {
        ID_COL,
        MONTH_COL,
        TARGET_COL,
    }

    features = [
        col
        for col in columnas
        if col not in excluir
    ]

    return (
        columnas,
        tipos,
        features,
    )


# ============================================================
# Lectura de datos
# ============================================================

def cargar_dataset(
    columnas: list[str],
) -> pd.DataFrame:

    con = duckdb.connect()

    cols_sql = ", ".join(
        f'"{c}"'
        for c in columnas
    )

    meses_sql = ", ".join(
        str(m)
        for m in MESES
    )

    print(
        "\nLeyendo dataset...",
        flush=True,
    )

    df = con.execute(
        f"""
        SELECT
            {cols_sql}
        FROM read_csv_auto(
            '{DATASET}',
            header=true
        )
        WHERE "{MONTH_COL}"
              IN ({meses_sql})
        ORDER BY
            "{ID_COL}",
            "{MONTH_COL}"
        """
    ).fetchdf()

    con.close()

    return df


# ============================================================
# Auditoría inicial
# ============================================================

def auditoria_inicial(
    df: pd.DataFrame,
    features: list[str],
):

    print(
        "\n"
        + "=" * 80,
        flush=True,
    )

    print(
        "AUDITORÍA INICIAL",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    print(
        f"\nFilas: "
        f"{len(df):,}",
        flush=True,
    )

    print(
        f"Columnas originales: "
        f"{df.shape[1]}",
        flush=True,
    )

    print(
        f"Predictores originales: "
        f"{len(features)}",
        flush=True,
    )

    print(
        "\nFilas por mes:",
        flush=True,
    )

    print(
        df[
            MONTH_COL
        ]
        .value_counts()
        .sort_index()
        .to_string(),
        flush=True,
    )

    duplicados = int(
        df.duplicated(
            subset=[
                ID_COL,
                MONTH_COL,
            ]
        ).sum()
    )

    print(
        f"\nDuplicados "
        f"cliente/mes: "
        f"{duplicados:,}",
        flush=True,
    )

    if duplicados != 0:
        raise ValueError(
            "Hay duplicados cliente/mes. "
            "No es seguro construir lags."
        )

    meses_encontrados = sorted(
        df[
            MONTH_COL
        ]
        .unique()
        .tolist()
    )

    if meses_encontrados != MESES:
        raise ValueError(
            f"Meses encontrados: "
            f"{meses_encontrados}; "
            f"esperados: {MESES}"
        )


# ============================================================
# Construcción de lag1
# ============================================================

def construir_historico(
    df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:

    print(
        "\n"
        + "=" * 80,
        flush=True,
    )

    print(
        "CONSTRUCCIÓN LAG1 + DELTA_LAG1",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    # --------------------------------------------------------
    # Tabla actual
    # --------------------------------------------------------

    actual = (
        df.copy()
    )

    # Mes que necesitamos encontrar
    # en la tabla histórica.
    mapa_mes_anterior = {
        mes: mes_anterior(
            mes
        )
        for mes in MESES
    }

    actual[
        "_mes_lag1"
    ] = (
        actual[
            MONTH_COL
        ]
        .map(
            mapa_mes_anterior
        )
        .astype(
            np.int32
        )
    )

    # --------------------------------------------------------
    # Tabla de valores del pasado
    # --------------------------------------------------------

    pasado = (
        df[
            [
                ID_COL,
                MONTH_COL,
            ]
            + features
        ]
        .copy()
    )

    rename_dict = {
        MONTH_COL:
            "_mes_lag1"
    }

    rename_dict.update(
        {
            feature:
                f"{feature}_lag1"
            for feature in features
        }
    )

    pasado.rename(
        columns=rename_dict,
        inplace=True,
    )

    # Indicador explícito de que existe
    # una observación en el mes anterior.
    pasado[
        "lag1_disponible"
    ] = np.int8(
        1
    )

    print(
        "\nHaciendo merge "
        "cliente + mes anterior...",
        flush=True,
    )

    historico = actual.merge(
        pasado,
        how="left",
        on=[
            ID_COL,
            "_mes_lag1",
        ],
        validate="many_to_one",
    )

    if len(
        historico
    ) != len(
        actual
    ):
        raise ValueError(
            "El merge cambió la "
            "cantidad de filas."
        )

    historico[
        "lag1_disponible"
    ] = (
        historico[
            "lag1_disponible"
        ]
        .fillna(
            0
        )
        .astype(
            np.int8
        )
    )

    # --------------------------------------------------------
    # Delta lag1
    # --------------------------------------------------------

    print(
        "Construyendo deltas...",
        flush=True,
    )

    delta_data = {}

    for i, feature in enumerate(
        features,
        start=1,
    ):

        lag_col = (
            f"{feature}_lag1"
        )

        delta_col = (
            f"{feature}_delta_lag1"
        )

        # Conversión numérica defensiva.
        actual_values = (
            pd.to_numeric(
                historico[
                    feature
                ],
                errors="coerce",
            )
        )

        lag_values = (
            pd.to_numeric(
                historico[
                    lag_col
                ],
                errors="coerce",
            )
        )

        delta_data[
            delta_col
        ] = (
            actual_values
            - lag_values
        )

        if (
            i % 25 == 0
            or i == len(
                features
            )
        ):
            print(
                f"  deltas: "
                f"{i}/{len(features)}",
                flush=True,
            )

    # Crear el bloque completo de una vez evita
    # fragmentar excesivamente el DataFrame.
    delta_df = pd.DataFrame(
        delta_data,
        index=historico.index,
    )

    historico = pd.concat(
        [
            historico,
            delta_df,
        ],
        axis=1,
    )

    historico.drop(
        columns=[
            "_mes_lag1"
        ],
        inplace=True,
    )

    return historico


# ============================================================
# Auditoría temporal
# ============================================================

def auditar_historico(
    historico: pd.DataFrame,
    features: list[str],
):

    print(
        "\n"
        + "=" * 80,
        flush=True,
    )

    print(
        "AUDITORÍA DEL DATASET HISTÓRICO",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    expected_cols = (
        3
        + len(features)
        + len(features)
        + len(features)
        + 1
    )

    print(
        f"\nFilas finales    : "
        f"{len(historico):,}",
        flush=True,
    )

    print(
        f"Columnas finales : "
        f"{historico.shape[1]:,}",
        flush=True,
    )

    print(
        f"Columnas esperadas: "
        f"{expected_cols:,}",
        flush=True,
    )

    if (
        historico.shape[1]
        != expected_cols
    ):
        raise ValueError(
            "Cantidad inesperada "
            "de columnas."
        )

    # --------------------------------------------------------
    # Cobertura lag por mes
    # --------------------------------------------------------

    cobertura = (
        historico
        .groupby(
            MONTH_COL,
            as_index=False,
        )
        .agg(
            filas=(
                ID_COL,
                "size",
            ),
            con_lag1=(
                "lag1_disponible",
                "sum",
            ),
        )
    )

    cobertura[
        "sin_lag1"
    ] = (
        cobertura[
            "filas"
        ]
        - cobertura[
            "con_lag1"
        ]
    )

    cobertura[
        "cobertura_pct"
    ] = (
        100.0
        * cobertura[
            "con_lag1"
        ]
        / cobertura[
            "filas"
        ]
    )

    print(
        "\nCobertura de lag1:",
        flush=True,
    )

    print(
        cobertura.to_string(
            index=False,
            formatters={
                "cobertura_pct":
                    lambda x:
                        f"{x:.4f}%"
            },
        ),
        flush=True,
    )

    # Marzo no debe tener lag porque
    # febrero no está en el dataset.
    marzo = historico.loc[
        historico[
            MONTH_COL
        ].eq(
            202103
        )
    ]

    if int(
        marzo[
            "lag1_disponible"
        ].sum()
    ) != 0:
        raise ValueError(
            "202103 tiene lags cuando "
            "no debería tenerlos."
        )

    # --------------------------------------------------------
    # Validación explícita de continuidad
    # --------------------------------------------------------

    # Para cada fila marcada con lag verificamos que
    # efectivamente exista (cliente, mes anterior)
    # en el dataset original reconstruyendo las claves.
    keys_actuales = set(
        zip(
            historico[
                ID_COL
            ].astype(
                np.int64
            ),
            historico[
                MONTH_COL
            ].astype(
                np.int32
            ),
        )
    )

    errores_temporales = 0

    con_lag = historico.loc[
        historico[
            "lag1_disponible"
        ].eq(
            1
        ),
        [
            ID_COL,
            MONTH_COL,
        ],
    ]

    for row in con_lag.itertuples(
        index=False
    ):

        cliente = int(
            getattr(
                row,
                ID_COL
            )
        )

        mes = int(
            getattr(
                row,
                MONTH_COL
            )
        )

        esperado = (
            cliente,
            mes_anterior(
                mes
            ),
        )

        if esperado not in (
            keys_actuales
        ):
            errores_temporales += 1

    print(
        f"\nErrores de continuidad "
        f"temporal: "
        f"{errores_temporales:,}",
        flush=True,
    )

    if errores_temporales != 0:
        raise ValueError(
            "Se detectaron lags que "
            "no corresponden al mes "
            "calendario anterior."
        )

    # --------------------------------------------------------
    # NA de features
    # --------------------------------------------------------

    lag_cols = [
        f"{x}_lag1"
        for x in features
    ]

    delta_cols = [
        f"{x}_delta_lag1"
        for x in features
    ]

    auditoria_na = []

    for mes in MESES:

        parte = historico.loc[
            historico[
                MONTH_COL
            ].eq(
                mes
            )
        ]

        n = len(
            parte
        )

        na_original = int(
            parte[
                features
            ]
            .isna()
            .sum()
            .sum()
        )

        na_lag = int(
            parte[
                lag_cols
            ]
            .isna()
            .sum()
            .sum()
        )

        na_delta = int(
            parte[
                delta_cols
            ]
            .isna()
            .sum()
            .sum()
        )

        auditoria_na.append(
            {
                "foto_mes":
                    mes,
                "filas":
                    n,
                "na_original":
                    na_original,
                "na_lag1":
                    na_lag,
                "na_delta_lag1":
                    na_delta,
            }
        )

    auditoria_na_df = pd.DataFrame(
        auditoria_na
    )

    print(
        "\nNA por bloque de variables:",
        flush=True,
    )

    print(
        auditoria_na_df.to_string(
            index=False
        ),
        flush=True,
    )

    # --------------------------------------------------------
    # Cobertura individual por feature
    # --------------------------------------------------------

    cobertura_features = []

    # Excluimos 202103 porque por diseño
    # no tiene mes anterior disponible.
    meses_con_historia = (
        historico[
            MONTH_COL
        ].ne(
            202103
        )
    )

    base = historico.loc[
        meses_con_historia
    ]

    for feature in features:

        lag_col = (
            f"{feature}_lag1"
        )

        delta_col = (
            f"{feature}_delta_lag1"
        )

        cobertura_features.append(
            {
                "feature":
                    feature,
                "actual_no_na":
                    int(
                        base[
                            feature
                        ]
                        .notna()
                        .sum()
                    ),
                "lag1_no_na":
                    int(
                        base[
                            lag_col
                        ]
                        .notna()
                        .sum()
                    ),
                "delta_no_na":
                    int(
                        base[
                            delta_col
                        ]
                        .notna()
                        .sum()
                    ),
            }
        )

    cobertura_features_df = (
        pd.DataFrame(
            cobertura_features
        )
    )

    return (
        cobertura,
        auditoria_na_df,
        cobertura_features_df,
    )


# ============================================================
# Main
# ============================================================

def main():

    inicio = time.time()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    print(
        "z523 - FEATURE ENGINEERING HISTÓRICO",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    print(
        f"\nInput:\n  {DATASET}",
        flush=True,
    )

    print(
        f"\nOutput:\n  {OUTPUT_DIR}",
        flush=True,
    )

    columnas, tipos, features = (
        obtener_esquema()
    )

    print(
        f"\nColumnas originales: "
        f"{len(columnas)}",
        flush=True,
    )

    print(
        f"Predictores originales: "
        f"{len(features)}",
        flush=True,
    )

    if len(
        features
    ) != 152:
        raise ValueError(
            f"Se esperaban 152 "
            f"predictores; encontrados "
            f"{len(features)}"
        )

    print(
        "\nMes anterior esperado:",
        flush=True,
    )

    for mes in MESES:
        print(
            f"  {mes} <- "
            f"{mes_anterior(mes)}",
            flush=True,
        )

    df = cargar_dataset(
        columnas=columnas
    )

    auditoria_inicial(
        df=df,
        features=features,
    )

    historico = construir_historico(
        df=df,
        features=features,
    )

    (
        cobertura,
        auditoria_na,
        cobertura_features,
    ) = auditar_historico(
        historico=historico,
        features=features,
    )

    # --------------------------------------------------------
    # Guardado
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 80,
        flush=True,
    )

    print(
        "GUARDANDO RESULTADOS",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    print(
        "\nGuardando dataset "
        "histórico en Parquet...",
        flush=True,
    )

    historico.to_parquet(
        OUTPUT_DATASET,
        index=False,
        compression="snappy",
    )

    cobertura.to_csv(
        OUTPUT_DIR
        / "cobertura_lag1_por_mes.csv",
        index=False,
    )

    auditoria_na.to_csv(
        OUTPUT_DIR
        / "auditoria_na_por_mes.csv",
        index=False,
    )

    cobertura_features.to_csv(
        OUTPUT_DIR
        / "cobertura_por_feature.csv",
        index=False,
    )

    resumen = {
        "input":
            str(
                DATASET
            ),
        "output_dataset":
            str(
                OUTPUT_DATASET
            ),
        "meses":
            MESES,
        "filas":
            int(
                len(
                    historico
                )
            ),
        "columnas_originales":
            int(
                len(
                    columnas
                )
            ),
        "features_originales":
            int(
                len(
                    features
                )
            ),
        "features_lag1":
            int(
                len(
                    features
                )
            ),
        "features_delta_lag1":
            int(
                len(
                    features
                )
            ),
        "columnas_finales":
            int(
                historico.shape[
                    1
                ]
            ),
        "features_modelables":
            int(
                len(
                    features
                )
                * 3
                + 1
            ),
        "leakage_target":
            False,
        "lag_tipo":
            "mes_calendario_anterior",
    }

    with (
        OUTPUT_DIR
        / "resumen_z523.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            resumen,
            f,
            indent=2,
        )

    elapsed = (
        time.time()
        - inicio
    )

    print(
        f"\nDataset:\n"
        f"  {OUTPUT_DATASET}",
        flush=True,
    )

    print(
        f"\nFilas finales    : "
        f"{len(historico):,}",
        flush=True,
    )

    print(
        f"Columnas finales : "
        f"{historico.shape[1]:,}",
        flush=True,
    )

    print(
        f"Features para modelo "
        f"(sin ID/mes/target): "
        f"{len(features) * 3 + 1:,}",
        flush=True,
    )

    print(
        f"\nTiempo total: "
        f"{elapsed:.1f} s",
        flush=True,
    )

    print(
        "\nIMPORTANTE:",
        flush=True,
    )

    print(
        "  z523 solamente construyó "
        "y auditó las variables.",
        flush=True,
    )

    print(
        "  No se entrenó ningún modelo.",
        flush=True,
    )

    print(
        "  No se utilizó el target "
        "para crear features.",
        flush=True,
    )

    print(
        "  No se generó ningún submit.",
        flush=True,
    )


if __name__ == "__main__":
    main()