#!/usr/bin/env python3
"""
z532_competencia_01_feature_engineering_lag2.py

Construcción y auditoría de features históricas lag2.

Objetivo
--------
Extender el dataset histórico de z523 agregando, para cada una de las
152 variables originales:

    variable_lag2 = valor de la misma variable dos meses calendario atrás

Ejemplos:
    202105 <- 202103
    202106 <- 202104
    202107 <- 202105
    202108 <- 202106

IMPORTANTE
----------
- No se utiliza target para construir las features.
- No se entrena ningún modelo.
- No se genera ningún submit.
- 202108 puede tener lag2 porque proviene de 202106.
- 202104 no puede tener lag2 porque 202102 no está disponible.
- El join es por numero_de_cliente y mes calendario exacto.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import duckdb
import pandas as pd


# ============================================================
# Configuración
# ============================================================

INPUT_DATASET = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/"
    "competencia_01/feature_engineering_z523/"
    "competencia_01_historico_lag1.parquet"
)

OUTPUT_DIR = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/"
    "competencia_01/feature_engineering_z532"
)

OUTPUT_PARQUET = (
    OUTPUT_DIR
    / "competencia_01_historico_lag1_lag2.parquet"
)

OUTPUT_AUDIT = (
    OUTPUT_DIR
    / "auditoria_lag2_por_mes.csv"
)

OUTPUT_METADATA = (
    OUTPUT_DIR
    / "metadata_z532.json"
)

ID_COL = "numero_de_cliente"
MONTH_COL = "foto_mes"
TARGET_COL = "clase_ternaria"
LAG1_AVAILABLE_COL = "lag1_disponible"
LAG2_AVAILABLE_COL = "lag2_disponible"

EXPECTED_ORIGINAL_FEATURES = 152
EXPECTED_Z523_COLUMNS = 460
EXPECTED_FINAL_COLUMNS = 613

MONTHS = [
    202103,
    202104,
    202105,
    202106,
    202107,
    202108,
]


# ============================================================
# Helpers
# ============================================================

def previous_month(
    foto_mes: int,
    n: int = 1,
) -> int:
    """
    Resta n meses calendario a un YYYYMM.
    """

    year = foto_mes // 100
    month = foto_mes % 100

    total = (
        year * 12
        + (month - 1)
        - n
    )

    new_year = total // 12
    new_month = total % 12 + 1

    return (
        new_year * 100
        + new_month
    )


def quote_identifier(
    name: str,
) -> str:
    """
    Quote seguro para identificadores DuckDB.
    """

    return (
        '"'
        + name.replace(
            '"',
            '""',
        )
        + '"'
    )


def identificar_originales(
    columns: list[str],
) -> list[str]:

    excluir = {
        ID_COL,
        MONTH_COL,
        TARGET_COL,
        LAG1_AVAILABLE_COL,
        LAG2_AVAILABLE_COL,
    }

    originales = [
        c
        for c in columns
        if (
            c not in excluir
            and not c.endswith("_lag1")
            and not c.endswith("_delta_lag1")
            and not c.endswith("_lag2")
        )
    ]

    return originales


# ============================================================
# Main
# ============================================================

def main() -> None:

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
        "z532 - FEATURE ENGINEERING LAG2",
        flush=True,
    )
    print(
        "=" * 80,
        flush=True,
    )

    print(
        f"\nInput:\n  {INPUT_DATASET}",
        flush=True,
    )

    print(
        f"\nOutput:\n  {OUTPUT_PARQUET}",
        flush=True,
    )

    if not INPUT_DATASET.exists():
        raise FileNotFoundError(
            INPUT_DATASET
        )

    # ========================================================
    # Inspección del esquema
    # ========================================================

    con = duckdb.connect()

    schema_df = con.execute(
        f"""
        DESCRIBE
        SELECT *
        FROM read_parquet(
            '{INPUT_DATASET}'
        )
        """
    ).df()

    columns = (
        schema_df[
            "column_name"
        ]
        .astype(str)
        .tolist()
    )

    print(
        f"\nColumnas input: "
        f"{len(columns):,}",
        flush=True,
    )

    if len(columns) != EXPECTED_Z523_COLUMNS:
        raise ValueError(
            f"Se esperaban "
            f"{EXPECTED_Z523_COLUMNS} "
            f"columnas en z523 y hay "
            f"{len(columns)}."
        )

    originales = (
        identificar_originales(
            columns
        )
    )

    print(
        f"Features originales: "
        f"{len(originales):,}",
        flush=True,
    )

    if len(originales) != EXPECTED_ORIGINAL_FEATURES:
        raise ValueError(
            "Se esperaban 152 "
            "features originales."
        )

    lag2_columns = [
        f"{c}_lag2"
        for c in originales
    ]

    colisiones = (
        set(lag2_columns)
        & set(columns)
    )

    if colisiones:
        raise ValueError(
            "Ya existen columnas lag2: "
            + ", ".join(
                sorted(colisiones)[:10]
            )
        )

    # ========================================================
    # Auditoría input
    # ========================================================

    print(
        "\nAuditando dataset de entrada...",
        flush=True,
    )

    input_stats = con.execute(
        f"""
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT
                CAST({quote_identifier(ID_COL)} AS VARCHAR)
                || '|'
                || CAST({quote_identifier(MONTH_COL)} AS VARCHAR)
            ) AS unique_client_month,
            MIN({quote_identifier(MONTH_COL)}) AS min_month,
            MAX({quote_identifier(MONTH_COL)}) AS max_month
        FROM read_parquet(
            '{INPUT_DATASET}'
        )
        """
    ).df().iloc[0]

    input_rows = int(
        input_stats["rows"]
    )

    unique_client_month = int(
        input_stats[
            "unique_client_month"
        ]
    )

    print(
        f"Filas input              : "
        f"{input_rows:,}",
        flush=True,
    )

    print(
        f"Cliente/mes únicos       : "
        f"{unique_client_month:,}",
        flush=True,
    )

    print(
        f"Mes mínimo               : "
        f"{int(input_stats['min_month'])}",
        flush=True,
    )

    print(
        f"Mes máximo               : "
        f"{int(input_stats['max_month'])}",
        flush=True,
    )

    if input_rows != unique_client_month:
        raise ValueError(
            "Hay duplicados cliente/mes "
            "en el dataset de entrada."
        )

    # ========================================================
    # Construcción lag2
    # ========================================================

    print(
        "\nConstruyendo lag2...",
        flush=True,
    )

    lag2_select = ",\n".join(
        [
            (
                f"p.{quote_identifier(c)} "
                f"AS "
                f"{quote_identifier(c + '_lag2')}"
            )
            for c in originales
        ]
    )

    # La relación temporal se expresa usando fechas reales.
    # Para una fila actual c:
    #
    # p.foto_mes debe ser exactamente c.foto_mes - 2 meses.
    #
    # strptime transforma YYYYMM -> fecha primer día del mes.
    join_condition = f"""
        p.{quote_identifier(ID_COL)}
            = c.{quote_identifier(ID_COL)}
        AND
        strftime(
            date_add(
                strptime(
                    CAST(
                        c.{quote_identifier(MONTH_COL)}
                        AS VARCHAR
                    ),
                    '%Y%m'
                ),
                INTERVAL '-2 months'
            ),
            '%Y%m'
        )::INTEGER
            = p.{quote_identifier(MONTH_COL)}
    """

    query_create = f"""
        COPY (
            WITH base AS (
                SELECT *
                FROM read_parquet(
                    '{INPUT_DATASET}'
                )
            )
            SELECT
                c.*,
                CASE
                    WHEN p.{quote_identifier(ID_COL)}
                         IS NOT NULL
                    THEN 1
                    ELSE 0
                END::TINYINT
                    AS {quote_identifier(LAG2_AVAILABLE_COL)},
                {lag2_select}
            FROM base c
            LEFT JOIN base p
                ON {join_condition}
            ORDER BY
                c.{quote_identifier(MONTH_COL)},
                c.{quote_identifier(ID_COL)}
        )
        TO '{OUTPUT_PARQUET}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        )
    """

    t0 = time.time()

    con.execute(
        query_create
    )

    print(
        f"Parquet construido en "
        f"{time.time() - t0:.1f}s",
        flush=True,
    )

    # ========================================================
    # Auditoría output
    # ========================================================

    print(
        "\nAuditando output...",
        flush=True,
    )

    output_schema = con.execute(
        f"""
        DESCRIBE
        SELECT *
        FROM read_parquet(
            '{OUTPUT_PARQUET}'
        )
        """
    ).df()

    output_columns = (
        output_schema[
            "column_name"
        ]
        .astype(str)
        .tolist()
    )

    output_rows = int(
        con.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet(
                '{OUTPUT_PARQUET}'
            )
            """
        ).fetchone()[0]
    )

    print(
        f"Filas output             : "
        f"{output_rows:,}",
        flush=True,
    )

    print(
        f"Columnas output          : "
        f"{len(output_columns):,}",
        flush=True,
    )

    expected_columns = (
        len(columns)
        + 1
        + len(originales)
    )

    if expected_columns != EXPECTED_FINAL_COLUMNS:
        raise ValueError(
            "Error interno en cantidad "
            "esperada de columnas."
        )

    if len(output_columns) != EXPECTED_FINAL_COLUMNS:
        raise ValueError(
            f"Se esperaban "
            f"{EXPECTED_FINAL_COLUMNS} "
            f"columnas y hay "
            f"{len(output_columns)}."
        )

    if output_rows != input_rows:
        raise ValueError(
            "La cantidad de filas cambió "
            "durante el join."
        )

    output_unique = int(
        con.execute(
            f"""
            SELECT COUNT(DISTINCT
                CAST({quote_identifier(ID_COL)} AS VARCHAR)
                || '|'
                || CAST({quote_identifier(MONTH_COL)} AS VARCHAR)
            )
            FROM read_parquet(
                '{OUTPUT_PARQUET}'
            )
            """
        ).fetchone()[0]
    )

    if output_unique != output_rows:
        raise ValueError(
            "El output contiene "
            "duplicados cliente/mes."
        )

    # ========================================================
    # Cobertura lag2 por mes
    # ========================================================

    audit_df = con.execute(
        f"""
        SELECT
            {quote_identifier(MONTH_COL)}
                AS foto_mes,
            COUNT(*) AS filas,
            SUM(
                {quote_identifier(LAG2_AVAILABLE_COL)}
            ) AS con_lag2,
            COUNT(*)
              - SUM(
                    {quote_identifier(LAG2_AVAILABLE_COL)}
                )
                AS sin_lag2,
            100.0
              * SUM(
                    {quote_identifier(LAG2_AVAILABLE_COL)}
                )
              / COUNT(*)
                AS cobertura_pct
        FROM read_parquet(
            '{OUTPUT_PARQUET}'
        )
        GROUP BY
            {quote_identifier(MONTH_COL)}
        ORDER BY
            {quote_identifier(MONTH_COL)}
        """
    ).df()

    audit_df.to_csv(
        OUTPUT_AUDIT,
        index=False,
    )

    print(
        "\n"
        + "=" * 80,
        flush=True,
    )
    print(
        "COBERTURA LAG2 POR MES",
        flush=True,
    )
    print(
        "=" * 80,
        flush=True,
    )

    for _, row in (
        audit_df.iterrows()
    ):

        print(
            f"{int(row['foto_mes'])} "
            f"| filas="
            f"{int(row['filas']):>7,} "
            f"| con_lag2="
            f"{int(row['con_lag2']):>7,} "
            f"| sin_lag2="
            f"{int(row['sin_lag2']):>7,} "
            f"| cobertura="
            f"{row['cobertura_pct']:7.4f}%",
            flush=True,
        )

    # ========================================================
    # Validaciones temporales esperadas
    # ========================================================

    cobertura = {
        int(row["foto_mes"]):
            int(row["con_lag2"])
        for _, row
        in audit_df.iterrows()
    }

    # No existe 202101 para 202103.
    if cobertura.get(
        202103,
        -1,
    ) != 0:
        raise ValueError(
            "202103 no debería "
            "tener lag2."
        )

    # No existe 202102 para 202104.
    if cobertura.get(
        202104,
        -1,
    ) != 0:
        raise ValueError(
            "202104 no debería "
            "tener lag2."
        )

    # A partir de 202105 esperamos
    # cobertura real > 0.
    for month in [
        202105,
        202106,
        202107,
        202108,
    ]:

        if cobertura.get(
            month,
            0,
        ) <= 0:
            raise ValueError(
                f"{month} debería "
                f"tener lag2."
            )

    # ========================================================
    # Auditoría fuerte de continuidad
    # ========================================================

    print(
        "\nAuditando continuidad "
        "calendario exacta...",
        flush=True,
    )

    continuity_errors = int(
        con.execute(
            f"""
            WITH out AS (
                SELECT
                    {quote_identifier(ID_COL)},
                    {quote_identifier(MONTH_COL)},
                    {quote_identifier(LAG2_AVAILABLE_COL)}
                FROM read_parquet(
                    '{OUTPUT_PARQUET}'
                )
            ),
            expected AS (
                SELECT
                    c.{quote_identifier(ID_COL)},
                    c.{quote_identifier(MONTH_COL)},
                    CASE
                        WHEN p.{quote_identifier(ID_COL)}
                             IS NOT NULL
                        THEN 1
                        ELSE 0
                    END AS expected_lag2
                FROM out c
                LEFT JOIN out p
                    ON
                    p.{quote_identifier(ID_COL)}
                        = c.{quote_identifier(ID_COL)}
                    AND
                    strftime(
                        date_add(
                            strptime(
                                CAST(
                                    c.{quote_identifier(MONTH_COL)}
                                    AS VARCHAR
                                ),
                                '%Y%m'
                            ),
                            INTERVAL '-2 months'
                        ),
                        '%Y%m'
                    )::INTEGER
                        = p.{quote_identifier(MONTH_COL)}
            )
            SELECT COUNT(*)
            FROM expected e
            JOIN out o
                USING (
                    {quote_identifier(ID_COL)},
                    {quote_identifier(MONTH_COL)}
                )
            WHERE
                e.expected_lag2
                <> o.{quote_identifier(LAG2_AVAILABLE_COL)}
            """
        ).fetchone()[0]
    )

    print(
        f"Errores continuidad      : "
        f"{continuity_errors}",
        flush=True,
    )

    if continuity_errors != 0:
        raise ValueError(
            "Falló auditoría de "
            "continuidad lag2."
        )

    # ========================================================
    # Missingness lag2
    # ========================================================

    print(
        "\nCalculando missingness lag2...",
        flush=True,
    )

    lag2_null_expr = " + ".join(
        [
            (
                f"CASE WHEN "
                f"{quote_identifier(c)} "
                f"IS NULL "
                f"THEN 1 ELSE 0 END"
            )
            for c in lag2_columns
        ]
    )

    missing_df = con.execute(
        f"""
        SELECT
            {quote_identifier(MONTH_COL)}
                AS foto_mes,
            COUNT(*) AS filas,
            SUM(
                {lag2_null_expr}
            ) AS lag2_nulls
        FROM read_parquet(
            '{OUTPUT_PARQUET}'
        )
        GROUP BY
            {quote_identifier(MONTH_COL)}
        ORDER BY
            {quote_identifier(MONTH_COL)}
        """
    ).df()

    print(
        "\n"
        + "=" * 80,
        flush=True,
    )
    print(
        "NULLS EN FEATURES LAG2",
        flush=True,
    )
    print(
        "=" * 80,
        flush=True,
    )

    for _, row in (
        missing_df.iterrows()
    ):

        print(
            f"{int(row['foto_mes'])} "
            f"| filas="
            f"{int(row['filas']):>7,} "
            f"| lag2_nulls="
            f"{int(row['lag2_nulls']):>10,}",
            flush=True,
        )

    missing_df.to_csv(
        OUTPUT_DIR
        / "missingness_lag2_por_mes.csv",
        index=False,
    )

    # ========================================================
    # Muestra de correspondencia temporal
    # ========================================================

    print(
        "\n"
        + "=" * 80,
        flush=True,
    )
    print(
        "CORRESPONDENCIA TEMPORAL ESPERADA",
        flush=True,
    )
    print(
        "=" * 80,
        flush=True,
    )

    temporal_rows = []

    for month in MONTHS:

        source_month = (
            previous_month(
                month,
                n=2,
            )
        )

        source_exists = (
            source_month
            in MONTHS
        )

        temporal_rows.append(
            {
                "foto_mes":
                    month,
                "lag2_source_month":
                    source_month,
                "source_in_dataset":
                    source_exists,
            }
        )

        print(
            f"{month} <- "
            f"{source_month} "
            f"| fuente disponible="
            f"{source_exists}",
            flush=True,
        )

    pd.DataFrame(
        temporal_rows
    ).to_csv(
        OUTPUT_DIR
        / "mapa_temporal_lag2.csv",
        index=False,
    )

    # ========================================================
    # Metadata
    # ========================================================

    metadata = {
        "script":
            "z532_competencia_01_feature_engineering_lag2.py",

        "input_dataset":
            str(INPUT_DATASET),

        "output_dataset":
            str(OUTPUT_PARQUET),

        "input_rows":
            input_rows,

        "output_rows":
            output_rows,

        "input_columns":
            len(columns),

        "output_columns":
            len(output_columns),

        "original_features":
            len(originales),

        "lag2_features_added":
            len(lag2_columns),

        "indicator_added":
            LAG2_AVAILABLE_COL,

        "lag2_definition":
            (
                "same numero_de_cliente, "
                "exactly two calendar months earlier"
            ),

        "uses_target_for_features":
            False,

        "trains_model":
            False,

        "uses_202108_as_target":
            False,

        "generates_submit":
            False,

        "continuity_errors":
            continuity_errors,
    }

    with OUTPUT_METADATA.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    # ========================================================
    # Final
    # ========================================================

    elapsed = (
        time.time()
        - inicio
    )

    print(
        "\n"
        + "=" * 80,
        flush=True,
    )
    print(
        "AUDITORÍA FINAL",
        flush=True,
    )
    print(
        "=" * 80,
        flush=True,
    )

    print(
        f"Filas preservadas        : "
        f"{output_rows == input_rows}",
        flush=True,
    )

    print(
        f"Cliente/mes únicos       : "
        f"{output_unique == output_rows}",
        flush=True,
    )

    print(
        f"Features originales      : "
        f"{len(originales)}",
        flush=True,
    )

    print(
        f"Features lag2 agregadas  : "
        f"{len(lag2_columns)}",
        flush=True,
    )

    print(
        f"Columnas finales         : "
        f"{len(output_columns)}",
        flush=True,
    )

    print(
        f"Errores continuidad      : "
        f"{continuity_errors}",
        flush=True,
    )

    print(
        f"\nTiempo total: "
        f"{elapsed:.1f}s",
        flush=True,
    )

    print(
        f"\nDataset generado:\n  "
        f"{OUTPUT_PARQUET}",
        flush=True,
    )

    print(
        "\nEste experimento NO entrenó "
        "modelos y NO generó submits.",
        flush=True,
    )

    con.close()


if __name__ == "__main__":
    main()