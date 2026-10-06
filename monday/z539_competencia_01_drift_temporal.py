#!/usr/bin/env python3

"""
z539_competencia_01_drift_temporal.py

Diagnostico temporal de Competencia 01.

Objetivos:
1. Medir cantidad y tasa de BAJA+2 por mes observable.
2. Analizar drift mensual de las variables originales.
3. Detectar variables con cambios abruptos de:
   - missingness
   - proporcion de ceros
   - mediana
   - distribucion
4. Identificar variables sospechosas para posteriores
   experimentos de ablacion.

IMPORTANTE:
- Es EDA. No entrena modelos.
- No elimina variables.
- No usa 202108 como target.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import duckdb


# ============================================================
# CONFIGURACION
# ============================================================

DATASET = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/"
    "competencia_01/feature_engineering_z523/"
    "competencia_01_historico_lag1.parquet"
)

OUTPUT_DIR = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/"
    "competencia_01/drift_temporal_z539"
)

ID_COL = "numero_de_cliente"
MONTH_COL = "foto_mes"
TARGET_COL = "clase_ternaria"

# Meses con BAJA+2 completamente observable
TARGET_MONTHS = [
    202103,
    202104,
    202105,
    202106,
]

# Para drift de X sí podemos mirar todos los meses.
DRIFT_MONTHS = [
    202103,
    202104,
    202105,
    202106,
    202107,
    202108,
]

EXPECTED_ORIGINALS = 152


# ============================================================
# HELPERS
# ============================================================

def quote_col(name):
    return '"' + name.replace('"', '""') + '"'


def robust_relative_change(a, b):
    """
    Cambio relativo simetrico robusto.

    Evita dividir solamente por el valor del mes anterior,
    que puede estar cerca de cero.
    """
    denom = abs(a) + abs(b) + 1e-12
    return abs(b - a) / denom


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("Z539 - DIAGNOSTICO TEMPORAL")
    print("=" * 80)
    print(f"Dataset: {DATASET}")
    print(f"Output : {OUTPUT_DIR}")

    con = duckdb.connect()

    # --------------------------------------------------------
    # 1. ESTRUCTURA DEL DATASET
    # --------------------------------------------------------

    schema = con.execute(
        f"""
        DESCRIBE
        SELECT *
        FROM read_parquet('{DATASET}')
        """
    ).df()

    all_cols = schema["column_name"].tolist()

    print()
    print(f"Columnas totales: {len(all_cols)}")

    # En Z523:
    # primeras columnas originales +
    # posteriormente lag/delta/features auxiliares.
    #
    # Detectamos originales excluyendo derivados conocidos.

    derived_patterns = (
        "_lag1",
        "_delta1",
    )

    auxiliary_cols = {
        ID_COL,
        MONTH_COL,
        TARGET_COL,
        "lag1_disponible",
    }

    original_features = []

    for col in all_cols:

        if col in auxiliary_cols:
            continue

        if any(
            pattern in col
            for pattern in derived_patterns
        ):
            continue

        original_features.append(col)

    print(
        "Features originales detectadas:",
        len(original_features),
    )

    if len(original_features) != EXPECTED_ORIGINALS:
        print(
            "ADVERTENCIA: se esperaban",
            EXPECTED_ORIGINALS,
            "features originales.",
        )

    pd.DataFrame(
        {"feature": original_features}
    ).to_csv(
        OUTPUT_DIR / "features_originales.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 2. BAJAS POR MES
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("BAJAS POR MES")
    print("=" * 80)

    bajas = con.execute(
        f"""
        SELECT
            {MONTH_COL},

            COUNT(*) AS n_clientes,

            SUM(
                CASE
                    WHEN {TARGET_COL} = 'BAJA+2'
                    THEN 1 ELSE 0
                END
            ) AS baja2,

            SUM(
                CASE
                    WHEN {TARGET_COL} = 'BAJA+1'
                    THEN 1 ELSE 0
                END
            ) AS baja1,

            SUM(
                CASE
                    WHEN {TARGET_COL} = 'CONTINUA'
                    THEN 1 ELSE 0
                END
            ) AS continua

        FROM read_parquet('{DATASET}')

        WHERE {MONTH_COL}
            IN ({",".join(map(str, TARGET_MONTHS))})

        GROUP BY {MONTH_COL}
        ORDER BY {MONTH_COL}
        """
    ).df()

    bajas["pct_baja2"] = (
        bajas["baja2"]
        / bajas["n_clientes"]
    )

    bajas["pct_baja1"] = (
        bajas["baja1"]
        / bajas["n_clientes"]
    )

    bajas["delta_baja2"] = (
        bajas["baja2"].diff()
    )

    bajas["delta_pct_baja2"] = (
        bajas["pct_baja2"].diff()
    )

    print(
        bajas.to_string(
            index=False,
        )
    )

    bajas.to_csv(
        OUTPUT_DIR / "bajas_por_mes.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 3. ESTADISTICAS MENSUALES
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("CALCULANDO DRIFT")
    print("=" * 80)

    rows = []

    for i, feature in enumerate(
        original_features,
        start=1,
    ):

        print(
            f"[{i:03d}/{len(original_features):03d}] "
            f"{feature}"
        )

        q = quote_col(feature)

        stats = con.execute(
            f"""
            SELECT
                {MONTH_COL},

                COUNT(*) AS n,

                COUNT(*) FILTER (
                    WHERE {q} IS NULL
                ) AS n_na,

                COUNT(*) FILTER (
                    WHERE {q} = 0
                ) AS n_zero,

                AVG({q}) AS mean,

                MEDIAN({q}) AS median,

                QUANTILE_CONT(
                    {q},
                    0.25
                ) AS q25,

                QUANTILE_CONT(
                    {q},
                    0.75
                ) AS q75

            FROM read_parquet('{DATASET}')

            WHERE {MONTH_COL}
                IN ({",".join(map(str, DRIFT_MONTHS))})

            GROUP BY {MONTH_COL}
            ORDER BY {MONTH_COL}
            """
        ).df()

        stats["feature"] = feature

        stats["pct_na"] = (
            stats["n_na"]
            / stats["n"]
        )

        stats["pct_zero"] = (
            stats["n_zero"]
            / stats["n"]
        )

        rows.append(stats)

    monthly = pd.concat(
        rows,
        ignore_index=True,
    )

    monthly.to_csv(
        OUTPUT_DIR
        / "estadisticas_variables_por_mes.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 4. CAMBIOS MES A MES
    # --------------------------------------------------------

    comparisons = []

    for feature, g in monthly.groupby(
        "feature",
        sort=False,
    ):

        g = (
            g.sort_values(MONTH_COL)
            .reset_index(drop=True)
        )

        for i in range(1, len(g)):

            prev = g.iloc[i - 1]
            curr = g.iloc[i]

            comparisons.append(
                {
                    "feature": feature,
                    "mes_anterior":
                        int(prev[MONTH_COL]),
                    "mes_actual":
                        int(curr[MONTH_COL]),

                    "delta_pct_na":
                        curr["pct_na"]
                        - prev["pct_na"],

                    "delta_pct_zero":
                        curr["pct_zero"]
                        - prev["pct_zero"],

                    "median_prev":
                        prev["median"],

                    "median_curr":
                        curr["median"],

                    "median_relative_change":
                        robust_relative_change(
                            prev["median"],
                            curr["median"],
                        )
                        if (
                            pd.notna(prev["median"])
                            and pd.notna(curr["median"])
                        )
                        else np.nan,
                }
            )

    changes = pd.DataFrame(
        comparisons
    )

    changes.to_csv(
        OUTPUT_DIR
        / "cambios_mes_a_mes.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 5. SCORE DE SOSPECHA
    # --------------------------------------------------------

    # Máximo salto observado por variable.
    summary = (
        changes
        .groupby("feature")
        .agg(
            max_delta_na=(
                "delta_pct_na",
                lambda x:
                    np.nanmax(np.abs(x))
            ),
            max_delta_zero=(
                "delta_pct_zero",
                lambda x:
                    np.nanmax(np.abs(x))
            ),
            max_median_change=(
                "median_relative_change",
                "max",
            ),
        )
        .reset_index()
    )

    # Ranking robusto:
    # usamos percentiles/ranks para que ninguna escala
    # domine automáticamente a las demás.

    for col in [
        "max_delta_na",
        "max_delta_zero",
        "max_median_change",
    ]:
        summary[
            f"rank_{col}"
        ] = (
            summary[col]
            .rank(
                pct=True,
                method="average",
            )
        )

    summary["drift_score"] = (
        summary["rank_max_delta_na"]
        + summary["rank_max_delta_zero"]
        + summary["rank_max_median_change"]
    ) / 3

    summary = summary.sort_values(
        "drift_score",
        ascending=False,
    )

    # Marcamos préstamos.
    summary["es_prestamo"] = (
        summary["feature"]
        .str.lower()
        .str.contains(
            "prest|loan",
            regex=True,
        )
    )

    summary.to_csv(
        OUTPUT_DIR
        / "ranking_drift_variables.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 6. TOP SOSPECHOSAS
    # --------------------------------------------------------

    top = summary.head(30)

    print()
    print("=" * 80)
    print("TOP 30 VARIABLES CON MAYOR DRIFT")
    print("=" * 80)

    print(
        top[
            [
                "feature",
                "max_delta_na",
                "max_delta_zero",
                "max_median_change",
                "drift_score",
                "es_prestamo",
            ]
        ].to_string(
            index=False
        )
    )

    prestamos = summary[
        summary["es_prestamo"]
    ]

    print()
    print("=" * 80)
    print("VARIABLES DE PRESTAMOS")
    print("=" * 80)

    print(
        prestamos.to_string(
            index=False
        )
    )

    top.to_csv(
        OUTPUT_DIR
        / "top30_variables_sospechosas.csv",
        index=False,
    )

    prestamos.to_csv(
        OUTPUT_DIR
        / "drift_variables_prestamos.csv",
        index=False,
    )

    print()
    print("=" * 80)
    print("Z539 FINALIZADO")
    print("=" * 80)
    print(f"Resultados: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()