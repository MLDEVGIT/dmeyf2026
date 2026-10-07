from pathlib import Path
import json
import time

import numpy as np
import pandas as pd


# ============================================================
# Configuración
# ============================================================

DATASET = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/"
    "competencia_01/feature_engineering_z523/"
    "competencia_01_historico_lag1.parquet"
)

RANK_Z537 = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/"
    "competencia_01/submits_z537_ponderado/"
    "ranking_historico_ensemble5.csv"
)

RANK_Z550 = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/"
    "competencia_01/submits_z550_tarjetas_ponderado/"
    "ranking_tarjetas_ponderado_ensemble5.csv"
)

OUTPUT_DIR = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/"
    "competencia_01/anatomia_intercambios_z552"
)

ID = "numero_de_cliente"
MONTH = "foto_mes"
SCORE_MONTH = 202108
N = 12000

TARJETAS = [
    "ctarjeta_debito_transacciones",
    "ctarjeta_visa_transacciones",
    "ctarjeta_master_transacciones",
    "ccuenta_debitos_automaticos",
    "ctarjeta_visa_debitos_automaticos",
    "ctarjeta_master_debitos_automaticos",
]

ACTIVIDAD = "actividad_tarjetas"


# ============================================================
# Helpers
# ============================================================

def smd(x1, x2):
    x1 = pd.to_numeric(
        x1, errors="coerce"
    ).dropna()

    x2 = pd.to_numeric(
        x2, errors="coerce"
    ).dropna()

    if len(x1) < 2 or len(x2) < 2:
        return np.nan

    v1 = x1.var(ddof=1)
    v2 = x2.var(ddof=1)

    pooled = np.sqrt(
        (v1 + v2) / 2
    )

    if (
        not np.isfinite(pooled)
        or pooled == 0
    ):
        return np.nan

    # positivo = mayor en SOLO_Z550
    return (
        x2.mean() - x1.mean()
    ) / pooled


def resumen_variable(df, variable):
    rows = []

    for grupo in [
        "SOLO_Z537",
        "SOLO_Z550",
    ]:
        x = pd.to_numeric(
            df.loc[
                df["grupo"].eq(grupo),
                variable,
            ],
            errors="coerce",
        )

        rows.append({
            "variable": variable,
            "grupo": grupo,
            "n": len(x),
            "n_validos":
                int(x.notna().sum()),
            "pct_na":
                float(
                    x.isna().mean() * 100
                ),
            "media":
                float(x.mean()),
            "mediana":
                float(x.median()),
            "p25":
                float(x.quantile(0.25)),
            "p75":
                float(x.quantile(0.75)),
            "pct_cero":
                float(
                    x.eq(0).mean() * 100
                ),
            "pct_positivo":
                float(
                    x.gt(0).mean() * 100
                ),
        })

    return rows


# ============================================================
# Inicio
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

t0 = time.time()

print("=" * 90)
print("Z552 - ANATOMIA INTERCAMBIOS Z537 vs Z550")
print("=" * 90)


# ============================================================
# Rankings
# ============================================================

z537 = pd.read_csv(
    RANK_Z537
)[
    [ID, "prob", "ranking"]
].rename(
    columns={
        "prob": "prob_z537",
        "ranking": "rank_z537",
    }
)

z550 = pd.read_csv(
    RANK_Z550
)[
    [ID, "prob", "ranking"]
].rename(
    columns={
        "prob": "prob_z550",
        "ranking": "rank_z550",
    }
)

ranking = z537.merge(
    z550,
    on=ID,
    validate="one_to_one",
)

ranking["delta_rank"] = (
    ranking["rank_z550"]
    - ranking["rank_z537"]
)

ranking["grupo"] = "NINGUNO"

ranking.loc[
    (ranking["rank_z537"] <= N)
    & (ranking["rank_z550"] <= N),
    "grupo",
] = "AMBOS"

ranking.loc[
    (ranking["rank_z537"] <= N)
    & (ranking["rank_z550"] > N),
    "grupo",
] = "SOLO_Z537"

ranking.loc[
    (ranking["rank_z537"] > N)
    & (ranking["rank_z550"] <= N),
    "grupo",
] = "SOLO_Z550"

print()
print("GRUPOS TOP 12.000")
print(
    ranking[
        "grupo"
    ].value_counts()
)

assert (
    ranking["grupo"]
    .eq("SOLO_Z537")
    .sum()
    == 444
)

assert (
    ranking["grupo"]
    .eq("SOLO_Z550")
    .sum()
    == 444
)


# ============================================================
# Agosto: componentes + lag + delta
# ============================================================

columnas = [
    ID,
    MONTH,
    *TARJETAS,
]

for v in TARJETAS:
    columnas.extend([
        f"{v}_lag1",
        f"{v}_delta_lag1",
    ])

# preservar orden y evitar duplicados
columnas = list(
    dict.fromkeys(columnas)
)

agosto = pd.read_parquet(
    DATASET,
    columns=columnas,
)

agosto = agosto[
    agosto[MONTH].eq(
        SCORE_MONTH
    )
].copy()

if agosto[ID].duplicated().any():
    raise ValueError(
        "IDs duplicados en agosto"
    )

agosto[ACTIVIDAD] = (
    agosto[TARJETAS]
    .gt(0)
    .sum(axis=1)
    .astype("int8")
)

# también reconstruimos actividad del lag
TARJETAS_LAG = [
    f"{v}_lag1"
    for v in TARJETAS
]

agosto[
    "actividad_tarjetas_lag1"
] = (
    agosto[TARJETAS_LAG]
    .gt(0)
    .sum(axis=1)
    .astype("int8")
)

agosto[
    "actividad_tarjetas_delta"
] = (
    agosto[ACTIVIDAD]
    - agosto[
        "actividad_tarjetas_lag1"
    ]
)

datos = ranking.merge(
    agosto.drop(
        columns=[MONTH]
    ),
    on=ID,
    how="left",
    validate="one_to_one",
)

intercambios = datos[
    datos["grupo"].isin(
        [
            "SOLO_Z537",
            "SOLO_Z550",
        ]
    )
].copy()

print()
print(
    f"Clientes intercambiados: "
    f"{len(intercambios):,}"
)


# ============================================================
# Movimiento de ranking y probabilidad
# ============================================================

print()
print("=" * 90)
print("RANKING / PROBABILIDAD")
print("=" * 90)

mov = (
    intercambios
    .groupby("grupo")
    .agg(
        n=(ID, "size"),
        rank537_media=(
            "rank_z537", "mean"
        ),
        rank537_mediana=(
            "rank_z537", "median"
        ),
        rank550_media=(
            "rank_z550", "mean"
        ),
        rank550_mediana=(
            "rank_z550", "median"
        ),
        delta_rank_media=(
            "delta_rank", "mean"
        ),
        delta_rank_mediana=(
            "delta_rank", "median"
        ),
        prob537_media=(
            "prob_z537", "mean"
        ),
        prob550_media=(
            "prob_z550", "mean"
        ),
    )
)

print(
    mov.to_string(
        float_format=lambda x:
            f"{x:.6f}"
    )
)

mov.to_csv(
    OUTPUT_DIR
    / "resumen_movimientos.csv"
)


# ============================================================
# Actividad agregada
# ============================================================

print()
print("=" * 90)
print("ACTIVIDAD TARJETAS")
print("=" * 90)

actividad_cols = [
    ACTIVIDAD,
    "actividad_tarjetas_lag1",
    "actividad_tarjetas_delta",
]

resumen_actividad = []

for v in actividad_cols:
    resumen_actividad.extend(
        resumen_variable(
            intercambios,
            v,
        )
    )

resumen_actividad = pd.DataFrame(
    resumen_actividad
)

print(
    resumen_actividad[
        [
            "variable",
            "grupo",
            "media",
            "mediana",
            "pct_cero",
            "pct_positivo",
        ]
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)

resumen_actividad.to_csv(
    OUTPUT_DIR
    / "resumen_actividad_agregada.csv",
    index=False,
)


# ============================================================
# Distribución 0..6
# ============================================================

dist_rows = []

for grupo in [
    "SOLO_Z537",
    "SOLO_Z550",
]:
    d = intercambios[
        intercambios[
            "grupo"
        ].eq(grupo)
    ]

    for variable in [
        ACTIVIDAD,
        "actividad_tarjetas_lag1",
    ]:
        vc = (
            d[variable]
            .value_counts()
            .reindex(
                range(7),
                fill_value=0,
            )
            .sort_index()
        )

        for valor, cantidad in vc.items():
            dist_rows.append({
                "grupo": grupo,
                "variable": variable,
                "valor": int(valor),
                "cantidad":
                    int(cantidad),
                "porcentaje":
                    cantidad
                    / len(d)
                    * 100,
            })

dist = pd.DataFrame(
    dist_rows
)

dist.to_csv(
    OUTPUT_DIR
    / "distribucion_actividad.csv",
    index=False,
)

print()
print("=" * 90)
print("DISTRIBUCION ACTIVIDAD ACTUAL")
print("=" * 90)

tabla_dist = (
    dist[
        dist["variable"].eq(
            ACTIVIDAD
        )
    ]
    .pivot(
        index="valor",
        columns="grupo",
        values="porcentaje",
    )
)

print(
    tabla_dist.to_string(
        float_format=lambda x:
            f"{x:.2f}"
    )
)


# ============================================================
# Componentes individuales
# ============================================================

print()
print("=" * 90)
print("COMPONENTES ACTUALES")
print("=" * 90)

componentes_rows = []

for v in TARJETAS:
    componentes_rows.extend(
        resumen_variable(
            intercambios,
            v,
        )
    )

componentes = pd.DataFrame(
    componentes_rows
)

componentes.to_csv(
    OUTPUT_DIR
    / "resumen_componentes_actuales.csv",
    index=False,
)

tabla_componentes = (
    componentes.pivot(
        index="variable",
        columns="grupo",
        values=[
            "media",
            "pct_positivo",
        ],
    )
)

print(
    tabla_componentes.to_string(
        float_format=lambda x:
            f"{x:.3f}"
    )
)


# ============================================================
# SMD: actuales, lag y delta
# ============================================================

variables_smd = [
    ACTIVIDAD,
    "actividad_tarjetas_lag1",
    "actividad_tarjetas_delta",
]

for v in TARJETAS:
    variables_smd.extend([
        v,
        f"{v}_lag1",
        f"{v}_delta_lag1",
    ])

variables_smd = list(
    dict.fromkeys(
        variables_smd
    )
)

smd_rows = []

a = intercambios[
    intercambios[
        "grupo"
    ].eq("SOLO_Z537")
]

b = intercambios[
    intercambios[
        "grupo"
    ].eq("SOLO_Z550")
]

for v in variables_smd:

    x1 = pd.to_numeric(
        a[v],
        errors="coerce",
    )

    x2 = pd.to_numeric(
        b[v],
        errors="coerce",
    )

    smd_rows.append({
        "variable": v,
        "media_solo_z537":
            float(x1.mean()),
        "media_solo_z550":
            float(x2.mean()),
        "mediana_solo_z537":
            float(x1.median()),
        "mediana_solo_z550":
            float(x2.median()),
        "smd_z550_menos_z537":
            smd(x1, x2),
    })

smd_df = pd.DataFrame(
    smd_rows
)

smd_df[
    "abs_smd"
] = (
    smd_df[
        "smd_z550_menos_z537"
    ].abs()
)

smd_df = (
    smd_df
    .sort_values(
        "abs_smd",
        ascending=False,
    )
)

smd_df.to_csv(
    OUTPUT_DIR
    / "smd_intercambios.csv",
    index=False,
)

print()
print("=" * 90)
print("SMD SOLO_Z550 - SOLO_Z537")
print("positivo = mayor en Z550")
print("=" * 90)

print(
    smd_df[
        [
            "variable",
            "media_solo_z537",
            "media_solo_z550",
            "mediana_solo_z537",
            "mediana_solo_z550",
            "smd_z550_menos_z537",
        ]
    ]
    .head(30)
    .to_string(
        index=False,
        float_format=lambda x:
            f"{x:+.4f}",
    )
)


# ============================================================
# Cambio de estado activo/inactivo por componente
# ============================================================

transiciones_rows = []

for grupo in [
    "SOLO_Z537",
    "SOLO_Z550",
]:
    d = intercambios[
        intercambios[
            "grupo"
        ].eq(grupo)
    ]

    for v in TARJETAS:

        actual = d[v].gt(0)

        lag = d[
            f"{v}_lag1"
        ].gt(0)

        transiciones_rows.append({
            "grupo": grupo,
            "variable": v,
            "pct_inactivo_inactivo":
                float(
                    ((~lag) & (~actual))
                    .mean() * 100
                ),
            "pct_activo_inactivo":
                float(
                    (lag & (~actual))
                    .mean() * 100
                ),
            "pct_inactivo_activo":
                float(
                    ((~lag) & actual)
                    .mean() * 100
                ),
            "pct_activo_activo":
                float(
                    (lag & actual)
                    .mean() * 100
                ),
        })

transiciones = pd.DataFrame(
    transiciones_rows
)

transiciones.to_csv(
    OUTPUT_DIR
    / "transiciones_componentes.csv",
    index=False,
)

print()
print("=" * 90)
print("TRANSICIONES ACTIVO/INACTIVO")
print("=" * 90)

print(
    transiciones.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.2f}",
    )
)


# ============================================================
# Archivo individual para inspección
# ============================================================

cols_salida = [
    ID,
    "grupo",
    "prob_z537",
    "prob_z550",
    "rank_z537",
    "rank_z550",
    "delta_rank",
    ACTIVIDAD,
    "actividad_tarjetas_lag1",
    "actividad_tarjetas_delta",
]

for v in TARJETAS:
    cols_salida.extend([
        v,
        f"{v}_lag1",
        f"{v}_delta_lag1",
    ])

cols_salida = list(
    dict.fromkeys(
        cols_salida
    )
)

intercambios[
    cols_salida
].sort_values(
    [
        "grupo",
        "rank_z550",
    ]
).to_csv(
    OUTPUT_DIR
    / "clientes_intercambiados_12000.csv",
    index=False,
)


# ============================================================
# Metadata
# ============================================================

metadata = {
    "dataset":
        str(DATASET),
    "ranking_z537":
        str(RANK_Z537),
    "ranking_z550":
        str(RANK_Z550),
    "score_month":
        SCORE_MONTH,
    "cut":
        N,
    "n_solo_z537":
        int(
            ranking["grupo"]
            .eq("SOLO_Z537")
            .sum()
        ),
    "n_solo_z550":
        int(
            ranking["grupo"]
            .eq("SOLO_Z550")
            .sum()
        ),
    "componentes_tarjetas":
        TARJETAS,
    "runtime_segundos":
        time.time() - t0,
}

with open(
    OUTPUT_DIR
    / "metadata_z552.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        metadata,
        f,
        indent=2,
        ensure_ascii=False,
    )


print()
print(
    f"Runtime total: "
    f"{time.time() - t0:.2f}s"
)

print()
print("Z552 FINALIZADO")
print("Output:", OUTPUT_DIR)
