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

OUTPUT_DIR = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/"
    "competencia_01/diagnostico_tarjetas_z551"
)

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

TARJETAS = [
    "ctarjeta_debito_transacciones",
    "ctarjeta_visa_transacciones",
    "ctarjeta_master_transacciones",
    "ccuenta_debitos_automaticos",
    "ctarjeta_visa_debitos_automaticos",
    "ctarjeta_master_debitos_automaticos",
]

FEATURE_AGREGADA = "actividad_tarjetas"

SALTOS_CLAVE = [
    (202105, 202106, "may_jun"),
    (202106, 202107, "jun_jul"),
    (202107, 202108, "jul_ago"),
    (202106, 202108, "jun_ago"),
]


# ============================================================
# Inicio
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

t0 = time.time()

print("=" * 90)
print("Z551 - DIAGNOSTICO TEMPORAL ACTIVIDAD TARJETAS")
print("=" * 90)

df = pd.read_parquet(
    DATASET,
    columns=[
        MONTH_COL,
        TARGET_COL,
        *TARJETAS,
    ],
)

df = df[
    df[MONTH_COL].isin(MESES)
].copy()

print(f"Filas: {len(df):,}")
print(f"Meses: {sorted(df[MONTH_COL].unique())}")

faltantes = [
    c for c in TARJETAS
    if c not in df.columns
]

if faltantes:
    raise ValueError(
        f"Faltan variables: {faltantes}"
    )


# ============================================================
# Actividad agregada
# ============================================================

df[FEATURE_AGREGADA] = (
    df[TARJETAS]
    .gt(0)
    .sum(axis=1)
    .astype("int8")
)


# ============================================================
# Estadísticas de componentes por mes
# ============================================================

rows = []

for mes in MESES:

    d = df[
        df[MONTH_COL].eq(mes)
    ]

    for variable in TARJETAS:

        x = d[variable]

        rows.append({
            "foto_mes": mes,
            "variable": variable,
            "n": len(x),
            "n_validos": int(x.notna().sum()),
            "pct_na": float(x.isna().mean() * 100),
            "media": float(x.mean()),
            "mediana": float(x.median()),
            "p25": float(x.quantile(0.25)),
            "p75": float(x.quantile(0.75)),
            "p90": float(x.quantile(0.90)),
            "p95": float(x.quantile(0.95)),
            "pct_cero": float(
                x.eq(0).mean() * 100
            ),
            "pct_activo": float(
                x.gt(0).mean() * 100
            ),
        })

stats = pd.DataFrame(rows)

stats.to_csv(
    OUTPUT_DIR
    / "estadisticas_componentes_por_mes.csv",
    index=False,
)


# ============================================================
# Tabla compacta: porcentaje activo
# ============================================================

activo = (
    stats.pivot(
        index="variable",
        columns="foto_mes",
        values="pct_activo",
    )
    .reindex(columns=MESES)
)

print()
print("=" * 90)
print("% ACTIVO (>0) POR COMPONENTE Y MES")
print("=" * 90)

print(
    activo.to_string(
        float_format=lambda x: f"{x:7.2f}"
    )
)

activo.to_csv(
    OUTPUT_DIR
    / "pct_activo_componentes.csv"
)


# ============================================================
# Tabla compacta: media
# ============================================================

medias = (
    stats.pivot(
        index="variable",
        columns="foto_mes",
        values="media",
    )
    .reindex(columns=MESES)
)

print()
print("=" * 90)
print("MEDIA POR COMPONENTE Y MES")
print("=" * 90)

print(
    medias.to_string(
        float_format=lambda x: f"{x:9.3f}"
    )
)

medias.to_csv(
    OUTPUT_DIR
    / "media_componentes.csv"
)


# ============================================================
# Saltos en porcentaje activo
# ============================================================

saltos_rows = []

for variable in TARJETAS:

    s = (
        stats[
            stats["variable"].eq(variable)
        ]
        .set_index("foto_mes")
    )

    for mes_a, mes_b, nombre in SALTOS_CLAVE:

        a = s.loc[
            mes_a,
            "pct_activo",
        ]

        b = s.loc[
            mes_b,
            "pct_activo",
        ]

        saltos_rows.append({
            "variable": variable,
            "salto": nombre,
            "mes_desde": mes_a,
            "mes_hasta": mes_b,
            "pct_activo_desde": a,
            "pct_activo_hasta": b,
            "delta_pp_activo": b - a,
        })

saltos_activo = pd.DataFrame(
    saltos_rows
)

saltos_activo.to_csv(
    OUTPUT_DIR
    / "saltos_pct_activo.csv",
    index=False,
)

print()
print("=" * 90)
print("SALTOS % ACTIVO - PUNTOS PORCENTUALES")
print("=" * 90)

for nombre in [
    "may_jun",
    "jun_jul",
    "jul_ago",
    "jun_ago",
]:

    print()
    print(nombre.upper())

    z = (
        saltos_activo[
            saltos_activo[
                "salto"
            ].eq(nombre)
        ]
        .sort_values(
            "delta_pp_activo",
            key=lambda s:
                s.abs(),
            ascending=False,
        )
    )

    print(
        z[
            [
                "variable",
                "pct_activo_desde",
                "pct_activo_hasta",
                "delta_pp_activo",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:+.3f}",
        )
    )


# ============================================================
# Saltos de medias
# ============================================================

saltos_media_rows = []

for variable in TARJETAS:

    s = (
        stats[
            stats["variable"].eq(variable)
        ]
        .set_index("foto_mes")
    )

    for mes_a, mes_b, nombre in SALTOS_CLAVE:

        a = s.loc[
            mes_a,
            "media",
        ]

        b = s.loc[
            mes_b,
            "media",
        ]

        if (
            np.isfinite(a)
            and a != 0
        ):
            cambio_pct = (
                (b / a) - 1
            ) * 100
        else:
            cambio_pct = np.nan

        saltos_media_rows.append({
            "variable": variable,
            "salto": nombre,
            "mes_desde": mes_a,
            "mes_hasta": mes_b,
            "media_desde": a,
            "media_hasta": b,
            "delta_media": b - a,
            "cambio_pct_media":
                cambio_pct,
        })

saltos_media = pd.DataFrame(
    saltos_media_rows
)

saltos_media.to_csv(
    OUTPUT_DIR
    / "saltos_media.csv",
    index=False,
)

print()
print("=" * 90)
print("CAMBIO DE MEDIA JUNIO -> AGOSTO")
print("=" * 90)

z = (
    saltos_media[
        saltos_media[
            "salto"
        ].eq("jun_ago")
    ]
    .sort_values(
        "cambio_pct_media",
        key=lambda s:
            s.abs(),
        ascending=False,
    )
)

print(
    z[
        [
            "variable",
            "media_desde",
            "media_hasta",
            "delta_media",
            "cambio_pct_media",
        ]
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:+.3f}",
    )
)


# ============================================================
# Distribución actividad_tarjetas
# ============================================================

dist_rows = []

for mes in MESES:

    d = df[
        df[MONTH_COL].eq(mes)
    ]

    vc = (
        d[FEATURE_AGREGADA]
        .value_counts()
        .reindex(
            range(
                len(TARJETAS) + 1
            ),
            fill_value=0,
        )
        .sort_index()
    )

    total = len(d)

    for actividad, cantidad in vc.items():

        dist_rows.append({
            "foto_mes": mes,
            "actividad_tarjetas":
                int(actividad),
            "cantidad":
                int(cantidad),
            "porcentaje":
                cantidad
                / total
                * 100,
        })

dist = pd.DataFrame(
    dist_rows
)

dist.to_csv(
    OUTPUT_DIR
    / "distribucion_actividad_tarjetas.csv",
    index=False,
)

dist_pct = (
    dist.pivot(
        index="actividad_tarjetas",
        columns="foto_mes",
        values="porcentaje",
    )
    .reindex(columns=MESES)
)

print()
print("=" * 90)
print("DISTRIBUCION ACTIVIDAD_TARJETAS (%)")
print("=" * 90)

print(
    dist_pct.to_string(
        float_format=lambda x:
            f"{x:7.3f}"
    )
)


# ============================================================
# Resumen agregado por mes
# ============================================================

resumen_rows = []

for mes in MESES:

    d = df[
        df[MONTH_COL].eq(mes)
    ]

    a = d[
        FEATURE_AGREGADA
    ]

    resumen_rows.append({
        "foto_mes": mes,
        "n": len(d),
        "actividad_media":
            float(a.mean()),
        "actividad_mediana":
            float(a.median()),
        "pct_actividad_0":
            float(
                a.eq(0).mean()
                * 100
            ),
        "pct_actividad_le1":
            float(
                a.le(1).mean()
                * 100
            ),
        "pct_actividad_ge4":
            float(
                a.ge(4).mean()
                * 100
            ),
        "pct_actividad_6":
            float(
                a.eq(6).mean()
                * 100
            ),
    })

resumen = pd.DataFrame(
    resumen_rows
)

resumen.to_csv(
    OUTPUT_DIR
    / "resumen_actividad_por_mes.csv",
    index=False,
)

print()
print("=" * 90)
print("RESUMEN ACTIVIDAD AGREGADA")
print("=" * 90)

print(
    resumen.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


# ============================================================
# BAJA+2 vs RESTO en meses observables
# ============================================================

target_rows = []

for mes in [
    202103,
    202104,
    202105,
    202106,
]:

    d = df[
        df[MONTH_COL].eq(mes)
    ].copy()

    d["grupo"] = np.where(
        d[TARGET_COL].eq("BAJA+2"),
        "BAJA+2",
        "RESTO",
    )

    for grupo, g in d.groupby(
        "grupo"
    ):

        target_rows.append({
            "foto_mes": mes,
            "grupo": grupo,
            "n": len(g),
            "actividad_tarjetas_media":
                float(
                    g[
                        FEATURE_AGREGADA
                    ].mean()
                ),
            "actividad_tarjetas_mediana":
                float(
                    g[
                        FEATURE_AGREGADA
                    ].median()
                ),
            "pct_actividad_0":
                float(
                    g[
                        FEATURE_AGREGADA
                    ]
                    .eq(0)
                    .mean()
                    * 100
                ),
            "pct_actividad_ge4":
                float(
                    g[
                        FEATURE_AGREGADA
                    ]
                    .ge(4)
                    .mean()
                    * 100
                ),
        })

target_stats = pd.DataFrame(
    target_rows
)

target_stats.to_csv(
    OUTPUT_DIR
    / "actividad_tarjetas_baja2_vs_resto.csv",
    index=False,
)

print()
print("=" * 90)
print("ACTIVIDAD TARJETAS - BAJA+2 VS RESTO")
print("=" * 90)

print(
    target_stats.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


# ============================================================
# Ranking de cambios anómalos jun -> ago
# ============================================================

jun_ago_activo = (
    saltos_activo[
        saltos_activo[
            "salto"
        ].eq("jun_ago")
    ][
        [
            "variable",
            "delta_pp_activo",
        ]
    ]
)

jun_ago_media = (
    saltos_media[
        saltos_media[
            "salto"
        ].eq("jun_ago")
    ][
        [
            "variable",
            "cambio_pct_media",
        ]
    ]
)

ranking_cambios = (
    jun_ago_activo
    .merge(
        jun_ago_media,
        on="variable",
        validate="one_to_one",
    )
)

ranking_cambios[
    "abs_delta_pp_activo"
] = (
    ranking_cambios[
        "delta_pp_activo"
    ].abs()
)

ranking_cambios[
    "abs_cambio_pct_media"
] = (
    ranking_cambios[
        "cambio_pct_media"
    ].abs()
)

ranking_cambios = (
    ranking_cambios
    .sort_values(
        [
            "abs_delta_pp_activo",
            "abs_cambio_pct_media",
        ],
        ascending=False,
    )
)

ranking_cambios.to_csv(
    OUTPUT_DIR
    / "ranking_cambios_junio_agosto.csv",
    index=False,
)

print()
print("=" * 90)
print("RANKING CAMBIOS JUNIO -> AGOSTO")
print("=" * 90)

print(
    ranking_cambios.to_string(
        index=False,
        float_format=lambda x:
            f"{x:+.3f}",
    )
)


# ============================================================
# Metadata
# ============================================================

metadata = {
    "dataset": str(DATASET),
    "meses": MESES,
    "variables_tarjetas":
        TARJETAS,
    "feature_agregada":
        FEATURE_AGREGADA,
    "saltos_clave":
        SALTOS_CLAVE,
    "runtime_segundos":
        time.time() - t0,
}

with open(
    OUTPUT_DIR
    / "metadata_z551.json",
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
print("Z551 FINALIZADO")
print("Output:", OUTPUT_DIR)
