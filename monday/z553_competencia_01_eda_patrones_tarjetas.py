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
    "competencia_01/eda_patrones_tarjetas_z553"
)

MONTH = "foto_mes"
TARGET = "clase_ternaria"

MESES_TARGET = [
    202103,
    202104,
    202105,
    202106,
]

VARS = {
    "debito_trx":
        "ctarjeta_debito_transacciones",
    "visa_trx":
        "ctarjeta_visa_transacciones",
    "master_trx":
        "ctarjeta_master_transacciones",
    "cuenta_da":
        "ccuenta_debitos_automaticos",
    "visa_da":
        "ctarjeta_visa_debitos_automaticos",
    "master_da":
        "ctarjeta_master_debitos_automaticos",
}

TRX = [
    "debito_trx",
    "visa_trx",
    "master_trx",
]

DA = [
    "cuenta_da",
    "visa_da",
    "master_da",
]


# ============================================================
# Inicio
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

t0 = time.time()

print("=" * 95)
print("Z553 - EDA PATRONES DE ACTIVIDAD DE TARJETAS")
print("=" * 95)

columnas = [
    MONTH,
    TARGET,
    *VARS.values(),
]

df = pd.read_parquet(
    DATASET,
    columns=columnas,
)

df = df[
    df[MONTH].isin(MESES_TARGET)
].copy()

print(f"Filas: {len(df):,}")
print(f"Meses: {MESES_TARGET}")


# ============================================================
# Target
# ============================================================

df["baja2"] = (
    df[TARGET]
    .eq("BAJA+2")
    .astype("int8")
)

print(
    f"BAJA+2 total: "
    f"{df['baja2'].sum():,}"
)


# ============================================================
# Banderas binarias
# ============================================================

for alias, variable in VARS.items():
    df[alias] = (
        df[variable]
        .gt(0)
        .astype("int8")
    )

# Orden explícito del patrón:
# debito_trx, visa_trx, master_trx,
# cuenta_da, visa_da, master_da

aliases = list(VARS.keys())

df["patron6"] = (
    df[aliases]
    .astype(str)
    .agg("".join, axis=1)
)

df["actividad_tarjetas"] = (
    df[aliases]
    .sum(axis=1)
    .astype("int8")
)

df["actividad_trx"] = (
    df[TRX]
    .sum(axis=1)
    .astype("int8")
)

df["actividad_da"] = (
    df[DA]
    .sum(axis=1)
    .astype("int8")
)


# ============================================================
# 1. Cada componente: BAJA+2 activo vs inactivo
# ============================================================

componentes_rows = []

for mes in MESES_TARGET:
    d = df[
        df[MONTH].eq(mes)
    ]

    for alias in aliases:
        for activo in [0, 1]:

            z = d[
                d[alias].eq(activo)
            ]

            n = len(z)
            bajas = int(
                z["baja2"].sum()
            )

            tasa = (
                bajas / n
                if n > 0
                else np.nan
            )

            componentes_rows.append({
                "foto_mes": mes,
                "componente": alias,
                "activo": activo,
                "n": n,
                "baja2": bajas,
                "tasa_baja2": tasa,
                "tasa_baja2_pct":
                    tasa * 100,
            })

componentes = pd.DataFrame(
    componentes_rows
)

componentes.to_csv(
    OUTPUT_DIR
    / "baja2_por_componente_mes.csv",
    index=False,
)

print()
print("=" * 95)
print("TASA BAJA+2: ACTIVO VS INACTIVO")
print("=" * 95)

for alias in aliases:

    z = componentes[
        componentes[
            "componente"
        ].eq(alias)
    ]

    tabla = z.pivot(
        index="foto_mes",
        columns="activo",
        values="tasa_baja2_pct",
    )

    tabla = tabla.rename(
        columns={
            0: "inactivo_pct",
            1: "activo_pct",
        }
    )

    tabla["ratio_inactivo_activo"] = (
        tabla["inactivo_pct"]
        / tabla["activo_pct"]
    )

    print()
    print(alias.upper())

    print(
        tabla.to_string(
            float_format=lambda x:
                f"{x:.4f}"
        )
    )


# ============================================================
# 2. Actividad total 0..6
# ============================================================

actividad_rows = []

for mes in MESES_TARGET:
    d = df[
        df[MONTH].eq(mes)
    ]

    for actividad in range(7):

        z = d[
            d[
                "actividad_tarjetas"
            ].eq(actividad)
        ]

        n = len(z)
        bajas = int(
            z["baja2"].sum()
        )

        tasa = (
            bajas / n
            if n > 0
            else np.nan
        )

        actividad_rows.append({
            "foto_mes": mes,
            "actividad_tarjetas":
                actividad,
            "n": n,
            "baja2": bajas,
            "tasa_baja2": tasa,
            "tasa_baja2_pct":
                tasa * 100,
        })

actividad = pd.DataFrame(
    actividad_rows
)

actividad.to_csv(
    OUTPUT_DIR
    / "baja2_por_actividad_total_mes.csv",
    index=False,
)

tabla_actividad = actividad.pivot(
    index="actividad_tarjetas",
    columns="foto_mes",
    values="tasa_baja2_pct",
)

print()
print("=" * 95)
print("TASA BAJA+2 (%) SEGUN ACTIVIDAD TOTAL 0..6")
print("=" * 95)

print(
    tabla_actividad.to_string(
        float_format=lambda x:
            f"{x:.4f}"
    )
)


# ============================================================
# 3. Separación transacciones vs débitos automáticos
# ============================================================

trx_da_rows = []

for mes in MESES_TARGET:

    d = df[
        df[MONTH].eq(mes)
    ]

    for trx in range(4):
        for da in range(4):

            z = d[
                d[
                    "actividad_trx"
                ].eq(trx)
                &
                d[
                    "actividad_da"
                ].eq(da)
            ]

            n = len(z)
            bajas = int(
                z["baja2"].sum()
            )

            tasa = (
                bajas / n
                if n > 0
                else np.nan
            )

            trx_da_rows.append({
                "foto_mes": mes,
                "actividad_trx": trx,
                "actividad_da": da,
                "n": n,
                "baja2": bajas,
                "tasa_baja2": tasa,
                "tasa_baja2_pct":
                    tasa * 100,
            })

trx_da = pd.DataFrame(
    trx_da_rows
)

trx_da.to_csv(
    OUTPUT_DIR
    / "baja2_trx_vs_da_mes.csv",
    index=False,
)

print()
print("=" * 95)
print("TASA BAJA+2 (%) - TRANSACCIONES x DEBITOS AUTOMATICOS")
print("=" * 95)

for mes in MESES_TARGET:

    z = trx_da[
        trx_da[MONTH].eq(mes)
    ]

    tabla = z.pivot(
        index="actividad_trx",
        columns="actividad_da",
        values="tasa_baja2_pct",
    )

    print()
    print(f"MES {mes}")

    print(
        tabla.to_string(
            float_format=lambda x:
                f"{x:.4f}"
        )
    )


# ============================================================
# 4. Patrones completos de seis banderas
# ============================================================

patrones_rows = []

for mes in MESES_TARGET:

    d = df[
        df[MONTH].eq(mes)
    ]

    g = (
        d.groupby(
            "patron6",
            observed=True,
        )
        .agg(
            n=("baja2", "size"),
            baja2=("baja2", "sum"),
        )
        .reset_index()
    )

    g["tasa_baja2"] = (
        g["baja2"]
        / g["n"]
    )

    g["tasa_baja2_pct"] = (
        g["tasa_baja2"]
        * 100
    )

    g[MONTH] = mes

    patrones_rows.append(g)

patrones_mes = pd.concat(
    patrones_rows,
    ignore_index=True,
)

patrones_mes.to_csv(
    OUTPUT_DIR
    / "patrones6_por_mes.csv",
    index=False,
)


# ============================================================
# 5. Consolidado de patrones + estabilidad
# ============================================================

patron_total = (
    df.groupby(
        "patron6",
        observed=True,
    )
    .agg(
        n=("baja2", "size"),
        baja2=("baja2", "sum"),
        actividad_tarjetas=(
            "actividad_tarjetas",
            "first",
        ),
        actividad_trx=(
            "actividad_trx",
            "first",
        ),
        actividad_da=(
            "actividad_da",
            "first",
        ),
    )
    .reset_index()
)

patron_total[
    "tasa_baja2"
] = (
    patron_total["baja2"]
    / patron_total["n"]
)

patron_total[
    "tasa_baja2_pct"
] = (
    patron_total[
        "tasa_baja2"
    ] * 100
)

tasas_mes = (
    patrones_mes.pivot(
        index="patron6",
        columns=MONTH,
        values="tasa_baja2_pct",
    )
)

tasas_mes.columns = [
    f"tasa_{c}"
    for c in tasas_mes.columns
]

patron_total = (
    patron_total
    .merge(
        tasas_mes,
        on="patron6",
        how="left",
    )
)

cols_tasa = [
    f"tasa_{m}"
    for m in MESES_TARGET
]

patron_total[
    "tasa_media_meses"
] = (
    patron_total[
        cols_tasa
    ].mean(
        axis=1,
        skipna=True,
    )
)

patron_total[
    "tasa_std_meses"
] = (
    patron_total[
        cols_tasa
    ].std(
        axis=1,
        skipna=True,
    )
)

patron_total[
    "tasa_min_meses"
] = (
    patron_total[
        cols_tasa
    ].min(
        axis=1,
        skipna=True,
    )
)

patron_total[
    "tasa_max_meses"
] = (
    patron_total[
        cols_tasa
    ].max(
        axis=1,
        skipna=True,
    )
)

patron_total[
    "rango_tasa_meses"
] = (
    patron_total[
        "tasa_max_meses"
    ]
    - patron_total[
        "tasa_min_meses"
    ]
)

patron_total[
    "pct_dataset"
] = (
    patron_total["n"]
    / len(df)
    * 100
)

patron_total = (
    patron_total
    .sort_values(
        [
            "n",
            "tasa_baja2_pct",
        ],
        ascending=[
            False,
            False,
        ],
    )
)

patron_total.to_csv(
    OUTPUT_DIR
    / "patrones6_consolidado.csv",
    index=False,
)


# ============================================================
# Patrones frecuentes
# ============================================================

# Umbral para evitar interpretar patrones minúsculos
frecuentes = patron_total[
    patron_total["n"] >= 1000
].copy()

print()
print("=" * 95)
print("PATRONES FRECUENTES (n >= 1000)")
print("orden bits: debito_trx visa_trx master_trx cuenta_da visa_da master_da")
print("=" * 95)

cols_print = [
    "patron6",
    "n",
    "pct_dataset",
    "actividad_tarjetas",
    "actividad_trx",
    "actividad_da",
    "tasa_baja2_pct",
    *cols_tasa,
    "tasa_std_meses",
    "rango_tasa_meses",
]

print(
    frecuentes[
        cols_print
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


# ============================================================
# Patrones de mayor riesgo con soporte suficiente
# ============================================================

riesgo = (
    patron_total[
        patron_total["n"] >= 500
    ]
    .sort_values(
        "tasa_baja2_pct",
        ascending=False,
    )
)

print()
print("=" * 95)
print("TOP PATRONES POR RIESGO (n >= 500)")
print("=" * 95)

print(
    riesgo[
        cols_print
    ]
    .head(25)
    .to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


# ============================================================
# 6. ¿Mismo contador, distinto riesgo?
# ============================================================

mismo_conteo_rows = []

for k in range(7):

    z = patron_total[
        patron_total[
            "actividad_tarjetas"
        ].eq(k)
        &
        patron_total["n"].ge(500)
    ].copy()

    if len(z) == 0:
        continue

    mismo_conteo_rows.append({
        "actividad_tarjetas": k,
        "n_patrones_n500":
            len(z),
        "tasa_min_pct":
            z[
                "tasa_baja2_pct"
            ].min(),
        "tasa_max_pct":
            z[
                "tasa_baja2_pct"
            ].max(),
        "rango_riesgo_pp":
            (
                z[
                    "tasa_baja2_pct"
                ].max()
                -
                z[
                    "tasa_baja2_pct"
                ].min()
            ),
    })

mismo_conteo = pd.DataFrame(
    mismo_conteo_rows
)

mismo_conteo.to_csv(
    OUTPUT_DIR
    / "heterogeneidad_mismo_conteo.csv",
    index=False,
)

print()
print("=" * 95)
print("HETEROGENEIDAD DE RIESGO DENTRO DEL MISMO CONTEO")
print("(sólo patrones con n >= 500)")
print("=" * 95)

print(
    mismo_conteo.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


# ============================================================
# 7. Pares extremos con mismo número de canales
# ============================================================

extremos_rows = []

for k in range(1, 6):

    z = patron_total[
        patron_total[
            "actividad_tarjetas"
        ].eq(k)
        &
        patron_total["n"].ge(500)
    ].copy()

    if len(z) < 2:
        continue

    bajo = z.loc[
        z[
            "tasa_baja2_pct"
        ].idxmin()
    ]

    alto = z.loc[
        z[
            "tasa_baja2_pct"
        ].idxmax()
    ]

    extremos_rows.append({
        "actividad_tarjetas": k,
        "patron_menor_riesgo":
            bajo["patron6"],
        "n_menor_riesgo":
            int(bajo["n"]),
        "tasa_menor_riesgo_pct":
            bajo[
                "tasa_baja2_pct"
            ],
        "patron_mayor_riesgo":
            alto["patron6"],
        "n_mayor_riesgo":
            int(alto["n"]),
        "tasa_mayor_riesgo_pct":
            alto[
                "tasa_baja2_pct"
            ],
        "diferencia_pp":
            (
                alto[
                    "tasa_baja2_pct"
                ]
                -
                bajo[
                    "tasa_baja2_pct"
                ]
            ),
    })

extremos = pd.DataFrame(
    extremos_rows
)

extremos.to_csv(
    OUTPUT_DIR
    / "extremos_mismo_conteo.csv",
    index=False,
)

print()
print("=" * 95)
print("PATRONES EXTREMOS CON EL MISMO NUMERO DE CANALES")
print("=" * 95)

print(
    extremos.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


# ============================================================
# Metadata
# ============================================================

metadata = {
    "dataset":
        str(DATASET),
    "meses_target":
        MESES_TARGET,
    "variables":
        VARS,
    "orden_patron":
        aliases,
    "transacciones":
        TRX,
    "debitos_automaticos":
        DA,
    "runtime_segundos":
        time.time() - t0,
}

with open(
    OUTPUT_DIR
    / "metadata_z553.json",
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
print("Z553 FINALIZADO")
print("Output:", OUTPUT_DIR)
