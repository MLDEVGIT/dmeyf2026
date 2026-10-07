from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    roc_auc_score,
)


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
    "competencia_01/actividad_familias_ablation_z548"
)

ID_COL = "numero_de_cliente"
MONTH_COL = "foto_mes"
TARGET_COL = "clase_ternaria"
LAG_AVAILABLE_COL = "lag1_disponible"

TRAIN_MONTHS = [202104, 202105]
TEST_MONTH = 202106

SEED = 290497

PESOS_TEMPORALES = {
    202104: 0.75,
    202105: 1.00,
}

CUTS = [
    8000,
    9500,
    10000,
    10500,
    11000,
    11500,
    12000,
    12500,
    13000,
    13500,
    14000,
    14500,
    15000,
]

ZONA = [
    10000,
    10500,
    11000,
    11500,
    12000,
    12500,
    13000,
    13500,
    14000,
]

PARAMS = {
    "objective": "binary",
    "n_estimators": 1200,
    "learning_rate": 0.02,
    "num_leaves": 750,
    "max_depth": -1,
    "min_child_samples": 5000,
    "max_bin": 31,
    "colsample_bytree": 0.5,
    "subsample": 1.0,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": -1,
    "importance_type": "gain",
}


FAMILIAS = {
    "tarjetas": [
        "ctarjeta_debito_transacciones",
        "ctarjeta_visa_transacciones",
        "ctarjeta_master_transacciones",
        "ccuenta_debitos_automaticos",
        "ctarjeta_visa_debitos_automaticos",
        "ctarjeta_master_debitos_automaticos",
    ],
    "payroll": [
        "cpayroll_trx",
        "cpayroll2_trx",
    ],
}


# ============================================================
# Utilidades
# ============================================================

def identificar_features(df):
    excluir = {
        ID_COL,
        MONTH_COL,
        TARGET_COL,
        LAG_AVAILABLE_COL,
    }

    delta = [
        c for c in df.columns
        if c.endswith("_delta_lag1")
    ]

    lag = [
        c for c in df.columns
        if (
            c.endswith("_lag1")
            and not c.endswith("_delta_lag1")
        )
    ]

    originales = [
        c for c in df.columns
        if (
            c not in excluir
            and not c.endswith("_lag1")
            and not c.endswith("_delta_lag1")
        )
    ]

    return (
        originales
        + lag
        + delta
        + [LAG_AVAILABLE_COL]
    )


def target_binario(s):
    return (
        s.eq("BAJA+2")
        .astype(np.int8)
    )


def crear_familia(df, nombre, variables):
    actual = (
        df[variables]
        .gt(0)
        .sum(axis=1)
        .astype("float32")
    )

    lag_cols = [
        f"{c}_lag1"
        for c in variables
    ]

    faltantes = [
        c for c in lag_cols
        if c not in df.columns
    ]

    if faltantes:
        raise ValueError(
            f"{nombre}: faltan {faltantes}"
        )

    lag1 = (
        df[lag_cols]
        .gt(0)
        .sum(axis=1)
        .astype("float32")
    )

    disponible = (
        df[LAG_AVAILABLE_COL]
        .fillna(0)
        .eq(1)
    )

    lag1 = lag1.where(
        disponible,
        np.nan,
    )

    delta = actual - lag1

    c_actual = f"actividad_{nombre}"
    c_lag = f"actividad_{nombre}_lag1"
    c_delta = f"actividad_{nombre}_delta_lag1"

    df[c_actual] = actual
    df[c_lag] = lag1
    df[c_delta] = delta

    return {
        "actual": c_actual,
        "lag1": c_lag,
        "delta": c_delta,
    }


def ganancia_por_corte(y, prob):
    orden = np.argsort(-prob)
    y_ord = np.asarray(y)[orden]
    acum = np.cumsum(y_ord)

    rows = []

    for n in CUTS:
        n_real = min(n, len(y_ord))

        positivos = int(
            acum[n_real - 1]
        )

        negativos = (
            n_real - positivos
        )

        gain = (
            positivos * 1_072_500
            - negativos * 27_500
        )

        rows.append({
            "cut": n_real,
            "positivos": positivos,
            "negativos": negativos,
            "ganancia": gain,
            "ganancia_millones":
                gain / 1_000_000,
        })

    return pd.DataFrame(rows)


# ============================================================
# Lectura
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

t0 = time.time()

print("=" * 80)
print("Z548 - ABLACION TARJETAS / PAYROLL")
print("=" * 80)

df = pd.read_parquet(DATASET)

print(f"Filas: {len(df):,}")
print(f"Columnas: {len(df.columns):,}")

features_base = identificar_features(df)

if len(features_base) != 457:
    raise ValueError(
        f"FULL457 esperado; "
        f"encontrado {len(features_base)}"
    )

print(
    f"Features baseline: "
    f"{len(features_base)}"
)


# ============================================================
# Construcción de FE
# ============================================================

features_familia = {}

for nombre, variables in FAMILIAS.items():

    faltantes = [
        c for c in variables
        if c not in df.columns
    ]

    if faltantes:
        raise ValueError(
            f"{nombre}: faltan {faltantes}"
        )

    features_familia[nombre] = (
        crear_familia(
            df,
            nombre,
            variables,
        )
    )

    print()
    print(nombre.upper())

    for k, c in (
        features_familia[nombre]
        .items()
    ):
        print(
            f"  {k:<6} "
            f"{c:<38} "
            f"mean={df[c].mean():.4f} "
            f"NA={df[c].isna().mean():.4%}"
        )


# ============================================================
# Train / test
# ============================================================

train = df[
    df[MONTH_COL].isin(
        TRAIN_MONTHS
    )
].copy()

test = df[
    df[MONTH_COL] == TEST_MONTH
].copy()

y_train = target_binario(
    train[TARGET_COL]
)

y_test = target_binario(
    test[TARGET_COL]
)

weights = (
    train[MONTH_COL]
    .map(PESOS_TEMPORALES)
    .astype(float)
    .to_numpy()
)

if not np.isfinite(weights).all():
    raise ValueError(
        "Pesos inválidos"
    )

print()
print(
    f"Train: {len(train):,} "
    f"| BAJA+2={int(y_train.sum()):,}"
)

print(
    f"Test:  {len(test):,} "
    f"| BAJA+2={int(y_test.sum()):,}"
)


# ============================================================
# Configuraciones
# ============================================================

t = features_familia["tarjetas"]
p = features_familia["payroll"]

EXPERIMENTOS = [
    (
        "BASE",
        features_base,
    ),

    (
        "TARJETAS_ACTUAL",
        features_base
        + [t["actual"]],
    ),

    (
        "TARJETAS_ACTUAL_LAG",
        features_base
        + [
            t["actual"],
            t["lag1"],
        ],
    ),

    (
        "TARJETAS_ACTUAL_LAG_DELTA",
        features_base
        + [
            t["actual"],
            t["lag1"],
            t["delta"],
        ],
    ),

    (
        "PAYROLL_ACTUAL",
        features_base
        + [p["actual"]],
    ),

    (
        "PAYROLL_ACTUAL_LAG",
        features_base
        + [
            p["actual"],
            p["lag1"],
        ],
    ),

    (
        "PAYROLL_ACTUAL_LAG_DELTA",
        features_base
        + [
            p["actual"],
            p["lag1"],
            p["delta"],
        ],
    ),

    (
        "TARJETAS_PAYROLL_ACTUAL",
        features_base
        + [
            t["actual"],
            p["actual"],
        ],
    ),
]

print()
print(
    f"Experimentos: {len(EXPERIMENTOS)}"
)


# ============================================================
# Entrenamiento
# ============================================================

metricas_rows = []
curvas = []
importancias = []

pred = pd.DataFrame({
    ID_COL:
        test[ID_COL].to_numpy(),
    TARGET_COL:
        test[TARGET_COL].to_numpy(),
})

for i, (
    nombre,
    features,
) in enumerate(
    EXPERIMENTOS,
    start=1,
):

    print()
    print("=" * 80)
    print(
        f"[{i}/{len(EXPERIMENTOS)}] "
        f"{nombre}"
    )
    print("=" * 80)

    print(
        f"Features: {len(features)}"
    )

    inicio = time.time()

    model = LGBMClassifier(
        **PARAMS
    )

    model.fit(
        train[features],
        y_train,
        sample_weight=weights,
    )

    prob = model.predict_proba(
        test[features]
    )[:, 1]

    segundos = (
        time.time() - inicio
    )

    auc = roc_auc_score(
        y_test,
        prob,
    )

    ap = average_precision_score(
        y_test,
        prob,
    )

    ll = log_loss(
        y_test,
        prob,
        labels=[0, 1],
    )

    curva = ganancia_por_corte(
        y_test,
        prob,
    )

    mejor = curva.loc[
        curva["ganancia"].idxmax()
    ]

    print(
        f"AUC={auc:.6f} "
        f"AP={ap:.6f} "
        f"LL={ll:.6f}"
    )

    print(
        f"Best N="
        f"{int(mejor['cut']):,} "
        f"| pos="
        f"{int(mejor['positivos'])} "
        f"| gain="
        f"{mejor['ganancia_millones']:.2f} M"
    )

    print()
    print(
        curva[
            [
                "cut",
                "positivos",
                "ganancia_millones",
            ]
        ].to_string(
            index=False
        )
    )

    metricas_rows.append({
        "modelo": nombre,
        "n_features": len(features),
        "auc": auc,
        "average_precision": ap,
        "logloss": ll,
        "best_cut":
            int(mejor["cut"]),
        "best_positivos":
            int(mejor["positivos"]),
        "best_gain_millones":
            float(
                mejor[
                    "ganancia_millones"
                ]
            ),
        "segundos": segundos,
    })

    curva["modelo"] = nombre
    curvas.append(curva)

    imp = pd.DataFrame({
        "modelo": nombre,
        "feature": features,
        "importance_gain":
            model.feature_importances_,
    })

    imp["rank_importance"] = (
        imp["importance_gain"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    importancias.append(imp)

    pred[
        f"prob_{nombre}"
    ] = prob


# ============================================================
# Resultados
# ============================================================

metricas = pd.DataFrame(
    metricas_rows
)

curvas = pd.concat(
    curvas,
    ignore_index=True,
)

importancias = pd.concat(
    importancias,
    ignore_index=True,
)

metricas.to_csv(
    OUTPUT_DIR
    / "metricas_modelos.csv",
    index=False,
)

curvas.to_csv(
    OUTPUT_DIR
    / "ganancia_por_corte.csv",
    index=False,
)

importancias.to_csv(
    OUTPUT_DIR
    / "importancias.csv",
    index=False,
)

pred.to_csv(
    OUTPUT_DIR
    / "predicciones_202106.csv",
    index=False,
)


# ============================================================
# Delta contra baseline
# ============================================================

tabla = curvas.pivot(
    index="cut",
    columns="modelo",
    values="ganancia_millones",
)

delta = tabla.subtract(
    tabla["BASE"],
    axis=0,
)

delta.to_csv(
    OUTPUT_DIR
    / "delta_gain_vs_base.csv"
)

print()
print("=" * 80)
print("RESUMEN METRICAS")
print("=" * 80)

print(
    metricas[
        [
            "modelo",
            "n_features",
            "auc",
            "average_precision",
            "logloss",
            "best_cut",
            "best_gain_millones",
        ]
    ].to_string(
        index=False
    )
)

print()
print("=" * 80)
print("DELTA GAIN VS BASE - MILLONES")
print("=" * 80)

print(
    delta.to_string(
        float_format=lambda x: f"{x:+.2f}"
    )
)


# ============================================================
# Robustez zona 10k-14k
# ============================================================

rows = []

for modelo in delta.columns:

    if modelo == "BASE":
        continue

    x = delta.loc[
        ZONA,
        modelo,
    ]

    rows.append({
        "modelo": modelo,
        "delta_medio_M":
            x.mean(),
        "delta_mediano_M":
            x.median(),
        "delta_min_M":
            x.min(),
        "delta_max_M":
            x.max(),
        "cortes_positivos":
            int((x > 0).sum()),
        "cortes_neutros":
            int((x == 0).sum()),
        "cortes_negativos":
            int((x < 0).sum()),
    })

robustez = (
    pd.DataFrame(rows)
    .sort_values(
        [
            "delta_medio_M",
            "cortes_positivos",
        ],
        ascending=[
            False,
            False,
        ],
    )
)

robustez.to_csv(
    OUTPUT_DIR
    / "robustez_10k_14k.csv",
    index=False,
)

print()
print("=" * 80)
print("ROBUSTEZ 10K-14K")
print("=" * 80)

print(
    robustez.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}",
    )
)


# ============================================================
# Comparaciones incrementales
# ============================================================

print()
print("=" * 80)
print("EFECTOS INCREMENTALES 10K-14K")
print("=" * 80)

comparaciones = [
    (
        "TARJETAS: actual vs BASE",
        "BASE",
        "TARJETAS_ACTUAL",
    ),
    (
        "TARJETAS: +lag",
        "TARJETAS_ACTUAL",
        "TARJETAS_ACTUAL_LAG",
    ),
    (
        "TARJETAS: +delta",
        "TARJETAS_ACTUAL_LAG",
        "TARJETAS_ACTUAL_LAG_DELTA",
    ),
    (
        "PAYROLL: actual vs BASE",
        "BASE",
        "PAYROLL_ACTUAL",
    ),
    (
        "PAYROLL: +lag",
        "PAYROLL_ACTUAL",
        "PAYROLL_ACTUAL_LAG",
    ),
    (
        "PAYROLL: +delta",
        "PAYROLL_ACTUAL_LAG",
        "PAYROLL_ACTUAL_LAG_DELTA",
    ),
    (
        "PAYROLL actual sobre TARJETAS actual",
        "TARJETAS_ACTUAL",
        "TARJETAS_PAYROLL_ACTUAL",
    ),
]

comparaciones_rows = []

for etiqueta, desde, hasta in comparaciones:

    efecto = (
        tabla.loc[ZONA, hasta]
        - tabla.loc[ZONA, desde]
    )

    row = {
        "comparacion": etiqueta,
        "delta_medio_M":
            efecto.mean(),
        "delta_mediano_M":
            efecto.median(),
        "delta_min_M":
            efecto.min(),
        "delta_max_M":
            efecto.max(),
        "cortes_positivos":
            int((efecto > 0).sum()),
        "cortes_neutros":
            int((efecto == 0).sum()),
        "cortes_negativos":
            int((efecto < 0).sum()),
    }

    comparaciones_rows.append(
        row
    )

comparaciones_df = pd.DataFrame(
    comparaciones_rows
)

print(
    comparaciones_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}",
    )
)

comparaciones_df.to_csv(
    OUTPUT_DIR
    / "efectos_incrementales.csv",
    index=False,
)


# ============================================================
# Importancia sólo FE nuevas
# ============================================================

nuevas = {
    t["actual"],
    t["lag1"],
    t["delta"],
    p["actual"],
    p["lag1"],
    p["delta"],
}

imp_nuevas = (
    importancias[
        importancias[
            "feature"
        ].isin(nuevas)
    ]
    .sort_values(
        [
            "modelo",
            "rank_importance",
        ]
    )
)

imp_nuevas.to_csv(
    OUTPUT_DIR
    / "importancias_features_nuevas.csv",
    index=False,
)

print()
print("=" * 80)
print("IMPORTANCIA FEATURES NUEVAS")
print("=" * 80)

print(
    imp_nuevas[
        [
            "modelo",
            "feature",
            "importance_gain",
            "rank_importance",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# Metadata
# ============================================================

metadata = {
    "dataset": str(DATASET),
    "train_months": TRAIN_MONTHS,
    "test_month": TEST_MONTH,
    "seed": SEED,
    "pesos_temporales":
        PESOS_TEMPORALES,
    "cuts": CUTS,
    "zona_robustez": ZONA,
    "familias": FAMILIAS,
    "n_features_base":
        len(features_base),
    "runtime_minutos":
        (time.time() - t0)
        / 60.0,
}

with open(
    OUTPUT_DIR
    / "metadata_z548.json",
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
    f"{(time.time() - t0) / 60:.2f} min"
)

print()
print("Z548 FINALIZADO")
print("Output:", OUTPUT_DIR)
