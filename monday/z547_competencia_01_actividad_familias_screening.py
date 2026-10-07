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
    "competencia_01/actividad_familias_z547"
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


# ============================================================
# Familias
# ============================================================

FAMILIAS = {
    "tarjetas": [
        "ctarjeta_debito_transacciones",
        "ctarjeta_visa_transacciones",
        "ctarjeta_master_transacciones",
        "ccuenta_debitos_automaticos",
        "ctarjeta_visa_debitos_automaticos",
        "ctarjeta_master_debitos_automaticos",
    ],

    "digital": [
        "chomebanking_transacciones",
        "cmobile_app_trx",
    ],

    "movimientos": [
        "ctransferencias_recibidas",
        "ctransferencias_emitidas",
        "cextraccion_autoservicio",
        "catm_trx",
        "catm_trx_other",
        "cpagomiscuentas",
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

    features_delta = [
        c
        for c in df.columns
        if c.endswith("_delta_lag1")
    ]

    features_lag = [
        c
        for c in df.columns
        if (
            c.endswith("_lag1")
            and not c.endswith("_delta_lag1")
        )
    ]

    features_originales = [
        c
        for c in df.columns
        if (
            c not in excluir
            and not c.endswith("_lag1")
            and not c.endswith("_delta_lag1")
        )
    ]

    return (
        features_originales
        + features_lag
        + features_delta
        + [LAG_AVAILABLE_COL]
    )


def target_binario(s):
    return (
        s.eq("BAJA+2")
        .astype(np.int8)
    )


def ganancia_por_corte(y, prob, cortes):
    orden = np.argsort(-prob)

    y_ord = (
        np.asarray(y)
        [orden]
    )

    acumulada = np.cumsum(y_ord)

    rows = []

    for n in cortes:
        n_real = min(
            n,
            len(y_ord),
        )

        positivos = int(
            acumulada[n_real - 1]
        )

        negativos = (
            n_real
            - positivos
        )

        ganancia = (
            positivos * 1_072_500
            - negativos * 27_500
        )

        rows.append(
            {
                "cut": n_real,
                "positivos": positivos,
                "negativos": negativos,
                "ganancia": ganancia,
                "ganancia_millones":
                    ganancia / 1_000_000,
            }
        )

    return pd.DataFrame(rows)


def crear_familia(
    df,
    nombre,
    variables,
):
    """
    Crea:
      actividad_<familia>
      actividad_<familia>_lag1
      actividad_<familia>_delta_lag1

    Actividad = cantidad de componentes > 0.
    """

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
        c
        for c in lag_cols
        if c not in df.columns
    ]

    if faltantes:
        raise ValueError(
            f"{nombre}: faltan lags: "
            f"{faltantes}"
        )

    lag1 = (
        df[lag_cols]
        .gt(0)
        .sum(axis=1)
        .astype("float32")
    )

    # Sin historia disponible, el lag agregado
    # no debe interpretarse como cero.
    disponible = (
        df[LAG_AVAILABLE_COL]
        .fillna(0)
        .eq(1)
    )

    lag1 = lag1.where(
        disponible,
        np.nan,
    )

    delta = (
        actual
        - lag1
    )

    col_actual = (
        f"actividad_{nombre}"
    )

    col_lag = (
        f"actividad_{nombre}_lag1"
    )

    col_delta = (
        f"actividad_{nombre}_delta_lag1"
    )

    df[col_actual] = actual
    df[col_lag] = lag1
    df[col_delta] = delta

    return [
        col_actual,
        col_lag,
        col_delta,
    ]


# ============================================================
# Inicio
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

t0 = time.time()

print("=" * 80)
print("Z547 - SCREENING FAMILIAS DE ACTIVIDAD")
print("=" * 80)

print("\nLeyendo dataset...")

df = pd.read_parquet(
    DATASET
)

print(
    f"Filas: {len(df):,}"
)

print(
    f"Columnas iniciales: "
    f"{len(df.columns):,}"
)

features_base = (
    identificar_features(df)
)

if len(features_base) != 457:
    raise ValueError(
        f"Se esperaban 457 features "
        f"base; hay {len(features_base)}."
    )

print(
    f"Features FULL457: "
    f"{len(features_base)}"
)


# ============================================================
# Validación de variables
# ============================================================

for nombre, variables in FAMILIAS.items():

    faltantes = [
        c
        for c in variables
        if c not in df.columns
    ]

    if faltantes:
        raise ValueError(
            f"Familia {nombre}: "
            f"faltan {faltantes}"
        )


# ============================================================
# Construcción FE -> historia
# ============================================================

features_familias = {}

for nombre, variables in FAMILIAS.items():

    nuevas = crear_familia(
        df,
        nombre,
        variables,
    )

    features_familias[
        nombre
    ] = nuevas

    print()
    print(
        f"{nombre}: "
        f"{len(variables)} componentes"
    )

    print(
        "  "
        + ", ".join(nuevas)
    )


# ============================================================
# Auditoría rápida de las nuevas features
# ============================================================

print()
print("=" * 80)
print("AUDITORIA FEATURES NUEVAS")
print("=" * 80)

for nombre, nuevas in (
    features_familias.items()
):
    for c in nuevas:
        print(
            f"{c:<42} "
            f"NA={df[c].isna().mean():.4%} "
            f"mean={df[c].mean():.4f}"
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
    df[MONTH_COL]
    == TEST_MONTH
].copy()

y_train = target_binario(
    train[TARGET_COL]
)

y_test = target_binario(
    test[TARGET_COL]
)

sample_weight = (
    train[MONTH_COL]
    .map(PESOS_TEMPORALES)
    .astype(float)
    .to_numpy()
)

if not np.isfinite(
    sample_weight
).all():
    raise ValueError(
        "Hay pesos inválidos."
    )

print()
print(
    f"Train filas: {len(train):,}"
)

print(
    f"Train BAJA+2: "
    f"{int(y_train.sum()):,}"
)

print(
    f"Test filas: {len(test):,}"
)

print(
    f"Test BAJA+2: "
    f"{int(y_test.sum()):,}"
)


# ============================================================
# Experimentos
# ============================================================

todas_familias = []

for nuevas in (
    features_familias.values()
):
    todas_familias.extend(
        nuevas
    )

EXPERIMENTOS = [
    (
        "BASE_PONDERADO",
        features_base,
    ),
    (
        "TARJETAS",
        features_base
        + features_familias[
            "tarjetas"
        ],
    ),
    (
        "DIGITAL",
        features_base
        + features_familias[
            "digital"
        ],
    ),
    (
        "MOVIMIENTOS",
        features_base
        + features_familias[
            "movimientos"
        ],
    ),
    (
        "PAYROLL",
        features_base
        + features_familias[
            "payroll"
        ],
    ),
    (
        "TODAS_FAMILIAS",
        features_base
        + todas_familias,
    ),
]

# Son 6 fits: baseline + 4 familias + conjunto.
print()
print(
    f"Experimentos: "
    f"{len(EXPERIMENTOS)}"
)


# ============================================================
# Modelado
# ============================================================

metricas_rows = []
cortes_all = []
importancias_all = []

predicciones = pd.DataFrame(
    {
        ID_COL:
            test[ID_COL].to_numpy(),

        TARGET_COL:
            test[TARGET_COL].to_numpy(),
    }
)

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

    modelo = LGBMClassifier(
        **PARAMS
    )

    modelo.fit(
        train[features],
        y_train,
        sample_weight=sample_weight,
    )

    prob = modelo.predict_proba(
        test[features]
    )[:, 1]

    segundos = (
        time.time()
        - inicio
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
        CUTS,
    )

    mejor = curva.loc[
        curva[
            "ganancia"
        ].idxmax()
    ]

    print(
        f"AUC={auc:.6f} "
        f"AP={ap:.6f} "
        f"LL={ll:.6f}"
    )

    print(
        f"Best: "
        f"N={int(mejor['cut']):,} "
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

    metricas_rows.append(
        {
            "modelo": nombre,
            "n_features":
                len(features),
            "auc": auc,
            "average_precision": ap,
            "logloss": ll,
            "best_cut":
                int(mejor["cut"]),
            "best_positivos":
                int(
                    mejor[
                        "positivos"
                    ]
                ),
            "best_gain_millones":
                float(
                    mejor[
                        "ganancia_millones"
                    ]
                ),
            "segundos": segundos,
        }
    )

    curva["modelo"] = nombre

    cortes_all.append(
        curva
    )

    imp = pd.DataFrame(
        {
            "modelo": nombre,
            "feature": features,
            "importance_gain":
                modelo.feature_importances_,
        }
    )

    imp["rank_importance"] = (
        imp[
            "importance_gain"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    importancias_all.append(
        imp
    )

    predicciones[
        f"prob_{nombre}"
    ] = prob


# ============================================================
# Consolidación
# ============================================================

metricas = pd.DataFrame(
    metricas_rows
)

cortes = pd.concat(
    cortes_all,
    ignore_index=True,
)

importancias = pd.concat(
    importancias_all,
    ignore_index=True,
)

metricas.to_csv(
    OUTPUT_DIR
    / "metricas_modelos.csv",
    index=False,
)

cortes.to_csv(
    OUTPUT_DIR
    / "ganancia_por_corte.csv",
    index=False,
)

importancias.to_csv(
    OUTPUT_DIR
    / "importancias.csv",
    index=False,
)

predicciones.to_csv(
    OUTPUT_DIR
    / "predicciones_202106.csv",
    index=False,
)


# ============================================================
# Comparación contra baseline
# ============================================================

tabla_gain = cortes.pivot(
    index="cut",
    columns="modelo",
    values="ganancia_millones",
)

baseline = (
    tabla_gain[
        "BASE_PONDERADO"
    ]
)

delta_gain = (
    tabla_gain
    .subtract(
        baseline,
        axis=0,
    )
)

print()
print("=" * 80)
print("RESUMEN")
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
print("DELTA GANANCIA VS BASELINE - MILLONES")
print("=" * 80)

print(
    delta_gain.to_string(
        float_format=lambda x: f"{x:+.2f}"
    )
)


# ============================================================
# Robustez en zona 10k-14k
# ============================================================

zona = [
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

print()
print("=" * 80)
print("RESUMEN ZONA 10K-14K VS BASELINE")
print("=" * 80)

zona_rows = []

for modelo in (
    delta_gain.columns
):

    if modelo == "BASE_PONDERADO":
        continue

    x = (
        delta_gain
        .loc[zona, modelo]
    )

    zona_rows.append(
        {
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
        }
    )

zona_df = (
    pd.DataFrame(
        zona_rows
    )
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

print(
    zona_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}",
    )
)

zona_df.to_csv(
    OUTPUT_DIR
    / "robustez_zona_10k_14k.csv",
    index=False,
)


# ============================================================
# Importancia de features nuevas
# ============================================================

nuevas_set = set(
    todas_familias
)

imp_nuevas = (
    importancias[
        importancias[
            "feature"
        ].isin(
            nuevas_set
        )
    ]
    .sort_values(
        [
            "modelo",
            "rank_importance",
        ]
    )
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

imp_nuevas.to_csv(
    OUTPUT_DIR
    / "importancias_features_nuevas.csv",
    index=False,
)


# ============================================================
# Metadata
# ============================================================

metadata = {
    "dataset":
        str(DATASET),
    "train_months":
        TRAIN_MONTHS,
    "test_month":
        TEST_MONTH,
    "seed":
        SEED,
    "pesos_temporales":
        PESOS_TEMPORALES,
    "cuts":
        CUTS,
    "familias":
        FAMILIAS,
    "n_features_base":
        len(features_base),
    "runtime_minutos":
        (time.time() - t0)
        / 60.0,
}

with open(
    OUTPUT_DIR
    / "metadata_z547.json",
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
print("Z547 FINALIZADO")
print("Output:", OUTPUT_DIR)
