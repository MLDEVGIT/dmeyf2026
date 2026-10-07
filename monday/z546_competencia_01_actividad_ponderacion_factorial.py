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
    "competencia_01/actividad_ponderacion_z546"
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

TRANSACTION_VARS = [
    "ctarjeta_debito_transacciones",
    "ctarjeta_visa_transacciones",
    "ctarjeta_master_transacciones",
    "cpayroll_trx",
    "cpayroll2_trx",
    "ccuenta_debitos_automaticos",
    "ctarjeta_visa_debitos_automaticos",
    "ctarjeta_master_debitos_automaticos",
    "cpagodeservicios",
    "cpagomiscuentas",
    "ccajeros_propios_descuentos",
    "ctarjeta_visa_descuentos",
    "ctarjeta_master_descuentos",
    "ccomisiones_mantenimiento",
    "ccomisiones_otras",
    "cforex_buy",
    "cforex_sell",
    "ctransferencias_recibidas",
    "ctransferencias_emitidas",
    "cextraccion_autoservicio",
    "ccheques_depositados",
    "ccheques_emitidos",
    "ccheques_depositados_rechazados",
    "ccheques_emitidos_rechazados",
    "ccallcenter_transacciones",
    "chomebanking_transacciones",
    "ccajas_transacciones",
    "ccajas_consultas",
    "ccajas_depositos",
    "ccajas_extracciones",
    "ccajas_otras",
    "catm_trx",
    "catm_trx_other",
    "cmobile_app_trx",
]


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

    features_historicas = (
        features_originales
        + features_lag
        + features_delta
        + [LAG_AVAILABLE_COL]
    )

    return features_historicas


def target_binario(s):
    return (
        s.eq("BAJA+2")
        .astype(np.int8)
    )


def ganancia_por_corte(y, prob, cortes):
    orden = np.argsort(-prob)
    y_ord = np.asarray(y)[orden]

    acumulada_positivos = np.cumsum(y_ord)

    rows = []

    for n in cortes:
        n_real = min(n, len(y_ord))

        positivos = int(
            acumulada_positivos[n_real - 1]
        )

        negativos = n_real - positivos

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


# ============================================================
# Lectura
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

t0 = time.time()

print("=" * 80)
print("Z546 - ACTIVIDAD x PONDERACION TEMPORAL")
print("=" * 80)

print("\nLeyendo dataset...")

df = pd.read_parquet(DATASET)

print(f"Filas: {len(df):,}")
print(f"Columnas: {len(df.columns):,}")

features_base = identificar_features(df)

print(
    f"Features FULL: {len(features_base):,}"
)

if len(features_base) != 457:
    raise ValueError(
        f"Se esperaban 457 features; "
        f"se encontraron {len(features_base)}."
    )

faltantes = [
    c
    for c in TRANSACTION_VARS
    if c not in df.columns
]

if faltantes:
    raise ValueError(
        f"Faltan variables transaccionales: {faltantes}"
    )

# ------------------------------------------------------------
# actividad_canales
# ------------------------------------------------------------

df["actividad_canales"] = (
    df[TRANSACTION_VARS]
    .gt(0)
    .sum(axis=1)
    .astype(np.int16)
)

features_actividad = (
    features_base
    + ["actividad_canales"]
)

print(
    f"Features ACTUAL: {len(features_actividad):,}"
)

# ------------------------------------------------------------
# Train / test
# ------------------------------------------------------------

train = df[
    df[MONTH_COL].isin(TRAIN_MONTHS)
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

print()
print(f"Train filas: {len(train):,}")
print(
    f"Train BAJA+2: {int(y_train.sum()):,}"
)
print(f"Test filas: {len(test):,}")
print(
    f"Test BAJA+2: {int(y_test.sum()):,}"
)

# ------------------------------------------------------------
# Configuraciones factoriales
# ------------------------------------------------------------

experimentos = [
    {
        "nombre": "A_BASE_SIN_PESO",
        "features": features_base,
        "ponderado": False,
    },
    {
        "nombre": "B_ACTIVIDAD_SIN_PESO",
        "features": features_actividad,
        "ponderado": False,
    },
    {
        "nombre": "C_BASE_PONDERADO",
        "features": features_base,
        "ponderado": True,
    },
    {
        "nombre": "D_ACTIVIDAD_PONDERADO",
        "features": features_actividad,
        "ponderado": True,
    },
]

resultados_metricas = []
resultados_cortes = []
importancias = []

predicciones = {
    ID_COL: test[ID_COL].to_numpy(),
    TARGET_COL: test[TARGET_COL].to_numpy(),
}

for exp in experimentos:

    nombre = exp["nombre"]
    features = exp["features"]
    ponderado = exp["ponderado"]

    print()
    print("=" * 80)
    print(nombre)
    print("=" * 80)

    X_train = train[features]
    X_test = test[features]

    if ponderado:
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
                f"{nombre}: pesos inválidos."
            )

        print(
            "Pesos:",
            PESOS_TEMPORALES,
        )
    else:
        sample_weight = None
        print("Pesos: ninguno")

    modelo = LGBMClassifier(
        **PARAMS
    )

    inicio = time.time()

    modelo.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
    )

    prob = modelo.predict_proba(
        X_test
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
        curva["ganancia"].idxmax()
    ]

    print(
        f"AUC={auc:.6f} "
        f"AP={ap:.6f} "
        f"LL={ll:.6f}"
    )

    print(
        f"Mejor corte probado: "
        f"{int(mejor['cut']):,} "
        f"| positivos="
        f"{int(mejor['positivos']):,} "
        f"| ganancia="
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
        ].to_string(index=False)
    )

    resultados_metricas.append(
        {
            "modelo": nombre,
            "n_features": len(features),
            "ponderado": ponderado,
            "auc": auc,
            "average_precision": ap,
            "logloss": ll,
            "best_cut": int(
                mejor["cut"]
            ),
            "best_positivos": int(
                mejor["positivos"]
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

    resultados_cortes.append(
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
        imp["importance_gain"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    importancias.append(imp)

    predicciones[
        f"prob_{nombre}"
    ] = prob


# ============================================================
# Resultados
# ============================================================

metricas = pd.DataFrame(
    resultados_metricas
)

cortes = pd.concat(
    resultados_cortes,
    ignore_index=True,
)

imp = pd.concat(
    importancias,
    ignore_index=True,
)

pred = pd.DataFrame(
    predicciones
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

imp.to_csv(
    OUTPUT_DIR
    / "importancias.csv",
    index=False,
)

pred.to_csv(
    OUTPUT_DIR
    / "predicciones_202106.csv",
    index=False,
)

# ------------------------------------------------------------
# Tabla comparativa por corte
# ------------------------------------------------------------

tabla_gain = cortes.pivot(
    index="cut",
    columns="modelo",
    values="ganancia_millones",
)

tabla_pos = cortes.pivot(
    index="cut",
    columns="modelo",
    values="positivos",
)

print()
print("=" * 80)
print("RESUMEN METRICAS")
print("=" * 80)

print(
    metricas[
        [
            "modelo",
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
print("GANANCIA POR CORTE - MILLONES")
print("=" * 80)

print(
    tabla_gain.to_string(
        float_format=lambda x: f"{x:.2f}"
    )
)

print()
print("=" * 80)
print("POSITIVOS POR CORTE")
print("=" * 80)

print(
    tabla_pos.to_string()
)

# ------------------------------------------------------------
# Efectos factoriales importantes
# ------------------------------------------------------------

print()
print("=" * 80)
print("EFECTO DE ACTIVIDAD")
print("B-A = actividad sin ponderacion")
print("D-C = actividad con ponderacion")
print("=" * 80)

for cut in CUTS:

    g = (
        cortes[
            cortes["cut"] == cut
        ]
        .set_index("modelo")[
            "ganancia_millones"
        ]
    )

    efecto_sin = (
        g["B_ACTIVIDAD_SIN_PESO"]
        - g["A_BASE_SIN_PESO"]
    )

    efecto_con = (
        g["D_ACTIVIDAD_PONDERADO"]
        - g["C_BASE_PONDERADO"]
    )

    print(
        f"N={cut:5d} "
        f"| B-A={efecto_sin:+7.2f} M "
        f"| D-C={efecto_con:+7.2f} M"
    )

# ------------------------------------------------------------
# Importancia de actividad
# ------------------------------------------------------------

print()
print("=" * 80)
print("IMPORTANCIA ACTIVIDAD_CANALES")
print("=" * 80)

actividad_imp = (
    imp[
        imp["feature"]
        == "actividad_canales"
    ][
        [
            "modelo",
            "importance_gain",
            "rank_importance",
        ]
    ]
    .sort_values(
        "rank_importance"
    )
)

print(
    actividad_imp.to_string(
        index=False
    )
)

# ------------------------------------------------------------
# Metadata
# ------------------------------------------------------------

metadata = {
    "dataset": str(DATASET),
    "train_months": TRAIN_MONTHS,
    "test_month": TEST_MONTH,
    "seed": SEED,
    "pesos_temporales":
        PESOS_TEMPORALES,
    "cuts": CUTS,
    "n_features_base":
        len(features_base),
    "n_features_actividad":
        len(features_actividad),
    "runtime_minutos":
        (time.time() - t0) / 60.0,
}

with open(
    OUTPUT_DIR
    / "metadata_z546.json",
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
print("Z546 FINALIZADO")
print("Output:", OUTPUT_DIR)
