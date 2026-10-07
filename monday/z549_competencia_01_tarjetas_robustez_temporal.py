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
    "competencia_01/tarjetas_robustez_temporal_z549"
)

ID_COL = "numero_de_cliente"
MONTH_COL = "foto_mes"
TARGET_COL = "clase_ternaria"
LAG_AVAILABLE_COL = "lag1_disponible"

SEED = 290497

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

VENTANAS = {
    "A_202103_202104": {
        "train_months": [202103],
        "test_month": 202104,
        "weights": {
            202103: 1.00,
        },
    },

    "B_20210304_202105": {
        "train_months": [202103, 202104],
        "test_month": 202105,
        "weights": {
            202103: 0.75,
            202104: 1.00,
        },
    },

    "C_20210405_202106": {
        "train_months": [202104, 202105],
        "test_month": 202106,
        "weights": {
            202104: 0.75,
            202105: 1.00,
        },
    },
}

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

TARJETAS = [
    "ctarjeta_debito_transacciones",
    "ctarjeta_visa_transacciones",
    "ctarjeta_master_transacciones",
    "ccuenta_debitos_automaticos",
    "ctarjeta_visa_debitos_automaticos",
    "ctarjeta_master_debitos_automaticos",
]


# ============================================================
# Funciones
# ============================================================

def identificar_features(df):
    excluir = {
        ID_COL,
        MONTH_COL,
        TARGET_COL,
        LAG_AVAILABLE_COL,
    }

    delta = [
        c
        for c in df.columns
        if c.endswith("_delta_lag1")
    ]

    lag = [
        c
        for c in df.columns
        if (
            c.endswith("_lag1")
            and not c.endswith("_delta_lag1")
        )
    ]

    originales = [
        c
        for c in df.columns
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


def ganancia_por_corte(y, prob):
    orden = np.argsort(-prob)
    y_ord = np.asarray(y)[orden]
    acumulada = np.cumsum(y_ord)

    rows = []

    for n in CUTS:
        n_real = min(n, len(y_ord))

        positivos = int(
            acumulada[n_real - 1]
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
# Inicio
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

t0 = time.time()

print("=" * 80)
print("Z549 - ROBUSTEZ TEMPORAL ACTIVIDAD TARJETAS")
print("=" * 80)

df = pd.read_parquet(DATASET)

print(f"Filas: {len(df):,}")
print(f"Columnas: {len(df.columns):,}")

features_base = identificar_features(df)

if len(features_base) != 457:
    raise ValueError(
        f"Se esperaban 457 features; "
        f"hay {len(features_base)}"
    )

faltantes = [
    c
    for c in TARJETAS
    if c not in df.columns
]

if faltantes:
    raise ValueError(
        f"Faltan variables: {faltantes}"
    )


# ============================================================
# FE actual de tarjetas
# ============================================================

FEATURE_TARJETAS = "actividad_tarjetas"

df[FEATURE_TARJETAS] = (
    df[TARJETAS]
    .gt(0)
    .sum(axis=1)
    .astype("float32")
)

print(
    f"FULL457: {len(features_base)}"
)

print(
    f"{FEATURE_TARJETAS}: "
    f"mean={df[FEATURE_TARJETAS].mean():.4f} "
    f"| min={df[FEATURE_TARJETAS].min():.0f} "
    f"| max={df[FEATURE_TARJETAS].max():.0f}"
)


# ============================================================
# Corridas
# ============================================================

metricas_rows = []
curvas_rows = []
comparacion_rows = []
importancias_rows = []

for nombre_ventana, cfg in VENTANAS.items():

    print()
    print("#" * 80)
    print(f"VENTANA {nombre_ventana}")
    print("#" * 80)

    train_months = cfg[
        "train_months"
    ]

    test_month = cfg[
        "test_month"
    ]

    pesos = cfg[
        "weights"
    ]

    train = df[
        df[MONTH_COL].isin(
            train_months
        )
    ].copy()

    test = df[
        df[MONTH_COL].eq(
            test_month
        )
    ].copy()

    y_train = target_binario(
        train[TARGET_COL]
    )

    y_test = target_binario(
        test[TARGET_COL]
    )

    sample_weight = (
        train[MONTH_COL]
        .map(pesos)
        .astype(float)
        .to_numpy()
    )

    if not np.isfinite(
        sample_weight
    ).all():
        raise ValueError(
            f"Pesos inválidos en "
            f"{nombre_ventana}"
        )

    print(
        f"Train {train_months}: "
        f"{len(train):,} filas "
        f"| BAJA+2="
        f"{int(y_train.sum()):,}"
    )

    print(
        f"Test {test_month}: "
        f"{len(test):,} filas "
        f"| BAJA+2="
        f"{int(y_test.sum()):,}"
    )

    experimentos = [
        (
            "BASE",
            features_base,
        ),
        (
            "TARJETAS_ACTUAL",
            features_base
            + [FEATURE_TARJETAS],
        ),
    ]

    curvas_ventana = {}

    for modelo_nombre, features in experimentos:

        print()
        print(
            f"{nombre_ventana} "
            f"| {modelo_nombre}"
        )

        inicio = time.time()

        model = LGBMClassifier(
            **PARAMS
        )

        model.fit(
            train[features],
            y_train,
            sample_weight=sample_weight,
        )

        prob = model.predict_proba(
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
            f"Best N="
            f"{int(mejor['cut']):,} "
            f"| pos="
            f"{int(mejor['positivos'])} "
            f"| gain="
            f"{mejor['ganancia_millones']:.2f} M"
        )

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
            "ventana":
                nombre_ventana,
            "train_months":
                ",".join(
                    map(
                        str,
                        train_months,
                    )
                ),
            "test_month":
                test_month,
            "modelo":
                modelo_nombre,
            "n_features":
                len(features),
            "auc":
                auc,
            "average_precision":
                ap,
            "logloss":
                ll,
            "best_cut":
                int(mejor["cut"]),
            "best_gain_millones":
                float(
                    mejor[
                        "ganancia_millones"
                    ]
                ),
            "segundos":
                segundos,
        })

        curva["ventana"] = (
            nombre_ventana
        )

        curva["modelo"] = (
            modelo_nombre
        )

        curvas_rows.append(
            curva
        )

        curvas_ventana[
            modelo_nombre
        ] = curva.set_index(
            "cut"
        )

        imp = pd.DataFrame({
            "feature":
                features,
            "importance_gain":
                model.feature_importances_,
        })

        imp[
            "rank_importance"
        ] = (
            imp[
                "importance_gain"
            ]
            .rank(
                method="min",
                ascending=False,
            )
            .astype(int)
        )

        if (
            FEATURE_TARJETAS
            in features
        ):
            fila_imp = imp[
                imp["feature"].eq(
                    FEATURE_TARJETAS
                )
            ].iloc[0]

            importancias_rows.append({
                "ventana":
                    nombre_ventana,
                "test_month":
                    test_month,
                "feature":
                    FEATURE_TARJETAS,
                "importance_gain":
                    float(
                        fila_imp[
                            "importance_gain"
                        ]
                    ),
                "rank_importance":
                    int(
                        fila_imp[
                            "rank_importance"
                        ]
                    ),
            })

    # --------------------------------------------------------
    # Comparación TARJETAS - BASE
    # --------------------------------------------------------

    base = (
        curvas_ventana["BASE"]
    )

    tarjetas = (
        curvas_ventana[
            "TARJETAS_ACTUAL"
        ]
    )

    delta = (
        tarjetas[
            "ganancia_millones"
        ]
        - base[
            "ganancia_millones"
        ]
    )

    print()
    print("DELTA TARJETAS - BASE")
    print(
        delta.to_string(
            float_format=lambda x:
                f"{x:+.2f}"
        )
    )

    zona = delta.loc[ZONA]

    resumen = {
        "ventana":
            nombre_ventana,
        "test_month":
            test_month,
        "delta_medio_M":
            zona.mean(),
        "delta_mediano_M":
            zona.median(),
        "delta_min_M":
            zona.min(),
        "delta_max_M":
            zona.max(),
        "cortes_positivos":
            int((zona > 0).sum()),
        "cortes_neutros":
            int((zona == 0).sum()),
        "cortes_negativos":
            int((zona < 0).sum()),
    }

    comparacion_rows.append(
        resumen
    )

    print()
    print("ROBUSTEZ 10K-14K")
    print(
        pd.DataFrame(
            [resumen]
        ).to_string(
            index=False,
            float_format=lambda x:
                f"{x:.2f}",
        )
    )


# ============================================================
# Consolidación
# ============================================================

metricas = pd.DataFrame(
    metricas_rows
)

curvas = pd.concat(
    curvas_rows,
    ignore_index=True,
)

comparacion = pd.DataFrame(
    comparacion_rows
)

importancias = pd.DataFrame(
    importancias_rows
)

metricas.to_csv(
    OUTPUT_DIR
    / "metricas_por_ventana.csv",
    index=False,
)

curvas.to_csv(
    OUTPUT_DIR
    / "ganancia_por_corte.csv",
    index=False,
)

comparacion.to_csv(
    OUTPUT_DIR
    / "robustez_tarjetas_vs_base.csv",
    index=False,
)

importancias.to_csv(
    OUTPUT_DIR
    / "importancia_actividad_tarjetas.csv",
    index=False,
)


# ============================================================
# Resumen final
# ============================================================

print()
print("=" * 80)
print("RESUMEN METRICAS")
print("=" * 80)

print(
    metricas[
        [
            "ventana",
            "test_month",
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
print("ROBUSTEZ TEMPORAL TARJETAS VS BASE")
print("=" * 80)

print(
    comparacion.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.2f}",
    )
)

print()
print("=" * 80)
print("IMPORTANCIA ACTIVIDAD_TARJETAS")
print("=" * 80)

print(
    importancias.to_string(
        index=False
    )
)


# ============================================================
# Resumen agregado de ventanas
# ============================================================

print()
print("=" * 80)
print("RESUMEN AGREGADO")
print("=" * 80)

resumen_agregado = {
    "ventanas":
        len(comparacion),
    "delta_medio_entre_ventanas_M":
        comparacion[
            "delta_medio_M"
        ].mean(),
    "delta_mediano_entre_ventanas_M":
        comparacion[
            "delta_mediano_M"
        ].median(),
    "ventanas_delta_medio_positivo":
        int(
            (
                comparacion[
                    "delta_medio_M"
                ] > 0
            ).sum()
        ),
    "total_cortes_positivos":
        int(
            comparacion[
                "cortes_positivos"
            ].sum()
        ),
    "total_cortes_neutros":
        int(
            comparacion[
                "cortes_neutros"
            ].sum()
        ),
    "total_cortes_negativos":
        int(
            comparacion[
                "cortes_negativos"
            ].sum()
        ),
}

for k, v in (
    resumen_agregado.items()
):
    print(
        f"{k}: {v}"
    )


# ============================================================
# Metadata
# ============================================================

metadata = {
    "dataset":
        str(DATASET),
    "seed":
        SEED,
    "feature":
        FEATURE_TARJETAS,
    "componentes_tarjetas":
        TARJETAS,
    "ventanas":
        VENTANAS,
    "cuts":
        CUTS,
    "zona_robustez":
        ZONA,
    "n_features_base":
        len(features_base),
    "runtime_minutos":
        (time.time() - t0)
        / 60.0,
}

with open(
    OUTPUT_DIR
    / "metadata_z549.json",
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
print("Z549 FINALIZADO")
print("Output:", OUTPUT_DIR)
