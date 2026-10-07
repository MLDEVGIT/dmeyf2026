from pathlib import Path
import json
import time
import gc

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss


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
    "competencia_01/actividad_trx_da_z554"
)

MONTH = "foto_mes"
TARGET = "clase_ternaria"
ID = "numero_de_cliente"

SEED = 290497

CUTS = list(range(8000, 16001, 500))
ROBUST_CUTS = list(range(10000, 14001, 500))

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

TRX_VARS = [
    "ctarjeta_debito_transacciones",
    "ctarjeta_visa_transacciones",
    "ctarjeta_master_transacciones",
]

DA_VARS = [
    "ccuenta_debitos_automaticos",
    "ctarjeta_visa_debitos_automaticos",
    "ctarjeta_master_debitos_automaticos",
]

ALL_CARD_VARS = TRX_VARS + DA_VARS

EXPERIMENTS = {
    "BASE": [],
    "TRX": [
        "actividad_trx",
    ],
    "DA": [
        "actividad_da",
    ],
    "TRX_DA": [
        "actividad_trx",
        "actividad_da",
    ],
    "TOTAL": [
        "actividad_tarjetas",
    ],
}

WINDOWS = [
    {
        "window": "A_202104",
        "train_months": [202103],
        "test_month": 202104,
        "weights": {
            202103: 1.0,
        },
    },
    {
        "window": "B_202105",
        "train_months": [
            202103,
            202104,
        ],
        "test_month": 202105,
        "weights": {
            202103: 0.75,
            202104: 1.0,
        },
    },
    {
        "window": "C_202106",
        "train_months": [
            202104,
            202105,
        ],
        "test_month": 202106,
        "weights": {
            202104: 0.75,
            202105: 1.0,
        },
    },
]


# ============================================================
# Helpers
# ============================================================

def ganancia(y_true, prob, n):
    order = np.argsort(-prob)
    selected = order[:n]

    y_sel = y_true[selected]

    tp = int(y_sel.sum())
    fp = int(len(y_sel) - tp)

    return (
        tp * 1_072_500
        - fp * 27_500
    )


def add_activity_features(df):
    df = df.copy()

    df["actividad_trx"] = (
        df[TRX_VARS]
        .gt(0)
        .sum(axis=1)
        .astype("int8")
    )

    df["actividad_da"] = (
        df[DA_VARS]
        .gt(0)
        .sum(axis=1)
        .astype("int8")
    )

    df["actividad_tarjetas"] = (
        df[ALL_CARD_VARS]
        .gt(0)
        .sum(axis=1)
        .astype("int8")
    )

    return df


# ============================================================
# Carga
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

t0 = time.time()

print("=" * 95)
print("Z554 - ROBUSTEZ TEMPORAL: TRX vs DA")
print("=" * 95)

df = pd.read_parquet(DATASET)

print(
    f"Dataset: {df.shape[0]:,} filas, "
    f"{df.shape[1]} columnas"
)

df = add_activity_features(df)

print()
print("Features construidas:")
print(
    df[
        [
            "actividad_trx",
            "actividad_da",
            "actividad_tarjetas",
        ]
    ]
    .describe()
    .round(4)
)


# ============================================================
# FULL457
# ============================================================

original_cols = [
    c for c in df.columns
    if (
        c not in {
            ID,
            MONTH,
            TARGET,
            "lag1_disponible",
            "actividad_trx",
            "actividad_da",
            "actividad_tarjetas",
        }
        and not c.endswith("_lag1")
        and not c.endswith("_delta_lag1")
    )
]

# Z523 = 152 originales + 152 lag1 + 152 delta + lag1_disponible
lag_cols = [
    f"{c}_lag1"
    for c in original_cols
    if f"{c}_lag1" in df.columns
]

delta_cols = [
    f"{c}_delta_lag1"
    for c in original_cols
    if f"{c}_delta_lag1" in df.columns
]

FULL457 = (
    original_cols
    + lag_cols
    + delta_cols
    + ["lag1_disponible"]
)

FULL457 = list(dict.fromkeys(FULL457))

print()
print(f"Originales: {len(original_cols)}")
print(f"Lag1:       {len(lag_cols)}")
print(f"Delta:      {len(delta_cols)}")
print(f"FULL:       {len(FULL457)}")

if len(FULL457) != 457:
    raise ValueError(
        f"FULL457 debería tener 457 features y tiene {len(FULL457)}"
    )


# ============================================================
# Experimentos
# ============================================================

metric_rows = []
gain_rows = []
importance_rows = []

for w in WINDOWS:

    window = w["window"]
    train_months = w["train_months"]
    test_month = w["test_month"]

    print()
    print("=" * 95)
    print(
        f"{window}: train={train_months} "
        f"-> test={test_month}"
    )
    print("=" * 95)

    train_mask = df[MONTH].isin(
        train_months
    )
    test_mask = df[MONTH].eq(
        test_month
    )

    y_train = (
        df.loc[
            train_mask,
            TARGET,
        ]
        .eq("BAJA+2")
        .astype("int8")
        .to_numpy()
    )

    y_test = (
        df.loc[
            test_mask,
            TARGET,
        ]
        .eq("BAJA+2")
        .astype("int8")
        .to_numpy()
    )

    sample_weight = (
        df.loc[
            train_mask,
            MONTH,
        ]
        .map(w["weights"])
        .astype(float)
        .to_numpy()
    )

    print(
        f"Train: {train_mask.sum():,} "
        f"| BAJA+2: {y_train.sum():,}"
    )
    print(
        f"Test:  {test_mask.sum():,} "
        f"| BAJA+2: {y_test.sum():,}"
    )

    predictions = {}

    for experiment, extras in EXPERIMENTS.items():

        features = FULL457 + extras

        print()
        print(
            f"[{window}] {experiment} "
            f"({len(features)} features)"
        )

        model = lgb.LGBMClassifier(
            **PARAMS
        )

        model.fit(
            df.loc[
                train_mask,
                features,
            ],
            y_train,
            sample_weight=sample_weight,
        )

        prob = model.predict_proba(
            df.loc[
                test_mask,
                features,
            ]
        )[:, 1]

        predictions[experiment] = prob

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
        )

        gains = {
            n: ganancia(
                y_test,
                prob,
                n,
            )
            for n in CUTS
        }

        best_n = max(
            gains,
            key=gains.get,
        )

        best_gain = gains[
            best_n
        ]

        metric_rows.append({
            "window": window,
            "test_month": test_month,
            "experiment": experiment,
            "n_features": len(features),
            "auc": auc,
            "average_precision": ap,
            "logloss": ll,
            "best_cut": best_n,
            "best_gain": best_gain,
        })

        for n, gain in gains.items():
            gain_rows.append({
                "window": window,
                "test_month": test_month,
                "experiment": experiment,
                "cut": n,
                "gain": gain,
            })

        imp = pd.DataFrame({
            "feature": features,
            "gain_importance":
                model.booster_
                .feature_importance(
                    importance_type="gain"
                ),
        })

        imp = imp.sort_values(
            "gain_importance",
            ascending=False,
        ).reset_index(drop=True)

        imp["rank"] = (
            np.arange(len(imp)) + 1
        )

        for feature in extras:

            row = imp[
                imp["feature"].eq(
                    feature
                )
            ]

            importance_rows.append({
                "window": window,
                "test_month": test_month,
                "experiment": experiment,
                "feature": feature,
                "gain_importance":
                    float(
                        row[
                            "gain_importance"
                        ].iloc[0]
                    ),
                "rank":
                    int(
                        row[
                            "rank"
                        ].iloc[0]
                    ),
            })

        print(
            f"AUC={auc:.6f} "
            f"AP={ap:.6f} "
            f"LL={ll:.6f} "
            f"BEST={best_gain / 1e6:.2f}M"
            f"@{best_n}"
        )

        del model, imp
        gc.collect()

    # --------------------------------------------------------
    # Deltas económicos vs BASE
    # --------------------------------------------------------

    base_prob = predictions["BASE"]

    print()
    print("ROBUSTEZ 10k-14k vs BASE")

    for experiment in [
        "TRX",
        "DA",
        "TRX_DA",
        "TOTAL",
    ]:

        deltas = []

        for n in ROBUST_CUTS:

            g_base = ganancia(
                y_test,
                base_prob,
                n,
            )

            g_exp = ganancia(
                y_test,
                predictions[
                    experiment
                ],
                n,
            )

            deltas.append(
                g_exp - g_base
            )

        arr = np.array(
            deltas,
            dtype=float,
        )

        pos = int(
            (arr > 0).sum()
        )

        neg = int(
            (arr < 0).sum()
        )

        neu = int(
            (arr == 0).sum()
        )

        print(
            f"{experiment:8s} "
            f"mean={arr.mean()/1e6:+7.2f}M "
            f"median={np.median(arr)/1e6:+7.2f}M "
            f"min={arr.min()/1e6:+7.2f}M "
            f"max={arr.max()/1e6:+7.2f}M "
            f"+/0/-={pos}/{neu}/{neg}"
        )

    del predictions
    gc.collect()


# ============================================================
# Resultados
# ============================================================

metrics = pd.DataFrame(
    metric_rows
)

gains = pd.DataFrame(
    gain_rows
)

importances = pd.DataFrame(
    importance_rows
)

metrics.to_csv(
    OUTPUT_DIR
    / "metricas_z554.csv",
    index=False,
)

gains.to_csv(
    OUTPUT_DIR
    / "ganancias_por_cut_z554.csv",
    index=False,
)

importances.to_csv(
    OUTPUT_DIR
    / "importancias_features_z554.csv",
    index=False,
)


# ============================================================
# Deltas vs BASE por ventana/cut
# ============================================================

base_gains = (
    gains[
        gains[
            "experiment"
        ].eq("BASE")
    ][
        [
            "window",
            "cut",
            "gain",
        ]
    ]
    .rename(
        columns={
            "gain":
                "gain_base"
        }
    )
)

delta_df = (
    gains[
        ~gains[
            "experiment"
        ].eq("BASE")
    ]
    .merge(
        base_gains,
        on=[
            "window",
            "cut",
        ],
        validate="many_to_one",
    )
)

delta_df[
    "delta_gain_vs_base"
] = (
    delta_df["gain"]
    - delta_df["gain_base"]
)

delta_df.to_csv(
    OUTPUT_DIR
    / "deltas_vs_base_z554.csv",
    index=False,
)


# ============================================================
# Robustez agregada
# ============================================================

rob = delta_df[
    delta_df[
        "cut"
    ].isin(ROBUST_CUTS)
].copy()

robust_rows = []

for experiment in [
    "TRX",
    "DA",
    "TRX_DA",
    "TOTAL",
]:

    z = rob[
        rob[
            "experiment"
        ].eq(experiment)
    ]

    for window in [
        *[
            w["window"]
            for w in WINDOWS
        ],
        "ALL",
    ]:

        if window == "ALL":
            q = z
        else:
            q = z[
                z[
                    "window"
                ].eq(window)
            ]

        d = q[
            "delta_gain_vs_base"
        ].to_numpy(
            dtype=float
        )

        robust_rows.append({
            "experiment":
                experiment,
            "window":
                window,
            "n_observations":
                len(d),
            "mean_delta":
                d.mean(),
            "median_delta":
                np.median(d),
            "min_delta":
                d.min(),
            "max_delta":
                d.max(),
            "positive":
                int(
                    (d > 0).sum()
                ),
            "neutral":
                int(
                    (d == 0).sum()
                ),
            "negative":
                int(
                    (d < 0).sum()
                ),
        })

robust = pd.DataFrame(
    robust_rows
)

robust.to_csv(
    OUTPUT_DIR
    / "robustez_z554.csv",
    index=False,
)


# ============================================================
# Resumen final
# ============================================================

print()
print("=" * 95)
print("RESUMEN DE METRICAS")
print("=" * 95)

print(
    metrics.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}",
    )
)

print()
print("=" * 95)
print("ROBUSTEZ AGREGADA 10k-14k - TRES OOT")
print("=" * 95)

final_rob = robust[
    robust[
        "window"
    ].eq("ALL")
].copy()

print(
    final_rob[
        [
            "experiment",
            "n_observations",
            "mean_delta",
            "median_delta",
            "min_delta",
            "max_delta",
            "positive",
            "neutral",
            "negative",
        ]
    ].assign(
        mean_delta=lambda x:
            x["mean_delta"] / 1e6,
        median_delta=lambda x:
            x["median_delta"] / 1e6,
        min_delta=lambda x:
            x["min_delta"] / 1e6,
        max_delta=lambda x:
            x["max_delta"] / 1e6,
    ).to_string(
        index=False,
        float_format=lambda x:
            f"{x:+.3f}",
    )
)

print()
print("=" * 95)
print("IMPORTANCIA DE FEATURES NUEVAS")
print("=" * 95)

print(
    importances.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.2f}",
    )
)


# ============================================================
# Metadata
# ============================================================

metadata = {
    "dataset":
        str(DATASET),
    "seed":
        SEED,
    "params":
        PARAMS,
    "windows":
        WINDOWS,
    "experiments":
        EXPERIMENTS,
    "robust_cuts":
        ROBUST_CUTS,
    "trx_vars":
        TRX_VARS,
    "da_vars":
        DA_VARS,
    "runtime_seconds":
        time.time() - t0,
}

with open(
    OUTPUT_DIR
    / "metadata_z554.json",
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
print("Z554 FINALIZADO")
print("Output:", OUTPUT_DIR)
