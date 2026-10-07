from pathlib import Path
import gc
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
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
    "competencia_01/actividad_da_seeds_z555"
)

ID = "numero_de_cliente"
MONTH = "foto_mes"
TARGET = "clase_ternaria"

TRAIN_MONTHS = [202104, 202105]
TEST_MONTH = 202106

MONTH_WEIGHTS = {
    202104: 0.75,
    202105: 1.00,
}

SEEDS = [
    290497,
    100003,
    200003,
    300007,
    400009,
]

CUTS = list(range(8000, 16001, 500))
ROBUST_CUTS = list(range(10000, 14001, 500))

DA_VARS = [
    "ccuenta_debitos_automaticos",
    "ctarjeta_visa_debitos_automaticos",
    "ctarjeta_master_debitos_automaticos",
]

PARAMS_BASE = {
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
    "n_jobs": -1,
    "verbosity": -1,
    "importance_type": "gain",
}


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


def metricas(y_true, prob):
    return {
        "auc": roc_auc_score(
            y_true,
            prob,
        ),
        "average_precision":
            average_precision_score(
                y_true,
                prob,
            ),
        "logloss": log_loss(
            y_true,
            prob,
        ),
    }


def evaluar_ganancias(y_true, prob):
    return {
        n: ganancia(
            y_true,
            prob,
            n,
        )
        for n in CUTS
    }


def resumen_robustez(
    gains_base,
    gains_da,
):
    deltas = np.array(
        [
            gains_da[n]
            - gains_base[n]
            for n in ROBUST_CUTS
        ],
        dtype=float,
    )

    return {
        "mean_delta":
            float(deltas.mean()),
        "median_delta":
            float(np.median(deltas)),
        "min_delta":
            float(deltas.min()),
        "max_delta":
            float(deltas.max()),
        "positive":
            int((deltas > 0).sum()),
        "neutral":
            int((deltas == 0).sum()),
        "negative":
            int((deltas < 0).sum()),
    }


# ============================================================
# Inicio
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

t0 = time.time()

print("=" * 95)
print("Z555 - ACTIVIDAD DA: ROBUSTEZ POR SEEDS + ENSEMBLE")
print("=" * 95)

df = pd.read_parquet(DATASET)

print(
    f"Dataset: {df.shape[0]:,} filas, "
    f"{df.shape[1]} columnas"
)


# ============================================================
# Feature DA
# ============================================================

df["actividad_da"] = (
    df[DA_VARS]
    .gt(0)
    .sum(axis=1)
    .astype("int8")
)

print()
print("ACTIVIDAD_DA GLOBAL")

print(
    df["actividad_da"]
    .describe()
    .round(4)
)

print()
print("ACTIVIDAD_DA POR MES")

print(
    df.groupby(MONTH)[
        "actividad_da"
    ]
    .agg(
        ["mean", "median"]
    )
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
            "actividad_da",
        }
        and not c.endswith("_lag1")
        and not c.endswith("_delta_lag1")
    )
]

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

FULL457 = list(
    dict.fromkeys(
        original_cols
        + lag_cols
        + delta_cols
        + ["lag1_disponible"]
    )
)

if len(FULL457) != 457:
    raise ValueError(
        f"FULL457 tiene {len(FULL457)} "
        "features; esperaba 457"
    )

DA458 = (
    FULL457
    + ["actividad_da"]
)

print()
print(
    f"BASE: {len(FULL457)} features"
)
print(
    f"DA:   {len(DA458)} features"
)


# ============================================================
# Train / test
# ============================================================

train_mask = df[
    MONTH
].isin(TRAIN_MONTHS)

test_mask = df[
    MONTH
].eq(TEST_MONTH)

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
    .map(MONTH_WEIGHTS)
    .astype(float)
    .to_numpy()
)

print()
print(
    f"Train {TRAIN_MONTHS}: "
    f"{train_mask.sum():,} filas "
    f"| BAJA+2={y_train.sum():,}"
)

print(
    f"Test {TEST_MONTH}: "
    f"{test_mask.sum():,} filas "
    f"| BAJA+2={y_test.sum():,}"
)


# ============================================================
# Entrenamiento por seed
# ============================================================

metric_rows = []
gain_rows = []
robust_rows = []
importance_rows = []

pred_base = {}
pred_da = {}

for seed in SEEDS:

    print()
    print("=" * 95)
    print(f"SEED {seed}")
    print("=" * 95)

    params = dict(
        PARAMS_BASE
    )

    params[
        "random_state"
    ] = seed

    seed_predictions = {}

    for experiment, features in [
        ("BASE", FULL457),
        ("DA", DA458),
    ]:

        print()
        print(
            f"[{seed}] {experiment} "
            f"({len(features)} features)"
        )

        model = lgb.LGBMClassifier(
            **params
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

        seed_predictions[
            experiment
        ] = prob

        if experiment == "BASE":
            pred_base[seed] = prob
        else:
            pred_da[seed] = prob

        mets = metricas(
            y_test,
            prob,
        )

        gains = evaluar_ganancias(
            y_test,
            prob,
        )

        best_cut = max(
            gains,
            key=gains.get,
        )

        best_gain = gains[
            best_cut
        ]

        metric_rows.append({
            "seed": seed,
            "experiment":
                experiment,
            **mets,
            "best_cut":
                best_cut,
            "best_gain":
                best_gain,
        })

        for cut, gain in gains.items():

            gain_rows.append({
                "seed": seed,
                "experiment":
                    experiment,
                "cut": cut,
                "gain": gain,
            })

        if experiment == "DA":

            imp = pd.DataFrame({
                "feature":
                    features,
                "gain_importance":
                    model.booster_
                    .feature_importance(
                        importance_type="gain"
                    ),
            })

            imp = (
                imp.sort_values(
                    "gain_importance",
                    ascending=False,
                )
                .reset_index(drop=True)
            )

            imp["rank"] = (
                np.arange(len(imp))
                + 1
            )

            row = imp[
                imp["feature"]
                .eq("actividad_da")
            ].iloc[0]

            importance_rows.append({
                "seed": seed,
                "gain_importance":
                    float(
                        row[
                            "gain_importance"
                        ]
                    ),
                "rank":
                    int(row["rank"]),
            })

        print(
            f"AUC={mets['auc']:.6f} "
            f"AP={mets['average_precision']:.6f} "
            f"LL={mets['logloss']:.6f} "
            f"BEST="
            f"{best_gain/1e6:.2f}M"
            f"@{best_cut}"
        )

        del model
        gc.collect()

    gains_base = (
        evaluar_ganancias(
            y_test,
            seed_predictions[
                "BASE"
            ],
        )
    )

    gains_da = (
        evaluar_ganancias(
            y_test,
            seed_predictions[
                "DA"
            ],
        )
    )

    rob = resumen_robustez(
        gains_base,
        gains_da,
    )

    robust_rows.append({
        "seed": seed,
        "type": "SEED",
        **rob,
    })

    print()
    print(
        "DA vs BASE 10k-14k: "
        f"mean="
        f"{rob['mean_delta']/1e6:+.2f}M "
        f"median="
        f"{rob['median_delta']/1e6:+.2f}M "
        f"min="
        f"{rob['min_delta']/1e6:+.2f}M "
        f"max="
        f"{rob['max_delta']/1e6:+.2f}M "
        f"+/0/-="
        f"{rob['positive']}/"
        f"{rob['neutral']}/"
        f"{rob['negative']}"
    )

    del seed_predictions
    gc.collect()


# ============================================================
# Ensemble
# ============================================================

print()
print("=" * 95)
print("ENSEMBLE 5 SEEDS")
print("=" * 95)

ensemble_base = np.mean(
    np.column_stack(
        [
            pred_base[s]
            for s in SEEDS
        ]
    ),
    axis=1,
)

ensemble_da = np.mean(
    np.column_stack(
        [
            pred_da[s]
            for s in SEEDS
        ]
    ),
    axis=1,
)

for experiment, prob in [
    ("BASE", ensemble_base),
    ("DA", ensemble_da),
]:

    mets = metricas(
        y_test,
        prob,
    )

    gains = evaluar_ganancias(
        y_test,
        prob,
    )

    best_cut = max(
        gains,
        key=gains.get,
    )

    best_gain = gains[
        best_cut
    ]

    metric_rows.append({
        "seed": "ENSEMBLE5",
        "experiment":
            experiment,
        **mets,
        "best_cut":
            best_cut,
        "best_gain":
            best_gain,
    })

    for cut, gain in gains.items():

        gain_rows.append({
            "seed": "ENSEMBLE5",
            "experiment":
                experiment,
            "cut": cut,
            "gain": gain,
        })

    print(
        f"{experiment:4s} "
        f"AUC={mets['auc']:.6f} "
        f"AP={mets['average_precision']:.6f} "
        f"LL={mets['logloss']:.6f} "
        f"BEST="
        f"{best_gain/1e6:.2f}M"
        f"@{best_cut}"
    )

ensemble_gains_base = (
    evaluar_ganancias(
        y_test,
        ensemble_base,
    )
)

ensemble_gains_da = (
    evaluar_ganancias(
        y_test,
        ensemble_da,
    )
)

rob_ensemble = resumen_robustez(
    ensemble_gains_base,
    ensemble_gains_da,
)

robust_rows.append({
    "seed": "ENSEMBLE5",
    "type": "ENSEMBLE",
    **rob_ensemble,
})

print()
print(
    "ENSEMBLE DA vs BASE 10k-14k: "
    f"mean="
    f"{rob_ensemble['mean_delta']/1e6:+.2f}M "
    f"median="
    f"{rob_ensemble['median_delta']/1e6:+.2f}M "
    f"min="
    f"{rob_ensemble['min_delta']/1e6:+.2f}M "
    f"max="
    f"{rob_ensemble['max_delta']/1e6:+.2f}M "
    f"+/0/-="
    f"{rob_ensemble['positive']}/"
    f"{rob_ensemble['neutral']}/"
    f"{rob_ensemble['negative']}"
)


# ============================================================
# Estabilidad de rankings entre seeds
# ============================================================

print()
print("=" * 95)
print("ESTABILIDAD TOP-N ENTRE SEEDS")
print("=" * 95)

jaccard_rows = []

for experiment, pred_dict in [
    ("BASE", pred_base),
    ("DA", pred_da),
]:

    rankings = {
        seed:
            np.argsort(
                -pred_dict[seed]
            )
        for seed in SEEDS
    }

    for cut in [
        10000,
        12000,
        14000,
    ]:

        vals = []

        for i in range(
            len(SEEDS)
        ):
            for j in range(
                i + 1,
                len(SEEDS),
            ):

                a = set(
                    rankings[
                        SEEDS[i]
                    ][:cut]
                )

                b = set(
                    rankings[
                        SEEDS[j]
                    ][:cut]
                )

                jac = (
                    len(a & b)
                    / len(a | b)
                )

                vals.append(jac)

                jaccard_rows.append({
                    "experiment":
                        experiment,
                    "cut": cut,
                    "seed_a":
                        SEEDS[i],
                    "seed_b":
                        SEEDS[j],
                    "jaccard":
                        jac,
                })

        print(
            f"{experiment:4s} "
            f"N={cut:5d} "
            f"mean Jaccard="
            f"{np.mean(vals):.6f} "
            f"min="
            f"{np.min(vals):.6f} "
            f"max="
            f"{np.max(vals):.6f}"
        )


# ============================================================
# Tablas finales
# ============================================================

metrics_df = pd.DataFrame(
    metric_rows
)

gains_df = pd.DataFrame(
    gain_rows
)

robust_df = pd.DataFrame(
    robust_rows
)

importance_df = pd.DataFrame(
    importance_rows
)

jaccard_df = pd.DataFrame(
    jaccard_rows
)

metrics_df.to_csv(
    OUTPUT_DIR
    / "metricas_z555.csv",
    index=False,
)

gains_df.to_csv(
    OUTPUT_DIR
    / "ganancias_por_cut_z555.csv",
    index=False,
)

robust_df.to_csv(
    OUTPUT_DIR
    / "robustez_seeds_z555.csv",
    index=False,
)

importance_df.to_csv(
    OUTPUT_DIR
    / "importancia_actividad_da_z555.csv",
    index=False,
)

jaccard_df.to_csv(
    OUTPUT_DIR
    / "jaccard_seeds_z555.csv",
    index=False,
)


# ============================================================
# Delta detallado por seed/cut
# ============================================================

delta_rows = []

for seed in [
    *SEEDS,
    "ENSEMBLE5",
]:

    b = gains_df[
        (gains_df["seed"] == seed)
        &
        (
            gains_df[
                "experiment"
            ] == "BASE"
        )
    ].set_index("cut")[
        "gain"
    ]

    d = gains_df[
        (gains_df["seed"] == seed)
        &
        (
            gains_df[
                "experiment"
            ] == "DA"
        )
    ].set_index("cut")[
        "gain"
    ]

    for cut in CUTS:

        delta_rows.append({
            "seed": seed,
            "cut": cut,
            "gain_base":
                int(b.loc[cut]),
            "gain_da":
                int(d.loc[cut]),
            "delta_da_vs_base":
                int(
                    d.loc[cut]
                    - b.loc[cut]
                ),
        })

delta_df = pd.DataFrame(
    delta_rows
)

delta_df.to_csv(
    OUTPUT_DIR
    / "deltas_da_vs_base_z555.csv",
    index=False,
)


# ============================================================
# Resumen
# ============================================================

print()
print("=" * 95)
print("RESUMEN ROBUSTEZ DA vs BASE")
print("=" * 95)

tmp = robust_df.copy()

for c in [
    "mean_delta",
    "median_delta",
    "min_delta",
    "max_delta",
]:
    tmp[c] = (
        tmp[c] / 1e6
    )

print(
    tmp.to_string(
        index=False,
        float_format=lambda x:
            f"{x:+.3f}",
    )
)

print()
print("=" * 95)
print("IMPORTANCIA ACTIVIDAD_DA")
print("=" * 95)

print(
    importance_df.to_string(
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
    "train_months":
        TRAIN_MONTHS,
    "test_month":
        TEST_MONTH,
    "month_weights":
        MONTH_WEIGHTS,
    "seeds":
        SEEDS,
    "robust_cuts":
        ROBUST_CUTS,
    "da_vars":
        DA_VARS,
    "params_base":
        PARAMS_BASE,
    "runtime_seconds":
        time.time() - t0,
}

with open(
    OUTPUT_DIR
    / "metadata_z555.json",
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
print("Z555 FINALIZADO")
print("Output:", OUTPUT_DIR)
