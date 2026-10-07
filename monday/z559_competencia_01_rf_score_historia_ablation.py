from pathlib import Path
import gc
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
)


# ============================================================
# CONFIG
# ============================================================

DATASET = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/"
    "competencia_01/feature_engineering_z523/"
    "competencia_01_historico_lag1.parquet"
)

OUTPUT_DIR = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/"
    "competencia_01/rf_score_historia_z559"
)

ID = "numero_de_cliente"
MONTH = "foto_mes"
TARGET = "clase_ternaria"

SEED = 290497

CUTS = list(range(8000, 16001, 500))
ROBUST_CUTS = list(range(10000, 14001, 500))

# Para construir score_t:
# RF entrenado en t-1 -> predice t.
SCORE_PAIRS = [
    (202103, 202104),
    (202104, 202105),
    (202105, 202106),
]

# Ventanas donde ya tenemos score actual + score lag.
WINDOWS = [
    {
        "window": "B_202105",
        "train_months": [202104],
        "test_month": 202105,
        "weights": {
            202104: 1.0,
        },
    },
    {
        "window": "C_202106",
        "train_months": [202104, 202105],
        "test_month": 202106,
        "weights": {
            202104: 0.75,
            202105: 1.0,
        },
    },
]

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_leaf": 100,
    "max_features": "sqrt",
    "class_weight": "balanced_subsample",
    "random_state": SEED,
    "n_jobs": -1,
}

LGB_PARAMS = {
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
# HELPERS
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


def evaluar(y_true, prob):
    gains = {
        n: ganancia(
            y_true,
            prob,
            n,
        )
        for n in CUTS
    }

    best_cut = max(
        gains,
        key=gains.get,
    )

    return {
        "auc":
            roc_auc_score(
                y_true,
                prob,
            ),
        "ap":
            average_precision_score(
                y_true,
                prob,
            ),
        "logloss":
            log_loss(
                y_true,
                prob,
            ),
        "best_cut":
            best_cut,
        "best_gain":
            gains[best_cut],
        "gains":
            gains,
    }


def preparar_rf(frame, cols):
    return (
        frame[cols]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
        .astype("float32")
    )


def resumen_delta(
    gains_ref,
    gains_candidate,
):
    arr = np.array(
        [
            gains_candidate[n]
            - gains_ref[n]
            for n in ROBUST_CUTS
        ],
        dtype=float,
    )

    return {
        "mean_delta":
            float(arr.mean()),
        "median_delta":
            float(np.median(arr)),
        "min_delta":
            float(arr.min()),
        "max_delta":
            float(arr.max()),
        "positive":
            int((arr > 0).sum()),
        "neutral":
            int((arr == 0).sum()),
        "negative":
            int((arr < 0).sum()),
    }


# ============================================================
# LOAD
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

t0 = time.time()

print("=" * 100)
print("Z559 - HISTORIA DEL RF_SCORE: ABLATION")
print("=" * 100)

df = pd.read_parquet(DATASET)

print(
    f"Dataset: {len(df):,} filas, "
    f"{df.shape[1]} columnas"
)


# ============================================================
# FEATURES BASE
# ============================================================

original_cols = [
    c for c in df.columns
    if (
        c not in {
            ID,
            MONTH,
            TARGET,
            "lag1_disponible",
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

if len(original_cols) != 152:
    raise ValueError(
        f"Esperaba 152 originales; "
        f"hay {len(original_cols)}"
    )

if len(FULL457) != 457:
    raise ValueError(
        f"Esperaba FULL457=457; "
        f"hay {len(FULL457)}"
    )

print(f"Originales: {len(original_cols)}")
print(f"FULL457:    {len(FULL457)}")


# ============================================================
# GENERAR SCORE TEMPORAL ESTRICTO
#
# score abril <- RF marzo
# score mayo  <- RF abril
# score junio <- RF mayo
# ============================================================

df["rf_score_actual"] = np.nan

rf_score_rows = []

print()
print("=" * 100)
print("CONSTRUCCION TEMPORAL DE RF_SCORE")
print("=" * 100)

for i, (
    fit_month,
    score_month,
) in enumerate(
    SCORE_PAIRS,
    start=1,
):

    print()
    print(
        f"RF {fit_month} "
        f"-> score {score_month}"
    )

    fit_mask = df[
        MONTH
    ].eq(fit_month)

    score_mask = df[
        MONTH
    ].eq(score_month)

    fit_df = df.loc[
        fit_mask
    ]

    score_df = df.loc[
        score_mask
    ]

    y_fit = (
        fit_df[TARGET]
        .eq("BAJA+2")
        .astype("int8")
        .to_numpy()
    )

    y_score = (
        score_df[TARGET]
        .eq("BAJA+2")
        .astype("int8")
        .to_numpy()
    )

    X_fit = preparar_rf(
        fit_df,
        original_cols,
    )

    X_score = preparar_rf(
        score_df,
        original_cols,
    )

    rf = RandomForestClassifier(
        **{
            **RF_PARAMS,
            "random_state":
                SEED + i,
        }
    )

    rf.fit(
        X_fit,
        y_fit,
    )

    pred = (
        rf.predict_proba(
            X_score
        )[:, 1]
        .astype("float32")
    )

    df.loc[
        score_mask,
        "rf_score_actual",
    ] = pred

    auc = roc_auc_score(
        y_score,
        pred,
    )

    ap = average_precision_score(
        y_score,
        pred,
    )

    rf_score_rows.append({
        "fit_month":
            fit_month,
        "score_month":
            score_month,
        "n_fit":
            int(fit_mask.sum()),
        "n_score":
            int(score_mask.sum()),
        "baja2_fit":
            int(y_fit.sum()),
        "baja2_score":
            int(y_score.sum()),
        "auc":
            auc,
        "ap":
            ap,
    })

    print(
        f"AUC={auc:.6f} "
        f"AP={ap:.6f}"
    )

    del (
        fit_df,
        score_df,
        X_fit,
        X_score,
        rf,
        pred,
    )

    gc.collect()


# ============================================================
# LAG DEL RF_SCORE POR CLIENTE
# ============================================================

print()
print("=" * 100)
print("CONSTRUYENDO HISTORIA DEL RF_SCORE")
print("=" * 100)

score_lookup = (
    df[
        [
            ID,
            MONTH,
            "rf_score_actual",
        ]
    ]
    .copy()
)

score_lookup[
    MONTH
] = (
    score_lookup[MONTH]
    + 1
)

# foto_mes está YYYYMM y no se puede sumar 1
# en diciembre. En nuestro rango marzo-junio no cruzamos
# diciembre, pero hacemos el cálculo correcto igualmente.

def mes_siguiente(m):
    year = m // 100
    month = m % 100

    if month == 12:
        return (
            (year + 1) * 100
            + 1
        )

    return (
        year * 100
        + month
        + 1
    )


score_lookup[MONTH] = (
    df[
        [
            ID,
            MONTH,
            "rf_score_actual",
        ]
    ][MONTH]
    .map(mes_siguiente)
    .to_numpy()
)

score_lookup = (
    score_lookup.rename(
        columns={
            "rf_score_actual":
                "rf_score_lag1"
        }
    )
)

df = df.merge(
    score_lookup,
    on=[
        ID,
        MONTH,
    ],
    how="left",
    validate="one_to_one",
)

df[
    "delta_rf_score"
] = (
    df["rf_score_actual"]
    - df["rf_score_lag1"]
)

for m in [
    202104,
    202105,
    202106,
]:

    part = df[
        df[MONTH].eq(m)
    ]

    print()
    print(f"Mes {m}")

    print(
        "  actual nonNA="
        f"{part['rf_score_actual'].notna().mean():.4%}"
    )

    print(
        "  lag1 nonNA="
        f"{part['rf_score_lag1'].notna().mean():.4%}"
    )

    print(
        "  delta nonNA="
        f"{part['delta_rf_score'].notna().mean():.4%}"
    )

    if (
        part[
            "rf_score_actual"
        ].notna().any()
    ):
        print(
            "  actual mean="
            f"{part['rf_score_actual'].mean():.6f}"
        )

    if (
        part[
            "rf_score_lag1"
        ].notna().any()
    ):
        print(
            "  lag mean="
            f"{part['rf_score_lag1'].mean():.6f}"
        )

    if (
        part[
            "delta_rf_score"
        ].notna().any()
    ):
        print(
            "  delta mean="
            f"{part['delta_rf_score'].mean():+.6f}"
        )


# ============================================================
# EXPERIMENTOS
# ============================================================

EXPERIMENTS = {
    "BASE":
        FULL457,

    "ACTUAL":
        FULL457
        + [
            "rf_score_actual",
        ],

    "ACTUAL_LAG":
        FULL457
        + [
            "rf_score_actual",
            "rf_score_lag1",
        ],

    "ACTUAL_LAG_DELTA":
        FULL457
        + [
            "rf_score_actual",
            "rf_score_lag1",
            "delta_rf_score",
        ],
}


metric_rows = []
gain_rows = []
robust_rows = []
importance_rows = []


# ============================================================
# LOOP VENTANAS
# ============================================================

for w in WINDOWS:

    window = w[
        "window"
    ]

    train_months = w[
        "train_months"
    ]

    test_month = w[
        "test_month"
    ]

    weights = w[
        "weights"
    ]

    print()
    print("=" * 100)
    print(
        f"{window}: "
        f"train={train_months} "
        f"-> test={test_month}"
    )
    print("=" * 100)

    train_mask = df[
        MONTH
    ].isin(train_months)

    test_mask = df[
        MONTH
    ].eq(test_month)

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
        .map(weights)
        .astype(float)
        .to_numpy()
    )

    print(
        f"Train: "
        f"{train_mask.sum():,} "
        f"| BAJA+2="
        f"{y_train.sum():,}"
    )

    print(
        f"Test:  "
        f"{test_mask.sum():,} "
        f"| BAJA+2="
        f"{y_test.sum():,}"
    )

    results = {}

    for experiment, features in (
        EXPERIMENTS.items()
    ):

        print()
        print(
            f"{experiment} "
            f"({len(features)} features)"
        )

        model = lgb.LGBMClassifier(
            **LGB_PARAMS
        )

        model.fit(
            df.loc[
                train_mask,
                features,
            ],
            y_train,
            sample_weight=
                sample_weight,
        )

        prob = model.predict_proba(
            df.loc[
                test_mask,
                features,
            ]
        )[:, 1]

        res = evaluar(
            y_test,
            prob,
        )

        results[
            experiment
        ] = res

        metric_rows.append({
            "window":
                window,
            "test_month":
                test_month,
            "experiment":
                experiment,
            "n_features":
                len(features),
            "auc":
                res["auc"],
            "ap":
                res["ap"],
            "logloss":
                res["logloss"],
            "best_cut":
                res["best_cut"],
            "best_gain":
                res["best_gain"],
        })

        for cut, gain in (
            res["gains"].items()
        ):

            gain_rows.append({
                "window":
                    window,
                "test_month":
                    test_month,
                "experiment":
                    experiment,
                "cut":
                    cut,
                "gain":
                    gain,
            })

        print(
            f"AUC={res['auc']:.6f} "
            f"AP={res['ap']:.6f} "
            f"LL={res['logloss']:.6f} "
            f"BEST="
            f"{res['best_gain']/1e6:.2f}M"
            f"@{res['best_cut']}"
        )

        if experiment != "BASE":

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

            for f in [
                "rf_score_actual",
                "rf_score_lag1",
                "delta_rf_score",
            ]:

                if f not in features:
                    continue

                row = imp[
                    imp["feature"]
                    .eq(f)
                ].iloc[0]

                importance_rows.append({
                    "window":
                        window,
                    "test_month":
                        test_month,
                    "experiment":
                        experiment,
                    "feature":
                        f,
                    "gain_importance":
                        float(
                            row[
                                "gain_importance"
                            ]
                        ),
                    "rank":
                        int(row["rank"]),
                })

        del model
        gc.collect()


    # ========================================================
    # ROBUSTEZ vs BASE y vs ACTUAL
    # ========================================================

    print()
    print(
        "ROBUSTEZ 10k-14k"
    )

    for experiment in [
        "ACTUAL",
        "ACTUAL_LAG",
        "ACTUAL_LAG_DELTA",
    ]:

        rb = resumen_delta(
            results[
                "BASE"
            ]["gains"],
            results[
                experiment
            ]["gains"],
        )

        ra = resumen_delta(
            results[
                "ACTUAL"
            ]["gains"],
            results[
                experiment
            ]["gains"],
        )

        robust_rows.append({
            "window":
                window,
            "test_month":
                test_month,
            "experiment":
                experiment,

            "mean_vs_base":
                rb[
                    "mean_delta"
                ],
            "median_vs_base":
                rb[
                    "median_delta"
                ],
            "positive_vs_base":
                rb[
                    "positive"
                ],
            "neutral_vs_base":
                rb[
                    "neutral"
                ],
            "negative_vs_base":
                rb[
                    "negative"
                ],

            "mean_vs_actual":
                ra[
                    "mean_delta"
                ],
            "median_vs_actual":
                ra[
                    "median_delta"
                ],
            "positive_vs_actual":
                ra[
                    "positive"
                ],
            "neutral_vs_actual":
                ra[
                    "neutral"
                ],
            "negative_vs_actual":
                ra[
                    "negative"
                ],
        })

        print(
            f"{experiment:17s} "
            f"vs BASE "
            f"mean="
            f"{rb['mean_delta']/1e6:+.2f}M "
            f"med="
            f"{rb['median_delta']/1e6:+.2f}M "
            f"+/0/-="
            f"{rb['positive']}/"
            f"{rb['neutral']}/"
            f"{rb['negative']} "
            f"| vs ACTUAL "
            f"mean="
            f"{ra['mean_delta']/1e6:+.2f}M "
            f"med="
            f"{ra['median_delta']/1e6:+.2f}M "
            f"+/0/-="
            f"{ra['positive']}/"
            f"{ra['neutral']}/"
            f"{ra['negative']}"
        )


# ============================================================
# GUARDAR
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

rf_scores_df = pd.DataFrame(
    rf_score_rows
)

metrics_df.to_csv(
    OUTPUT_DIR
    / "metricas_z559.csv",
    index=False,
)

gains_df.to_csv(
    OUTPUT_DIR
    / "ganancias_z559.csv",
    index=False,
)

robust_df.to_csv(
    OUTPUT_DIR
    / "robustez_z559.csv",
    index=False,
)

importance_df.to_csv(
    OUTPUT_DIR
    / "importancias_rf_historia_z559.csv",
    index=False,
)

rf_scores_df.to_csv(
    OUTPUT_DIR
    / "metricas_rf_temporal_z559.csv",
    index=False,
)


# ============================================================
# RESUMEN
# ============================================================

print()
print("=" * 100)
print("RESUMEN ROBUSTEZ")
print("=" * 100)

tmp = robust_df.copy()

for c in [
    "mean_vs_base",
    "median_vs_base",
    "mean_vs_actual",
    "median_vs_actual",
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
print("=" * 100)
print("IMPORTANCIAS FEATURES RF")
print("=" * 100)

print(
    importance_df.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.2f}",
    )
)


metadata = {
    "dataset":
        str(DATASET),
    "seed":
        SEED,
    "score_pairs":
        SCORE_PAIRS,
    "windows":
        WINDOWS,
    "rf_params":
        RF_PARAMS,
    "lgb_params":
        LGB_PARAMS,
    "experiments":
        list(
            EXPERIMENTS.keys()
        ),
    "runtime_seconds":
        time.time() - t0,
}

with open(
    OUTPUT_DIR
    / "metadata_z559.json",
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
    f"{time.time()-t0:.2f}s"
)

print()
print("Z559 FINALIZADO")
print("Output:", OUTPUT_DIR)
