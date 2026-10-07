from pathlib import Path
import gc
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from scipy.stats import spearmanr


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
    "competencia_01/submits_z561_rf_crossfit"
)

ID = "numero_de_cliente"
MONTH = "foto_mes"
TARGET = "clase_ternaria"

TRAIN_MONTHS = [202104, 202105, 202106]
SCORE_MONTH = 202108

MONTH_WEIGHTS = {
    202104: 0.75,
    202105: 1.00,
    202106: 1.00,
}

SEEDS = [
    290497,
    100003,
    200003,
    300007,
    400009,
]

SUBMIT_CUTS = [
    10000,
    11000,
    11500,
    12000,
    12500,
    13000,
    13500,
    14000,
]

STABILITY_CUTS = [
    10000,
    11500,
    12000,
    12500,
    14000,
]

RF_PARAMS_BASE = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_leaf": 100,
    "max_features": "sqrt",
    "class_weight": "balanced_subsample",
    "n_jobs": -1,
}

LGB_PARAMS_BASE = {
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
# HELPERS
# ============================================================

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


def jaccard_top(
    pred_a,
    pred_b,
    n,
):
    idx_a = set(
        np.argsort(-pred_a)[:n]
    )

    idx_b = set(
        np.argsort(-pred_b)[:n]
    )

    return (
        len(idx_a & idx_b)
        / len(idx_a | idx_b)
    )


# ============================================================
# LOAD
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

t0 = time.time()

print("=" * 100)
print("Z561 - FINAL RF CROSS-FIT -> AGOSTO")
print("=" * 100)

df = pd.read_parquet(DATASET)

print(
    f"Dataset: {len(df):,} filas, "
    f"{df.shape[1]} columnas"
)


# ============================================================
# FEATURES
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

FEATURES_RF_LGB = (
    FULL457 + ["rf_score"]
)

print(
    f"Originales: {len(original_cols)}"
)

print(
    f"FULL457:    {len(FULL457)}"
)

print(
    f"RF+LGB:     {len(FEATURES_RF_LGB)}"
)


# ============================================================
# TRAIN / AUGUST
# ============================================================

train_df = (
    df[
        df[MONTH].isin(
            TRAIN_MONTHS
        )
    ]
    .copy()
    .reset_index(drop=True)
)

score_df = (
    df[
        df[MONTH].eq(
            SCORE_MONTH
        )
    ]
    .copy()
    .reset_index(drop=True)
)

y_train = (
    train_df[TARGET]
    .eq("BAJA+2")
    .astype("int8")
    .to_numpy()
)

weights_train = (
    train_df[MONTH]
    .map(MONTH_WEIGHTS)
    .astype(float)
    .to_numpy()
)

X_train_rf = preparar_rf(
    train_df,
    original_cols,
)

X_score_rf = preparar_rf(
    score_df,
    original_cols,
)

month_indices = {
    m: np.flatnonzero(
        train_df[MONTH]
        .eq(m)
        .to_numpy()
    )
    for m in TRAIN_MONTHS
}

print()
print(
    f"Train {TRAIN_MONTHS}: "
    f"{len(train_df):,} "
    f"| BAJA+2={y_train.sum():,}"
)

for m in TRAIN_MONTHS:

    idx = month_indices[m]

    print(
        f"  {m}: "
        f"{len(idx):,} filas "
        f"| BAJA+2="
        f"{y_train[idx].sum():,} "
        f"| weight="
        f"{MONTH_WEIGHTS[m]:.2f}"
    )

print(
    f"Score {SCORE_MONTH}: "
    f"{len(score_df):,} filas"
)


# ============================================================
# OUTPUTS
# ============================================================

predictions = {}
rf_score_predictions = {}

importance_rows = []
rf_crossfit_rows = []
seed_stability_rows = []


# ============================================================
# LOOP SEEDS
# ============================================================

for seed in SEEDS:

    seed_t0 = time.time()

    print()
    print("=" * 100)
    print(f"SEED {seed}")
    print("=" * 100)

    rf_train_score = np.zeros(
        len(train_df),
        dtype="float32",
    )

    # --------------------------------------------------------
    # LEAVE-ONE-MONTH-OUT RF SCORE PARA TRAIN
    # --------------------------------------------------------

    print()
    print(
        "RF TRAIN SCORE - "
        "LEAVE-ONE-MONTH-OUT"
    )

    for k, score_month in enumerate(
        TRAIN_MONTHS,
        start=1,
    ):

        idx_score = (
            month_indices[
                score_month
            ]
        )

        fit_months = [
            m
            for m in TRAIN_MONTHS
            if m != score_month
        ]

        idx_fit = np.concatenate(
            [
                month_indices[m]
                for m in fit_months
            ]
        )

        rf = RandomForestClassifier(
            **RF_PARAMS_BASE,
            random_state=seed + k,
        )

        rf.fit(
            X_train_rf.iloc[
                idx_fit
            ],
            y_train[
                idx_fit
            ],
            sample_weight=
                weights_train[
                    idx_fit
                ],
        )

        pred = rf.predict_proba(
            X_train_rf.iloc[
                idx_score
            ]
        )[:, 1]

        rf_train_score[
            idx_score
        ] = pred

        # Métrica descriptiva del score OOF
        y_part = y_train[
            idx_score
        ]

        from sklearn.metrics import (
            roc_auc_score,
            average_precision_score,
        )

        auc = roc_auc_score(
            y_part,
            pred,
        )

        ap = average_precision_score(
            y_part,
            pred,
        )

        rf_crossfit_rows.append({
            "seed": seed,
            "score_month":
                score_month,
            "fit_months":
                "+".join(
                    str(x)
                    for x in fit_months
                ),
            "auc": auc,
            "ap": ap,
        })

        print(
            f"score {score_month} "
            f"<- RF {fit_months}: "
            f"AUC={auc:.6f} "
            f"AP={ap:.6f}"
        )

        del rf, pred
        gc.collect()

    # --------------------------------------------------------
    # RF FINAL TRAIN COMPLETO -> AGOSTO
    # --------------------------------------------------------

    print()
    print(
        f"RF FINAL {TRAIN_MONTHS} "
        f"-> {SCORE_MONTH}"
    )

    rf_final = RandomForestClassifier(
        **RF_PARAMS_BASE,
        random_state=seed + 100,
    )

    rf_final.fit(
        X_train_rf,
        y_train,
        sample_weight=
            weights_train,
    )

    rf_score_aug = (
        rf_final.predict_proba(
            X_score_rf
        )[:, 1]
        .astype("float32")
    )

    rf_score_predictions[
        seed
    ] = rf_score_aug.copy()

    print(
        "rf_score agosto: "
        f"mean="
        f"{rf_score_aug.mean():.6f} "
        f"std="
        f"{rf_score_aug.std():.6f} "
        f"min="
        f"{rf_score_aug.min():.6f} "
        f"max="
        f"{rf_score_aug.max():.6f}"
    )

    del rf_final
    gc.collect()

    # --------------------------------------------------------
    # LGBM
    # --------------------------------------------------------

    train_df[
        "rf_score"
    ] = rf_train_score

    score_df[
        "rf_score"
    ] = rf_score_aug

    model = lgb.LGBMClassifier(
        **{
            **LGB_PARAMS_BASE,
            "random_state": seed,
        }
    )

    model.fit(
        train_df[
            FEATURES_RF_LGB
        ],
        y_train,
        sample_weight=
            weights_train,
    )

    prob = model.predict_proba(
        score_df[
            FEATURES_RF_LGB
        ]
    )[:, 1]

    predictions[
        seed
    ] = prob.copy()

    imp = pd.DataFrame({
        "feature":
            FEATURES_RF_LGB,
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
        np.arange(len(imp)) + 1
    )

    rf_row = imp[
        imp["feature"]
        .eq("rf_score")
    ].iloc[0]

    importance_rows.append({
        "seed": seed,
        "gain_importance":
            float(
                rf_row[
                    "gain_importance"
                ]
            ),
        "rank":
            int(
                rf_row["rank"]
            ),
    })

    print(
        "LGBM agosto: "
        f"prob mean="
        f"{prob.mean():.6f} "
        f"std="
        f"{prob.std():.6f}"
    )

    print(
        "rf_score importance: "
        f"gain="
        f"{rf_row['gain_importance']:.2f} "
        f"rank="
        f"{int(rf_row['rank'])}"
    )

    print(
        f"Seed runtime: "
        f"{time.time()-seed_t0:.1f}s"
    )

    del (
        model,
        rf_train_score,
        rf_score_aug,
        prob,
    )

    gc.collect()


# ============================================================
# ENSEMBLE 5 SEEDS
# ============================================================

print()
print("=" * 100)
print("ENSEMBLE 5 SEEDS")
print("=" * 100)

pred_matrix = np.column_stack(
    [
        predictions[s]
        for s in SEEDS
    ]
)

ensemble = pred_matrix.mean(
    axis=1
)

rf_score_matrix = np.column_stack(
    [
        rf_score_predictions[s]
        for s in SEEDS
    ]
)

rf_score_ensemble = (
    rf_score_matrix.mean(
        axis=1
    )
)

print(
    "LGB ensemble: "
    f"mean={ensemble.mean():.6f} "
    f"std={ensemble.std():.6f} "
    f"min={ensemble.min():.6f} "
    f"max={ensemble.max():.6f}"
)

print(
    "RF-score ensemble: "
    f"mean="
    f"{rf_score_ensemble.mean():.6f} "
    f"std="
    f"{rf_score_ensemble.std():.6f}"
)


# ============================================================
# ESTABILIDAD ENTRE SEEDS
# ============================================================

print()
print("=" * 100)
print("ESTABILIDAD ENTRE SEEDS")
print("=" * 100)

rankings = {
    s: np.argsort(
        -predictions[s]
    )
    for s in SEEDS
}

for cut in STABILITY_CUTS:

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

            seed_stability_rows.append({
                "cut": cut,
                "seed_a":
                    SEEDS[i],
                "seed_b":
                    SEEDS[j],
                "jaccard":
                    jac,
            })

    print(
        f"N={cut:5d} "
        f"Jaccard mean="
        f"{np.mean(vals):.6f} "
        f"min="
        f"{np.min(vals):.6f} "
        f"max="
        f"{np.max(vals):.6f}"
    )


# ============================================================
# RANKING COMPLETO
# ============================================================

ranking = pd.DataFrame({
    ID:
        score_df[ID]
        .to_numpy(),
    MONTH:
        score_df[MONTH]
        .to_numpy(),
    "prob_rf_crossfit":
        ensemble,
    "rf_score_aux":
        rf_score_ensemble,
})

for seed in SEEDS:

    ranking[
        f"prob_seed_{seed}"
    ] = predictions[
        seed
    ]

ranking[
    "rank_rf_crossfit"
] = (
    ranking[
        "prob_rf_crossfit"
    ]
    .rank(
        method="first",
        ascending=False,
    )
    .astype(int)
)

ranking = (
    ranking.sort_values(
        "prob_rf_crossfit",
        ascending=False,
    )
    .reset_index(drop=True)
)

ranking.to_csv(
    OUTPUT_DIR
    / "ranking_rf_crossfit_ensemble5.csv",
    index=False,
)

print()
print(
    "Ranking completo guardado:"
)

print(
    OUTPUT_DIR
    / "ranking_rf_crossfit_ensemble5.csv"
)


# ============================================================
# SUBMITS
# ============================================================

print()
print("=" * 100)
print("GENERANDO SUBMITS")
print("=" * 100)

order = np.argsort(
    -ensemble
)

for cut in SUBMIT_CUTS:

    selected = np.zeros(
        len(score_df),
        dtype="int8",
    )

    selected[
        order[:cut]
    ] = 1

    submit = pd.DataFrame({
        ID:
            score_df[ID]
            .to_numpy(),
        "Predicted":
            selected,
    })

    path = (
        OUTPUT_DIR
        / (
            "submit_rf_crossfit_"
            f"ensemble5_n{cut}.csv"
        )
    )

    submit.to_csv(
        path,
        index=False,
    )

    print(
        f"N={cut:5d} -> "
        f"{path.name}"
    )


# ============================================================
# IMPORTANCE / RF METRICS / STABILITY
# ============================================================

importance_df = pd.DataFrame(
    importance_rows
)

rf_crossfit_df = pd.DataFrame(
    rf_crossfit_rows
)

stability_df = pd.DataFrame(
    seed_stability_rows
)

importance_df.to_csv(
    OUTPUT_DIR
    / "importancia_rf_score_z561.csv",
    index=False,
)

rf_crossfit_df.to_csv(
    OUTPUT_DIR
    / "metricas_rf_crossfit_z561.csv",
    index=False,
)

stability_df.to_csv(
    OUTPUT_DIR
    / "jaccard_seeds_z561.csv",
    index=False,
)


# ============================================================
# CORRELACION ENTRE SEEDS
# ============================================================

corr_rows = []

for i in range(
    len(SEEDS)
):
    for j in range(
        i + 1,
        len(SEEDS),
    ):

        sa = SEEDS[i]
        sb = SEEDS[j]

        pearson = np.corrcoef(
            predictions[sa],
            predictions[sb],
        )[0, 1]

        spear = spearmanr(
            predictions[sa],
            predictions[sb],
        ).statistic

        corr_rows.append({
            "seed_a": sa,
            "seed_b": sb,
            "pearson":
                pearson,
            "spearman":
                spear,
        })

pd.DataFrame(
    corr_rows
).to_csv(
    OUTPUT_DIR
    / "correlaciones_seeds_z561.csv",
    index=False,
)


# ============================================================
# METADATA
# ============================================================

metadata = {
    "dataset":
        str(DATASET),
    "train_months":
        TRAIN_MONTHS,
    "score_month":
        SCORE_MONTH,
    "month_weights":
        MONTH_WEIGHTS,
    "seeds":
        SEEDS,
    "rf_params_base":
        RF_PARAMS_BASE,
    "lgb_params_base":
        LGB_PARAMS_BASE,
    "features_lgb":
        len(FEATURES_RF_LGB),
    "crossfit_strategy":
        (
            "leave-one-month-out RF "
            "within Apr-May-Jun"
        ),
    "score_strategy":
        (
            "RF trained on "
            "Apr-May-Jun -> Aug"
        ),
    "submit_cuts":
        SUBMIT_CUTS,
    "runtime_seconds":
        time.time() - t0,
}

with open(
    OUTPUT_DIR
    / "metadata_z561.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 100)
print("IMPORTANCIA RF_SCORE")
print("=" * 100)

print(
    importance_df.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.2f}",
    )
)

print()
print(
    f"Runtime total: "
    f"{time.time()-t0:.2f}s"
)

print()
print("Z561 FINALIZADO")
print("Output:", OUTPUT_DIR)
