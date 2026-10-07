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
from sklearn.model_selection import StratifiedKFold


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
    "competencia_01/rf_score_auxiliar_z556"
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

SEED = 290497

CUTS = list(range(8000, 16001, 500))
ROBUST_CUTS = list(range(10000, 14001, 500))

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

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_leaf": 100,
    "max_features": "sqrt",
    "class_weight": "balanced_subsample",
    "random_state": SEED,
    "n_jobs": -1,
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


def evaluate(y, prob):
    gains = {
        n: ganancia(y, prob, n)
        for n in CUTS
    }

    best_cut = max(
        gains,
        key=gains.get,
    )

    return {
        "auc":
            roc_auc_score(y, prob),
        "ap":
            average_precision_score(y, prob),
        "logloss":
            log_loss(y, prob),
        "best_cut":
            best_cut,
        "best_gain":
            gains[best_cut],
        "gains":
            gains,
    }


# ============================================================
# LOAD
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

t0 = time.time()

print("=" * 95)
print("Z556 - RANDOM FOREST AUXILIAR COMO FEATURE")
print("=" * 95)

df = pd.read_parquet(DATASET)

print(
    f"Dataset: {len(df):,} filas, "
    f"{df.shape[1]} columnas"
)


# ============================================================
# IDENTIFICAR 152 VARIABLES ORIGINALES
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

print()
print(
    f"Originales: {len(original_cols)}"
)
print(
    f"FULL457:    {len(FULL457)}"
)

if len(original_cols) != 152:
    raise ValueError(
        f"Esperaba 152 originales, "
        f"hay {len(original_cols)}"
    )

if len(FULL457) != 457:
    raise ValueError(
        f"Esperaba FULL457=457, "
        f"hay {len(FULL457)}"
    )


# ============================================================
# TRAIN / TEST
# ============================================================

train_mask = df[
    MONTH
].isin(TRAIN_MONTHS)

test_mask = df[
    MONTH
].eq(TEST_MONTH)

train_idx = np.flatnonzero(
    train_mask.to_numpy()
)

test_idx = np.flatnonzero(
    test_mask.to_numpy()
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

weights_train = (
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
    f"Train: {len(train_idx):,} "
    f"| BAJA+2={y_train.sum():,}"
)

print(
    f"Test:  {len(test_idx):,} "
    f"| BAJA+2={y_test.sum():,}"
)


# ============================================================
# PREPARAR MATRICES RF
#
# RandomForest sklearn no admite NaN en todas las versiones.
# Imputamos con 0 únicamente para el modelo auxiliar.
# No modificamos FULL457 del LightGBM.
# ============================================================

print()
print("=" * 95)
print("PREPARANDO MATRICES RF")
print("=" * 95)

X_rf_train = (
    df.loc[
        train_mask,
        original_cols,
    ]
    .replace(
        [np.inf, -np.inf],
        np.nan,
    )
    .fillna(0)
    .astype("float32")
)

X_rf_test = (
    df.loc[
        test_mask,
        original_cols,
    ]
    .replace(
        [np.inf, -np.inf],
        np.nan,
    )
    .fillna(0)
    .astype("float32")
)

print(
    f"RF train: {X_rf_train.shape}"
)
print(
    f"RF test:  {X_rf_test.shape}"
)


# ============================================================
# OOF RF SCORE PARA TRAIN
# ============================================================

print()
print("=" * 95)
print("GENERANDO RF_SCORE OOF - 5 FOLDS")
print("=" * 95)

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED,
)

rf_oof = np.zeros(
    len(y_train),
    dtype="float32",
)

fold_rows = []

for fold, (
    idx_fit,
    idx_val,
) in enumerate(
    skf.split(
        X_rf_train,
        y_train,
    ),
    start=1,
):

    fold_t0 = time.time()

    print()
    print(
        f"Fold {fold}/5 "
        f"| fit={len(idx_fit):,} "
        f"| val={len(idx_val):,}"
    )

    rf = RandomForestClassifier(
        **{
            **RF_PARAMS,
            "random_state":
                SEED + fold,
        }
    )

    rf.fit(
        X_rf_train.iloc[
            idx_fit
        ],
        y_train[idx_fit],
        sample_weight=
            weights_train[idx_fit],
    )

    pred_val = rf.predict_proba(
        X_rf_train.iloc[
            idx_val
        ]
    )[:, 1]

    rf_oof[
        idx_val
    ] = pred_val

    fold_auc = roc_auc_score(
        y_train[idx_val],
        pred_val,
    )

    fold_ap = (
        average_precision_score(
            y_train[idx_val],
            pred_val,
        )
    )

    fold_rows.append({
        "fold": fold,
        "auc": fold_auc,
        "ap": fold_ap,
        "runtime_seconds":
            time.time() - fold_t0,
    })

    print(
        f"AUC={fold_auc:.6f} "
        f"AP={fold_ap:.6f} "
        f"time="
        f"{time.time()-fold_t0:.1f}s"
    )

    del rf
    gc.collect()


print()
print(
    "RF OOF global: "
    f"AUC="
    f"{roc_auc_score(y_train, rf_oof):.6f} "
    f"AP="
    f"{average_precision_score(y_train, rf_oof):.6f}"
)


# ============================================================
# RF FINAL PARA JUNIO
# ============================================================

print()
print("=" * 95)
print("RF FINAL TRAIN ABR+MAY -> TEST JUN")
print("=" * 95)

rf_final = RandomForestClassifier(
    **RF_PARAMS
)

rf_final.fit(
    X_rf_train,
    y_train,
    sample_weight=weights_train,
)

rf_test = (
    rf_final.predict_proba(
        X_rf_test
    )[:, 1]
    .astype("float32")
)

print(
    f"RF junio AUC="
    f"{roc_auc_score(y_test, rf_test):.6f} "
    f"AP="
    f"{average_precision_score(y_test, rf_test):.6f}"
)


# ============================================================
# INSERTAR FEATURE
# ============================================================

df["rf_score"] = np.nan

df.loc[
    train_mask,
    "rf_score",
] = rf_oof

df.loc[
    test_mask,
    "rf_score",
] = rf_test

RF458 = (
    FULL457
    + ["rf_score"]
)


# ============================================================
# LIGHTGBM BASE vs RF_SCORE
# ============================================================

print()
print("=" * 95)
print("LIGHTGBM BASE vs RF_SCORE")
print("=" * 95)

results = {}
gain_rows = []
metric_rows = []

for experiment, features in [
    ("BASE", FULL457),
    ("RF_SCORE", RF458),
]:

    print()
    print(
        f"{experiment}: "
        f"{len(features)} features"
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
        sample_weight=weights_train,
    )

    prob = model.predict_proba(
        df.loc[
            test_mask,
            features,
        ]
    )[:, 1]

    res = evaluate(
        y_test,
        prob,
    )

    results[
        experiment
    ] = res

    metric_rows.append({
        "experiment":
            experiment,
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

    if experiment == "RF_SCORE":

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

        rf_imp = imp[
            imp["feature"]
            .eq("rf_score")
        ]

        print()
        print("Importancia rf_score:")
        print(
            rf_imp.to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.2f}",
            )
        )

        imp.to_csv(
            OUTPUT_DIR
            / "importancias_rf_score_z556.csv",
            index=False,
        )

    del model
    gc.collect()


# ============================================================
# ROBUSTEZ ECONÓMICA
# ============================================================

print()
print("=" * 95)
print("RF_SCORE vs BASE - ZONA 10k-14k")
print("=" * 95)

delta_rows = []

for cut in CUTS:

    gb = results[
        "BASE"
    ]["gains"][cut]

    gr = results[
        "RF_SCORE"
    ]["gains"][cut]

    delta = gr - gb

    delta_rows.append({
        "cut": cut,
        "gain_base": gb,
        "gain_rf_score": gr,
        "delta": delta,
    })

    if cut in ROBUST_CUTS:
        print(
            f"N={cut:5d} "
            f"BASE={gb/1e6:7.2f}M "
            f"RF={gr/1e6:7.2f}M "
            f"delta={delta/1e6:+7.2f}M"
        )


rob = np.array(
    [
        x["delta"]
        for x in delta_rows
        if x["cut"]
        in ROBUST_CUTS
    ],
    dtype=float,
)

print()
print(
    f"mean={rob.mean()/1e6:+.2f}M "
    f"median={np.median(rob)/1e6:+.2f}M "
    f"min={rob.min()/1e6:+.2f}M "
    f"max={rob.max()/1e6:+.2f}M "
    f"+/0/-="
    f"{(rob>0).sum()}/"
    f"{(rob==0).sum()}/"
    f"{(rob<0).sum()}"
)


# ============================================================
# GUARDAR
# ============================================================

pd.DataFrame(
    metric_rows
).to_csv(
    OUTPUT_DIR
    / "metricas_z556.csv",
    index=False,
)

pd.DataFrame(
    gain_rows
).to_csv(
    OUTPUT_DIR
    / "ganancias_z556.csv",
    index=False,
)

pd.DataFrame(
    delta_rows
).to_csv(
    OUTPUT_DIR
    / "deltas_rf_vs_base_z556.csv",
    index=False,
)

pd.DataFrame(
    fold_rows
).to_csv(
    OUTPUT_DIR
    / "rf_oof_folds_z556.csv",
    index=False,
)


# Guardamos scores para análisis posterior
scores_test = pd.DataFrame({
    ID:
        df.loc[
            test_mask,
            ID,
        ].to_numpy(),
    "foto_mes":
        TEST_MONTH,
    "rf_score":
        rf_test,
    "target":
        y_test,
})

scores_test.to_csv(
    OUTPUT_DIR
    / "rf_scores_junio_z556.csv",
    index=False,
)


metadata = {
    "dataset":
        str(DATASET),
    "train_months":
        TRAIN_MONTHS,
    "test_month":
        TEST_MONTH,
    "seed":
        SEED,
    "rf_params":
        RF_PARAMS,
    "lgb_params":
        LGB_PARAMS,
    "runtime_seconds":
        time.time() - t0,
}

with open(
    OUTPUT_DIR
    / "metadata_z556.json",
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
print("Z556 FINALIZADO")
print("Output:", OUTPUT_DIR)
