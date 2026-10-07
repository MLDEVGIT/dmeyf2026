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
    "competencia_01/rf_crossfit_z560"
)

ID = "numero_de_cliente"
MONTH = "foto_mes"
TARGET = "clase_ternaria"

TRAIN_MONTHS = [202103, 202104, 202105]
TEST_MONTH = 202106

# Peso creciente hacia el presente.
# Es la extensión natural del esquema .75 / 1.
MONTH_WEIGHTS = {
    202103: 0.50,
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

def ganancia(y_true, prob, n):
    order = np.argsort(-prob)
    idx = order[:n]
    y_sel = y_true[idx]

    tp = int(y_sel.sum())
    fp = int(len(y_sel) - tp)

    return tp * 1_072_500 - fp * 27_500


def evaluar(y_true, prob):
    gains = {
        n: ganancia(y_true, prob, n)
        for n in CUTS
    }

    best_cut = max(gains, key=gains.get)

    return {
        "auc": roc_auc_score(y_true, prob),
        "ap": average_precision_score(y_true, prob),
        "logloss": log_loss(y_true, prob),
        "best_cut": best_cut,
        "best_gain": gains[best_cut],
        "gains": gains,
    }


def preparar_rf(frame, cols):
    return (
        frame[cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .astype("float32")
    )


def robustez(gains_base, gains_rf):
    arr = np.array(
        [
            gains_rf[n] - gains_base[n]
            for n in ROBUST_CUTS
        ],
        dtype=float,
    )

    return {
        "mean_delta": float(arr.mean()),
        "median_delta": float(np.median(arr)),
        "min_delta": float(arr.min()),
        "max_delta": float(arr.max()),
        "positive": int((arr > 0).sum()),
        "neutral": int((arr == 0).sum()),
        "negative": int((arr < 0).sum()),
    }


# ============================================================
# LOAD
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

t0 = time.time()

print("=" * 100)
print("Z560 - VALIDACION DE RF SCORE CON CROSS-FITTING POR MES")
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
        f"Esperaba 152 originales; hay {len(original_cols)}"
    )

if len(FULL457) != 457:
    raise ValueError(
        f"Esperaba FULL457=457; hay {len(FULL457)}"
    )

print(f"Originales: {len(original_cols)}")
print(f"FULL457:    {len(FULL457)}")


# ============================================================
# TRAIN / TEST
# ============================================================

train_df = (
    df[df[MONTH].isin(TRAIN_MONTHS)]
    .copy()
    .reset_index(drop=True)
)

test_df = (
    df[df[MONTH].eq(TEST_MONTH)]
    .copy()
    .reset_index(drop=True)
)

y_train = (
    train_df[TARGET]
    .eq("BAJA+2")
    .astype("int8")
    .to_numpy()
)

y_test = (
    test_df[TARGET]
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

X_test_rf = preparar_rf(
    test_df,
    original_cols,
)

month_indices = {
    m: np.flatnonzero(
        train_df[MONTH].eq(m).to_numpy()
    )
    for m in TRAIN_MONTHS
}

print()
print(
    f"Train {TRAIN_MONTHS}: "
    f"{len(train_df):,} | BAJA+2={y_train.sum():,}"
)

print(
    f"Test {TEST_MONTH}: "
    f"{len(test_df):,} | BAJA+2={y_test.sum():,}"
)

for m in TRAIN_MONTHS:
    idx = month_indices[m]

    print(
        f"  {m}: {len(idx):,} filas "
        f"| BAJA+2={y_train[idx].sum():,} "
        f"| weight={MONTH_WEIGHTS[m]:.2f}"
    )


# ============================================================
# OUTPUT STRUCTURES
# ============================================================

metric_rows = []
gain_rows = []
robust_rows = []
importance_rows = []
rf_metric_rows = []

pred_base = {}
pred_rf = {}


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

    crossfit_rows = []

    # --------------------------------------------------------
    # LEAVE-ONE-MONTH-OUT RF CROSS-FIT
    # --------------------------------------------------------

    print()
    print("RF TRAIN SCORE - LEAVE-ONE-MONTH-OUT")

    for k, score_month in enumerate(
        TRAIN_MONTHS,
        start=1,
    ):

        idx_score = month_indices[
            score_month
        ]

        idx_fit = np.concatenate(
            [
                month_indices[m]
                for m in TRAIN_MONTHS
                if m != score_month
            ]
        )

        fit_months = [
            m for m in TRAIN_MONTHS
            if m != score_month
        ]

        rf = RandomForestClassifier(
            **RF_PARAMS_BASE,
            random_state=seed + k,
        )

        rf.fit(
            X_train_rf.iloc[idx_fit],
            y_train[idx_fit],
            sample_weight=
                weights_train[idx_fit],
        )

        pred = rf.predict_proba(
            X_train_rf.iloc[idx_score]
        )[:, 1]

        rf_train_score[
            idx_score
        ] = pred

        auc = roc_auc_score(
            y_train[idx_score],
            pred,
        )

        ap = average_precision_score(
            y_train[idx_score],
            pred,
        )

        crossfit_rows.append({
            "seed": seed,
            "score_month": score_month,
            "fit_months": "+".join(
                str(x) for x in fit_months
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
    # RF FINAL: MAR+APR+MAY -> JUN
    # --------------------------------------------------------

    rf_final = RandomForestClassifier(
        **RF_PARAMS_BASE,
        random_state=seed + 100,
    )

    rf_final.fit(
        X_train_rf,
        y_train,
        sample_weight=weights_train,
    )

    rf_test_score = (
        rf_final.predict_proba(
            X_test_rf
        )[:, 1]
        .astype("float32")
    )

    rf_train_auc = roc_auc_score(
        y_train,
        rf_train_score,
    )

    rf_train_ap = average_precision_score(
        y_train,
        rf_train_score,
    )

    rf_test_auc = roc_auc_score(
        y_test,
        rf_test_score,
    )

    rf_test_ap = average_precision_score(
        y_test,
        rf_test_score,
    )

    print()
    print(
        f"RF crossfit train "
        f"AUC={rf_train_auc:.6f} "
        f"AP={rf_train_ap:.6f}"
    )

    print(
        f"RF junio "
        f"AUC={rf_test_auc:.6f} "
        f"AP={rf_test_ap:.6f}"
    )

    rf_metric_rows.extend(
        crossfit_rows
    )

    rf_metric_rows.append({
        "seed": seed,
        "score_month": TEST_MONTH,
        "fit_months": "+".join(
            str(x) for x in TRAIN_MONTHS
        ),
        "auc": rf_test_auc,
        "ap": rf_test_ap,
    })

    del rf_final
    gc.collect()

    # --------------------------------------------------------
    # LGBM
    # --------------------------------------------------------

    train_df["rf_score"] = (
        rf_train_score
    )

    test_df["rf_score"] = (
        rf_test_score
    )

    results = {}

    for experiment, features in [
        ("BASE", FULL457),
        (
            "RF_SCORE",
            FULL457 + ["rf_score"],
        ),
    ]:

        model = lgb.LGBMClassifier(
            **{
                **LGB_PARAMS_BASE,
                "random_state": seed,
            }
        )

        model.fit(
            train_df[features],
            y_train,
            sample_weight=
                weights_train,
        )

        prob = model.predict_proba(
            test_df[features]
        )[:, 1]

        res = evaluar(
            y_test,
            prob,
        )

        results[experiment] = res

        if experiment == "BASE":
            pred_base[seed] = prob.copy()
        else:
            pred_rf[seed] = prob.copy()

        metric_rows.append({
            "seed": seed,
            "experiment": experiment,
            "auc": res["auc"],
            "ap": res["ap"],
            "logloss": res["logloss"],
            "best_cut": res["best_cut"],
            "best_gain": res["best_gain"],
        })

        for cut, gain in res[
            "gains"
        ].items():

            gain_rows.append({
                "seed": seed,
                "experiment": experiment,
                "cut": cut,
                "gain": gain,
            })

        print()
        print(
            f"{experiment:8s} "
            f"AUC={res['auc']:.6f} "
            f"AP={res['ap']:.6f} "
            f"LL={res['logloss']:.6f} "
            f"BEST="
            f"{res['best_gain']/1e6:.2f}M"
            f"@{res['best_cut']}"
        )

        if experiment == "RF_SCORE":

            imp = pd.DataFrame({
                "feature": features,
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

            row = imp[
                imp["feature"].eq(
                    "rf_score"
                )
            ].iloc[0]

            importance_rows.append({
                "seed": seed,
                "gain_importance":
                    float(
                        row["gain_importance"]
                    ),
                "rank":
                    int(row["rank"]),
            })

            print(
                f"rf_score: gain="
                f"{row['gain_importance']:.2f} "
                f"rank={int(row['rank'])}"
            )

        del model
        gc.collect()

    rob = robustez(
        results["BASE"]["gains"],
        results["RF_SCORE"]["gains"],
    )

    robust_rows.append({
        "seed": seed,
        "type": "SEED",
        **rob,
    })

    print()
    print(
        "RF_SCORE vs BASE 10k-14k: "
        f"mean={rob['mean_delta']/1e6:+.2f}M "
        f"median={rob['median_delta']/1e6:+.2f}M "
        f"min={rob['min_delta']/1e6:+.2f}M "
        f"max={rob['max_delta']/1e6:+.2f}M "
        f"+/0/-="
        f"{rob['positive']}/"
        f"{rob['neutral']}/"
        f"{rob['negative']}"
    )

    print(
        f"Seed runtime: "
        f"{time.time()-seed_t0:.1f}s"
    )

    del (
        rf_train_score,
        rf_test_score,
        results,
    )

    gc.collect()


# ============================================================
# ENSEMBLE
# ============================================================

print()
print("=" * 100)
print("ENSEMBLE 5 SEEDS")
print("=" * 100)

ensemble_base = np.mean(
    np.column_stack(
        [pred_base[s] for s in SEEDS]
    ),
    axis=1,
)

ensemble_rf = np.mean(
    np.column_stack(
        [pred_rf[s] for s in SEEDS]
    ),
    axis=1,
)

ensemble_results = {}

for experiment, prob in [
    ("BASE", ensemble_base),
    ("RF_SCORE", ensemble_rf),
]:

    res = evaluar(
        y_test,
        prob,
    )

    ensemble_results[
        experiment
    ] = res

    metric_rows.append({
        "seed": "ENSEMBLE5",
        "experiment": experiment,
        "auc": res["auc"],
        "ap": res["ap"],
        "logloss": res["logloss"],
        "best_cut": res["best_cut"],
        "best_gain": res["best_gain"],
    })

    for cut, gain in res[
        "gains"
    ].items():

        gain_rows.append({
            "seed": "ENSEMBLE5",
            "experiment": experiment,
            "cut": cut,
            "gain": gain,
        })

    print(
        f"{experiment:8s} "
        f"AUC={res['auc']:.6f} "
        f"AP={res['ap']:.6f} "
        f"LL={res['logloss']:.6f} "
        f"BEST="
        f"{res['best_gain']/1e6:.2f}M"
        f"@{res['best_cut']}"
    )


rob_ensemble = robustez(
    ensemble_results["BASE"]["gains"],
    ensemble_results["RF_SCORE"]["gains"],
)

robust_rows.append({
    "seed": "ENSEMBLE5",
    "type": "ENSEMBLE",
    **rob_ensemble,
})

print()
print(
    "ENSEMBLE RF_SCORE vs BASE 10k-14k: "
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
# TOP-N SEED STABILITY
# ============================================================

print()
print("=" * 100)
print("ESTABILIDAD TOP-N")
print("=" * 100)

jaccard_rows = []

for experiment, preds in [
    ("BASE", pred_base),
    ("RF_SCORE", pred_rf),
]:

    rankings = {
        s: np.argsort(-preds[s])
        for s in SEEDS
    }

    for cut in [
        10000,
        12000,
        14000,
    ]:

        vals = []

        for i in range(len(SEEDS)):
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
            f"{experiment:8s} "
            f"N={cut:5d} "
            f"mean={np.mean(vals):.6f} "
            f"min={np.min(vals):.6f} "
            f"max={np.max(vals):.6f}"
        )


# ============================================================
# SAVE
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

rf_metrics_df = pd.DataFrame(
    rf_metric_rows
)

jaccard_df = pd.DataFrame(
    jaccard_rows
)

metrics_df.to_csv(
    OUTPUT_DIR / "metricas_z560.csv",
    index=False,
)

gains_df.to_csv(
    OUTPUT_DIR / "ganancias_z560.csv",
    index=False,
)

robust_df.to_csv(
    OUTPUT_DIR / "robustez_z560.csv",
    index=False,
)

importance_df.to_csv(
    OUTPUT_DIR / "importancia_rf_score_z560.csv",
    index=False,
)

rf_metrics_df.to_csv(
    OUTPUT_DIR / "metricas_rf_crossfit_z560.csv",
    index=False,
)

jaccard_df.to_csv(
    OUTPUT_DIR / "jaccard_seeds_z560.csv",
    index=False,
)


# ============================================================
# DETAILED DELTAS
# ============================================================

delta_rows = []

for seed in [
    *SEEDS,
    "ENSEMBLE5",
]:

    gb = (
        gains_df[
            (gains_df["seed"] == seed)
            &
            (
                gains_df["experiment"]
                == "BASE"
            )
        ]
        .set_index("cut")["gain"]
    )

    gr = (
        gains_df[
            (gains_df["seed"] == seed)
            &
            (
                gains_df["experiment"]
                == "RF_SCORE"
            )
        ]
        .set_index("cut")["gain"]
    )

    for cut in CUTS:

        delta_rows.append({
            "seed": seed,
            "cut": cut,
            "gain_base":
                int(gb.loc[cut]),
            "gain_rf":
                int(gr.loc[cut]),
            "delta":
                int(
                    gr.loc[cut]
                    - gb.loc[cut]
                ),
        })

pd.DataFrame(
    delta_rows
).to_csv(
    OUTPUT_DIR
    / "deltas_rf_vs_base_z560.csv",
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 100)
print("RESUMEN ROBUSTEZ")
print("=" * 100)

tmp = robust_df.copy()

for c in [
    "mean_delta",
    "median_delta",
    "min_delta",
    "max_delta",
]:
    tmp[c] = tmp[c] / 1e6

print(
    tmp.to_string(
        index=False,
        float_format=lambda x:
            f"{x:+.3f}",
    )
)

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


metadata = {
    "dataset": str(DATASET),
    "train_months": TRAIN_MONTHS,
    "test_month": TEST_MONTH,
    "month_weights": MONTH_WEIGHTS,
    "seeds": SEEDS,
    "rf_params_base": RF_PARAMS_BASE,
    "lgb_params_base": LGB_PARAMS_BASE,
    "robust_cuts": ROBUST_CUTS,
    "crossfit_strategy":
        "leave-one-month-out RF",
    "runtime_seconds":
        time.time() - t0,
}

with open(
    OUTPUT_DIR / "metadata_z560.json",
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
print("Z560 FINALIZADO")
print("Output:", OUTPUT_DIR)
