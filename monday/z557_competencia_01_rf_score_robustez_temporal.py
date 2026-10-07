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
    "competencia_01/rf_score_temporal_z557"
)

ID = "numero_de_cliente"
MONTH = "foto_mes"
TARGET = "clase_ternaria"

SEED = 290497

CUTS = list(range(8000, 16001, 500))
ROBUST_CUTS = list(range(10000, 14001, 500))

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
        "train_months": [202103, 202104],
        "test_month": 202105,
        "weights": {
            202103: 0.75,
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


def preparar_rf(df_part, cols):
    return (
        df_part[cols]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
        .astype("float32")
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
print("Z557 - RF_SCORE: ROBUSTEZ TEMPORAL ESTRICTA")
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

print(f"Originales: {len(original_cols)}")
print(f"FULL457:    {len(FULL457)}")


# ============================================================
# RESULTADOS
# ============================================================

metric_rows = []
gain_rows = []
robust_rows = []
rf_rows = []
importance_rows = []


# ============================================================
# LOOP TEMPORAL
# ============================================================

for wi, w in enumerate(WINDOWS, start=1):

    window = w["window"]
    train_months = w["train_months"]
    test_month = w["test_month"]
    month_weights = w["weights"]

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

    train_df = df.loc[
        train_mask
    ].copy()

    test_df = df.loc[
        test_mask
    ].copy()

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
        .map(month_weights)
        .astype(float)
        .to_numpy()
    )

    print(
        f"Train: {len(train_df):,} "
        f"| BAJA+2={y_train.sum():,}"
    )

    print(
        f"Test:  {len(test_df):,} "
        f"| BAJA+2={y_test.sum():,}"
    )


    # ========================================================
    # MATRICES RF
    # ========================================================

    X_train_rf = preparar_rf(
        train_df,
        original_cols,
    )

    X_test_rf = preparar_rf(
        test_df,
        original_cols,
    )

    rf_train_score = np.zeros(
        len(train_df),
        dtype="float32",
    )


    # ========================================================
    # SCORE DEL TRAIN
    #
    # 1 mes:
    #   OOF estratificado interno.
    #
    # 2 meses:
    #   cada mes se predice con RF entrenado
    #   exclusivamente en el otro mes.
    # ========================================================

    if len(train_months) == 1:

        print()
        print(
            "RF train score: "
            "OOF interno 5 folds "
            "(train de un solo mes)"
        )

        skf = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=SEED,
        )

        for fold, (
            idx_fit,
            idx_val,
        ) in enumerate(
            skf.split(
                X_train_rf,
                y_train,
            ),
            start=1,
        ):

            ft0 = time.time()

            rf = RandomForestClassifier(
                **{
                    **RF_PARAMS,
                    "random_state":
                        SEED + fold,
                }
            )

            rf.fit(
                X_train_rf.iloc[
                    idx_fit
                ],
                y_train[idx_fit],
                sample_weight=
                    weights_train[idx_fit],
            )

            pred = rf.predict_proba(
                X_train_rf.iloc[
                    idx_val
                ]
            )[:, 1]

            rf_train_score[
                idx_val
            ] = pred

            print(
                f"  fold {fold}/5 "
                f"AUC="
                f"{roc_auc_score(y_train[idx_val], pred):.6f} "
                f"AP="
                f"{average_precision_score(y_train[idx_val], pred):.6f} "
                f"time="
                f"{time.time()-ft0:.1f}s"
            )

            del rf
            gc.collect()

    else:

        print()
        print(
            "RF train score: "
            "cross-month estricto"
        )

        if len(train_months) != 2:
            raise ValueError(
                "Este diseño espera "
                "1 o 2 meses de train"
            )

        m1, m2 = train_months

        for score_month, fit_month in [
            (m1, m2),
            (m2, m1),
        ]:

            idx_fit = np.flatnonzero(
                train_df[
                    MONTH
                ].eq(
                    fit_month
                ).to_numpy()
            )

            idx_score = np.flatnonzero(
                train_df[
                    MONTH
                ].eq(
                    score_month
                ).to_numpy()
            )

            print(
                f"  score {score_month} "
                f"<- RF entrenado en "
                f"{fit_month}"
            )

            rf = RandomForestClassifier(
                **{
                    **RF_PARAMS,
                    "random_state":
                        SEED
                        + int(score_month)
                        % 1000,
                }
            )

            rf.fit(
                X_train_rf.iloc[
                    idx_fit
                ],
                y_train[idx_fit],
                sample_weight=
                    weights_train[idx_fit],
            )

            pred = rf.predict_proba(
                X_train_rf.iloc[
                    idx_score
                ]
            )[:, 1]

            rf_train_score[
                idx_score
            ] = pred

            print(
                f"    AUC="
                f"{roc_auc_score(y_train[idx_score], pred):.6f} "
                f"AP="
                f"{average_precision_score(y_train[idx_score], pred):.6f}"
            )

            del rf
            gc.collect()


    # ========================================================
    # RF FINAL: TODO TRAIN -> TEST
    # ========================================================

    print()
    print(
        f"RF final {train_months} "
        f"-> {test_month}"
    )

    rf_final = RandomForestClassifier(
        **{
            **RF_PARAMS,
            "random_state":
                SEED + 100 + wi,
        }
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

    rf_train_auc = (
        roc_auc_score(
            y_train,
            rf_train_score,
        )
    )

    rf_train_ap = (
        average_precision_score(
            y_train,
            rf_train_score,
        )
    )

    rf_test_auc = (
        roc_auc_score(
            y_test,
            rf_test_score,
        )
    )

    rf_test_ap = (
        average_precision_score(
            y_test,
            rf_test_score,
        )
    )

    print(
        f"RF train-score "
        f"AUC={rf_train_auc:.6f} "
        f"AP={rf_train_ap:.6f}"
    )

    print(
        f"RF test "
        f"AUC={rf_test_auc:.6f} "
        f"AP={rf_test_ap:.6f}"
    )

    rf_rows.append({
        "window": window,
        "test_month": test_month,
        "rf_train_auc":
            rf_train_auc,
        "rf_train_ap":
            rf_train_ap,
        "rf_test_auc":
            rf_test_auc,
        "rf_test_ap":
            rf_test_ap,
    })


    # ========================================================
    # LGBM
    # ========================================================

    train_df[
        "rf_score"
    ] = rf_train_score

    test_df[
        "rf_score"
    ] = rf_test_score

    experiments = {
        "BASE":
            FULL457,
        "RF_SCORE":
            FULL457
            + ["rf_score"],
    }

    results = {}

    print()
    print("-" * 100)
    print("LIGHTGBM")
    print("-" * 100)

    for experiment, features in (
        experiments.items()
    ):

        model = lgb.LGBMClassifier(
            **LGB_PARAMS
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

        results[
            experiment
        ] = res

        metric_rows.append({
            "window": window,
            "test_month":
                test_month,
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
                .eq("rf_score")
            ].iloc[0]

            importance_rows.append({
                "window":
                    window,
                "test_month":
                    test_month,
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
                f"           rf_score "
                f"importance="
                f"{row['gain_importance']:.2f} "
                f"rank="
                f"{int(row['rank'])}"
            )

        del model
        gc.collect()


    # ========================================================
    # ROBUSTEZ
    # ========================================================

    deltas = []

    print()
    print(
        "RF_SCORE vs BASE "
        "10k-14k"
    )

    for cut in ROBUST_CUTS:

        gb = results[
            "BASE"
        ]["gains"][cut]

        gr = results[
            "RF_SCORE"
        ]["gains"][cut]

        d = gr - gb
        deltas.append(d)

        print(
            f"N={cut:5d} "
            f"delta="
            f"{d/1e6:+7.2f}M"
        )

    arr = np.array(
        deltas,
        dtype=float,
    )

    rob = {
        "window":
            window,
        "test_month":
            test_month,
        "mean_delta":
            arr.mean(),
        "median_delta":
            np.median(arr),
        "min_delta":
            arr.min(),
        "max_delta":
            arr.max(),
        "positive":
            int(
                (arr > 0).sum()
            ),
        "neutral":
            int(
                (arr == 0).sum()
            ),
        "negative":
            int(
                (arr < 0).sum()
            ),
    }

    robust_rows.append(rob)

    print(
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

    del (
        train_df,
        test_df,
        X_train_rf,
        X_test_rf,
        rf_final,
        rf_train_score,
        rf_test_score,
        results,
    )

    gc.collect()


# ============================================================
# OUTPUT
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

rf_df = pd.DataFrame(
    rf_rows
)

importance_df = pd.DataFrame(
    importance_rows
)

metrics_df.to_csv(
    OUTPUT_DIR
    / "metricas_z557.csv",
    index=False,
)

gains_df.to_csv(
    OUTPUT_DIR
    / "ganancias_z557.csv",
    index=False,
)

robust_df.to_csv(
    OUTPUT_DIR
    / "robustez_z557.csv",
    index=False,
)

rf_df.to_csv(
    OUTPUT_DIR
    / "metricas_rf_z557.csv",
    index=False,
)

importance_df.to_csv(
    OUTPUT_DIR
    / "importancia_rf_score_z557.csv",
    index=False,
)


# ============================================================
# RESUMEN AGREGADO
# ============================================================

print()
print("=" * 100)
print("RESUMEN ROBUSTEZ TEMPORAL")
print("=" * 100)

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

all_deltas = []

for _, row in gains_df[
    gains_df[
        "experiment"
    ].eq("RF_SCORE")
    &
    gains_df[
        "cut"
    ].isin(ROBUST_CUTS)
].iterrows():

    gb = gains_df[
        gains_df[
            "window"
        ].eq(row["window"])
        &
        gains_df[
            "experiment"
        ].eq("BASE")
        &
        gains_df[
            "cut"
        ].eq(row["cut"])
    ]["gain"].iloc[0]

    all_deltas.append(
        row["gain"] - gb
    )

all_deltas = np.array(
    all_deltas,
    dtype=float,
)

print()
print("AGREGADO 3 OOT:")
print(
    f"mean="
    f"{all_deltas.mean()/1e6:+.2f}M "
    f"median="
    f"{np.median(all_deltas)/1e6:+.2f}M "
    f"min="
    f"{all_deltas.min()/1e6:+.2f}M "
    f"max="
    f"{all_deltas.max()/1e6:+.2f}M "
    f"+/0/-="
    f"{(all_deltas>0).sum()}/"
    f"{(all_deltas==0).sum()}/"
    f"{(all_deltas<0).sum()}"
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


# ============================================================
# METADATA
# ============================================================

metadata = {
    "dataset":
        str(DATASET),
    "seed":
        SEED,
    "windows":
        WINDOWS,
    "rf_params":
        RF_PARAMS,
    "lgb_params":
        LGB_PARAMS,
    "robust_cuts":
        ROBUST_CUTS,
    "train_score_strategy":
        {
            "one_month":
                "5-fold stratified OOF",
            "two_months":
                "strict cross-month scoring",
        },
    "runtime_seconds":
        time.time() - t0,
}

with open(
    OUTPUT_DIR
    / "metadata_z557.json",
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
print("Z557 FINALIZADO")
print("Output:", OUTPUT_DIR)
