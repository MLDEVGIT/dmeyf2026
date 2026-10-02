"""
z509_seleccion_media2.py

DMEyF 2026
Selección incremental de features MEDIA2.

Objetivo
--------
Z508 mostró que:

    A0 = t0
    A2 = t0 + media2

mejora ligeramente la capacidad predictiva out-of-time al incorporar
la media de las dos últimas observaciones.

Z509 estudia si esa mejora requiere TODAS las variables media2 o si
puede obtenerse con un subconjunto pequeño.

IMPORTANTE
----------
La selección de media2 se realiza EXCLUSIVAMENTE utilizando TRAIN.

TEST no participa:
- en selección de variables;
- en ranking;
- en eliminación de columnas;
- en entrenamiento.

Diseño temporal
---------------
TRAIN = 202104
TEST  = 202105

Experimentos
------------
B0_t0
B1_t0_media2_top10
B2_t0_media2_top20
B3_t0_media2_top40
B4_t0_media2_top80
B5_t0_media2_all

La selección Top-N se obtiene a partir de importancia gain calculada
en TRAIN mediante modelos LightGBM entrenados únicamente en TRAIN.

Además se estudia la estabilidad del ranking entre semillas.
"""

from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier

from sklearn.metrics import (
    average_precision_score,
    log_loss,
    roc_auc_score,
)

warnings.filterwarnings("ignore")


# ======================================================================
# CONFIGURACIÓN
# ======================================================================

INPUT = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/"
    "prebaja_features/features_prebaja.csv"
)

OUT_DIR = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/"
    "seleccion_media2"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TRAIN_FOTO = 202104
TEST_FOTO = 202105

SEEDS = [
    100,
    200,
    300,
    400,
    500,
]

TOP_N = [
    10,
    20,
    40,
    80,
]

PARAMS = {
    "objective": "binary",
    "n_estimators": 500,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "verbosity": -1,
}


# ======================================================================
# UTILIDADES
# ======================================================================

def titulo(txt):
    print()
    print("=" * 78)
    print(txt)
    print("=" * 78)


def subtitulo(txt):
    print()
    print("-" * 78)
    print(txt)
    print("-" * 78)


def construir_target(df):
    df = df.copy()

    df["target"] = (
        df["grupo"]
        .eq("BAJA+2")
        .astype("int8")
    )

    return df


def columnas_validas_train(
    train,
    columnas,
):
    """
    Una columna se conserva si TRAIN contiene
    algún valor no nulo y más de un valor distinto.

    TEST jamás interviene en esta decisión.
    """

    validas = []
    eliminadas = []

    for col in columnas:

        x = pd.to_numeric(
            train[col],
            errors="coerce",
        )

        if x.notna().sum() == 0:
            eliminadas.append(
                (col, "all_na")
            )
            continue

        if x.nunique(
            dropna=True
        ) <= 1:
            eliminadas.append(
                (col, "constante")
            )
            continue

        validas.append(col)

    return validas, eliminadas


def preparar_xy(
    train,
    test,
    columnas,
):
    """
    Convierte exclusivamente las columnas elegidas
    a float32.

    LightGBM maneja NaN de forma nativa.
    """

    X_train = (
        train[columnas]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .astype("float32")
    )

    X_test = (
        test[columnas]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .astype("float32")
    )

    y_train = (
        train["target"]
        .to_numpy(
            dtype="int8"
        )
    )

    y_test = (
        test["target"]
        .to_numpy(
            dtype="int8"
        )
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


def nuevo_modelo(seed):
    params = PARAMS.copy()

    params["random_state"] = seed

    return LGBMClassifier(
        **params
    )


# ======================================================================
# CARGA
# ======================================================================

def cargar_datos():

    titulo(
        "CARGA DE FEATURE MATRIX"
    )

    print(INPUT)

    t0 = time.time()

    df = pd.read_csv(INPUT)

    print(
        f"Dimensiones : "
        f"{len(df):,} × "
        f"{df.shape[1]:,}"
    )

    print(
        f"Tiempo carga: "
        f"{time.time() - t0:.2f} s"
    )

    necesarias = {
        "numero_de_cliente",
        "grupo",
        "foto_ancla",
        "cluster",
    }

    faltantes = (
        necesarias
        - set(df.columns)
    )

    if faltantes:
        raise ValueError(
            f"Faltan columnas: "
            f"{sorted(faltantes)}"
        )

    df = construir_target(df)

    return df


# ======================================================================
# PARTICIÓN TEMPORAL
# ======================================================================

def particion_temporal(df):

    titulo(
        "PARTICIÓN TEMPORAL"
    )

    train = df[
        df["foto_ancla"]
        == TRAIN_FOTO
    ].copy()

    test = df[
        df["foto_ancla"]
        == TEST_FOTO
    ].copy()

    if train.empty:
        raise ValueError(
            "TRAIN vacío."
        )

    if test.empty:
        raise ValueError(
            "TEST vacío."
        )

    print(
        f"TRAIN {TRAIN_FOTO}: "
        f"{len(train):,}"
    )

    print(
        train["grupo"]
        .value_counts()
    )

    print()

    print(
        f"TEST {TEST_FOTO}: "
        f"{len(test):,}"
    )

    print(
        test["grupo"]
        .value_counts()
    )

    print()

    print(
        "Clientes TRAIN:",
        f"{train['numero_de_cliente'].nunique():,}"
    )

    print(
        "Clientes TEST :",
        f"{test['numero_de_cliente'].nunique():,}"
    )

    return train, test


# ======================================================================
# DETECCIÓN DE FAMILIAS
# ======================================================================

def detectar_features(
    train,
):

    titulo(
        "DETECCIÓN DE FEATURES"
    )

    t0_cols = sorted(
        [
            c
            for c in train.columns
            if c.endswith("__t0")
        ]
    )

    media2_cols = sorted(
        [
            c
            for c in train.columns
            if c.endswith("__media2")
        ]
    )

    print(
        f"t0 declaradas     : "
        f"{len(t0_cols):,}"
    )

    print(
        f"media2 declaradas : "
        f"{len(media2_cols):,}"
    )

    t0_validas, t0_elim = (
        columnas_validas_train(
            train,
            t0_cols,
        )
    )

    media_validas, media_elim = (
        columnas_validas_train(
            train,
            media2_cols,
        )
    )

    print()
    print(
        f"t0 válidas TRAIN     : "
        f"{len(t0_validas):,}"
    )

    print(
        f"media2 válidas TRAIN : "
        f"{len(media_validas):,}"
    )

    print()
    print(
        f"t0 eliminadas        : "
        f"{len(t0_elim):,}"
    )

    print(
        f"media2 eliminadas    : "
        f"{len(media_elim):,}"
    )

    eliminadas = []

    for col, motivo in t0_elim:
        eliminadas.append(
            {
                "familia": "t0",
                "feature": col,
                "motivo": motivo,
            }
        )

    for col, motivo in media_elim:
        eliminadas.append(
            {
                "familia": "media2",
                "feature": col,
                "motivo": motivo,
            }
        )

    pd.DataFrame(
        eliminadas
    ).to_csv(
        OUT_DIR
        / "columnas_eliminadas_train.csv",
        index=False,
    )

    return (
        t0_validas,
        media_validas,
    )


# ======================================================================
# RANKING MEDIA2 — SÓLO TRAIN
# ======================================================================

def ranking_media2_train(
    train,
    t0_cols,
    media2_cols,
):

    titulo(
        "RANKING MEDIA2 — EXCLUSIVAMENTE TRAIN"
    )

    columnas = (
        t0_cols
        + media2_cols
    )

    X_train = (
        train[columnas]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .astype("float32")
    )

    y_train = (
        train["target"]
        .to_numpy(
            dtype="int8"
        )
    )

    registros = []

    for seed in SEEDS:

        print(
            f"seed={seed} ...",
            end=" ",
            flush=True,
        )

        inicio = time.time()

        model = nuevo_modelo(seed)

        model.fit(
            X_train,
            y_train,
        )

        gain = (
            model.booster_
            .feature_importance(
                importance_type="gain"
            )
        )

        total = gain.sum()

        if total > 0:
            share = gain / total
        else:
            share = np.zeros_like(
                gain,
                dtype=float,
            )

        tmp = pd.DataFrame(
            {
                "seed": seed,
                "feature": columnas,
                "gain": gain,
                "gain_share": share,
            }
        )

        tmp = tmp[
            tmp["feature"]
            .str.endswith("__media2")
        ].copy()

        tmp = tmp.sort_values(
            [
                "gain_share",
                "feature",
            ],
            ascending=[
                False,
                True,
            ],
        )

        tmp["rank_seed"] = (
            np.arange(
                1,
                len(tmp) + 1,
            )
        )

        registros.append(tmp)

        print(
            f"{time.time() - inicio:.2f}s"
        )

    ranking_seed = pd.concat(
        registros,
        ignore_index=True,
    )

    ranking_seed.to_csv(
        OUT_DIR
        / "ranking_media2_por_seed.csv",
        index=False,
    )

    resumen = (
        ranking_seed
        .groupby(
            "feature",
            as_index=False,
        )
        .agg(
            gain_mean=(
                "gain",
                "mean",
            ),
            gain_std=(
                "gain",
                "std",
            ),
            gain_share_mean=(
                "gain_share",
                "mean",
            ),
            gain_share_std=(
                "gain_share",
                "std",
            ),
            rank_mean=(
                "rank_seed",
                "mean",
            ),
            rank_min=(
                "rank_seed",
                "min",
            ),
            rank_max=(
                "rank_seed",
                "max",
            ),
        )
    )

    # Cuántas veces entra en cada Top-N.
    for n in TOP_N:

        ids = (
            ranking_seed[
                ranking_seed[
                    "rank_seed"
                ]
                <= n
            ]
            .groupby(
                "feature"
            )
            .size()
        )

        resumen[
            f"veces_top{n}"
        ] = (
            resumen["feature"]
            .map(ids)
            .fillna(0)
            .astype(int)
        )

    resumen = resumen.sort_values(
        [
            "gain_share_mean",
            "rank_mean",
            "feature",
        ],
        ascending=[
            False,
            True,
            True,
        ],
    ).reset_index(
        drop=True
    )

    resumen[
        "rank_consenso"
    ] = (
        np.arange(
            1,
            len(resumen) + 1,
        )
    )

    resumen.to_csv(
        OUT_DIR
        / "ranking_media2_train.csv",
        index=False,
    )

    print()
    print(
        "TOP 25 MEDIA2 — ranking TRAIN"
    )

    print(
        resumen[
            [
                "rank_consenso",
                "feature",
                "gain_share_mean",
                "rank_mean",
                "rank_min",
                "rank_max",
                "veces_top10",
                "veces_top20",
                "veces_top40",
            ]
        ]
        .head(25)
        .to_string(
            index=False
        )
    )

    return (
        resumen,
        ranking_seed,
    )


# ======================================================================
# ESTABILIDAD DEL RANKING
# ======================================================================

def estabilidad_topn(
    ranking_seed,
):

    titulo(
        "ESTABILIDAD DEL RANKING ENTRE SEMILLAS"
    )

    registros = []

    for n in TOP_N:

        conjuntos = {}

        for seed in SEEDS:

            x = ranking_seed[
                (
                    ranking_seed["seed"]
                    == seed
                )
                &
                (
                    ranking_seed[
                        "rank_seed"
                    ]
                    <= n
                )
            ]

            conjuntos[seed] = set(
                x["feature"]
            )

        for i in range(
            len(SEEDS)
        ):

            for j in range(
                i + 1,
                len(SEEDS),
            ):

                s1 = SEEDS[i]
                s2 = SEEDS[j]

                a = conjuntos[s1]
                b = conjuntos[s2]

                inter = len(
                    a & b
                )

                union = len(
                    a | b
                )

                jaccard = (
                    inter / union
                    if union
                    else np.nan
                )

                registros.append(
                    {
                        "top_n": n,
                        "seed_1": s1,
                        "seed_2": s2,
                        "interseccion":
                            inter,
                        "jaccard":
                            jaccard,
                    }
                )

    est = pd.DataFrame(
        registros
    )

    est.to_csv(
        OUT_DIR
        / "estabilidad_ranking.csv",
        index=False,
    )

    resumen = (
        est
        .groupby(
            "top_n",
            as_index=False,
        )
        .agg(
            interseccion_media=(
                "interseccion",
                "mean",
            ),
            interseccion_min=(
                "interseccion",
                "min",
            ),
            jaccard_mean=(
                "jaccard",
                "mean",
            ),
            jaccard_min=(
                "jaccard",
                "min",
            ),
        )
    )

    print(
        resumen.to_string(
            index=False
        )
    )

    return est, resumen


# ======================================================================
# DISEÑOS B0-B5
# ======================================================================

def construir_disenos(
    t0_cols,
    media2_cols,
    ranking,
):

    titulo(
        "CONSTRUCCIÓN DE EXPERIMENTOS"
    )

    orden_media = (
        ranking["feature"]
        .tolist()
    )

    # Seguridad:
    # sólo features media2 realmente válidas.
    permitidas = set(
        media2_cols
    )

    orden_media = [
        c
        for c in orden_media
        if c in permitidas
    ]

    diseños = {
        "B0_t0":
            list(t0_cols),
    }

    for n in TOP_N:

        diseños[
            f"B{TOP_N.index(n)+1}"
            f"_t0_media2_top{n}"
        ] = (
            list(t0_cols)
            + orden_media[:n]
        )

    diseños[
        "B5_t0_media2_all"
    ] = (
        list(t0_cols)
        + list(media2_cols)
    )

    for nombre, cols in diseños.items():

        n_t0 = sum(
            c.endswith("__t0")
            for c in cols
        )

        n_media = sum(
            c.endswith("__media2")
            for c in cols
        )

        print(
            f"{nombre:25s} "
            f"{len(cols):4d} features "
            f"(t0={n_t0}, "
            f"media2={n_media})"
        )

    return diseños


# ======================================================================
# ENTRENAMIENTO / EVALUACIÓN
# ======================================================================

def evaluar_disenos(
    train,
    test,
    diseños,
):

    titulo(
        "VALIDACIÓN PREDICTIVA OUT-OF-TIME"
    )

    resultados = []
    predicciones = []
    importancias = []

    y_train = (
        train["target"]
        .to_numpy(
            dtype="int8"
        )
    )

    y_test = (
        test["target"]
        .to_numpy(
            dtype="int8"
        )
    )

    for nombre, columnas in (
        diseños.items()
    ):

        subtitulo(
            f"EXPERIMENTO {nombre}"
        )

        (
            X_train,
            X_test,
            _,
            _,
        ) = preparar_xy(
            train,
            test,
            columnas,
        )

        print(
            "TRAIN:",
            X_train.shape,
        )

        print(
            "TEST :",
            X_test.shape,
        )

        for seed in SEEDS:

            print(
                f"  seed={seed} ...",
                end=" ",
                flush=True,
            )

            inicio = time.time()

            model = nuevo_modelo(seed)

            model.fit(
                X_train,
                y_train,
            )

            p_train = (
                model.predict_proba(
                    X_train
                )[:, 1]
            )

            p_test = (
                model.predict_proba(
                    X_test
                )[:, 1]
            )

            auc_train = (
                roc_auc_score(
                    y_train,
                    p_train,
                )
            )

            auc_test = (
                roc_auc_score(
                    y_test,
                    p_test,
                )
            )

            ap_test = (
                average_precision_score(
                    y_test,
                    p_test,
                )
            )

            ll_test = (
                log_loss(
                    y_test,
                    p_test,
                )
            )

            segundos = (
                time.time()
                - inicio
            )

            resultados.append(
                {
                    "experimento":
                        nombre,
                    "seed":
                        seed,
                    "n_features":
                        len(columnas),
                    "auc_train":
                        auc_train,
                    "auc_test":
                        auc_test,
                    "ap_test":
                        ap_test,
                    "logloss_test":
                        ll_test,
                    "fit_seconds":
                        segundos,
                }
            )

            print(
                f"AUC={auc_test:.6f}  "
                f"AP={ap_test:.6f}  "
                f"LL={ll_test:.6f}  "
                f"{segundos:.2f}s"
            )

            # ---------------------------------------------
            # Predicciones TEST
            # ---------------------------------------------

            pred = pd.DataFrame(
                {
                    "experimento":
                        nombre,
                    "seed":
                        seed,
                    "numero_de_cliente":
                        test[
                            "numero_de_cliente"
                        ].to_numpy(),
                    "foto_ancla":
                        TEST_FOTO,
                    "target":
                        y_test,
                    "prob":
                        p_test,
                }
            )

            predicciones.append(
                pred
            )

            # ---------------------------------------------
            # Importancia
            # ---------------------------------------------

            gain = (
                model.booster_
                .feature_importance(
                    importance_type="gain"
                )
            )

            total_gain = (
                gain.sum()
            )

            if total_gain > 0:
                share = (
                    gain
                    / total_gain
                )
            else:
                share = (
                    np.zeros_like(
                        gain,
                        dtype=float,
                    )
                )

            imp = pd.DataFrame(
                {
                    "experimento":
                        nombre,
                    "seed":
                        seed,
                    "feature":
                        columnas,
                    "gain":
                        gain,
                    "gain_share":
                        share,
                }
            )

            importancias.append(
                imp
            )

    resultados = pd.DataFrame(
        resultados
    )

    predicciones = pd.concat(
        predicciones,
        ignore_index=True,
    )

    importancias = pd.concat(
        importancias,
        ignore_index=True,
    )

    return (
        resultados,
        predicciones,
        importancias,
    )


# ======================================================================
# RESÚMENES
# ======================================================================

def resumir_modelos(
    resultados,
):

    titulo(
        "RESUMEN DE MODELOS"
    )

    resumen = (
        resultados
        .groupby(
            "experimento",
            as_index=False,
        )
        .agg(
            seeds=(
                "seed",
                "nunique",
            ),
            n_features=(
                "n_features",
                "first",
            ),
            auc_train_mean=(
                "auc_train",
                "mean",
            ),
            auc_test_mean=(
                "auc_test",
                "mean",
            ),
            auc_test_std=(
                "auc_test",
                "std",
            ),
            auc_test_min=(
                "auc_test",
                "min",
            ),
            auc_test_max=(
                "auc_test",
                "max",
            ),
            ap_test_mean=(
                "ap_test",
                "mean",
            ),
            ap_test_std=(
                "ap_test",
                "std",
            ),
            logloss_test_mean=(
                "logloss_test",
                "mean",
            ),
            logloss_test_std=(
                "logloss_test",
                "std",
            ),
            fit_seconds_mean=(
                "fit_seconds",
                "mean",
            ),
        )
    )

    orden = {
        "B0_t0": 0,
        "B1_t0_media2_top10": 1,
        "B2_t0_media2_top20": 2,
        "B3_t0_media2_top40": 3,
        "B4_t0_media2_top80": 4,
        "B5_t0_media2_all": 5,
    }

    resumen["_orden"] = (
        resumen["experimento"]
        .map(orden)
    )

    resumen = (
        resumen
        .sort_values("_orden")
        .drop(
            columns="_orden"
        )
        .reset_index(
            drop=True
        )
    )

    print(
        resumen.to_string(
            index=False
        )
    )

    return resumen


def comparar_baseline(
    resultados,
):

    titulo(
        "COMPARACIÓN CONTRA B0"
    )

    base = (
        resultados[
            resultados[
                "experimento"
            ]
            == "B0_t0"
        ][
            [
                "seed",
                "auc_test",
                "ap_test",
                "logloss_test",
            ]
        ]
        .rename(
            columns={
                "auc_test":
                    "auc_base",
                "ap_test":
                    "ap_base",
                "logloss_test":
                    "ll_base",
            }
        )
    )

    x = resultados.merge(
        base,
        on="seed",
        how="left",
    )

    x["delta_auc"] = (
        x["auc_test"]
        - x["auc_base"]
    )

    x["delta_ap"] = (
        x["ap_test"]
        - x["ap_base"]
    )

    # Positivo = mejora.
    x["mejora_logloss"] = (
        x["ll_base"]
        - x["logloss_test"]
    )

    comp = (
        x.groupby(
            "experimento",
            as_index=False,
        )
        .agg(
            delta_auc_mean=(
                "delta_auc",
                "mean",
            ),
            delta_auc_std=(
                "delta_auc",
                "std",
            ),
            delta_auc_min=(
                "delta_auc",
                "min",
            ),
            delta_auc_max=(
                "delta_auc",
                "max",
            ),
            delta_ap_mean=(
                "delta_ap",
                "mean",
            ),
            delta_ap_std=(
                "delta_ap",
                "std",
            ),
            mejora_logloss_mean=(
                "mejora_logloss",
                "mean",
            ),
            semillas_mejor_auc=(
                "delta_auc",
                lambda s:
                    int(
                        (s > 0)
                        .sum()
                    ),
            ),
            semillas_mejor_ap=(
                "delta_ap",
                lambda s:
                    int(
                        (s > 0)
                        .sum()
                    ),
            ),
        )
    )

    orden = {
        "B0_t0": 0,
        "B1_t0_media2_top10": 1,
        "B2_t0_media2_top20": 2,
        "B3_t0_media2_top40": 3,
        "B4_t0_media2_top80": 4,
        "B5_t0_media2_all": 5,
    }

    comp["_orden"] = (
        comp["experimento"]
        .map(orden)
    )

    comp = (
        comp
        .sort_values("_orden")
        .drop(
            columns="_orden"
        )
        .reset_index(
            drop=True
        )
    )

    print(
        comp.to_string(
            index=False
        )
    )

    return comp


# ======================================================================
# ENSEMBLE
# ======================================================================

def evaluar_ensemble(
    predicciones,
):

    titulo(
        "ENSEMBLE DE SEMILLAS"
    )

    ens = (
        predicciones
        .groupby(
            [
                "experimento",
                "numero_de_cliente",
                "foto_ancla",
                "target",
            ],
            as_index=False,
        )
        .agg(
            prob_mean=(
                "prob",
                "mean",
            ),
            prob_std=(
                "prob",
                "std",
            ),
        )
    )

    registros = []

    for nombre, g in (
        ens.groupby(
            "experimento",
            sort=False,
        )
    ):

        y = (
            g["target"]
            .to_numpy()
        )

        p = (
            g["prob_mean"]
            .to_numpy()
        )

        registros.append(
            {
                "experimento":
                    nombre,
                "auc_ensemble":
                    roc_auc_score(
                        y,
                        p,
                    ),
                "ap_ensemble":
                    average_precision_score(
                        y,
                        p,
                    ),
                "logloss_ensemble":
                    log_loss(
                        y,
                        p,
                    ),
                "prob_std_mean":
                    g["prob_std"]
                    .mean(),
            }
        )

    resumen = pd.DataFrame(
        registros
    )

    print(
        resumen.to_string(
            index=False
        )
    )

    return ens, resumen


# ======================================================================
# IMPORTANCIA FINAL
# ======================================================================

def resumir_importancias(
    importancias,
):

    titulo(
        "IMPORTANCIA DE FEATURES"
    )

    resumen = (
        importancias
        .groupby(
            [
                "experimento",
                "feature",
            ],
            as_index=False,
        )
        .agg(
            gain_mean=(
                "gain",
                "mean",
            ),
            gain_std=(
                "gain",
                "std",
            ),
            gain_share_mean=(
                "gain_share",
                "mean",
            ),
            gain_share_std=(
                "gain_share",
                "std",
            ),
        )
    )

    resumen[
        "transformacion"
    ] = np.where(
        resumen["feature"]
        .str.endswith("__media2"),
        "media2",
        "t0",
    )

    resumen[
        "variable"
    ] = (
        resumen["feature"]
        .str.replace(
            "__media2$",
            "",
            regex=True,
        )
        .str.replace(
            "__t0$",
            "",
            regex=True,
        )
    )

    resumen[
        "rank_gain"
    ] = (
        resumen
        .groupby(
            "experimento"
        )[
            "gain_share_mean"
        ]
        .rank(
            method="first",
            ascending=False,
        )
    )

    for nombre in (
        resumen[
            "experimento"
        ].unique()
    ):

        print()
        print(
            f"TOP 15 — {nombre}"
        )

        cols = [
            "rank_gain",
            "feature",
            "gain_share_mean",
        ]

        print(
            resumen[
                resumen[
                    "experimento"
                ]
                == nombre
            ]
            .sort_values(
                "rank_gain"
            )[
                cols
            ]
            .head(15)
            .to_string(
                index=False
            )
        )

    return resumen


# ======================================================================
# SELECCIÓN DE DISEÑO
# ======================================================================

def diagnostico_saturacion(
    resumen,
    comparacion,
):

    titulo(
        "DIAGNÓSTICO DE SATURACIÓN"
    )

    x = resumen.merge(
        comparacion[
            [
                "experimento",
                "delta_auc_mean",
                "delta_ap_mean",
                "semillas_mejor_auc",
            ]
        ],
        on="experimento",
        how="left",
    )

    x = x[
        [
            "experimento",
            "n_features",
            "auc_test_mean",
            "auc_test_std",
            "ap_test_mean",
            "logloss_test_mean",
            "delta_auc_mean",
            "delta_ap_mean",
            "semillas_mejor_auc",
        ]
    ]

    print(
        x.to_string(
            index=False
        )
    )

    # Mejor AUC observado.
    mejor_idx = (
        x["auc_test_mean"]
        .idxmax()
    )

    mejor = x.loc[
        mejor_idx
    ]

    print()
    print(
        "Mayor AUC medio:"
    )

    print(
        f"  {mejor['experimento']}"
    )

    print(
        f"  AUC = "
        f"{mejor['auc_test_mean']:.6f}"
    )

    print(
        f"  features = "
        f"{int(mejor['n_features'])}"
    )

    return x


# ======================================================================
# EXPORTACIÓN
# ======================================================================

def exportar(
    resultados,
    resumen,
    comparacion,
    predicciones,
    ens,
    resumen_ens,
    importancias,
    importancia_resumen,
    ranking,
    estabilidad_resumen,
    saturacion,
):

    titulo(
        "EXPORTACIÓN"
    )

    archivos = {
        "metricas_corridas.csv":
            resultados,

        "resumen_modelos.csv":
            resumen,

        "comparacion_baseline.csv":
            comparacion,

        "predicciones_test.csv":
            predicciones,

        "predicciones_ensemble.csv":
            ens,

        "resumen_ensemble.csv":
            resumen_ens,

        "importancia_gain.csv":
            importancias,

        "importancia_resumen.csv":
            importancia_resumen,

        "diagnostico_saturacion.csv":
            saturacion,
    }

    for nombre, df in (
        archivos.items()
    ):

        destino = (
            OUT_DIR
            / nombre
        )

        df.to_csv(
            destino,
            index=False,
        )

        print(
            f"OK  "
            f"{nombre:35s} "
            f"{destino.stat().st_size:>12,} "
            "bytes"
        )

    # -------------------------------------------------------------
    # Resumen TXT
    # -------------------------------------------------------------

    txt = []

    txt.append(
        "Z509 — SELECCIÓN INCREMENTAL MEDIA2"
    )

    txt.append(
        "=" * 70
    )

    txt.append("")

    txt.append(
        f"TRAIN = {TRAIN_FOTO}"
    )

    txt.append(
        f"TEST  = {TEST_FOTO}"
    )

    txt.append("")

    txt.append(
        "Parámetros LightGBM:"
    )

    txt.append(
        json.dumps(
            PARAMS,
            indent=2,
        )
    )

    txt.append("")

    txt.append(
        "RESUMEN MODELOS"
    )

    txt.append(
        resumen.to_string(
            index=False
        )
    )

    txt.append("")

    txt.append(
        "COMPARACIÓN CONTRA B0"
    )

    txt.append(
        comparacion.to_string(
            index=False
        )
    )

    txt.append("")

    txt.append(
        "ESTABILIDAD TOP-N"
    )

    txt.append(
        estabilidad_resumen
        .to_string(
            index=False
        )
    )

    txt.append("")

    txt.append(
        "TOP 30 MEDIA2 — TRAIN"
    )

    txt.append(
        ranking[
            [
                "rank_consenso",
                "feature",
                "gain_share_mean",
                "rank_mean",
                "rank_min",
                "rank_max",
                "veces_top10",
                "veces_top20",
                "veces_top40",
                "veces_top80",
            ]
        ]
        .head(30)
        .to_string(
            index=False
        )
    )

    destino = (
        OUT_DIR
        / "resumen_z509.txt"
    )

    destino.write_text(
        "\n".join(txt),
        encoding="utf-8",
    )

    print(
        f"OK  "
        f"{'resumen_z509.txt':35s} "
        f"{destino.stat().st_size:>12,} "
        "bytes"
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    inicio_total = time.time()

    titulo(
        "Z509 — SELECCIÓN INCREMENTAL DE MEDIA2"
    )

    print(
        """
Pregunta experimental:

¿La mejora observada al incorporar MEDIA2 requiere
las 152 variables históricas o está concentrada en
un subconjunto pequeño?

Regla fundamental:

El ranking de MEDIA2 se construye EXCLUSIVAMENTE
con TRAIN 202104.

TEST 202105 permanece aislado hasta la evaluación
out-of-time.

El modelo y los hiperparámetros permanecen fijos.
Sólo cambia el número de features MEDIA2.
"""
    )

    # --------------------------------------------------------------
    # Datos
    # --------------------------------------------------------------

    df = cargar_datos()

    train, test = (
        particion_temporal(
            df
        )
    )

    # --------------------------------------------------------------
    # Familias
    # --------------------------------------------------------------

    (
        t0_cols,
        media2_cols,
    ) = detectar_features(
        train
    )

    # --------------------------------------------------------------
    # Ranking TRAIN
    # --------------------------------------------------------------

    (
        ranking,
        ranking_seed,
    ) = ranking_media2_train(
        train,
        t0_cols,
        media2_cols,
    )

    # --------------------------------------------------------------
    # Estabilidad ranking
    # --------------------------------------------------------------

    (
        estabilidad,
        estabilidad_resumen,
    ) = estabilidad_topn(
        ranking_seed
    )

    # --------------------------------------------------------------
    # Diseños
    # --------------------------------------------------------------

    diseños = construir_disenos(
        t0_cols,
        media2_cols,
        ranking,
    )

    # Guardamos composición exacta.
    filas = []

    for nombre, cols in (
        diseños.items()
    ):

        for col in cols:

            filas.append(
                {
                    "experimento":
                        nombre,
                    "feature":
                        col,
                    "transformacion":
                        (
                            "media2"
                            if col.endswith(
                                "__media2"
                            )
                            else "t0"
                        ),
                }
            )

    pd.DataFrame(
        filas
    ).to_csv(
        OUT_DIR
        / "columnas_modelos.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Evaluación
    # --------------------------------------------------------------

    (
        resultados,
        predicciones,
        importancias,
    ) = evaluar_disenos(
        train,
        test,
        diseños,
    )

    # --------------------------------------------------------------
    # Resúmenes
    # --------------------------------------------------------------

    resumen = resumir_modelos(
        resultados
    )

    comparacion = (
        comparar_baseline(
            resultados
        )
    )

    (
        ens,
        resumen_ens,
    ) = evaluar_ensemble(
        predicciones
    )

    importancia_resumen = (
        resumir_importancias(
            importancias
        )
    )

    saturacion = (
        diagnostico_saturacion(
            resumen,
            comparacion,
        )
    )

    # --------------------------------------------------------------
    # Exportación
    # --------------------------------------------------------------

    exportar(
        resultados,
        resumen,
        comparacion,
        predicciones,
        ens,
        resumen_ens,
        importancias,
        importancia_resumen,
        ranking,
        estabilidad_resumen,
        saturacion,
    )

    # --------------------------------------------------------------
    # Final
    # --------------------------------------------------------------

    titulo(
        "RESULTADO FINAL Z509"
    )

    print(
        saturacion.to_string(
            index=False
        )
    )

    print()

    print(
        "Tiempo total:",
        f"{time.time() - inicio_total:.2f} s"
    )

    print()

    print(
        "Z509 FINALIZADO CORRECTAMENTE."
    )

    print()

    print(
        "Resultados:"
    )

    print(
        OUT_DIR
    )


if __name__ == "__main__":

    try:
        main()

    except Exception as e:

        titulo(
            "ERROR EN Z509"
        )

        print(
            f"{type(e).__name__}: "
            f"{e}"
        )

        raise