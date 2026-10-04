#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
z515_competencia_01.py

Primera Competencia - validación temporal LightGBM.

Permite dos modalidades de entrenamiento:

1) Un solo mes:
   python monday/z515_competencia_01.py \
       --train 202105 \
       --test 202106

2) Varios meses acumulados:
   python monday/z515_competencia_01.py \
       --train-desde 202103 \
       --train-hasta 202105 \
       --test 202106

TARGET:
    BAJA+2 = 1
    BAJA+1 / CONTINUA = 0

Se utilizan exclusivamente las 152 variables originales del dataset.
numero_de_cliente, foto_mes y clase_ternaria NO entran al modelo.

La evaluación principal es la ganancia económica sobre TEST para
distintos cortes del ranking de probabilidades.

Esta versión NO genera submits para 202108.
"""

import argparse
from pathlib import Path
import time

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    roc_auc_score,
)


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

DATASET = Path("/data/dmeyf/datasets/competencia_01.csv")

OUTPUT_DIR = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/competencia_01"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_MES_DEFAULT = 202105
TEST_MES_DEFAULT = 202106

SEMILLA = 290497

GANANCIA_ACIERTO = 1_072_500
COSTO_ESTIMULO = -27_500

CORTES = list(range(4000, 19001, 500))


# Mismos hiperparámetros usados en los experimentos anteriores.
# NO se modifican para poder comparar limpiamente los experimentos.
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
    "random_state": SEMILLA,
    "n_jobs": -1,
    "verbosity": -1,
}


# =============================================================================
# ARGUMENTOS
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Validación temporal para Primera Competencia."
    )

    parser.add_argument(
        "--train",
        type=int,
        default=None,
        help=(
            "Un único foto_mes de entrenamiento. "
            f"Si no se especifica ninguna modalidad se usa "
            f"{TRAIN_MES_DEFAULT}."
        ),
    )

    parser.add_argument(
        "--train-desde",
        type=int,
        default=None,
        help="Primer foto_mes del entrenamiento acumulado.",
    )

    parser.add_argument(
        "--train-hasta",
        type=int,
        default=None,
        help="Último foto_mes del entrenamiento acumulado.",
    )

    parser.add_argument(
        "--test",
        type=int,
        default=TEST_MES_DEFAULT,
        help=(
            f"foto_mes de validación "
            f"(default: {TEST_MES_DEFAULT})."
        ),
    )

    return parser.parse_args()


def resolver_periodos(args):
    """
    Determina qué meses se usarán como TRAIN y cuál como TEST.

    No permite mezclar --train con --train-desde/--train-hasta.
    """

    usa_train_simple = args.train is not None

    usa_rango = (
        args.train_desde is not None
        or args.train_hasta is not None
    )

    if usa_train_simple and usa_rango:
        raise ValueError(
            "No se puede combinar --train con "
            "--train-desde/--train-hasta."
        )

    if usa_rango:

        if args.train_desde is None or args.train_hasta is None:
            raise ValueError(
                "Para entrenamiento acumulado deben especificarse "
                "tanto --train-desde como --train-hasta."
            )

        if args.train_desde > args.train_hasta:
            raise ValueError(
                "--train-desde no puede ser posterior a "
                "--train-hasta."
            )

        train_desde = args.train_desde
        train_hasta = args.train_hasta

    else:

        train_mes = (
            args.train
            if args.train is not None
            else TRAIN_MES_DEFAULT
        )

        train_desde = train_mes
        train_hasta = train_mes

    test_mes = args.test

    if train_hasta >= test_mes:
        raise ValueError(
            f"El último mes de TRAIN ({train_hasta}) debe ser "
            f"anterior a TEST ({test_mes})."
        )

    return train_desde, train_hasta, test_mes


# =============================================================================
# UTILIDADES
# =============================================================================

def titulo(texto):
    print()
    print("=" * 78)
    print(texto)
    print("=" * 78)


def cargar_datos(train_desde, train_hasta, test_mes):
    """
    Carga mediante DuckDB los meses necesarios para TRAIN y TEST.
    """

    titulo("CARGA DE DATOS")

    con = duckdb.connect()

    query = f"""
        SELECT *
        FROM read_csv_auto('{DATASET}')
        WHERE
            (
                foto_mes BETWEEN {train_desde} AND {train_hasta}
            )
            OR foto_mes = {test_mes}
        ORDER BY foto_mes, numero_de_cliente
    """

    df = con.execute(query).df()

    con.close()

    print(f"Dataset: {DATASET}")
    print(f"Filas cargadas: {len(df):,}")
    print()
    print(df.groupby("foto_mes").size())

    return df


def preparar_datos(
    df,
    train_desde,
    train_hasta,
    test_mes,
):
    """
    Construye target binario y matrices TRAIN / TEST.
    """

    titulo("PREPARACIÓN DEL EXPERIMENTO")

    df = df.copy()

    df["target"] = (
        df["clase_ternaria"].eq("BAJA+2")
    ).astype(np.int8)

    excluir = {
        "numero_de_cliente",
        "foto_mes",
        "clase_ternaria",
        "target",
    }

    features = [
        c
        for c in df.columns
        if c not in excluir
    ]

    train = df[
        (df["foto_mes"] >= train_desde)
        & (df["foto_mes"] <= train_hasta)
    ].copy()

    test = df[
        df["foto_mes"] == test_mes
    ].copy()

    if train.empty:
        raise ValueError(
            "No se encontraron observaciones para TRAIN."
        )

    if test.empty:
        raise ValueError(
            f"No se encontraron datos para TEST={test_mes}."
        )

    if train["clase_ternaria"].isna().any():
        meses_problematicos = sorted(
            train.loc[
                train["clase_ternaria"].isna(),
                "foto_mes",
            ].unique()
        )

        raise ValueError(
            "TRAIN contiene targets desconocidos en: "
            f"{meses_problematicos}"
        )

    if test["clase_ternaria"].isna().any():
        raise ValueError(
            f"TEST={test_mes} contiene targets desconocidos."
        )

    if len(features) != 152:
        raise ValueError(
            f"Se esperaban 152 variables predictoras y se "
            f"encontraron {len(features)}."
        )

    print(
        f"TRAIN: {train_desde}"
        + (
            f" - {train_hasta}"
            if train_desde != train_hasta
            else ""
        )
    )

    print(f"TEST : {test_mes}")

    print(
        f"Observaciones TRAIN: {len(train):,}"
    )

    print(
        f"Observaciones TEST : {len(test):,}"
    )

    print(
        f"Variables predictoras: {len(features)}"
    )

    print("\nDistribución TRAIN por foto_mes y clase:")

    print(
        train.groupby(
            ["foto_mes", "clase_ternaria"]
        )
        .size()
        .unstack(fill_value=0)
    )

    print("\nDistribución TEST:")

    print(
        test["clase_ternaria"]
        .value_counts(dropna=False)
    )

    print("\nTarget binario TRAIN:")

    print(
        train["target"]
        .value_counts()
        .sort_index()
    )

    print("\nTarget binario TEST:")

    print(
        test["target"]
        .value_counts()
        .sort_index()
    )

    X_train = train[features]
    y_train = train["target"]

    X_test = test[features]
    y_test = test["target"]

    return (
        train,
        test,
        features,
        X_train,
        y_train,
        X_test,
        y_test,
    )


def entrenar_modelo(X_train, y_train):
    """
    Entrena LightGBM con los mismos parámetros de los
    experimentos anteriores.
    """

    titulo("ENTRENAMIENTO LIGHTGBM")

    print("Parámetros:")

    for k, v in PARAMS.items():
        print(f"  {k}: {v}")

    modelo = lgb.LGBMClassifier(
        **PARAMS
    )

    inicio = time.time()

    modelo.fit(
        X_train,
        y_train,
        categorical_feature="auto",
    )

    segundos = time.time() - inicio

    print(
        f"\nEntrenamiento finalizado en "
        f"{segundos:.2f} s"
    )

    return modelo


def evaluar_metricas(
    modelo,
    X_test,
    y_test,
):
    """
    Calcula métricas probabilísticas sobre TEST.
    """

    titulo("MÉTRICAS PREDICTIVAS")

    prob = modelo.predict_proba(
        X_test
    )[:, 1]

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

    print(f"AUC     : {auc:.6f}")
    print(f"AP      : {ap:.6f}")
    print(f"LogLoss : {ll:.6f}")

    return prob, auc, ap, ll


def evaluar_ganancia(
    test,
    prob,
):
    """
    Ordena TEST por probabilidad descendente y calcula
    ganancia acumulada para cada corte.
    """

    titulo("GANANCIA POR CORTE")

    ranking = pd.DataFrame({
        "numero_de_cliente":
            test["numero_de_cliente"].to_numpy(),

        "clase_ternaria":
            test["clase_ternaria"].to_numpy(),

        "target":
            test["target"].to_numpy(),

        "prob":
            prob,
    })

    ranking = (
        ranking
        .sort_values(
            "prob",
            ascending=False,
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    ranking["ganancia_individual"] = np.where(
        ranking["target"].eq(1),
        GANANCIA_ACIERTO,
        COSTO_ESTIMULO,
    )

    ranking["ganancia_acumulada"] = (
        ranking["ganancia_individual"]
        .cumsum()
    )

    total_positivos = int(
        ranking["target"].sum()
    )

    resultados = []

    for corte in CORTES:

        if corte > len(ranking):
            continue

        seleccion = ranking.iloc[:corte]

        positivos = int(
            seleccion["target"].sum()
        )

        precision = (
            positivos / corte
        )

        recall = (
            positivos / total_positivos
        )

        ganancia = int(
            seleccion[
                "ganancia_individual"
            ].sum()
        )

        resultados.append({
            "corte": corte,
            "positivos": positivos,
            "precision": precision,
            "recall": recall,
            "ganancia": ganancia,
            "ganancia_millones":
                ganancia / 1_000_000,
        })

    tabla = pd.DataFrame(
        resultados
    )

    mejor_idx = (
        tabla["ganancia"]
        .idxmax()
    )

    mejor = tabla.loc[
        mejor_idx
    ]

    print(
        tabla.to_string(
            index=False,
            formatters={
                "precision":
                    lambda x: f"{x:.4f}",

                "recall":
                    lambda x: f"{x:.4f}",

                "ganancia_millones":
                    lambda x: f"{x:.3f}",
            },
        )
    )

    titulo("MEJOR CORTE")

    print(
        f"Corte       : "
        f"{int(mejor['corte']):,}"
    )

    print(
        f"BAJA+2      : "
        f"{int(mejor['positivos']):,}"
    )

    print(
        f"Precision   : "
        f"{mejor['precision']:.4f}"
    )

    print(
        f"Recall      : "
        f"{mejor['recall']:.4f}"
    )

    print(
        f"Ganancia    : "
        f"${int(mejor['ganancia']):,} "
        f"({mejor['ganancia_millones']:.3f} millones)"
    )

    return ranking, tabla, mejor


def guardar_resultados(
    ranking,
    tabla,
    modelo,
    features,
    auc,
    ap,
    ll,
    mejor,
    train_desde,
    train_hasta,
    test_mes,
):
    """
    Guarda resultados de validación.

    Cada experimento queda identificado por sus meses.

    NO genera archivos para Zulip.
    """

    titulo("GUARDADO DE RESULTADOS")

    if train_desde == train_hasta:

        prefijo = (
            f"z515_train{train_desde}"
            f"_test{test_mes}"
        )

    else:

        prefijo = (
            f"z515_train"
            f"{train_desde}-{train_hasta}"
            f"_test{test_mes}"
        )

    ranking_path = (
        OUTPUT_DIR
        / f"{prefijo}_ranking.csv"
    )

    cortes_path = (
        OUTPUT_DIR
        / f"{prefijo}_ganancia_cortes.csv"
    )

    importancia_path = (
        OUTPUT_DIR
        / f"{prefijo}_importancia.csv"
    )

    resumen_path = (
        OUTPUT_DIR
        / f"{prefijo}_resumen.txt"
    )

    ranking.to_csv(
        ranking_path,
        index=False,
    )

    tabla.to_csv(
        cortes_path,
        index=False,
    )

    importancia = pd.DataFrame({
        "feature":
            features,

        "importance_gain":
            modelo.booster_.feature_importance(
                importance_type="gain"
            ),

        "importance_split":
            modelo.booster_.feature_importance(
                importance_type="split"
            ),
    })

    importancia = (
        importancia
        .sort_values(
            "importance_gain",
            ascending=False,
        )
    )

    importancia.to_csv(
        importancia_path,
        index=False,
    )

    with open(
        resumen_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "z515 - Competencia 01 "
            "- validación temporal\n"
        )

        f.write(
            "=" * 60 + "\n"
        )

        f.write(
            f"TRAIN_DESDE: "
            f"{train_desde}\n"
        )

        f.write(
            f"TRAIN_HASTA: "
            f"{train_hasta}\n"
        )

        f.write(
            f"TEST: "
            f"{test_mes}\n"
        )

        f.write(
            "TARGET: BAJA+2 vs resto\n"
        )

        f.write(
            f"Features: "
            f"{len(features)}\n"
        )

        f.write(
            f"Semilla: "
            f"{SEMILLA}\n"
        )

        f.write(
            f"AUC: "
            f"{auc:.6f}\n"
        )

        f.write(
            f"AP: "
            f"{ap:.6f}\n"
        )

        f.write(
            f"LogLoss: "
            f"{ll:.6f}\n"
        )

        f.write(
            f"Mejor corte: "
            f"{int(mejor['corte'])}\n"
        )

        f.write(
            f"BAJA+2 capturados: "
            f"{int(mejor['positivos'])}\n"
        )

        f.write(
            f"Precision: "
            f"{mejor['precision']:.6f}\n"
        )

        f.write(
            f"Recall: "
            f"{mejor['recall']:.6f}\n"
        )

        f.write(
            f"Ganancia: "
            f"{int(mejor['ganancia'])}\n"
        )

    print(ranking_path)
    print(cortes_path)
    print(importancia_path)
    print(resumen_path)


# =============================================================================
# MAIN
# =============================================================================

def main():

    args = parse_args()

    (
        train_desde,
        train_hasta,
        test_mes,
    ) = resolver_periodos(args)

    titulo(
        "Z515 - PRIMERA COMPETENCIA"
    )

    print(
        "Validación temporal LightGBM"
    )

    if train_desde == train_hasta:

        print(
            f"TRAIN: {train_desde}"
        )

    else:

        print(
            f"TRAIN: "
            f"{train_desde} - {train_hasta}"
        )

    print(
        f"TEST : {test_mes}"
    )

    print(
        "Esta ejecución NO genera submits."
    )

    df = cargar_datos(
        train_desde,
        train_hasta,
        test_mes,
    )

    (
        train,
        test,
        features,
        X_train,
        y_train,
        X_test,
        y_test,
    ) = preparar_datos(
        df,
        train_desde,
        train_hasta,
        test_mes,
    )

    modelo = entrenar_modelo(
        X_train,
        y_train,
    )

    (
        prob,
        auc,
        ap,
        ll,
    ) = evaluar_metricas(
        modelo,
        X_test,
        y_test,
    )

    (
        ranking,
        tabla,
        mejor,
    ) = evaluar_ganancia(
        test,
        prob,
    )

    guardar_resultados(
        ranking,
        tabla,
        modelo,
        features,
        auc,
        ap,
        ll,
        mejor,
        train_desde,
        train_hasta,
        test_mes,
    )

    titulo("FIN")

    print(
        "Validación completada."
    )

    print(
        "Todavía NO se generó ningún "
        "archivo para Zulip."
    )


if __name__ == "__main__":
    main()