#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
z516_competencia_01_final.py

Primera Competencia - generación de candidatos finales.

TRAIN supervisado:
    202103, 202104, 202105, 202106

SCORE:
    202108

202107 se excluye porque su target BAJA+2 no está completo.

TARGET:
    BAJA+2 = 1
    BAJA+1 / CONTINUA = 0

MODELOS:
    LightGBM con los mismos hiperparámetros utilizados en z515.
    Entre modelos cambia SOLAMENTE random_state.

SALIDA:
    CSV sin header, una única columna:
        numero_de_cliente

El script genera varios candidatos para distintos cortes y semillas,
pero NO realiza ningún envío a Zulip.
"""

from pathlib import Path
import time

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

DATASET = Path(
    "/data/dmeyf/datasets/competencia_01.csv"
)

OUTPUT_DIR = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/"
    "competencia_01/submits_z516"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TRAIN_DESDE = 202103
TRAIN_HASTA = 202106

SCORE_MES = 202108


# -------------------------------------------------------------------------
# Semillas
# -------------------------------------------------------------------------
#
# Todos los hiperparámetros permanecen idénticos.
# Únicamente cambia random_state.
#
# Cinco semillas permiten estudiar estabilidad sin generar todavía
# una cantidad excesiva de modelos/archivos.
#

SEMILLAS = [
    290497,
    100003,
    200003,
    300007,
    400009,
]


# -------------------------------------------------------------------------
# Cortes
# -------------------------------------------------------------------------
#
# Elegidos a partir de las validaciones temporales realizadas con z515.
#
# 8.000  : extremo bajo razonable
# 9.500  : óptimo observado en una ventana
# 12.000 : óptimo acumulado 202103-202105 -> 202106
# 12.500 : óptimo acumulado 202104-202105 -> 202106
# 14.500 : óptimo 202105 -> 202106
# 15.000 : extremo superior recomendado
#

CORTES = [
    8000,
    9500,
    12000,
    12500,
    14500,
    15000,
]


# -------------------------------------------------------------------------
# LightGBM
# -------------------------------------------------------------------------
#
# Mismos hiperparámetros utilizados en z515.
#

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
}


# =============================================================================
# UTILIDADES
# =============================================================================

def titulo(texto):
    print()
    print("=" * 78)
    print(texto)
    print("=" * 78)


def cargar_datos():
    """
    Carga únicamente:
        TRAIN 202103-202106
        SCORE 202108

    202107 queda deliberadamente afuera.
    """

    titulo("CARGA DE DATOS")

    con = duckdb.connect()

    query = f"""
        SELECT *
        FROM read_csv_auto('{DATASET}')
        WHERE
            foto_mes BETWEEN {TRAIN_DESDE} AND {TRAIN_HASTA}
            OR foto_mes = {SCORE_MES}
        ORDER BY foto_mes, numero_de_cliente
    """

    df = con.execute(query).df()

    con.close()

    print(f"Dataset: {DATASET}")
    print(f"Filas cargadas: {len(df):,}")

    print()
    print(
        df.groupby("foto_mes")
        .size()
    )

    return df


def preparar_datos(df):
    """
    Construye TRAIN y SCORE.

    Valida que:
        - TRAIN tenga target completo.
        - SCORE sea 202108.
        - SCORE tenga IDs únicos.
        - existan exactamente 152 predictores.
        - 202107 no haya ingresado accidentalmente.
    """

    titulo("PREPARACIÓN")

    meses_presentes = set(
        df["foto_mes"].unique()
    )

    if 202107 in meses_presentes:
        raise ValueError(
            "ERROR: 202107 ingresó al dataset de modelado."
        )

    train = df[
        (df["foto_mes"] >= TRAIN_DESDE)
        & (df["foto_mes"] <= TRAIN_HASTA)
    ].copy()

    score = df[
        df["foto_mes"] == SCORE_MES
    ].copy()

    if train.empty:
        raise ValueError(
            "TRAIN está vacío."
        )

    if score.empty:
        raise ValueError(
            "SCORE está vacío."
        )

    if train["clase_ternaria"].isna().any():

        meses_problematicos = sorted(
            train.loc[
                train["clase_ternaria"].isna(),
                "foto_mes",
            ].unique()
        )

        raise ValueError(
            "TRAIN contiene targets desconocidos en "
            f"{meses_problematicos}"
        )

    duplicados_score = int(
        score["numero_de_cliente"]
        .duplicated()
        .sum()
    )

    if duplicados_score != 0:
        raise ValueError(
            f"SCORE contiene {duplicados_score} "
            "numero_de_cliente duplicados."
        )

    excluir = {
        "numero_de_cliente",
        "foto_mes",
        "clase_ternaria",
    }

    features = [
        c
        for c in df.columns
        if c not in excluir
    ]

    if len(features) != 152:
        raise ValueError(
            f"Se esperaban 152 predictores y se "
            f"encontraron {len(features)}."
        )

    train["target"] = (
        train["clase_ternaria"]
        .eq("BAJA+2")
        .astype(np.int8)
    )

    print(
        f"TRAIN: {TRAIN_DESDE}-{TRAIN_HASTA}"
    )

    print(
        f"SCORE: {SCORE_MES}"
    )

    print(
        f"Observaciones TRAIN: {len(train):,}"
    )

    print(
        f"Clientes SCORE: {len(score):,}"
    )

    print(
        f"IDs duplicados SCORE: {duplicados_score}"
    )

    print(
        f"Variables predictoras: {len(features)}"
    )

    print("\nDistribución TRAIN:")

    print(
        train.groupby(
            ["foto_mes", "clase_ternaria"]
        )
        .size()
        .unstack(fill_value=0)
    )

    print("\nTarget binario TRAIN:")

    print(
        train["target"]
        .value_counts()
        .sort_index()
    )

    X_train = train[features]
    y_train = train["target"]

    X_score = score[features]

    return (
        train,
        score,
        features,
        X_train,
        y_train,
        X_score,
    )


def entrenar_y_predecir(
    X_train,
    y_train,
    X_score,
    semilla,
):
    """
    Entrena un LightGBM y devuelve probabilidades para 202108.
    """

    titulo(
        f"MODELO - SEMILLA {semilla}"
    )

    params = PARAMS_BASE.copy()

    params["random_state"] = semilla

    print("Parámetros:")

    for k, v in params.items():
        print(f"  {k}: {v}")

    inicio = time.time()

    modelo = lgb.LGBMClassifier(
        **params
    )

    modelo.fit(
        X_train,
        y_train,
        categorical_feature="auto",
    )

    prob = modelo.predict_proba(
        X_score
    )[:, 1]

    segundos = time.time() - inicio

    print(
        f"\nEntrenamiento + scoring: "
        f"{segundos:.2f} s"
    )

    print(
        f"Prob min : {prob.min():.8f}"
    )

    print(
        f"Prob mean: {prob.mean():.8f}"
    )

    print(
        f"Prob max : {prob.max():.8f}"
    )

    return modelo, prob


def construir_ranking(
    score,
    prob,
    semilla,
):
    """
    Construye ranking descendente para una semilla.
    """

    ranking = pd.DataFrame({
        "numero_de_cliente":
            score["numero_de_cliente"]
            .to_numpy(),

        "prob":
            prob,
    })

    ranking = (
        ranking
        .sort_values(
            ["prob", "numero_de_cliente"],
            ascending=[False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    ranking["rank"] = (
        np.arange(
            1,
            len(ranking) + 1,
        )
    )

    ranking["semilla"] = semilla

    return ranking


def validar_submit(
    path,
    corte,
):
    """
    Valida el archivo exactamente como será enviado.

    Requisitos:
        - existe;
        - exactamente N filas;
        - una sola columna;
        - sin duplicados;
        - todos los IDs son enteros;
        - ningún valor está vacío;
        - no hay header.
    """

    if not path.exists():
        raise ValueError(
            f"No existe {path}"
        )

    submit = pd.read_csv(
        path,
        header=None,
        dtype=str,
    )

    if submit.shape != (corte, 1):
        raise ValueError(
            f"{path.name}: shape incorrecto "
            f"{submit.shape}; esperado ({corte}, 1)."
        )

    serie = submit.iloc[:, 0]

    if serie.isna().any():
        raise ValueError(
            f"{path.name}: contiene valores vacíos."
        )

    if serie.duplicated().any():
        raise ValueError(
            f"{path.name}: contiene IDs duplicados."
        )

    if not serie.str.fullmatch(
        r"[0-9]+"
    ).all():

        raise ValueError(
            f"{path.name}: hay valores que no tienen "
            "notación entera."
        )

    if serie.iloc[0] == "numero_de_cliente":
        raise ValueError(
            f"{path.name}: contiene header."
        )

    return True


def guardar_candidatos(
    ranking,
    semilla,
):
    """
    Genera un CSV por corte para una semilla.

    Cada archivo contiene únicamente numero_de_cliente,
    sin header y sin índice.
    """

    titulo(
        f"CANDIDATOS - SEMILLA {semilla}"
    )

    registros = []

    for corte in CORTES:

        seleccion = (
            ranking
            .head(corte)
            [["numero_de_cliente"]]
            .copy()
        )

        seleccion[
            "numero_de_cliente"
        ] = (
            seleccion[
                "numero_de_cliente"
            ]
            .astype(np.int64)
        )

        nombre = (
            f"z516_s{semilla}"
            f"_n{corte}"
            f"_m{SCORE_MES}.csv"
        )

        path = (
            OUTPUT_DIR
            / nombre
        )

        seleccion.to_csv(
            path,
            index=False,
            header=False,
        )

        validar_submit(
            path,
            corte,
        )

        print(
            f"OK  {nombre:<40} "
            f"{corte:>6,} clientes"
        )

        registros.append({
            "semilla": semilla,
            "corte": corte,
            "archivo": nombre,
            "path": str(path),
            "cantidad": corte,
            "validado": True,
        })

    return registros


def guardar_ranking(
    ranking,
    semilla,
):
    """
    Guarda el ranking completo como artefacto de análisis.
    Este archivo NO es un submit.
    """

    nombre = (
        f"ranking_s{semilla}"
        f"_m{SCORE_MES}.csv"
    )

    path = (
        OUTPUT_DIR
        / nombre
    )

    ranking.to_csv(
        path,
        index=False,
    )

    return path


def calcular_estabilidad(
    rankings,
):
    """
    Calcula Jaccard entre semillas para cada corte.

    Esto permite medir cuánto cambia la selección final
    cuando cambia únicamente la semilla del LightGBM.
    """

    titulo("ESTABILIDAD ENTRE SEMILLAS")

    resultados = []

    semillas = list(
        rankings.keys()
    )

    for corte in CORTES:

        jaccards = []

        for i in range(
            len(semillas)
        ):

            for j in range(
                i + 1,
                len(semillas),
            ):

                s1 = semillas[i]
                s2 = semillas[j]

                ids1 = set(
                    rankings[s1]
                    .head(corte)
                    ["numero_de_cliente"]
                )

                ids2 = set(
                    rankings[s2]
                    .head(corte)
                    ["numero_de_cliente"]
                )

                inter = len(
                    ids1 & ids2
                )

                union = len(
                    ids1 | ids2
                )

                jaccard = (
                    inter / union
                    if union > 0
                    else np.nan
                )

                jaccards.append(
                    jaccard
                )

        media = float(
            np.mean(jaccards)
        )

        minimo = float(
            np.min(jaccards)
        )

        maximo = float(
            np.max(jaccards)
        )

        resultados.append({
            "corte": corte,
            "jaccard_mean": media,
            "jaccard_min": minimo,
            "jaccard_max": maximo,
        })

    tabla = pd.DataFrame(
        resultados
    )

    print(
        tabla.to_string(
            index=False,
            formatters={
                "jaccard_mean":
                    lambda x: f"{x:.4f}",

                "jaccard_min":
                    lambda x: f"{x:.4f}",

                "jaccard_max":
                    lambda x: f"{x:.4f}",
            },
        )
    )

    path = (
        OUTPUT_DIR
        / "estabilidad_semillas.csv"
    )

    tabla.to_csv(
        path,
        index=False,
    )

    print()
    print(path)

    return tabla


def guardar_manifest(
    registros,
):
    """
    Guarda inventario de todos los archivos aptos para submit.
    """

    titulo("MANIFEST")

    manifest = pd.DataFrame(
        registros
    )

    path = (
        OUTPUT_DIR
        / "manifest_submits.csv"
    )

    manifest.to_csv(
        path,
        index=False,
    )

    print(
        f"Archivos de submit: "
        f"{len(manifest)}"
    )

    print(
        f"Semillas: "
        f"{manifest['semilla'].nunique()}"
    )

    print(
        f"Cortes: "
        f"{manifest['corte'].nunique()}"
    )

    print()
    print(path)

    return manifest


# =============================================================================
# MAIN
# =============================================================================

def main():

    titulo(
        "Z516 - PRIMERA COMPETENCIA - FINAL"
    )

    print(
        f"TRAIN: {TRAIN_DESDE}-{TRAIN_HASTA}"
    )

    print(
        f"SCORE: {SCORE_MES}"
    )

    print(
        f"Semillas: {SEMILLAS}"
    )

    print(
        f"Cortes: {CORTES}"
    )

    print()
    print(
        "Este script genera candidatos."
    )

    print(
        "NO realiza envíos a Zulip."
    )

    df = cargar_datos()

    (
        train,
        score,
        features,
        X_train,
        y_train,
        X_score,
    ) = preparar_datos(
        df
    )

    rankings = {}
    registros = []

    for semilla in SEMILLAS:

        modelo, prob = (
            entrenar_y_predecir(
                X_train,
                y_train,
                X_score,
                semilla,
            )
        )

        ranking = construir_ranking(
            score,
            prob,
            semilla,
        )

        rankings[
            semilla
        ] = ranking

        ranking_path = (
            guardar_ranking(
                ranking,
                semilla,
            )
        )

        print(
            f"Ranking completo: "
            f"{ranking_path}"
        )

        registros_semilla = (
            guardar_candidatos(
                ranking,
                semilla,
            )
        )

        registros.extend(
            registros_semilla
        )

        del modelo

    estabilidad = (
        calcular_estabilidad(
            rankings
        )
    )

    manifest = (
        guardar_manifest(
            registros
        )
    )

    titulo("CONTROL FINAL")

    esperados = (
        len(SEMILLAS)
        * len(CORTES)
    )

    generados = len(
        manifest
    )

    print(
        f"Archivos esperados: {esperados}"
    )

    print(
        f"Archivos generados: {generados}"
    )

    if generados != esperados:
        raise ValueError(
            "La cantidad de archivos generados "
            "no coincide con la esperada."
        )

    if not manifest[
        "validado"
    ].all():

        raise ValueError(
            "Existe al menos un submit "
            "que no pasó las validaciones."
        )

    titulo("FIN")

    print(
        "Todos los candidatos fueron "
        "generados y validados."
    )

    print()
    print(
        f"Directorio:"
    )

    print(
        OUTPUT_DIR
    )

    print()
    print(
        "IMPORTANTE: todavía NO se envió "
        "ningún archivo a Zulip."
    )


if __name__ == "__main__":
    main()