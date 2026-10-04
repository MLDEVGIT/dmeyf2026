#!/usr/bin/env python3
"""
z520_competencia_01_baseline_vs_trial46.py

Comparación final controlada:

    A) LightGBM baseline
    B) LightGBM Trial 46 de z517

TRAIN:
    202103, 202104, 202105, 202106

SCORE:
    202108

Target:
    BAJA+2 = 1
    BAJA+1 / CONTINUA = 0

Diseño:
    - 152 variables originales
    - mismas 5 semillas para ambas familias
    - N fijo = 12000
    - solo cambian los hiperparámetros
    - genera rankings completos
    - genera CSV de submit SIN header
    - calcula estabilidad entre semillas
    - calcula similitud Baseline vs Trial46

No:
    - usa 202107
    - usa target de 202108
    - ejecuta Optuna
    - envía archivos al bot
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd


# ============================================================
# Configuración
# ============================================================

DATASET_DEFAULT = Path(
    "/data/dmeyf/datasets/competencia_01.csv"
)

OUTPUT_DEFAULT = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/"
    "competencia_01/submits_z520"
)

TRAIN_MONTHS = [
    202103,
    202104,
    202105,
    202106,
]

SCORE_MONTH = 202108

ID_COL = "numero_de_cliente"
MONTH_COL = "foto_mes"
TARGET = "clase_ternaria"

N_SUBMIT = 12000

SEEDS = [
    290497,
    100003,
    200003,
    300007,
    400009,
]


# ============================================================
# Hiperparámetros
# ============================================================

BASELINE_PARAMS = {
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
}


TRIAL46_PARAMS = {
    "n_estimators": 1828,
    "learning_rate": 0.0115735794442499,
    "num_leaves": 665,
    "max_depth": -1,
    "min_child_samples": 1261,
    "max_bin": 31,
    "colsample_bytree": 0.9178523224363776,
    "subsample": 1.0,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
}


# ============================================================
# Argumentos
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Comparación final Baseline vs Trial46 "
            "para Competencia 01."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_DEFAULT,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DEFAULT,
    )

    parser.add_argument(
        "--n-submit",
        type=int,
        default=N_SUBMIT,
    )

    return parser.parse_args()


# ============================================================
# Utilidades
# ============================================================

def jaccard(
    a: set,
    b: set,
) -> float:

    union = a | b

    if not union:
        return 1.0

    return len(a & b) / len(union)


def validar_submit(
    path: Path,
    expected_n: int,
    score_ids: set,
):
    """
    Verifica:
      - cantidad exacta de filas
      - una sola columna
      - IDs enteros
      - sin duplicados
      - todos pertenecen a 202108
      - no tiene header
    """

    df = pd.read_csv(
        path,
        header=None,
    )

    if df.shape != (
        expected_n,
        1,
    ):
        raise ValueError(
            f"{path.name}: shape inválido "
            f"{df.shape}; esperado "
            f"({expected_n}, 1)"
        )

    serie = df.iloc[:, 0]

    if serie.isna().any():
        raise ValueError(
            f"{path.name}: contiene NA"
        )

    if serie.duplicated().any():
        raise ValueError(
            f"{path.name}: contiene IDs duplicados"
        )

    ids = set(
        serie.astype(
            np.int64
        ).tolist()
    )

    if not ids.issubset(
        score_ids
    ):
        raise ValueError(
            f"{path.name}: contiene IDs "
            "que no pertenecen a 202108"
        )

    # Control explícito de formato.
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        primera_linea = (
            f.readline()
            .strip()
        )

    if primera_linea.lower().startswith(
        ID_COL.lower()
    ):
        raise ValueError(
            f"{path.name}: tiene header"
        )

    try:
        int(
            primera_linea
        )
    except ValueError as exc:
        raise ValueError(
            f"{path.name}: primera línea "
            f"no es un ID entero: "
            f"{primera_linea}"
        ) from exc


# ============================================================
# Lectura de datos
# ============================================================

def cargar_datos(
    dataset: Path,
):
    if not dataset.exists():
        raise FileNotFoundError(
            f"No existe dataset: {dataset}"
        )

    meses = (
        TRAIN_MONTHS
        + [SCORE_MONTH]
    )

    meses_sql = ", ".join(
        str(x)
        for x in meses
    )

    query = f"""
        SELECT *
        FROM read_csv_auto(
            '{dataset}',
            header = true,
            sample_size = -1
        )
        WHERE foto_mes IN ({meses_sql})
    """

    print(
        "\nCargando datos con DuckDB...",
        flush=True,
    )

    inicio = time.time()

    con = duckdb.connect()

    try:
        df = (
            con.execute(
                query
            )
            .fetchdf()
        )
    finally:
        con.close()

    elapsed = (
        time.time()
        - inicio
    )

    print(
        f"Datos cargados: "
        f"{len(df):,} filas "
        f"en {elapsed:.1f} s",
        flush=True,
    )

    print(
        "\nFilas por mes:",
        flush=True,
    )

    print(
        df.groupby(
            MONTH_COL
        )
        .size()
        .to_string(),
        flush=True,
    )

    return df


# ============================================================
# Entrenamiento de una familia
# ============================================================

def entrenar_familia(
    nombre: str,
    params: dict,
    seeds: list[int],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_score: pd.DataFrame,
    score_ids: np.ndarray,
    output_dir: Path,
    n_submit: int,
):
    """
    Entrena una familia con todas las semillas.

    Devuelve:
        rankings
        top_sets
        manifest rows
    """

    rankings = {}
    top_sets = {}
    manifest_rows = []

    familia_dir = (
        output_dir
        / nombre
    )

    familia_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\n"
        + "=" * 80,
        flush=True,
    )

    print(
        f"FAMILIA: {nombre.upper()}",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    for i, seed in enumerate(
        seeds,
        start=1,
    ):
        print(
            f"\n[{i}/{len(seeds)}] "
            f"{nombre} - seed {seed}",
            flush=True,
        )

        modelo = lgb.LGBMClassifier(
            objective="binary",
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
            **params,
        )

        inicio = time.time()

        modelo.fit(
            X_train,
            y_train,
        )

        prob = (
            modelo.predict_proba(
                X_score
            )[:, 1]
        )

        elapsed = (
            time.time()
            - inicio
        )

        ranking = pd.DataFrame(
            {
                ID_COL: score_ids,
                "prob": prob,
            }
        )

        ranking = (
            ranking
            .sort_values(
                "prob",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

        rankings[seed] = ranking

        ranking_path = (
            familia_dir
            / (
                f"ranking_{nombre}"
                f"_seed{seed}.csv"
            )
        )

        ranking.to_csv(
            ranking_path,
            index=False,
        )

        top = (
            ranking
            .head(
                n_submit
            )[ID_COL]
            .astype(
                np.int64
            )
        )

        top_sets[seed] = set(
            top.tolist()
        )

        submit_path = (
            familia_dir
            / (
                f"submit_{nombre}"
                f"_seed{seed}"
                f"_n{n_submit}.csv"
            )
        )

        top.to_csv(
            submit_path,
            index=False,
            header=False,
        )

        manifest_rows.append(
            {
                "familia": nombre,
                "seed": seed,
                "n_submit": n_submit,
                "prob_min": float(
                    prob.min()
                ),
                "prob_max": float(
                    prob.max()
                ),
                "prob_mean": float(
                    prob.mean()
                ),
                "prob_std": float(
                    prob.std()
                ),
                "tiempo_segundos": float(
                    elapsed
                ),
                "ranking": str(
                    ranking_path
                ),
                "submit": str(
                    submit_path
                ),
            }
        )

        print(
            f"  tiempo    : "
            f"{elapsed:.1f} s",
            flush=True,
        )

        print(
            f"  prob mean : "
            f"{prob.mean():.8f}",
            flush=True,
        )

        print(
            f"  prob max  : "
            f"{prob.max():.8f}",
            flush=True,
        )

        print(
            f"  submit    : "
            f"{submit_path.name}",
            flush=True,
        )

    return (
        rankings,
        top_sets,
        manifest_rows,
    )


# ============================================================
# Estabilidad dentro de una familia
# ============================================================

def estabilidad_interna(
    nombre: str,
    top_sets: dict[int, set],
) -> pd.DataFrame:

    filas = []

    for seed_a, seed_b in itertools.combinations(
        top_sets.keys(),
        2,
    ):
        valor = jaccard(
            top_sets[seed_a],
            top_sets[seed_b],
        )

        filas.append(
            {
                "familia": nombre,
                "seed_a": seed_a,
                "seed_b": seed_b,
                "jaccard": valor,
            }
        )

    return pd.DataFrame(
        filas
    )


# ============================================================
# Comparación Baseline vs Trial46
# ============================================================

def comparar_familias(
    baseline_sets: dict[int, set],
    trial46_sets: dict[int, set],
) -> pd.DataFrame:

    filas = []

    # Primero: misma semilla contra misma semilla.
    for seed in SEEDS:

        a = baseline_sets[seed]
        b = trial46_sets[seed]

        inter = len(
            a & b
        )

        valor = jaccard(
            a,
            b,
        )

        filas.append(
            {
                "comparacion":
                    "misma_semilla",
                "seed_baseline":
                    seed,
                "seed_trial46":
                    seed,
                "interseccion":
                    inter,
                "jaccard":
                    valor,
            }
        )

    # Luego: todas las combinaciones.
    for seed_b, set_b in (
        baseline_sets.items()
    ):
        for seed_t, set_t in (
            trial46_sets.items()
        ):

            inter = len(
                set_b & set_t
            )

            valor = jaccard(
                set_b,
                set_t,
            )

            filas.append(
                {
                    "comparacion":
                        "todas_combinaciones",
                    "seed_baseline":
                        seed_b,
                    "seed_trial46":
                        seed_t,
                    "interseccion":
                        inter,
                    "jaccard":
                        valor,
                }
            )

    return pd.DataFrame(
        filas
    )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    print(
        "z520 - BASELINE VS TRIAL46 - COMPETENCIA 01",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    print(
        f"\nDataset : "
        f"{args.dataset}",
        flush=True,
    )

    print(
        f"Output  : "
        f"{args.output_dir}",
        flush=True,
    )

    print(
        f"TRAIN   : "
        f"{TRAIN_MONTHS}",
        flush=True,
    )

    print(
        f"SCORE   : "
        f"{SCORE_MONTH}",
        flush=True,
    )

    print(
        f"N       : "
        f"{args.n_submit}",
        flush=True,
    )

    print(
        f"Seeds   : "
        f"{SEEDS}",
        flush=True,
    )

    # --------------------------------------------------------
    # Datos
    # --------------------------------------------------------

    df = cargar_datos(
        args.dataset
    )

    train = df[
        df[MONTH_COL].isin(
            TRAIN_MONTHS
        )
    ].copy()

    score = df[
        df[MONTH_COL]
        == SCORE_MONTH
    ].copy()

    if train[
        TARGET
    ].isna().any():
        raise ValueError(
            "Hay targets NA en TRAIN."
        )

    if len(
        score
    ) == 0:
        raise ValueError(
            "No hay registros para SCORE."
        )

    if score[
        ID_COL
    ].duplicated().any():
        raise ValueError(
            "Hay numero_de_cliente "
            "duplicados en SCORE."
        )

    feature_cols = [
        c
        for c in df.columns
        if c not in {
            ID_COL,
            MONTH_COL,
            TARGET,
        }
    ]

    if len(
        feature_cols
    ) != 152:
        raise ValueError(
            f"Se esperaban 152 variables; "
            f"se encontraron "
            f"{len(feature_cols)}"
        )

    X_train = train[
        feature_cols
    ]

    X_score = score[
        feature_cols
    ]

    y_train = (
        train[TARGET]
        .eq("BAJA+2")
        .astype(
            np.int8
        )
        .to_numpy()
    )

    score_ids = (
        score[ID_COL]
        .astype(
            np.int64
        )
        .to_numpy()
    )

    score_ids_set = set(
        score_ids.tolist()
    )

    print(
        f"\nTRAIN filas : "
        f"{len(train):,}",
        flush=True,
    )

    print(
        f"SCORE filas : "
        f"{len(score):,}",
        flush=True,
    )

    print(
        f"Variables   : "
        f"{len(feature_cols)}",
        flush=True,
    )

    print(
        f"TRAIN BAJA+2: "
        f"{int(y_train.sum()):,}",
        flush=True,
    )

    print(
        f"SCORE IDs únicos: "
        f"{len(score_ids_set):,}",
        flush=True,
    )

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    (
        baseline_rankings,
        baseline_sets,
        manifest_baseline,
    ) = entrenar_familia(
        nombre="baseline",
        params=BASELINE_PARAMS,
        seeds=SEEDS,
        X_train=X_train,
        y_train=y_train,
        X_score=X_score,
        score_ids=score_ids,
        output_dir=args.output_dir,
        n_submit=args.n_submit,
    )

    # --------------------------------------------------------
    # Trial 46
    # --------------------------------------------------------

    (
        trial46_rankings,
        trial46_sets,
        manifest_trial46,
    ) = entrenar_familia(
        nombre="trial46",
        params=TRIAL46_PARAMS,
        seeds=SEEDS,
        X_train=X_train,
        y_train=y_train,
        X_score=X_score,
        score_ids=score_ids,
        output_dir=args.output_dir,
        n_submit=args.n_submit,
    )

    # Evita warnings por variables no usadas
    _ = (
        baseline_rankings,
        trial46_rankings,
    )

    # --------------------------------------------------------
    # Validación física de los 10 submits
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 80,
        flush=True,
    )

    print(
        "VALIDANDO ARCHIVOS DE SUBMIT",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    manifest = pd.DataFrame(
        manifest_baseline
        + manifest_trial46
    )

    for submit_str in manifest[
        "submit"
    ]:

        submit_path = Path(
            submit_str
        )

        validar_submit(
            path=submit_path,
            expected_n=args.n_submit,
            score_ids=score_ids_set,
        )

        print(
            f"OK: {submit_path.name}",
            flush=True,
        )

    # --------------------------------------------------------
    # Estabilidad interna
    # --------------------------------------------------------

    estabilidad_baseline = (
        estabilidad_interna(
            "baseline",
            baseline_sets,
        )
    )

    estabilidad_trial46 = (
        estabilidad_interna(
            "trial46",
            trial46_sets,
        )
    )

    estabilidad = pd.concat(
        [
            estabilidad_baseline,
            estabilidad_trial46,
        ],
        ignore_index=True,
    )

    estabilidad_path = (
        args.output_dir
        / "estabilidad_interna_z520.csv"
    )

    estabilidad.to_csv(
        estabilidad_path,
        index=False,
    )

    # --------------------------------------------------------
    # Comparación entre familias
    # --------------------------------------------------------

    comparacion = (
        comparar_familias(
            baseline_sets,
            trial46_sets,
        )
    )

    comparacion_path = (
        args.output_dir
        / "comparacion_baseline_trial46_z520.csv"
    )

    comparacion.to_csv(
        comparacion_path,
        index=False,
    )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest_path = (
        args.output_dir
        / "manifest_z520.csv"
    )

    manifest.to_csv(
        manifest_path,
        index=False,
    )

    # --------------------------------------------------------
    # Resumen
    # --------------------------------------------------------

    resumen_estabilidad = (
        estabilidad
        .groupby(
            "familia"
        )["jaccard"]
        .agg(
            [
                "mean",
                "min",
                "max",
                "std",
            ]
        )
    )

    misma_semilla = comparacion[
        comparacion[
            "comparacion"
        ]
        == "misma_semilla"
    ]

    todas = comparacion[
        comparacion[
            "comparacion"
        ]
        == "todas_combinaciones"
    ]

    resumen = {
        "train_months":
            TRAIN_MONTHS,
        "score_month":
            SCORE_MONTH,
        "n_submit":
            args.n_submit,
        "seeds":
            SEEDS,
        "n_features":
            len(feature_cols),
        "train_rows":
            len(train),
        "score_rows":
            len(score),
        "train_baja2":
            int(y_train.sum()),

        "baseline_internal_jaccard_mean":
            float(
                estabilidad_baseline[
                    "jaccard"
                ].mean()
            ),

        "trial46_internal_jaccard_mean":
            float(
                estabilidad_trial46[
                    "jaccard"
                ].mean()
            ),

        "baseline_vs_trial46_same_seed_jaccard_mean":
            float(
                misma_semilla[
                    "jaccard"
                ].mean()
            ),

        "baseline_vs_trial46_all_jaccard_mean":
            float(
                todas[
                    "jaccard"
                ].mean()
            ),
    }

    resumen_path = (
        args.output_dir
        / "resumen_z520.json"
    )

    with resumen_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            resumen,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Mostrar resultado
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 80,
        flush=True,
    )

    print(
        "RESULTADO FINAL z520",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    print(
        "\nEstabilidad interna "
        "N=12000:",
        flush=True,
    )

    print(
        resumen_estabilidad.to_string(
            float_format=lambda x:
                f"{x:.6f}"
        ),
        flush=True,
    )

    print(
        "\nBaseline vs Trial46 "
        "- misma semilla:",
        flush=True,
    )

    print(
        misma_semilla[
            [
                "seed_baseline",
                "seed_trial46",
                "interseccion",
                "jaccard",
            ]
        ].to_string(
            index=False,
            formatters={
                "jaccard":
                    lambda x:
                        f"{x:.6f}"
            },
        ),
        flush=True,
    )

    print(
        "\nResumen entre familias:",
        flush=True,
    )

    print(
        f"  Jaccard misma semilla mean : "
        f"{misma_semilla['jaccard'].mean():.6f}",
        flush=True,
    )

    print(
        f"  Jaccard misma semilla min  : "
        f"{misma_semilla['jaccard'].min():.6f}",
        flush=True,
    )

    print(
        f"  Jaccard misma semilla max  : "
        f"{misma_semilla['jaccard'].max():.6f}",
        flush=True,
    )

    print(
        f"  Jaccard todas comb. mean   : "
        f"{todas['jaccard'].mean():.6f}",
        flush=True,
    )

    print(
        "\nProbabilidades:",
        flush=True,
    )

    print(
        manifest[
            [
                "familia",
                "seed",
                "prob_mean",
                "prob_std",
                "prob_max",
                "tiempo_segundos",
            ]
        ].to_string(
            index=False,
            formatters={
                "prob_mean":
                    lambda x:
                        f"{x:.8f}",
                "prob_std":
                    lambda x:
                        f"{x:.8f}",
                "prob_max":
                    lambda x:
                        f"{x:.8f}",
                "tiempo_segundos":
                    lambda x:
                        f"{x:.1f}",
            },
        ),
        flush=True,
    )

    print(
        "\nArchivos generados:",
        flush=True,
    )

    print(
        f"  {manifest_path}",
        flush=True,
    )

    print(
        f"  {estabilidad_path}",
        flush=True,
    )

    print(
        f"  {comparacion_path}",
        flush=True,
    )

    print(
        f"  {resumen_path}",
        flush=True,
    )

    print(
        "\nLos 10 submits fueron "
        "validados correctamente.",
        flush=True,
    )

    print(
        "\nIMPORTANTE:",
        flush=True,
    )

    print(
        "z520 NO decide el ganador usando 202108.",
        flush=True,
    )

    print(
        "Solo genera rankings y mide estabilidad.",
        flush=True,
    )

    print(
        "La comparación con el Public debe "
        "mantener N=12000 para ambas familias.",
        flush=True,
    )


if __name__ == "__main__":
    main()