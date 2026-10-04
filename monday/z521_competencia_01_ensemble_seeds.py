#!/usr/bin/env python3
"""
z521_competencia_01_ensemble_seeds.py

Ensemble de las 5 semillas del modelo baseline de z520.

No entrena modelos.

Lee los rankings completos:
    ranking_baseline_seed*.csv

Para cada numero_de_cliente:
    prob_ensemble = promedio(prob_seed1, ..., prob_seed5)

Luego:
    - ordena por prob_ensemble descendente
    - genera ranking completo
    - selecciona TOP N=12000
    - genera submit sin header
    - valida el submit
    - compara el TOP ensemble contra cada semilla individual
      mediante intersección y Jaccard

Objetivo:
    reducir la varianza entre semillas sin modificar
    hiperparámetros, datos de entrenamiento ni corte.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuración
# ============================================================

INPUT_DEFAULT = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/"
    "competencia_01/submits_z520/baseline"
)

OUTPUT_DEFAULT = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/"
    "competencia_01/ensemble_z521"
)

ID_COL = "numero_de_cliente"
PROB_COL = "prob"

SEEDS = [
    290497,
    100003,
    200003,
    300007,
    400009,
]

N_SUBMIT = 12000

EXPECTED_SCORE_ROWS = 164647


# ============================================================
# Argumentos
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Ensemble por promedio de probabilidades "
            "de las semillas baseline de z520."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DEFAULT,
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
    a: set[int],
    b: set[int],
) -> float:

    union = a | b

    if not union:
        return 1.0

    return len(a & b) / len(union)


def validar_ranking(
    df: pd.DataFrame,
    seed: int,
):

    columnas = set(
        df.columns
    )

    requeridas = {
        ID_COL,
        PROB_COL,
    }

    if not requeridas.issubset(
        columnas
    ):
        raise ValueError(
            f"Seed {seed}: faltan columnas. "
            f"Columnas encontradas: "
            f"{list(df.columns)}"
        )

    if len(df) != EXPECTED_SCORE_ROWS:
        raise ValueError(
            f"Seed {seed}: "
            f"{len(df):,} filas; "
            f"esperadas "
            f"{EXPECTED_SCORE_ROWS:,}"
        )

    if df[
        ID_COL
    ].isna().any():
        raise ValueError(
            f"Seed {seed}: IDs NA"
        )

    if df[
        PROB_COL
    ].isna().any():
        raise ValueError(
            f"Seed {seed}: probabilidades NA"
        )

    if df[
        ID_COL
    ].duplicated().any():
        raise ValueError(
            f"Seed {seed}: IDs duplicados"
        )

    if not np.isfinite(
        df[PROB_COL].to_numpy()
    ).all():
        raise ValueError(
            f"Seed {seed}: "
            "probabilidades no finitas"
        )


def validar_submit(
    path: Path,
    expected_n: int,
    valid_ids: set[int],
):

    df = pd.read_csv(
        path,
        header=None,
    )

    if df.shape != (
        expected_n,
        1,
    ):
        raise ValueError(
            f"Shape inválido: "
            f"{df.shape}; "
            f"esperado "
            f"({expected_n}, 1)"
        )

    serie = df.iloc[
        :,
        0,
    ]

    if serie.isna().any():
        raise ValueError(
            "Submit contiene NA"
        )

    if serie.duplicated().any():
        raise ValueError(
            "Submit contiene IDs duplicados"
        )

    try:
        ids = set(
            serie.astype(
                np.int64
            ).tolist()
        )
    except Exception as exc:
        raise ValueError(
            "Los IDs no pudieron "
            "convertirse a enteros"
        ) from exc

    if not ids.issubset(
        valid_ids
    ):
        raise ValueError(
            "Submit contiene IDs "
            "fuera del universo 202108"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        primera = (
            f.readline()
            .strip()
        )

    if primera.lower() == (
        ID_COL.lower()
    ):
        raise ValueError(
            "El submit tiene header"
        )

    try:
        int(
            primera
        )
    except ValueError as exc:
        raise ValueError(
            f"Primera línea no entera: "
            f"{primera}"
        ) from exc


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
        "z521 - ENSEMBLE DE SEMILLAS BASELINE",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    print(
        f"\nInput : "
        f"{args.input_dir}",
        flush=True,
    )

    print(
        f"Output: "
        f"{args.output_dir}",
        flush=True,
    )

    print(
        f"N     : "
        f"{args.n_submit}",
        flush=True,
    )

    print(
        f"Seeds : "
        f"{SEEDS}",
        flush=True,
    )

    # --------------------------------------------------------
    # Lectura de rankings
    # --------------------------------------------------------

    frames = []

    ids_referencia = None

    top_individuales = {}

    print(
        "\nLeyendo rankings...",
        flush=True,
    )

    for seed in SEEDS:

        path = (
            args.input_dir
            / (
                f"ranking_baseline_"
                f"seed{seed}.csv"
            )
        )

        if not path.exists():
            raise FileNotFoundError(
                f"No existe: {path}"
            )

        df = pd.read_csv(
            path
        )

        validar_ranking(
            df=df,
            seed=seed,
        )

        # Convertimos explícitamente ID.
        df[ID_COL] = (
            df[ID_COL]
            .astype(
                np.int64
            )
        )

        # El ranking ya viene ordenado,
        # pero no dependemos de eso.
        top_seed = (
            df
            .sort_values(
                PROB_COL,
                ascending=False,
            )
            .head(
                args.n_submit
            )[ID_COL]
        )

        top_individuales[
            seed
        ] = set(
            top_seed.tolist()
        )

        ids_actuales = set(
            df[
                ID_COL
            ].tolist()
        )

        if ids_referencia is None:

            ids_referencia = (
                ids_actuales
            )

        elif ids_actuales != (
            ids_referencia
        ):

            raise ValueError(
                f"Seed {seed}: "
                "el universo de clientes "
                "no coincide con las "
                "otras semillas"
            )

        df_seed = (
            df[
                [
                    ID_COL,
                    PROB_COL,
                ]
            ]
            .rename(
                columns={
                    PROB_COL:
                        f"prob_{seed}"
                }
            )
        )

        frames.append(
            df_seed
        )

        print(
            f"  OK seed {seed}: "
            f"{len(df):,} clientes | "
            f"prob mean="
            f"{df[PROB_COL].mean():.8f}",
            flush=True,
        )

    # --------------------------------------------------------
    # Merge cliente por cliente
    # --------------------------------------------------------

    print(
        "\nUniendo probabilidades "
        "por numero_de_cliente...",
        flush=True,
    )

    ensemble = (
        frames[0]
        .copy()
    )

    for frame in frames[
        1:
    ]:

        ensemble = (
            ensemble.merge(
                frame,
                on=ID_COL,
                how="inner",
                validate="one_to_one",
            )
        )

    if len(
        ensemble
    ) != EXPECTED_SCORE_ROWS:

        raise ValueError(
            f"Merge produjo "
            f"{len(ensemble):,} filas; "
            f"esperadas "
            f"{EXPECTED_SCORE_ROWS:,}"
        )

    prob_cols = [
        f"prob_{seed}"
        for seed in SEEDS
    ]

    # --------------------------------------------------------
    # Ensemble
    # --------------------------------------------------------

    ensemble[
        "prob_ensemble"
    ] = (
        ensemble[
            prob_cols
        ]
        .mean(
            axis=1
        )
    )

    # También dejamos desvío entre semillas
    # por cliente como diagnóstico.
    ensemble[
        "prob_std_seeds"
    ] = (
        ensemble[
            prob_cols
        ]
        .std(
            axis=1,
            ddof=0,
        )
    )

    ensemble = (
        ensemble
        .sort_values(
            "prob_ensemble",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    ensemble[
        "rank_ensemble"
    ] = (
        np.arange(
            1,
            len(ensemble) + 1,
            dtype=np.int64,
        )
    )

    # --------------------------------------------------------
    # Ranking completo
    # --------------------------------------------------------

    ranking_path = (
        args.output_dir
        / "ranking_baseline_ensemble_5seeds.csv"
    )

    ensemble.to_csv(
        ranking_path,
        index=False,
    )

    # --------------------------------------------------------
    # TOP N y submit
    # --------------------------------------------------------

    top_ensemble = (
        ensemble
        .head(
            args.n_submit
        )[ID_COL]
        .astype(
            np.int64
        )
    )

    top_ensemble_set = set(
        top_ensemble.tolist()
    )

    submit_path = (
        args.output_dir
        / (
            "submit_baseline_"
            "ensemble5seeds_"
            f"n{args.n_submit}.csv"
        )
    )

    top_ensemble.to_csv(
        submit_path,
        index=False,
        header=False,
    )

    validar_submit(
        path=submit_path,
        expected_n=args.n_submit,
        valid_ids=ids_referencia,
    )

    # --------------------------------------------------------
    # Comparación contra cada semilla
    # --------------------------------------------------------

    comparaciones = []

    for seed in SEEDS:

        top_seed = (
            top_individuales[
                seed
            ]
        )

        inter = len(
            top_ensemble_set
            & top_seed
        )

        union = len(
            top_ensemble_set
            | top_seed
        )

        jac = jaccard(
            top_ensemble_set,
            top_seed,
        )

        solo_ensemble = len(
            top_ensemble_set
            - top_seed
        )

        solo_seed = len(
            top_seed
            - top_ensemble_set
        )

        comparaciones.append(
            {
                "seed": seed,
                "interseccion":
                    inter,
                "union":
                    union,
                "jaccard":
                    jac,
                "solo_ensemble":
                    solo_ensemble,
                "solo_seed":
                    solo_seed,
            }
        )

    comparacion_df = (
        pd.DataFrame(
            comparaciones
        )
    )

    comparacion_path = (
        args.output_dir
        / (
            "comparacion_ensemble_"
            "vs_seeds_z521.csv"
        )
    )

    comparacion_df.to_csv(
        comparacion_path,
        index=False,
    )

    # --------------------------------------------------------
    # Consenso de semillas dentro del TOP ensemble
    # --------------------------------------------------------

    # Para cada cliente del TOP ensemble,
    # contamos en cuántos TOP-12000
    # individuales aparece.
    consenso = []

    for cliente in (
        top_ensemble.tolist()
    ):

        apariciones = sum(
            cliente
            in top_individuales[
                seed
            ]
            for seed in SEEDS
        )

        consenso.append(
            apariciones
        )

    consenso_serie = pd.Series(
        consenso,
        name="n_seeds_top",
    )

    consenso_counts = (
        consenso_serie
        .value_counts()
        .sort_index()
    )

    consenso_df = pd.DataFrame(
        {
            "n_seeds_top":
                consenso_counts.index,
            "clientes":
                consenso_counts.values,
        }
    )

    consenso_df[
        "porcentaje"
    ] = (
        100.0
        * consenso_df[
            "clientes"
        ]
        / args.n_submit
    )

    consenso_path = (
        args.output_dir
        / "consenso_top_z521.csv"
    )

    consenso_df.to_csv(
        consenso_path,
        index=False,
    )

    # --------------------------------------------------------
    # Resumen
    # --------------------------------------------------------

    cutoff_prob = float(
        ensemble.iloc[
            args.n_submit - 1
        ][
            "prob_ensemble"
        ]
    )

    resumen = {
        "seeds":
            SEEDS,
        "n_score":
            len(ensemble),
        "n_submit":
            args.n_submit,
        "prob_ensemble_mean":
            float(
                ensemble[
                    "prob_ensemble"
                ].mean()
            ),
        "prob_ensemble_std":
            float(
                ensemble[
                    "prob_ensemble"
                ].std(
                    ddof=0
                )
            ),
        "prob_ensemble_max":
            float(
                ensemble[
                    "prob_ensemble"
                ].max()
            ),
        "prob_cutoff_n":
            cutoff_prob,
        "jaccard_mean_vs_seeds":
            float(
                comparacion_df[
                    "jaccard"
                ].mean()
            ),
        "jaccard_min_vs_seeds":
            float(
                comparacion_df[
                    "jaccard"
                ].min()
            ),
        "jaccard_max_vs_seeds":
            float(
                comparacion_df[
                    "jaccard"
                ].max()
            ),
    }

    resumen_path = (
        args.output_dir
        / "resumen_z521.json"
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
    # Salida
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 80,
        flush=True,
    )

    print(
        "RESULTADO FINAL z521",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    print(
        "\nEnsemble:",
        flush=True,
    )

    print(
        f"  Clientes score : "
        f"{len(ensemble):,}",
        flush=True,
    )

    print(
        f"  TOP seleccionado: "
        f"{args.n_submit:,}",
        flush=True,
    )

    print(
        f"  Prob mean       : "
        f"{ensemble['prob_ensemble'].mean():.8f}",
        flush=True,
    )

    print(
        f"  Prob max        : "
        f"{ensemble['prob_ensemble'].max():.8f}",
        flush=True,
    )

    print(
        f"  Prob corte N    : "
        f"{cutoff_prob:.8f}",
        flush=True,
    )

    print(
        "\nEnsemble vs semillas individuales:",
        flush=True,
    )

    print(
        comparacion_df.to_string(
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
        "\nJaccard:",
        flush=True,
    )

    print(
        f"  mean: "
        f"{comparacion_df['jaccard'].mean():.6f}",
        flush=True,
    )

    print(
        f"  min : "
        f"{comparacion_df['jaccard'].min():.6f}",
        flush=True,
    )

    print(
        f"  max : "
        f"{comparacion_df['jaccard'].max():.6f}",
        flush=True,
    )

    print(
        "\nConsenso dentro del TOP ensemble:",
        flush=True,
    )

    print(
        consenso_df.to_string(
            index=False,
            formatters={
                "porcentaje":
                    lambda x:
                        f"{x:.2f}%"
            },
        ),
        flush=True,
    )

    print(
        "\nArchivos:",
        flush=True,
    )

    print(
        f"  Ranking    : "
        f"{ranking_path}",
        flush=True,
    )

    print(
        f"  Submit     : "
        f"{submit_path}",
        flush=True,
    )

    print(
        f"  Comparación: "
        f"{comparacion_path}",
        flush=True,
    )

    print(
        f"  Consenso   : "
        f"{consenso_path}",
        flush=True,
    )

    print(
        f"  Resumen    : "
        f"{resumen_path}",
        flush=True,
    )

    print(
        "\nSubmit validado correctamente:",
        flush=True,
    )

    print(
        f"  {args.n_submit:,} IDs",
        flush=True,
    )

    print(
        "  sin header",
        flush=True,
    )

    print(
        "  sin duplicados",
        flush=True,
    )

    print(
        "  IDs pertenecientes al universo 202108",
        flush=True,
    )


if __name__ == "__main__":
    main()