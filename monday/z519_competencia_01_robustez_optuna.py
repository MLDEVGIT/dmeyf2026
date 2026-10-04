#!/usr/bin/env python3
"""
z519_competencia_01_robustez_optuna.py

Validación de robustez temporal de los mejores trials de z517.

Origen de candidatos:
    trials_z517.csv

Selección:
    TOP N trials según ganancia en TEST 202106.

Validación independiente:
    TRAIN: 202103, 202104
    TEST : 202105

Para cada candidato:
    - reentrena con sus hiperparámetros congelados
    - calcula AUC, AP y LogLoss
    - calcula curva de ganancia
    - obtiene mejor corte y ganancia en 202105
    - compara contra el baseline de cada período

No:
    - ejecuta Optuna
    - usa 202107
    - usa 202108
    - genera submits
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    log_loss,
    roc_auc_score,
)


# ============================================================
# Configuración
# ============================================================

DATASET_DEFAULT = Path(
    "/data/dmeyf/datasets/competencia_01.csv"
)

TRIALS_DEFAULT = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/"
    "competencia_01/optuna_z517/trials_z517.csv"
)

OUTPUT_DEFAULT = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/"
    "competencia_01/robustez_z519"
)

TRAIN_MONTHS = [202103, 202104]
TEST_MONTH = 202105

TARGET = "clase_ternaria"
ID_COL = "numero_de_cliente"
MONTH_COL = "foto_mes"

GANANCIA_BAJA2 = 1_072_500
COSTO_ESTIMULO = -27_500

SEMILLA = 290497

CORTES = np.arange(
    4000,
    19001,
    500,
    dtype=int,
)

# Baselines reproducidos previamente con z515/z518.
BASELINE_GAIN_202105 = 248_050_000
BASELINE_GAIN_202106 = 398_200_000


# ============================================================
# Argumentos
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Robustez temporal TOP trials z517."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_DEFAULT,
    )

    parser.add_argument(
        "--trials-file",
        type=Path,
        default=TRIALS_DEFAULT,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DEFAULT,
    )

    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help=(
            "Cantidad de mejores trials de z517 "
            "a validar."
        ),
    )

    return parser.parse_args()


# ============================================================
# Utilidades
# ============================================================

def fmt_millones(x: float) -> str:
    return f"{x / 1_000_000:.3f} M"


def calcular_ganancias(
    y_true: np.ndarray,
    prob: np.ndarray,
) -> pd.DataFrame:

    orden = np.argsort(-prob)
    y_ord = y_true[orden]

    ganancias_individuales = np.where(
        y_ord == 1,
        GANANCIA_BAJA2,
        COSTO_ESTIMULO,
    )

    acumulada = np.cumsum(
        ganancias_individuales
    )

    filas = []

    for corte in CORTES:
        if corte > len(y_ord):
            continue

        y_top = y_ord[:corte]

        positivos = int(y_top.sum())

        precision = (
            positivos / corte
        )

        total_positivos = int(
            y_true.sum()
        )

        recall = (
            positivos / total_positivos
            if total_positivos > 0
            else np.nan
        )

        ganancia = int(
            acumulada[corte - 1]
        )

        filas.append(
            {
                "corte": int(corte),
                "positivos": positivos,
                "precision": precision,
                "recall": recall,
                "ganancia": ganancia,
            }
        )

    return pd.DataFrame(filas)


# ============================================================
# Lectura de candidatos z517
# ============================================================

def cargar_candidatos(
    trials_file: Path,
    top_n: int,
) -> pd.DataFrame:

    if not trials_file.exists():
        raise FileNotFoundError(
            f"No existe: {trials_file}"
        )

    trials = pd.read_csv(
        trials_file
    )

    columnas_necesarias = {
        "trial",
        "ganancia",
        "corte",
        "auc",
        "ap",
        "num_iterations",
        "learning_rate",
        "feature_fraction",
        "num_leaves",
        "min_data_in_leaf",
    }

    faltantes = (
        columnas_necesarias
        - set(trials.columns)
    )

    if faltantes:
        raise ValueError(
            "Faltan columnas en trials_z517.csv: "
            f"{sorted(faltantes)}"
        )

    candidatos = (
        trials
        .sort_values(
            ["ganancia", "auc"],
            ascending=[False, False],
        )
        .head(top_n)
        .copy()
        .reset_index(drop=True)
    )

    print("\nCandidatos seleccionados de z517:")

    print(
        candidatos[
            [
                "trial",
                "ganancia",
                "corte",
                "auc",
                "ap",
            ]
        ].to_string(
            index=False
        )
    )

    return candidatos


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
        + [TEST_MONTH]
    )

    meses_sql = ", ".join(
        str(x) for x in meses
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

    print("\nCargando datos con DuckDB...")

    inicio = time.time()

    con = duckdb.connect()

    try:
        df = (
            con.execute(query)
            .fetchdf()
        )
    finally:
        con.close()

    print(
        f"Datos cargados: {len(df):,} filas "
        f"en {time.time() - inicio:.1f} s"
    )

    print("\nFilas por mes:")

    print(
        df.groupby(MONTH_COL)
        .size()
        .to_string()
    )

    return df


# ============================================================
# Evaluación de un trial
# ============================================================

def evaluar_trial(
    fila: pd.Series,
    X_train,
    y_train,
    X_test,
    y_test,
):

    trial_num = int(
        fila["trial"]
    )

    params = {
        "num_iterations": int(
            fila["num_iterations"]
        ),
        "learning_rate": float(
            fila["learning_rate"]
        ),
        "feature_fraction": float(
            fila["feature_fraction"]
        ),
        "num_leaves": int(
            fila["num_leaves"]
        ),
        "min_data_in_leaf": int(
            fila["min_data_in_leaf"]
        ),
    }

    print("\n" + "=" * 70)
    print(
        f"TRIAL {trial_num} - VALIDACION 202105"
    )
    print("=" * 70)

    print(
        f"Ganancia original 202106: "
        f"{fmt_millones(fila['ganancia'])}"
    )

    print(
        f"Corte original 202106   : "
        f"{int(fila['corte'])}"
    )

    modelo = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=params[
            "num_iterations"
        ],
        learning_rate=params[
            "learning_rate"
        ],
        num_leaves=params[
            "num_leaves"
        ],
        max_depth=-1,
        min_child_samples=params[
            "min_data_in_leaf"
        ],
        max_bin=31,
        colsample_bytree=params[
            "feature_fraction"
        ],
        subsample=1.0,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=SEMILLA,
        n_jobs=-1,
        verbosity=-1,
    )

    inicio = time.time()

    modelo.fit(
        X_train,
        y_train,
    )

    prob = modelo.predict_proba(
        X_test
    )[:, 1]

    elapsed = (
        time.time()
        - inicio
    )

    auc_105 = roc_auc_score(
        y_test,
        prob,
    )

    ap_105 = average_precision_score(
        y_test,
        prob,
    )

    logloss_105 = log_loss(
        y_test,
        prob,
    )

    tabla = calcular_ganancias(
        y_true=y_test,
        prob=prob,
    )

    idx = tabla[
        "ganancia"
    ].idxmax()

    mejor = tabla.loc[idx]

    gain_105 = int(
        mejor["ganancia"]
    )

    corte_105 = int(
        mejor["corte"]
    )

    delta_105 = (
        gain_105
        - BASELINE_GAIN_202105
    )

    delta_106 = (
        int(fila["ganancia"])
        - BASELINE_GAIN_202106
    )

    pct_105 = (
        100.0
        * delta_105
        / BASELINE_GAIN_202105
    )

    pct_106 = (
        100.0
        * delta_106
        / BASELINE_GAIN_202106
    )

    print(
        f"\nAUC 202105      : "
        f"{auc_105:.6f}"
    )

    print(
        f"AP 202105       : "
        f"{ap_105:.6f}"
    )

    print(
        f"LogLoss 202105  : "
        f"{logloss_105:.6f}"
    )

    print(
        f"Mejor N 202105  : "
        f"{corte_105}"
    )

    print(
        f"Ganancia 202105 : "
        f"{fmt_millones(gain_105)}"
    )

    print(
        f"Delta baseline  : "
        f"{fmt_millones(delta_105)} "
        f"({pct_105:+.2f}%)"
    )

    print(
        f"Tiempo           : "
        f"{elapsed:.1f} s"
    )

    return {
        "trial": trial_num,

        "gain_202105": gain_105,
        "corte_202105": corte_105,
        "auc_202105": float(
            auc_105
        ),
        "ap_202105": float(
            ap_105
        ),
        "logloss_202105": float(
            logloss_105
        ),

        "gain_202106": int(
            fila["ganancia"]
        ),
        "corte_202106": int(
            fila["corte"]
        ),
        "auc_202106": float(
            fila["auc"]
        ),
        "ap_202106": float(
            fila["ap"]
        ),

        "delta_gain_202105": (
            delta_105
        ),
        "delta_pct_202105": (
            pct_105
        ),

        "delta_gain_202106": (
            delta_106
        ),
        "delta_pct_202106": (
            pct_106
        ),

        "num_iterations": params[
            "num_iterations"
        ],
        "learning_rate": params[
            "learning_rate"
        ],
        "feature_fraction": params[
            "feature_fraction"
        ],
        "num_leaves": params[
            "num_leaves"
        ],
        "min_data_in_leaf": params[
            "min_data_in_leaf"
        ],

        "tiempo_segundos": float(
            elapsed
        ),
    }


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print(
        "z519 - ROBUSTEZ TEMPORAL TOP TRIALS OPTUNA"
    )
    print("=" * 70)

    print(
        f"\nDataset     : {args.dataset}"
    )

    print(
        f"Trials      : {args.trials_file}"
    )

    print(
        f"Output      : {args.output_dir}"
    )

    print(
        f"TOP trials  : {args.top}"
    )

    print(
        f"TRAIN       : {TRAIN_MONTHS}"
    )

    print(
        f"TEST        : {TEST_MONTH}"
    )

    # --------------------------------------------------------
    # Candidatos
    # --------------------------------------------------------

    candidatos = cargar_candidatos(
        trials_file=args.trials_file,
        top_n=args.top,
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

    test = df[
        df[MONTH_COL]
        == TEST_MONTH
    ].copy()

    feature_cols = [
        c
        for c in df.columns
        if c not in {
            ID_COL,
            MONTH_COL,
            TARGET,
        }
    ]

    X_train = train[
        feature_cols
    ]

    X_test = test[
        feature_cols
    ]

    y_train = (
        train[TARGET]
        .eq("BAJA+2")
        .astype(np.int8)
        .to_numpy()
    )

    y_test = (
        test[TARGET]
        .eq("BAJA+2")
        .astype(np.int8)
        .to_numpy()
    )

    print(
        f"\nTRAIN filas : "
        f"{len(train):,}"
    )

    print(
        f"TEST filas  : "
        f"{len(test):,}"
    )

    print(
        f"Variables   : "
        f"{len(feature_cols)}"
    )

    print(
        f"TRAIN BAJA+2: "
        f"{int(y_train.sum()):,}"
    )

    print(
        f"TEST BAJA+2 : "
        f"{int(y_test.sum()):,}"
    )

    # --------------------------------------------------------
    # Evaluar TOP trials
    # --------------------------------------------------------

    resultados = []

    inicio_total = time.time()

    for _, fila in candidatos.iterrows():

        resultado = evaluar_trial(
            fila=fila,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        resultados.append(
            resultado
        )

    tiempo_total = (
        time.time()
        - inicio_total
    )

    # --------------------------------------------------------
    # Tabla comparativa
    # --------------------------------------------------------

    resultados_df = pd.DataFrame(
        resultados
    )

    # Promedio de mejora porcentual en ambos períodos.
    resultados_df[
        "delta_pct_promedio"
    ] = (
        resultados_df[
            "delta_pct_202105"
        ]
        + resultados_df[
            "delta_pct_202106"
        ]
    ) / 2.0

    # Peor resultado relativo entre ambos períodos.
    # Es una medida simple de robustez temporal.
    resultados_df[
        "delta_pct_peor_periodo"
    ] = resultados_df[
        [
            "delta_pct_202105",
            "delta_pct_202106",
        ]
    ].min(
        axis=1
    )

    resultados_df = (
        resultados_df
        .sort_values(
            [
                "delta_pct_peor_periodo",
                "delta_pct_promedio",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    output_csv = (
        args.output_dir
        / "comparacion_top_trials_z519.csv"
    )

    resultados_df.to_csv(
        output_csv,
        index=False,
    )

    # --------------------------------------------------------
    # Presentación
    # --------------------------------------------------------

    print("\n" + "=" * 100)
    print(
        "RESULTADO FINAL - ROBUSTEZ TEMPORAL"
    )
    print("=" * 100)

    print(
        "\nBaselines:"
    )

    print(
        f"  202105: "
        f"{fmt_millones(BASELINE_GAIN_202105)}"
    )

    print(
        f"  202106: "
        f"{fmt_millones(BASELINE_GAIN_202106)}"
    )

    columnas = [
        "trial",
        "gain_202105",
        "gain_202106",
        "delta_pct_202105",
        "delta_pct_202106",
        "delta_pct_promedio",
        "delta_pct_peor_periodo",
        "corte_202105",
        "corte_202106",
    ]

    mostrar = (
        resultados_df[
            columnas
        ]
        .copy()
    )

    mostrar[
        "gain_202105"
    ] = (
        mostrar[
            "gain_202105"
        ]
        / 1_000_000
    )

    mostrar[
        "gain_202106"
    ] = (
        mostrar[
            "gain_202106"
        ]
        / 1_000_000
    )

    print(
        "\nOrdenado por robustez "
        "(mejor peor-período):\n"
    )

    print(
        mostrar.to_string(
            index=False,
            formatters={
                "gain_202105":
                    lambda x: f"{x:.3f}",
                "gain_202106":
                    lambda x: f"{x:.3f}",
                "delta_pct_202105":
                    lambda x: f"{x:+.2f}",
                "delta_pct_202106":
                    lambda x: f"{x:+.2f}",
                "delta_pct_promedio":
                    lambda x: f"{x:+.2f}",
                "delta_pct_peor_periodo":
                    lambda x: f"{x:+.2f}",
            },
        )
    )

    print(
        f"\nTiempo total modelos: "
        f"{tiempo_total / 60:.1f} minutos"
    )

    print(
        f"\nResultado guardado en:\n"
        f"  {output_csv}"
    )

    # --------------------------------------------------------
    # Candidato robusto
    # --------------------------------------------------------

    mejor_robusto = (
        resultados_df.iloc[0]
    )

    print("\n" + "=" * 70)
    print("CANDIDATO MAS ROBUSTO")
    print("=" * 70)

    print(
        f"\nTrial: "
        f"{int(mejor_robusto['trial'])}"
    )

    print(
        f"Gain 202105: "
        f"{fmt_millones(mejor_robusto['gain_202105'])} "
        f"("
        f"{mejor_robusto['delta_pct_202105']:+.2f}%"
        f")"
    )

    print(
        f"Gain 202106: "
        f"{fmt_millones(mejor_robusto['gain_202106'])} "
        f"("
        f"{mejor_robusto['delta_pct_202106']:+.2f}%"
        f")"
    )

    print(
        f"Promedio delta: "
        f"{mejor_robusto['delta_pct_promedio']:+.2f}%"
    )

    print(
        f"Peor período: "
        f"{mejor_robusto['delta_pct_peor_periodo']:+.2f}%"
    )

    print("\nHiperparámetros:")

    print(
        f"  num_iterations: "
        f"{int(mejor_robusto['num_iterations'])}"
    )

    print(
        f"  learning_rate: "
        f"{mejor_robusto['learning_rate']}"
    )

    print(
        f"  feature_fraction: "
        f"{mejor_robusto['feature_fraction']}"
    )

    print(
        f"  num_leaves: "
        f"{int(mejor_robusto['num_leaves'])}"
    )

    print(
        f"  min_data_in_leaf: "
        f"{int(mejor_robusto['min_data_in_leaf'])}"
    )

    print(
        "\nIMPORTANTE:"
    )

    print(
        "Esta selección usa únicamente períodos "
        "con target conocido."
    )

    print(
        "Todavía no se utiliza 202108 ni se "
        "generan archivos para el bot."
    )


if __name__ == "__main__":
    main()