from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier


# ============================================================
# Configuración
# ============================================================

DATASET = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/"
    "competencia_01/feature_engineering_z523/"
    "competencia_01_historico_lag1.parquet"
)

OUTPUT_DIR = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/"
    "competencia_01/submits_z550_tarjetas_ponderado"
)

ID_COL = "numero_de_cliente"
MONTH_COL = "foto_mes"
TARGET_COL = "clase_ternaria"
LAG_AVAILABLE_COL = "lag1_disponible"

TRAIN_MONTHS = [202104, 202105, 202106]
SCORE_MONTH = 202108

PESOS_TEMPORALES = {
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

CUTS = [
    11500,
    12000,
    12500,
    13000,
]

N_PRINCIPAL = 12000

TARJETAS = [
    "ctarjeta_debito_transacciones",
    "ctarjeta_visa_transacciones",
    "ctarjeta_master_transacciones",
    "ccuenta_debitos_automaticos",
    "ctarjeta_visa_debitos_automaticos",
    "ctarjeta_master_debitos_automaticos",
]

FEATURE_TARJETAS = "actividad_tarjetas"


# ============================================================
# Funciones
# ============================================================

def identificar_features(df):
    excluir = {
        ID_COL,
        MONTH_COL,
        TARGET_COL,
        LAG_AVAILABLE_COL,
    }

    delta = [
        c for c in df.columns
        if c.endswith("_delta_lag1")
    ]

    lag = [
        c for c in df.columns
        if (
            c.endswith("_lag1")
            and not c.endswith("_delta_lag1")
        )
    ]

    originales = [
        c for c in df.columns
        if (
            c not in excluir
            and not c.endswith("_lag1")
            and not c.endswith("_delta_lag1")
        )
    ]

    return (
        originales
        + lag
        + delta
        + [LAG_AVAILABLE_COL]
    )


def target_binario(s):
    return s.eq("BAJA+2").astype(np.int8)


def params_modelo(seed):
    return {
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
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": -1,
        "importance_type": "gain",
    }


def validar_submit(path, score_ids, n):
    sub = pd.read_csv(
        path,
        header=None,
        names=[ID_COL],
    )

    if len(sub) != n:
        raise ValueError(
            f"{path.name}: "
            f"{len(sub)} filas != {n}"
        )

    if sub[ID_COL].isna().any():
        raise ValueError(
            f"{path.name}: IDs NA"
        )

    if sub[ID_COL].duplicated().any():
        raise ValueError(
            f"{path.name}: IDs duplicados"
        )

    universo = set(score_ids)

    if not set(sub[ID_COL]).issubset(universo):
        raise ValueError(
            f"{path.name}: IDs fuera de 202108"
        )

    print(
        f"VALIDADO {path.name}: "
        f"{len(sub):,} IDs"
    )


# ============================================================
# Inicio
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

t0 = time.time()

print("=" * 80)
print("Z550 - FINAL TARJETAS + PONDERACION + ENSEMBLE")
print("=" * 80)

print()
print("Leyendo dataset...")

df = pd.read_parquet(DATASET)

print(f"Filas: {len(df):,}")
print(f"Columnas iniciales: {len(df.columns):,}")


# ============================================================
# FULL457
# ============================================================

features_base = identificar_features(df)

if len(features_base) != 457:
    raise ValueError(
        f"Se esperaban 457 features base; "
        f"se encontraron {len(features_base)}"
    )

print(
    f"Features FULL457: {len(features_base)}"
)


# ============================================================
# Feature nueva
# ============================================================

faltantes = [
    c for c in TARJETAS
    if c not in df.columns
]

if faltantes:
    raise ValueError(
        f"Faltan variables de tarjetas: {faltantes}"
    )

df[FEATURE_TARJETAS] = (
    df[TARJETAS]
    .gt(0)
    .sum(axis=1)
    .astype("float32")
)

features = (
    features_base
    + [FEATURE_TARJETAS]
)

if len(features) != 458:
    raise ValueError(
        f"Se esperaban 458 features; "
        f"hay {len(features)}"
    )

print(
    f"Feature nueva: {FEATURE_TARJETAS}"
)

print(
    f"Componentes: {len(TARJETAS)}"
)

print(
    f"Media global: "
    f"{df[FEATURE_TARJETAS].mean():.4f}"
)

print(
    f"Rango: "
    f"{df[FEATURE_TARJETAS].min():.0f}"
    f" - "
    f"{df[FEATURE_TARJETAS].max():.0f}"
)

print(
    f"Features finales: {len(features)}"
)


# ============================================================
# Auditoría
# ============================================================

duplicados = (
    df[
        [ID_COL, MONTH_COL]
    ]
    .duplicated()
    .sum()
)

print()
print("AUDITORIA")
print(
    f"Duplicados cliente/mes: "
    f"{duplicados:,}"
)

if duplicados != 0:
    raise ValueError(
        "Hay duplicados cliente/mes"
    )

train = df[
    df[MONTH_COL].isin(
        TRAIN_MONTHS
    )
].copy()

score = df[
    df[MONTH_COL].eq(
        SCORE_MONTH
    )
].copy()

y_train = target_binario(
    train[TARGET_COL]
)

weights = (
    train[MONTH_COL]
    .map(PESOS_TEMPORALES)
    .astype(float)
    .to_numpy()
)

if not np.isfinite(weights).all():
    raise ValueError(
        "Hay sample weights inválidos"
    )

print(
    f"Train meses: {TRAIN_MONTHS}"
)

print(
    f"Train filas: {len(train):,}"
)

print(
    f"Train BAJA+2: "
    f"{int(y_train.sum()):,}"
)

print(
    f"Train negativos: "
    f"{len(y_train) - int(y_train.sum()):,}"
)

print(
    f"Score mes: {SCORE_MONTH}"
)

print(
    f"Score filas: {len(score):,}"
)

print(
    f"Score IDs únicos: "
    f"{score[ID_COL].nunique():,}"
)

print(
    f"Score target no-NA: "
    f"{score[TARGET_COL].notna().sum():,}"
)

if (
    score[ID_COL].nunique()
    != len(score)
):
    raise ValueError(
        "IDs duplicados en score"
    )

print()
print("PESOS TRAIN")

for mes in TRAIN_MONTHS:
    mask = train[MONTH_COL].eq(mes)

    print(
        f"{mes}: "
        f"n={mask.sum():,} "
        f"peso={PESOS_TEMPORALES[mes]:.2f}"
    )

print()
print("ACTIVIDAD TARJETAS POR MES")

for mes in TRAIN_MONTHS + [SCORE_MONTH]:
    x = df.loc[
        df[MONTH_COL].eq(mes),
        FEATURE_TARJETAS,
    ]

    print(
        f"{mes}: "
        f"mean={x.mean():.4f} "
        f"median={x.median():.1f} "
        f"zero={(x == 0).mean():.4%}"
    )


# ============================================================
# Matrices
# ============================================================

X_train = train[features]
X_score = score[features]

print()
print(
    f"X_train: {X_train.shape}"
)

print(
    f"X_score: {X_score.shape}"
)


# ============================================================
# Entrenamiento 5 seeds
# ============================================================

predicciones = []
importancias = []

for i, seed in enumerate(
    SEEDS,
    start=1,
):
    print()
    print("=" * 80)
    print(
        f"SEED {i}/{len(SEEDS)}: {seed}"
    )
    print("=" * 80)

    inicio = time.time()

    model = LGBMClassifier(
        **params_modelo(seed)
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=weights,
    )

    prob = model.predict_proba(
        X_score
    )[:, 1]

    predicciones.append(prob)

    imp = pd.DataFrame({
        "seed": seed,
        "feature": features,
        "importance_gain":
            model.feature_importances_,
    })

    imp[
        "rank_importance"
    ] = (
        imp["importance_gain"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    importancias.append(imp)

    fila_tarjetas = imp[
        imp["feature"].eq(
            FEATURE_TARJETAS
        )
    ].iloc[0]

    print(
        f"Tiempo: "
        f"{time.time() - inicio:.1f}s"
    )

    print(
        f"Prob mean="
        f"{prob.mean():.8f} "
        f"| max={prob.max():.8f}"
    )

    print(
        f"{FEATURE_TARJETAS}: "
        f"gain="
        f"{fila_tarjetas['importance_gain']:.2f} "
        f"| rank="
        f"{int(fila_tarjetas['rank_importance'])}"
    )


# ============================================================
# Ensemble
# ============================================================

pred_matrix = np.column_stack(
    predicciones
)

prob_ensemble = (
    pred_matrix.mean(axis=1)
)

print()
print("=" * 80)
print("ENSEMBLE")
print("=" * 80)

print(
    f"Mean prob: "
    f"{prob_ensemble.mean():.8f}"
)

print(
    f"Max prob: "
    f"{prob_ensemble.max():.8f}"
)


# ============================================================
# Estabilidad top-N entre seeds
# ============================================================

def top_ids(prob, n):
    idx = np.argsort(-prob)[:n]

    return set(
        score.iloc[idx][ID_COL]
        .tolist()
    )


for n in CUTS:
    tops = [
        top_ids(prob, n)
        for prob in predicciones
    ]

    jaccards = []

    for i in range(len(tops)):
        for j in range(
            i + 1,
            len(tops),
        ):
            inter = len(
                tops[i] & tops[j]
            )

            union = len(
                tops[i] | tops[j]
            )

            jaccards.append(
                inter / union
            )

    print(
        f"Top {n:,} seeds: "
        f"Jaccard mean="
        f"{np.mean(jaccards):.6f} "
        f"| min="
        f"{np.min(jaccards):.6f} "
        f"| max="
        f"{np.max(jaccards):.6f}"
    )


# ============================================================
# Ranking final
# ============================================================

ranking = pd.DataFrame({
    ID_COL:
        score[ID_COL].to_numpy(),
    "prob":
        prob_ensemble,
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

ranking[
    "ranking"
] = np.arange(
    1,
    len(ranking) + 1,
)

ranking_path = (
    OUTPUT_DIR
    / "ranking_tarjetas_ponderado_ensemble5.csv"
)

ranking.to_csv(
    ranking_path,
    index=False,
)

print()
print(
    f"Ranking guardado: "
    f"{ranking_path}"
)


# ============================================================
# Rankings individuales
# ============================================================

for seed, prob in zip(
    SEEDS,
    predicciones,
):
    r = pd.DataFrame({
        ID_COL:
            score[ID_COL].to_numpy(),
        "prob":
            prob,
    })

    r = (
        r
        .sort_values(
            "prob",
            ascending=False,
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    r[
        "ranking"
    ] = np.arange(
        1,
        len(r) + 1,
    )

    r.to_csv(
        OUTPUT_DIR
        / (
            f"ranking_tarjetas_"
            f"seed{seed}.csv"
        ),
        index=False,
    )


# ============================================================
# Submits ensemble
# ============================================================

print()
print("=" * 80)
print("GENERANDO SUBMITS")
print("=" * 80)

score_ids = (
    score[ID_COL]
    .tolist()
)

for n in CUTS:
    path = (
        OUTPUT_DIR
        / (
            "submit_tarjetas_ponderado_"
            f"ensemble5_n{n}.csv"
        )
    )

    ranking[
        [ID_COL]
    ].head(n).to_csv(
        path,
        index=False,
        header=False,
    )

    validar_submit(
        path,
        score_ids,
        n,
    )


# ============================================================
# Submit canonical seed 290497
# Sólo para diagnóstico/control, NO es el principal
# ============================================================

prob_canonical = (
    predicciones[
        SEEDS.index(290497)
    ]
)

ranking_canonical = pd.DataFrame({
    ID_COL:
        score[ID_COL].to_numpy(),
    "prob":
        prob_canonical,
})

ranking_canonical = (
    ranking_canonical
    .sort_values(
        "prob",
        ascending=False,
        kind="mergesort",
    )
    .reset_index(drop=True)
)

ranking_canonical[
    "ranking"
] = np.arange(
    1,
    len(ranking_canonical) + 1,
)

ranking_canonical.to_csv(
    OUTPUT_DIR
    / "ranking_tarjetas_seed290497.csv",
    index=False,
)

for n in CUTS:
    path = (
        OUTPUT_DIR
        / (
            "submit_tarjetas_ponderado_"
            f"seed290497_n{n}.csv"
        )
    )

    ranking_canonical[
        [ID_COL]
    ].head(n).to_csv(
        path,
        index=False,
        header=False,
    )

    validar_submit(
        path,
        score_ids,
        n,
    )


# ============================================================
# Comparación ensemble vs canonical
# ============================================================

for n in CUTS:
    a = set(
        ranking.head(n)[ID_COL]
    )

    b = set(
        ranking_canonical
        .head(n)[ID_COL]
    )

    inter = len(a & b)
    union = len(a | b)

    print(
        f"N={n:,}: "
        f"ensemble vs canonical "
        f"inter={inter:,} "
        f"| Jaccard="
        f"{inter / union:.6f}"
    )


# ============================================================
# Importancias
# ============================================================

importancias = pd.concat(
    importancias,
    ignore_index=True,
)

importancias.to_csv(
    OUTPUT_DIR
    / "importancias_5seeds.csv",
    index=False,
)

imp_tarjetas = (
    importancias[
        importancias[
            "feature"
        ].eq(
            FEATURE_TARJETAS
        )
    ]
    .copy()
)

imp_tarjetas.to_csv(
    OUTPUT_DIR
    / "importancia_actividad_tarjetas_5seeds.csv",
    index=False,
)

print()
print("IMPORTANCIA ACTIVIDAD TARJETAS")

print(
    imp_tarjetas[
        [
            "seed",
            "importance_gain",
            "rank_importance",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# Predicciones para reproducibilidad
# ============================================================

pred_df = pd.DataFrame({
    ID_COL:
        score[ID_COL].to_numpy(),
})

for seed, prob in zip(
    SEEDS,
    predicciones,
):
    pred_df[
        f"prob_seed_{seed}"
    ] = prob

pred_df[
    "prob_ensemble"
] = prob_ensemble

pred_df.to_csv(
    OUTPUT_DIR
    / "predicciones_202108.csv",
    index=False,
)


# ============================================================
# Metadata
# ============================================================

metadata = {
    "dataset":
        str(DATASET),
    "train_months":
        TRAIN_MONTHS,
    "score_month":
        SCORE_MONTH,
    "pesos_temporales":
        PESOS_TEMPORALES,
    "seeds":
        SEEDS,
    "cuts":
        CUTS,
    "n_principal":
        N_PRINCIPAL,
    "n_features_base":
        len(features_base),
    "n_features_final":
        len(features),
    "feature_nueva":
        FEATURE_TARJETAS,
    "componentes_tarjetas":
        TARJETAS,
    "runtime_minutos":
        (time.time() - t0)
        / 60.0,
}

with open(
    OUTPUT_DIR
    / "metadata_z550.json",
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
    f"{(time.time() - t0) / 60:.2f} min"
)

print()
print("Z550 FINALIZADO")
print("Output:", OUTPUT_DIR)

print()
print("SUBMIT PRINCIPAL A PROBAR:")
print(
    OUTPUT_DIR
    / (
        "submit_tarjetas_ponderado_"
        f"ensemble5_n{N_PRINCIPAL}.csv"
    )
)
