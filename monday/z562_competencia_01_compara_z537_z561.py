from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ID = "numero_de_cliente"
MONTH = "foto_mes"

Z537 = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/"
    "competencia_01/submits_z537_ponderado/"
    "ranking_historico_ensemble5.csv"
)

Z561 = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/"
    "competencia_01/submits_z561_rf_crossfit/"
    "ranking_rf_crossfit_ensemble5.csv"
)

DATASET = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/"
    "competencia_01/feature_engineering_z523/"
    "competencia_01_historico_lag1.parquet"
)

OUTPUT = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/"
    "competencia_01/comparacion_z537_z561_z562"
)

CUTS = [
    8000,
    10000,
    11000,
    11500,
    12000,
    12500,
    13000,
    14000,
    15000,
]


OUTPUT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD RANKINGS
# ============================================================

print("=" * 100)
print("Z562 - COMPARACION Z537 vs Z561")
print("=" * 100)

a = pd.read_csv(Z537)
b = pd.read_csv(Z561)

print()
print("Z537:", a.shape)
print("Z561:", b.shape)

print()
print("Columnas Z537:")
print(a.columns.tolist())

print()
print("Columnas Z561:")
print(b.columns.tolist())


# ============================================================
# DETECTAR PROB Z537
# ============================================================

candidate_cols = [
    c for c in a.columns
    if c not in {
        ID,
        MONTH,
    }
    and (
        "prob" in c.lower()
        or "pred" in c.lower()
        or "ensemble" in c.lower()
    )
]

numeric_candidates = [
    c for c in candidate_cols
    if pd.api.types.is_numeric_dtype(
        a[c]
    )
]

if not numeric_candidates:
    raise ValueError(
        "No pude detectar columna de "
        "probabilidad en Z537."
    )

print()
print(
    "Candidatas prob Z537:",
    numeric_candidates,
)

# Preferimos una columna que mencione ensemble.
ensemble_candidates = [
    c for c in numeric_candidates
    if "ensemble" in c.lower()
]

if ensemble_candidates:
    prob537 = ensemble_candidates[0]
else:
    prob537 = numeric_candidates[0]

prob561 = "prob_rf_crossfit"

print()
print("Prob Z537 elegida:", prob537)
print("Prob Z561:", prob561)


# ============================================================
# MERGE
# ============================================================

cols_a = [
    ID,
    prob537,
]

if MONTH in a.columns:
    cols_a.append(MONTH)

aa = a[
    cols_a
].copy()

aa = aa.rename(
    columns={
        prob537:
            "prob_z537",
    }
)

bb = b[
    [
        ID,
        MONTH,
        prob561,
        "rf_score_aux",
    ]
].copy()

bb = bb.rename(
    columns={
        prob561:
            "prob_z561",
    }
)

if MONTH in aa.columns:

    merged = aa.merge(
        bb,
        on=[
            ID,
            MONTH,
        ],
        how="inner",
        validate="one_to_one",
    )

else:

    merged = aa.merge(
        bb,
        on=ID,
        how="inner",
        validate="one_to_one",
    )

print()
print(
    f"Clientes comparados: "
    f"{len(merged):,}"
)

if len(merged) != len(bb):
    print(
        "WARNING: merge no contiene "
        "todo Z561."
    )


# ============================================================
# CORRELACIONES
# ============================================================

pearson = np.corrcoef(
    merged["prob_z537"],
    merged["prob_z561"],
)[0, 1]

spearman = spearmanr(
    merged["prob_z537"],
    merged["prob_z561"],
).statistic

print()
print("=" * 100)
print("CORRELACION")
print("=" * 100)

print(
    f"Pearson probabilidades: "
    f"{pearson:.6f}"
)

print(
    f"Spearman ranking:       "
    f"{spearman:.6f}"
)


# ============================================================
# RANKS
# ============================================================

merged["rank_z537"] = (
    merged["prob_z537"]
    .rank(
        ascending=False,
        method="first",
    )
    .astype(int)
)

merged["rank_z561"] = (
    merged["prob_z561"]
    .rank(
        ascending=False,
        method="first",
    )
    .astype(int)
)

merged["delta_rank"] = (
    merged["rank_z537"]
    - merged["rank_z561"]
)

# positivo = Z561 lo promovió


# ============================================================
# OVERLAP POR CUT
# ============================================================

overlap_rows = []

print()
print("=" * 100)
print("OVERLAP TOP-N")
print("=" * 100)

for n in CUTS:

    s537 = set(
        merged.loc[
            merged["rank_z537"] <= n,
            ID,
        ]
    )

    s561 = set(
        merged.loc[
            merged["rank_z561"] <= n,
            ID,
        ]
    )

    ambos = len(
        s537 & s561
    )

    solo537 = len(
        s537 - s561
    )

    solo561 = len(
        s561 - s537
    )

    union = len(
        s537 | s561
    )

    jaccard = (
        ambos / union
    )

    pct_reemplazo = (
        solo561 / n
    )

    overlap_rows.append({
        "cut":
            n,
        "ambos":
            ambos,
        "solo_z537":
            solo537,
        "solo_z561":
            solo561,
        "jaccard":
            jaccard,
        "pct_reemplazo":
            pct_reemplazo,
    })

    print(
        f"N={n:5d} "
        f"ambos={ambos:5d} "
        f"solo537={solo537:4d} "
        f"solo561={solo561:4d} "
        f"Jaccard={jaccard:.6f} "
        f"reemplazo="
        f"{pct_reemplazo:.2%}"
    )


overlap_df = pd.DataFrame(
    overlap_rows
)

overlap_df.to_csv(
    OUTPUT
    / "overlap_z537_z561.csv",
    index=False,
)


# ============================================================
# ANATOMIA N=12000
# ============================================================

N = 12000

merged["grupo_n12000"] = "NINGUNO"

mask537 = (
    merged["rank_z537"]
    <= N
)

mask561 = (
    merged["rank_z561"]
    <= N
)

merged.loc[
    mask537 & mask561,
    "grupo_n12000",
] = "AMBOS"

merged.loc[
    mask537 & ~mask561,
    "grupo_n12000",
] = "SOLO_Z537"

merged.loc[
    ~mask537 & mask561,
    "grupo_n12000",
] = "SOLO_Z561"


print()
print("=" * 100)
print("ANATOMIA N=12000")
print("=" * 100)

print(
    merged[
        "grupo_n12000"
    ].value_counts()
)


# ============================================================
# VARIABLES AGOSTO PARA INTERCAMBIOS
# ============================================================

aug = pd.read_parquet(
    DATASET,
    filters=[
        (MONTH, "==", 202108)
    ],
)

exclude = {
    ID,
    MONTH,
    "clase_ternaria",
}

original_cols = [
    c for c in aug.columns
    if (
        c not in exclude
        and c != "lag1_disponible"
        and not c.endswith("_lag1")
        and not c.endswith(
            "_delta_lag1"
        )
    )
]

aug = aug[
    [ID] + original_cols
]

merged = merged.merge(
    aug,
    on=ID,
    how="left",
    validate="one_to_one",
)


# ============================================================
# RESUMEN RF SCORE / RANKS
# ============================================================

interesting_groups = [
    "AMBOS",
    "SOLO_Z537",
    "SOLO_Z561",
]

summary_rows = []

for group in interesting_groups:

    x = merged[
        merged[
            "grupo_n12000"
        ].eq(group)
    ]

    summary_rows.append({
        "grupo":
            group,
        "n":
            len(x),
        "prob_z537_mean":
            x[
                "prob_z537"
            ].mean(),
        "prob_z561_mean":
            x[
                "prob_z561"
            ].mean(),
        "rf_score_aux_mean":
            x[
                "rf_score_aux"
            ].mean(),
        "rank_z537_mean":
            x[
                "rank_z537"
            ].mean(),
        "rank_z561_mean":
            x[
                "rank_z561"
            ].mean(),
        "delta_rank_mean":
            x[
                "delta_rank"
            ].mean(),
    })


summary_df = pd.DataFrame(
    summary_rows
)

print()
print("Resumen grupos:")
print(
    summary_df.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}",
    )
)

summary_df.to_csv(
    OUTPUT
    / "resumen_grupos_n12000.csv",
    index=False,
)


# ============================================================
# DIFERENCIAS DE VARIABLES:
# SOLO_Z561 vs SOLO_Z537
# ============================================================

g537 = merged[
    merged[
        "grupo_n12000"
    ].eq("SOLO_Z537")
]

g561 = merged[
    merged[
        "grupo_n12000"
    ].eq("SOLO_Z561")
]

diff_rows = []

for col in original_cols:

    if not pd.api.types.is_numeric_dtype(
        merged[col]
    ):
        continue

    mean537 = g537[
        col
    ].mean()

    mean561 = g561[
        col
    ].mean()

    std_all = merged[
        col
    ].std()

    if (
        pd.isna(std_all)
        or std_all == 0
    ):
        standardized_diff = np.nan
    else:
        standardized_diff = (
            mean561 - mean537
        ) / std_all

    diff_rows.append({
        "feature":
            col,
        "mean_solo_z537":
            mean537,
        "mean_solo_z561":
            mean561,
        "diff":
            mean561 - mean537,
        "standardized_diff":
            standardized_diff,
        "abs_standardized_diff":
            (
                abs(
                    standardized_diff
                )
                if not pd.isna(
                    standardized_diff
                )
                else np.nan
            ),
    })


diff_df = (
    pd.DataFrame(
        diff_rows
    )
    .sort_values(
        "abs_standardized_diff",
        ascending=False,
    )
)

print()
print("=" * 100)
print(
    "TOP 25 DIFERENCIAS "
    "SOLO_Z561 vs SOLO_Z537"
)
print("=" * 100)

print(
    diff_df.head(25)[
        [
            "feature",
            "mean_solo_z537",
            "mean_solo_z561",
            "standardized_diff",
        ]
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}",
    )
)

diff_df.to_csv(
    OUTPUT
    / "diferencias_variables_n12000.csv",
    index=False,
)


# ============================================================
# GUARDAR ANATOMIA
# ============================================================

keep_cols = [
    ID,
    "prob_z537",
    "prob_z561",
    "rf_score_aux",
    "rank_z537",
    "rank_z561",
    "delta_rank",
    "grupo_n12000",
]

merged[
    keep_cols
].sort_values(
    "rank_z561"
).to_csv(
    OUTPUT
    / "anatomia_clientes_n12000.csv",
    index=False,
)


print()
print("Z562 FINALIZADO")
print("Output:", OUTPUT)
