"""
Análisis de transición entre soluciones KMeans para Miranda.

Objetivo
--------
Estudiar cómo se subdividen los clusters al pasar:

    k=2 -> k=3 -> k=4 -> k=5

sobre el Experimento B de z502:

    BAJA+2 + fieles

Se reutiliza exactamente:
- muestra balanceada de z501
- Random Forest de z501
- representación cliente × atributos P
- semilla 214363
- KMeans con n_init=20

IMPORTANTE
----------
share_baja2 describe la composición de la muestra balanceada.
NO representa probabilidad real de baja.
"""

from pathlib import Path
import sys

import pandas as pd
from sklearn.cluster import KMeans


# ============================================================
# Importar z501
# ============================================================

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent),
)

import z501_cluster_rf as z501


# ============================================================
# Configuración
# ============================================================

CSV = Path(
    "/data/dmeyf/datasets/competencia_01.csv"
)

OUT = Path(
    "/data/dmeyf/datasets/evaluacion_clusters"
)

SEED = 214363

N_TREES = 300
MIN_SAMPLES_LEAF = 50

K_VALUES = [2, 3, 4, 5]


# ============================================================
# Clustering
# ============================================================

def clusterizar(
    P: pd.DataFrame,
    k: int,
) -> pd.Series:

    km = KMeans(
        n_clusters=k,
        n_init=20,
        random_state=SEED,
    )

    labels = km.fit_predict(
        P.values
    )

    return pd.Series(
        labels,
        index=P.index,
        name=f"k{k}",
    )


# ============================================================
# Composición de una solución
# ============================================================

def composicion(
    labels: pd.Series,
    grupo_por_id: pd.Series,
    k: int,
) -> pd.DataFrame:

    tmp = pd.DataFrame(
        {
            "cluster": labels,
            "grupo": grupo_por_id,
        }
    )

    tabla = (
        tmp.groupby("cluster")["grupo"]
        .agg(
            n_total="size",
            n_baja2="sum",
            share_baja2="mean",
        )
        .reset_index()
    )

    tabla["n_baja2"] = (
        tabla["n_baja2"]
        .astype(int)
    )

    tabla["n_fieles"] = (
        tabla["n_total"]
        - tabla["n_baja2"]
    )

    tabla["share_baja2"] *= 100

    tabla.insert(
        0,
        "k",
        k,
    )

    return tabla


# ============================================================
# Transición k -> k+1
# ============================================================

def transicion(
    labels_origen: pd.Series,
    labels_destino: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    # Conteos absolutos.
    conteos = pd.crosstab(
        labels_origen,
        labels_destino,
    )

    # Porcentaje de cada cluster de origen que termina
    # en cada cluster de destino.
    porcentajes = pd.crosstab(
        labels_origen,
        labels_destino,
        normalize="index",
    ) * 100

    return conteos, porcentajes


# ============================================================
# Main
# ============================================================

def main() -> None:

    print(
        "=== CONSTRUCCIÓN DE LA REPRESENTACIÓN P ==="
    )

    if not CSV.exists():
        raise SystemExit(
            f"No encuentro {CSV}"
        )

    df = z501.cargar_muestra(
        CSV,
        SEED,
    )

    feats = z501.columnas_features(
        df
    )

    X = df[feats].to_numpy(
        dtype="float32"
    )

    y = df[
        z501.GRUPO_COL
    ].to_numpy()

    ids = df[
        z501.ID_COL
    ].to_numpy()

    print(
        f"Muestra: {len(df):,} filas"
    )

    print(
        f"Features: {len(feats)}"
    )

    print(
        "Entrenando Random Forest..."
    )

    rf = z501.entrenar_rf(
        X,
        y,
        N_TREES,
        MIN_SAMPLES_LEAF,
        SEED,
    )

    print(
        f"OOB accuracy: {rf.oob_score_:.4f}"
    )

    print(
        "Construyendo P..."
    )

    C = z501.contar_hojas_por_id(
        rf,
        X,
        ids,
        feats,
    )

    P = z501.normalizar(C)

    grupo_por_id = (
        df.groupby(
            z501.ID_COL
        )[z501.GRUPO_COL]
        .max()
        .reindex(P.index)
    )

    print(
        f"P: {P.shape}"
    )

    # --------------------------------------------------------
    # Calcular soluciones
    # --------------------------------------------------------

    soluciones = {}

    composiciones = []

    for k in K_VALUES:

        print(
            f"Clusterizando k={k}..."
        )

        labels = clusterizar(
            P,
            k,
        )

        soluciones[k] = labels

        comp = composicion(
            labels,
            grupo_por_id,
            k,
        )

        composiciones.append(
            comp
        )

    composicion_total = pd.concat(
        composiciones,
        ignore_index=True,
    )

    print()
    print("=" * 70)
    print("COMPOSICIÓN BAJA+2 / FIELES")
    print("=" * 70)
    print()

    print(
        composicion_total.to_string(
            index=False,
            formatters={
                "share_baja2":
                    "{:.2f}".format,
            },
        )
    )

    # --------------------------------------------------------
    # Transiciones
    # --------------------------------------------------------

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    composicion_total.to_csv(
        OUT / "transiciones_composicion.csv",
        index=False,
    )

    for k_origen, k_destino in zip(
        K_VALUES[:-1],
        K_VALUES[1:],
    ):

        print()
        print("=" * 70)

        print(
            f"TRANSICIÓN "
            f"k={k_origen} -> k={k_destino}"
        )

        print("=" * 70)

        conteos, porcentajes = transicion(
            soluciones[k_origen],
            soluciones[k_destino],
        )

        print()
        print("CONTEOS")
        print()

        print(
            conteos.to_string()
        )

        print()
        print(
            "% DE CADA CLUSTER DE ORIGEN"
        )
        print()

        print(
            porcentajes.to_string(
                float_format=lambda x: f"{x:.2f}"
            )
        )

        conteos.to_csv(
            OUT
            / (
                f"transicion_"
                f"k{k_origen}_k{k_destino}_"
                f"conteos.csv"
            )
        )

        porcentajes.to_csv(
            OUT
            / (
                f"transicion_"
                f"k{k_origen}_k{k_destino}_"
                f"porcentajes.csv"
            )
        )

    # --------------------------------------------------------
    # Matriz completa de asignaciones
    # --------------------------------------------------------

    asignaciones = pd.DataFrame(
        {
            f"k{k}": soluciones[k]
            for k in K_VALUES
        }
    )

    asignaciones[
        "grupo"
    ] = grupo_por_id

    asignaciones.to_csv(
        OUT / "asignaciones_k2_k5.csv"
    )

    print()
    print("=" * 70)
    print("RESULTADOS GUARDADOS")
    print("=" * 70)

    print(
        OUT / "transiciones_composicion.csv"
    )

    print(
        OUT / "asignaciones_k2_k5.csv"
    )

    print()
    print(
        "NOTA: share_baja2 corresponde "
        "a la muestra balanceada."
    )

    print(
        "NO representa probabilidad "
        "real de baja."
    )


if __name__ == "__main__":
    main()