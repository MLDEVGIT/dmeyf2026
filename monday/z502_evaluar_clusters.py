"""
Evaluación exploratoria del número de clusters para el ejercicio de Miranda.

Reutiliza la representación cliente × atributos construida en
z501_cluster_rf.py.

Experimentos
------------
A) Clusterizar solamente clientes BAJA+2.
B) Clusterizar BAJA+2 + fieles de la muestra balanceada.

Para cada k calcula:
- inercia (WCSS)
- Calinski-Harabasz
- Davies-Bouldin
- tamaño mínimo de cluster
- tamaño máximo de cluster
- porcentaje representado por el cluster más pequeño
- estabilidad de KMeans mediante Adjusted Rand Index (ARI)

En el experimento B también informa la composición BAJA+2/fieles
de cada cluster.

IMPORTANTE
----------
La muestra del experimento B está balanceada artificialmente entre
BAJA+2 y fieles. Por lo tanto, share_baja2 NO debe interpretarse
como probabilidad de baja.

La estabilidad mediante ARI mantiene fija la representación P.
Por lo tanto, mide la sensibilidad de KMeans a su inicialización,
no la estabilidad del pipeline completo.
"""

from pathlib import Path
import sys

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)


# ============================================================
# Importar funciones del ejercicio z501 de la cátedra
# ============================================================

# Permite importar z501 cuando ejecutamos el script
# desde la raíz del repositorio.
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

SEED = 214363

N_TREES = 300
MIN_SAMPLES_LEAF = 50

# Cantidades de clusters a explorar.
K_VALUES = range(2, 11)

# Semillas utilizadas exclusivamente para evaluar la estabilidad
# de KMeans manteniendo fija la representación P.
STABILITY_SEEDS = [
    101,
    211,
    307,
    401,
    503,
    601,
    701,
    809,
    907,
    1009,
    1103,
    1201,
    1301,
    1409,
    1511,
    1601,
    1709,
    1801,
    1901,
    2003,
]


# ============================================================
# Evaluación de una solución KMeans
# ============================================================

def evaluar_k(
    P: pd.DataFrame,
    k: int,
    seed: int,
) -> tuple[dict, pd.Series]:
    """
    Ejecuta KMeans para un determinado k y devuelve:

    1. métricas generales de la solución;
    2. cluster asignado a cada cliente.

    Estas métricas no se utilizan como criterio automático
    para elegir k. Son evidencia que después se combina con
    estabilidad, tamaños e interpretabilidad.
    """

    km = KMeans(
        n_clusters=k,
        n_init=20,
        random_state=seed,
    )

    labels = pd.Series(
        km.fit_predict(P.values),
        index=P.index,
        name="cluster",
    )

    tamanios = labels.value_counts()

    metricas = {
        "k": k,
        "inercia": km.inertia_,
        "calinski_harabasz":
            calinski_harabasz_score(
                P.values,
                labels.values,
            ),
        "davies_bouldin":
            davies_bouldin_score(
                P.values,
                labels.values,
            ),
        "cluster_min":
            int(tamanios.min()),
        "cluster_max":
            int(tamanios.max()),
        "cluster_min_pct":
            100 * tamanios.min() / len(labels),
    }

    return metricas, labels


# ============================================================
# Estabilidad de KMeans
# ============================================================

def evaluar_estabilidad(
    P: pd.DataFrame,
    k: int,
    seeds: list[int],
) -> dict:
    """
    Evalúa la estabilidad de KMeans frente a distintas semillas.

    La matriz P permanece fija.

    Por lo tanto, esta prueba mide únicamente cuánto cambia la
    partición debido a la inicialización de KMeans.

    NO mide todavía la estabilidad del pipeline completo:

        muestra de fieles
            -> Random Forest
            -> representación P
            -> KMeans

    Para comparar las particiones se utiliza Adjusted Rand Index
    (ARI).

    ARI = 1:
        las dos particiones son idénticas, independientemente
        del número utilizado para nombrar cada cluster.

    ARI cercano a 0:
        concordancia aproximadamente compatible con azar.

    Se comparan todos los pares posibles de soluciones.
    Con 20 semillas existen 190 pares.
    """

    soluciones = []

    for seed in seeds:

        km = KMeans(
            n_clusters=k,
            n_init=20,
            random_state=seed,
        )

        labels = km.fit_predict(
            P.values
        )

        soluciones.append(labels)

    aris = []

    for i in range(len(soluciones)):

        for j in range(
            i + 1,
            len(soluciones),
        ):

            ari = adjusted_rand_score(
                soluciones[i],
                soluciones[j],
            )

            aris.append(ari)

    s = pd.Series(
        aris,
        dtype="float64",
    )

    return {
        "ari_mean": s.mean(),
        "ari_min": s.min(),
        "ari_max": s.max(),
        "ari_std": s.std(),
        "ari_pairs": len(s),
    }


# ============================================================
# Formato de tablas
# ============================================================

FORMATTERS_METRICAS = {
    "inercia":
        "{:.4f}".format,
    "calinski_harabasz":
        "{:.2f}".format,
    "davies_bouldin":
        "{:.4f}".format,
    "cluster_min_pct":
        "{:.2f}".format,
    "ari_mean":
        "{:.4f}".format,
    "ari_min":
        "{:.4f}".format,
    "ari_max":
        "{:.4f}".format,
    "ari_std":
        "{:.4f}".format,
}


# ============================================================
# Programa principal
# ============================================================

def main() -> None:

    # --------------------------------------------------------
    # 1. Cargar la misma muestra que utiliza z501
    # --------------------------------------------------------

    print(
        "=== CARGA Y REPRESENTACIÓN DEL BOSQUE ==="
    )

    if not CSV.exists():
        raise SystemExit(
            f"No encuentro el dataset: {CSV}"
        )

    df = z501.cargar_muestra(
        CSV,
        SEED,
    )

    feats = z501.columnas_features(
        df
    )

    print(
        f"Dataset            : {CSV}"
    )

    print(
        f"Filas de la muestra: {len(df):,}"
    )

    print(
        f"Features numéricas : {len(feats):,}"
    )

    print(
        f"Clientes únicos    : "
        f"{df[z501.ID_COL].nunique():,}"
    )

    # --------------------------------------------------------
    # 2. Random Forest
    # --------------------------------------------------------

    X = df[feats].to_numpy(
        dtype="float32"
    )

    y = df[
        z501.GRUPO_COL
    ].to_numpy()

    ids = df[
        z501.ID_COL
    ].to_numpy()

    print()

    print(
        f"Entrenando Random Forest "
        f"({N_TREES} árboles, "
        f"min_samples_leaf="
        f"{MIN_SAMPLES_LEAF})..."
    )

    rf = z501.entrenar_rf(
        X,
        y,
        N_TREES,
        MIN_SAMPLES_LEAF,
        SEED,
    )

    print(
        f"OOB accuracy: "
        f"{rf.oob_score_:.4f}"
    )

    # --------------------------------------------------------
    # 3. Construir matriz cliente × atributos
    # --------------------------------------------------------

    print()

    print(
        "Construyendo matriz "
        "cliente × atributos..."
    )

    C = z501.contar_hojas_por_id(
        rf,
        X,
        ids,
        feats,
    )

    P = z501.normalizar(C)

    print(
        f"Matriz P completa: {P.shape}"
    )

    # --------------------------------------------------------
    # 4. Determinar grupo de cada cliente
    # --------------------------------------------------------

    # grupo = 1 -> cliente BAJA+2
    # grupo = 0 -> cliente fiel

    grupo_por_id = (
        df.groupby(
            z501.ID_COL
        )[z501.GRUPO_COL]
        .max()
        .reindex(P.index)
    )

    n_baja2 = int(
        (grupo_por_id == 1).sum()
    )

    n_fieles = int(
        (grupo_por_id == 0).sum()
    )

    print(
        f"Clientes BAJA+2    : "
        f"{n_baja2:,}"
    )

    print(
        f"Clientes fieles    : "
        f"{n_fieles:,}"
    )

    # Matriz únicamente para BAJA+2.

    ids_baja2 = grupo_por_id[
        grupo_por_id == 1
    ].index

    P_baja2 = P.loc[
        ids_baja2
    ]

    print(
        f"Matriz P BAJA+2   : "
        f"{P_baja2.shape}"
    )

    print()

    print(
        f"Estabilidad KMeans: "
        f"{len(STABILITY_SEEDS)} semillas"
    )

    n_pares = (
        len(STABILITY_SEEDS)
        * (len(STABILITY_SEEDS) - 1)
        // 2
    )

    print(
        f"Comparaciones ARI por k: "
        f"{n_pares}"
    )

    # ========================================================
    # EXPERIMENTO A
    # Clusterizar solamente BAJA+2
    # ========================================================

    print()

    print("=" * 70)

    print(
        "EXPERIMENTO A: "
        "SOLO CLIENTES BAJA+2"
    )

    print("=" * 70)

    resultados_a = []

    for k in K_VALUES:

        print(
            f"Evaluando k={k}..."
        )

        metricas, _ = evaluar_k(
            P_baja2,
            k,
            SEED,
        )

        estabilidad = (
            evaluar_estabilidad(
                P_baja2,
                k,
                STABILITY_SEEDS,
            )
        )

        metricas.update(
            estabilidad
        )

        resultados_a.append(
            metricas
        )

    tabla_a = pd.DataFrame(
        resultados_a
    )

    print()

    print(
        "=== MÉTRICAS — "
        "SOLO BAJA+2 ==="
    )

    print()

    print(
        tabla_a.to_string(
            index=False,
            formatters=FORMATTERS_METRICAS,
        )
    )

    # ========================================================
    # EXPERIMENTO B
    # Clusterizar BAJA+2 + fieles
    # ========================================================

    print()

    print("=" * 70)

    print(
        "EXPERIMENTO B: "
        "BAJA+2 + FIELES"
    )

    print("=" * 70)

    resultados_b = []
    composiciones = []

    for k in K_VALUES:

        print(
            f"Evaluando k={k}..."
        )

        metricas, labels = evaluar_k(
            P,
            k,
            SEED,
        )

        estabilidad = (
            evaluar_estabilidad(
                P,
                k,
                STABILITY_SEEDS,
            )
        )

        metricas.update(
            estabilidad
        )

        resultados_b.append(
            metricas
        )

        # ----------------------------------------------------
        # Composición BAJA+2 / fieles por cluster
        # ----------------------------------------------------

        tmp = pd.DataFrame(
            {
                "cluster": labels,
                "grupo": grupo_por_id,
            }
        )

        comp = (
            tmp.groupby(
                "cluster"
            )["grupo"]
            .agg(
                n_total="size",
                n_baja2="sum",
                share_baja2="mean",
            )
            .reset_index()
        )

        comp[
            "n_baja2"
        ] = comp[
            "n_baja2"
        ].astype(int)

        comp[
            "n_fieles"
        ] = (
            comp["n_total"]
            - comp["n_baja2"]
        )

        comp[
            "share_baja2"
        ] *= 100

        # Agregar k como primera columna.

        comp.insert(
            0,
            "k",
            k,
        )

        composiciones.append(
            comp
        )

    tabla_b = pd.DataFrame(
        resultados_b
    )

    composicion_b = pd.concat(
        composiciones,
        ignore_index=True,
    )

    print()

    print(
        "=== MÉTRICAS — "
        "BAJA+2 + FIELES ==="
    )

    print()

    print(
        tabla_b.to_string(
            index=False,
            formatters=FORMATTERS_METRICAS,
        )
    )

    print()

    print(
        "=== COMPOSICIÓN DE LOS CLUSTERS "
        "— EXPERIMENTO B ==="
    )

    print()

    print(
        composicion_b.to_string(
            index=False,
            formatters={
                "share_baja2":
                    "{:.2f}".format,
            },
        )
    )

    print()

    print(
        "NOTA: share_baja2 describe "
        "la composición de esta "
        "muestra balanceada."
    )

    print(
        "NO debe interpretarse como "
        "probabilidad real de baja."
    )

    # ========================================================
    # Guardar resultados
    # ========================================================

    out = Path(
        "/data/dmeyf/datasets/"
        "evaluacion_clusters"
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    archivo_a = (
        out
        / "metricas_solo_baja2.csv"
    )

    archivo_b = (
        out
        / "metricas_baja2_fieles.csv"
    )

    archivo_comp = (
        out
        / "composicion_baja2_fieles.csv"
    )

    tabla_a.to_csv(
        archivo_a,
        index=False,
    )

    tabla_b.to_csv(
        archivo_b,
        index=False,
    )

    composicion_b.to_csv(
        archivo_comp,
        index=False,
    )

    print()

    print("=" * 70)

    print(
        "RESULTADOS GUARDADOS"
    )

    print("=" * 70)

    print(
        archivo_a
    )

    print(
        archivo_b
    )

    print(
        archivo_comp
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()