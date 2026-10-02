"""
z504_perfil_clusters.py

Perfil e interpretación de clusters del ejercicio de Miranda.

Objetivo
--------
Los clusters fueron construidos a partir de:

    muestra balanceada BAJA+2 + clientes fieles
        -> Random Forest
        -> matriz P cliente × atributo
        -> KMeans

Este script NO usa P para interpretar directamente los clusters.
Una vez obtenidas las asignaciones, vuelve a las variables originales
del dataset para caracterizar cada segmento.

Se generan dos perfiles:

A) HISTORICO
   Para cada cliente se calcula la media histórica de cada feature.

   Pregunta:
       ¿Qué características estructurales distinguen al cluster?

B) ULTIMA FOTO
   Para cada cliente se toma su última observación disponible.

   Pregunta:
       ¿Cómo se encontraba el cliente al final de su historia observada?

Para comparar variables de escalas diferentes se utiliza:

    SMD = standardized mean difference

También se incorpora el diccionario oficial de la materia:
    - campo
    - unidad
    - Significado

IMPORTANTE
----------
La muestra contiene igual cantidad de clientes BAJA+2 y fieles.

Por lo tanto:

    share_baja2

describe la composición de ESTA MUESTRA BALANCEADA.

NO representa una probabilidad real de baja.

Asimismo, una SMD alta indica asociación descriptiva con el cluster.
NO implica causalidad ni importancia predictiva.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


# ============================================================
# Importar funciones utilizadas en z501
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

sys.path.insert(
    0,
    str(SCRIPT_DIR),
)

import z501_cluster_rf as z501


# ============================================================
# Configuración
# ============================================================

CSV = Path(
    "/data/dmeyf/datasets/competencia_01.csv"
)

DICCIONARIO = Path(
    "/data/dmeyf/datasets/DiccionarioDatos_2026.ods"
)

OUT = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/perfiles"
)

SEED = 214363

N_TREES = 300
MIN_SAMPLES_LEAF = 50

K_VALUES = [2, 3, 4, 5]

TOP_N = 15


# ============================================================
# Diccionario oficial
# ============================================================

def cargar_diccionario(
    path: Path,
) -> pd.DataFrame:

    if not path.exists():
        raise SystemExit(
            f"No encuentro el diccionario: {path}"
        )

    dic = pd.read_excel(
        path,
        sheet_name="Diccionario",
        engine="odf",
    )

    requeridas = {
        "campo",
        "unidad",
        "Significado",
    }

    faltantes = (
        requeridas
        - set(dic.columns)
    )

    if faltantes:
        raise SystemExit(
            "Faltan columnas esperadas en el diccionario: "
            + ", ".join(sorted(faltantes))
        )

    dic = (
        dic[
            [
                "campo",
                "unidad",
                "Significado",
            ]
        ]
        .dropna(
            subset=["campo"]
        )
        .copy()
    )

    dic["campo"] = (
        dic["campo"]
        .astype(str)
        .str.strip()
    )

    # Por seguridad, una fila por campo.
    dic = (
        dic
        .drop_duplicates(
            subset=["campo"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    return dic


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
        name="cluster",
    )


# ============================================================
# Representaciones originales por cliente
# ============================================================

def construir_perfil_historico(
    df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """
    Una fila por cliente.

    Cada feature representa la media de todas las fotos disponibles
    de ese cliente en la muestra utilizada.
    """

    return (
        df
        .groupby(z501.ID_COL)[features]
        .mean()
    )


def construir_perfil_ultima_foto(
    df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """
    Una fila por cliente.

    Se conserva la última foto_mes disponible para cada cliente.
    """

    columnas = [
        z501.ID_COL,
        z501.MES_COL,
        *features,
    ]

    ultimo = (
        df[columnas]
        .sort_values(
            [
                z501.ID_COL,
                z501.MES_COL,
            ]
        )
        .drop_duplicates(
            subset=[z501.ID_COL],
            keep="last",
        )
        .set_index(z501.ID_COL)
    )

    return ultimo[features]


# ============================================================
# Standardized Mean Difference
# ============================================================

def calcular_smd(
    media_cluster: float,
    media_resto: float,
    sd_cluster: float,
    sd_resto: float,
) -> float:
    """
    Diferencia estandarizada de medias:

        (mean_cluster - mean_rest) / pooled_sd

    con:

        pooled_sd = sqrt((sd_cluster² + sd_rest²) / 2)

    Se usa como medida descriptiva de separación.
    """

    valores = [
        media_cluster,
        media_resto,
        sd_cluster,
        sd_resto,
    ]

    if any(
        pd.isna(v)
        for v in valores
    ):
        return np.nan

    pooled_var = (
        sd_cluster ** 2
        + sd_resto ** 2
    ) / 2.0

    if (
        not np.isfinite(pooled_var)
        or pooled_var <= 0
    ):
        return np.nan

    return (
        media_cluster
        - media_resto
    ) / np.sqrt(pooled_var)


# ============================================================
# Perfil de un cluster
# ============================================================

def perfilar_cluster(
    perfil_clientes: pd.DataFrame,
    labels: pd.Series,
    cluster: int,
    diccionario: pd.DataFrame,
    tipo_perfil: str,
) -> pd.DataFrame:

    ids_cluster = labels[
        labels == cluster
    ].index

    ids_resto = labels[
        labels != cluster
    ].index

    A = perfil_clientes.loc[
        perfil_clientes.index.intersection(
            ids_cluster
        )
    ]

    B = perfil_clientes.loc[
        perfil_clientes.index.intersection(
            ids_resto
        )
    ]

    filas = []

    for feature in perfil_clientes.columns:

        a = A[feature]
        b = B[feature]

        media_cluster = a.mean()
        media_resto = b.mean()

        mediana_cluster = a.median()
        mediana_resto = b.median()

        sd_cluster = a.std()
        sd_resto = b.std()

        valor_smd = calcular_smd(
            media_cluster,
            media_resto,
            sd_cluster,
            sd_resto,
        )

        filas.append(
            {
                "tipo_perfil": tipo_perfil,
                "cluster": cluster,
                "feature": feature,

                "n_cluster_validos":
                    int(a.notna().sum()),

                "n_resto_validos":
                    int(b.notna().sum()),

                "media_cluster":
                    media_cluster,

                "media_resto":
                    media_resto,

                "mediana_cluster":
                    mediana_cluster,

                "mediana_resto":
                    mediana_resto,

                "sd_cluster":
                    sd_cluster,

                "sd_resto":
                    sd_resto,

                "dif_media":
                    media_cluster
                    - media_resto,

                "smd":
                    valor_smd,

                "abs_smd":
                    abs(valor_smd)
                    if pd.notna(valor_smd)
                    else np.nan,
            }
        )

    resultado = pd.DataFrame(
        filas
    )

    resultado = resultado.merge(
        diccionario,
        how="left",
        left_on="feature",
        right_on="campo",
    )

    resultado = resultado.sort_values(
        "abs_smd",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    return resultado


# ============================================================
# Composición BAJA+2 / fieles
# ============================================================

def composicion_cluster(
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
        tmp
        .groupby("cluster")["grupo"]
        .agg(
            n_total="size",
            n_baja2="sum",
            share_baja2="mean",
        )
        .reset_index()
    )

    tabla["n_total"] = (
        tabla["n_total"]
        .astype(int)
    )

    tabla["n_baja2"] = (
        tabla["n_baja2"]
        .astype(int)
    )

    tabla["n_fieles"] = (
        tabla["n_total"]
        - tabla["n_baja2"]
    )

    tabla["share_baja2"] = (
        tabla["share_baja2"]
        * 100
    )

    tabla.insert(
        0,
        "k",
        k,
    )

    return tabla


# ============================================================
# Impresión de TOP features
# ============================================================

def imprimir_top(
    perfil: pd.DataFrame,
    k: int,
    cluster: int,
    tipo: str,
) -> pd.DataFrame:

    top = (
        perfil
        .dropna(
            subset=["abs_smd"]
        )
        .head(TOP_N)
        .copy()
    )

    top.insert(
        0,
        "k",
        k,
    )

    print()
    print(
        f"--- k={k} "
        f"cluster={cluster} "
        f"{tipo.upper()} "
        f"TOP {TOP_N} ---"
    )

    columnas = [
        "feature",
        "unidad",
        "media_cluster",
        "media_resto",
        "smd",
        "Significado",
    ]

    if len(top) == 0:

        print(
            "Sin features con SMD calculable."
        )

    else:

        print(
            top[columnas]
            .to_string(
                index=False,
                formatters={
                    "media_cluster":
                        "{:.3f}".format,

                    "media_resto":
                        "{:.3f}".format,

                    "smd":
                        "{:.3f}".format,
                },
            )
        )

    return top


# ============================================================
# Main
# ============================================================

def main() -> None:

    print(
        "=== PERFIL E INTERPRETACIÓN "
        "DE CLUSTERS ==="
    )

    # --------------------------------------------------------
    # Archivos
    # --------------------------------------------------------

    if not CSV.exists():
        raise SystemExit(
            f"No encuentro el dataset: {CSV}"
        )

    if not DICCIONARIO.exists():
        raise SystemExit(
            f"No encuentro el diccionario: "
            f"{DICCIONARIO}"
        )

    # --------------------------------------------------------
    # Diccionario
    # --------------------------------------------------------

    print()
    print(
        "Leyendo diccionario oficial..."
    )

    diccionario = cargar_diccionario(
        DICCIONARIO
    )

    print(
        f"Campos en diccionario: "
        f"{len(diccionario):,}"
    )

    # --------------------------------------------------------
    # Muestra idéntica a z501/z502/z503
    # --------------------------------------------------------

    print()
    print(
        "Construyendo muestra..."
    )

    df = z501.cargar_muestra(
        CSV,
        SEED,
    )

    features = z501.columnas_features(
        df
    )

    print(
        f"Filas              : "
        f"{len(df):,}"
    )

    print(
        f"Clientes únicos    : "
        f"{df[z501.ID_COL].nunique():,}"
    )

    print(
        f"Features numéricas : "
        f"{len(features)}"
    )

    # --------------------------------------------------------
    # Cobertura del diccionario
    # --------------------------------------------------------

    campos_diccionario = set(
        diccionario["campo"]
    )

    encontradas = [
        feature
        for feature in features
        if feature in campos_diccionario
    ]

    faltantes = [
        feature
        for feature in features
        if feature not in campos_diccionario
    ]

    print()
    print("=" * 70)
    print(
        "COBERTURA DEL DICCIONARIO"
    )
    print("=" * 70)

    print(
        f"Features encontradas: "
        f"{len(encontradas)}/{len(features)}"
    )

    if faltantes:

        print()
        print(
            "Features sin entrada en "
            "el diccionario:"
        )

        for feature in faltantes:
            print(
                f"  - {feature}"
            )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    X = (
        df[features]
        .to_numpy(
            dtype=np.float32
        )
    )

    y = (
        df[z501.GRUPO_COL]
        .to_numpy()
    )

    ids = (
        df[z501.ID_COL]
        .to_numpy()
    )

    print()
    print(
        "Entrenando Random Forest "
        f"({N_TREES} árboles)..."
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
    # P
    # --------------------------------------------------------

    print()
    print(
        "Construyendo matriz P..."
    )

    C = z501.contar_hojas_por_id(
        rf,
        X,
        ids,
        features,
    )

    P = z501.normalizar(C)

    print(
        f"P: {P.shape}"
    )

    # --------------------------------------------------------
    # Variables originales por cliente
    # --------------------------------------------------------

    print()
    print(
        "Construyendo perfil histórico..."
    )

    perfil_historico = (
        construir_perfil_historico(
            df,
            features,
        )
        .reindex(P.index)
    )

    print(
        f"Histórico: "
        f"{perfil_historico.shape}"
    )

    print(
        "Construyendo perfil "
        "de última foto..."
    )

    perfil_ultima = (
        construir_perfil_ultima_foto(
            df,
            features,
        )
        .reindex(P.index)
    )

    print(
        f"Última foto: "
        f"{perfil_ultima.shape}"
    )

    # --------------------------------------------------------
    # BAJA+2 / fiel por cliente
    # --------------------------------------------------------

    grupo_por_id = (
        df
        .groupby(z501.ID_COL)[
            z501.GRUPO_COL
        ]
        .max()
        .reindex(P.index)
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    composiciones = []

    resumen_historico = []
    resumen_ultima = []

    asignaciones = pd.DataFrame(
        index=P.index
    )

    # --------------------------------------------------------
    # Soluciones k=2..5
    # --------------------------------------------------------

    for k in K_VALUES:

        print()
        print("=" * 70)
        print(
            f"K = {k}"
        )
        print("=" * 70)

        labels = clusterizar(
            P,
            k,
        )

        asignaciones[
            f"k{k}"
        ] = labels

        comp = composicion_cluster(
            labels,
            grupo_por_id,
            k,
        )

        composiciones.append(
            comp
        )

        print()
        print(
            "COMPOSICIÓN"
        )

        print(
            comp.to_string(
                index=False,
                formatters={
                    "share_baja2":
                        "{:.2f}".format,
                },
            )
        )

        # ----------------------------------------------------
        # Cada cluster
        # ----------------------------------------------------

        for cluster in sorted(
            labels.unique()
        ):

            # HISTÓRICO

            perfil_h = perfilar_cluster(
                perfil_historico,
                labels,
                cluster,
                diccionario,
                tipo_perfil="historico",
            )

            archivo_h = (
                OUT
                / (
                    f"perfil_historico_"
                    f"k{k}_cluster{cluster}.csv"
                )
            )

            perfil_h.to_csv(
                archivo_h,
                index=False,
            )

            top_h = imprimir_top(
                perfil_h,
                k,
                cluster,
                "historico",
            )

            resumen_historico.append(
                top_h
            )

            # ÚLTIMA FOTO

            perfil_u = perfilar_cluster(
                perfil_ultima,
                labels,
                cluster,
                diccionario,
                tipo_perfil="ultima_foto",
            )

            archivo_u = (
                OUT
                / (
                    f"perfil_ultima_foto_"
                    f"k{k}_cluster{cluster}.csv"
                )
            )

            perfil_u.to_csv(
                archivo_u,
                index=False,
            )

            top_u = imprimir_top(
                perfil_u,
                k,
                cluster,
                "ultima_foto",
            )

            resumen_ultima.append(
                top_u
            )

    # ========================================================
    # Resultados agregados
    # ========================================================

    composicion_total = pd.concat(
        composiciones,
        ignore_index=True,
    )

    top_historico_total = pd.concat(
        resumen_historico,
        ignore_index=True,
    )

    top_ultima_total = pd.concat(
        resumen_ultima,
        ignore_index=True,
    )

    composicion_total.to_csv(
        OUT / "composicion_k2_k5.csv",
        index=False,
    )

    top_historico_total.to_csv(
        OUT / "top_historico_k2_k5.csv",
        index=False,
    )

    top_ultima_total.to_csv(
        OUT / "top_ultima_foto_k2_k5.csv",
        index=False,
    )

    asignaciones.index.name = (
        z501.ID_COL
    )

    asignaciones.to_csv(
        OUT / "asignaciones_k2_k5.csv"
    )

    # ========================================================
    # Final
    # ========================================================

    print()
    print("=" * 70)
    print(
        "RESULTADOS GUARDADOS"
    )
    print("=" * 70)

    print(
        OUT / "composicion_k2_k5.csv"
    )

    print(
        OUT / "top_historico_k2_k5.csv"
    )

    print(
        OUT / "top_ultima_foto_k2_k5.csv"
    )

    print(
        OUT / "asignaciones_k2_k5.csv"
    )

    print()
    print(
        "También se generó un perfil "
        "completo por combinación "
        "k/cluster para histórico "
        "y última foto."
    )

    print()
    print(
        "NOTA 1: share_baja2 corresponde "
        "a la muestra balanceada."
    )

    print(
        "NO representa probabilidad real "
        "de baja."
    )

    print()
    print(
        "NOTA 2: SMD es una medida "
        "descriptiva de separación."
    )

    print(
        "NO implica causalidad ni "
        "importancia predictiva."
    )


if __name__ == "__main__":
    main()