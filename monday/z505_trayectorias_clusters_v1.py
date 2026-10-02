"""
z505_trayectorias_clusters.py

DMEyF 2026
Análisis temporal de los clusters obtenidos en el espacio supervisado
del Random Forest.

Objetivo
--------
Tomar las asignaciones de clusters k=5 generadas previamente y estudiar
cómo evolucionan temporalmente las variables originales antes de la
última observación disponible de cada cliente.

IMPORTANTE
----------
- Este script NO vuelve a entrenar el Random Forest.
- Este script NO vuelve a clusterizar.
- Los clusters se consideran una segmentación ya obtenida.
- El análisis es descriptivo.
- Las trayectorias NO implican causalidad.

Salida
------
Genera:
    trayectorias_medias_k5.csv
    trayectorias_relativas_k5.csv
    deltas_k5.csv
    pendientes_k5.csv
    resumen_trayectorias_k5.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

DATASET = Path(
    "/data/dmeyf/datasets/competencia_01.csv"
)

DIR_RESULTADOS = Path(
    "/data/dmeyf/datasets/evaluacion_clusters"
)

DIR_PERFILES = DIR_RESULTADOS / "perfiles"

ASIGNACIONES = DIR_PERFILES / "asignaciones_k2_k5.csv"

DIR_SALIDA = DIR_RESULTADOS / "trayectorias"
DIR_SALIDA.mkdir(parents=True, exist_ok=True)


# Cantidad máxima de fotos hacia atrás.
#
# t=0  : última foto disponible del cliente
# t=-1 : foto inmediatamente anterior
# ...
MAX_LAGS = 12


# Variables seleccionadas a partir de los perfiles de z504.
#
# Se incluyen distintas dimensiones:
# - actividad general
# - productos
# - canales
# - tarjetas
# - pagos / débitos
# - comisiones
# - estados
# - mora
VARIABLES_CANDIDATAS = [

    # --------------------------------------------------------
    # Actividad general
    # --------------------------------------------------------
    "active_quarter",
    "ctrx_quarter",
    "cproductos",

    # --------------------------------------------------------
    # Canales digitales
    # --------------------------------------------------------
    "internet",
    "thomebanking",
    "chomebanking_transacciones",
    "cmobile_app_trx",

    # --------------------------------------------------------
    # Débito / ATM
    # --------------------------------------------------------
    "ctarjeta_debito_transacciones",
    "mautoservicio",
    "catm_trx",
    "cextraccion_autoservicio",
    "mextraccion_autoservicio",

    # --------------------------------------------------------
    # Transferencias / pagos
    # --------------------------------------------------------
    "ctransferencias_emitidas",
    "cpagomiscuentas",
    "ccuenta_debitos_automaticos",
    "cpayroll_trx",
    "mpayroll",

    # --------------------------------------------------------
    # Productos tarjeta
    # --------------------------------------------------------
    "ctarjeta_visa",
    "ctarjeta_master",

    # --------------------------------------------------------
    # Uso Visa
    # --------------------------------------------------------
    "ctarjeta_visa_transacciones",
    "Visa_cconsumos",
    "mtarjeta_visa_consumo",
    "ctarjeta_visa_debitos_automaticos",
    "Visa_mpagospesos",

    # --------------------------------------------------------
    # Estados de tarjeta
    # --------------------------------------------------------
    "Visa_status",
    "Master_status",

    # --------------------------------------------------------
    # Fechas / mora
    # --------------------------------------------------------
    "Visa_Finiciomora",
    "Master_Finiciomora",
    "Visa_fechaalta",
    "Master_fechaalta",

    # --------------------------------------------------------
    # Comisiones
    # --------------------------------------------------------
    "ccomisiones_otras",
    "mcomisiones_otras",
    "ccomisiones_mantenimiento",
    "mcomisiones_mantenimiento",
    "mcomisiones",
]


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def encabezado(texto):
    print()
    print("=" * 70)
    print(texto)
    print("=" * 70)


def buscar_columna_cluster_k5(asignaciones):
    """
    Busca automáticamente la columna correspondiente a k=5.

    Admite nombres habituales:
        k5
        cluster_k5
        cluster5

    Si no encuentra coincidencia exacta, busca una columna que contenga
    simultáneamente 'cluster' y '5'.
    """

    candidatos = [
        "k5",
        "cluster_k5",
        "cluster5",
        "cluster_5",
    ]

    for c in candidatos:
        if c in asignaciones.columns:
            return c

    for c in asignaciones.columns:
        cl = c.lower()

        if "cluster" in cl and "5" in cl:
            return c

    raise ValueError(
        "No se pudo identificar la columna de clusters k=5.\n"
        f"Columnas disponibles: {asignaciones.columns.tolist()}"
    )


def buscar_columna_cliente(df):
    """
    Identifica la columna ID de cliente.
    """

    candidatos = [
        "numero_de_cliente",
        "cliente",
        "customer_id",
        "id_cliente",
    ]

    for c in candidatos:
        if c in df.columns:
            return c

    raise ValueError(
        "No se pudo identificar la columna del cliente.\n"
        f"Columnas disponibles: {df.columns.tolist()}"
    )


def preparar_asignaciones(asignaciones):
    """
    Devuelve:
        numero_de_cliente
        cluster
    """

    col_cliente = buscar_columna_cliente(asignaciones)
    col_cluster = buscar_columna_cluster_k5(asignaciones)

    resultado = asignaciones[
        [col_cliente, col_cluster]
    ].copy()

    resultado.columns = [
        "numero_de_cliente",
        "cluster",
    ]

    resultado = resultado.drop_duplicates(
        subset=["numero_de_cliente"]
    )

    resultado["cluster"] = pd.to_numeric(
        resultado["cluster"],
        errors="coerce",
    )

    resultado = resultado.dropna(
        subset=["cluster"]
    )

    resultado["cluster"] = (
        resultado["cluster"].astype(int)
    )

    return resultado


def construir_tiempo_relativo(df):
    """
    Construye una escala temporal relativa por cliente.

    Ejemplo:

        foto_mes       posicion
        202101           -3
        202102           -2
        202103           -1
        202104            0

    t=0 representa siempre la última foto observada del cliente.
    """

    df = df.sort_values(
        ["numero_de_cliente", "foto_mes"]
    ).copy()

    df["_orden"] = (
        df.groupby("numero_de_cliente")
        .cumcount()
    )

    df["_n_fotos"] = (
        df.groupby("numero_de_cliente")[
            "numero_de_cliente"
        ]
        .transform("size")
    )

    df["t"] = (
        df["_orden"]
        - df["_n_fotos"]
        + 1
    )

    return df


def calcular_trayectorias_medias(
    df,
    variables,
):
    """
    Calcula media, mediana, cantidad de observaciones y desvío
    para cada:

        cluster × t × variable
    """

    registros = []

    grupos = df.groupby(
        ["cluster", "t"],
        sort=True,
    )

    for (cluster, t), g in grupos:

        for variable in variables:

            serie = pd.to_numeric(
                g[variable],
                errors="coerce",
            )

            serie = serie.replace(
                [np.inf, -np.inf],
                np.nan,
            )

            validos = serie.dropna()

            if len(validos) == 0:
                continue

            registros.append(
                {
                    "cluster": int(cluster),
                    "t": int(t),
                    "variable": variable,
                    "n": int(validos.size),
                    "media": float(validos.mean()),
                    "mediana": float(validos.median()),
                    "std": float(validos.std()),
                    "q25": float(validos.quantile(0.25)),
                    "q75": float(validos.quantile(0.75)),
                }
            )

    return pd.DataFrame(registros)


def calcular_trayectorias_relativas(
    trayectorias,
):
    """
    Expresa cada media relativa a la media del mismo cluster en t=0.

    Dos métricas:

        delta_vs_t0
            media(t) - media(t0)

        ratio_vs_t0
            media(t) / media(t0)

    Esto permite observar dirección y magnitud de la trayectoria.
    """

    base = (
        trayectorias[
            trayectorias["t"] == 0
        ][
            [
                "cluster",
                "variable",
                "media",
            ]
        ]
        .rename(
            columns={
                "media": "media_t0"
            }
        )
    )

    resultado = trayectorias.merge(
        base,
        on=[
            "cluster",
            "variable",
        ],
        how="left",
    )

    resultado["delta_vs_t0"] = (
        resultado["media"]
        - resultado["media_t0"]
    )

    resultado["ratio_vs_t0"] = np.where(
        resultado["media_t0"].abs() > 1e-12,
        resultado["media"]
        / resultado["media_t0"],
        np.nan,
    )

    return resultado


def calcular_deltas_cliente(
    df,
    variables,
):
    """
    Calcula cambios individuales entre distintas ventanas.

    Para cada cliente:

        delta_1 = t0 - t-1
        delta_2 = t0 - t-2
        delta_3 = t0 - t-3
        delta_6 = t0 - t-6

    Luego resume esos deltas por cluster.
    """

    ventanas = [
        1,
        2,
        3,
        6,
    ]

    registros = []

    indexado = df.set_index(
        [
            "numero_de_cliente",
            "t",
        ]
    )

    clientes = (
        df[
            [
                "numero_de_cliente",
                "cluster",
            ]
        ]
        .drop_duplicates()
    )

    for _, fila in clientes.iterrows():

        cliente = fila["numero_de_cliente"]
        cluster = int(fila["cluster"])

        try:
            actual = indexado.loc[
                (cliente, 0)
            ]
        except KeyError:
            continue

        # En caso improbable de duplicados.
        if isinstance(actual, pd.DataFrame):
            actual = actual.iloc[-1]

        for lag in ventanas:

            try:
                anterior = indexado.loc[
                    (cliente, -lag)
                ]
            except KeyError:
                continue

            if isinstance(anterior, pd.DataFrame):
                anterior = anterior.iloc[-1]

            for variable in variables:

                x0 = pd.to_numeric(
                    pd.Series(
                        [actual[variable]]
                    ),
                    errors="coerce",
                ).iloc[0]

                xl = pd.to_numeric(
                    pd.Series(
                        [anterior[variable]]
                    ),
                    errors="coerce",
                ).iloc[0]

                if pd.isna(x0) or pd.isna(xl):
                    continue

                registros.append(
                    {
                        "numero_de_cliente": cliente,
                        "cluster": cluster,
                        "variable": variable,
                        "lag": lag,
                        "valor_t0": float(x0),
                        "valor_t_lag": float(xl),
                        "delta": float(x0 - xl),
                    }
                )

    detalle = pd.DataFrame(registros)

    if detalle.empty:
        return detalle, pd.DataFrame()

    resumen = (
        detalle
        .groupby(
            [
                "cluster",
                "variable",
                "lag",
            ],
            as_index=False,
        )
        .agg(
            n=("delta", "size"),
            delta_media=("delta", "mean"),
            delta_mediana=("delta", "median"),
            delta_std=("delta", "std"),
            q25=("delta", lambda x: x.quantile(0.25)),
            q75=("delta", lambda x: x.quantile(0.75)),
        )
    )

    return detalle, resumen


def pendiente_individual(
    tiempos,
    valores,
):
    """
    Pendiente de una regresión lineal simple:

        valor ~ t

    Requiere al menos 3 observaciones válidas.
    """

    tiempos = np.asarray(
        tiempos,
        dtype=float,
    )

    valores = np.asarray(
        valores,
        dtype=float,
    )

    mascara = (
        np.isfinite(tiempos)
        & np.isfinite(valores)
    )

    tiempos = tiempos[mascara]
    valores = valores[mascara]

    if len(valores) < 3:
        return np.nan

    if np.std(tiempos) == 0:
        return np.nan

    pendiente = np.polyfit(
        tiempos,
        valores,
        deg=1,
    )[0]

    return float(pendiente)


def calcular_pendientes(
    df,
    variables,
):
    """
    Calcula la pendiente temporal individual de cada variable
    durante las observaciones disponibles dentro de MAX_LAGS.

    Después resume por cluster.
    """

    registros = []

    for cliente, g in df.groupby(
        "numero_de_cliente"
    ):

        cluster = int(
            g["cluster"].iloc[0]
        )

        tiempos = g["t"].to_numpy()

        for variable in variables:

            valores = pd.to_numeric(
                g[variable],
                errors="coerce",
            ).to_numpy(
                dtype=float
            )

            pendiente = pendiente_individual(
                tiempos,
                valores,
            )

            if np.isnan(pendiente):
                continue

            registros.append(
                {
                    "numero_de_cliente": cliente,
                    "cluster": cluster,
                    "variable": variable,
                    "pendiente": pendiente,
                }
            )

    detalle = pd.DataFrame(registros)

    if detalle.empty:
        return detalle, pd.DataFrame()

    resumen = (
        detalle
        .groupby(
            [
                "cluster",
                "variable",
            ],
            as_index=False,
        )
        .agg(
            n=("pendiente", "size"),
            pendiente_media=(
                "pendiente",
                "mean",
            ),
            pendiente_mediana=(
                "pendiente",
                "median",
            ),
            pendiente_std=(
                "pendiente",
                "std",
            ),
            q25=(
                "pendiente",
                lambda x: x.quantile(0.25),
            ),
            q75=(
                "pendiente",
                lambda x: x.quantile(0.75),
            ),
        )
    )

    return detalle, resumen


def construir_resumen(
    trayectorias,
    pendientes,
):
    """
    Construye una tabla compacta para inspección.

    Para cada cluster-variable muestra:

        media t=0
        media t=-1
        media t=-3
        media t=-6
        pendiente media
    """

    tiempos_interes = [
        0,
        -1,
        -3,
        -6,
    ]

    partes = []

    for t in tiempos_interes:

        aux = (
            trayectorias[
                trayectorias["t"] == t
            ][
                [
                    "cluster",
                    "variable",
                    "media",
                ]
            ]
            .rename(
                columns={
                    "media": f"media_t{t}"
                }
            )
        )

        partes.append(aux)

    if not partes:
        return pd.DataFrame()

    resumen = partes[0]

    for aux in partes[1:]:

        resumen = resumen.merge(
            aux,
            on=[
                "cluster",
                "variable",
            ],
            how="outer",
        )

    if not pendientes.empty:

        resumen = resumen.merge(
            pendientes[
                [
                    "cluster",
                    "variable",
                    "pendiente_media",
                    "pendiente_mediana",
                ]
            ],
            on=[
                "cluster",
                "variable",
            ],
            how="left",
        )

    # Cambio de seis fotos a la última.
    if (
        "media_t0" in resumen.columns
        and "media_t-6" in resumen.columns
    ):

        resumen["delta_t0_t6"] = (
            resumen["media_t0"]
            - resumen["media_t-6"]
        )

    return resumen


def imprimir_variables_destacadas(
    resumen,
    top_n=12,
):
    """
    Muestra las variables con mayor cambio estandarizado relativo
    entre t=-6 y t=0 dentro de cada cluster.

    Para evitar problemas de escala se usa:

        abs(delta) / (abs(media_t-6) + 1)

    Esta métrica es SOLO para ordenar la salida exploratoria.
    """

    if resumen.empty:
        return

    if (
        "media_t0" not in resumen.columns
        or "media_t-6" not in resumen.columns
    ):
        return

    r = resumen.copy()

    r["cambio_relativo_orden"] = (
        (
            r["media_t0"]
            - r["media_t-6"]
        ).abs()
        /
        (
            r["media_t-6"].abs()
            + 1.0
        )
    )

    for cluster in sorted(
        r["cluster"].dropna().unique()
    ):

        encabezado(
            f"CLUSTER {int(cluster)} — "
            f"MAYORES CAMBIOS t=-6 -> t=0"
        )

        aux = (
            r[
                r["cluster"] == cluster
            ]
            .sort_values(
                "cambio_relativo_orden",
                ascending=False,
            )
            .head(top_n)
            .copy()
        )

        columnas = [
            "variable",
            "media_t-6",
            "media_t-3",
            "media_t-1",
            "media_t0",
            "delta_t0_t6",
            "pendiente_media",
        ]

        columnas = [
            c
            for c in columnas
            if c in aux.columns
        ]

        print(
            aux[columnas]
            .to_string(
                index=False,
                float_format=lambda x: f"{x:,.3f}",
            )
        )


# ============================================================
# MAIN
# ============================================================

def main():

    encabezado(
        "Z505 — TRAYECTORIAS TEMPORALES DE LOS CLUSTERS"
    )

    # --------------------------------------------------------
    # 1. Cargar asignaciones
    # --------------------------------------------------------

    print(
        f"\nCargando asignaciones:\n{ASIGNACIONES}"
    )

    if not ASIGNACIONES.exists():
        raise FileNotFoundError(
            f"No existe:\n{ASIGNACIONES}\n\n"
            "Ejecutar primero z504."
        )

    asignaciones_raw = pd.read_csv(
        ASIGNACIONES
    )

    print(
        "Asignaciones:",
        asignaciones_raw.shape,
    )

    print(
        "Columnas:",
        asignaciones_raw.columns.tolist(),
    )

    asignaciones = preparar_asignaciones(
        asignaciones_raw
    )

    print(
        "\nClientes con cluster k=5:",
        f"{len(asignaciones):,}",
    )

    print(
        "\nDistribución k=5:"
    )

    print(
        asignaciones[
            "cluster"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # 2. Cargar dataset
    # --------------------------------------------------------

    encabezado(
        "CARGA DEL DATASET HISTÓRICO"
    )

    print(DATASET)

    df = pd.read_csv(
        DATASET
    )

    print(
        "Dataset:",
        df.shape,
    )

    # --------------------------------------------------------
    # 3. Validaciones
    # --------------------------------------------------------

    obligatorias = [
        "numero_de_cliente",
        "foto_mes",
    ]

    faltantes = [
        c
        for c in obligatorias
        if c not in df.columns
    ]

    if faltantes:
        raise ValueError(
            "Faltan columnas obligatorias: "
            + str(faltantes)
        )

    variables = [
        c
        for c in VARIABLES_CANDIDATAS
        if c in df.columns
    ]

    variables_faltantes = [
        c
        for c in VARIABLES_CANDIDATAS
        if c not in df.columns
    ]

    print(
        f"\nVariables disponibles: "
        f"{len(variables)}"
    )

    if variables_faltantes:

        print(
            "\nVariables candidatas no encontradas:"
        )

        for c in variables_faltantes:
            print(" -", c)

    # --------------------------------------------------------
    # 4. Filtrar clientes del experimento
    # --------------------------------------------------------

    clientes = set(
        asignaciones[
            "numero_de_cliente"
        ]
    )

    df = df[
        df[
            "numero_de_cliente"
        ].isin(clientes)
    ].copy()

    print(
        "\nFilas de clientes clusterizados:",
        f"{len(df):,}",
    )

    print(
        "Clientes:",
        f"{df['numero_de_cliente'].nunique():,}",
    )

    # --------------------------------------------------------
    # 5. Agregar cluster
    # --------------------------------------------------------

    df = df.merge(
        asignaciones,
        on="numero_de_cliente",
        how="inner",
        validate="many_to_one",
    )

    # --------------------------------------------------------
    # 6. Orden temporal
    # --------------------------------------------------------

    df["foto_mes"] = pd.to_numeric(
        df["foto_mes"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["foto_mes"]
    )

    df["foto_mes"] = (
        df["foto_mes"].astype(int)
    )

    df = construir_tiempo_relativo(
        df
    )

    # Conservar únicamente la ventana elegida.
    df = df[
        (df["t"] <= 0)
        & (df["t"] >= -MAX_LAGS)
    ].copy()

    encabezado(
        "COBERTURA TEMPORAL"
    )

    cobertura = (
        df.groupby("t")
        .agg(
            filas=(
                "numero_de_cliente",
                "size",
            ),
            clientes=(
                "numero_de_cliente",
                "nunique",
            ),
        )
        .reset_index()
        .sort_values("t")
    )

    print(
        cobertura.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 7. Trayectorias agregadas
    # --------------------------------------------------------

    encabezado(
        "CALCULANDO TRAYECTORIAS MEDIAS"
    )

    trayectorias = (
        calcular_trayectorias_medias(
            df,
            variables,
        )
    )

    print(
        "Registros:",
        f"{len(trayectorias):,}",
    )

    # --------------------------------------------------------
    # 8. Trayectorias relativas
    # --------------------------------------------------------

    encabezado(
        "CALCULANDO CAMBIOS RELATIVOS"
    )

    trayectorias_relativas = (
        calcular_trayectorias_relativas(
            trayectorias
        )
    )

    print(
        "Registros:",
        f"{len(trayectorias_relativas):,}",
    )

    # --------------------------------------------------------
    # 9. Deltas individuales
    # --------------------------------------------------------

    encabezado(
        "CALCULANDO DELTAS INDIVIDUALES"
    )

    (
        deltas_detalle,
        deltas_resumen,
    ) = calcular_deltas_cliente(
        df,
        variables,
    )

    print(
        "Deltas individuales:",
        f"{len(deltas_detalle):,}",
    )

    print(
        "Resumen deltas:",
        f"{len(deltas_resumen):,}",
    )

    # --------------------------------------------------------
    # 10. Pendientes
    # --------------------------------------------------------

    encabezado(
        "CALCULANDO PENDIENTES TEMPORALES"
    )

    (
        pendientes_detalle,
        pendientes_resumen,
    ) = calcular_pendientes(
        df,
        variables,
    )

    print(
        "Pendientes individuales:",
        f"{len(pendientes_detalle):,}",
    )

    print(
        "Resumen pendientes:",
        f"{len(pendientes_resumen):,}",
    )

    # --------------------------------------------------------
    # 11. Resumen
    # --------------------------------------------------------

    resumen = construir_resumen(
        trayectorias,
        pendientes_resumen,
    )

    imprimir_variables_destacadas(
        resumen,
        top_n=15,
    )

    # --------------------------------------------------------
    # 12. Guardar
    # --------------------------------------------------------

    encabezado(
        "GUARDANDO RESULTADOS"
    )

    archivo_trayectorias = (
        DIR_SALIDA
        / "trayectorias_medias_k5.csv"
    )

    archivo_relativas = (
        DIR_SALIDA
        / "trayectorias_relativas_k5.csv"
    )

    archivo_deltas = (
        DIR_SALIDA
        / "deltas_k5.csv"
    )

    archivo_deltas_detalle = (
        DIR_SALIDA
        / "deltas_detalle_k5.csv"
    )

    archivo_pendientes = (
        DIR_SALIDA
        / "pendientes_k5.csv"
    )

    archivo_pendientes_detalle = (
        DIR_SALIDA
        / "pendientes_detalle_k5.csv"
    )

    archivo_resumen = (
        DIR_SALIDA
        / "resumen_trayectorias_k5.csv"
    )

    archivo_base = (
        DIR_SALIDA
        / "base_temporal_k5.csv"
    )

    trayectorias.to_csv(
        archivo_trayectorias,
        index=False,
    )

    trayectorias_relativas.to_csv(
        archivo_relativas,
        index=False,
    )

    deltas_resumen.to_csv(
        archivo_deltas,
        index=False,
    )

    deltas_detalle.to_csv(
        archivo_deltas_detalle,
        index=False,
    )

    pendientes_resumen.to_csv(
        archivo_pendientes,
        index=False,
    )

    pendientes_detalle.to_csv(
        archivo_pendientes_detalle,
        index=False,
    )

    resumen.to_csv(
        archivo_resumen,
        index=False,
    )

    columnas_base = [
        "numero_de_cliente",
        "foto_mes",
        "cluster",
        "t",
    ] + variables

    df[
        columnas_base
    ].to_csv(
        archivo_base,
        index=False,
    )

    for archivo in [
        archivo_trayectorias,
        archivo_relativas,
        archivo_deltas,
        archivo_deltas_detalle,
        archivo_pendientes,
        archivo_pendientes_detalle,
        archivo_resumen,
        archivo_base,
    ]:
        print(archivo)

    encabezado(
        "INTERPRETACIÓN"
    )

    print(
        """
t = 0
    última foto disponible de cada cliente

t = -1
    foto inmediatamente anterior

t = -6
    seis observaciones antes de la última

delta_t0_t6
    media(t=0) - media(t=-6)

pendiente > 0
    la variable tiende a aumentar hacia la última foto

pendiente < 0
    la variable tiende a disminuir hacia la última foto

IMPORTANTE:
Los clusters fueron construidos previamente.
Este análisis describe sus trayectorias históricas.

No implica causalidad.

Además, la composición BAJA+2/fieles proviene de una muestra
balanceada y no representa probabilidades poblacionales.
"""
    )


if __name__ == "__main__":
    main()