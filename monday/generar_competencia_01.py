from pathlib import Path

import duckdb


# ============================================================
# Rutas del proyecto
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

# En EQUIPO333 los datasets pesados se almacenan en el HDD.
# En otros equipos (por ejemplo, la laptop) se usa data/ dentro del repo.
DATA_DIR_EQUIPO333 = Path("/data/dmeyf/datasets")
DATA_DIR = DATA_DIR_EQUIPO333 if DATA_DIR_EQUIPO333.exists() else ROOT / "data"

CSV_ENTRADA = DATA_DIR / "competencia_01_crudo.csv"
CSV_SALIDA = DATA_DIR / "competencia_01.csv"


# ============================================================
# Validación inicial
# ============================================================

if not CSV_ENTRADA.exists():
    raise FileNotFoundError(
        f"No se encontró el archivo de entrada: {CSV_ENTRADA}"
    )

print(f"Entrada: {CSV_ENTRADA}")
print(f"Salida : {CSV_SALIDA}")

# DuckDB maneja mejor las rutas con "/"
entrada_sql = CSV_ENTRADA.as_posix()
salida_sql = CSV_SALIDA.as_posix()


# ============================================================
# Conexión a DuckDB
# ============================================================

con = duckdb.connect()


# ============================================================
# Dataset original
# ============================================================

con.execute(f"""
    CREATE OR REPLACE VIEW competencia_01_crudo AS
    SELECT *
    FROM read_csv_auto('{entrada_sql}')
""")


cantidad_original = con.execute("""
    SELECT COUNT(*)
    FROM competencia_01_crudo
""").fetchone()[0]

print(f"\nRegistros originales: {cantidad_original:,}")


# ============================================================
# Construcción de clase_ternaria
#
# Para cada cliente se genera una grilla con todos los períodos.
# mes_0 indica presencia en el período actual.
# mes_1 indica presencia un mes después.
# mes_2 indica presencia dos meses después.
#
# Casos:
#
# mes_1  mes_2   clase
# ------  ------  ----------------
#   0       0     BAJA+1
#   0       1     CONTINUA
#   1       0     BAJA+2
#   1       1     CONTINUA
#   0      NULL   BAJA+1
#   1      NULL   NULL
#  NULL    NULL   NULL
#
# El caso 0 -> 1 se considera CONTINUA porque el cliente
# reaparece en el segundo mes.
#
# En 202107 puede conocerse BAJA+1 cuando mes_1 = 0,
# aunque mes_2 todavía no esté disponible.
# ============================================================

con.execute(f"""
COPY (

    WITH periodos AS (
        SELECT DISTINCT
            foto_mes
        FROM competencia_01_crudo
    ),

    clientes AS (
        SELECT DISTINCT
            numero_de_cliente
        FROM competencia_01_crudo
    ),

    grilla AS (
        SELECT
            numero_de_cliente,
            foto_mes
        FROM clientes
        CROSS JOIN periodos
    ),

    presencia AS (
        SELECT
            g.numero_de_cliente,
            g.foto_mes,

            CASE
                WHEN c.numero_de_cliente IS NULL THEN 0
                ELSE 1
            END AS mes_0

        FROM grilla g

        LEFT JOIN competencia_01_crudo c
            USING (numero_de_cliente, foto_mes)
    ),

    futuro AS (
        SELECT
            numero_de_cliente,
            foto_mes,
            mes_0,

            LEAD(mes_0, 1) OVER (
                PARTITION BY numero_de_cliente
                ORDER BY foto_mes
            ) AS mes_1,

            LEAD(mes_0, 2) OVER (
                PARTITION BY numero_de_cliente
                ORDER BY foto_mes
            ) AS mes_2

        FROM presencia
    ),

    target AS (
        SELECT
            numero_de_cliente,
            foto_mes,
            mes_0,

            CASE
                -- Si reaparece o continúa presente en t+2,
                -- se considera CONTINUA.
                WHEN mes_2 = 1
                    THEN 'CONTINUA'

                -- Está presente en t+1 pero desaparece en t+2.
                WHEN mes_1 = 1
                     AND mes_2 = 0
                    THEN 'BAJA+2'

                -- Desaparece inmediatamente en t+1.
                -- También permite identificar BAJA+1 en 202107.
                WHEN mes_1 = 0
                    THEN 'BAJA+1'

                -- No hay suficiente información futura.
                ELSE NULL
            END AS clase_ternaria

        FROM futuro
    )

    SELECT
        c.*,
        t.clase_ternaria

    FROM competencia_01_crudo c

    INNER JOIN target t
        USING (numero_de_cliente, foto_mes)

    -- Conservamos solamente observaciones reales.
    WHERE t.mes_0 = 1

    ORDER BY
        c.foto_mes,
        c.numero_de_cliente

)
TO '{salida_sql}'
(
    FORMAT CSV,
    HEADER
)
""")


# ============================================================
# Validación del resultado
# ============================================================

con.execute(f"""
    CREATE OR REPLACE VIEW competencia_01 AS
    SELECT *
    FROM read_csv_auto('{salida_sql}')
""")


cantidad_final = con.execute("""
    SELECT COUNT(*)
    FROM competencia_01
""").fetchone()[0]


print(f"Registros finales   : {cantidad_final:,}")

if cantidad_original != cantidad_final:
    raise ValueError(
        "La cantidad de registros del archivo final no coincide "
        "con el archivo original."
    )

print("OK: se conservaron todos los registros.")


# ============================================================
# Distribución de clase_ternaria por período
# ============================================================

resumen = con.execute("""
    SELECT
        foto_mes,
        clase_ternaria,
        COUNT(*) AS cantidad

    FROM competencia_01

    GROUP BY
        foto_mes,
        clase_ternaria

    ORDER BY
        foto_mes,
        clase_ternaria
""").fetchdf()


print("\nDistribución de clase_ternaria:")
print(resumen)


# ============================================================
# Validación de totales por período
# ============================================================

totales = con.execute("""
    SELECT
        foto_mes,
        COUNT(*) AS cantidad

    FROM competencia_01

    GROUP BY foto_mes

    ORDER BY foto_mes
""").fetchdf()

print("\nCantidad total de registros por período:")
print(totales)


# ============================================================
# Tamaño del archivo generado
# ============================================================

tamano_mb = CSV_SALIDA.stat().st_size / (1024 ** 2)

print(f"\nTamaño del archivo: {tamano_mb:,.2f} MB")


# ============================================================
# Cierre
# ============================================================

con.close()

print("\nProceso finalizado correctamente.")