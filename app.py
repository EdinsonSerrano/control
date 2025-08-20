# app.py
# ---------------------------------------------------------------
# Aplicativo de Finanzas Personales
# Stack: Python 3.10+, Streamlit, SQLite, Pandas, Plotly, OpenPyXL
# Autor: ChatGPT (GPT-5 Thinking)
# Licencia: MIT (puedes reutilizar y modificar)
# ---------------------------------------------------------------
"""
INSTRUCCIONES RÁPIDAS DE DESPLIEGUE (HOSTING GRATUITO)
1) Crea un repositorio en GitHub con estos archivos:
   - app.py (este archivo)
   - requirements.txt (contenido sugerido al final de este archivo, cópialo en un archivo nuevo)
   - (opcional) .streamlit/config.toml para personalizar tema.

2) Ve a https://share.streamlit.io/ (Streamlit Community Cloud) e inicia sesión con GitHub.
3) Crea una nueva app → selecciona tu repo y el archivo principal `app.py`.
4) Espera el build. ¡Listo! Quedará pública y gratuita.

NOTAS IMPORTANTES
- La base de datos es SQLite y se crea como `data.db` en el directorio de trabajo.
- En despliegue gratuito, el almacenamiento es efímero; exporta a Excel con frecuencia para respaldo.
- Puedes conectar Google Drive / S3 más adelante si necesitas persistencia fuerte.
"""

import os
import io
import sqlite3
from datetime import datetime, date

import pandas as pd
import plotly.express as px
import streamlit as st

# ----------------------------
# Configuración general UI
# ----------------------------
st.set_page_config(
    page_title="Finanzas Personales",
    page_icon="💸",
    layout="wide",
)

PRIMARY_COLOR = "#3B82F6"  # azul
SUCCESS_COLOR = "#10B981"   # verde
DANGER_COLOR = "#EF4444"    # rojo

# ----------------------------
# Utilidades de base de datos
# ----------------------------
DB_PATH = "data.db"

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()

    # Ingresos
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ingresos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha TEXT NOT NULL,
            fuente TEXT,
            categoria TEXT,
            monto REAL NOT NULL,
            descripcion TEXT
        );
        """
    )

    # Gastos
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS gastos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha TEXT NOT NULL,
            categoria TEXT,
            metodo TEXT,
            proveedor TEXT,
            monto REAL NOT NULL,
            descripcion TEXT
        );
        """
    )

    # Deuda - movimientos (Desembolso = entra dinero; Pago = sale dinero)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS deuda_movimientos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha TEXT NOT NULL,
            deuda TEXT NOT NULL,
            tipo TEXT CHECK(tipo IN ('Desembolso','Pago')) NOT NULL,
            monto REAL NOT NULL,
            descripcion TEXT
        );
        """
    )

    # Inversiones
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS inversiones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha TEXT NOT NULL,
            instrumento TEXT,
            monto REAL NOT NULL,
            descripcion TEXT
        );
        """
    )

    # Ahorros
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ahorros (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha TEXT NOT NULL,
            meta TEXT,
            monto REAL NOT NULL,
            descripcion TEXT
        );
        """
    )

    # Categorías personalizables
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS categorias (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tipo TEXT CHECK(tipo IN ('ingreso','gasto')) NOT NULL,
            nombre TEXT NOT NULL UNIQUE
        );
        """
    )

    # Semilla de categorías básicas
    categorias_seed = {
        'ingreso': ["Salario", "Negocio", "Freelance", "Intereses", "Otros"],
        'gasto': [
            "Alimentación", "Transporte", "Vivienda", "Servicios", "Entretenimiento",
            "Salud", "Educación", "Ropa", "Mascotas", "Imprevistos", "Suscripciones",
            "Cafés/Snacks"  # útil para gasto hormiga
        ],
    }
    for tipo, items in categorias_seed.items():
        for nombre in items:
            try:
                cur.execute("INSERT INTO categorias(tipo,nombre) VALUES(?,?)", (tipo, nombre))
            except sqlite3.IntegrityError:
                pass

    conn.commit()
    conn.close()

init_db()

# ----------------------------
# Funciones helper
# ----------------------------

def df_from_sql(query: str, params: tuple | None = None) -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def insert_row(table: str, data: dict):
    conn = get_conn()
    cols = ",".join(data.keys())
    placeholders = ",".join(["?"] * len(data))
    values = tuple(data.values())
    conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({placeholders})", values)
    conn.commit()
    conn.close()


def parse_date(d: date | str) -> str:
    if isinstance(d, date):
        return d.isoformat()
    return d


def human_money(x: float) -> str:
    try:
        return f"$ {x:,.0f}".replace(",", ".").replace(".", ",", 1).replace(",", ".", 1)
    except Exception:
        return f"$ {x:,.0f}"


def load_all():
    ingresos = df_from_sql("SELECT * FROM ingresos")
    gastos = df_from_sql("SELECT * FROM gastos")
    deudas = df_from_sql("SELECT * FROM deuda_movimientos")
    inversiones = df_from_sql("SELECT * FROM inversiones")
    ahorros = df_from_sql("SELECT * FROM ahorros")
    return ingresos, gastos, deudas, inversiones, ahorros


def add_period_columns(df: pd.DataFrame, date_col: str = "fecha") -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["anio"] = df[date_col].dt.year
    df["mes"] = df[date_col].dt.month
    df["dia"] = df[date_col].dt.day
    # Quincena: Q1 = días 1-15, Q2 = resto
    df["quincena"] = df[date_col].dt.day.apply(lambda d: 1 if d <= 15 else 2)
    # Trimestre/Semestre
    df["trimestre"] = df[date_col].dt.quarter
    df["semestre"] = df[date_col].dt.month.apply(lambda m: 1 if m <= 6 else 2)
    # Etiquetas útiles
    df["YQ"] = df.apply(lambda r: f"{r['anio']}-Q{int(r['trimestre'])}", axis=1)
    df["YS"] = df.apply(lambda r: f"{r['anio']}-S{int(r['semestre'])}", axis=1)
    df["YM"] = df[date_col].dt.to_period('M').astype(str)
    df["YQn"] = df[date_col].dt.to_period('Q').astype(str)
    return df


def aggregate_by_period(df: pd.DataFrame, value_col: str, date_col: str, period: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["periodo", value_col])
    df = add_period_columns(df, date_col)
    if period == "Diario":
        grp = df.groupby(df[date_col].dt.date)[value_col].sum().reset_index(name=value_col)
        grp.rename(columns={date_col: "periodo"}, inplace=True)
    elif period == "Quincenal":
        grp = df.groupby(["anio", "mes", "quincena"], as_index=False)[value_col].sum()
        grp["periodo"] = grp.apply(lambda r: f"{int(r['anio'])}-{int(r['mes']):02d}-Q{int(r['quincena'])}", axis=1)
        grp = grp[["periodo", value_col]]
    elif period == "Mensual":
        grp = df.groupby("YM", as_index=False)[value_col].sum().rename(columns={"YM": "periodo"})
    elif period == "Trimestral":
        grp = df.groupby("YQ", as_index=False)[value_col].sum().rename(columns={"YQ": "periodo"})
    elif period == "Semestral":
        grp = df.groupby("YS", as_index=False)[value_col].sum().rename(columns={"YS": "periodo"})
    elif period == "Anual":
        grp = df.groupby("anio", as_index=False)[value_col].sum().rename(columns={"anio": "periodo"})
    else:
        grp = df.groupby("YM", as_index=False)[value_col].sum().rename(columns={"YM": "periodo"})
    return grp


def detect_gastos_hormiga(gastos: pd.DataFrame, threshold: float = 15000.0, min_count: int = 5):
    """Detecta gastos pequeños y frecuentes por mes/categoría/proveedor/descripcion."""
    if gastos.empty:
        return pd.DataFrame(columns=["anio", "mes", "categoria", "proveedor", "descripcion", "transacciones", "total"])
    g = gastos.copy()
    g = add_period_columns(g, "fecha")
    g_small = g[g["monto"] <= threshold]
    grp = (
        g_small.groupby(["anio", "mes", "categoria", "proveedor", "descripcion"], as_index=False)
        .agg(transacciones=("monto", "count"), total=("monto", "sum"))
    )
    grp = grp[grp["transacciones"] >= min_count]
    grp.sort_values(["anio", "mes", "total"], ascending=[True, True, False], inplace=True)
    return grp

# ----------------------------
# Sidebar: navegación + filtros globales
# ----------------------------
PAGES = ["Dashboard", "Registrar", "Reportes", "Deudas", "Configurar", "Exportar"]
page = st.sidebar.radio("Navegación", PAGES)

st.sidebar.markdown("---")

# Filtros de periodo globales
periodo = st.sidebar.selectbox(
    "Periodicidad para gráficos",
    ["Mensual", "Diario", "Quincenal", "Trimestral", "Semestral", "Anual"],
    index=0,
)

# Parámetros de gastos hormiga
st.sidebar.markdown("### Gastos hormiga")
threshold_hormiga = st.sidebar.number_input("Umbral por transacción (<=)", min_value=1000.0, value=15000.0, step=1000.0)
min_count_hormiga = st.sidebar.number_input("Mín. repeticiones/mes", min_value=2, value=5, step=1)

st.sidebar.markdown("---")
st.sidebar.caption("Consejo: exporta a Excel regularmente para respaldo.")

# ----------------------------
# Dashboard
# ----------------------------
if page == "Dashboard":
    st.title("💸 Finanzas Personales – Dashboard")
    ingresos, gastos, deudas, inversiones, ahorros = load_all()

    # Conversión de fechas
    for df in [ingresos, gastos, deudas, inversiones, ahorros]:
        if not df.empty:
            df["fecha"] = pd.to_datetime(df["fecha"]) 

    total_ingresos = ingresos["monto"].sum() if not ingresos.empty else 0.0
    total_gastos = gastos["monto"].sum() if not gastos.empty else 0.0

    # Deuda: pagos (outflow) y desembolsos (inflow)
    desembolsos = deudas.loc[deudas["tipo"] == "Desembolso", "monto"].sum() if not deudas.empty else 0.0
    pagos_deuda = deudas.loc[deudas["tipo"] == "Pago", "monto"].sum() if not deudas.empty else 0.0

    total_ahorro = ahorros["monto"].sum() if not ahorros.empty else 0.0
    total_inversion = inversiones["monto"].sum() if not inversiones.empty else 0.0

    # Flujo de caja neto (cashflow)
    cashflow = total_ingresos + desembolsos - total_gastos - pagos_deuda

    # KPI Cards
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Ingresos", human_money(total_ingresos))
    c2.metric("Gastos", human_money(total_gastos))
    c3.metric("Pagos Deuda", human_money(pagos_deuda))
    c4.metric("Cashflow", human_money(cashflow))
    c5.metric("Ahorro + Inversión", human_money(total_ahorro + total_inversion))

    # Serie temporal ingresos vs gastos
    col1, col2 = st.columns(2)

    if not ingresos.empty:
        ag_i = aggregate_by_period(ingresos, "monto", "fecha", periodo)
        fig_i = px.line(ag_i, x="periodo", y="monto", title=f"Ingresos por periodo ({periodo})")
        col1.plotly_chart(fig_i, use_container_width=True)
    else:
        col1.info("Sin datos de ingresos aún.")

    if not gastos.empty:
        ag_g = aggregate_by_period(gastos, "monto", "fecha", periodo)
        fig_g = px.line(ag_g, x="periodo", y="monto", title=f"Gastos por periodo ({periodo})")
        col2.plotly_chart(fig_g, use_container_width=True)
    else:
        col2.info("Sin datos de gastos aún.")

    # Distribución de gastos e ingresos
    col3, col4 = st.columns(2)
    if not gastos.empty:
        by_cat_g = gastos.groupby("categoria", as_index=False)["monto"].sum().sort_values("monto", ascending=False)
        fig_gc = px.bar(by_cat_g, x="categoria", y="monto", title="Distribución de gastos por categoría")
        col3.plotly_chart(fig_gc, use_container_width=True)
    else:
        col3.info("Registra gastos para ver su distribución.")

    if not ingresos.empty:
        by_src = ingresos.groupby("fuente", as_index=False)["monto"].sum().sort_values("monto", ascending=False)
        fig_is = px.pie(by_src, names="fuente", values="monto", title="Fuentes principales de ingreso")
        col4.plotly_chart(fig_is, use_container_width=True)
    else:
        col4.info("Registra ingresos para ver las fuentes principales.")

    # Gastos hormiga
    st.markdown("### 🐜 Gastos hormiga detectados")
    gh = detect_gastos_hormiga(gastos, threshold_hormiga, min_count_hormiga)
    if gh.empty:
        st.success("Sin gastos hormiga detectados con los parámetros actuales.")
    else:
        st.dataframe(gh, use_container_width=True)

# ----------------------------
# Registrar movimientos
# ----------------------------
elif page == "Registrar":
    st.title("📝 Registrar movimientos")

    tab_ing, tab_gto, tab_deu, tab_inv, tab_aho = st.tabs([
        "Ingresos", "Gastos", "Deuda (mov.)", "Inversiones", "Ahorros"
    ])

    # Helper para categorías
    def load_categorias(tipo: str) -> list[str]:
        df = df_from_sql("SELECT nombre FROM categorias WHERE tipo = ? ORDER BY nombre", (tipo,))
        return df["nombre"].tolist()

    with tab_ing:
        with st.form("form_ingresos"):
            fecha = st.date_input("Fecha", value=date.today())
            fuente = st.text_input("Fuente (ej: Salario Empresa X)")
            categoria = st.selectbox("Categoría de ingreso", options=load_categorias("ingreso"))
            monto = st.number_input("Monto", min_value=0.0, step=1000.0)
            descripcion = st.text_area("Descripción", placeholder="Detalle opcional")
            submitted = st.form_submit_button("Agregar ingreso", use_container_width=True)
        if submitted:
            insert_row("ingresos", {
                "fecha": parse_date(fecha),
                "fuente": fuente.strip() or categoria,
                "categoria": categoria,
                "monto": float(monto),
                "descripcion": descripcion.strip(),
            })
            st.success("Ingreso registrado ✅")

    with tab_gto:
        with st.form("form_gastos"):
            fecha = st.date_input("Fecha", value=date.today(), key="g_fecha")
            categoria = st.selectbox("Categoría de gasto", options=load_categorias("gasto"))
            metodo = st.selectbox("Método de pago", options=["Efectivo", "Tarjeta débito", "Tarjeta crédito", "Transferencia", "Otro"])
            proveedor = st.text_input("Proveedor / Lugar (opcional)")
            monto = st.number_input("Monto", min_value=0.0, step=1000.0, key="g_monto")
            descripcion = st.text_area("Descripción", placeholder="Detalle opcional", key="g_desc")
            submitted = st.form_submit_button("Agregar gasto", use_container_width=True)
        if submitted:
            insert_row("gastos", {
                "fecha": parse_date(fecha),
                "categoria": categoria,
                "metodo": metodo,
                "proveedor": proveedor.strip(),
                "monto": float(monto),
                "descripcion": descripcion.strip(),
            })
            st.success("Gasto registrado ✅")

    with tab_deu:
        with st.form("form_deudas"):
            fecha = st.date_input("Fecha", value=date.today(), key="d_fecha")
            deuda = st.text_input("Nombre de la deuda (ej: Tarjeta X / Préstamo Y)")
            tipo = st.selectbox("Tipo de movimiento", options=["Desembolso", "Pago"]) 
            monto = st.number_input("Monto", min_value=0.0, step=1000.0, key="d_monto")
            descripcion = st.text_area("Descripción", placeholder="Detalle opcional", key="d_desc")
            submitted = st.form_submit_button("Agregar movimiento de deuda", use_container_width=True)
        if submitted:
            insert_row("deuda_movimientos", {
                "fecha": parse_date(fecha),
                "deuda": deuda.strip(),
                "tipo": tipo,
                "monto": float(monto),
                "descripcion": descripcion.strip(),
            })
            st.success("Movimiento de deuda registrado ✅")

    with tab_inv:
        with st.form("form_inversiones"):
            fecha = st.date_input("Fecha", value=date.today(), key="i_fecha")
            instrumento = st.text_input("Instrumento (ej: CDT, Fondo, Acciones)")
            monto = st.number_input("Monto", min_value=0.0, step=1000.0, key="i_monto")
            descripcion = st.text_area("Descripción", placeholder="Detalle opcional", key="i_desc")
            submitted = st.form_submit_button("Agregar inversión", use_container_width=True)
        if submitted:
            insert_row("inversiones", {
                "fecha": parse_date(fecha),
                "instrumento": instrumento.strip(),
                "monto": float(monto),
                "descripcion": descripcion.strip(),
            })
            st.success("Inversión registrada ✅")

    with tab_aho:
        with st.form("form_ahorros"):
            fecha = st.date_input("Fecha", value=date.today(), key="a_fecha")
            meta = st.text_input("Meta / Fondo (ej: Fondo de emergencia)")
            monto = st.number_input("Monto", min_value=0.0, step=1000.0, key="a_monto")
            descripcion = st.text_area("Descripción", placeholder="Detalle opcional", key="a_desc")
            submitted = st.form_submit_button("Agregar ahorro", use_container_width=True)
        if submitted:
            insert_row("ahorros", {
                "fecha": parse_date(fecha),
                "meta": meta.strip(),
                "monto": float(monto),
                "descripcion": descripcion.strip(),
            })
            st.success("Ahorro registrado ✅")

# ----------------------------
# Reportes
# ----------------------------
elif page == "Reportes":
    st.title("📈 Reportes")
    ingresos, gastos, deudas, inversiones, ahorros = load_all()

    # Rango de fechas
    min_date = pd.Timestamp("2020-01-01")
    max_date = pd.Timestamp.today()
    for df in [ingresos, gastos, deudas, inversiones, ahorros]:
        if not df.empty:
            df["fecha"] = pd.to_datetime(df["fecha"]) 
            min_date = min(min_date, df["fecha"].min())
            max_date = max(max_date, df["fecha"].max())

    d1, d2 = st.date_input(
        "Rango de fechas",
        value=(min_date.date(), max_date.date()),
    )

    def filter_range(df):
        if df.empty:
            return df
        m = (df["fecha"] >= pd.to_datetime(d1)) & (df["fecha"] <= pd.to_datetime(d2))
        return df[m]

    ingresos = filter_range(ingresos)
    gastos = filter_range(gastos)
    deudas = filter_range(deudas)
    inversiones = filter_range(inversiones)
    ahorros = filter_range(ahorros)

    # Cashflow por periodo
    ag_i = aggregate_by_period(ingresos, "monto", "fecha", periodo) if not ingresos.empty else pd.DataFrame(columns=["periodo","monto"])
    ag_g = aggregate_by_period(gastos, "monto", "fecha", periodo) if not gastos.empty else pd.DataFrame(columns=["periodo","monto"])

    # Deuda movimientos
    if not deudas.empty:
        ag_des = aggregate_by_period(deudas[deudas["tipo"]=="Desembolso"], "monto", "fecha", periodo)
        ag_pag = aggregate_by_period(deudas[deudas["tipo"]=="Pago"], "monto", "fecha", periodo)
    else:
        ag_des = pd.DataFrame(columns=["periodo","monto"]) 
        ag_pag = pd.DataFrame(columns=["periodo","monto"]) 

    # Unir para cashflow: ingresos + desembolsos - gastos - pagos
    all_periods = sorted(set(ag_i["periodo"]).union(ag_g["periodo"]).union(ag_des["periodo"]).union(ag_pag["periodo"]))
    cf = pd.DataFrame({"periodo": all_periods})
    for lbl, df_ in [("ingresos", ag_i), ("gastos", ag_g), ("desembolsos", ag_des), ("pagos_deuda", ag_pag)]:
        cf = cf.merge(df_.rename(columns={"monto": lbl}), on="periodo", how="left")
    cf = cf.fillna(0)
    cf["cashflow"] = cf["ingresos"] + cf["desembolsos"] - cf["gastos"] - cf["pagos_deuda"]

    c1, c2 = st.columns(2)
    if not cf.empty:
        c1.plotly_chart(px.line(cf, x="periodo", y=["ingresos", "gastos"], title=f"Ingresos vs Gastos ({periodo})"), use_container_width=True)
        c2.plotly_chart(px.line(cf, x="periodo", y=["cashflow"], title=f"Cashflow ({periodo})"), use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    if not gastos.empty:
        top_gastos = gastos.groupby("categoria", as_index=False)["monto"].sum().sort_values("monto", ascending=False).head(5)
        col1.subheader("Top 5 categorías de gasto")
        col1.dataframe(top_gastos, use_container_width=True)
        col1.plotly_chart(px.bar(top_gastos, x="categoria", y="monto", title="Top 5 gastos"), use_container_width=True)
    else:
        col1.info("No hay gastos en el rango.")

    if not ingresos.empty:
        top_ing = ingresos.groupby("fuente", as_index=False)["monto"].sum().sort_values("monto", ascending=False).head(5)
        col2.subheader("Top 5 fuentes de ingreso")
        col2.dataframe(top_ing, use_container_width=True)
        col2.plotly_chart(px.pie(top_ing, names="fuente", values="monto", title="Top 5 ingresos"), use_container_width=True)
    else:
        col2.info("No hay ingresos en el rango.")

    st.markdown("---")
    st.subheader("🐜 Gastos hormiga (en el rango)")
    gh = detect_gastos_hormiga(gastos, threshold_hormiga, min_count_hormiga)
    st.dataframe(gh, use_container_width=True)

# ----------------------------
# Deudas – estado por deuda
# ----------------------------
elif page == "Deudas":
    st.title("💳 Deudas – Estado por deuda")
    deudas = df_from_sql("SELECT * FROM deuda_movimientos")
    if not deudas.empty:
        deudas["fecha"] = pd.to_datetime(deudas["fecha"]) 
        estado = deudas.groupby("deuda", as_index=False).agg(
            desembolsado=("monto", lambda s: deudas.loc[(deudas["deuda"]==s.name) & (deudas["tipo"]=="Desembolso"), "monto"].sum()),
            pagado=("monto", lambda s: deudas.loc[(deudas["deuda"]==s.name) & (deudas["tipo"]=="Pago"), "monto"].sum()),
        )
        estado["saldo_pendiente"] = estado["desembolsado"] - estado["pagado"]
        st.dataframe(estado.sort_values("saldo_pendiente", ascending=False), use_container_width=True)
        st.plotly_chart(px.bar(estado, x="deuda", y="saldo_pendiente", title="Saldo pendiente por deuda"), use_container_width=True)
    else:
        st.info("Aún no hay movimientos de deuda.")

# ----------------------------
# Configurar – categorías, metas
# ----------------------------
elif page == "Configurar":
    st.title("⚙️ Configuración")

    st.subheader("Categorías personalizadas")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Agregar categoría de ingreso**")
        cat_i = st.text_input("Nombre nueva categoría (ingreso)")
        if st.button("Agregar ingreso"):
            try:
                insert_row("categorias", {"tipo": "ingreso", "nombre": cat_i.strip()})
                st.success("Categoría de ingreso agregada ✅")
            except Exception as e:
                st.error(f"No se pudo agregar: {e}")

    with col2:
        st.markdown("**Agregar categoría de gasto**")
        cat_g = st.text_input("Nombre nueva categoría (gasto)")
        if st.button("Agregar gasto"):
            try:
                insert_row("categorias", {"tipo": "gasto", "nombre": cat_g.strip()})
                st.success("Categoría de gasto agregada ✅")
            except Exception as e:
                st.error(f"No se pudo agregar: {e}")

    st.markdown("---")
    st.caption("Sugerencia: usa categorías simples al inicio y ve refinándolas.")

# ----------------------------
# Exportar – Excel con hojas por módulo y resúmenes
# ----------------------------
elif page == "Exportar":
    st.title("📤 Exportar a Excel")
    ingresos, gastos, deudas, inversiones, ahorros = load_all()

    # Preparar resumen por periodo
    ag_ing = aggregate_by_period(ingresos, "monto", "fecha", "Mensual") if not ingresos.empty else pd.DataFrame(columns=["periodo","monto"])
    ag_gto = aggregate_by_period(gastos, "monto", "fecha", "Mensual") if not gastos.empty else pd.DataFrame(columns=["periodo","monto"])

    if not deudas.empty:
        ag_des = aggregate_by_period(deudas[deudas["tipo"]=="Desembolso"], "monto", "fecha", "Mensual")
        ag_pag = aggregate_by_period(deudas[deudas["tipo"]=="Pago"], "monto", "fecha", "Mensual")
    else:
        ag_des = pd.DataFrame(columns=["periodo","monto"]) 
        ag_pag = pd.DataFrame(columns=["periodo","monto"]) 

    allp = sorted(set(ag_ing["periodo"]).union(ag_gto["periodo"]).union(ag_des["periodo"]).union(ag_pag["periodo"]))
    resumen = pd.DataFrame({"periodo": allp})
    for lbl, df_ in [("ingresos", ag_ing), ("gastos", ag_gto), ("desembolsos", ag_des), ("pagos_deuda", ag_pag)]:
        resumen = resumen.merge(df_.rename(columns={"monto": lbl}), on="periodo", how="left")
    resumen = resumen.fillna(0)
    resumen["cashflow"] = resumen["ingresos"] + resumen["desembolsos"] - resumen["gastos"] - resumen["pagos_deuda"]

    # Gastos hormiga (global)
    gh = detect_gastos_hormiga(gastos)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        resumen.to_excel(writer, index=False, sheet_name="ResumenMensual")
        if not ingresos.empty:
            ingresos.to_excel(writer, index=False, sheet_name="Ingresos")
        if not gastos.empty:
            gastos.to_excel(writer, index=False, sheet_name="Gastos")
        if not deudas.empty:
            deudas.to_excel(writer, index=False, sheet_name="DeudaMov")
        if not inversiones.empty:
            inversiones.to_excel(writer, index=False, sheet_name="Inversiones")
        if not ahorros.empty:
            ahorros.to_excel(writer, index=False, sheet_name="Ahorros")
        if not gh.empty:
            gh.to_excel(writer, index=False, sheet_name="GastosHormiga")

    st.download_button(
        label="Descargar Excel",
        data=buffer.getvalue(),
        file_name=f"finanzas_personales_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

# ---------------------------------------------------------------
# CONTENIDO PARA requirements.txt (cópialo a un archivo separado)
# ---------------------------------------------------------------
# streamlit==1.36.0
# pandas==2.2.2
# plotly==5.22.0
# openpyxl==3.1.5
# numpy==1.26.4
# """sqlite3 viene en la librería estándar de Python"""
