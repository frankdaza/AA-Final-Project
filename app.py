import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Configuración de la página
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predicción de Precio de Propiedad",
    page_icon="🏡",
    layout="centered"
)

st.markdown(
    """
    <style>
        .stApp {
            background-image: linear-gradient(to right, #fefefe, #e2f0ff);
            background-attachment: fixed;
        }

        h1 {
            color: #003366;
            font-size: 2.8em;
            text-align: center;
        }

        .stNumberInput > div > div > input {
            background-color: #ffffff !important;
        }

        .stForm {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }

        section {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Predicción de Precio de Propiedades 🏠")

with st.expander("📊 ¿Cómo se calcula esta predicción?"):
    st.markdown("""
    - El modelo fue entrenado con datos históricos de propiedades en Argentina.
    - Utiliza un algoritmo de regresión y considera tanto variables numéricas como categóricas.
    """)

# ──────────────────────────────────────────────────────────────────────────────
# Cargar modelo
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_path: str = "xgb_model_price.pkl"):
    path = Path(model_path)
    if not path.exists():
        st.error(f"Modelo no encontrado en {model_path}")
        st.stop()
    with open(path, "rb") as f:
        model_loaded = pickle.load(f)
        return model_loaded

model = load_model()

# ──────────────────────────────────────────────────────────────────────────────
# Cargar el CSV de ciudades y vecindarios
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_ubicaciones(path="df_limpio.csv"):
    df = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
    df = df.rename(columns={"neighbourhood": "neighborhood"})  # por si acaso
    return df[["city", "neighborhood"]].dropna().drop_duplicates()

df_ubicaciones = load_ubicaciones()


# ─────── Selector dinámico CIUDAD y VECINDARIO (fuera del form) ───────
st.markdown("## 🗺️ Formulario")

col_city, col_neigh = st.columns(2)
with col_city:
    city = st.selectbox("Ciudad", sorted(df_ubicaciones["city"].dropna().unique()))
with col_neigh:
    neighborhoods = df_ubicaciones[df_ubicaciones["city"] == city]["neighborhood"].dropna().unique()
    neighborhood = st.selectbox("Vecindario", sorted(neighborhoods))


# ──────────────────────────────────────────────────────────────────────────────
# Formulario - Entrada de datos
# ──────────────────────────────────────────────────────────────────────────────
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        lat = st.number_input("Latitud", value=-34.60, format="%.6f")
        province = st.selectbox("Provincia", ["Bs.As. G.B.A. Zona Norte", "Bs.As. G.B.A. Zona Sur", "Bs.As. G.B.A. Zona Oeste", "Capital Federal"])

        properaty_type = st.selectbox("Tipo de Propiedad", ["Casa", "Departamento", "Oficina", "PH"])
        rooms = st.number_input("Ambientes (rooms)", min_value=0.0, value=3.0, step=1.0)

    with col2:
        lon = st.number_input("Longitud", value=-58.42, format="%.6f")
        bedrooms = st.number_input("Dormitorios (bedrooms)", min_value=0.0, value=2.0, step=1.0)
        bathrooms = st.number_input("Baños (bathrooms)", min_value=0.0, value=1.0, step=1.0)
        total_surface = st.number_input("Superficie total (m²)", min_value=0.0, value=100.0, step=1.0)
        covered_surface = st.number_input("Superficie cubierta (m²)", min_value=0.0, value=80.0, step=1.0)

    submitted = st.form_submit_button("Predecir Precio")

# ──────────────────────────────────────────────────────────────────────────────
# Inferencia y resultado
# ──────────────────────────────────────────────────────────────────────────────
if submitted:
    input_data = pd.DataFrame([{
        "lat": lat,
        "lon": lon,
        "province": province,
        "city": city,
        "neighborhood": neighborhood,
        "rooms": rooms,
        "bathrooms": bathrooms,
        "bedrooms": bedrooms,
        "total_surface": total_surface,
        "covered_surface": covered_surface,
        "properaty_type": properaty_type
    }])

    df_limpio = pd.read_csv("df_limpio.csv")
    df_limpio = df_limpio.rename(columns={"neighbourhood": "neighborhood"})

    # ─────── Rankings ───────
    precio_prom_city = df_limpio.groupby('neighborhood')['price'].mean().sort_values(ascending=False)
    ranking_neighborhood = {city: rank for rank, city in enumerate(precio_prom_city.index, start=1)}
    input_data['neighborhood_rank'] = input_data['neighborhood'].map(ranking_neighborhood).fillna(len(ranking_neighborhood) + 1)

    precio_prom_city = df_limpio.groupby('city')['price'].mean().sort_values(ascending=False)
    ranking_city = {city: rank for rank, city in enumerate(precio_prom_city.index, start=1)}
    input_data['city_rank'] = input_data['city'].map(ranking_city).fillna(len(ranking_city) + 1)

    # ─────── Provincias one-hot ───────
    df_provincias = pd.get_dummies(df_limpio[['province']], drop_first=False, dtype=int)
    expected_province_cols = df_provincias.columns.tolist()

    input_prov = pd.get_dummies(input_data[['province']], drop_first=False, dtype=int)
    for col in expected_province_cols:
        if col not in input_prov.columns:
            input_prov[col] = 0
    input_prov = input_prov[expected_province_cols]

    # ─────── Codificar properaty_type manualmente ───────
    property_map = {
        "Casa": 0,
        "Departamento": 1,
        "Oficina": 2,
        "PH": 3
    }

    # ─────── Combinar y seleccionar columnas finales ───────
    input_data_final = pd.concat([
        input_data.drop(columns=['province', 'city', 'neighborhood', 'properaty_type']),
        input_prov
    ], axis=1)

    final_columns = [
        'lat', 'lon', 'rooms', 'bedrooms', 'bathrooms', 'total_surface',
        'covered_surface', 'properaty_type_encoded', 'neighborhood_rank',
        'city_rank', 'province_Bs.As. G.B.A. Zona Norte',
        'province_Bs.As. G.B.A. Zona Oeste', 'province_Bs.As. G.B.A. Zona Sur',
        'province_Capital Federal'
    ]

    for col in final_columns:
        if col not in input_data_final.columns:
            input_data_final[col] = 0

    input_data_final = input_data_final[final_columns]

    # ─────── Predicción ───────
    try:
        predicted_price = model.predict(input_data_final.to_numpy())[0]
    except Exception as e:
        st.error(f"Error al realizar la predicción: {e}")
        st.stop()

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=predicted_price,
        title={'text': "Precio estimado (USD)"},
        gauge={
            'axis': {'range': [0, 1000000]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 100000], 'color': '#f2d7d5'},
                {'range': [100000, 300000], 'color': '#f9e79f'},
                {'range': [300000, 600000], 'color': '#aed6f1'},
                {'range': [600000, 1000000], 'color': '#a3e4d7'}
            ]
        }
    ))

    st.plotly_chart(fig_gauge)

    with st.expander("Ver datos de entrada usados ➤"):
        st.write(input_data_final)
