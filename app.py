import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 1. ARQUITECTURA DEL MODELO ---
class SignalClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SignalClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(1, 7), padding=(0, 3))
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=(2, 7), padding=(0, 3))
        self.bn2 = nn.BatchNorm2d(64)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 128, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# --- 2. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="RF Classifier Dashboard", 
    page_icon="📡", # Aquí pones el emoji que prefieras
    layout="wide"
)

# Estilo CSS para mejorar la estética de los contenedores
st.markdown("""
    <style>
    .stMetric { background-color: #1e2530; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    .stAlert { border-radius: 10px; }
    [data-testid="stSidebar"] { background-color: #161b22; }
    </style>
    """, unsafe_allow_html=True)

st.title("📡 Dashboard de Clasificación de Señales RF")

# --- 3. CARGA DE ACTIVOS ---
@st.cache_resource
def load_assets():
    data_path = "data/processed/RML2016_limpio.pt"
    model_path = "models/signal_model_v3.pth"
    
    if not os.path.exists(data_path) or not os.path.exists(model_path):
        return None
        
    try:
        checkpoint = torch.load(data_path, map_location=torch.device('cpu'))
        weights = torch.load(model_path, map_location=torch.device('cpu'))
        return {"checkpoint": checkpoint, "weights": weights}
    except Exception as e:
        st.error(f"Error cargando archivos: {e}")
        return None

assets = load_assets()

if assets is None:
    st.error("❌ No se encontraron los archivos necesarios en el repositorio.")
    st.info("Asegúrate de que 'data/processed/RML2016_limpio.pt' esté en GitHub.")
else:
    X = assets['checkpoint']['X']
    lbl = assets['checkpoint']['lbl']
    mods = assets['checkpoint']['mods']
    
    model = SignalClassifier(num_classes=len(mods))
    model.load_state_dict(assets['weights'])
    model.eval()

    # --- 4. BARRA LATERAL ---
    st.sidebar.header("🕹️ Control de Señal")
    idx = st.sidebar.slider("Seleccionar índice de muestra", 0, len(X)-1, 100)
    
    # Explicación de la Arquitectura
    with st.sidebar.expander("🔬 Detalles de la Arquitectura"):
        st.markdown("""
        **CNN SignalClassifier V3**
        Esta red procesa señales I/Q de 2 canales x 128 muestras.
        
        **Capas:**
        - **Conv2D (128):** Filtros de 1x7 para extraer patrones temporales.
        - **BatchNorm:** Normalización de activaciones.
        - **Conv2D (64):** Reducción de dimensionalidad y abstracción.
        - **Dropout (0.5):** Regularización para evitar sobreajuste.
        - **Softmax:** Clasificación en 10 tipos de modulación.
        """)
    
    st.sidebar.divider()
    st.sidebar.info(f"Total de muestras: {len(X)}")

    # --- 5. PROCESAMIENTO E INFERENCIA ---
    signal = X[idx]
    with torch.no_grad():
        output = model(signal.unsqueeze(0))
        probs = F.softmax(output, dim=1).numpy()[0]
        pred_idx = np.argmax(probs)
    
    col1, col2 = st.columns(2)
    ALTURA = 5

    with col1:
        st.subheader("📊 Dominio del Tiempo (I/Q)")
        fig_time, ax_time = plt.subplots(figsize=(7, ALTURA))
        fig_time.patch.set_facecolor('white') # Fondo claro para contraste
        
        ax_time.plot(signal[0], label="Fase (I)", color="#1f77b4", lw=1.5)
        ax_time.plot(signal[1], label="Cuadratura (Q)", color="#ff7f0e", lw=1.5)
        ax_time.set_title(f"Clase Real: {mods[lbl[idx]]}", fontweight='bold')
        ax_time.set_xlabel("Muestras")
        ax_time.set_ylabel("Amplitud")
        ax_time.legend(loc='upper right')
        ax_time.grid(True, alpha=0.3)
        fig_time.tight_layout()
        st.pyplot(fig_time, use_container_width=True)

    with col2:
        st.subheader("🎯 Probabilidades del Modelo")
        fig_prob, ax_prob = plt.subplots(figsize=(7, ALTURA))
        fig_prob.patch.set_facecolor('white')
        
        colors = ['#2ca02c' if i == pred_idx else '#aec7e8' for i in range(len(mods))]
        ax_prob.barh(mods, probs, color=colors)
        ax_prob.set_xlim(0, 1.0)
        ax_prob.set_title(f"Predicho: {mods[pred_idx]} ({probs[pred_idx]*100:.1f}%)", fontweight='bold')
        ax_prob.set_xlabel("Confianza")
        fig_prob.tight_layout()
        st.pyplot(fig_prob, use_container_width=True)

    # --- 6. PANEL DE RESULTADOS ---
    st.divider()
    es_correcta = (pred_idx == lbl[idx])
    
    res_c1, res_c2 = st.columns([1, 2])
    with res_c1:
        status = "CORRECTO" if es_correcta else "ERROR"
        color_delta = "normal" if es_correcta else "inverse"
        st.metric("Validación", status, delta="Match" if es_correcta else "Mismatch", delta_color=color_delta)

    with res_c2:
        if es_correcta:
            st.success(f"### ✅ Identificación exitosa: **{mods[pred_idx]}**")
        else:
            st.error(f"### ❌ Error de Clasificación: Predicho **{mods[pred_idx]}** vs Real **{mods[lbl[idx]]}**")

    with st.expander("🔍 Ver detalles del vector de entrada (Tensores)"):
        st.write("Muestra procesada del dataset:")
        st.dataframe(signal.numpy(), use_container_width=True)