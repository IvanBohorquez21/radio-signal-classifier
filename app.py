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

# --- 2. CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="RF Classifier Dashboard", layout="wide")
st.title("📡 Dashboard de Clasificación de Señales RF (V3)")

# --- 3. CARGA DE ACTIVOS (Rutas Optimizadas para Cloud) ---
@st.cache_resource
def load_assets():
    # En Streamlit Cloud, las rutas son relativas a la raíz del repo
    data_path = "data/processed/RML2016_limpio.pt"
    model_path = "models/signal_model_v3.pth"
    
    # Verificación de existencia para evitar errores silenciosos
    if not os.path.exists(data_path) or not os.path.exists(model_path):
        return None
        
    try:
        # Cargamos siempre en CPU para el servidor
        checkpoint = torch.load(data_path, map_location=torch.device('cpu'))
        weights = torch.load(model_path, map_location=torch.device('cpu'))
        return {"checkpoint": checkpoint, "weights": weights}
    except Exception as e:
        st.error(f"Error al cargar archivos: {e}")
        return None

assets = load_assets()

if assets is None:
    st.error("❌ No se encontraron los archivos necesarios.")
    st.info("Asegúrate de que 'data/processed/RML2016_limpio.pt' y 'models/signal_model_v3.pth' estén en tu repositorio de GitHub.")
else:
    X = assets['checkpoint']['X']
    lbl = assets['checkpoint']['lbl']
    mods = assets['checkpoint']['mods']
    
    model = SignalClassifier(num_classes=len(mods))
    model.load_state_dict(assets['weights'])
    model.eval()

    # --- 4. INTERFAZ DE USUARIO ---
    st.sidebar.header("🕹️ Selección de Señal")
    idx = st.sidebar.slider("Índice de muestra", 0, len(X)-1, 100)
    
    col1, col2 = st.columns(2)
    ALTURA_COMUN = 5 

    signal = X[idx]
    with torch.no_grad():
        output = model(signal.unsqueeze(0))
        probs = F.softmax(output, dim=1).numpy()[0]
        pred_idx = np.argmax(probs)
    
    with col1:
        st.subheader("📊 Dominio del Tiempo (I/Q)")
        fig_time, ax_time = plt.subplots(figsize=(7, ALTURA_COMUN))
        ax_time.plot(signal[0], label="Fase (I)", color="#1f77b4", lw=1.5)
        ax_time.plot(signal[1], label="Cuadratura (Q)", color="#ff7f0e", lw=1.5)
        ax_time.set_title(f"Clase Real: {mods[lbl[idx]]}", fontsize=12, fontweight='bold')
        ax_time.set_xlabel("Muestras")
        ax_time.set_ylabel("Amplitud")
        ax_time.legend(loc='upper right')
        ax_time.grid(True, alpha=0.3)
        st.pyplot(fig_time, use_container_width=True)

    with col2:
        st.subheader("🎯 Predicción del Modelo")
        fig_prob, ax_prob = plt.subplots(figsize=(7, ALTURA_COMUN))
        colors = ['#2ca02c' if i == pred_idx else '#aec7e8' for i in range(len(mods))]
        
        ax_prob.barh(mods, probs, color=colors)
        ax_prob.set_xlim(0, 1.0)
        ax_prob.set_title(f"Predicho: {mods[pred_idx]} ({probs[pred_idx]*100:.1f}%)", fontsize=12, fontweight='bold')
        ax_prob.set_xlabel("Probabilidad")
        plt.tight_layout()
        st.pyplot(fig_prob, use_container_width=True)

    # --- 5. RESULTADO FINAL ---
    st.divider()
    es_correcta = (pred_idx == lbl[idx])
    
    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        if es_correcta:
            st.metric("Resultado", "CORRECTO", delta="Match", delta_color="normal")
        else:
            st.metric("Resultado", "ERROR", delta="Mismatch", delta_color="inverse")

    with res_col2:
        if es_correcta:
            st.success(f"✅ El modelo identificó correctamente la señal como **{mods[pred_idx]}**.")
        else:
            st.error(f"❌ El modelo predijo **{mods[pred_idx]}**, pero la realidad es **{mods[lbl[idx]]}**.")

    with st.expander("Detalles Técnicos"):
        st.write(f"Clase Real ID: `{lbl[idx]}`")
        st.write(f"Probabilidad Máxima: `{np.max(probs):.4f}`")
        st.write(f"Ruta de datos: `{os.path.abspath('data/processed/RML2016_limpio.pt')}`")