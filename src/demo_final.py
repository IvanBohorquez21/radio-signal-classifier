import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 1. CONFIGURACIÓN Y CARGA ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATH_PROCESSED = "data/processed/RML2016_limpio.pt"
MODEL_PATH = "models/signal_model_v3.pth"
IMG_SAVE_DIR = "img"

# Cargar datos procesados
if not os.path.exists(PATH_PROCESSED):
    print(f"❌ Error: No se encuentra {PATH_PROCESSED}")
    exit()

checkpoint = torch.load(PATH_PROCESSED)
X, lbl, mods = checkpoint['X'], checkpoint['lbl'], checkpoint['mods']

# --- 2. DEFINICIÓN DEL MODELO (Idéntico a train.ipynb) ---
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

# Cargar modelo entrenado
model = SignalClassifier(num_classes=len(mods)).to(device)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    print(f"✅ Modelo cargado: {MODEL_PATH}")
else:
    print(f"❌ No se encontró el modelo en {MODEL_PATH}")
    exit()

# --- 3. TOMAR MUESTRA ALEATORIA ---
# Creamos un loader temporal para sacar una muestra
dataset = TensorDataset(X, lbl)
loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
signals, labels = next(iter(loader))

# Elegimos una al azar
idx = np.random.randint(0, len(signals))
signal_raw = signals[idx]
label_real = labels[idx].item()

# Preparar para inferencia
signal_input = signal_raw.unsqueeze(0).to(device)

# --- 4. INFERENCIA ---
with torch.no_grad():
    output = model(signal_input)
    probs = F.softmax(output, dim=1).cpu().numpy()[0]
    pred_idx = np.argmax(probs)

# --- 5. VISUALIZACIÓN "PRO" ---
plt.figure(figsize=(15, 6))

# Subplot 1: Señal en el tiempo (I/Q)
plt.subplot(1, 2, 1)
signal_np = signal_raw.numpy()
plt.plot(signal_np[0], label="Componente I (Fase)", color='#1f77b4', alpha=0.8)
plt.plot(signal_np[1], label="Componente Q (Cuadratura)", color='#ff7f0e', alpha=0.8)
plt.title(f"Señal Real: {mods[label_real]}", fontsize=14, fontweight='bold')
plt.xlabel("Muestras de Tiempo")
plt.ylabel("Amplitud")
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.6)

# Subplot 2: Probabilidades (Barras)
plt.subplot(1, 2, 2)
colors = ['#aec7e8'] * len(mods) # Azul claro por defecto
# Si acertó: verde. Si falló: rojo, y marcamos la correcta en gris.
if pred_idx == label_real:
    colors[pred_idx] = '#2ca02c' # Verde
else:
    colors[pred_idx] = '#d62728' # Rojo
    colors[label_real] = '#7f7f7f' # Gris (la que debió ser)

bars = plt.bar(mods, probs, color=colors)
plt.xticks(rotation=45)
plt.title(f"Predicción IA: {mods[pred_idx]} ({probs[pred_idx]*100:.1f}% confianza)", 
          fontsize=14, color=colors[pred_idx], fontweight='bold')
plt.ylabel("Probabilidad")
plt.ylim(0, 1.1)

# Añadir etiquetas de valor sobre las barras
for bar in bars:
    yval = bar.get_height()
    if yval > 0.05: # Solo mostrar si la probabilidad es relevante
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()

# Guardar el resultado en img/
os.makedirs(IMG_SAVE_DIR, exist_ok=True)
SAVE_NAME = f"{IMG_SAVE_DIR}/demo_prediccion_{mods[label_real]}.png"
plt.savefig(SAVE_NAME, dpi=300)
print(f"📊 Demo guardado como: {os.path.abspath(SAVE_NAME)}")

plt.show()