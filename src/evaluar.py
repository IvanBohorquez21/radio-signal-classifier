import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os

# --- 1. CONFIGURACIÓN ---
# Ajustamos rutas para que funcionen desde la raíz del proyecto
PATH_PROCESSED = "data/processed/RML2016_limpio.pt"
MODEL_PATH = "models/signal_model_v3.pth"
IMG_SAVE_PATH = "img/matriz_confusion_v3.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. CARGA DE DATOS ---
if not os.path.exists(PATH_PROCESSED):
    print(f"❌ Error: No se encuentra {PATH_PROCESSED}")
    exit()

print(f"📦 Cargando datos procesados...")
checkpoint = torch.load(PATH_PROCESSED)
X, lbl, mods = checkpoint['X'], checkpoint['lbl'], checkpoint['mods']

# Creamos el loader de prueba (Test)
dataset = TensorDataset(X, lbl)
# Usamos el mismo split que en el entrenamiento para ser coherentes
_, test_ds = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))])
test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False)

# --- 3. DEFINICIÓN DEL MODELO ---
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

# --- 4. EVALUACIÓN ---
print(f"🧠 Cargando modelo desde {MODEL_PATH}...")
model = SignalClassifier(num_classes=len(mods)).to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

y_true, y_pred = [], []
print("🚀 Realizando inferencia...")

with torch.no_grad():
    for signals, labels in test_loader:
        outputs = model(signals.to(device))
        y_pred.extend(torch.argmax(outputs, 1).cpu().numpy())
        y_true.extend(labels.numpy())

# --- 5. RESULTADOS Y GUARDADO ---
os.makedirs("img", exist_ok=True)

# Generar Matriz de Confusión
cm = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(12, 10))
sns.heatmap(cm_norm, annot=True, fmt='.2f', xticklabels=mods, yticklabels=mods, cmap='Greens')
plt.title("Matriz de Confusión - Modelo V3")
plt.xlabel("Predicción")
plt.ylabel("Real")

plt.savefig(IMG_SAVE_PATH, dpi=300, bbox_inches='tight')
print(f"✅ Matriz guardada en: {IMG_SAVE_PATH}")

print("\n📝 REPORTE DE CLASIFICACIÓN:")
print(classification_report(y_true, y_pred, target_names=mods))

plt.show()