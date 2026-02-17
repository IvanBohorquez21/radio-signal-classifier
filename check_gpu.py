import torch

print("--- Verificación de Hardware ---")
if torch.cuda.is_available():
    print(f"✅ ¡Éxito! PyTorch reconoce tu GPU.")
    print(f"🎮 Tarjeta detectada: {torch.cuda.get_device_name(0)}")
    print(f"💾 Memoria VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("❌ No se detectó la GPU. Verifica la instalación de PyTorch.")