# -*- coding: utf-8 -*-
"""
Treino e utilidades do classificador de sementes.

• Transfer-learning (ResNet-18 pré-treinada ImageNet)
• Split treino/val usando **dois** ImageFolder independentes
• Balanceamento de classes via WeightedRandomSampler
• Early-Stopping + ReduceLROnPlateau
• Função load_trained_model() – carrega ou treina se não existir best_model.pth
"""

import os, time, json, random
from pathlib import Path

import torch, torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import transforms, datasets
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import classification_report, confusion_matrix

# -------------------- Configurações --------------------
data_dir      = r'C:\Users\faculdade\.cache\kagglehub\datasets\warcoder\soyabean-seeds\versions\2\Soybean Seeds'
MODEL_PATH    = 'best_model.pth'
CLASS_NAMES_JSON = 'class_names.json'

IMG_SIZE      = 224
BATCH_SIZE    = 32
EPOCHS        = 25           # máx. (early-stop interrompe antes)
HEAD_EPOCHS   = 3            # quantas épocas só com FC treinável
LR_HEAD       = 1e-3
LR_FINE       = 1e-4
PATIENCE      = 5            # early-stopping
NUM_WORKERS   = 0            # coloque >0 se estiver em Linux

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed   = 42
random.seed(seed); torch.manual_seed(seed)

# -------------------- Transforms -----------------------
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

transform_val = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# -------------------------------------------------------
def _build_dataloaders():
    """Cria datasets independentes (evita bug de transform)."""
    full_ds_train = datasets.ImageFolder(data_dir, transform=transform_train)
    full_ds_val   = datasets.ImageFolder(data_dir, transform=transform_val)

    assert full_ds_train.classes == full_ds_val.classes
    class_names = full_ds_train.classes
    num_classes = len(class_names)

    # split estável
    indices = list(range(len(full_ds_train)))
    random.shuffle(indices)
    split_idx = int(0.8 * len(indices))
    idx_train, idx_val = indices[:split_idx], indices[split_idx:]

    train_ds = Subset(full_ds_train, idx_train)
    val_ds   = Subset(full_ds_val,   idx_val)

    # -------------- balanceamento (sampler) --------------
    targets = [full_ds_train.imgs[i][1] for i in idx_train]
    class_counts = torch.bincount(torch.tensor(targets))
    class_weights = 1.0 / class_counts.float()
    sample_weights = torch.tensor([class_weights[t] for t in targets])

    sampler = WeightedRandomSampler(sample_weights,
                                    num_samples=len(sample_weights),
                                    replacement=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=NUM_WORKERS,
                              pin_memory=True)

    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS,
                              pin_memory=True)

    return train_loader, val_loader, class_names, num_classes

# -------------------------------------------------------
def _build_model(num_classes):
    """ResNet-18 pré-treinada + troca do FC."""
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    for p in model.parameters():
        p.requires_grad = False            # congela tudo
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)

# -------------------------------------------------------
def _train_model():
    train_loader, val_loader, class_names, num_classes = _build_dataloaders()
    model = _build_model(num_classes)

    # critérios e otimizadores
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR_HEAD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=2, factor=0.5)

    best_val_acc = 0
    patience_counter = 0

    for epoch in range(EPOCHS):
        t0 = time.time()
        model.train()
        running_loss, correct = 0.0, 0

        # ------------------- treino -------------------
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            correct += (out.argmax(1) == yb).sum().item()

        train_loss = running_loss / len(train_loader.dataset)
        train_acc  = correct      / len(train_loader.dataset)

        # ------------------- validação ----------------
        model.eval()
        val_correct, val_loss_sum = 0, 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                val_loss = criterion(out, yb)
                val_loss_sum += val_loss.item() * xb.size(0)
                val_correct += (out.argmax(1) == yb).sum().item()

        val_loss = val_loss_sum / len(val_loader.dataset)
        val_acc  = val_correct  / len(val_loader.dataset)
        scheduler.step(val_loss)

        dur = time.time() - t0
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
              f"trn_loss {train_loss:.3f} acc {train_acc:.2%} | "
              f"val_loss {val_loss:.3f} acc {val_acc:.2%} | {dur:.1f}s")

        # ----------- unfreeze após HEAD_EPOCHS --------
        if epoch+1 == HEAD_EPOCHS:
            for p in model.parameters():
                p.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), lr=LR_FINE)
            print(">> Backbone descongelado (fine-tuning) <<")

        # ---------------- early-stopping --------------
        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            patience_counter = 0
            print("  * modelo salvo")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping.")
                break

    # ---------------- relatório final ----------------
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            out = model(xb)
            y_true.extend(yb)
            y_pred.extend(out.argmax(1).cpu())
    print('\n==== Classification Report ====')
    print(classification_report(y_true, y_pred, target_names=class_names))
    cm = confusion_matrix(y_true, y_pred)
    print('Confusion-matrix:\n', cm)

    # salva class names em json (para scripts externos)
    with open(CLASS_NAMES_JSON, 'w', encoding='utf-8') as f:
        json.dump(class_names, f, ensure_ascii=False)

    return model, transform_val, class_names

# -------------------------------------------------------
def load_trained_model():
    """
    Retorna (model, transform_val, class_names, device, IMG_SIZE)
    Se best_model.pth não existir, dispara o treino.
    """
    if not os.path.exists(MODEL_PATH):
        print(">> best_model.pth não encontrado – iniciando treino...")
        model, transform, class_names = _train_model()
    else:
        # pega class names salvos
        if os.path.exists(CLASS_NAMES_JSON):
            with open(CLASS_NAMES_JSON, 'r', encoding='utf-8') as f:
                class_names = json.load(f)
        else:
            # fallback: abre ImageFolder só pra pegar nomes
            class_names = datasets.ImageFolder(data_dir).classes
        num_classes = len(class_names)
        model = _build_model(num_classes)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        transform = transform_val
        print(f">> Modelo carregado de '{MODEL_PATH}'.")
    return model, transform, class_names, device, IMG_SIZE

# -------------------------------------------------------
if __name__ == '__main__':
    # executa treino explícito se rodado direto
    _train_model()