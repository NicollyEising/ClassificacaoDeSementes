# identificarSementesModelo.py
# ------------------------------------------------------------
#  Train & load – versão 2024-06-12
# ------------------------------------------------------------
import os, time, json, random
from pathlib import Path
from collections import Counter

import torch, torch.nn as nn
from torch.utils.data      import DataLoader, Subset, WeightedRandomSampler
from torchvision           import datasets, transforms
from torchvision.models    import resnet18, ResNet18_Weights
from sklearn.metrics       import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# ---------- paths ----------
DATA_DIR   = r'C:\Users\faculdade\.cache\kagglehub\datasets\warcoder\soyabean-seeds\versions\2\Soybean Seeds'
MODEL_PATH = 'best_model.pth'
NAMES_JSON = 'class_names.json'

# ---------- hiperparâmetros ----------
IMG_SIZE      = 224
BATCH_SIZE    = 32
EPOCHS        = 40
LR_MAX        = 3e-4            # pico do One-Cycle
WEIGHT_DECAY  = 1e-4
EARLY_STOP    = 8               # paciência em épocas
SEED          = 42
NUM_WORKERS   = 0               # >0 no Linux acelera

torch.manual_seed(SEED); random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------- transforms ----------
tf_train = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(25),
    transforms.ColorJitter(0.25, 0.25, 0.25, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])
tf_val = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])


# ═════════════════ helpers ═════════════════
def build_datasets():
    ds_full_train = datasets.ImageFolder(DATA_DIR, transform=tf_train)
    ds_full_val   = datasets.ImageFolder(DATA_DIR, transform=tf_val)

    assert ds_full_train.classes == ds_full_val.classes
    class_names = ds_full_train.classes
    num_classes = len(class_names)

    # stratified split 80/20
    indices = list(range(len(ds_full_train)))
    labels  = [ds_full_train.imgs[i][1] for i in indices]
    random.shuffle(indices)

    split = int(0.8*len(indices))
    idx_train, idx_val = indices[:split], indices[split:]

    ds_train = Subset(ds_full_train, idx_train)
    ds_val   = Subset(ds_full_val,   idx_val)

    # ───── mostra distribuição ─────
    cnt_train = Counter([labels[i] for i in idx_train])
    cnt_val   = Counter([labels[i] for i in idx_val])
    print('\nDistribuição de classes (treino):')
    for i,n in enumerate(class_names):
        print(f'  {n:<18}: {cnt_train[i]}')
    print('Distribuição (val):')
    for i,n in enumerate(class_names):
        print(f'  {n:<18}: {cnt_val[i]}')

    # ───── sampler balanceado ─────
    weights = torch.tensor([1.0/cnt_train[i] for i in labels], dtype=torch.double)
    sampler_weights = weights[idx_train]
    sampler = WeightedRandomSampler(sampler_weights,
                                    num_samples=len(idx_train),
                                    replacement=True)

    train_ld = DataLoader(ds_train, batch_size=BATCH_SIZE,
                          sampler=sampler, num_workers=NUM_WORKERS,
                          pin_memory=True)
    val_ld   = DataLoader(ds_val,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=NUM_WORKERS,
                          pin_memory=True)
    return train_ld, val_ld, class_names, num_classes


def build_model(num_classes):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


def train():
    train_ld, val_ld, class_names, num_classes = build_datasets()
    model = build_model(num_classes)

    # pesos para a CrossEntropy (inverso da freq.)
    targets = [y for _,y in datasets.ImageFolder(DATA_DIR).imgs]
    freq = torch.bincount(torch.tensor(targets, dtype=torch.int64))
    ce_weights = (1.0 / freq.float()).to(device)

    criterion = nn.CrossEntropyLoss(weight=ce_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX,
                                  weight_decay=WEIGHT_DECAY)
    steps_per_epoch = len(train_ld)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR_MAX, epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch, pct_start=0.3)

    best_acc, patience = 0, 0
    for epoch in range(1, EPOCHS+1):
        t0 = time.time()
        model.train(); running_loss = 0; correct = 0

        for xb,yb in train_ld:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb); loss = criterion(out, yb)
            loss.backward(); optimizer.step(); scheduler.step()

            running_loss += loss.item()*xb.size(0)
            correct += (out.argmax(1)==yb).sum().item()

        train_loss = running_loss/len(train_ld.dataset)
        train_acc  = correct     /len(train_ld.dataset)

        # ---- validação ----
        model.eval(); val_loss, val_correct = 0,0
        with torch.no_grad():
            for xb,yb in val_ld:
                xb,yb = xb.to(device), yb.to(device)
                out = model(xb); loss = criterion(out,yb)
                val_loss += loss.item()*xb.size(0)
                val_correct += (out.argmax(1)==yb).sum().item()
        val_loss /= len(val_ld.dataset)
        val_acc  = val_correct/ len(val_ld.dataset)

        print(f'Ep {epoch:02d}/{EPOCHS}  '
              f'trn_loss {train_loss:.3f} acc {train_acc:.2%} | '
              f'val_loss {val_loss:.3f} acc {val_acc:.2%}  '
              f'({time.time()-t0:.1f}s)')

        # ---- early stop ----
        if val_acc > best_acc + 1e-4:
            best_acc = val_acc; patience = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print('  * modelo salvo')
        else:
            patience += 1
            if patience >= EARLY_STOP:
                print('Early-stopping acionado.')
                break

    # -------- relatório final --------
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval(); y_true,y_pred=[],[]
    with torch.no_grad():
        for xb,yb in val_ld:
            out = model(xb.to(device))
            y_true.extend(yb); y_pred.extend(out.argmax(1).cpu())
    print('\nClassification report:')
    print(classification_report(y_true,y_pred, target_names=class_names))
    cm = confusion_matrix(y_true,y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.title('Matriz de Confusão'); plt.xlabel('Predito'); plt.ylabel('Verdadeiro')
    plt.tight_layout(); plt.show()

    # salva nomes
    with open(NAMES_JSON,'w',encoding='utf-8') as f:
        json.dump(class_names,f,ensure_ascii=False)

    return model, tf_val, class_names


# ═════════════════ interface externa ═════════════════
def load_trained_model():
    """
    Retorna (model, transform_val, class_names, device, IMG_SIZE).
    Treina se best_model.pth não existir.
    """
    if not os.path.exists(MODEL_PATH):
        print('> best_model.pth ausente – iniciando treino...')
        model, transform, names = train()
    else:
        if os.path.exists(NAMES_JSON):
            with open(NAMES_JSON,'r',encoding='utf-8') as f:
                names = json.load(f)
        else:
            names = datasets.ImageFolder(DATA_DIR).classes
        model = build_model(len(names))
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval(); transform = tf_val
        print('> modelo carregado de disco.')
    return model, transform, names, device, IMG_SIZE


# ═════════════════ execução direta ═════════════════
if __name__ == '__main__':
    train()