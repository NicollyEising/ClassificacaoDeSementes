# identificarSementesModelo.py
# ------------------------------------------------------------
#  Train & load – versão completa com checkpoints, F1 e IoU
# ------------------------------------------------------------
from datetime import datetime
import os, time, json, random
from pathlib import Path
from collections import Counter

import torch, torch.nn as nn
from torch.utils.data      import DataLoader, Subset, WeightedRandomSampler
from torchvision           import datasets, transforms
from torchvision.models    import resnet18, ResNet18_Weights
from sklearn.metrics       import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from bancoDeDados import Database
from io import BytesIO

# ---------- paths ----------
DATA_DIR    = r'C:\Users\faculdade\Desktop\SementesVisaoComputacional\amostras'
MODEL_PATH  = 'best_model.pth'
NAMES_JSON  = 'class_names.json'
CKPT_DIR    = Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)

# ---------- hiperparâmetros ----------
IMG_SIZE      = 224
BATCH_SIZE    = 32
EPOCHS        = 50
LR_MAX        = 3e-4
WEIGHT_DECAY  = 1e-4
EARLY_STOP    = 8
SEED          = 42
NUM_WORKERS   = 0

torch.manual_seed(SEED); random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------- transforms ----------
tf_train = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(25),
    transforms.ColorJitter(0.25,0.25,0.25,0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
tf_val = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ═════════════════ helpers ═════════════════
def build_datasets():
    ds_full_train = datasets.ImageFolder(DATA_DIR, transform=tf_train)
    ds_full_val   = datasets.ImageFolder(DATA_DIR, transform=tf_val)

    assert ds_full_train.classes == ds_full_val.classes
    class_names = ds_full_train.classes
    num_classes = len(class_names)

    # split 80/20
    indices = list(range(len(ds_full_train)))
    labels = [ds_full_train.imgs[i][1] for i in indices]
    random.shuffle(indices)
    split = int(0.8*len(indices))
    idx_train, idx_val = indices[:split], indices[split:]

    ds_train = Subset(ds_full_train, idx_train)
    ds_val   = Subset(ds_full_val, idx_val)

    # pesos para sampler balanceado
    cnt_train = Counter([labels[i] for i in idx_train])
    weights = torch.tensor([1.0/cnt_train[labels[i]] for i in idx_train], dtype=torch.double)
    sampler = WeightedRandomSampler(weights, num_samples=len(idx_train), replacement=True)

    train_ld = DataLoader(ds_train, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True)
    val_ld   = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    print("\nDistribuição classes (treino):", cnt_train)
    cnt_val = Counter([labels[i] for i in idx_val])
    print("Distribuição classes (val):", cnt_val)

    return train_ld, val_ld, class_names, num_classes

def build_model(num_classes):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)

def save_checkpoint(epoch, model, optimizer, scheduler, best_acc, patience):
    ckpt_path = CKPT_DIR / f"epoch_{epoch}.pth"
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'best_acc': best_acc,
        'patience': patience
    }, ckpt_path)
    print(f'  * Checkpoint salvo em {ckpt_path}')

def load_checkpoint(model, optimizer, scheduler, filename):
    ckpt = torch.load(filename, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    scheduler.load_state_dict(ckpt['scheduler_state'])
    return ckpt['epoch'], ckpt['best_acc'], ckpt['patience'], model, optimizer, scheduler

def iou_score(y_true, y_pred, num_classes):
    iou = []
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)
    for cls in range(num_classes):
        intersection = ((y_pred==cls) & (y_true==cls)).sum().item()
        union = ((y_pred==cls) | (y_true==cls)).sum().item()
        iou.append(intersection/union if union>0 else 0)
    return sum(iou)/len(iou)

def train(resume_ckpt=None):
    train_ld, val_ld, class_names, num_classes = build_datasets()
    model = build_model(num_classes)

    targets = [y for _,y in datasets.ImageFolder(DATA_DIR).imgs]
    freq = torch.bincount(torch.tensor(targets, dtype=torch.int64))
    ce_weights = (1.0 / freq.float()).to(device)
    criterion = nn.CrossEntropyLoss(weight=ce_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = len(train_ld)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR_MAX, epochs=EPOCHS, steps_per_epoch=steps_per_epoch, pct_start=0.3)

    start_epoch, best_acc, patience = 1, 0, 0
    if resume_ckpt:
        start_epoch, best_acc, patience, model, optimizer, scheduler = load_checkpoint(model, optimizer, scheduler, resume_ckpt)
        start_epoch += 1

    for epoch in range(start_epoch, EPOCHS+1):
        t0 = time.time(); model.train(); running_loss=0; correct=0
        for xb,yb in train_ld:
            xb,yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb); loss = criterion(out,yb)
            loss.backward(); optimizer.step(); scheduler.step()
            running_loss += loss.item()*xb.size(0)
            correct += (out.argmax(1)==yb).sum().item()
        train_loss = running_loss/len(train_ld.dataset)
        train_acc  = correct/len(train_ld.dataset)

        model.eval(); val_loss, val_correct=0,0; y_true,y_pred=[],[]
        with torch.no_grad():
            for xb,yb in val_ld:
                xb,yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = criterion(out,yb)
                val_loss += loss.item()*xb.size(0)
                val_correct += (out.argmax(1)==yb).sum().item()
                y_true.extend(yb.cpu().tolist())
                y_pred.extend(out.argmax(1).cpu().tolist())
        val_loss /= len(val_ld.dataset)
        val_acc  = val_correct/len(val_ld.dataset)
        f1 = f1_score(y_true, y_pred, average='weighted')
        iou = iou_score(y_true, y_pred, num_classes)

        print(f'Ep {epoch:02d}/{EPOCHS} | trn_loss {train_loss:.3f} acc {train_acc:.2%} | val_loss {val_loss:.3f} acc {val_acc:.2%} | F1 {f1:.3f} IoU {iou:.3f} ({time.time()-t0:.1f}s)')

        if val_acc > best_acc + 1e-4:
            best_acc = val_acc; patience = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print('  * Modelo salvo como melhor')
        else:
            patience += 1
            if patience >= EARLY_STOP:
                print('Early-stopping acionado.'); break
        save_checkpoint(epoch, model, optimizer, scheduler, best_acc, patience)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval(); print('\nClassification report:')
    print(classification_report(y_true, y_pred, target_names=class_names))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.title('Matriz de Confusão'); plt.xlabel('Predito'); plt.ylabel('Verdadeiro'); plt.tight_layout(); plt.show()

    with open(NAMES_JSON,'w',encoding='utf-8') as f:
        json.dump(class_names,f,ensure_ascii=False)


        # --- salvar relatório em JSON e TXT ---
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    results_doc = {
        "epochs": epoch,
        "best_val_acc": best_acc,
        "f1_weighted": f1,
        "iou": iou,
        "class_report": report,
        "train_samples": len(train_ld.dataset),
        "val_samples": len(val_ld.dataset),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    # salvar em JSON
    with open("training_results.json", "w", encoding="utf-8") as f:
        json.dump(results_doc, f, indent=4, ensure_ascii=False)

    # salvar em TXT (formato legível)
    with open("training_results.txt", "w", encoding="utf-8") as f:
        f.write("Resumo do Treinamento de Classificação de Sementes\n")
        f.write("="*60 + "\n")
        f.write(f"Data/Hora: {results_doc['timestamp']}\n")
        f.write(f"Melhor Acurácia Validação: {results_doc['best_val_acc']:.4f}\n")
        f.write(f"F1-Weighted: {results_doc['f1_weighted']:.4f}\n")
        f.write(f"IoU médio: {results_doc['iou']:.4f}\n")
        f.write(f"Épocas executadas: {results_doc['epochs']}\n\n")
        f.write("Relatório por classe:\n")
        for cls, metrics in report.items():
            if isinstance(metrics, dict):
                f.write(f"- {cls} | precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} f1={metrics['f1-score']:.3f}\n")
        f.write("\nDistribuição de amostras:\n")
        f.write(f"Treino: {results_doc['train_samples']}, Validação: {results_doc['val_samples']}\n")



# ============================
# SALVAR NO BANCO DE DADOS
# ============================


    # salvar no banco de dados
    # criar conexão com o banco
    db = Database(dbname="sementesdb", user="postgres", password="123")

    # aqui você precisa do ID do usuário; exemplo:
    usuario_id = 1  # substitua pelo ID correto do usuário que está executando o treino

    # salvar na tabela resultados
    salvar_resultado_treino(db, usuario_id, results_doc, nome_arquivo="training_results.json")

    db.fechar()



def salvar_resultado_treino(db: Database, usuario_id: int, results_doc: dict, nome_arquivo: str = None):
    """
    Salva os resultados do treino na tabela 'resultados' para um usuário específico.
    """
    import json

    # converter relatório de classes para string JSON para salvar na coluna 'classe_prevista'
    classe_prevista_json = json.dumps(results_doc['class_report'], ensure_ascii=False)

    # salvar
    db.inserir_resultado(
        usuario_id=usuario_id,
        img_bytes=None,  # não há imagem aqui, mas poderia salvar uma figura se quiser
        classe_prevista=classe_prevista_json,
        probabilidade=results_doc['best_val_acc'],  # pode usar best_val_acc como "probabilidade"
        cidade=None,
        temperatura=None,
        condicao=None,
        chance_chuva=None,
        nome_arquivo=nome_arquivo,
        data_hora=datetime.now(),
    )
    print(f"Resultado do treino salvo para usuario_id={usuario_id}")


def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        print('> best_model.pth ausente – iniciando treino...')
        model, transform, names = train()
    else:
        if os.path.exists(NAMES_JSON):
            with open(NAMES_JSON,'r',encoding='utf-8') as f: names = json.load(f)
        else: names = datasets.ImageFolder(DATA_DIR).classes
        model = build_model(len(names))
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval(); transform=tf_val
        print('> modelo carregado de disco.')
    return model, transform, names, device, IMG_SIZE

if __name__ == '__main__':
    train()
