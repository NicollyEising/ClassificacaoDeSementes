# main.py
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Caminho para as imagens
data_dir = r'C:\Users\faculdade\.cache\kagglehub\datasets\warcoder\soyabean-seeds\versions\2\Soybean Seeds'

# Transforms
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Dataset
dataset = datasets.ImageFolder(root=data_dir, transform=transform_train)
class_names = dataset.classes
num_classes = len(class_names)

# Treino/Validação
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

# Corrige transform do val_ds
val_ds.dataset.transform = transform_val

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

# Modelo pré-treinado
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
epochs = 20

# Treinamento
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == y_batch).sum().item()

    acc = correct / len(train_ds)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.2f}, Accuracy: {acc:.4f}")

# Avaliação
model.eval()
correct = 0
y_true, y_pred = [], []

with torch.no_grad():
    for x_batch, y_batch in val_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        outputs = model(x_batch)
        preds = outputs.argmax(1)
        correct += (preds == y_batch).sum().item()
        y_true.extend(y_batch.cpu())
        y_pred.extend(preds.cpu())

val_acc = correct / len(val_ds)
print(f"\nValidation Accuracy: {val_acc:.4f}")
print("\nRelatório de Classificação:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Matriz de Confusão
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Matriz de Confusão")
plt.xlabel("Predito")
plt.ylabel("Verdadeiro")
plt.show()

# Função de predição
from PIL import Image

def predict_image(img_path):
    image = Image.open(img_path).convert('RGB')
    transform_predict = transform_val  # usar transform do val
    image = transform_predict(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(image)
        pred_idx = output.argmax(1).item()
        return class_names[pred_idx]