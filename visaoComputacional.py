import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from identificarSementesModelo import *

# Classes do dataset (adicione "Não Semente" se incluir no treino)
class_names = ['Broken', 'Immature', 'Intact', 'Skin-damaged', 'Spotted']

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

cap = cv2.VideoCapture(0)
print("Pressione 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_copy = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Filtros morfológicos para reduzir ruído
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected = False

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000 or area > 5000:  # Limite inferior e superior da área
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:  # Faixa de proporção aceitável
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity < 0.3:  # Limite para circularidade
            continue

        roi = frame_copy[y:y+h, x:x+w]
        image = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            max_prob, pred_class = torch.max(probs, dim=1)

            if max_prob.item() < 0.7:  # Threshold de confiança
                label = "Não Semente"
            else:
                label = class_names[pred_class.item()]

        if label != "Não Semente":
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'{label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2, cv2.LINE_AA)
            detected = True
            break  # Apenas 1 contorno principal por frame

    if not detected:
        cv2.putText(frame, "Semente nao detectada", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Identificacao de Sementes', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
