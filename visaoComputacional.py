# -*- coding: utf-8 -*-
"""
visaocomputacional.py — Pipeline de detecção + classificação de sementes (atualizado)

Novidades:
- Lista explícita das 5 classes conhecidas (em ordem coerente com treino).
- Pós-processamento expandido: evita que 'Intact soybeans' ou 'Immature soybeans'
  sejam confundidas como 'Skin-damaged' ou 'Broken' quando a evidência for fraca.
"""

import os
import cv2
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from collections import Counter

# ---------- carga do modelo ----------
try:
    from identificarSementesModelo import load_trained_model
except Exception as e:
    raise RuntimeError(
        "Não foi possível importar load_trained_model de identificarSementesModelo.py"
    ) from e

# Carrega modelo, etc.
model, transform_val, class_names, device, IMG_SIZE = load_trained_model()

# Lista explícita de classes conforme treino
class_names = [
    "Broken soybeans",
    "Immature soybeans",
    "Intact soybeans",
    "Skin-damaged soybeans",
    "Spotted soybeans"
]

model = model.to(device).eval()

# ---------- parâmetros principais ----------
IMAGE_PATH = r"C:\Users\faculdade\.cache\kagglehub\datasets\warcoder\soyabean-seeds\versions\2\Soybean Seeds\Spotted soybeans\10.jpg"

OUT_DIR = Path("debug_rois")
OUT_DIR.mkdir(exist_ok=True)

MIN_AREA_RATIO = 0.00015
MAX_AREA_RATIO = 0.12
ASPECT_MIN, ASPECT_MAX = 0.25, 4.0
CIRC_MIN = 0.10

USE_TTA = True
CONFIDENCE_PRINT_TOPK = 3

ALPHA_AREA = 0.6
BETA_CONF = 1.0
MIN_KEEPED_ROIS = 1

WHOLE_WEIGHT = 0.6

# Ajuste de nomes completos
BROKEN_NAME = "Broken soybeans"
SKIN_NAME = "Skin-damaged soybeans"

BROKEN_SKIN_MARGIN = 0.08
SMALL_ROI_RATIO = 0.01

# ---------- utilidades ----------
def infer_tta(pil_img: Image.Image) -> np.ndarray:
    if USE_TTA:
        imgs = [
            pil_img,
            pil_img.transpose(Image.FLIP_LEFT_RIGHT),
            pil_img.transpose(Image.FLIP_TOP_BOTTOM),
            pil_img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM),
        ]
    else:
        imgs = [pil_img]

    probs_sum = None
    with torch.no_grad():
        for im in imgs:
            t = transform_val(im).unsqueeze(0).to(device)
            out = model(t)
            p = torch.softmax(out, dim=1).cpu().numpy().ravel()
            probs_sum = p if probs_sum is None else probs_sum + p

    return probs_sum / len(imgs)

def infer_whole(frame_bgr: np.ndarray) -> np.ndarray:
    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).resize((IMG_SIZE, IMG_SIZE))
    return infer_tta(pil)

def softmax_normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    m = np.max(v)
    ex = np.exp(v - m)
    return ex / (np.sum(ex) + eps)

def topk_info(probs: np.ndarray, k: int = 3):
    idxs = np.argsort(probs)[::-1][:k]
    return [(int(i), class_names[int(i)], float(probs[int(i)])) for i in idxs]

def class_index(name: str) -> int:
    try:
        return class_names.index(name)
    except ValueError:
        return -1

BROKEN_IDX = class_index(BROKEN_NAME)
SKIN_IDX = class_index(SKIN_NAME)

# ---------- leitura da imagem ----------
frame = cv2.imread(IMAGE_PATH)
if frame is None:
    raise FileNotFoundError(IMAGE_PATH)

H, W = frame.shape[:2]
IMG_AREA = H * W
print(f"[INFO] Image: {W}x{H}")
print(f"[INFO] Classes: {class_names}")

# ---------- segmentação ----------
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
_, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, 1)
th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, 1)

contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"[INFO] Contornos encontrados: {len(contours)}")

min_area = int(MIN_AREA_RATIO * IMG_AREA)
max_area = int(MAX_AREA_RATIO * IMG_AREA)

# ---------- avaliação de ROIs ----------
counters = Counter()
rois = []

for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    per = cv2.arcLength(cnt, True)

    if per == 0:
        counters["perimeter_zero"] += 1
        reason = "perimeter_zero"
    elif area < min_area or area > max_area:
        counters["area_filtered"] += 1
        reason = "area_filtered"
    else:
        ar = w / h if h > 0 else 0
        circ = 0.0 if per == 0 else 4.0 * np.pi * (area / (per * per))
        if ar < ASPECT_MIN or ar > ASPECT_MAX:
            counters["aspect_filtered"] += 1
            reason = "aspect_filtered"
        elif circ < CIRC_MIN:
            counters["circularity_filtered"] += 1
            reason = "circularity_filtered"
        else:
            reason = "kept"
            counters["kept"] += 1

    margin = max(2, int(0.04 * max(w, h)))
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(W, x + w + margin)
    y1 = min(H, y + h + margin)

    roi = frame[y0:y1, x0:x1]
    cv2.imwrite(str(OUT_DIR / f"roi_{i:03d}_{reason}.png"), roi)

    pil_roi = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    probs = infer_tta(pil_roi)
    peak = float(np.max(probs))
    pred_idx = int(np.argmax(probs))

    rois.append({
        "i": i,
        "bbox": (x0, y0, x1, y1),
        "area": area,
        "aspect": ar if per != 0 else 0.0,
        "circularity": circ if per != 0 else 0.0,
        "reason": reason,
        "probs": probs,
        "peak": peak,
        "pred_idx": pred_idx,
        "pred_name": class_names[pred_idx],
    })

    if i < 20:
        print(f"[ROI {i:03d}] area={area:.1f} aspect={ar:.2f} circ={circ:.3f} "
              f"reason={reason} pred={class_names[pred_idx]}({peak:.3f}) "
              f"top{CONFIDENCE_PRINT_TOPK}: {topk_info(probs, CONFIDENCE_PRINT_TOPK)}")

print("[INFO] Contadores:", dict(counters))

# ---------- agregação de ROIs ----------
fused = np.zeros(len(class_names), dtype=np.float64)
sum_w = 0.0
kept_for_fusion = 0

for r in rois:
    if r["reason"] != "kept":
        continue
    w_area = (max(r["area"], 1.0) / float(IMG_AREA)) ** ALPHA_AREA
    w_conf = (max(r["peak"], 1e-6)) ** BETA_CONF
    weight = w_area * w_conf
    fused += weight * r["probs"]
    sum_w += weight
    kept_for_fusion += 1

if kept_for_fusion < MIN_KEEPED_ROIS and len(rois) > 0:
    r_best = max(rois, key=lambda r: r["peak"])
    w_area = (max(r_best["area"], 1.0) / float(IMG_AREA)) ** ALPHA_AREA
    w_conf = (max(r_best["peak"], 1e-6)) ** BETA_CONF
    weight = w_area * w_conf
    fused += weight * r_best["probs"]
    sum_w += weight
    kept_for_fusion = 1

if sum_w > 0:
    fused /= sum_w

# ---------- integração da imagem inteira ----------
whole_probs = infer_whole(frame)

if sum_w > 0:
    final_probs = (1.0 - WHOLE_WEIGHT) * fused + WHOLE_WEIGHT * whole_probs
else:
    final_probs = whole_probs.copy()

final_probs = softmax_normalize(final_probs)
final_idx = int(np.argmax(final_probs))
final_name = class_names[final_idx]
final_prob = float(final_probs[final_idx])

# ---------- pós-processamento: Broken vs Skin-damaged ----------
if BROKEN_IDX >= 0 and SKIN_IDX >= 0:
    p_broken = float(final_probs[BROKEN_IDX])
    p_skin = float(final_probs[SKIN_IDX])
    if abs(p_broken - p_skin) <= BROKEN_SKIN_MARGIN:
        if whole_probs[SKIN_IDX] > whole_probs[BROKEN_IDX]:
            final_idx = SKIN_IDX
            final_name = class_names[final_idx]
            final_prob = float(final_probs[final_idx])
    if final_idx == BROKEN_IDX and len(rois) > 0:
        r_best = max(rois, key=lambda r: r["peak"])
        roi_ratio = r_best["area"] / float(IMG_AREA)
        if roi_ratio < SMALL_ROI_RATIO and whole_probs[SKIN_IDX] >= whole_probs[BROKEN_IDX] * 0.95:
            final_idx = SKIN_IDX
            final_name = class_names[final_idx]
            final_prob = float(final_probs[final_idx])

# ---------- pós-processamento: Classes saudáveis vs. danificadas ----------
INTACT_IDX = class_index("Intact soybeans")
IMMATURE_IDX = class_index("Immature soybeans")

if INTACT_IDX >= 0 and IMMATURE_IDX >= 0:
    p_intact = float(final_probs[INTACT_IDX])
    p_immature = float(final_probs[IMMATURE_IDX])
    p_skin = float(final_probs[SKIN_IDX]) if SKIN_IDX >= 0 else 0.0
    p_broken = float(final_probs[BROKEN_IDX]) if BROKEN_IDX >= 0 else 0.0

    if final_idx in (SKIN_IDX, BROKEN_IDX) and len(rois) > 0:
        r_best = max(rois, key=lambda r: r["peak"])
        roi_ratio = r_best["area"] / float(IMG_AREA)
        if roi_ratio < 0.02:
            if whole_probs[INTACT_IDX] > max(whole_probs[SKIN_IDX], whole_probs[BROKEN_IDX]):
                final_idx = INTACT_IDX
                final_name = class_names[final_idx]
                final_prob = float(final_probs[final_idx])
            elif whole_probs[IMMATURE_IDX] > max(whole_probs[SKIN_IDX], whole_probs[BROKEN_IDX]):
                final_idx = IMMATURE_IDX
                final_name = class_names[final_idx]
                final_prob = float(final_probs[final_idx])

    healthy_max = max(p_intact, p_immature)
    damage_max = max(p_skin, p_broken)
    if healthy_max >= damage_max * 0.98:
        if p_intact >= p_immature:
            final_idx = INTACT_IDX
        else:
            final_idx = IMMATURE_IDX
        final_name = class_names[final_idx]
        final_prob = float(final_probs[final_idx])

# ---------- escolha de bbox ----------
chosen_bbox = (0, 0, W, H)
if len(rois) > 0:
    same_class = [r for r in rois if r["pred_idx"] == final_idx]
    if same_class:
        r_draw = max(same_class, key=lambda r: r["peak"])
    else:
        r_draw = max(rois, key=lambda r: r["peak"])
    chosen_bbox = r_draw["bbox"]

# ---------- anotação ----------
x0, y0, x1, y1 = chosen_bbox
color = (0, 200, 0)
out = frame.copy()
cv2.rectangle(out, (x0, y0), (x1, y1), color, 3)
cv2.putText(out, f"{final_name} ({final_prob:.3f})", (x0, max(30, y0 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

top_margin = max(50, int(0.12 * H))
annot = cv2.copyMakeBorder(out, top_margin, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
label_text = f"Estado: {final_name}    Prob: {final_prob:.3f}"
(fs_w, fs_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, max(0.6, top_margin/80), 2)
cv2.putText(annot, label_text,
            ((annot.shape[1] - fs_w)//2, top_margin//2 + fs_h//2),
            cv2.FONT_HERSHEY_SIMPLEX, max(0.6, top_margin/80), (0, 0, 0), 2, cv2.LINE_AA)

mx0 = int((x0 / W) * annot.shape[1])
mx1 = int((x1 / W) * annot.shape[1])
mini_h = max(6, int(top_margin * 0.20))
myc = int(top_margin * 0.75)
my0 = myc - mini_h // 2
my1 = myc + mini_h // 2
cv2.rectangle(annot, (mx0, my0), (mx1, my1), (0, 200, 0), -1)
cv2.rectangle(annot, (mx0, my0), (mx1, my1), (0, 120, 0), 1)
cv2.arrowedLine(annot, ((mx0 + mx1)//2, my1 + 1), ((x0 + x1)//2, y0 + top_margin),
                (0, 120, 0), 2, tipLength=0.04)

save_path = OUT_DIR / "annotated_result.png"
cv2.imwrite(str(save_path), annot)
print("[INFO] Resultado anotado salvo em:", save_path)

# ---------- resumo ----------
summary = {
    "image": IMAGE_PATH,
    "classes": class_names,
    "final_class": final_name,
    "final_prob": final_prob,
    "top3_final": topk_info(final_probs, 3),
    "whole_top3": topk_info(whole_probs, 3),
    "fusion_params": {
        "ALPHA_AREA": ALPHA_AREA,
        "BETA_CONF": BETA_CONF,
        "WHOLE_WEIGHT": WHOLE_WEIGHT
    },
    "counters": dict(counters),
    "num_rois": len(rois),
}
with open(OUT_DIR / "summary_best.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

pred_dist = Counter([r["pred_idx"] for r in rois])
print("[INFO] Distribuição de predições por ROI:", {class_names[i]: c for i, c in pred_dist.items()})
print("[DONE]")