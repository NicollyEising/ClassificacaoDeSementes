import os
import io
import cv2
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from collections import Counter
import torch.nn.functional as F
import requests
import re
import base64
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, Form, UploadFile, File, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from bancoDeDados import *

# Importações do script original (assumindo arquivos no mesmo diretório)
from bancoDeDados import Database
from cadastroLogin import login, usuario_logado  # Ajuste se necessário
try:
    from identificarSementesModelo import load_trained_model
except Exception as e:
    raise RuntimeError("Não foi possível importar load_trained_model") from e



# Instância global do BD (ajuste credenciais)
db = Database(dbname="sementesdb", user="postgres", password="123")

# Configurações globais do modelo
model, transform_val, class_names, device, IMG_SIZE = load_trained_model()
class_names = [
    "Broken soybeans",
    "Immature soybeans",
    "Intact soybeans",
    "Skin-damaged soybeans",
    "Spotted soybeans"
]
model = model.to(device).eval()

# Configurações de API
app = FastAPI(title="API Visão Computacional Sementes")
security = HTTPBasic()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] ,  # endereço do seu frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chaves API (do script original)
WEATHER_API_KEY = "fa1b810b93d5467994f30008251705"
AGRO_CONSUMER_KEY = "fyfa_Jsspy6meIrgVpFEonTxeUIa"
AGRO_CONSUMER_SECRET = "14CHcEvIvoThD37ObxwlzgKPrn0a"

# Diretórios e constantes (do script)
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
BROKEN_SKIN_MARGIN = 0.08
SMALL_ROI_RATIO = 0.01
BROKEN_NAME = "Broken soybeans"
SKIN_NAME = "Skin-damaged soybeans"
SPOTTED_NAME = "Spotted soybeans"

# Modelos Pydantic para respostas
class ClimaResponse(BaseModel):
    cidade: str
    temperatura: float
    condicao: str
    chance_chuva: int

class ResultadoProcessamento(BaseModel):
    classe_prevista: str
    probabilidade: float
    imagem_anotada_base64: str  # Imagem anotada em base64
    clima: ClimaResponse
    recomendacoes: List[dict]  # Resultados da API Agro

class LoginResponse(BaseModel):
    usuario_id: int
    mensagem: str

# Funções auxiliares (do script original, adaptadas)

def get_weather(cidade="São Paulo", dias=3):
    url = "https://api.weatherapi.com/v1/forecast.json"
    try:
        resposta = requests.get(url, params={
            "key": WEATHER_API_KEY,
            "q": cidade,
            "days": dias,
            "lang": "pt"
        })
        resposta.raise_for_status()
        dados = resposta.json()
        return dados
    except requests.exceptions.RequestException as erro:
        raise HTTPException(status_code=500, detail=f"Erro ao consultar clima: {erro}")

def get_access_token(consumer_key, consumer_secret):
    credentials = f"{consumer_key}:{consumer_secret}"
    b64_credentials = base64.b64encode(credentials.encode()).decode()
    token_url = "https://api.cnptia.embrapa.br/token"
    data = {"grant_type": "client_credentials"}
    headers = {"Authorization": f"Basic {b64_credentials}"}
    
    response = requests.post(token_url, data=data, headers=headers)
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        raise HTTPException(status_code=500, detail=f"Erro ao obter token Agro: {response.text}")

def consulta_respondeagro(access_token, query, from_record=0, size=10):
    url = "https://api.cnptia.embrapa.br/respondeagro/v1/_search/template"
    headers = {"Authorization": f"Bearer {access_token}"}
    payload = {
        "id": "query_all",
        "params": {
            "query_string": query,
            "from": from_record,
            "size": size
        }
    }
    
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        raise HTTPException(status_code=500, detail=f"Erro na consulta Agro: {response.text}")

# Classe GradCAM (do script)
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()

    def __call__(self, x, class_idx=None):
        x = x.to(device)
        out = self.model(x)

        if class_idx is None:
            class_idx = out.argmax(dim=1).item()

        loss = out[:, class_idx]
        self.model.zero_grad()
        loss.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

def overlay_cam(img_bgr: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    img_bgr: imagem no formato BGR (como retornado pelo OpenCV)
    cam: mapa de ativação normalizado em [0,1], shape=(H_cam, W_cam)
    Retorna: imagem BGR com heatmap sobreposta
    """
    # Garantir tipo e escala corretos para applyColorMap
    heatmap_uint8 = np.uint8(255 * np.clip(cam, 0.0, 1.0))
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)  # já retorna BGR
    # Ajustar tamanho do heatmap para o da imagem, se necessário
    if (heatmap_bgr.shape[0] != img_bgr.shape[0]) or (heatmap_bgr.shape[1] != img_bgr.shape[1]):
        heatmap_bgr = cv2.resize(heatmap_bgr, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    # Combinar heatmap e imagem original (ambos em BGR)
    result = cv2.addWeighted(img_bgr, 1.0 - alpha, heatmap_bgr, alpha, 0)
    return result

# Funções de inferência (do script)
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
    return [(class_names[int(i)], float(probs[int(i)])) for i in idxs]

def class_index(name: str) -> int:
    try:
        return class_names.index(name)
    except ValueError:
        return -1

# Processamento principal (adaptado do script para função)
def processar_imagem(frame: np.ndarray, cidade: str = "Jaragua do Sul") -> dict:
    # Obter clima
    clima_data = get_weather(cidade, 1)
    if "erro" in clima_data:
        raise HTTPException(status_code=500, detail=clima_data["detalhe"])
    temp = clima_data["current"]["temp_c"]
    condicao = clima_data["current"]["condition"]["text"]
    chance_chuva = clima_data["forecast"]["forecastday"][0]["day"]["daily_chance_of_rain"]

    # Configurar GradCAM
    target_layer = model.layer4[-1]  # Ajuste conforme modelo
    gradcam = GradCAM(model, target_layer)

    # Inferência whole
    whole_probs = infer_whole(frame)

    # Segmentação e ROIs (do script)
    H, W = frame.shape[:2]
    IMG_AREA = H * W
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, 1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, 1)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = int(MIN_AREA_RATIO * IMG_AREA)
    max_area = int(MAX_AREA_RATIO * IMG_AREA)
    counters = Counter()
    rois = []

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        per = cv2.arcLength(cnt, True)
        ar = w / h if h > 0 else 0.0
        circ = 4.0 * np.pi * (area / (per * per)) if per != 0 else 0.0

        if per == 0:
            counters["perimeter_zero"] += 1
            reason = "perimeter_zero"
        elif area < min_area or area > max_area:
            counters["area_filtered"] += 1
            reason = "area_filtered"
        elif ar < ASPECT_MIN or ar > ASPECT_MAX:
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
        pil_roi = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        probs = infer_tta(pil_roi)
        peak = float(np.max(probs))
        pred_idx = int(np.argmax(probs))

        rois.append({
            "i": i,
            "bbox": (x0, y0, x1, y1),
            "area": area,
            "aspect": ar,
            "circularity": circ,
            "reason": reason,
            "probs": probs,
            "peak": peak,
            "pred_idx": pred_idx,
            "pred_name": class_names[pred_idx],
        })

    # Agregação e fusão (do script)
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

    final_probs = (1.0 - WHOLE_WEIGHT) * fused + WHOLE_WEIGHT * whole_probs if sum_w > 0 else whole_probs.copy()
    final_probs = softmax_normalize(final_probs)
    final_idx = int(np.argmax(final_probs))
    final_name = class_names[final_idx]
    final_prob = float(final_probs[final_idx])

    # Pós-processamento (Broken vs Skin, Healthy vs Damaged) - do script
    BROKEN_IDX = class_index(BROKEN_NAME)
    SKIN_IDX = class_index(SKIN_NAME)
    SPOTTED_IDX = class_index(SPOTTED_NAME)
    INTACT_IDX = class_index("Intact soybeans")
    IMMATURE_IDX = class_index("Immature soybeans")

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

    if INTACT_IDX >= 0 and IMMATURE_IDX >= 0 and SPOTTED_IDX >= 0:
        p_intact = float(final_probs[INTACT_IDX])
        p_immature = float(final_probs[IMMATURE_IDX])
        p_skin = float(final_probs[SKIN_IDX]) if SKIN_IDX >= 0 else 0.0
        p_broken = float(final_probs[BROKEN_IDX]) if BROKEN_IDX >= 0 else 0.0
        p_spotted = float(final_probs[SPOTTED_IDX])

        if final_idx in (SKIN_IDX, BROKEN_IDX, SPOTTED_IDX) and len(rois) > 0:
            r_best = max(rois, key=lambda r: r["peak"])
            roi_ratio = r_best["area"] / float(IMG_AREA)
            if roi_ratio < 0.02:
                if whole_probs[INTACT_IDX] > max(whole_probs[SKIN_IDX], whole_probs[BROKEN_IDX], whole_probs[SPOTTED_IDX]):
                    final_idx = INTACT_IDX
                    final_name = class_names[final_idx]
                    final_prob = float(final_probs[final_idx])
                elif whole_probs[IMMATURE_IDX] > max(whole_probs[SKIN_IDX], whole_probs[BROKEN_IDX], whole_probs[SPOTTED_IDX]):
                    final_idx = IMMATURE_IDX
                    final_name = class_names[final_idx]
                    final_prob = float(final_probs[final_idx])

        healthy_max = max(p_intact, p_immature)
        damage_max = max(p_skin, p_broken, p_spotted)
        if healthy_max >= damage_max * 0.98:
            if p_intact >= p_immature:
                final_idx = INTACT_IDX
            else:
                final_idx = IMMATURE_IDX
            final_name = class_names[final_idx]
            final_prob = float(final_probs[final_idx])

    # Anotação da imagem (com bbox e texto)
    chosen_bbox = (0, 0, W, H)
    if len(rois) > 0:
        same_class = [r for r in rois if r["pred_idx"] == final_idx]
        r_draw = max(same_class, key=lambda r: r["peak"]) if same_class else max(rois, key=lambda r: r["peak"])
        chosen_bbox = r_draw["bbox"]

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

    # --------------------------
    # Gerar Grad-CAM e overlay
    # --------------------------
    # Preparar o tensor de entrada (tamanho esperado pelo modelo)
    pil_whole = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((IMG_SIZE, IMG_SIZE))
    input_t = transform_val(pil_whole).unsqueeze(0).to(device)

    # Gera o mapa Grad-CAM para a classe final
    cam_map = gradcam(input_t, class_idx=final_idx)  # resultado em tamanho IMG_SIZE x IMG_SIZE, normalizado 0..1

    # Redimensionar cam para o tamanho original da imagem
    cam_resized = cv2.resize(cam_map, (W, H), interpolation=cv2.INTER_LINEAR)

    # Criar overlay Grad-CAM sobre a imagem original (BGR)
    overlay = overlay_cam(frame, cam_resized, alpha=0.5)

    # Codificar imagem overlay para base64 (PNG)
    is_success, buffer = cv2.imencode(".png", overlay)
    if not is_success:
        raise HTTPException(status_code=500, detail="Falha ao codificar imagem Grad-CAM")
    img_bytes = buffer.tobytes()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")


    # Recomendações Agro (adaptado)
    recomendacao_map = {
        "Intact soybeans": "sementes de soja saudáveis",
        "Immature soybeans": "sementes de soja imaturas",
        "Broken soybeans": "soja quebrada",
        "Skin-damaged soybeans": "soja com casca danificada",
        "Spotted soybeans": "soja com manchas",
    }
    query_variants = {
        "Intact soybeans": ["características fisicas de sementes de soja saudáveis", "recomendações para sementes de soja saudáveis"],
        "Immature soybeans": ["características de sementes de soja imaturas", "recomendações para sementes de soja imaturas"],
        "Broken soybeans": ["características de sementes de soja quebradas", "recomendações para sementes de soja quebradas"],
        "Skin-damaged soybeans": ["características de sementes de soja com casca danificada", "recomendações para sementes de soja casca danificada"],
        "Spotted soybeans": ["características de sementes de soja com manchas", "recomendações para sementes de soja com manchas"]
    }
    generic_queries = ["manejo sementes de soja", "qualidade de sementes de soja", "controle de doenças em sementes de soja", "boas práticas sementes soja"]

    base_term = recomendacao_map.get(final_name, "soja")
    queries = query_variants.get(final_name, []) + [base_term] + generic_queries

    token = get_access_token(AGRO_CONSUMER_KEY, AGRO_CONSUMER_SECRET)
    recomendacoes = []
    for q in queries:
        resultados = consulta_respondeagro(token, q, 0, 10)
        hits = resultados.get("hits", {}).get("hits", [])
        if hits:
            for item in hits:
                source = item["_source"]
                resposta_texto = re.sub(r'<.*?>', '', source.get('answer', ''))
                recomendacoes.append({
                    "pergunta": source.get('question'),
                    "resposta": resposta_texto,
                    "capitulo": source.get('chapter'),
                    "livro": source.get('book'),
                    "ano": source.get('year'),
                    "pdf": source.get('pdf'),
                    "epub": source.get('epub')
                })
            break  # Para ao encontrar resultados

    # Retorno
    return {
        "classe_prevista": final_name,
        "probabilidade": final_prob,
        "imagem_anotada_base64": img_base64,
        "clima": {
            "cidade": cidade,
            "temperatura": temp,
            "condicao": condicao,
            "chance_chuva": chance_chuva
        },
        "recomendacoes": recomendacoes
    }

# Endpoints da API


class UsuarioRequest(BaseModel):
    usuario_id: int

# Modelo de resposta
class UsuarioResponse(BaseModel):
    usuario_id: int
    mensagem: str

@app.post("/usuario", response_model=UsuarioResponse)
def api_usuario(request: UsuarioRequest):
    # Validação simples do ID
    if request.usuario_id <= 0:
        raise HTTPException(status_code=400, detail="ID inválido")

    # Retorna o ID recebido e uma mensagem
    return {"usuario_id": request.usuario_id, "mensagem": "Usuário recebido com sucesso"}

import qrcode
import base64
from io import BytesIO

def gerar_url_detalhes(semente_id: int) -> str:
    """
    Gera a URL de detalhes para um resultado de semente.
    """
    return f"https://meusistema.com/resultados/{semente_id}"

def gerar_qrcode_base64(url: str) -> str:
    """
    Gera um QR Code a partir de uma URL e retorna em formato base64.
    """
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")

@app.post("/processar_imagem", response_model=ResultadoProcessamento)
async def api_processar_imagem(
    arquivo: UploadFile = File(...),
    cidade: Optional[str] = "Jaragua do Sul",
    usuario_id: int = Form(...)
):
    try:
        contents = await arquivo.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Imagem inválida")

        resultado = processar_imagem(frame, cidade)

        # --- geração de recomendações inteligentes ---
        try:
            recomendacoes = gerar_recomendacoes_smart(resultado, usuario_profile={"usuario_id": usuario_id})
            resultado["recomendacoes"] = recomendacoes
        except Exception as e:
            print("Aviso: erro ao gerar recomendações:", e)
            resultado["recomendacoes"] = []

        semente_id = db.obter_proximo_id()  # ou o ID que será usado
        url_detalhes = f"http://127.0.0.1:5500/frontend/item.html?id={semente_id}"
        qrcode_base64 = gerar_qrcode_base64(url_detalhes)

        # Salvar no BD (adaptado do script)
        is_success, buffer = cv2.imencode(".png", cv2.imdecode(np.frombuffer(base64.b64decode(resultado["imagem_anotada_base64"]), np.uint8), cv2.IMREAD_COLOR))
        img_bytes = buffer.tobytes()

        db.inserir_resultado(
            usuario_id=usuario_id,
            img_bytes=img_bytes,
            classe_prevista=resultado["classe_prevista"],
            probabilidade=resultado["probabilidade"],
            cidade=resultado["clima"]["cidade"],
            temperatura=resultado["clima"]["temperatura"],
            condicao=resultado["clima"]["condicao"],
            chance_chuva=resultado["clima"]["chance_chuva"],
            nome_arquivo=arquivo.filename,
            data_hora=datetime.now(),
            url_detalhes=url_detalhes,
            qrcode_base64=qrcode_base64
        )


        if not resultado.get("imagem_anotada_base64"):
            raise HTTPException(status_code=400, detail="Imagem anotada não gerada pela função processar_imagem")

    # Conferir tamanho
        print("Tamanho da imagem em bytes:", len(img_bytes))

        return resultado
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/resultado/{sement_id}", response_model=dict)
def obter_resultado(sement_id: int):
    resultado = db.obter_resultado_por_id(sement_id)
    usuario_id: int = Form(...)
    if not resultado:
        raise HTTPException(status_code=404, detail="Resultado não encontrado")

    # Garantir URL e QR Code
    resultado["url_detalhes"] = resultado.get("url_detalhes") or f"http://127.0.0.1:5500/frontend/item.html?id={sement_id}"
    resultado["qrcode_base64"] = resultado.get("qrcode_base64") or gerar_qrcode_base64(resultado["url_detalhes"])

    # Converter bytes para base64
    img_bytes = resultado.get("img_bytes")
    if img_bytes:
        resultado["imagem_anotada_base64"] = base64.b64encode(img_bytes).decode("utf-8")
    else:
        resultado["imagem_anotada_base64"] = None

    

    resultado["data_hora"] = resultado.get("data_hora").isoformat() if resultado.get("data_hora") else datetime.now().isoformat()

        # --- geração de recomendações inteligentes ---
    try:
        recomendacoes = gerar_recomendacoes_smart(resultado, usuario_profile={"usuario_id": usuario_id})
        resultado["recomendacoes"] = recomendacoes

    except Exception as e:
            print("Aviso: erro ao gerar recomendações:", e)
            resultado["recomendacoes"] = []
    return resultado

@app.get("/clima", response_model=ClimaResponse)
def api_clima(cidade: str = "Jaragua do Sul"):
    clima = get_weather(cidade, 1)
    return {
        "cidade": clima["location"]["name"],
        "temperatura": clima["current"]["temp_c"],
        "condicao": clima["current"]["condition"]["text"],
        "chance_chuva": clima["forecast"]["forecastday"][0]["day"]["daily_chance_of_rain"]
    }




@app.get("/recomendacoes_agro/{classe}")
def api_recomendacoes_agro(classe: str):
    # Chamada direta à função de recomendações
    # Nota: Esta é uma simplificação; integre com processar_imagem se necessário
    token = get_access_token(AGRO_CONSUMER_KEY, AGRO_CONSUMER_SECRET)
    # Use uma query simples baseada na classe
    query = f"recomendações para {classe.lower()}"
    resultados = consulta_respondeagro(token, query, 0, 10)
    hits = resultados.get("hits", {}).get("hits", [])
    recomendacoes = []
    for item in hits:
        source = item["_source"]
        resposta_texto = re.sub(r'<.*?>', '', source.get('answer', ''))
        recomendacoes.append({
            "pergunta": source.get('question'),
            "resposta": resposta_texto,
            "capitulo": source.get('chapter'),
            "livro": source.get('book'),
            "ano": source.get('year'),
            "pdf": source.get('pdf'),
            "epub": source.get('epub')
        })
    return {"classe": classe, "recomendacoes": recomendacoes}



from visaoComputacional import *

# Mapas de sinônimos ou variações de consulta por classe
query_variants = {
    "Intact soybeans": [
        "soja intacta",
        "sementes de soja sadias",
        "qualidade de grãos de soja"
    ],
    "Immature soybeans": [
        "soja imatura",
        "maturação de sementes de soja",
        "soja verde"
    ],
    "Broken soybeans": [
        "soja quebrada",
        "sementes danificadas mecanicamente",
        "quebra de grãos de soja"
    ],
    "Skin-damaged soybeans": [
        "soja com dano de casca",
        "danos de tegumento em sementes de soja",
        "ferimentos em casca de soja"
    ],
    "Spotted soybeans": [
        "soja manchada",
        "manchas em sementes de soja",
        "infecção fúngica em grãos de soja"
    ]
}

def clima_risk_factor(clima):
    # Exemplo simples: quanto maior chance_chuva, maior o risco
    try:
        chance = int(clima.get("chance_chuva", 0))
    except:
        chance = 0
    # normaliza 0..1
    return min(1.0, max(0.0, chance / 100.0))

# Mapa de severidade manual (0..1)
SEVERITY_MAP = {
    "Intact soybeans": 0.0,
    "Immature soybeans": 0.3,
    "Broken soybeans": 0.6,
    "Skin-damaged soybeans": 0.7,
    "Spotted soybeans": 0.8
}

def gerar_recomendacoes_smart(resultado, usuario_profile=None):
    """
    resultado: dict retornado por processar_imagem
    usuario_profile: dict opcional com preferências (ex: 'mercado_exportacao':True)
    Retorna: lista de recomendações estruturadas
    """
    classe = resultado["classe_prevista"]
    prob = float(resultado["probabilidade"])
    clima = resultado.get("clima", {})
    rois = resultado.get("rois", [])  # se quiser manter rois no retorno
    # calcular max roi ratio (se rois presentes)
    max_roi_ratio = 0.0
    if rois:
        H = resultado.get("imagem_shape", {}).get("H", None)
        W = resultado.get("imagem_shape", {}).get("W", None)
        if H and W:
            IMG_AREA = H * W
            max_roi_ratio = max([r["area"] / float(IMG_AREA) for r in rois]) if rois else 0.0
        else:
            # fallback: use roi area fraction se salva
            max_roi_ratio = max([r.get("area_ratio", 0.0) for r in rois]) if rois else 0.0

    # parâmetros de peso (ajustáveis)
    w_class, w_roi, w_clima, w_user = 0.5, 0.2, 0.2, 0.1
    alpha_area = ALPHA_AREA if 'ALPHA_AREA' in globals() else 0.6

    score = (w_class * prob
             + w_roi * (max_roi_ratio ** alpha_area)
             + w_clima * clima_risk_factor(clima)
             + w_user * (1.0 if usuario_profile and usuario_profile.get("sensitive_market") else 0.0)
            )
    # normalizar (apenas heurístico)
    score = min(1.0, max(0.0, score))

    # Prioridade por score
    if score >= 0.75:
        prioridade = "alta"
    elif score >= 0.4:
        prioridade = "media"
    else:
        prioridade = "baixa"

    # Buscar recomendações em RespondeAgro (tenta primeiro queries já existentes)
    token = get_access_token(AGRO_CONSUMER_KEY, AGRO_CONSUMER_SECRET)
    base_queries = query_variants.get(classe, [classe])
    recomendacoes_texts = []
    for q in base_queries + ["manejo sementes de soja", "qualidade de sementes de soja"]:
        resp = consulta_respondeagro(token, q, 0, 5)
        hits = resp.get("hits", {}).get("hits", [])
        for h in hits:
            src = h.get("_source", {})
            texto = re.sub(r'<.*?>', '', src.get("answer", "") or "")
            if texto:
                recomendacoes_texts.append({
                    "pergunta": src.get("question"),
                    "resposta": texto,
                    "fonte": {
                        "capitulo": src.get("chapter"),
                        "livro": src.get("book"),
                        "ano": src.get("year"),
                        "pdf": src.get("pdf")
                    }
                })
        if recomendacoes_texts:
            break

    # Se não encontrou nada, gerar recomendações internas (templates)
    if not recomendacoes_texts:
        # exemplo simples
        if classe == "Spotted soybeans":
            recomendacoes_texts.append({
                "pergunta": "Providências iniciais",
                "resposta": "Isolar amostras e enviar para análise laboratorial; reduzir umidade de armazenamento; rastrear lote."
            })
        else:
            recomendacoes_texts.append({
                "pergunta": "Boas práticas",
                "resposta": "Separar lotes, reduzir tempo de armazenamento, monitorar temperatura e umidade."
            })

    # Monta lista final com metadata
    recs_struct = []
    for r in recomendacoes_texts:
        recs_struct.append({
            "acao": r.get("resposta"),
            "motivo": f"Classe: {classe}, prob: {prob:.3f}, prioridade calculada: {prioridade}",
            "prioridade": prioridade,
            "score": score,
            "fonte": r.get("fonte", {}),
            "evidencia": {
                "classe": classe,
                "probabilidade": prob,
                "max_roi_ratio": max_roi_ratio,
                "clima": clima
            }
        })

    return recs_struct





if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)