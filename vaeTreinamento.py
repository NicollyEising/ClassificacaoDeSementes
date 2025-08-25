import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, utils, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import lpips
from PIL import Image
import numpy as np

# ------------------- Hiperparâmetros -------------------
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE      = 128
BATCH_SIZE    = 32
LATENT_DIM    = 512
EPOCHS        = 200
WARM_REC_EPS  = 2
LR_GEN        = 2e-4
LR_DISC       = 5e-5
BASE_KLD      = 1e-2
BASE_ADV      = 5e-2
BETA          = 75
R1_GAMMA      = 10.0

categorias = [
    "Broken soybeans",
    "Immature soybeans",
    "Intact soybeans",
    "Skin-damaged soybeans",
    "Spotted soybeans"
]

# ------------------- Bloco residual up -----------------
class ResBlockUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.skip  = nn.Conv2d(in_ch, out_ch, 1, 1, 0)
        self.up    = nn.Upsample(scale_factor=2, mode='nearest')
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.bn2   = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        up = self.up(x)
        out = self.bn1(F.relu(self.conv1(up)))
        out = self.bn2(F.relu(self.conv2(out)))
        skip = self.up(self.skip(x))
        return out + skip

# ------------------------- VAE -------------------------
class VAE(nn.Module):
    def __init__(self, z_dim, n_cls):
        super().__init__()
        enc, c = [], 3
        for o in [64, 128, 256, 512]:
            enc += [
                nn.Conv2d(c, o, 4, 2, 1),
                nn.BatchNorm2d(o),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            c = o
        self.encoder = nn.Sequential(*enc, nn.Flatten())
        self.flat    = 512 * (IMG_SIZE // 16) ** 2
        self.mu      = nn.Linear(self.flat, z_dim)
        self.lv      = nn.Linear(self.flat, z_dim)
        self.fc      = nn.Linear(z_dim + n_cls, self.flat)

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (512, IMG_SIZE // 16, IMG_SIZE // 16)),
            ResBlockUp(512, 256),
            nn.Dropout2d(0.3),
            ResBlockUp(256, 128),
            nn.Dropout2d(0.3),
            ResBlockUp(128, 64),
            nn.Dropout2d(0.3),
            ResBlockUp(64, 32),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Tanh()
        )

    def rep(self, mu, lv):
        return mu + torch.randn_like(lv) * torch.exp(0.5 * lv)

    def forward(self, x, oh):
        h      = self.encoder(x)
        mu, lv = self.mu(h), self.lv(h)
        z      = self.rep(mu, lv)
        f      = self.fc(torch.cat([z, oh], dim=1))
        out    = self.decoder(f)
        return out, mu, lv

    def generate(self, z, oh):
        f   = self.fc(torch.cat([z, oh], dim=1))
        out = self.decoder(f)
        return out

# -------------------- Discriminador --------------------
class D(nn.Module):
    def __init__(self):
        super().__init__()
        c, layers = 3, []
        for o in [64, 128, 256, 512]:
            layers += [
                nn.utils.spectral_norm(nn.Conv2d(c, o, 4, 2, 1)),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            c = o
        self.cnn = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.utils.spectral_norm(nn.Linear(512, 1))

    def forward(self, x):
        x = self.cnn(x)
        x = self.gap(x).view(x.size(0), -1)
        return self.fc(x)

# ---------------------- Perdas -------------------------
lpips_loss = lpips.LPIPS(net='vgg').to(DEVICE)

def kld(mu, lv):
    return -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())

def r1_penalty(discriminator, real):
    real_req = real.detach().requires_grad_(True)
    pred     = discriminator(real_req).sum()
    grad     = torch.autograd.grad(pred, real_req, create_graph=True)[0]
    return grad.view(real_req.size(0), -1).pow(2).sum(1).mean()

def load_classifier():
    # função mantida para compatibilidade; pode ser removida se não utilizada
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(categorias))
    model = model.to(DEVICE)
    model.eval()
    return model


def train(data_dir, checkpoint_path="checkpoint.pth", best_model_path="best_model_vae.pth", seed: int = 0, resume: bool = False, single_class: bool = True):
    """Treina um VAE (possivelmente em modo single-class).

    Args:
        data_dir: pasta contendo as imagens (estrutura compatível com ImageFolder).
        checkpoint_path: caminho do checkpoint temporário.
        best_model_path: arquivo onde será salvo o melhor modelo (contém 'model' e 'num_classes').
        seed: semente aleatória.
        resume: se True, tenta retomar do checkpoint_path.
        single_class: se True, mapeia todos os rótulos para classe 0 e treina com n_cls=1.
    Returns:
        n_cls (int): número de classes usadas no treinamento.
    """
    torch.manual_seed(seed)
    tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    ds = datasets.ImageFolder(data_dir, tf)
    ds_classes = ds.classes
    # validações mínimas
    for c in ds_classes:
        if c not in categorias and not single_class:
            raise ValueError(f"Classe '{c}' não está em 'categorias'.")

    # Se pedirmos treino por categoria (modo único), definimos n_cls=1 e mapeamos tudo para a classe 0
    if single_class:
        n_cls = 1
    else:
        n_cls = len(categorias)

    dl = DataLoader(ds, BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    # Se não quisermos retomar e o checkpoint existir, removemos para começar do zero
    if os.path.exists(checkpoint_path) and not resume:
        try:
            os.remove(checkpoint_path)
        except Exception:
            pass

    vae  = VAE(LATENT_DIM, n_cls).to(DEVICE)
    d1   = D().to(DEVICE)
    d2   = D().to(DEVICE)
    optG = torch.optim.AdamW(vae.parameters(), lr=LR_GEN, betas=(0.5, 0.999))
    optD = torch.optim.AdamW(list(d1.parameters()) + list(d2.parameters()), lr=LR_DISC, betas=(0.5, 0.999))

    start_epoch = 1
    best_loss_g = float("inf")
    if resume and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        vae.load_state_dict(ckpt["vae"])
        d1.load_state_dict(ckpt["d1"])
        d2.load_state_dict(ckpt["d2"])
        optG.load_state_dict(ckpt["optG"])
        optD.load_state_dict(ckpt["optD"])
        start_epoch = ckpt["epoch"] + 1
        best_loss_g = ckpt.get("best_loss_g", best_loss_g)
        print(f"Checkpoint carregado: retomando da época {start_epoch}")

    os.makedirs('debug', exist_ok=True)
    # classifier carregado apenas se necessário
    classifier = load_classifier()

    for ep in range(start_epoch, EPOCHS + 1):
        totG = 0.0
        for i, (x, y_ds) in enumerate(tqdm(dl, desc=f'Época {ep}/{EPOCHS}')):
            x = x.to(DEVICE)
            if single_class:
                # mapeia todos os rótulos para 0 (apenas uma classe)
                y = torch.zeros(x.size(0), dtype=torch.long, device=DEVICE)
            else:
                y = torch.tensor([categorias.index(ds.classes[int(v)]) for v in y_ds], dtype=torch.long, device=DEVICE)
            oh = F.one_hot(y, n_cls).float().to(DEVICE)

            fake, mu, lv = vae(x, oh)
            fake_det = fake.detach()
            optD.zero_grad()
            if ep > WARM_REC_EPS:
                lossD = 0.0
                for Dm, sz in [(d1, IMG_SIZE), (d2, IMG_SIZE//2)]:
                    real = x if sz == IMG_SIZE else F.interpolate(x, (sz, sz))
                    fake_s = fake_det if sz == IMG_SIZE else F.interpolate(fake_det, (sz, sz))
                    lossD += -(Dm(real).mean() - Dm(fake_s).mean()) + R1_GAMMA * r1_penalty(Dm, real)
                lossD.backward()
                optD.step()
            else:
                lossD = torch.tensor(0.0, device=DEVICE)

            fake, mu, lv = vae(x, oh)
            fake = fake.clamp(-1, 1)
            rec = 0.3 * F.l1_loss(fake, x) + 0.7 * lpips_loss(fake, x).mean()
            if ep > WARM_REC_EPS:
                adv = 0.0
                for Dm, sz in [(d1, IMG_SIZE), (d2, IMG_SIZE//2)]:
                    f = fake if sz == IMG_SIZE else F.interpolate(fake, (sz, sz))
                    adv += -Dm(f).mean()
                λ_adv = BASE_ADV * ((ep - WARM_REC_EPS) / (EPOCHS - WARM_REC_EPS))**2 * 1.5
                λ_kld = BASE_KLD
            else:
                adv = torch.tensor(0.0, device=DEVICE)
                λ_adv = 0.0
                λ_kld = 0.0
            lossG = rec + BETA * λ_kld * kld(mu, lv) + λ_adv * adv
            optG.zero_grad()
            lossG.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            optG.step()
            totG += lossG.item()

            if i % 200 == 0:
                with torch.no_grad():
                    for cls in range(n_cls):
                        z_rand = torch.randn(1, LATENT_DIM, device=DEVICE)
                        oh_rand = F.one_hot(torch.tensor([cls], device=DEVICE), n_cls).float()
                        fake_rand = vae.generate(z_rand, oh_rand).clamp(-1, 1)
                        utils.save_image((fake_rand + 1) / 2, f'debug/ep{ep}_cls{cls}_rand.png')

        avg_loss_g = totG / max(1, len(dl))
        print(f'Época {ep} completa — Loss_G={avg_loss_g:.4f}  Loss_D={lossD.item():.4f}')
        torch.save({
            "epoch": ep,
            "vae": vae.state_dict(),
            "d1": d1.state_dict(),
            "d2": d2.state_dict(),
            "optG": optG.state_dict(),
            "optD": optD.state_dict(),
            "best_loss_g": best_loss_g
        }, checkpoint_path)
        if avg_loss_g < best_loss_g:
            best_loss_g = avg_loss_g
            # salvo o melhor modelo com num_classes (para geração posterior)
            torch.save({'model': vae.state_dict(), 'num_classes': n_cls}, best_model_path)
            print(f"Novo melhor modelo salvo na época {ep} com Loss_G={best_loss_g:.4f}")

    return n_cls

# ----------- Geração de Amostras após o Treino (1 categoria) -----------
def gerar_amostras(peso_arquivo, pasta_saida, categoria, num_imgs=100, seed: int = 0):
    torch.manual_seed(seed)

    checkpoint = torch.load(peso_arquivo, map_location=DEVICE)
    n_cls = checkpoint.get('num_classes', None)

    if n_cls is None:
        raise ValueError("Arquivo de peso não contém 'num_classes'.")

    # como o treino foi feito apenas para uma categoria, n_cls deve ser 1
    if n_cls != 1:
        raise ValueError(f"Modelo deveria ter 1 classe, mas possui {n_cls} classes.")

    vae = VAE(LATENT_DIM, n_cls).to(DEVICE)
    vae.load_state_dict(checkpoint['model'])
    vae.eval()

    # pasta principal da categoria
    pasta_classe = os.path.join(pasta_saida, categoria.replace(" ", "_"))
    os.makedirs(pasta_classe, exist_ok=True)

    with torch.no_grad():
        for idx in range(num_imgs):
            z = torch.randn(1, LATENT_DIM, device=DEVICE)
            oh = F.one_hot(torch.tensor([0], device=DEVICE), n_cls).float().to(DEVICE)  # apenas classe 0
            fake = vae.generate(z, oh).clamp(-1, 1)
            caminho_arquivo = os.path.join(pasta_classe, f'amostra_{idx:04d}_{categoria.replace(" ", "_")}.png')
            utils.save_image((fake + 1) / 2, caminho_arquivo)


if __name__ == "__main__":
    base_dir = r"C:\Users\faculdade\.cache\kagglehub\datasets\warcoder\soyabean-seeds\versions\2\Soybean Seeds"

    for categoria in categorias:
        print(f"\n--- Treinando categoria isolada: {categoria} ---")

        data_dir = os.path.join(base_dir, categoria)

        checkpoint = f"checkpoint_{categoria.replace(' ', '_')}.pth"
        best_model = f"best_model_vae_{categoria.replace(' ', '_')}.pth"
        pasta_amostras = os.path.join("amostras", categoria.replace(" ", "_"))

        # treino do zero para a categoria atual (single_class=True)
        train(
            data_dir=data_dir,
            checkpoint_path=checkpoint,
            best_model_path=best_model,
            seed=42,
            resume=True,
            single_class=True
        )

        # geração de amostras apenas dessa categoria
        gerar_amostras(
            peso_arquivo=best_model,
            pasta_saida="amostras",
            categoria=categoria,
            num_imgs=1000,
            seed=42
        )
