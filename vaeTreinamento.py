import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
from tqdm import tqdm
import lpips

# ------------------- Hiperparâmetros -------------------
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE      = 128
BATCH_SIZE    = 32
LATENT_DIM    = 512

EPOCHS        = 70
WARM_REC_EPS  = 2

LR_GEN        = 2e-4
LR_DISC       = 5e-5
BASE_KLD      = 1e-2
BASE_ADV      = 5e-2
BETA          = 50
R1_GAMMA      = 10.0

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

        # Camada de saída do decoder
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

# ------------------- Treino ---------------------------
def train(data_dir):
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
    dl = DataLoader(ds, BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    n_cls = len(ds.classes)

    vae  = VAE(LATENT_DIM, n_cls).to(DEVICE)
    d1   = D().to(DEVICE)
    d2   = D().to(DEVICE)
    optG = torch.optim.AdamW(vae.parameters(), lr=LR_GEN, betas=(0.5, 0.999))
    optD = torch.optim.AdamW(
        list(d1.parameters()) + list(d2.parameters()),
        lr=LR_DISC, betas=(0.5, 0.999)
    )

    os.makedirs('debug', exist_ok=True)

    for ep in range(1, EPOCHS + 1):
        totG = 0.0
        for i, (x, y) in enumerate(tqdm(dl, desc=f'Época {ep}/{EPOCHS}')):
            x  = x.to(DEVICE)
            oh = F.one_hot(y, n_cls).float().to(DEVICE)

            fake, mu, lv = vae(x, oh)
            fake_det     = fake.detach()
            optD.zero_grad()
            if ep > WARM_REC_EPS:
                lossD = 0.0
                for Dm, sz in [(d1, IMG_SIZE), (d2, IMG_SIZE//2)]:
                    real   = x if sz == IMG_SIZE else F.interpolate(x, (sz, sz))
                    fake_s = fake_det if sz == IMG_SIZE else F.interpolate(fake_det, (sz, sz))
                    lossD += -(Dm(real).mean() - Dm(fake_s).mean()) + R1_GAMMA * r1_penalty(Dm, real)
                lossD.backward()
                optD.step()
            else:
                lossD = torch.tensor(0.0, device=DEVICE)


            fake, mu, lv = vae(x, oh)
            fake         = fake.clamp(-1, 1)
            rec          = 0.3 * F.l1_loss(fake, x) + 0.7 * lpips_loss(fake, x).mean()

            if ep > WARM_REC_EPS:
                adv    = 0.0
                for Dm, sz in [(d1, IMG_SIZE), (d2, IMG_SIZE//2)]:
                    f = fake if sz == IMG_SIZE else F.interpolate(fake, (sz, sz))
                    adv += -Dm(f).mean()
                λ_adv = BASE_ADV * ((ep - WARM_REC_EPS) / (EPOCHS - WARM_REC_EPS))**2
                λ_kld = BASE_KLD
            else:
                adv    = torch.tensor(0.0, device=DEVICE)
                λ_adv  = 0.0
                λ_kld  = 0.0

            lossG = rec + BETA * λ_kld * kld(mu, lv) + λ_adv * adv
            optG.zero_grad()
            lossG.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            optG.step()
            totG += lossG.item()

            if i % 200 == 0:
                # Reconstrução
                utils.save_image((fake[:16] + 1) / 2, f'debug/ep{ep}_batch{i}_fake.png', nrow=4)
                utils.save_image((x[:16] + 1)    / 2, f'debug/ep{ep}_batch{i}_real.png', nrow=4)
                with torch.no_grad():
                    z_rand = torch.randn(16, LATENT_DIM, device=DEVICE)
                    y_rand = torch.randint(0, n_cls, (16,), device=DEVICE)
                    oh_rand = F.one_hot(y_rand, n_cls).float().to(DEVICE)
                    fake_rand = vae.generate(z_rand, oh_rand).clamp(-1, 1)
                    utils.save_image((fake_rand + 1) / 2, f'debug/ep{ep}_batch{i}_rand.png', nrow=4)
                with torch.no_grad():
                    z1 = torch.randn(1, LATENT_DIM, device=DEVICE)
                    z2 = torch.randn(1, LATENT_DIM, device=DEVICE)
                    y_interp = torch.randint(0, n_cls, (1,), device=DEVICE)
                    oh_interp = F.one_hot(y_interp, n_cls).float().to(DEVICE)
                    imgs = []
                    for alpha in torch.linspace(0, 1, steps=8):
                        z = (1 - alpha) * z1 + alpha * z2
                        img = vae.generate(z, oh_interp).clamp(-1, 1)
                        imgs.append(img)
                    imgs = torch.cat(imgs, 0)
                    utils.save_image((imgs + 1) / 2, f'debug/ep{ep}_batch{i}_interp.png', nrow=8)

        print(f'Época {ep} completa — Loss_G={(totG/len(dl)):.4f}  Loss_D={lossD.item():.4f}')

    torch.save({'model': vae.state_dict(), 'num_classes': n_cls}, 'vae_wgp.pth')
    return n_cls

# ----------- Geração de Amostras após o Treino ---------
def gerar_amostras(peso_arquivo, pasta_saida, num_imgs=100):
    os.makedirs(pasta_saida, exist_ok=True)

    checkpoint = torch.load(peso_arquivo, map_location=DEVICE)
    n_cls = checkpoint['num_classes']

    vae = VAE(LATENT_DIM, n_cls).to(DEVICE)
    vae.load_state_dict(checkpoint['model'])
    vae.eval()

    with torch.no_grad():
        for i in range(num_imgs):
            z = torch.randn(1, LATENT_DIM, device=DEVICE)
            y = torch.randint(0, n_cls, (1,), device=DEVICE)
            oh = F.one_hot(y, n_cls).float().to(DEVICE)

            fake = vae.generate(z, oh).clamp(-1, 1)
            utils.save_image((fake + 1) / 2,
                             f'{pasta_saida}/amostra_{i:03d}.png')

# ----------------------- Main -------------------------
if __name__ == "__main__":
    data_dir     = r'C:\Users\faculdade\.cache\kagglehub\datasets\warcoder\soyabean-seeds\versions\2\Soybean Seeds'
    train(data_dir)
    gerar_amostras('vae_wgp.pth', 'amostras', num_imgs=100)