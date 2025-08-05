# Sistema de Identificação e Geração de Imagens de Sementes com Visão Computacional

Este projeto implementa um sistema baseado em visão computacional para identificação automática de sementes agrícolas e geração de amostras sintéticas utilizando técnicas de VAE-GAN. A aplicação final inclui um pipeline de captura por webcam, classificação em tempo real e visualização dos resultados. O modelo foi treinado com o dataset Soybean Seeds.

## Estrutura do Projeto

- Treinamento de VAE-GAN: Rede geradora de imagens sintéticas condicionadas às classes reais.  
- Treinamento de Classificador ResNet18: Classificação das imagens em cinco categorias de sementes.  
- Identificação via Webcam: Detecção de sementes em vídeo e rotulagem em tempo real.  
- Geração de Amostras: Produção de imagens sintéticas por classe para simulação ou reforço de dataset.  

## Requisitos

- Python ≥ 3.8  
- PyTorch ≥ 1.12  
- torchvision  
- lpips  
- scikit-learn  
- OpenCV  
- matplotlib  
- seaborn  
- PIL  

## Treinamento do Gerador VAE-GAN

O treinamento utiliza um modelo VAE com blocos residuais e duas redes discriminadoras (WGAN-GP). As imagens geradas são salvas a cada 200 iterações.

    python vae_gan_treino.py

Parâmetros principais:  
- LATENT_DIM = 512  
- IMG_SIZE = 128  
- BETA, BASE_KLD, BASE_ADV regulam as perdas de reconstrução, KLD e adversária.  
- WARM_REC_EPS = 2 (warm-up para evitar ruído no início do treino adversário).  

Saídas:  
- Reconstruções reais e geradas (debug/ep*_fake.png)  
- Amostras aleatórias condicionadas a rótulos  
- Interpolações latentes  

## Geração de Amostras Sintéticas

Gera imagens sintéticas de sementes condicionadas às classes aprendidas:

    gerar_amostras('vae_wgp.pth', 'amostras', num_imgs=100)  

## Classificação Supervisionada com ResNet18

Treinamento de um classificador para as cinco classes de sementes:

    python classificador_resnet18.py  

Fluxo:  
- Pré-processamento com transforms.Compose  
- Treinamento com CrossEntropyLoss  
- Avaliação com classification_report e matriz de confusão  
- Predição de imagem individual via predict_image('caminho_da_imagem.png')  

## Identificação em Tempo Real com Webcam

Executa captura em tempo real e detecção de sementes por contornos:

    python identificar_sementes_webcam.py  

Características:  
- Filtragem por área, circularidade e proporção  
- Classificação com limiar de confiança  
- Exibição da predição na tela com cv2.putText  

## Instruções

- Faça o download do dataset *Soybean Seeds* disponível em:  
  https://www.kaggle.com/datasets/warcoder/soyabean-seeds

- Após o download, insira o caminho local do dataset na variável `data_dir` nos seguintes arquivos:  
    - `identificarSementesModelo.py`  
    - `vaeTreinamento.py`

## Referências

- Dataset: Soybean Seeds Dataset (Kaggle): https://www.kaggle.com/datasets/warcoder/soyabean-seeds
- Técnicas: VAE, GAN, WGAN-GP, ResNet18  
- Métrica de qualidade: LPIPS (Perceptual Similarity)
