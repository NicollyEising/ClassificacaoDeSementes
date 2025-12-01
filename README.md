# Sistema de Identificação e Análise de Sementes

Projeto de Visão Computacional para Classificação e Avaliação de Qualidade

## Objetivo

O objetivo deste projeto é desenvolver um sistema automatizado de identificação e análise de sementes agrícolas, empregando técnicas de visão computacional, aprendizado de máquina e métodos de explicabilidade. O sistema realiza a classificação do tipo de semente, avalia sua qualidade, identifica defeitos visuais e apresenta os resultados em uma interface web responsiva com rastreabilidade por QR Code.

O desenvolvimento do modelo de classificação foi apoiado por um conjunto de dados ampliado artificialmente. Para isso, foi utilizado um modelo VAE-GAN responsável pela geração de **3.000 imagens sintéticas por classe**, permitindo a construção de um dataset equilibrado, adequado ao treinamento da rede neural e essencial para alcançar maior precisão e robustez nas análises.

## Escopo

### Funcionalidades Principais

**Classificação e Análise**

* Identificação automática do tipo de semente.
* Avaliação da qualidade com base em características visuais.
* Detecção de defeitos, como mofo, fissuras e descoloração.
* Segmentação de regiões afetadas por imperfeições.

**Interface do Usuário**

* Upload de imagens.
* Visualização dos resultados.
* Histórico de análises.
* Acesso a páginas de resultado via QR Code.

**Processamento e Armazenamento**

* Armazenamento estruturado de imagens e resultados.
* Geração de imagens sintéticas para ampliação do dataset.

**Integrações**

* Consulta a APIs climáticas.
* Recomendações baseadas em condições ambientais.

## Contexto

A avaliação manual da qualidade de sementes é suscetível a variabilidade e limitações operacionais. O emprego de visão computacional aumenta a precisão, reduz falhas e contribui para práticas de agricultura de precisão.

## Restrições

* Dependência da qualidade das imagens fornecidas pelo usuário.
* Necessidade de conjuntos de dados equilibrados para cada classe.
* Análise restrita a características visuais.

## Diferenciais

* Aplicação de redes neurais convolucionais para classificação e segmentação.
* Utilização de Grad-CAM para interpretabilidade.
* Uso de VAE-GAN para geração de imagens sintéticas e balanceamento do dataset.
* Disponibilização de resultados via QR Code.

## Modelo Arquitetural

O sistema compreende:

* **Frontend** em ReactJS, com páginas dinâmicas, visualização das análises e leitura de QR Code.
* **Backend** em Python (FastAPI), responsável pelo processamento, inferência e integração com APIs externas.
* **Banco de Dados** PostgreSQL para armazenamento de metadados e histórico.
* **Infraestrutura em Nuvem** com frontend hospedado na Vercel e backend na AWS.

## Requisitos

### Requisitos Funcionais

* **RF1:** Upload de imagens.
* **RF2:** Classificação automática.
* **RF3:** Avaliação de qualidade.
* **RF4:** Detecção e segmentação de defeitos.
* **RF5:** Histórico de análises.
* **RF6:** Geração de QR Code.
* **RF7:** Página dinâmica com resultados e previsões ambientais.
* **RF8:** Geração de imagens sintéticas com VAE-GAN.

### Requisitos Não Funcionais

* **RNF1:** Segurança de dados.
* **RNF2:** Interface responsiva.
* **RNF3:** Código modular e escalável.
* **RNF4:** Acessibilidade conforme diretrizes WCAG 2.1.

## Pilha Tecnológica

**Frontend**

* ReactJS
* React Router
* Tailwind CSS
* Semantic UI
* Axios

**Backend**

* Python
* FastAPI
* OpenCV
* TensorFlow / PyTorch
* PostgreSQL

**Ferramentas Adicionais**

* Trello
* GitHub Actions
* Vercel
* AWS

## Metodologia de Desenvolvimento

Metodologia FDD, distribuída em pacotes:

**Pacote 1 – Configuração e Infraestrutura**

* Configuração inicial.
* Estruturação do projeto.
* Integração inicial entre frontend e backend.

**Pacote 2 – Upload e Pré-processamento**

* Interface de envio.
* Pré-processamento e segmentação básica.

**Pacote 3 – Classificação e Avaliação**

* Modelos de classificação.
* Detecção de defeitos.
* Registro dos resultados.

**Pacote 4 – Interface e Histórico**

* Visualização completa das análises.
* Histórico detalhado.
* Implementação do QR Code.

## Coleta de Dados

### Conjuntos Utilizados

* Soybean Seeds Classification Dataset (Kaggle).
* Base própria com imagens coletadas.

### Expansão e Balanceamento do Dataset (Ponto de Ênfase)

Para garantir equilíbrio entre classes e volume suficiente para o treinamento da rede neural, o dataset final foi construído majoritariamente por meio de um modelo **VAE-GAN**, utilizado para geração de imagens sintéticas realistas.

* Foram produzidas **3.000 imagens sintéticas por classe**, totalizando um conjunto amplo e balanceado.
* O modelo foi responsável tanto pela ampliação quanto pela padronização visual dos exemplos, garantindo maior robustez no treinamento.

## Treinamento de Modelos

* Arquitetura principal: ResNet-50 com fine-tuning.
* Parâmetros estimados: batch size 32, 50 épocas, taxa de aprendizado 1e-4.
* Validação cruzada utilizando 5 folds.
* Implementação adicional de VAE e VAE-GAN para síntese e melhoria do dataset.

## Métricas de Avaliação

* Acurácia global.
* F1-score.
* IoU para segmentação.

## Interface Web

* Tela de upload.
* Página de análise contendo imagem, resultados, defeitos e dados ambientais.
* Histórico com listagem e gráficos.
* Página específica gerada via QR Code.

## Monitoramento

* Registro de logs.
* Indicadores de desempenho da API.
* Métricas de uso.

## Infraestrutura

* **Frontend:** Vercel
* **Backend:** AWS
* **Banco:** PostgreSQL
* **CI/CD:** GitHub Actions
* **Planejamento:** Trello

## Riscos e Mitigações

* **Imagens de baixa qualidade:** aplicação de técnicas de pré-processamento.
* **Imbalance de classes:** utilização de VAE-GAN para geração balanceada de dados.
* **Integração entre módulos:** testes contínuos e versionamento.

## Considerações Finais

O sistema integra visão computacional, modelos generativos e serviços web para fornecer uma solução completa de análise de sementes. A utilização extensiva de imagens sintéticas geradas por VAE-GAN permitiu a formação de um dataset equilibrado, com 3.000 amostras por categoria, elevando a precisão e a robustez do modelo de classificação. A arquitetura modular possibilita expansão futura para novas espécies e funcionalidades.
