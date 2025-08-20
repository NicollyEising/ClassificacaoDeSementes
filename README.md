# Sistema de Identificação e Análise de Sementes

## Objetivo do Projeto

Desenvolver um sistema automatizado de classificação de sementes agrícolas, utilizando visão computacional e aprendizado de máquina, que permita identificar diferentes tipos de sementes (soja, trigo, quiabo, etc.) e avaliar sua qualidade com base em características como cor, tamanho, textura e presença de defeitos (mofo, fissuras, descoloração).

---

## Escopo

**Classificação de sementes:**
O sistema deve ser capaz de:

* Identificar o tipo de semente automaticamente.
* Avaliar a qualidade das sementes.
* Detectar defeitos em cada semente.

**Interface do usuário:**

* Upload de imagens das sementes.
* Visualização dos resultados de classificação e qualidade.
* Histórico de análises realizadas.

**Segurança e Privacidade:**

* Armazenamento seguro das imagens e resultados.
* Controle de acesso aos dados sensíveis.

**Testes e Validação:**

* Testes unitários e de integração para assegurar a precisão do modelo e funcionamento do sistema.

---

## Tecnologias Utilizadas

**Frontend:**

* React: Construção de interfaces de usuário.
* React Router: Navegação entre páginas.
* Axios: Requisições HTTP ao backend.
* HTML e CSS: Estrutura e estilização.
* Tailwind CSS e Semantic UI: Framework para estilização rápida.

**Backend:**

* Python: Processamento de imagens e lógica do modelo.
* Flask / FastAPI: Criação de APIs para comunicação com o frontend.
* PyTorch / TensorFlow: Treinamento e execução do modelo de classificação.
* OpenCV: Processamento e análise de imagens.
* MongoDB / PostgreSQL: Armazenamento de resultados e imagens processadas.

**Testes e Qualidade:**

* PyTest: Testes unitários e integração.
* SonarQube: Análise de qualidade do código.

---

## Requisitos do Projeto

### Requisitos Funcionais

* **RF1:** Upload de imagens pelo usuário.
* **RF2:** Classificação automática da semente.
* **RF3:** Avaliação da qualidade da semente.
* **RF4:** Detecção de defeitos nas sementes.
* **RF5:** Histórico de análises anteriores.

### Requisitos Não Funcionais

* **RNF1:** Segurança no armazenamento de imagens e resultados.
* **RNF2:** Alta precisão e desempenho no processamento das imagens.
* **RNF3:** Interface intuitiva e de fácil navegação.
* **RNF4:** Código organizado e documentado para manutenção futura.

---

## Metodologia de Organização de Tarefas

O projeto adotará a metodologia **FDD (Feature Driven Development)**, com foco na entrega incremental de funcionalidades:

* Cada funcionalidade (upload, classificação, análise de qualidade, histórico) será desenvolvida de forma independente, testada e validada.
* Permite acompanhamento preciso do progresso e priorização das funcionalidades essenciais.

---

## Pacotes de Entrega

**Pacote 1: Configuração e Infraestrutura**

* Configuração do ambiente de desenvolvimento.
* Estruturação do projeto (pastas e arquivos).
* Integração entre backend e frontend.

**Pacote 2: Upload e Processamento de Imagens**

* Interface para upload de imagens de sementes.
* Processamento inicial das imagens (pré-processamento e segmentação).

**Pacote 3: Classificação e Avaliação**

* Implementação do modelo de classificação de sementes.
* Avaliação de qualidade e detecção de defeitos.
* Armazenamento dos resultados no banco de dados.

**Pacote 4: Interface e Histórico**

* Visualização dos resultados pelo usuário.
* Histórico de análises.
* Sistema de autenticação e segurança.

**Pacote 5: Testes e Qualidade**

* Testes unitários e de integração do backend e frontend.
* Monitoramento de qualidade do código.
* Integração contínua (CI/CD).

---

## Infraestrutura

* **Frontend:** Vercel
* **Backend:** AWS
* **Banco de Dados:** PostgreSQL
* **CI/CD:** GitHub
* **Trello:** https://trello.com/b/CmD4xiWe/tcc-sementes

---



