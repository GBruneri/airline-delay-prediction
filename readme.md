# ✈️ Airline Delay Risk Prediction — End-to-End Data Science Project

## 📌 Visão geral

Este projeto implementa um **pipeline completo de Ciência de Dados**, desde a ingestão e exploração dos dados até a disponibilização de um **modelo preditivo via API**, com foco em **organização, reprodutibilidade e decisões arquiteturais realistas**.

O objetivo final é **estimar a probabilidade histórica de atraso de voos (≥ 15 minutos)** para um determinado **aeroporto, mês e companhia aérea**, fornecendo um **score de risco interpretável**, adequado para consulta.

O projeto foi desenvolvido com **mentalidade de mercado**, priorizando:
- separação clara de responsabilidades  
- ausência de data leakage  
- foco no usuário final  
- boas práticas de versionamento e arquitetura  

---

## 🎯 Problema e framing

Em vez de tentar prever atrasos individuais de voos (o que exigiria dados operacionais e climáticos em tempo real), o projeto responde à seguinte pergunta:

> **“Dado um aeroporto, uma companhia aérea e um mês do ano, qual é o risco histórico de atraso segundo os dados disponíveis?”**

Esse framing resulta em:
- um modelo **sazonal e estrutural**
- mais estável
- mais interpretável
- mais adequado para consumo via API

---

## 📊 Dataset

- **Fonte:** Kaggle  
- **Dataset:** *Airline Delay Cause*  
- **Granularidade:** agregações mensais por aeroporto e companhia aérea  

O dataset **não é versionado no Git**. Apenas o código responsável por baixá-lo e carregá-lo faz parte do repositório, garantindo reprodutibilidade sem inflar o histórico.

---

## 🧠 Decisões importantes de modelagem

### 🔹 Definição do target

O target foi definido como uma **probabilidade normalizada de atraso**, evitando vieses de volume:

$$delay_probability = arr_del15 / arr_flights$$


Posteriormente, o problema foi tratado como **classificação binária**, usando um **limiar baseado em quantil**, o que permite:
- lidar com assimetria forte
- focar nos casos de maior risco
- evitar decisões arbitrárias de threshold

---

### 🔹 Prevenção de data leakage

Variáveis **pós-evento**, como:
- `arr_delay`
- delays por causa específica
- contagens detalhadas de atraso

foram **explicitamente removidas** do pipeline de modelagem.

Essa decisão reduz métricas infladas artificialmente, mas resulta em um modelo honesto e defensável.

---

### 🔹 Avaliação e trade-offs

Foram testados:
- baseline linear
- árvores de decisão
- Random Forest com regularização

O ganho incremental de modelos mais complexos foi **marginal**, o que levou à decisão consciente de **parar a complexificação** e priorizar:
- estabilidade
- interpretabilidade
- custo computacional

---

## 🏗️ Arquitetura do projeto

A organização do projeto segue padrões utilizados em ambientes profissionais:
```airline-delay-risk-prediction/
├── api/ # FastAPI (serving do modelo)
│ └── main.py
│
├── artifacts/ # Artefatos gerados (não versionados)
│
├── scripts/ # Execuções pontuais (treino offline)
│ └── train_and_save_model.py
│
├── src/ # Biblioteca do projeto
│ ├── data/ # Ingestão e limpeza
│ ├── features/ # Target e feature engineering
│ ├── models/ # Preprocessing, treino e avaliação
│ └── utils/
│
├── notebooks/ # EDA e análise (orquestração)
│
├── README.md
└── .gitignore
```

### Princípios adotados
- **Notebooks não contêm lógica crítica**
- **`src/` funciona como biblioteca reutilizável**
- **Treino e inferência são etapas separadas**
- **Artefatos de modelo não são versionados**

---

## 🚀 Inferência e API

### 🔹 Inferência offline

O modelo final é treinado por meio de um **script dedicado**, que gera um artefato serializado:

```bash
python scripts/train_and_save_model.py


