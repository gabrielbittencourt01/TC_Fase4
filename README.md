# Previsão de Preço de Ações com LSTM - Tech Challenge Fase 4 (FIAP)

Este projeto foi desenvolvido para o Tech Challenge da Fase 4 da Pós em Engenharia de Machine Learning da FIAP. Ele consiste na criação de um modelo de Deep Learning com LSTM para prever o preço de fechamento de uma ação da bolsa, além do deploy de uma API RESTful e uma interface interativa com Streamlit.

---

## 🧠 Modelo

- Tipo: Rede Neural LSTM
- Dados: Preço de fechamento da ação `AAPL` (Apple Inc.) de 2015 a 2024
- Coleta: Usando `yfinance`
- Pré-processamento: Normalização MinMax + janelas deslizantes
- Hiperparâmetros otimizados com Keras Tuner
- Avaliação com: MAE, RMSE, MAPE

---

## 🚀 API (FastAPI)

- Rota: `POST /predict`
- Entrada: JSON com os últimos 60 valores **normalizados**
- Saída: Previsão do próximo preço (em escala original)
- Rota: `GET /health` para checagem de status
- Rota: `GET /` com mensagem de boas-vindas HTML

### Exemplo de entrada:

```json
{
  "values": [0.51, 0.52, ..., 0.53]  // 60 valores normalizados
}
```

### Como rodar a API:

```bash
# 1. Clonar repositório
# 2. Instalar dependências
pip install -r requirements.txt

# 3. Rodar API localmente
uvicorn main:app --reload

# ou com Docker
docker build -t lstm-api .
docker run -p 8000:8000 lstm-api
```

---

## 💻 App Streamlit

Interface interativa para prever preços de duas formas:

1. Upload de arquivo `.xlsx` com preços históricos
2. Entrada manual de 60 valores

### Como rodar:

```bash
streamlit run app_streamlit.py
```

---

## 🌐 Integração

O app Streamlit envia os dados para a API FastAPI (local ou em nuvem), recebe a previsão e exibe:
- Gráfico
- Preço previsto
- Métricas (MAE, RMSE, MAPE)

---

## 📦 Deploy (opcional)

- Sugestão: deploy gratuito via [Render](https://render.com)
- Subir a pasta com `main.py`, `models/`, `requirements.txt`, `Dockerfile`

---

## ✅ Testes

### 1. Testar API diretamente
Abra no navegador:
- `http://localhost:8000/` → Página inicial
- `http://localhost:8000/docs` → Swagger interativo

### 2. Testar API com `curl`:
```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"values": [0.5, 0.51, ..., 0.52]}'
```

### 3. Testar App Streamlit
- Escolha entre upload de Excel ou digitar manualmente
- Clique em "Prever preço"

---

## 🎥 Roteiro para o Vídeo de Apresentação

1. Introdução: objetivo do projeto (previsão de preço com LSTM)
2. Explicação do notebook (coleta, modelo, avaliação)
3. Demonstração da API via Swagger
4. Teste da API via `curl` ou Postman
5. Demonstração do app Streamlit com upload + entrada manual
6. Encerramento com o que foi aprendido

---

## 📁 Estrutura de Arquivos

```
├── app_streamlit.py
├── main.py
├── Dockerfile
├── requirements.txt
├── README.md
├── models/
│   ├── lstm_model.h5
│   └── scaler.pkl
└── notebook/
    └── TechChallengeFase4_notebook.ipynb
 
