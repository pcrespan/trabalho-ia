# Sistema de Análise de Risco de Crédito

Sistema de machine learning para análise e predição de risco de crédito utilizando modelos de regressão logística e ensemble (Random Forest).

## Estrutura do Projeto

```
trabalho-ia/
├── app/                          # Aplicação Streamlit
│   ├── pages/
│   │   └── main.py              # Interface principal
│   ├── constants.py             # Constantes e configurações
│   ├── feedback.py              # Sistema de feedback
│   ├── train.py                 # Treinamento de modelos
│   └── utils.py                 # Funções auxiliares
├── data/
│   ├── german.csv               # Dataset original (German Credit Data)
│   └── training_data.csv        # Dados de treinamento com feedback
├── models/
│   ├── base_model.py            # Classe base para modelos
│   ├── logistic_model.py        # Modelo de regressão logística
│   ├── ensemble_model.py        # Modelo Random Forest
│   ├── ensemble/                # Modelos ensemble treinados
│   └── regression/              # Modelos de regressão treinados
├── train_pipeline/
│   ├── pipeline.py              # Pipeline de treinamento
│   └── preprocessor/
│       └── create_preprocessor.py
└── scripts/
    └── retrain_watcher.py       # Monitoramento automático
```

## Funcionalidades

- **Predição de Risco**: Análise de crédito usando dois modelos (Regressão Logística e Random Forest)
- **Interface Web**: Dashboard interativo desenvolvido com Streamlit
- **Sistema de Feedback**: Registro de predições incorretas para retreinamento
- **Retreinamento Automático**: Atualização dos modelos com novos dados
- **Métricas de Avaliação**: Visualização de acurácia e matriz de confusão

## Modelos

### Regressão Logística
- Modelo linear para classificação binária
- Treinado com `max_iter=1000`

### Random Forest
- Modelo ensemble com 200 árvores de decisão
- Maior robustez e capacidade de capturar padrões complexos

## Instalação

Instale as dependências:
```bash
pip install -r requirements.txt
```

## Uso

### Executar a Interface Web

```bash
streamlit run app/pages/main.py
```

### Treinar Modelos

Execute o pipeline de treinamento:

```bash
python -m train_pipeline.pipeline
```

### Retreinamento com Feedback

1. Use a interface para fazer predições
2. Marque predições incorretas através do sistema de feedback
3. Clique em "Retrain models" para atualizar os modelos com os novos dados

## Dataset

O projeto utiliza o **German Credit Data**, que contém informações de clientes para análise de risco de crédito:

- **Total**: 1000 registros
- **Features**: 20 atributos (balanço de conta, duração do crédito, status de pagamento, propósito, idade, etc.)
- **Target**: Creditability (1 = Bom, 0 = Ruim)

## Pré-processamento

- **Features Numéricas**: Normalização com StandardScaler
- **Features Categóricas**: OneHotEncoder
- **Split**: 80% treino / 20% validação

## Métricas

O sistema avalia os modelos usando:
- Acurácia (treino e teste)
- Matriz de confusão
- Classification report (precision, recall, f1-score)
