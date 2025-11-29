# 🏦 Sistema de Análise de Risco de Crédito com IA

## Descrição do Projeto

Este projeto implementa um **sistema completo de análise de risco de crédito** utilizando técnicas avançadas de Machine Learning e Deep Learning. O sistema foi desenvolvido como trabalho final da disciplina de Inteligência Artificial e combina:

- **3 modelos de IA diferentes** (Regressão Logística, Random Forest, MLP)
- **Interface web** construída com Streamlit
- **Análise explicativa via LLM** (Qwen 3-1.7B) para interpretação dos resultados
- **Pipeline completo de ML** desde o pré-processamento até a inferência
- **Arquitetura modular** seguindo boas práticas de engenharia de software

O sistema permite que usuários insiram dados de clientes e recebam instantaneamente:
1. Predições de 3 modelos diferentes
2. Probabilidades de aprovação
3. Análise explicativa em português gerada por IA

## Objetivos

- Implementar e comparar diferentes algoritmos de classificação binária
- Avaliar performance de ML tradicional vs. Deep Learning
- Criar uma aplicação web funcional para inferência em tempo real
- Integrar Large Language Models para explicabilidade
- Aplicar padrões de projeto e boas práticas de engenharia de software

## Dataset

### German Credit Data

O projeto utiliza o **German Credit Data**, um dataset clássico para problemas de credit scoring.

#### Características do Dataset
- **Total de amostras**: 1.000 registros
- **Features**: 20 variáveis preditoras
- **Target**: `Creditability` (1 = bom pagador, 0 = mau pagador)
- **Formato**: CSV
- **Fonte**: `data/german.csv`

#### Variáveis do Dataset

**Features Numéricas** (6 variáveis):
- `Duration_of_Credit_monthly`: Duração do crédito em meses (1-500)
- `Credit_Amount`: Valor do crédito solicitado (0-1.000.000)
- `Instalment_per_cent`: Percentual de parcela (0-100)
- `Duration_in_Current_address`: Tempo no endereço atual (0-100)
- `Age_years`: Idade do cliente (16-120)
- `No_of_dependents`: Número de dependentes (0-50)

**Features Categóricas** (14 variáveis):
- `Account_Balance`: Saldo em conta (1=Sem conta, 2=Sem saldo, 3=Com saldo)
- `Payment_Status_of_Previous_Credit`: Histórico de pagamentos
- `Purpose`: Propósito do empréstimo (0-10: Carro novo, Usado, Móveis, TV/Rádio, etc.)
- `Value_Savings_Stocks`: Valor em poupança/investimentos
- `Length_of_current_employment`: Tempo de emprego atual
- `Sex_Marital_Status`: Gênero e estado civil
- `Guarantors`: Presença de fiadores
- `Most_valuable_available_asset`: Ativo mais valioso
- `Concurrent_Credits`: Créditos concorrentes
- `Type_of_apartment`: Tipo de moradia
- `No_of_Credits_at_this_Bank`: Número de créditos neste banco
- `Occupation`: Ocupação profissional
- `Telephone`: Possui telefone
- `Foreign_Worker`: Trabalhador estrangeiro

**Pré-processamento:** Retirada a feature categórica de `Foreign_Worker` por poder fornecer análise discriminnatória.

#### Distribuição de Classes
- **Classe 1 (Bom crédito)**: ~70% dos casos
- **Classe 0 (Mau crédito)**: ~30% dos casos
- Dataset levemente desbalanceado.

---

## Arquitetura do Sistema

```
trabalho-ia/
│
├── 📁 data/
│   └── german.csv                    # Dataset original (1000 registros)
│
├── 📁 models/                        # Modelos de ML
│   ├── base_model.py                 # ABC - Interface base
│   ├── logistic_model.py             # Regressão Logística
│   ├── ensemble_model.py             # Random Forest
│   ├── mlp_model.py                  # Multi-Layer Perceptron (PyTorch)
│   │
│   ├── 📁 regression/                # Modelos treinados
│   │   └── logistic.pkl              # Modelo serializado
│   ├── 📁 ensemble/
│   │   └── random_forest.pkl
│   └── 📁 mlp/
│       ├── mlp.pt                    # Pesos da rede neural
│       └── loss_curve.png            # Visualização do treinamento
│
├── 📁 train_pipeline/                # Pipeline de treinamento
│   ├── pipeline.py                   # Script principal de treinamento
│   └── 📁 preprocessor/
│       └── create_preprocessor.py    # Geração do preprocessador
│
├── 📁 app/                           # Aplicação Streamlit
│   ├── __init__.py
│   ├── constants.py                  # Constantes e configurações
│   ├── utils.py                      # Funções utilitárias
│   ├── qwen_3.py                     # Integração com LLM Qwen
│   │
│   ├── 📁 pages/
│   │   └── main.py                   # Interface principal
│   │
│   └── 📁 llm/
│       └── __init__.py
│
├── preprocessor.joblib               # Preprocessador treinado
├── .gitignore
├── LICENSE                           # GPL v3
└── README.md                         # Este arquivo
```

---

## Tecnologias e Dependências

### Stack Principal

#### Machine Learning & Deep Learning
- **PyTorch**: Framework para redes neurais profundas
  - Implementação do MLP customizado
  - Suporte a GPU/CPU automático
  
- **Scikit-learn**: Biblioteca de ML tradicional
  - Random Forest Classifier
  - Logistic Regression
  - Preprocessamento (StandardScaler, OneHotEncoder)
  - ColumnTransformer para pipelines
  - Métricas de avaliação

#### Dados e Processamento
- **Pandas**: Manipulação e análise de dados
  - Leitura de CSV
  - Transformações de dataframes
  
- **NumPy**: Operações numéricas de alta performance

#### Interface Web
- **Streamlit**: Framework para aplicações web de ML
  - Interface interativa responsiva
  - Inputs dinâmicos baseados em features
  - Visualização de resultados em tempo real

#### LLM e NLP
- **Transformers (HuggingFace)**: Biblioteca para modelos de linguagem
  - Qwen 3-1.7B para análise explicativa
  - Geração de texto em português

#### Utilitários
- **Joblib**: Serialização eficiente de modelos scikit-learn
- **Matplotlib**: Visualização de curvas de treinamento
- **Pathlib**: Manipulação moderna de caminhos

### Instalação de Dependências

```bash

pip install -r requirements.txt

```

## Modelos de Machine Learning

### 1. Regressão Logística (`LogisticModel`)

#### Descrição
Modelo linear baseline que estabelece a performance mínima esperada.

#### Arquitetura
- **Algoritmo**: Regressão Logística com regularização L2
- **Solver**: LBFGS (Limited-memory BFGS)
- **Max Iterações**: 1.000
- **Função de ativação**: Sigmoid

#### Armazenamento
- **Localização**: `models/regression/logistic.pkl`

### 2. Random Forest (`EnsembleModel`)

#### Descrição
Ensemble de árvores de decisão que captura relações não-lineares complexas.

#### Arquitetura
- **N Estimadores**: 200 árvores
- **Critério**: Gini Impurity
- **Max Features**: `sqrt(n_features)`
- **Bootstrap**: Ativado

#### Armazenamento
- **Localização**: `models/ensemble/random_forest.pkl`

### 3. Multi-Layer Perceptron (`MLPModel`)

#### Descrição
Rede neural profunda implementada em PyTorch para aprendizado de representações complexas.

#### Hiperparâmetros de Treinamento
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Batch Size**: 32
- **Epochs**: 20
- **Loss Function**: Binary Cross-Entropy (BCELoss)
- **Device**: Auto-detect (CUDA/CPU)

---

#### Monitoramento
Durante o treinamento, o modelo:
1. Calcula loss no conjunto de treino
2. Valida no conjunto de validação a cada época
3. Imprime métricas em tempo real
4. Gera gráfico de convergência

#### Visualização
Curva de loss salva em: `models/mlp/loss_curve.png`

#### Armazenamento
- **Localização**: `models/mlp/mlp.pt`
- **Formato**: PyTorch state_dict
- **Conteúdo**: Pesos e biases de todas as camadas

---

## Aplicação Web 

### Visão Geral
Interface web construída com **Streamlit** que permite análise de crédito em tempo real.

### Funcionalidades

#### 1. **Entrada de Dados Dinâmica**
- **Sidebar interativa** com todos os 20 campos do dataset
- **Validação automática** de tipos e ranges
- **Inputs adaptivos**:
  - Campos numéricos: `number_input` com min/max
  - Campos categóricos: `selectbox` com opções pré-definidas
- **Labels humanizados**: Substituição de underscores por espaços

#### 2. **Validação de Dados**
```python
def validate_row(df_row: pd.DataFrame) -> Tuple[bool, str]:
    # Verifica campos obrigatórios
    # Valida ranges numéricos
    # Retorna (sucesso, mensagem_erro)
```
#### 3. **Carregamento Inteligente de Modelos**
A função `load_models_via_load_method()` implementa:

- **Busca em múltiplos caminhos**
  ```python
  MODEL_FILENAMES = {
      "logistic": [
          "models/regression/logistic.pkl",
          "train_pipeline/models/logistic.pkl",
          ...
      ],
      ...
  }
  ```

- **Suporte a múltiplos formatos**
  - Scikit-learn: `.pkl`, `.joblib`
  - PyTorch: `.pt`, `.pth`

- **Fallback automático**
  - Tenta carregar via método `load()` da classe
  - Se falhar, carrega arquivo diretamente
  - Compatível com modelos de diferentes versões

#### 4. **Pré-processamento Automático**
```python
preprocessor = load_preprocessor(find_preprocessor_path())
X_transformed = preprocessor.transform(input_df)
```

- Busca automática do preprocessador em múltiplos locais
- Aplicação das mesmas transformações do treinamento
- Garantia de compatibilidade dimensional

#### 5. **Predição Multi-Modelo**
```python
results_df = predict_all(preprocessor, models, input_df)
```

**Retorna dataframe com:**
```
model       | prediction | probability
------------|------------|------------
logistic    | Good       | 0.73
ensemble    | Good       | 0.81
mlp         | Bad        | 0.42
```

#### 6. **Visualização de Resultados**
- **Tabela formatada** com predições de todos os modelos
- **Probabilidades** em formato decimal (0-1)
- **Labels intuitivas**: "Good" (bom pagador) vs "Bad" (mau pagador)

---

## Integração com LLM (Qwen)

### Modelo Utilizado
**Qwen 3-1.7B** - Large Language Model da Alibaba

### Características
- **Parâmetros**: 1.7 bilhões
- **Contexto**: Até 32K tokens
- **Idioma**: Multilíngue com excelente suporte ao português
- **Device**: Auto-detect (GPU/CPU)

### Funcionalidade

#### Análise Explicativa Automática
Após as predições, o LLM gera um texto explicativo em português analisando:

1. **Consenso entre modelos**
   - Todos concordam? → Alta confiança
   - Discordância? → Caso marginal

2. **Fatores relevantes**
   - Idade do cliente
   - Valor do crédito
   - Histórico de pagamentos
   - Propósito do empréstimo

3. **Justificativa da decisão**
   - Pontos positivos encontrados
   - Fatores de risco identificados
   - Recomendação final

#### Implementação

**Prompt Engineering:**
```python
def answer(model, tokenizer, classification: pd.DataFrame, input_df: pd.DataFrame):
    prompt = """
    Você receberá informações sobre um aplicante a linha de crédito,
    e a classificação final do mesmo. Você deverá fazer um texto bem
    curto e resumido apresentando possíveis razões para os resultados.
    
    Classificações:
    {classificações dos 3 modelos}
    
    Inputs do usuário:
    {20 features do cliente}
    """
```
--- 

## Padrões de Projeto

### 1. Abstract Base Class (ABC)

#### Implementação
```python
from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def fit(self, X_train, y_train, X_val, y_val):
        """Treina o modelo e avalia no conjunto de validação."""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Realiza predições em novos dados."""
        pass
    
    @abstractmethod
    def save(self, path):
        """Serializa o modelo treinado."""
        pass
```

#### Benefícios
- **Interface consistente**: Todos os modelos têm os mesmos métodos
- **Polimorfismo**: Modelos intercambiáveis sem alterar código cliente
- **Extensibilidade**: Novos modelos seguem o contrato automaticamente
- **Type safety**: IDE detecta métodos faltantes em tempo de desenvolvimento

#### Uso
```python
models: List[BaseModel] = [
    LogisticModel(),
    EnsembleModel(),
    MLPModel(input_dim=61)
]

for model in models:
    model.fit(X_train, y_train, X_val, y_val)  # Interface uniforme
    model.save(f"models/{model.__class__.__name__}.pkl")
```

### 2. Encapsulamento

Cada modelo encapsula:
- **Estado interno**: `self.model`, `self.history`, `self.device`
- **Lógica de treinamento**: Implementação específica do algoritmo
- **Serialização**: Formato adequado (pickle vs. PyTorch state_dict)

## 🔗 Deploy

### Link do Deploy


## 📚 Referências e Recursos

### Datasets
- **German Credit Data**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))

### Frameworks e Bibliotecas
- **PyTorch**: https://pytorch.org/docs/
- **Scikit-learn**: https://scikit-learn.org/stable/
- **Streamlit**: https://docs.streamlit.io/
- **Transformers**: https://huggingface.co/docs/transformers/

### Artigos e Tutoriais
- Credit Scoring com ML: [Towards Data Science](https://towardsdatascience.com/)
- Binary Classification Best Practices: [Google ML Guide](https://developers.google.com/machine-learning)

---

## 📄 Licença

Este projeto está licenciado sob a **GNU General Public License v3.0**.

Principais permissões:
- ✅ Uso comercial
- ✅ Modificação
- ✅ Distribuição
- ✅ Uso privado

Principais condições:
- ⚠️ Código fonte deve ser disponibilizado
- ⚠️ Mesma licença deve ser mantida
- ⚠️ Mudanças devem ser documentadas

Veja o arquivo [LICENSE](LICENSE) para detalhes completos.

---

## 👥 Autores

**Disciplina**: Inteligência Artificial  
**Período**: Novembro 2025  

---

**Última atualização**: 29 de Novembro de 2025  
**Versão**: 2.0.0
