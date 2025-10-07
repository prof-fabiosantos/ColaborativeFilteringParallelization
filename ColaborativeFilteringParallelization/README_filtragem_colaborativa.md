# 🧠 Filtragem Colaborativa com Paralelismo

Este projeto demonstra um sistema **básico de recomendação colaborativa** (Collaborative Filtering) implementado em **Python**, com **cálculo de similaridades em paralelo** para aumentar o desempenho em grandes volumes de dados (ex: 10.000 avaliações).

---

## ⚙️ Como funciona

### 1. Geração do Dataset
O script cria um **dataset sintético** com 1000 usuários e 100 itens, simulando notas de 1 a 5, incluindo valores ausentes (`NaN`) para representar itens não avaliados.

```python
ratings = pd.DataFrame(
    np.random.choice([1, 2, 3, 4, 5, np.nan], size=(n_users, n_items), p=[0.15, 0.15, 0.2, 0.2, 0.2, 0.1]),
    columns=[f"item_{i}" for i in range(n_items)],
    index=[f"user_{u}" for u in range(n_users)]
)
```

---

### 2. Tratamento dos Dados
Como o `scikit-learn` não aceita `NaN`, os valores ausentes são substituídos pela **média das notas do próprio usuário**, ou pela **média global** caso o usuário não tenha avaliado nada.

```python
ratings_filled = ratings.apply(lambda row: row.fillna(row.mean()), axis=1)
global_mean = ratings_filled.stack().mean()
ratings_filled = ratings_filled.fillna(global_mean)
```

Em seguida, os dados são **normalizados** para que as diferenças de escala não influenciem a similaridade:

```python
ratings_norm = pd.DataFrame(StandardScaler().fit_transform(ratings_filled),
                            index=ratings.index,
                            columns=ratings.columns)
```

---

### 3. Cálculo de Similaridades entre Usuários
A similaridade é medida usando o **Cosseno de Similaridade**, que compara o ângulo entre os vetores de avaliação dos usuários.

\[
\text{similaridade}(A, B) = \frac{A \cdot B}{||A|| \, ||B||}
\]

Cada usuário tem sua similaridade calculada com todos os outros:

```python
similarities = cosine_similarity(user_vector, ratings_norm.values)[0]
```

---

### 4. Paralelismo com `ProcessPoolExecutor`
O cálculo das similaridades é distribuído entre vários processos (núcleos da CPU), o que acelera a execução em datasets grandes.

```python
from concurrent.futures import ProcessPoolExecutor, as_completed

with ProcessPoolExecutor(max_workers=6) as executor:
    futures = {executor.submit(compute_similarity_for_user, u): u for u in ratings_norm.index}
```

Cada processo calcula a similaridade de um subconjunto de usuários de forma independente, retornando os resultados quando prontos.  
Isso permite aproveitar todos os **núcleos de CPU disponíveis**, reduzindo significativamente o tempo total.

---

### 5. Geração das Recomendações
Para um usuário específico:
1. Pegamos os **usuários mais similares (top_k)**.
2. Calculamos a média das notas deles.
3. Recomendamos itens que o usuário **ainda não avaliou (`NaN`)**.

```python
similar_users = user_similarities[user_id].head(top_k).index
similar_ratings = ratings.loc[similar_users].mean().sort_values(ascending=False)
already_rated = ratings.loc[user_id]
recommendations = similar_ratings[already_rated.isna()].head(n_recs)
```

---

## 🚀 Como Executar

### 1️⃣ Instalar dependências
```bash
pip install -r requirements.txt
```

### 2️⃣ Executar o script principal
```bash
python filtragemColaborativaParalelismo.py
```

### 3️⃣ Exemplo de saída esperada
```
🧠 Calculando similaridades em paralelo...

🎯 Recomendações para user_0:
item_12    4.8
item_34    4.6
item_47    4.5
item_05    4.3
item_89    4.2
```

---

## 📊 Benefícios do Paralelismo
| Técnica | Tempo Médio | Escalabilidade |
|----------|--------------|----------------|
| Cálculo sequencial | ❌ Lento para >1000 usuários | Baixa |
| `ProcessPoolExecutor` (CPU-bound) | ✅ Muito mais rápido | Alta |
| `ThreadPoolExecutor` | Ineficiente (GIL) | Média |

---

## 🧩 Conceitos-Chave

| Conceito | Descrição |
|-----------|------------|
| **Filtragem Colaborativa** | Técnica que recomenda itens com base nas preferências de usuários similares. |
| **Cosseno de Similaridade** | Mede o grau de semelhança entre dois vetores de avaliações. |
| **Paralelismo** | Divide o trabalho entre múltiplos processos, acelerando a execução. |
| **Normalização** | Padroniza os dados para que todos os usuários tenham escalas comparáveis. |

---

## 🧠 Próximos Passos
- Implementar **predição de notas** usando média ponderada pela similaridade.  
- Aplicar **métodos baseados em itens** em vez de usuários.  
- Usar **FAISS** ou **Annoy** para busca de vizinhos aproximados (recomendações em tempo real).  
- Integrar com **banco de dados** para armazenar perfis e histórico de recomendações.

---

## 👩‍💻 Autor
Desenvolvido como exemplo educacional para demonstrar **Filtragem Colaborativa + Processamento Paralelo em Python**.
