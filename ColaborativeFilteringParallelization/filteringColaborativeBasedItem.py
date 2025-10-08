import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ===============================
# 1️⃣ Nova base com mais filmes
# ===============================
dados = {
    "Usuário": [
        "Ana", "Ana", "Ana",
        "Bruno", "Bruno", "Bruno",
        "Carla", "Carla", "Carla",
        "Daniel", "Daniel", "Daniel",
        "Eduarda", "Eduarda"
    ],
    "Filme": [
        "Matrix", "Titanic", "Avatar",
        "Matrix", "Avatar", "Vingadores",
        "Titanic", "Avatar", "Interestelar",
        "Matrix", "Interestelar", "Vingadores",
        "Vingadores", "Interestelar"
    ],
    "Nota": [
        5, 4, 4,
        5, 4, 4,
        5, 4, 5,
        4, 5, 5,
        5, 4
    ]
}

df = pd.DataFrame(dados)

# ===============================
# 2️⃣ Matriz usuário × item
# ===============================
matriz = df.pivot_table(index="Usuário", columns="Filme", values="Nota").fillna(0)
print("📈 Matriz Usuário × Filme:")
print(matriz)

# ===============================
# 3️⃣ Similaridade entre itens
# ===============================
similaridade = cosine_similarity(matriz.T)
sim_df = pd.DataFrame(similaridade, index=matriz.columns, columns=matriz.columns)
print("\n🤝 Similaridade entre filmes:")
print(sim_df.round(2))

# ===============================
# 4️⃣ Recomendações para um usuário
# ===============================
usuario = "Ana"
avaliacoes = matriz.loc[usuario]
filmes_avaliados = avaliacoes[avaliacoes > 0].index
print(f"\n🎬 Filmes que {usuario} já assistiu:", list(filmes_avaliados))

recomendacoes = pd.Series(0, index=matriz.columns)
for filme in filmes_avaliados:
    recomendacoes += sim_df[filme] * avaliacoes[filme]

# Remover filmes já vistos
recomendacoes = recomendacoes.drop(filmes_avaliados)
recomendacoes = recomendacoes.sort_values(ascending=False)

print(f"\n✨ Recomendações para {usuario}:")
print(recomendacoes.round(2))

