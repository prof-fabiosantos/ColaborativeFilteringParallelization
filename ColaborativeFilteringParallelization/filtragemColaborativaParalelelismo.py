import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# ============================
# 🎯 1. Cria dataset sintético (10.000 avaliações)
# ============================
n_users, n_items = 1000, 100
np.random.seed(42)

ratings = pd.DataFrame(
    np.random.choice(
        [1, 2, 3, 4, 5, np.nan],
        size=(n_users, n_items),
        p=[0.15, 0.15, 0.2, 0.2, 0.2, 0.1]
    ),
    columns=[f"item_{i}" for i in range(n_items)],
    index=[f"user_{u}" for u in range(n_users)]
)

# ============================
# 🧮 2. Preenche valores ausentes (NaN)
# ============================
# Substitui NaN pela média das notas do próprio usuário
ratings_filled = ratings.apply(lambda row: row.fillna(row.mean()), axis=1)

# Caso algum usuário tenha todas as notas NaN, usa a média global
global_mean = ratings_filled.stack().mean()
ratings_filled = ratings_filled.fillna(global_mean)

# ============================
# ⚖️ 3. Normaliza as notas (importante para cosseno)
# ============================
ratings_norm = pd.DataFrame(
    StandardScaler().fit_transform(ratings_filled),
    index=ratings.index,
    columns=ratings.columns
)

# ============================
# ⚙️ 4. Função para calcular similaridade de um usuário
# ============================
def compute_similarity_for_user(user_id):
    """Calcula as similaridades de um usuário com todos os outros."""
    user_vector = ratings_norm.loc[user_id].values.reshape(1, -1)
    similarities = cosine_similarity(user_vector, ratings_norm.values)[0]
    sim_df = pd.Series(similarities, index=ratings_norm.index)
    sim_df = sim_df.drop(user_id)  # remove auto-similaridade
    return user_id, sim_df.sort_values(ascending=False)

# ============================
# ⚡ 5. Paraleliza o cálculo
# ============================
def compute_all_similarities_parallel(n_workers=6):
    results = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(compute_similarity_for_user, u): u for u in ratings_norm.index}
        for future in as_completed(futures):
            try:
                user_id, sim_series = future.result()
                results[user_id] = sim_series
            except Exception as e:
                print(f"⚠️ Erro ao processar {futures[future]}: {e}")
    return results

# ============================
# 🧩 6. Gera recomendações
# ============================
def recommend_for_user(user_id, user_similarities, top_k=5, n_recs=5):
    """Gera recomendações para um usuário, com base nos top_k mais similares."""
    similar_users = user_similarities[user_id].head(top_k).index
    similar_ratings = ratings.loc[similar_users].mean().sort_values(ascending=False)
    already_rated = ratings.loc[user_id]

    # Seleciona apenas itens não avaliados (NaN) para recomendar
    mask = already_rated.isna()
    recommendations = similar_ratings[mask].head(n_recs)

    if recommendations.empty:
        return pd.Series(["⚠️ Nenhuma recomendação disponível"], index=["info"])

    return recommendations

# ============================
# 🚀 7. Execução principal
# ============================
if __name__ == "__main__":
    print("🧠 Calculando similaridades em paralelo...")
    user_similarities = compute_all_similarities_parallel(n_workers=6)

    # Exemplo: recomendações para um usuário específico
    user = "user_0"
    recs = recommend_for_user(user, user_similarities)

    print(f"\n🎯 Recomendações para {user}:\n{recs}")
