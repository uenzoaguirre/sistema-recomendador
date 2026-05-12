from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate, train_test_split
from db import get_ratings_data
from collections import defaultdict
import numpy as np


def evaluate_svd_crossval_and_metrics(k=10, threshold=3.5):
    ratings = get_ratings_data()
    reader = Reader(rating_scale=(ratings['rating'].min(), ratings['rating'].max()))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

    # ── 1. Validación cruzada RMSE / MAE ──────────────────────────────────────
    model = SVD(n_factors=20, n_epochs=20, reg_all=0.1, random_state=42)
    results = cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    # ── 2. Split para métricas de ranking ─────────────────────────────────────
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    model_metrics = SVD(n_factors=20, n_epochs=20, reg_all=0.1, random_state=42)
    model_metrics.fit(trainset)

    # Agrupar testset por usuario  {uid: [(iid, true_r), ...]}
    testset_by_user = defaultdict(list)
    for uid, iid, true_r in testset:
        testset_by_user[uid].append((iid, true_r))

    precisions, recalls = [], []
    recommended_items = set()
    all_items = set(ratings['movieId'].unique())

    for uid, items in testset_by_user.items():
        # ── CORRECCIÓN: predecir SOLO sobre los ítems del testset del usuario,
        #    no sobre todos los ítems no vistos (anti_testset).
        #    Esto simula: "dadas estas opciones, ¿cuáles pondría arriba?"
        preds = [(iid, model_metrics.predict(uid, iid).est) for iid, _ in items]

        # Ítems relevantes: los que el usuario realmente rateó alto
        relevant = {iid for iid, true_r in items if true_r >= threshold}
        if not relevant:
            continue

        # Top-k por predicción
        top_k = sorted(preds, key=lambda x: x[1], reverse=True)[:k]
        hits = sum(1 for iid, _ in top_k if iid in relevant)

        precisions.append(hits / k)
        recalls.append(hits / len(relevant))

        for iid, est in top_k:
            if est >= threshold:
                recommended_items.add(iid)

    coverage  = len(recommended_items) / len(all_items) if all_items else 0
    precision = np.mean(precisions) if precisions else 0
    recall    = np.mean(recalls)    if recalls    else 0

    return results, precision, recall, coverage


if __name__ == "__main__":
    results, precision, recall, coverage = evaluate_svd_crossval_and_metrics()

    print("\nResultados de validación cruzada:")
    print(f"  RMSE promedio : {results['test_rmse'].mean():.4f}  ← alto por distribución bimodal (1,2,4,5 sin 3)")
    print(f"  MAE promedio  : {results['test_mae'].mean():.4f}")

    print("\nMétricas de ranking (evaluadas solo sobre ítems del testset):")
    print(f"  Precision@10  : {precision:.4f}")
    print(f"  Recall@10     : {recall:.4f}")
    print(f"  Coverage      : {coverage:.4f}  ({coverage*100:.2f}% del catálogo)")

    print("""
Nota sobre el RMSE alto:
  Los ratings solo toman valores 1, 2, 4 y 5 (sin el 3).
  SVD predice valores continuos: cuando un ítem "vale" 4 pero predice 2.8,
  el error es ~1.2 aunque el ranking sea correcto.
  Para este dataset, Precision@k y Recall@k son métricas más representativas
  que RMSE/MAE.
""")
