
from collections import defaultdict
from sklearn.metrics import ndcg_score
from scipy import sparse
from utils import get_top_n


def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """Return precision and recall at k metrics for each user"""

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.

        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return precisions, recalls


def get_conversion_rate(predictions, k=10):
    """conversion rate for each item, number of users in the test set and the prediction/number of users in the
    prediction """
    top_n = get_top_n(predictions, n=k)
    pred_items_users = defaultdict(list)
    for uid, user_ratings in top_n.items():
        for (iid, _) in user_ratings:
            if uid not in pred_items_users[iid]:
                pred_items_users[iid].append(uid)

    top_n = defaultdict(list)
    real_items_users = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings
    for uid, user_ratings in top_n.items():
        for (iid, _) in user_ratings:
            if uid not in real_items_users[iid]:
                real_items_users[iid].append(uid)
    all_in_items_users = defaultdict(list)
    for iid in real_items_users:
        all_in_items_users[iid].extend(list(set(real_items_users[iid]).intersection(set(pred_items_users[iid]))))
    sum_cr = 0
    for iid in pred_items_users:
        if not pred_items_users[iid]:
            continue
        if iid in all_in_items_users:
            sum_cr += len(all_in_items_users[iid]) / len(pred_items_users[iid])
    return sum_cr/len(pred_items_users)


def get_ndcg(predictions, k_highest_scores=None):
    """
    Calculates the ndcg (normalized discounted cumulative gain) from surprise predictions, using sklearn.metrics.ndcg_score and scipy.sparse

    Parameters:
    surprise_predictions (List of surprise.prediction_algorithms.predictions.Prediction): list of predictions
    k_highest_scores (positive integer): Only consider the highest k scores in the ranking. If None, use all.

    Returns:
    float in [0., 1.]: The averaged NDCG scores over all recommendations

    """
    r_uis = [p.r_ui for p in predictions]
    ests = [p.est for p in predictions]

    sparse_preds = sparse.coo_matrix(ests)
    sparse_vals = sparse.coo_matrix(r_uis)

    dense_preds = sparse_preds.toarray()
    dense_vals = sparse_vals.toarray()
    return ndcg_score(y_true=dense_vals, y_score=dense_preds, k=k_highest_scores)