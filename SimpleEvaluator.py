import numpy as np
import scipy.sparse as sp

def recall(recommendations: np.array, relevant_items: np.array) -> float:
    is_relevant = np.in1d(recommendations, relevant_items, assume_unique=True)

    recall_score = np.sum(is_relevant) / relevant_items.shape[0]

    return recall_score


def precision(recommendations: np.array, relevant_items: np.array) -> float:
    is_relevant = np.isin(recommendations, relevant_items, assume_unique=True)
    precision_score = np.sum(is_relevant) / recommendations.shape[0]

    return precision_score


def mean_average_precision(recommendations: np.array, relevant_items: np.array) -> float:
    is_relevant = np.isin(recommendations, relevant_items, assume_unique=True)
    precision_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))
    map_score = np.sum(precision_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])

    return map_score


def evaluator(recommender: object, urm_test:sp.csr_matrix, cutoff = 10):
    recommendation_length = cutoff
    accum_precision = 0
    accum_recall = 0
    accum_map = 0
    urm_train = recommender.get_URM_train()
    num_users = urm_train.shape[0]

    num_users_evaluated = 0
    num_users_skipped = 0
    for user_id in range(num_users):
        user_profile_start = urm_test.indptr[user_id]
        user_profile_end = urm_test.indptr[user_id + 1]

        relevant_items = urm_test.indices[user_profile_start:user_profile_end]

        if relevant_items.size == 0:
            num_users_skipped += 1
            continue

        recommendations = recommender.recommend(user_id_array=user_id,
                                                cutoff=recommendation_length)

        accum_precision += precision(np.array(recommendations), relevant_items)
        accum_recall += recall(np.array(recommendations), relevant_items)
        accum_map += mean_average_precision(np.array(recommendations), relevant_items)

        num_users_evaluated += 1

    accum_precision /= max(num_users_evaluated, 1)
    accum_recall /= max(num_users_evaluated, 1)
    accum_map /= max(num_users_evaluated, 1)

    result_dict = {'MAP': accum_map, 'Precision':accum_precision, 'Recall':accum_recall}

    return result_dict
