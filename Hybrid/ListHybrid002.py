from Base.BaseRecommender import BaseRecommender
import numpy as np


def _round_robin_classic(recommendations_list, cutoff=10):
    final_ranking = []
    idx = 0
    while len(final_ranking) < cutoff:
        for l in recommendations_list:
            if not l[idx] in final_ranking:
                final_ranking.append(l[idx])
        idx += 1
    return final_ranking[:cutoff]


def _weighted_majority_voting(recommendations_list, cutoff=10):
    items_dict = {}
    list_len= len(recommendations_list[0])
    for l in recommendations_list:
        for i in range(list_len):
            if l[i] in items_dict.keys():
                items_dict[l[i]] += 1/(i+1)
            else:
                items_dict[l[i]] = 1/(i+1)

    items_dict = dict(sorted(items_dict.items(), key=lambda item: item[1], reverse=True))
    return list(items_dict.keys())[:cutoff]


class ListHybrid002(BaseRecommender):
    def _compute_item_score(self, user_id_array, items_to_compute=None):
        raise NotImplementedError(
            f"{self.RECOMMENDER_NAME}: _compute_item_score not assigned for current recommender")

    def save_model(self, folder_path, file_name=None):
        raise NotImplementedError(
            f"{self.RECOMMENDER_NAME}: save_model not assigned for current recommender")

    RECOMMENDER_NAME = "ListHybrid002"

    def __init__(self, URM_train, recommender_list, seed=1205):
        super(ListHybrid002, self).__init__(URM_train)
        self.__recommender_list = recommender_list
        self.URM_train = URM_train

    def fit(self):
        pass

    def recommend(self, user_id_array, cutoff = None, remove_seen_flag=True, items_to_compute = None,
                  remove_top_pop_flag = False, remove_custom_items_flag = False, return_scores = False):

        # If is a scalar transform it in a 1-cell array
        single_user = False
        if np.isscalar(user_id_array):
            user_id_array = np.atleast_1d(user_id_array)
            single_user=True

        if cutoff is None:
            cutoff = self.URM_train.shape[1] - 1

        ranking_list = []

        for user_id in user_id_array:
            recommendations = []
            for rec in self.__recommender_list:
                recommendations.append(rec.recommend(user_id,
                                                     cutoff=cutoff*2,
                                                     remove_seen_flag=remove_seen_flag,
                                                     items_to_compute=items_to_compute,
                                                     remove_top_pop_flag=remove_top_pop_flag,
                                                     remove_custom_items_flag=remove_top_pop_flag,
                                                     return_scores=False))
            final_ranking = _weighted_majority_voting(recommendations)
            ranking_list.append(final_ranking)

        # Return single list for one user, instead of list of lists
        if single_user:
            ranking_list = ranking_list[0]
        return ranking_list