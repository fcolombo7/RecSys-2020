from Base.BaseRecommender import BaseRecommender
from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Base.NonPersonalizedRecommender import TopPop
from Hybrid.LinearHybrid005 import LinearHybrid005

import numpy as np


class ColdUserEmbedding002(BaseRecommender):
    RECOMMENDER_NAME = "ColdUserEmbedding002"

    def __init__(self, URM_train, ICM_train, submission=False, verbose=True, seed=1205):
        super(ColdUserEmbedding002, self).__init__(URM_train, verbose=verbose)
        self.__warm_recommender = LinearHybrid005(URM_train, ICM_train, submission=submission, verbose=verbose,
                                                  seed=seed)
        self.__warm_params = {'alpha': 0.3553383791480798, 'l1_ratio': 0.000435281815357902}
        self.__cold_recommender = TopPop(URM_train)
        self.__cold_params = {}

    def fit(self):
        self.__cold_recommender.fit()
        self.__warm_recommender.fit(**self.__warm_params)

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None,
                  remove_top_pop_flag=False, remove_custom_items_flag=False, return_scores=False):
        # If is a scalar transform it in a 1-cell array
        if np.isscalar(user_id_array):
            user_id_array = np.atleast_1d(user_id_array)
            single_user = True
        else:
            single_user = False

        if cutoff is None:
            cutoff = self.URM_train.shape[1] - 1

        ranking_list = []
        for user_id in user_id_array:
            user_profile_length = len(self.URM_train[user_id].data)
            recommender = self.__get_recommender_by_profile_length(user_profile_length)
            ranking_list.append(recommender.recommend(user_id,
                                                      cutoff=cutoff,
                                                      remove_seen_flag=remove_seen_flag,
                                                      items_to_compute=items_to_compute,
                                                      remove_top_pop_flag=remove_top_pop_flag,
                                                      remove_custom_items_flag=remove_custom_items_flag,
                                                      return_scores=False)
                                )
        # Return single list for one user, instead of list of lists
        if single_user:
            ranking_list = ranking_list[0]
        return ranking_list

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        raise NotImplementedError(
            f"{self.RECOMMENDER_NAME}: compute_item_score not assigned for current recommender, unable to compute "
            f"prediction scores")

    def save_model(self, folder_path, file_name=None):
        raise NotImplementedError(
            f"{self.RECOMMENDER_NAME}: save_model not assigned for current recommender, unable to save the model")

    def __get_recommender_by_profile_length(self, user_profile_length):
        if user_profile_length < 1:
            return self.__cold_recommender
        return self.__warm_recommender
