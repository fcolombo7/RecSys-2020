from Base.BaseRecommender import BaseRecommender
import numpy as np

from Hybrid.LinearHybrid005 import LinearHybrid005
from Hybrid.LinearHybridC001 import LinearHybridC001
from Hybrid.LinearHybridC002 import LinearHybridC002
from Hybrid.LinearHybridW001 import LinearHybridW001


class UserWiseHybrid003(BaseRecommender):
    RECOMMENDER_NAME = "UserWiseHybrid002"

    def __init__(self, URM_train, ICM_train, submission=False, verbose=True, seed=1205):
        super(UserWiseHybrid003, self).__init__(URM_train, verbose=verbose)
        self.__recommender_segmentation = [
            ((0, 3),
             LinearHybridC002(URM_train, ICM_train, submission=submission, verbose=verbose, seed=seed),
             {'alpha': 0.904995781837142, 'l1_ratio': 0.824227167167697}),
            ((3, -1),
             LinearHybridW001(URM_train, ICM_train, submission=submission, verbose=verbose, seed=seed),
             {'alpha': 0.6455264484875096, 'l1_ratio': 0.7800132947163301}),
        ]

    def fit(self):
        for _, recommender, params in self.__recommender_segmentation:
            print("Fitting "+recommender.RECOMMENDER_NAME)
            recommender.fit(**params)
            print("Done.")

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
        for f_range, recommender, _ in self.__recommender_segmentation:
            if user_profile_length >= f_range[0] and (user_profile_length < f_range[1] or f_range[1] == -1):
                return recommender
        raise ValueError(
            f"{self.RECOMMENDER_NAME}: there is no recommender for users with profile length equal to {user_profile_length}.")
