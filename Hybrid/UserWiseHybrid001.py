from Base.BaseRecommender import BaseRecommender
from Base.NonPersonalizedRecommender import TopPop
from GraphBased.P3alphaRecommender import P3alphaRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender

import numpy as np
import os as os


class UserWiseHybrid001(BaseRecommender):
    """
    Class representing a UserWise hybrid recommender.
    Description: (range => recommender)
    > [0, 1) => TopPop (?)
    > [1, 25) => P3alpha (?)
    > [25, 100) => RP3beta (?)
    > [100, 200) => RP3beta (?)
    > [200, end) => RP3beta (?)
    """
    RECOMMENDER_NAME = "UserWiseHybrid001"

    def __init__(self, URM_train, ICM_all, verbose=True):
        super(UserWiseHybrid001, self).__init__(URM_train, verbose=verbose)
        self.URM_train = URM_train
        self.ICM_all = ICM_all

        # range and recommender definition ---> element structure: ( (start, end), recommender, fit_args* )
        self.__recommender_segmentation = [
            ((0, 1), TopPop(URM_train), {}),
            ((1, 25), P3alphaRecommender(URM_train),
             {'topK': 729, 'alpha': 0.4104229220476686, 'normalize_similarity': False}),
            ((25, 100), RP3betaRecommender(URM_train),
             {'topK': 1000, 'alpha': 0.35039375652835403, 'beta': 0.0, 'normalize_similarity': False}),
            ((100, 200), RP3betaRecommender(URM_train),
             {'topK': 1000, 'alpha': 0.32110178834628456, 'beta': 0.0, 'normalize_similarity': True}),
            ((200, -1), RP3betaRecommender(URM_train),
             {'topK': 272, 'alpha': 0.4375286551430567, 'beta': 0.16773603374534, 'normalize_similarity': True}),
        ]
        self.loaded = False

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
            f"{self.RECOMMENDER_NAME}: compute_item_score not assigned for current recommender, unable to compute prediction scores")

    def save_model(self, folder_path, file_name=None):
        if file_name is None:
            file_name = self.RECOMMENDER_NAME
        model_root_folder = folder_path + '/' + file_name
        self._print("Saving the sub-models in directory '{}'".format(folder_path + file_name))
        if not os.path.exists(model_root_folder):
            os.makedirs(model_root_folder)

        for f_range, recommender, _ in self.__recommender_segmentation:
            rec_fn = f_range[0] + '-' + f_range[1] + '_' + recommender.RECOMMENDER_NAME
            recommender.save_model(model_root_folder + '/', rec_fn)

    def load_model(self, folder_path, file_name=None):
        if file_name is None:
            file_name = self.RECOMMENDER_NAME
        model_root_folder = folder_path + '/' + file_name
        try:
            for f_range, recommender, _ in self.__recommender_segmentation:
                rec_fn = f_range[0] + '-' + f_range[1] + '_' + recommender.RECOMMENDER_NAME
                recommender.load_model(model_root_folder + '/', rec_fn)

            self.loaded = True
            print("INFO: All the models have been loaded.")
        except:
            print("WARNING: Errors occur in loading the model. Recommenders will be trained.")
            self.loaded = False

    def fit(self):
        if not self.loaded:
            print("INFO: Start fitting the recommenders... ")
            for f_range, recommender, best_parameters in self.__recommender_segmentation:
                print(f"Fitting {recommender.RECOMMENDER_NAME} [{f_range[0]} - {f_range[1]}]")
                recommender.fit(**best_parameters)


    def __get_recommender_by_profile_length(self, user_profile_length):
        for f_range, recommender, _ in self.__recommender_segmentation:
            if user_profile_length >= f_range[0] and (user_profile_length < f_range[1] or f_range[1] == -1):
                return recommender
        raise ValueError(f"{self.RECOMMENDER_NAME}: there is no recommender for users with profile length equal to {user_profile_length}.")
