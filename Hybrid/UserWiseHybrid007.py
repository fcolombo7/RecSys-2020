from Base.BaseRecommender import BaseRecommender
import numpy as np

from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Base.DataIO import DataIO

from GraphBased.P3alphaRecommender import P3alphaRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython

class UserWiseHybrid007(BaseRecommender):
    RECOMMENDER_NAME = "UserWiseHybrid007" #4 Range

    def __init__(self, URM_train, ICM_train, submission=False, verbose=True, seed=1205):
        super(UserWiseHybrid007, self).__init__(URM_train, verbose=verbose)
        recommenders = {
            'rp3b': RP3betaRecommender(URM_train),
            'p3a': P3alphaRecommender(URM_train),
            'sen': SLIMElasticNetRecommender(URM_train),
            'sbpr': SLIM_BPR_Cython(URM_train),
            'icb': ItemKNNCBFRecommender(URM_train,ICM_train),
            'icf': ItemKNNCFRecommender(URM_train),
            'ucf': UserKNNCFRecommender(URM_train)
        }
        print("Fitting rp3b...")
        params = {'topK': 1000, 'alpha': 0.38192761611274967, 'beta': 0.0, 'normalize_similarity': False}
        recommenders['rp3b'].fit(**params)
        print("done.")
        print("Fitting p3a...")
        params = {'topK': 131, 'alpha': 0.33660811631883863, 'normalize_similarity': False}
        recommenders['p3a'].fit(**params)
        print("done.")
        print("Fitting sen...")
        params = {'topK': 992, 'l1_ratio': 0.004065081925341167, 'alpha': 0.003725005053334143}
        recommenders['sen'].fit(**params)
        print("done.")
        print("Fitting sbpr...")
        params = {'topK': 979, 'epochs': 130, 'symmetric': False, 'sgd_mode': 'adam', 'lambda_i': 0.004947329669424629,
                  'lambda_j': 1.1534760845071758e-05, 'learning_rate': 0.0001}
        recommenders['sbpr'].fit(**params)
        print("done.")
        print("Fitting icb...")
        params = {'topK': 65, 'shrink': 0, 'similarity': 'dice', 'normalize': True}
        recommenders['icb'].fit(**params)
        print("done.")
        print("Fitting icf...")
        params = {'topK': 55, 'shrink': 1000, 'similarity': 'asymmetric', 'normalize': True, 'asymmetric_alpha': 0.0}
        recommenders['icf'].fit(**params)
        print("done.")
        print("Fitting ucf...")
        params = {'topK': 190, 'shrink': 0, 'similarity': 'cosine', 'normalize': True}
        recommenders['ucf'].fit(**params)
        print("done.")

        self.__recommender_segmentation = [
            ((0,5), HiddenRecommender(URM_train, ICM_train, [
                recommenders['p3a'],
                recommenders['ucf'],
                recommenders['icb']
            ], submission=submission, verbose=verbose, seed=seed),
             {'alpha': 0.3987236515679141, 'l1_ratio': 0.15489605895390016}),

            ((5,10), HiddenRecommender(URM_train, ICM_train, [
                recommenders['rp3b'],
                recommenders['icb'],
                recommenders['sen']
            ], submission=submission, verbose=verbose, seed=seed),
             {'alpha': 1.0, 'l1_ratio': 0.3951763029766836}),

            ((10,22), HiddenRecommender(URM_train, ICM_train, [
                recommenders['icb'],
                recommenders['sen'],
                recommenders['sbpr']
            ], submission=submission, verbose=verbose, seed=seed),
             {'alpha': 0.7321778261479165, 'l1_ratio': 0.15333729621089734}),

            ((22,-1), HiddenRecommender(URM_train, ICM_train, [
                recommenders['icb'],
                recommenders['sen'],
                recommenders['sbpr']
            ], submission=submission, verbose=verbose, seed=seed),
             {'alpha': 0.7321778261479165, 'l1_ratio': 0.15333729621089734})
        ]

    def fit(self):
        for f_range, recommender, params in self.__recommender_segmentation:
            print(f"Fitting {recommender.RECOMMENDER_NAME} in {f_range}")
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


class HiddenRecommender(BaseItemSimilarityMatrixRecommender):
    RECOMMENDER_NAME = "HiddenRecommender"

    # set the seed equal to the one of the parameter search!!!!
    def __init__(self, URM_train, ICM_train, rec_list, submission=False, verbose=True, seed=1205):
        super(HiddenRecommender, self).__init__(URM_train, verbose = verbose)
        self.URM_train = URM_train
        self.ICM_train = ICM_train

        self.__rec1 = rec_list[0]
        self.__rec2 = rec_list[1]
        self.__rec3 = rec_list[2]

        self.__a = self.__b = self.__c = None
        self.seed = seed
        self.__submission = submission

    def fit(self, alpha=0.5, l1_ratio=0.5):
        self.__a = alpha * l1_ratio
        self.__b = alpha - self.__a
        self.__c = 1 - self.__a - self.__b

    def _compute_item_score(self, user_id_array, items_to_compute=None):

        item_weights_1 = self.__rec1._compute_item_score(user_id_array)
        item_weights_2 = self.__rec2._compute_item_score(user_id_array)
        item_weights_3 = self.__rec3._compute_item_score(user_id_array)

        item_weights = item_weights_1 * self.__a + item_weights_2 * self.__b + item_weights_3 * self.__c

        return item_weights

    def save_model(self, folder_path, file_name = None):
        if file_name is None:
            file_name = self.RECOMMENDER_NAME
        self._print("Saving model in file '{}'".format(folder_path + file_name))
        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = {})
        self._print("Saving complete")


