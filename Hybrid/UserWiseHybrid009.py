from Base.BaseRecommender import BaseRecommender
import numpy as np

from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Base.DataIO import DataIO
from Base.Recommender_utils import similarityMatrixTopK

from GraphBased.P3alphaRecommender import P3alphaRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from SLIM_ElasticNet.SSLIM_ElasticNet import SSLIMElasticNet


class UserWiseHybrid009(BaseRecommender):
    RECOMMENDER_NAME = "UserWiseHybrid009"

    def __init__(self, URM_train, ICM_train, submission=False, verbose=True, seed=1205):
        super(UserWiseHybrid009, self).__init__(URM_train, verbose=verbose)
        recommenders = {
            'rp3b': RP3betaRecommender(URM_train),
            'p3a': P3alphaRecommender(URM_train),
            'sen': SLIMElasticNetRecommender(URM_train),
            'sbpr': SLIM_BPR_Cython(URM_train),
            'icb': ItemKNNCBFRecommender(URM_train, ICM_train),
            'icf': ItemKNNCFRecommender(URM_train),
            'ucf': UserKNNCFRecommender(URM_train),
            'sslim': SSLIMElasticNet(URM_train, ICM_train)
        }

        params = {'topK': 1000, 'alpha': 0.38192761611274967, 'beta': 0.0, 'normalize_similarity': False}
        try:
            recommenders['rp3b'].load_model(f'stored_recommenders/seed_{str(seed)}_hybrid_sub/',
                                            f'{recommenders["rp3b"].RECOMMENDER_NAME}_for_sub')
            print(f"{recommenders['rp3b'].RECOMMENDER_NAME} loaded.")
        except:
            print(f"Fitting {recommenders['rp3b'].RECOMMENDER_NAME} ...")
            recommenders['rp3b'].fit(**params)
            recommenders['rp3b'].save_model(f'stored_recommenders/seed_{str(seed)}_hybrid_sub/',
                                            f'{recommenders["rp3b"].RECOMMENDER_NAME}_for_sub')
            print(f"done.")


        params = {'topK': 131, 'alpha': 0.33660811631883863, 'normalize_similarity': False}
        try:
            recommenders['p3a'].load_model(f'stored_recommenders/seed_{str(seed)}_hybrid_sub/',
                                            f'{recommenders["p3a"].RECOMMENDER_NAME}_for_sub')
            print(f"{recommenders['p3a'].RECOMMENDER_NAME} loaded.")
        except:
            print(f"Fitting {recommenders['p3a'].RECOMMENDER_NAME} ...")
            recommenders['p3a'].fit(**params)
            recommenders['p3a'].save_model(f'stored_recommenders/seed_{str(seed)}_hybrid_sub/',
                                            f'{recommenders["p3a"].RECOMMENDER_NAME}_for_sub')
            print(f"done.")

        params = {'topK': 992, 'l1_ratio': 0.004065081925341167, 'alpha': 0.003725005053334143}
        try:
            recommenders['sen'].load_model(f'stored_recommenders/seed_{str(seed)}_hybrid_sub/',
                                           f'{recommenders["sen"].RECOMMENDER_NAME}_for_sub')
            print(f"{recommenders['sen'].RECOMMENDER_NAME} loaded.")
        except:
            print(f"Fitting {recommenders['sen'].RECOMMENDER_NAME} ...")
            recommenders['sen'].fit(**params)
            recommenders['sen'].save_model(f'stored_recommenders/seed_{str(seed)}_hybrid_sub/',
                                           f'{recommenders["sen"].RECOMMENDER_NAME}_for_sub')
            print(f"done.")


        params = {'topK': 979, 'epochs': 130, 'symmetric': False, 'sgd_mode': 'adam', 'lambda_i': 0.004947329669424629,
                  'lambda_j': 1.1534760845071758e-05, 'learning_rate': 0.0001}
        try:
            recommenders['sbpr'].load_model(f'stored_recommenders/seed_{str(seed)}_hybrid_sub/',
                                           f'{recommenders["sbpr"].RECOMMENDER_NAME}_for_sub')
            print(f"{recommenders['sbpr'].RECOMMENDER_NAME} loaded.")
        except:
            print(f"Fitting {recommenders['sbpr'].RECOMMENDER_NAME} ...")
            recommenders['sbpr'].fit(**params)
            recommenders['sbpr'].save_model(f'stored_recommenders/seed_{str(seed)}_hybrid_sub/',
                                           f'{recommenders["sbpr"].RECOMMENDER_NAME}_for_sub')
            print(f"done.")


        params = {'topK': 65, 'shrink': 0, 'similarity': 'dice', 'normalize': True}
        try:
            recommenders['icb'].load_model(f'stored_recommenders/seed_{str(seed)}_hybrid_sub/',
                                            f'{recommenders["icb"].RECOMMENDER_NAME}_for_sub')
            print(f"{recommenders['icb'].RECOMMENDER_NAME} loaded.")
        except:
            print(f"Fitting {recommenders['icb'].RECOMMENDER_NAME} ...")
            recommenders['icb'].fit(**params)
            recommenders['icb'].save_model(f'stored_recommenders/seed_{str(seed)}_hybrid_sub/',
                                            f'{recommenders["icb"].RECOMMENDER_NAME}_for_sub')
            print(f"done.")


        params = {'topK': 55, 'shrink': 1000, 'similarity': 'asymmetric', 'normalize': True, 'asymmetric_alpha': 0.0}
        try:
            recommenders['icf'].load_model(f'stored_recommenders/seed_{str(seed)}_hybrid_sub/',
                                           f'{recommenders["icf"].RECOMMENDER_NAME}_for_sub')
            print(f"{recommenders['icf'].RECOMMENDER_NAME} loaded.")
        except:
            print(f"Fitting {recommenders['icf'].RECOMMENDER_NAME} ...")
            recommenders['icf'].fit(**params)
            recommenders['icf'].save_model(f'stored_recommenders/seed_{str(seed)}_hybrid_sub/',
                                           f'{recommenders["icf"].RECOMMENDER_NAME}_for_sub')
            print(f"done.")


        params = {'topK': 190, 'shrink': 0, 'similarity': 'cosine', 'normalize': True}
        try:
            recommenders['ucf'].load_model(f'stored_recommenders/seed_{str(seed)}_hybrid_sub/',
                                           f'{recommenders["ucf"].RECOMMENDER_NAME}_for_sub')
            print(f"{recommenders['ucf'].RECOMMENDER_NAME} loaded.")
        except:
            print(f"Fitting {recommenders['ucf'].RECOMMENDER_NAME} ...")
            recommenders['ucf'].fit(**params)
            recommenders['ucf'].save_model(f'stored_recommenders/seed_{str(seed)}_hybrid_sub/',
                                           f'{recommenders["ucf"].RECOMMENDER_NAME}_for_sub')
            print(f"done.")


        params = {'beta': 0.4849594591575789, 'topK': 1000, 'l1_ratio': 1e-05, 'alpha': 0.001}
        try:
            recommenders['sslim'].load_model(f'stored_recommenders/seed_{str(seed)}_hybrid_sub/',
                                           f'{recommenders["sslim"].RECOMMENDER_NAME}_for_sub')
            print(f"{recommenders['sslim'].RECOMMENDER_NAME} loaded.")
        except:
            print(f"Fitting {recommenders['sslim'].RECOMMENDER_NAME} ...")
            recommenders['sslim'].fit(**params)
            recommenders['sslim'].save_model(f'stored_recommenders/seed_{str(seed)}_hybrid_sub/',
                                           f'{recommenders["sslim"].RECOMMENDER_NAME}_for_sub')
            print(f"done.")


        self.__recommender_segmentation = [
            ((0, 3), HiddenMergedRecommender(URM_train, ICM_train, [
                recommenders['rp3b'],
                recommenders['icb'],
                recommenders['icf']
            ], submission=submission, verbose=verbose, seed=seed),
             {'alpha': 0.7276553525851246, 'l1_ratio': 0.6891035546237165, 'topK': 1000}),

            ((3, 5), HiddenLinearRecommender(URM_train, ICM_train, [
                recommenders['sslim'],
                recommenders['p3a'],
                recommenders['icb']
            ], submission=submission, verbose=verbose, seed=seed),
             {'alpha': 0.9847198829156348, 'l1_ratio': 0.9996962519963414}),

            ((5, 10), HiddenLinearRecommender(URM_train, ICM_train, [
                recommenders['icb'],
                recommenders['rp3b'],
                recommenders['sen']
            ], submission=submission, verbose=verbose, seed=seed),
             {'alpha': 0.9949623682515907, 'l1_ratio': 0.007879399002699851}),

            ((10, 17), HiddenLinearRecommender(URM_train, ICM_train, [
                recommenders['sslim'],
                recommenders['icb'],
                recommenders['ucf']
            ], submission=submission, verbose=verbose, seed=seed),
             {'alpha': 0.6461624491197696, 'l1_ratio': 0.7617220099582368}),

            ((17, 30), HiddenLinearRecommender(URM_train, ICM_train, [
                recommenders['sslim'],
                recommenders['p3a'],
                recommenders['icb']
            ], submission=submission, verbose=verbose, seed=seed),
             {'alpha': 0.8416340030829476, 'l1_ratio': 0.6651408407090509}),

            ((30, 100), HiddenLinearRecommender(URM_train, ICM_train, [
                recommenders['sslim'],
                recommenders['icb'],
                recommenders['icf']
            ], submission=submission, verbose=verbose, seed=seed),
             {'alpha': 0.996772013761913, 'l1_ratio': 0.7831508517025596}),

            ((100, 200), HiddenLinearRecommender(URM_train, ICM_train, [
                recommenders['sslim'],
                recommenders['rp3b'],
                recommenders['icb']
            ], submission=submission, verbose=verbose, seed=seed),
             {'alpha': 0.8416340030829476, 'l1_ratio': 0.6651408407090509}),

            ((200, -1), HiddenMergedRecommender(URM_train, ICM_train, [
                recommenders['sslim'],
                recommenders['p3a'],
                recommenders['icb']
            ], submission=submission, verbose=verbose, seed=seed),
             {'alpha': 0.859343616443417, 'l1_ratio': 0.8995038091652459, 'topK': 900}),
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
                #print (f_range, recommender.RECOMMENDER_NAME)
                return recommender
        raise ValueError(
            f"{self.RECOMMENDER_NAME}: there is no recommender for users with profile length equal to {user_profile_length}.")


class HiddenLinearRecommender(BaseItemSimilarityMatrixRecommender):
    RECOMMENDER_NAME = "HiddenRecommender"

    # set the seed equal to the one of the parameter search!!!!
    def __init__(self, URM_train, ICM_train, rec_list, submission=False, verbose=True, seed=1205):
        super(HiddenLinearRecommender, self).__init__(URM_train, verbose = verbose)
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

        # normalization
        item_weights_1_max = item_weights_1.max()
        item_weights_2_max = item_weights_2.max()
        item_weights_3_max = item_weights_3.max()

        if not item_weights_1_max == 0:
            item_weights_1 /= item_weights_1_max
        if not item_weights_2_max == 0:
            item_weights_2 /= item_weights_2_max
        if not item_weights_3_max == 0:
            item_weights_3 /= item_weights_3_max

        item_weights = item_weights_1 * self.__a + item_weights_2 * self.__b + item_weights_3 * self.__c

        return item_weights

    def save_model(self, folder_path, file_name = None):
        if file_name is None:
            file_name = self.RECOMMENDER_NAME
        self._print("Saving model in file '{}'".format(folder_path + file_name))
        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = {})
        self._print("Saving complete")


class HiddenMergedRecommender(BaseItemSimilarityMatrixRecommender):
    RECOMMENDER_NAME = "HiddenMergedRecommender"

    # set the seed equal to the one of the parameter search!!!!
    def __init__(self, URM_train, ICM_train, rec_list, submission=False, verbose=True, seed=1205):
        super(HiddenMergedRecommender, self).__init__(URM_train, verbose = verbose)
        self.URM_train = URM_train
        self.ICM_train = ICM_train

        self.__rec1 = rec_list[0]
        self.__rec2 = rec_list[1]
        self.__rec3 = rec_list[2]

        if (self.__rec1.W_sparse.shape != self.__rec2.W_sparse.shape) or \
                (self.__rec1.W_sparse.shape != self.__rec3.W_sparse.shape):
            raise ValueError(
                f"HybridCombinationMergedSearch({self.__rec1.RECOMMENDER_NAME} - {self.__rec2.RECOMMENDER_NAME} - "
                f"{self.__rec3.RECOMMENDER_NAME}): similarities have different size, S1 is "
                f"{self.__rec1.W_sparse.shape}, S2 is {self.__rec2.W_sparse.shape}, S3 is {self.__rec3.W_sparse.shape}")

        # self.__W1 = check_matrix(self.__rec1.W_sparse.copy(), 'csr')
        # self.__W2 = check_matrix(self.__rec2.W_sparse.copy(), 'csr')
        # self.__W3 = check_matrix(self.__rec3.W_sparse.copy(), 'csr')

        self.__a = self.__b = self.__c = None
        self.topK = None
        self.W_sparse = None
        self.seed = seed

    def fit(self, alpha=0.5, l1_ratio=0.5, topK=100):
        self.__a = alpha * l1_ratio
        self.__b = alpha - self.__a
        self.__c = 1 - self.__a - self.__b
        self.topK = topK

        W1_max = self.__rec1.W_sparse.max()
        W2_max = self.__rec2.W_sparse.max()
        W3_max = self.__rec3.W_sparse.max()

        W1 = self.__rec1.W_sparse
        W2 = self.__rec2.W_sparse
        W3 = self.__rec3.W_sparse
        if W1_max != 0:
            W1 = W1 / W1_max
        if W2_max != 0:
            W2 = W2 / W2_max
        if W3_max != 0:
            W3 = W3 / W3_max

        W = W1 * self.__a + W2 * self.__b + W3 * self.__c

        self.W_sparse = similarityMatrixTopK(W, k=self.topK).tocsr()


    def save_model(self, folder_path, file_name = None):
        if file_name is None:
            file_name = self.RECOMMENDER_NAME
        self._print("Saving model in file '{}'".format(folder_path + file_name))
        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = {})
        self._print("Saving complete")

