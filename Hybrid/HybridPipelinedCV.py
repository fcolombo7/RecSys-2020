from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Base.DataIO import DataIO
from Base.Evaluation.Evaluator import EvaluatorHoldout
from Base.NonPersonalizedRecommender import TopPop
from DataParser import DataParser
from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from GraphBased.RP3betaRecommender import RP3betaRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
import numpy as np
from Base.Recommender_utils import check_matrix, similarityMatrixTopK

class HybridPipelinedCV(BaseItemSimilarityMatrixRecommender):
    RECOMMENDER_NAME = "HybridPipelinedCV"

    # set the seed equal to the one of the parameter search!!!!
    def __init__(self, URM_train, ICM_train, rec_couple, seed=None, fold=None, submission=False, verbose=True):
        """
                :params list_rec è una lista composta da (RecommenderClass, **fit_keywargs)
                """
        assert (seed is not None and fold is not None) or submission is True

        super(HybridPipelinedCV, self).__init__(URM_train, verbose=verbose)
        self.URM_train = URM_train
        self.ICM_train = ICM_train

        self.submission = submission
        self.seed = seed
        self.fold = fold
        self.rec_couple = rec_couple

        self.__rec1_class = rec_couple[0][0]
        self.__rec1_keywargs = rec_couple[0][1]

        self.__rec2_class = rec_couple[1][0]
        self.__rec2_keywargs = rec_couple[1][1]

        ### CONSTRUCT the 3 recs
        try:
            self.__rec1 = self.__rec1_class(URM_train, ICM_train, verbose=False)
        except:
            self.__rec1 = self.__rec1_class(URM_train, verbose=False)


    def fit(self, topK=100, normalize=False):
        self.topK = topK
        folder = 'for_sub' if self.submission else 'hybrid_search'
        filename = 'fors_sub' if self.submission else f'{str(self.seed)}_fold-{str(self.fold)}'
        # load the models if already trained for that particular seed and fold
        try:
            self.__rec1.load_model(
                f'stored_recommenders/seed_{str(self.seed)}_{folder}/{self.__rec1.RECOMMENDER_NAME}/', filename)
            print(f"{self.__rec1.RECOMMENDER_NAME} loaded. [seed={self.seed}, fold={self.fold}]")
        except:
            print(f"Fitting {self.__rec1.RECOMMENDER_NAME} ... [seed={self.seed}, fold={self.fold}]")
            self.__rec1.fit(**self.__rec1_keywargs)
            print(f"done.")
            self.__rec1.save_model(
                f'stored_recommenders/seed_{str(self.seed)}_{folder}/{self.__rec1.RECOMMENDER_NAME}/', filename)

        w_sparse = self.__rec1.W_sparse
        w_sparse = similarityMatrixTopK(w_sparse, k=self.topK).tocsr()

        URM_train = self.URM_train.dot(w_sparse)
        try:
            self.__rec2 = self.__rec2_class(URM_train, self.ICM_train, verbose=False)
        except:
            self.__rec2 = self.__rec2_class(URM_train, verbose=False)

        print(f"Fitting {self.__rec2.RECOMMENDER_NAME}... [topk={topK}, seed={self.seed}, fold={self.fold}]")
        self.__rec2.fit(**self.__rec2_keywargs)
        print(f"Overwriting the URM of the rec2...")
        self.__rec2.URM_train = self.URM_train
        print(f"done.")

    def recommend(self, user_id_array, cutoff = None, remove_seen_flag=True, items_to_compute = None,
                  remove_top_pop_flag = False, remove_custom_items_flag = False, return_scores = False):

        return self.__rec2.recommend(user_id_array,
                                     cutoff=cutoff,
                                     remove_seen_flag=remove_seen_flag,
                                     items_to_compute=items_to_compute,
                                     remove_top_pop_flag=remove_top_pop_flag,
                                     remove_custom_items_flag=remove_custom_items_flag,
                                     return_scores=return_scores)


    def save_model(self, folder_path, file_name=None):
        if file_name is None:
            file_name = self.RECOMMENDER_NAME
        self._print("Saving model in file '{}'".format(folder_path + file_name))
        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save={})
        self._print("Saving complete")