from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Base.Evaluation.Evaluator import EvaluatorHoldout
from DataParser import DataParser
from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from GraphBased.RP3betaRecommender import RP3betaRecommender
from GraphBased.P3alphaRecommender import P3alphaRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
import numpy as np
from Base.Recommender_utils import check_matrix
from SimpleEvaluator import evaluator


def round_robin_merging(ranking_lists):
    list_len = len(ranking_lists[0])
    final_ranking = []
    idx = 0
    while len(final_ranking) < list_len:
        for l in ranking_lists:
            if (l[idx] not in final_ranking) and (len(final_ranking)<list_len):
                final_ranking.append(l[idx])
        idx = idx+1
    return final_ranking


class ListMerged001(BaseItemSimilarityMatrixRecommender):
    RECOMMENDER_NAME = "ListMerged001"

    def __init__(self, URM_train, verbose=True):
        super(BaseItemSimilarityMatrixRecommender, self).__init__(URM_train, verbose=verbose)
        self.__rec1 = RP3betaRecommender(URM_train)
        self.__rec2 = P3alphaRecommender(URM_train)
        self.URM_train=URM_train

    def ger_recommenders(self):
        return self.__rec1,self.__rec2

    def fit(self):
        print("Fit 1/2...")
        self.__rec1.fit(alpha=0.4530815441932864,  beta=0.008742088319964482, topK=104, normalize_similarity=False)
        print("Fit 2/2...")
        self.__rec2.fit(alpha=0.4905425214201532, topK=1000, normalize_similarity=False)
        print("Done.")

    def recommend(self, user_id_array, cutoff = None, remove_seen_flag=True, items_to_compute = None,
                  remove_top_pop_flag = False, remove_custom_items_flag = False, return_scores = False):
        ranking_list1 =self.__rec1.recommend(user_id_array=user_id_array,
                                             cutoff=cutoff,
                                             remove_seen_flag=remove_seen_flag,
                                             items_to_compute=items_to_compute,
                                             remove_top_pop_flag=remove_top_pop_flag,
                                             remove_custom_items_flag=remove_custom_items_flag,
                                             return_scores=return_scores)

        ranking_list2 = self.__rec2.recommend(user_id_array=user_id_array,
                                              cutoff=cutoff,
                                              remove_seen_flag=remove_seen_flag,
                                              items_to_compute=items_to_compute,
                                              remove_top_pop_flag=remove_top_pop_flag,
                                              remove_custom_items_flag=remove_custom_items_flag,
                                              return_scores=return_scores)

        return round_robin_merging([ranking_list1, ranking_list2])


if __name__ == '__main__':
    seed = 1205
    parser = DataParser('../data')
    URM_all = parser.get_URM_all()
    ICM_all = parser.get_ICM_all()
    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.85, seed=seed)

    recommender = ListMerged001(URM_train)
    recommender.fit()
    result_dict = evaluator(recommender, URM_test, cutoff=10)
    print("Round robin:")
    print(result_dict)

    result_dict = evaluator(recommender.ger_recommenders()[0], URM_test, cutoff=10)
    print("RP3beta:")
    print(result_dict)

    result_dict = evaluator(recommender.ger_recommenders()[1], URM_test, cutoff=10)
    print("P3alpha:")
    print(result_dict)

