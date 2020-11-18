import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as pyplot

from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize

from Base.Evaluation.Evaluator import EvaluatorHoldout
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender

from DataParser import DataParser
from Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample

seed = 1234
"""
class CFItemKNN(object):

    def __init__(self, URM):
        self.URM = URM

    def get_URM_train(self):
        return self.URM

    def fit(self, topK=50, shrink=100, normalize=True, similarity="cosine"):
        obj = Compute_Similarity_Python(self.URM,
                                        shrink=shrink,
                                        topK=topK,
                                        normalize=normalize,
                                        similarity=similarity)
        self.W_sparse = obj.compute_similarity()

    def recommend(self, user_id, at=None, exclude_seen=True):
        user_profile = self.URM[user_id]
        scores = user_profile.dot(self.W_sparse).toarray().ravel()
        if exclude_seen:
            scores = self.__filter_seen(user_id, scores)

        # rank the items
        ranking = scores.argsort()[::-1]
        return ranking[:at]

    def __filter_seen(self, user_id, scores):
        start_pos = self.URM.indptr[user_id]
        end_pos = self.URM.indptr[user_id+1]
        user_profile = self.URM.indices[start_pos:end_pos]
        scores[user_profile] = -np.inf
        return scores


def dataset_splits(ratings, num_users, num_items, validation_percentage: float, testing_percentage: float):
    (user_ids_training, user_ids_test,
     item_ids_training, item_ids_test,
     ratings_training, ratings_test) = train_test_split(ratings.user_id,
                                                        ratings.item_id,
                                                        ratings.ratings,
                                                        test_size=testing_percentage,
                                                        shuffle=True,
                                                        random_state=seed)

    (user_ids_training, user_ids_validation,
     item_ids_training, item_ids_validation,
     ratings_training, ratings_validation) = train_test_split(user_ids_training,
                                                              item_ids_training,
                                                              ratings_training,
                                                              test_size=validation_percentage,
                                                              )

    urm_train = sp.csr_matrix((ratings_training, (user_ids_training, item_ids_training)),
                              shape=(num_users, num_items))

    urm_validation = sp.csr_matrix((ratings_validation, (user_ids_validation, item_ids_validation)),
                                   shape=(num_users, num_items))

    urm_test = sp.csr_matrix((ratings_test, (user_ids_test, item_ids_test)),
                             shape=(num_users, num_items))

    return urm_train, urm_validation, urm_test
"""

if __name__ == '__main__':
    parser = DataParser()
    URM_all = parser.get_URM_all()
    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.85)
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.80)

    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])

    run = 0
    topK_values = range(100,900, 50)
    shrink_values = range(100,900, 50)
    space = [Categorical(categories=topK_values, name='topK'),
             Categorical(categories=shrink_values, name='shrink')]

    @use_named_args(space)
    def objective(**params):
        recommender = ItemKNNCFRecommender(URM_train)
        recommender.fit(**params)
        print(f"RUN#{run}:\n> space: [ topK={recommender.topK}, shrink={recommender.shrink} ]")
        result_dict, _ = evaluator_validation.evaluateRecommender(recommender)
        MAP = result_dict[10]['MAP']
        print(f"> result: [ MAP={MAP} ]")
        return -MAP


    res_gp = gp_minimize(objective, space, n_calls=2, n_random_starts=2)

    print(f"Best parameters:\n- topK: {res_gp.x[0]},\n -shrink: {res_gp.x[1]}")