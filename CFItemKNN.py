import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as pyplot

from DataParser import DataParser
from Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
import SimpleEvaluator
seed = 1234

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


if __name__ == '__main__':
    parser = DataParser()

    # get the ratings
    ratings = parser.get_ratings()
    print(ratings)
    users_stats, items_stats, ratings_stats = parser.get_statistics()

    # split them into 3 sets
    URM_train, URM_valid, URM_test = dataset_splits(ratings, users_stats['max']+1, items_stats['max']+1, 0.15, 0.15)

    # testing the recommender
    """
    recommender = CFItemKNN(URM_train)
    recommender.fit(shrink=0, topK=50)
    
    for user_id in range(10):
        print(recommender.recommend(user_id=user_id, at=10))
    """

    x_tick = [10, 50, 100, 200, 500]
    MAP_per_k = []
    recommender = CFItemKNN(URM_train)
    for topK in x_tick:
        recommender.fit(shrink=0, topK=topK)

        result_dict = SimpleEvaluator.evaluator(recommender, urm_test=URM_valid, cutoff=10)
        MAP_per_k.append((result_dict['MAP']))

    pyplot.plot(x_tick, MAP_per_k)
    pyplot.ylabel('MAP')
    pyplot.xlabel('TopK')
    pyplot.show()


