from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Base.Evaluation.Evaluator import EvaluatorHoldout
from Base.NonPersonalizedRecommender import TopPop
from DataParser import DataParser
from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from GraphBased.RP3betaRecommender import RP3betaRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
import numpy as np
from Base.Recommender_utils import check_matrix


class PipeHybrid001(RP3betaRecommender):
    RECOMMENDER_NAME = "PipeHybrid001"

    def __init__(self, URM_train, ICM_train,verbose=True):
        super(PipeHybrid001, self).__init__(URM_train, verbose = verbose)
        self.URM_train_recommendation = URM_train
        self.ICM_train = ICM_train
        self.__content_recommender = ItemKNNCBFRecommender(URM_train, ICM_train)
        #print("fitting ItemKNNCBF...")
        try:
            self.__content_recommender.load_model('stored_recommenders/ItemKNNCBFRecommender/best_at_26_10_20')
        except:
            self.__content_recommender.fit(topK=140, shrink=1000, similarity='cosine', normalize=True,
                                           feature_weighting='BM25')  # best parameter up to now
            self.__content_recommender.save_model('stored_recommenders/ItemKNNCBFRecommender/best_at_26_10_20')

        #print("... done")

        #print(f"URM_train shape: {URM_train.shape}")
        #print(f"W_sparse knn shape: {self.__content_recommender.W_sparse.shape}")

        self.URM_train = URM_train.dot(self.__content_recommender.W_sparse)
        self._URM_train_format_checked = False
        self._W_sparse_format_checked = False

    def recommend(self, user_id_array, cutoff = None, remove_seen_flag=True, items_to_compute = None,
                  remove_top_pop_flag = False, remove_custom_items_flag = False, return_scores = False):
        """
        redefinition using self.URM_train_recommendation, not the new URM train of the RP3Beta algorithm
        """
        # If is a scalar transform it in a 1-cell array
        if np.isscalar(user_id_array):
            user_id_array = np.atleast_1d(user_id_array)
            single_user = True
        else:
            single_user = False

        if cutoff is None:
            cutoff = self.URM_train_recommendation.shape[1] - 1

        # Compute the scores using the model-specific function
        # Vectorize over all users in user_id_array
        scores_batch = self._compute_item_score(user_id_array, items_to_compute=items_to_compute)


        for user_index in range(len(user_id_array)):

            user_id = user_id_array[user_index]

            if remove_seen_flag:
                scores_batch[user_index,:] = self._remove_seen_on_scores(user_id, scores_batch[user_index, :])

            # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
            # - Partition the data to extract the set of relevant items
            # - Sort only the relevant items
            # - Get the original item index
            # relevant_items_partition = (-scores_user).argpartition(cutoff)[0:cutoff]
            # relevant_items_partition_sorting = np.argsort(-scores_user[relevant_items_partition])
            # ranking = relevant_items_partition[relevant_items_partition_sorting]
            #
            # ranking_list.append(ranking)

        if remove_top_pop_flag:
            scores_batch = self._remove_TopPop_on_scores(scores_batch)

        if remove_custom_items_flag:
            scores_batch = self._remove_custom_items_on_scores(scores_batch)

        # relevant_items_partition is block_size x cutoff
        relevant_items_partition = (-scores_batch).argpartition(cutoff, axis=1)[:,0:cutoff]

        # Get original value and sort it
        # [:, None] adds 1 dimension to the array, from (block_size,) to (block_size,1)
        # This is done to correctly get scores_batch value as [row, relevant_items_partition[row,:]]
        relevant_items_partition_original_value = scores_batch[np.arange(scores_batch.shape[0])[:, None], relevant_items_partition]
        relevant_items_partition_sorting = np.argsort(-relevant_items_partition_original_value, axis=1)
        ranking = relevant_items_partition[np.arange(relevant_items_partition.shape[0])[:, None], relevant_items_partition_sorting]

        ranking_list = [None] * ranking.shape[0]

        # Remove from the recommendation list any item that has a -inf score
        # Since -inf is a flag to indicate an item to remove
        for user_index in range(len(user_id_array)):
            user_recommendation_list = ranking[user_index]
            user_item_scores = scores_batch[user_index, user_recommendation_list]

            not_inf_scores_mask = np.logical_not(np.isinf(user_item_scores))

            user_recommendation_list = user_recommendation_list[not_inf_scores_mask]
            ranking_list[user_index] = user_recommendation_list.tolist()

            #TEST
            """
            user_profile_array = self.URM_train[user_id_array[user_index]]
            if np.empty(user_profile_array):
                print(f"WARNING! {user_index} is a cold user!")
                rec = TopPop(URM_train)
                rec.fit()
                ranking_list[user_index]=rec.recommend([user_id_array[user_index]], cutoff=cutoff)
            """


        # Return single list for one user, instead of list of lists
        if single_user:
            ranking_list = ranking_list[0]

        if return_scores:
            return ranking_list, scores_batch

        else:
            return ranking_list


if __name__ =='__main__':
    seed = 1205
    parser = DataParser('../data')
    URM_all = parser.get_URM_all()
    ICM_all = parser.get_ICM_all()
    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.85, seed=seed)

    f_range = (0, 2)

    # --------------------
    URM_test = parser.filter_URM_test_by_range(URM_train, URM_test, f_range)
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])

    recommender = PipeHybrid001(URM_train, ICM_all)
    recommender.fit(topK=946, alpha=0.47193263239089045, beta=0.0316773658685341, normalize_similarity=False)

    result, _ = evaluator_test.evaluateRecommender(recommender)
    print(result)

