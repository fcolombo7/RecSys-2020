from Base.Evaluation.Evaluator import EvaluatorHoldout
from DataParser import DataParser
from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Hybrid.MergedHybrid000 import MergedHybrid000
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender

if __name__ == '__main__':
    seed = 1205

    parser = DataParser()
    URM_all = parser.get_URM_all()
    ICM_all = parser.get_ICM_all()

    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.85, seed=seed)

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
    rec1 = ItemKNNCBFRecommender(URM_train, ICM_all)
    rec2 = SLIMElasticNetRecommender(URM_train)

    # 'topK': 40, 'shrink': 1000, 'similarity': 'cosine', 'normalize': True, 'feature_weighting': 'BM25'
    rec1.fit(topK=40, shrink=1000, similarity='cosine', feature_weighting='BM25')

    # topK': 140, 'l1_ratio': 1e-05, 'alpha': 0.386
    rec2.fit(topK=140, l1_ratio=1e-5, alpha=0.386)
    print("recomenders are ready")
    merged_recommender = MergedHybrid000(URM_train, content_recommender=rec1, collaborative_recommender=rec2)
    merged_recommender.fit(alpha=0.5)
    result, _ = evaluator_test.evaluateRecommender(merged_recommender)
    print(result[10]['MAP'])


