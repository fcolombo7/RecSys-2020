from Base.Evaluation.Evaluator import EvaluatorHoldout
from DataParser import DataParser
from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Hybrid.HybridCombinationSearch import HybridCombinationSearch
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from SLIM_ElasticNet.SSLIM_ElasticNet import SSLIMElasticNet

import pandas as pd

if __name__ == '__main__':
    seed = 1205
    parser = DataParser()
    URM_all = parser.get_URM_all()
    ICM_all = parser.get_ICM_all()

    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.90, seed=seed)

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])

    sslim = SSLIMElasticNet(URM_train, ICM_all, verbose=False)
    sslim_params = {'beta': 0.4849594591575789, 'topK': 1000, 'l1_ratio': 1e-05, 'alpha': 0.001}
    try:
        sslim.load_model(f'stored_recommenders/seed_1205_S-SLIMElasticNet/', 'for_notebook_analysis')
        print(f"{sslim.RECOMMENDER_NAME} loaded.")
    except:
        print(f"Fitting {sslim.RECOMMENDER_NAME} ...")
        sslim.fit(**sslim_params)
        print(f"done.")
        sslim.save_model(f'stored_recommenders/seed_{str(seed)}_{sslim.RECOMMENDER_NAME}/', 'for_notebook_analysis')

    ucf = UserKNNCFRecommender(URM_train, verbose=False)
    ucf_params = {'topK': 190, 'shrink': 0, 'similarity': 'cosine', 'normalize': True}
    try:
        ucf.load_model(f'stored_recommenders/seed_{str(seed)}_{ucf.RECOMMENDER_NAME}/', 'for_notebook_analysis')
        print(f"{ucf.RECOMMENDER_NAME} loaded.")
    except:
        print(f"Fitting {ucf.RECOMMENDER_NAME} ...")
        ucf.fit(**ucf_params)
        print(f"done.")
        ucf.save_model(f'stored_recommenders/seed_{str(seed)}_{ucf.RECOMMENDER_NAME}/', 'for_notebook_analysis')

    icb = ItemKNNCBFRecommender(URM_train, ICM_all, verbose=False)

    icb_params = {'topK': 65, 'shrink': 0, 'similarity': 'dice', 'normalize': True}
    try:
        icb.load_model(f'stored_recommenders/seed_{str(seed)}_{icb.RECOMMENDER_NAME}/', 'for_notebook_analysis')
        print(f"{icb.RECOMMENDER_NAME} loaded.")
    except:
        print(f"Fitting {icb.RECOMMENDER_NAME} ...")
        icb.fit(**icb_params)
        print(f"done.")
        icb.save_model(f'stored_recommenders/seed_{str(seed)}_{icb.RECOMMENDER_NAME}/', 'for_notebook_analysis')

    list_recommender = [sslim, icb, ucf]
    best_recommender = HybridCombinationSearch(URM_train, ICM_all, list_recommender)
    params = {'alpha': 0.6461624491197696, 'l1_ratio': 0.7617220099582368}
    best_recommender.fit(**params)

    user_ids = parser.get_ratings().user_id.unique()
    cutoff = 20
    user_recommendations_items = []
    user_recommendations_user_id = []
    target = []

    for n_user in user_ids:
        recommendations = best_recommender.recommend(n_user, cutoff=20)
        user_recommendations_items.extend(recommendations)
        user_recommendations_user_id.extend([n_user] * len(recommendations))

    for _ in user_ids:
        for _ in range(int(cutoff / 2)):
            target.append(1)
        for _ in range(int(cutoff / 2)):
            target.append(0)

    train_dataframe = pd.DataFrame({"user_id": user_recommendations_user_id,
                                    "item_id": user_recommendations_items,
                                    'target': target})

    
