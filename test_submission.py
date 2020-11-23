import numpy as np
import scipy.sparse as sps
import pandas as pd
import re

from Base.Evaluation.Evaluator import EvaluatorHoldout
from DataParser import DataParser
from datetime import datetime

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython

def create_csv(parser, recommender, name=None):

    out_userid = np.array([])
    out_itemlist = np.array([])

    target_data = parser.get_target_data()
    for user_id in target_data.user_id.unique():
        out_userid = np.append(out_userid, user_id)
        recommendation = recommender.recommend(user_id, cutoff=10)
        #print(type(recommendation))
        str_ = re.sub(' +', ' ', np.array_str(np.array(recommendation)))[1:-1]
        if str_[0] == ' ':
            str_ = str_[1:]
        #print(str_)
        out_itemlist = np.append(out_itemlist, str_)

    out_dataframe = pd.DataFrame(data={'user_id':out_userid, 'item_list':out_itemlist})
    out_dataframe = out_dataframe.astype({'user_id': 'int32'})

    filename = str(datetime.now().strftime("res_%Y%m%d-%H%M.csv"))
    if not name is None:
        name = name + '_' + filename
    else:
        name = filename
    out_path = "res_csv/" + name
    out_dataframe.to_csv(out_path, index=False)

    # remove the single line added
    fd = open(out_path, "r")
    d = fd.read()
    fd.close()
    m = d.split("\n")
    s = "\n".join(m[:-1])
    fd = open(out_path, "w+")
    for i in range(len(s)):
        fd.write(s[i])
    fd.close()


if __name__ == '__main__':
    #print("Making a submission... ")
    seed = 1205

    parser = DataParser()
    URM_all = parser.get_URM_all()
    ICM_all = parser.get_ICM_all()
    rec1 = ItemKNNCBFRecommender(URM_all, ICM_all)
    rec1.fit(topK=40, shrink=1000, similarity='cosine', feature_weighting='BM25')
    create_csv(parser,rec1,'itemCBF')
    """
    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.85, seed=seed)

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])

    # Define the recommenders
    #recommender_cfi = ItemKNNCFRecommender(URM_train)
    #recommender_cfi.fit(topK=967, shrink=356)

    #'topK': 120, 'l1_ratio': 1e-05, 'alpha': 0.066
    #recommender_slim = SLIMElasticNetRecommender(URM_train)
    #recommender_slim.fit(topK=120, l1_ratio=1e-5, alpha=0.066)

    #topK: 790, lambda_i: 0.008943099834373669, lambda_j: 1.1173145975517076e-05, learning_rate: 0.0001
    #recommender_slim_bpr = SLIM_BPR_Cython(URM_train)
    #recommender_slim_bpr.fit(topK=790, sgd_mode = 'sgd', epochs = 60, random_seed = seed, lambda_i = 0.008943099834373669, lambda_j = 1.1173145975517076e-05, learning_rate = 0.0001)

    #topK: 1000, lambda_i: 0.01, lambda_j: 0.01, learning_rate: 0.0001
    recommender_slim_bpr2 = SLIM_BPR_Cython(URM_train)
    recommender_slim_bpr2.fit(topK=1000, sgd_mode = 'adam', symmetric = False, epochs = 90, random_seed = seed, lambda_i = 0.01, lambda_j = 0.01, learning_rate = 0.0001)

    #result_dict, _ = evaluator_test.evaluateRecommender(recommender_cfi)
    #print('ItemKNNCFRecommender:\n')
    #print(f"MAP: {result_dict[10]['MAP']}") #MAP: 0.0487357913544006 , PROVE FATTE SENZA SETTARE IL SEED

    #result_dict, _ = evaluator_test.evaluateRecommender(recommender_slim)
    #print('\nSLIMElasticNetRecommender:\n')
    #print(f"MAP: {result_dict[10]['MAP']}") #MAP: 0.05416326850944763

    #result_dict, _ = evaluator_test.evaluateRecommender(recommender_slim_bpr)
    #print('SLIM_BPR_Recommender:\n')
    #print(f"MAP: {result_dict[10]['MAP']}") #MAP: 0.05335925676721219

    result_dict, _ = evaluator_test.evaluateRecommender(recommender_slim)
    print('\nSLIMElasticNetRecommender:\n')
    print(f"MAP: {result_dict[10]['MAP']}") #MAP: 0.05416326850944763
    
    
    recommender_slim_bpr2 = SLIM_BPR_Cython(URM_all)
    recommender_slim_bpr2.fit(topK=1000, sgd_mode = 'adam', symmetric = False, epochs = 90, random_seed = seed, lambda_i = 0.01, lambda_j = 0.01, learning_rate = 0.0001)

    #result_dict, _ = evaluator_test.evaluateRecommender(recommender_slim_bpr2)
    #print('SLIM_BPR_Recommender:\n')
    #print(f"MAP: {result_dict[10]['MAP']}") #MAP: 0.05335925676721219

    create_csv(parser, recommender_slim_bpr2, 'SLIM_BPR')
    """
