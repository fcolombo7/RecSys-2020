import numpy as np
import scipy.sparse as sps
import pandas as pd
import re

from Base.Evaluation.Evaluator import EvaluatorHoldout
from DataParser import DataParser
from datetime import datetime

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender


def create_csv(parser, recommender, name=None):

    out_userid = np.array([])
    out_itemlist = np.array([])

    target_data = parser.get_target_data()
    for user_id in target_data.user_id.unique():
        out_userid = np.append(out_userid, user_id)
        str_ = re.sub(' +', ' ', np.array_str(recommender.recommend(user_id, at=10)))[1:-1]
        if str_[0] == ' ':
            str_ = str_[1:]
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
    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.85, seed=seed)

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])

    # Define the recommenders
    recommender_cfi = ItemKNNCFRecommender(URM_train)
    recommender_cfi.fit(topK=967, shrink=356)

    #'topK': 120, 'l1_ratio': 1e-05, 'alpha': 0.066
    recommender_slim = SLIMElasticNetRecommender(URM_train)
    recommender_slim.fit(topK=120, l1_ratio=1e-5, alpha=0.066)

    result_dict, _ = evaluator_test.evaluateRecommender(recommender_cfi)
    print('ItemKNNCFRecommender:\n')
    print(f"MAP: {result_dict[10]['MAP']}") #MAP: 0.0487357913544006 , PROVE FATTE SENZA SETTARE IL SEED

    result_dict, _ = evaluator_test.evaluateRecommender(recommender_slim)
    print('\nSLIMElasticNetRecommender:\n')
    print(f"MAP: {result_dict[10]['MAP']}") #MAP: 0.05416326850944763

    #create_csv(parser, recommender, 'testCFItem')
