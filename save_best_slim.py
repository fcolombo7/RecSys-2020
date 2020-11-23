import numpy as np
import os
import scipy.sparse as sps
from DataParser import DataParser
from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample

from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender

if __name__ == '__main__':
    parser = DataParser()
    URM_all = parser.get_URM_all()
    random_seed = 1205
    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.85, seed=random_seed)
    slim = SLIMElasticNetRecommender(URM_train)
    slim.fit(topK=140, l1_ratio=1e-5, alpha=0.386)
    slim.save_model('stored_recommenders/slim_elastic_net/',
                    f'best_{random_seed}_23_10_20')