import numpy as np
import pandas as pd
import re

from sklearn.model_selection import KFold
import scipy.sparse as sps
from Base.Evaluation.Evaluator import EvaluatorHoldout
from DataParser import DataParser
from datetime import datetime

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from GraphBased.P3alphaRecommender import P3alphaRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
from Hybrid.HybridCombinationSearch import HybridCombinationMergedSearch
from Hybrid.HybridCombinationSearchCVtest import HybridCombinationSearchCVtest
from Hybrid.LinearHybrid007 import LinearHybrid007
from Hybrid.LinearHybrid008 import LinearHybrid008
from Hybrid.UserWiseHybrid009 import UserWiseHybrid009
from Hybrid.UserWiseRecommenderCV import UserWiseRecommenderCV
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.ItemKNN_CBF_CF import ItemKNN_CBF_CF
from KNN.SpecialItemKNNCBFRecommender import SpecialItemKNNCBFRecommender
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from SLIM_ElasticNet.SSLIM_ElasticNet import SSLIMElasticNet
from SimpleEvaluator import evaluator
from Hybrid.UserWiseHybrid005 import UserWiseHybrid005


def create_csv(parser, recommender, name=None):
    out_userid = np.array([])
    out_itemlist = np.array([])

    target_data = parser.get_target_data()
    for user_id in target_data.user_id.unique():
        out_userid = np.append(out_userid, user_id)
        recommendation = recommender.recommend(user_id, cutoff=10)
        # print(type(recommendation))
        str_ = re.sub(' +', ' ', np.array_str(np.array(recommendation)))[1:-1]
        if str_[0] == ' ':
            str_ = str_[1:]
        # print(str_)
        out_itemlist = np.append(out_itemlist, str_)

    out_dataframe = pd.DataFrame(data={'user_id': out_userid, 'item_list': out_itemlist})
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
    parser = DataParser()
    URM_all = parser.get_URM_all()
    ICM_all = parser.get_ICM_all()
    rec_range_list = [
        ((0, 3), ['icb', 'sbpr', 'sslim'], {'alpha': 0.5238332406148302, 'l1_ratio': 0.5907932058981844}),
        ((3, 5), ['icbsup', 'p3a', 'sslim'], {'alpha': 0.7989350513339316, 'l1_ratio': 0.4461703935735907}),
        ((5, 20), ['icbsup', 'icfcb', 'p3a'], {'alpha': 0.6094679138906033, 'l1_ratio': 0.41715098578193194}),
        ((20, 50), ['icbsup', 'rp3b', 'sslim'], {'alpha': 0.7881737847053779, 'l1_ratio': 0.400772981812612}),
        ((50, 70), ['icb', 'sbpr', 'sslim'], {'alpha': 0.5238332406148302, 'l1_ratio': 0.5907932058981844}),
        ((70, 200), ['icb', 'rp3b', 'sslim'], {'alpha': 0.7861575718386372, 'l1_ratio': 0.34264216397962627}),
        ((200, -1), ['icbsup', 'p3a', 'sslim'], {'alpha': 0.7989350513339316, 'l1_ratio': 0.4461703935735907}),
    ]
    rec = UserWiseRecommenderCV(URM_all, ICM_all, rec_range_list, verbose=True, submission=True)
    rec.fit()
    create_csv(parser, rec, "UserWiseRecommenderCV-selected-ranges_CV2")

    exit(0)
    # testing rec with CV
    seed = 1666
    k = 5
    kf = KFold(n_splits=5, random_state=1666, shuffle=True)
    URM_list = []
    URM_test_list = []
    evaluator_list = []
    shape = URM_all.shape
    indptr = URM_all.indptr
    indices = URM_all.indices
    data = URM_all.data
    for train_index, test_index in kf.split(data):
        # print("TRAIN:", train_index, "TEST:", test_index)
        data_train = np.ones(data.shape)
        data_test = np.ones(data.shape)
        data_train[test_index] = 0
        data_test[train_index] = 0
        _URM_train = sps.csr_matrix((data_train, indices, indptr), shape=shape).copy()
        _URM_test = sps.csr_matrix((data_test, indices, indptr), shape=shape).copy()
        _URM_train.eliminate_zeros()
        _URM_test.eliminate_zeros()
        URM_list.append(_URM_train)
        URM_test_list.append(_URM_test)
        evaluator_list.append(EvaluatorHoldout(_URM_test, cutoff_list=[10]))

    result = []
    rec_range_list = [
        ((0, 3), ['icb', 'sbpr', 'sslim'], {'alpha': 0.5238332406148302, 'l1_ratio': 0.5907932058981844}),
        ((3, 5), ['icbsup', 'p3a', 'sslim'], {'alpha': 0.7989350513339316, 'l1_ratio': 0.4461703935735907}),
        ((5, 8), ['icb', 'icfcb', 'p3a'], {'alpha': 0.6904008032335807, 'l1_ratio': 0.23371348584724985}),
        ((8, 17), ['icbsup', 'icfcb', 'sbpr'], {'alpha': 0.7949204415356104, 'l1_ratio': 0.2705694962114903}),
        ((17, -1), ['icbsup', 'p3a', 'sslim'], {'alpha': 0.7989350513339316, 'l1_ratio': 0.4461703935735907}),
        #((30, 200), ['icb', 'p3a', 'sslim'], {'alpha': 0.6325200648079097, 'l1_ratio': 0.4685932097360991}),
        #((100, 200), ['icb', 'p3a', 'sslim'], ),
        #((200, -1), ['icbsup', 'p3a', 'sslim'], {'alpha': 0.7989350513339316, 'l1_ratio': 0.4461703935735907}),
    ]
    for i in range(len(URM_list)):
        rec = UserWiseRecommenderCV(URM_list[i], ICM_all, rec_range_list, submission=False, seed=seed, fold=i,
                                    verbose=True)
        print(f'\n> {rec.RECOMMENDER_NAME} on FOLD-{i}')
        rec.fit()
        r = evaluator(rec, URM_test_list[i], cutoff=10)
        result.append(r['MAP'])
        print(f'> MAP: {str(r["MAP"])}')

    print("> result list: ", result)
    print("> AVG MAP: ", np.average(result))

    exit(0)
    print('making a sub.')
    rec_list = [
        (SpecialItemKNNCBFRecommender,
         {'topK': 1000, 'shrink': 1000, 'similarity': 'cosine', 'normalize': True, 'feature_weighting': 'BM25'}),
        (ItemKNN_CBF_CF, {'topK': 1000, 'shrink': 1000, 'similarity': 'asymmetric', 'normalize': True,
                          'asymmetric_alpha': 0.241892724784089, 'feature_weighting': 'TF-IDF', 'icm_weight': 1.0}),
        (SLIM_BPR_Cython,
         {'topK': 1000, 'epochs': 130, 'symmetric': False, 'sgd_mode': 'adam', 'lambda_i': 1e-05, 'lambda_j': 1e-05,
          'learning_rate': 0.0001})

    ]
    rec = HybridCombinationSearchCVtest(URM_all, ICM_all, list_rec=rec_list, verbose=True, submission=True)
    params = {'alpha': 0.7949204415356104, 'l1_ratio': 0.2705694962114903}
    rec.fit(**params)
    print("creating the first csv...")
    create_csv(parser, rec, "SpecialItemKNNCBF-ItemKNN_CBF_CF-SlimBPR_CV2")
    print("Done.")

    print("Now the second...")
    rec_list = [
        (ItemKNNCBFRecommender,
         {'topK': 164, 'shrink': 8, 'similarity': 'jaccard', 'normalize': True}),
        (ItemKNN_CBF_CF, {'topK': 1000, 'shrink': 1000, 'similarity': 'asymmetric', 'normalize': True,
                          'asymmetric_alpha': 0.241892724784089, 'feature_weighting': 'TF-IDF', 'icm_weight': 1.0}),
        (SLIM_BPR_Cython,
         {'topK': 1000, 'epochs': 130, 'symmetric': False, 'sgd_mode': 'adam', 'lambda_i': 1e-05, 'lambda_j': 1e-05,
          'learning_rate': 0.0001})

    ]
    rec2 = HybridCombinationSearchCVtest(URM_all, ICM_all, list_rec=rec_list, verbose=True, submission=True)
    params = {'alpha': 0.8991751672246813, 'l1_ratio': 0.11874637825106651}
    rec2.fit(**params)
    create_csv(parser, rec2, "ItemKNNCBF-ItemKNN_CBF_CF-SlimBPR_CV2")

    exit(0)

    # testing k_fold

    seed = 1666
    k = 5

    kf = KFold(n_splits=5, random_state=1666, shuffle=True)
    URM_list = []
    evaluator_list = []
    shape = URM_all.shape
    indptr = URM_all.indptr
    indices = URM_all.indices
    data = URM_all.data
    for train_index, test_index in kf.split(data):
        # print("TRAIN:", train_index, "TEST:", test_index)
        data_train = np.ones(data.shape)
        data_test = np.ones(data.shape)
        data_train[test_index] = 0
        data_test[train_index] = 0
        _URM_train = sps.csr_matrix((data_train, indices, indptr), shape=shape).copy()
        _URM_test = sps.csr_matrix((data_test, indices, indptr), shape=shape).copy()
        _URM_train.eliminate_zeros()
        _URM_test.eliminate_zeros()
        URM_list.append(_URM_train)
        evaluator_list.append(EvaluatorHoldout(_URM_test, cutoff_list=[10]))

    result = []
    for i in range(len(URM_list)):
        rec = SSLIMElasticNet(URM_list[i], ICM_all, verbose=True)
        print(f'\n> {rec.RECOMMENDER_NAME} on FOLD-{i}')
        params = {'beta': 0.567288665094892, 'topK': 1000, 'l1_ratio': 1e-05, 'alpha': 0.001}
        rec.fit(**params)
        rec.save_model(f'stored_recommenders/seed_{str(seed)}_hybrid_search/{rec.RECOMMENDER_NAME}/',
                       f'{str(seed)}_fold-{str(i)}')

        # r, _ =evaluator_list[i].evaluateRecommender(rec)
        # result.append(r[10]['MAP'])

    # print(result)
    # print(np.average(result))

    exit(0)
    print("Making a submission... ")
    parser = DataParser()
    URM_all = parser.get_URM_all()
    ICM_all = parser.get_ICM_all()

    dict_1 = {'ICM_train': ICM_all}
    rec = ItemKNNCBFRecommender(URM_all, *dict_1)
    # rec_sub = UserWiseHybrid009(URM_all, ICM_all, submission=True)
    # rec_sub.fit()
    # create_csv(parser, rec_sub, 'UserWiseHybrid009')

    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.90, seed=1205)
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])

    recommender = UserWiseHybrid009(URM_train, ICM_all, submission=True)
    recommender.fit()
    result, _ = evaluator_test.evaluateRecommender(recommender)
    print(result[10])
