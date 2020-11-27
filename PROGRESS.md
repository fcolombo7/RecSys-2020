# User segmentation
* Range: **[200, end)**

| Algorithm | MAP | optimal parameters | notes |
| ------ | ------| ------| ------|
| PureSVD | 0.0411772 | _'num_factors': 136_ | - |
| RP3beta** | 0.0573073 | _'topK': 653, 'alpha': 0.0, 'beta': 0.0, 'normalize_similarity': True_ | configurazione dubbia |
| RP3beta* | 0.0969482 | _'topK': 272, 'alpha': 0.4375286551430567, 'beta': 0.16773603374534, 'normalize_similarity': True_ | performance più basse sul validation |
| P3alpha | 0.0590533 | _'topK': 948, 'alpha': 0.9208017955796528, 'normalize_similarity': False_ | - |
| UserKNNCF_cosine | 0.0663246| _'topK': 635, 'shrink': 241, 'similarity': 'cosine', 'normalize': False_ | - |
| UserKNNCF_jaccard | 0.0761602 | _'topK': 5, 'shrink': 1000, 'similarity': 'jaccard', 'normalize': False_ | - |
| ItemKNNCF_cosine | 0.0702457 | _'topK': 215, 'shrink': 1000, 'similarity': 'cosine', 'normalize': True_ | - |
| ItemKNNCF_jaccard | 0.0681387 | _'topK': 120, 'shrink': 765, 'similarity': 'jaccard', 'normalize': True_ | - |
| ItemKNNCF_asym | 0.0767328 | _'topK': 225, 'shrink': 245, 'similarity': 'asymmetric', 'normalize': True, 'asymmetric_alpha': 0.6650406762508793_ | - |
| ItemKNNCBF_cosine | 0.0841648 | _'topK': 90, 'shrink': 1000, 'similarity': 'cosine', 'normalize': True, 'feature_weighting': 'BM25'_ | **incredibile** |
| ItemKNNCBF_jaccard | 0.0604119 | _'topK': 5, 'shrink': 1000, 'similarity': 'jaccard', 'normalize': False_ | - |
| ItemKNNCBF_asym | 0.0795049 | _'topK': 90, 'shrink': 990, 'similarity': 'asymmetric', 'normalize': True, 'asymmetric_alpha': 0.0, 'feature_weighting': 'BM25'_ | - |
| Slim_BPR | 0.1215404 | _'topK': 120, 'epochs': 20, 'symmetric': True, 'sgd_mode': 'adam', 'lambda_i': 0.01, 'lambda_j': 1e-05, 'learning_rate': 0.0001_ | - |
| SLIM_ElasticNet | 0.0971627 | _'topK': 1000, 'l1_ratio': 0.0007739617738737429, 'alpha': 0.001_ | - |

* Range: **[100, 200)**

| Algorithm | MAP | optimal parameters | notes |
| ------ | ------| ------| ------|
| PureSVD | 0.0536443 | _'num_factors': 161_ | - |
| RP3beta | 0.0835274 | _'topK': 1000, 'alpha': 0.32110178834628456, 'beta': 0.0, 'normalize_similarity': True_ | - |
| P3alpha | 0.0765238 | _'topK': 705, 'alpha': 0.14129549794480725, 'normalize_similarity': True_ | - |
| UserKNNCF_cosine | 0.0503889 | _'topK': 75, 'shrink': 1000, 'similarity': 'cosine', 'normalize': True_ | - |
| UserKNNCF_jaccard | 0.0534986 | _'topK': 35, 'shrink': 1000, 'similarity': 'jaccard', 'normalize': False_ | - |
| ItemKNNCF_cosine | 0.0499827 | _'topK': 165, 'shrink': 202, 'similarity': 'cosine', 'normalize': True_ | - |
| ItemKNNCF_jaccard | 0.0627071 | _'topK': 60, 'shrink': 88, 'similarity': 'jaccard', 'normalize': True_ | - |
| ItemKNNCBF_cosine | 0.0362179 | _'topK': 345, 'shrink': 184, 'similarity': 'cosine', 'normalize': False, 'feature_weighting': 'BM25'_ | - |
| ItemKNNCBF_jaccard | 0.0307958 | _'topK': 725, 'shrink': 79, 'similarity': 'jaccard', 'normalize': False_ | - |
| Slim_BPR | 0.0422478 | 'topK': 100, 'epochs': 90, 'symmetric': True, 'sgd_mode': 'adam', 'lambda_i': 0.01, 'lambda_j': 0.01, 'learning_rate': 0.0001_ | - |

* Range: **[25, 100)**

| Algorithm | MAP | optimal parameters | notes |
| ------ | ------| ------| ------|
| PureSVD | 0.0239736 | _'num_factors': 228_ | - |
| RP3beta | 0.0417560 | _'topK': 1000, 'alpha': 0.35039375652835403, 'beta': 0.0, 'normalize_similarity': False_ | - |
| P3alpha | 0.0417541 | _'topK': 1000, 'alpha': 0.34876889668012145, 'normalize_similarity': False_ | - |
| UserKNNCF_cosine | 0.0383413 | _'topK': 335, 'shrink': 89, 'similarity': 'cosine', 'normalize': True_ | - |
| UserKNNCF_jaccard | 0.0410465 | _'topK': 225, 'shrink': 38, 'similarity': 'jaccard', 'normalize': False_ | - |
| UserKNNCBF_asym | 0.0382498 | 'topK': 100, 'shrink': 563, 'similarity': 'asymmetric', 'normalize': True, 'asymmetric_alpha': 1.6685411937999246 | - |
| ItemKNNCF_cosine | 0.0375998 | _'topK': 425, 'shrink': 91, 'similarity': 'cosine', 'normalize': True_ | - |
| ItemKNNCF_jaccard | 0.0364897 | _'topK': 535, 'shrink': 220, 'similarity': 'jaccard', 'normalize': True_ | - |
| ItemKNNCBF_cosine | 0.0228656 | _'topK': 25, 'shrink': 1000, 'similarity': 'cosine', 'normalize': True, 'feature_weighting': 'BM25'_ | - |
| ItemKNNCBF_jaccard | 0.0189027 | _'topK': 70, 'shrink': 0, 'similarity': 'jaccard', 'normalize': True_ | - |

* Range: **[0, 25)**

| Algorithm | MAP | optimal parameters | notes |
| ------ | ------| ------| ------|
| PureSVD | 0.0261367 | _'num_factors': 406_ | - |
| RP3beta | 0.0459742 | _'topK': 898, 'alpha': 0.06846350192503016, 'beta': 0.17584831467422982, 'normalize_similarity': False_ | - |
| P3alpha | 0.0515973 | _'topK': 729, 'alpha': 0.4104229220476686, 'normalize_similarity': False_ | - |
| UserKNNCF_cosine | 0.0476226 | _'topK': 70, 'shrink': 0, 'similarity': 'cosine', 'normalize': True_ | - |
| UserKNNCF_jaccard | 0.0480116 | _'topK': 140, 'shrink': 0, 'similarity': 'jaccard', 'normalize': False_ | - |
| UserKNNCF_asym | 0.0484562 | _'topK': 530, 'shrink': 87, 'similarity': 'asymmetric', 'normalize': True, 'asymmetric_alpha': 1.248829420189138_ | - |
| ItemKNNCF_cosine | 0.0424653 | _'topK': 90, 'shrink': 77, 'similarity': 'cosine', 'normalize': True_ | - |
| ItemKNNCF_jaccard | 0.0438934 | _'topK': 725, 'shrink': 240, 'similarity': 'jaccard', 'normalize': False_ | - |
| ItemKNNCF_asym | 0.0429509 | 'topK': 130, 'shrink': 1000, 'similarity': 'asymmetric', 'normalize': True, 'asymmetric_alpha': 0.0 | - |
| ItemKNNCBF_cosine | 0.0330596 | _'topK': 165, 'shrink': 1000, 'similarity': 'cosine', 'normalize': True, 'feature_weighting': 'BM25'_ | - |
| ItemKNNCBF_jaccard | 0.0312501 | _'topK': 65, 'shrink': 56, 'similarity': 'jaccard', 'normalize': True_ | - |

# Progress

| Submission | Result | Note |
| ------ | ------| ------|
| ----.csv | 0.06583 | Item based KNN CF with parameters determined by hyperparam search.  (*Test submission*) |


# Notes
- check news on the discussion forum about **SLIM Elasticnet**
