# New setup
```
seed = 1205
# SPLIT TO GET TEST PARTITION
URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.90, seed=seed)

# SPLIT TO GET THE HYBRID VALID PARTITION
URM_train, URM_valid_hybrid = split_train_in_two_percentage_global_sample(URM_train, train_percentage = 0.85, seed=seed)

# SPLIT TO GET THE sub_rec VALID PARTITION
URM_train, URM_valid_sub = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.85, seed=seed)
```
## Sub-recommenders performance

| Algorithm | MAP on **validation_sub** | optimal parameters |
| ------ | ------| ------|
| ItemKNNCBF_asym_**err** | 0.0299853 | {'topK': 545, 'shrink': 790, 'similarity': 'asymmetric', 'normalize': True, 'asymmetric_alpha': 0.3706399568074206, 'feature_weighting': 'BM25'}
| ItemKNNCBF_cosine_**err** | 0.0306796 | {'topK': 205, 'shrink': 1000, 'similarity': 'cosine', 'normalize': True, 'feature_weighting': 'BM25'}
| ItemKNNCBF_jaccard_**err** | 0.0266469 | {'topK': 105, 'shrink': 82, 'similarity': 'jaccard', 'normalize': True}
| ItemKNNCBF_asym | 0.0243888 | {'topK': 775, 'shrink': 849, 'similarity': 'asymmetric', 'normalize': True, 'asymmetric_alpha': 0.5913208493761569, 'feature_weighting': 'TF-IDF'}
| ItemKNNCBF_cosine | 0.0246950 | {'topK': 630, 'shrink': 261, 'similarity': 'cosine', 'normalize': True, 'feature_weighting': 'TF-IDF'}
| ItemKNNCBF_jaccard | 0.0263456 | {'topK': 135, 'shrink': 246, 'similarity': 'jaccard', 'normalize': True}
| ItemKNNCBF_dice | 0.0273667 | {'topK': 65, 'shrink': 0, 'similarity': 'dice', 'normalize': True}
| ItemKNNCBF_tversky | 0.0269525 | {'topK': 235, 'shrink': 104, 'similarity': 'tversky', 'normalize': True, 'tversky_alpha': 0.7080647543936939, 'tversky_beta': 0.15722506241462988}
| ItemKNNCF_asym | 0.0353139 | {'topK': 55, 'shrink': 1000, 'similarity': 'asymmetric', 'normalize': True, 'asymmetric_alpha': 0.0}
| ItemKNNCF_cosine | 0.0350612 | {'topK': 655, 'shrink': 130, 'similarity': 'cosine', 'normalize': True}
| ItemKNNCF_jaccard | 0.0350433 | {'topK': 460, 'shrink': 234, 'similarity': 'jaccard', 'normalize': False}
| UserKNNCF_asym | 0.0384820 | {'topK': 285, 'shrink': 392, 'similarity': 'asymmetric', 'normalize': True, 'asymmetric_alpha': 1.7165427482216917}
| UserKNNCF_cosine | 0.0399308 | {'topK': 190, 'shrink': 0, 'similarity': 'cosine', 'normalize': True}
| UserKNNCF_jaccard | 0.0384524 | {'topK': 190, 'shrink': 0, 'similarity': 'cosine', 'normalize': True}
| RP3Beta | 0.0406500 | {'topK': 1000, 'alpha': 0.38192761611274967, 'beta': 0.0, 'normalize_similarity': False}
| P3alpha | 0.0415311 | {'topK': 131, 'alpha': 0.33660811631883863, 'normalize_similarity': False}
| SLIM_ElasticNet | 0.0380146 | {'topK': 992, 'l1_ratio': 0.004065081925341167, 'alpha': 0.003725005053334143}
| Slim_BPR | 0.0349263 | {'topK': 979, 'epochs': 130, 'symmetric': False, 'sgd_mode': 'adam', 'lambda_i': 0.004947329669424629, 'lambda_j': 1.1534760845071758e-05, 'learning_rate': 0.0001}

## Combo performance

| Combined Recs | MAP on **validation_hybrid** | optimal parameters |
| ------ | ------| ------|
| ItemKNNCF, UserKNNCF, ItemKNNCBF| 0.0659249 | 'alpha': 0.767469300493861, 'l1_ratio': 0.7325725081659069
| P3alpha, ItemKNNCF, ItemKNNCBF| 0.0677491 | 'alpha': 0.4066665999396494, 'l1_ratio': 0.7594645794234393
| P3alpha, ItemKNNCF, UserKNNCF| 0.0542544 | 'alpha': 0.5630173292967398, 'l1_ratio': 0.92733589638295
| P3alpha, UserKNNCF, ItemKNNCBF| 0.0681188 | 'alpha': 0.37776131907747645, 'l1_ratio': 0.44018901104481
| RP3Beta, ItemKNNCF, ItemKNNCBF| 0.0671482 | 'alpha': 0.40426999639005445, 'l1_ratio': 1.0
| RP3Beta, ItemKNNCF, UserKNNCF| 0.0533618 | 'alpha': 0.7416313012430469, 'l1_ratio': 0.8122593875086325
| RP3Beta, P3alpha, ItemKNNCBF| 0.0677085 | 'alpha': 0.3553383791480798, 'l1_ratio': 0.000435281815357902
| RP3Beta, P3alpha, ItemKNNCF| 0.0535307 | 'alpha': 0.367698407319822, 'l1_ratio': 0.5878133798647788
| RP3Beta, P3alpha, UserKNNCF| 0.0541693 | 'alpha': 0.6405838432360388, 'l1_ratio': 0.4188312253799342
| RP3Beta, UserKNNCF, ItemKNNCBF| 0.0680765 | 'alpha': 0.4648716125499346, 'l1_ratio': 0.292302921903516
| ItemKNNCF, ItemKNNCBF, SLIM_EN | 0.0668203 | 'alpha': 0.7783657178315921, 'l1_ratio': 0.9570845000744118
| P3alpha, ItemKNNCBF, SLIM_EN | 0.0676024 | 'alpha': 1.0, 'l1_ratio': 0.38257019222950617
| RP3beta, ItemKNNCBF, SLMI_EN | 0.0669610 | {'alpha': 0.9986952651067782, 'l1_ratio': 0.40273040559834994}
| UserKNNCF, ItemKNNCBF, SLIM_EN | 0.0668025 | {'alpha': 0.4960538848298478, 'l1_ratio': 0.3805091314133038}
| P3alpha, ItemKNNCBF, SLIM_BPR | 0.0681493 | 'alpha':0.5521014101582482, 'l1_ratio': 0.33884991586467056
| RP3beta, ItemKNNCBF, SLMI_BPR | 0.0676888 | {'alpha': 0.3592184105265153, 'l1_ratio': 0.2874563071761684}
| UserKNNCF, ItemKNNCBF, SLIM_BPR | 0.0666102 | {'alpha': 0.36976685355295563, 'l1_ratio': 0.46692198040584476}
| ItemKNNCBF, SLIM_EN, SLIM_BPR | 0.0667062 | {'alpha': 0.9651828293963296, 'l1_ratio': 0.14049121822460078}

## New Segmentation
* Range: **[0, 3)** //users with 1 or 2 interactions

| Algorithm | MAP | optimal parameters | Notes|
| ------ | ------| ------| ----|
| PureSVD | 0.0252548 | {'num_factors': 711} | configurazione al limite |
| SLIM_Elasticnet | 0.0329999 | {'topK': 954, 'l1_ratio': 3.87446082207643e-05, 'alpha': 0.07562657698792305} | - |
| Slim_BPR | 0.0288176 | {'topK': 1000, 'epochs': 45, 'symmetric': False, 'sgd_mode': 'sgd', 'lambda_i': 0.01, 'lambda_j': 1e-05, 'learning_rate': 0.0001} | - |
| IALS | 0.0397605 | {'num_factors': 83, 'confidence_scaling': 'linear', 'alpha': 28.4278070726612, 'epsilon': 1.0234211788885077, 'reg': 0.0027328110246575004, 'epochs': 20} | non ancora finito |
| ItemKNNCBF | 0.0377209 | {'topK': 225, 'shrink': 1000, 'similarity': 'cosine', 'normalize': True, 'feature_weighting': 'BM25'}
| ItemKNNCF | 0.0352769 | {'topK': 220, 'shrink': 175, 'similarity': 'cosine', 'normalize': False}
| UserKNNCF | 0.0384740 | {'topK': 405, 'shrink': 1000, 'similarity': 'asymmetric', 'normalize': True, 'asymmetric_alpha': 2.0}
| P3alpha | 0.0373801 | {'topK': 542, 'alpha': 0.0019504824518365997, 'normalize_similarity': False}
| RP3beta | 0.0368763 | {'topK': 974, 'alpha': 0.190821920493987, 'beta': 0.001834105482327875, 'normalize_similarity': False}

### Combo-performance [0,3)
| Combined Recs | MAP on **validation_hybrid** | optimal parameters |
| ------ | ------| ------|
| IALS, ItemKNNCBF, ItemKNNCF | 0.0749199 | 'alpha': 0.6728142273393015, 'l1_ratio': 0.6411491975484928
| IALS, ItemKNNCBF, P3alpha | 0.0731597 | 'alpha': 0.9978162748831466, 'l1_ratio': 0.6504590619983928
| IALS, ItemKNNCBF, PureSVD | 0.0730645 | 'alpha': 0.755832754442967, 'l1_ratio': 0.5589520785301885
| IALS, ItemKNNCBF, RP3beta | 0.0734170 | 'alpha': 0.9298391226293536, 'l1_ratio': 0.4542398844285046
| IALS, ItemKNNCBF, UserKNNCF | 0.0738064 | 'alpha': 0.21231738900634417, 'l1_ratio': 0.5983165993032866

* ItemKNNCBF, ItemKNNCF, P3alpha	-> MAP: 0.0653232
* ItemKNNCBF, ItemKNNCF, PureSVD	-> MAP: 0.0639673
* ItemKNNCBF, ItemKNNCF, RP3beta	-> MAP: 0.0702089
* ItemKNNCBF, ItemKNNCF, UserKNNCF-> MAP: 0.0676479
* ItemKNNCBF, P3alpha, PureSVD	-> MAP: 0.0677768
* ItemKNNCBF, P3alpha, RP3beta	-> MAP: 0.0689340
* ItemKNNCBF, RP3beta, PureSVD	-> MAP: 0.0696135
* ItemKNNCBF, UserKNNCF, P3alpha	-> MAP: 0.0651108
* IALS, ItemKNNCF, P3alpha		-> LOW (0.0516691)
* IALS, ItemKNNCF, PureSVD		-> LOW (0.0488379)
* IALS, ItemKNNCF, RP3beta		-> LOW (0.0523218)
* IALS, ItemKNNCF, UserKNNCF		-> LOW (0.0506494)
* IALS, P3alpha, PureSVD			-> LOW (0.0514158)
* IALS, P3alpha, RP3beta			-> LOW (0.0509690)
* IALS, RP3beta, PureSVD			-> LOW (0.0517357)
* IALS, UserKNNCF, P3alpha		-> LOW (0.0514009)
* IALS, UserKNNCF, PureSVD		-> LOW (0.0507924)
* IALS, UserKNNCF, RP3beta		-> LOW (0.0515633)
* ItemKNNCBF, UserKNNCF, PureSVD	-> LOW (0.0544276)
* ItemKNNCBF, UserKNNCF, RP3beta	-> LOW (0.0690862)
* ItemKNNCF, P3alpha, PureSVD		-> LOW (0.0490127)
* ItemKNNCF, P3alpha, RP3beta		-> LOW (0.0464365)
* ItemKNNCF, RP3beta, PureSVD		-> LOW (0.0479192)
* ItemKNNCF, UserKNNCF, P3alpha	-> LOW (0.0503812)
* ItemKNNCF, UserKNNCF, PureSVD	-> LOW (0.0487587)
* ItemKNNCF, UserKNNCF, RP3beta	-> LOW (0.0465236)
* P3alpha, RP3beta, PureSVD		-> LOW (0.0493235)
* UserKNNCF, P3alpha, RP3beta		-> LOW (0.0495286)
* UserKNNCF, P3alpha, PureSVD		-> LOW (0.0463064)
* UserKNNCF, RP3beta, PureSVD		-> LOW (0.0509870)

---
* Range: **[3, -1)** //users with more than 2 interactions

| Algorithm | MAP | optimal parameters | Notes|
| ------ | ------| ------| ----|
| ItemKNNCBF | 0.0302073 | {'topK': 205, 'shrink': 1000, 'similarity': 'cosine', 'normalize': True, 'feature_weighting': 'BM25'}
| ItemKNNCF | 0.0376856 | {'topK': 565, 'shrink': 554, 'similarity': 'tversky', 'normalize': True, 'tversky_alpha': 1.9109121434662428, 'tversky_beta': 1.7823834698905734}
| UserKNNCF | 0.0426341 | {'topK': 190, 'shrink': 0, 'similarity': 'cosine', 'normalize': True}
| P3alpha | 0.0447004 | {'topK': 438, 'alpha': 0.41923120471415165, 'normalize_similarity': False}
| RP3beta | 0.0435639 | {'topK': 753, 'alpha': 0.3873710051288722, 'beta': 0.0, 'normalize_similarity': False}
| SLIM_ElasticNet | 0.0412542 | {'topK': 517, 'l1_ratio': 2.164297353389958e-05, 'alpha': 0.006016185597042145}
| SLIM_BPR | 0.0384652 | {'topK': 732, 'epochs': 135, 'symmetric': True, 'sgd_mode': 'adagrad', 'lambda_i': 0.0010988284763975408, 'lambda_j': 0.0005670232873552422, 'learning_rate': 0.004876973627195952}
ItemKNNCBF, ItemKNNCF, P3alpha	-> MAP: 0.0686834  
ItemKNNCBF, ItemKNNCF, RP3beta	-> MAP: 0.0688763  
ItemKNNCBF, ItemKNNCF, UserKNNCF-> MAP: 0.0666803  
ItemKNNCBF, P3alpha, RP3beta	-> MAP: 0.0683870  
ItemKNNCBF, UserKNNCF, P3alpha	-> MAP: 0.0689529  
ItemKNNCBF, UserKNNCF, RP3beta	-> MAP: 0.0687652  
ItemKNNCF, P3alpha, RP3beta		-> MAP: 0.0563827  
ItemKNNCF, UserKNNCF, P3alpha	-> MAP: 0.0568228  
ItemKNNCF, UserKNNCF, RP3beta 	-> MAP: 0.0561869  
UserKNNCF, P3alpha, RP3beta		-> MAP: 0.0565108  
---
---
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
| Slim_BPR | 0.0422478 | _'topK': 100, 'epochs': 90, 'symmetric': True, 'sgd_mode': 'adam', 'lambda_i': 0.01, 'lambda_j': 0.01, 'learning_rate': 0.0001_ | - |
| SLIM_ElasticNet | 0.0618866 | _'topK': 585, 'l1_ratio': 0.0001006932689379397, 'alpha': 0.007119051585906447_ | - |


* Range: **[50, 100)**

| Algorithm | MAP | optimal parameters | notes |
| ------ | ------| ------| ------|
| PureSVD | 0.0355194 | _'num_factors': 233_ | **!!!** |
| RP3beta | 0.0429103 | _'topK': 174, 'alpha': 0.0, 'beta': 0.0, 'normalize_similarity': True_ | - |
| P3alpha | 0.0439474 | _'topK': 469, 'alpha': 0.0, 'normalize_similarity': True_ | - |
| UserKNNCF_cosine | 0.0477062 | _'topK': 90, 'shrink': 77, 'similarity': 'cosine', 'normalize': True_ | - |
| UserKNNCF_jaccard | 0.0424165 | _'topK': 55, 'shrink': 0, 'similarity': 'jaccard', 'normalize': False_ | - |
| ItemKNNCF_cosine | 0.0472611 | _'topK': 90, 'shrink': 77, 'similarity': 'cosine', 'normalize': True_ | - |
| ItemKNNCF_jaccard | 0.0445934 | _'topK': 140, 'shrink': 337, 'similarity': 'jaccard', 'normalize': False_ | - |
| ItemKNNCF_asym | 0.0455878 | _'topK': 90, 'shrink': 1000, 'similarity': 'asymmetric', 'normalize': True, 'asymmetric_alpha': 0.0_ | - |
| ItemKNNCBF_cosine | 0.0278491 | _'topK': 165, 'shrink': 95, 'similarity': 'cosine', 'normalize': True, 'feature_weighting': 'none'_ | - |
| ItemKNNCBF_jaccard | 0.0216655 | _'topK': 55, 'shrink': 0, 'similarity': 'jaccard', 'normalize': True_ | - |
| ItemKNNCBF_tversky | 0.0358659 | _{'topK': 15, 'shrink': 0, 'similarity': 'tversky', 'normalize': True, 'tversky_alpha': 0.0, 'tversky_beta': 0.0}_ | - |
| Slim_BPR | 0.0465718 | _'topK': 120, 'epochs': 110, 'symmetric': True, 'sgd_mode': 'adam', 'lambda_i': 1e-05, 'lambda_j': 1e-05, 'learning_rate': 0.0001_ | - |

* Range: **[25, 50)**

| Algorithm | MAP | optimal parameters | notes |
| ------ | ------| ------| ------|
| PureSVD | 0.0258015 | _'num_factors': 270_ | - |
| RP3beta | 0.0429689 | _'topK': 939, 'alpha': 0.6073516078011799, 'beta': 0.002238854541773972, 'normalize_similarity': False_ | - |
| P3alpha | 0.0373815 | _'topK': 186, 'alpha': 0.19068246754249213, 'normalize_similarity': True_ | - |
| UserKNNCF_asym | 0.0350267 | _'topK': 70, 'shrink': 1000, 'similarity': 'asymmetric', 'normalize': True, 'asymmetric_alpha': 2.0_ | - |
| UserKNNCF_cosine |0.0334044| _'topK': 90, 'shrink': 0, 'similarity': 'cosine', 'normalize': True_ |-|
| UserKNNCF_jaccard | 0.0368398 | _'topK': 65, 'shrink': 0, 'similarity': 'jaccard', 'normalize': False_ | - |
| ItemKNNCF_cosine | 0.0358939 | _'topK': 90, 'shrink': 77, 'similarity': 'cosine', 'normalize': True_ | - |
| ItemKNNCF_jaccard | 0.0357255 | _'topK': 140, 'shrink': 337, 'similarity': 'jaccard', 'normalize': False_ | - |
| ItemKNNCBF_cosine | 0.0205111 | _'topK': 370, 'shrink': 514, 'similarity': 'cosine', 'normalize': True, 'feature_weighting': 'BM25'_ | - |
| ItemKNNCBF_jaccard | 0.0190212 | _'topK': 105, 'shrink': 109, 'similarity': 'jaccard', 'normalize': False_ | - |
| Slim_BPR | 0.0345580 | _'topK': 420, 'epochs': 85, 'symmetric': True, 'sgd_mode': 'adam', 'lambda_i': 1e-05, 'lambda_j': 1e-05, 'learning_rate': 0.0001_ | - |

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
| Slim_BPR | 0.0461873 | _'topK': 1100, 'epochs': 135, 'symmetric': False, 'sgd_mode': 'adam', 'lambda_i': 0.0038059348762991855, 'lambda_j': 0.00039157722122128087, 'learning_rate': 0.0001_ | - |

---
* Range: **[25, 100)** <-- **OLD!**

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


# Progress

| Submission | Result | Note |
| ------ | ------| ------|
| ----.csv | 0.06583 | Item based KNN CF with parameters determined by hyperparam search.  (*Test submission*) |


# Notes
- check news on the discussion forum about **SLIM Elasticnet**
