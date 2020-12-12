#Cross-validation v.2
```
seed = 1666
# SPLIT ---- only one!
URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.85, seed=seed)
```
## Single-recommender performance

| Algorithm | MAP w/ **CV** | optimal parameters |
| ------ | ------| ------|
| ItemKNNCBF            | 0.0294841 | {'topK': 164, 'shrink': 8, 'similarity': 'jaccard', 'normalize': True}
| ItemKNNCBF_Special    | 0.0318313 | {'topK': 1000, 'shrink': 1000, 'similarity': 'cosine', 'normalize': True, 'feature_weighting': 'BM25'}
| ItemKNN_CBF_CF        | 0.0566790 | {'topK': 1000, 'shrink': 1000, 'similarity': 'asymmetric', 'normalize': True, 'asymmetric_alpha': 0.241892724784089, 'feature_weighting': 'TF-IDF', 'icm_weight': 1.0}
| ItemKNNCF             |  0.0441588 | {'topK': 1000, 'shrink': 1000, 'similarity': 'cosine', 'normalize': True, 'feature_weighting': 'TF-IDF'}
| UserKNNCF             | 0.0442522 | {'topK': 163, 'shrink': 846, 'similarity': 'cosine', 'normalize': True, 'feature_weighting': 'TF-IDF'}
| RP3Beta               | 0.0469976 | {'topK': 926, 'alpha': 0.4300109351916609, 'beta': 0.01807360750913967, 'normalize_similarity': False}
| P3alpha               | 0.0462770 | {'topK': 575, 'alpha': 0.48009885897470206, 'normalize_similarity': False}
| SLIM_BPR              | 0.0408258 | {'topK': 1000, 'epochs': 1500, 'symmetric': False, 'sgd_mode': 'adam', 'lambda_i': 1e-05, 'lambda_j': 1e-05, 'learning_rate': 0.0001, 'epochs0': 100, 'epochs1': 160, 'epochs2': 175, 'epochs3': 90, 'epochs4': 60}
| S-SLIM                | - | *to be added*
| PureSVD               | 0.0229563 | {'num_factors': 350}

## hybrid

| Algorithm | MAP w/ **CV** | optimal parameters |
| ------ | ------| ------|
| ItemKNNCBF-ItemKNN_CBF_CF-ItemKNNCF       | 0.0723551 | {'alpha': 0.9457062373092435, 'l1_ratio': 0.08675668683308081}
| ItemKNNCBF-ItemKNN_CBF_CF-P3alpha         | 0.0733132 | {'alpha': 0.6904008032335807, 'l1_ratio': 0.23371348584724985}
| ItemKNNCBF-ItemKNN_CBF_CF-RP3beta         | 0.0732289 | {'alpha': 0.8129428861277737, 'l1_ratio': 0.14069122456123387}
| ItemKNNCBF-ItemKNN_CBF_CF-UserKNNCF       | 0.0732549 | {'alpha': 0.8129428861277737, 'l1_ratio': 0.14069122456123387}
| ItemKNNCBF-ItemKNN_CBF_CF-Slim_BPR        | 0.0736289 | {'alpha': 0.8991751672246813, 'l1_ratio': 0.11874637825106651}
| ItemKNNCBF-ItemKNNCF-P3alpha              | 0.0705969 | {'alpha': 0.44569420382449637, 'l1_ratio': 0.6168742754254437}
| ItemKNNCBF-ItemKNNCF-RP3beta              | 0.0704951 | {'alpha': 0.3033751201293658, 'l1_ratio': 0.7822655054333012}
| ItemKNNCBF-ItemKNNCF-UserKNNCF            | 0.0700526 | {'alpha': 0.48973465405541533, 'l1_ratio': 0.3567553681370489}
| ItemKNNCBF-RP3beta-P3alpha                | 0.0703646 | {'alpha': 0.6171935935379741, 'l1_ratio': 0.4090970817953153}
| ItemKNNCBF-UserKNNCF-P3alpha              | 0.0711359 | {'alpha': 0.6310254748569911, 'l1_ratio': 0.42318219644561506}
| ItemKNNCBF-UserKNNCF-RP3beta              | 0.0710556 | {'alpha': 0.5388201552532741, 'l1_ratio': 0.35597718538152645}
| Special_KNNCBF-ItemKNN_CBF_CF-ItemKNNCF   | 0.0720238 | {'alpha': 0.8122492419947548, 'l1_ratio': 0.1448868067763832}
| Special_KNNCBF-ItemKNN_CBF_CF-P3alpha     | 0.0739867 | {'alpha': 0.6094679138906033, 'l1_ratio': 0.41715098578193194}
| Special_KNNCBF-ItemKNN_CBF_CF-RP3beta     | 0.0736200 | {'alpha': 0.5369911284211629, 'l1_ratio': 0.45078644454282}
| Special_KNNCBF-ItemKNN_CBF_CF-UserKNNCF   | 0.0729346 | {'alpha': 0.7020854096807992, 'l1_ratio': 0.20749665943340362}
| Special_KNNCBF-ItemKNN_CBF_CF-Slim_BPR    | 0.0739889 | {'alpha': 0.7949204415356104, 'l1_ratio': 0.2705694962114903}
| Special_KNNCBF-ItemKNNCF-P3alpha          | 0.0729078 | {'alpha': 0.4973334775913678, 'l1_ratio': 0.6790274919219236}
| Special_KNNCBF-ItemKNNCF-RP3beta **!!!**  | 0.0724929 | {'alpha': 0.45931982254221104, 'l1_ratio': 0.552400663560699}
| Special_KNNCBF-ItemKNNCF-UserKNNCF        | 0.0718252 | {'alpha': 0.5934342246244017, 'l1_ratio': 0.38793882442583466}
| Special_KNNCBF-ItemKNNCF-Slim_BPR         | 0.0723958 | {'alpha': 0.8170119826252383, 'l1_ratio': 0.33586027250410305}
| Special_KNNCBF-RP3beta-P3alpha            | 0.0729609 | {'alpha': 0.4001101186128191, 'l1_ratio': 0.9989082076640393}
| Special_KNNCBF-UserKNNCF-P3alpha          | 0.0732810 | {'alpha': 0.627349023025726, 'l1_ratio': 0.5343474311262314}
| Special_KNNCBF-UserKNNCF-RP3beta          | 0.0730215 | {'alpha': 0.5574853559798421, 'l1_ratio': 0.5221328226891215}
| Special_KNNCBF-UserKNNCF-Slim_BPR         | 0.0726270 | {'alpha': 0.9033873534648698, 'l1_ratio': 0.30678626017844307}
| ItemKNN_CBF_CF-ItemKNNCF-P3alpha          | 0.0700128 | {'alpha': 0.9849245373595942, 'l1_ratio': 0.9961549824044795}
| ItemKNN_CBF_CF-ItemKNNCF-RP3beta          | 0.0700284 | {'alpha': 0.9960369080995459, 'l1_ratio': 0.9991666164394999}
| ItemKNN_CBF_CF-ItemKNNCF-UserKNNCF        | 0.0700190 | {'alpha': 0.9978504378244369, 'l1_ratio': 0.9989504324961238}
| ItemKNN_CBF_CF-...                        |     =     | all the same with config almost 1,1



---
#Cross-validation
```
seed = 1666
# SPLIT TO GET TEST PARTITION
URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.85, seed=seed)

# SPLIT TO GET THE HYBRID VALID PARTITION
URM_train, URM_valid_hybrid = split_train_in_two_percentage_global_sample(URM_train, train_percentage = 0.85, seed=seed)
```
## Sub-recommenders performance

| Algorithm | MAP on **CV** | optimal parameters |
| ------ | ------| ------|
| ItemKNNCBF_jaccard | 0.0258054 | {'topK': 178, 'shrink': 100, 'similarity': 'jaccard', 'normalize': True}
| **ItemKNNCBF_dice** | 0.0262233 | {'topK': 22, 'shrink': 59, 'similarity': 'dice', 'normalize': False}
| ItemKNNCBF_tversky | 0.0262111 | {'topK': 282, 'shrink': 13, 'similarity': 'tversky', 'normalize': True, 'tversky_alpha': 1.1415654194011693, 'tversky_beta': 1.8776506336950853}
| **ItemKNNCF_asym** | 0.0351144 | {'topK': 994, 'shrink': 981, 'similarity': 'asymmetric', 'normalize': True, 'asymmetric_alpha': 0.05110465631417439, 'feature_weighting': 'TF-IDF'}
| ItemKNNCF_cosine | 0.0338102 | {'topK': 127, 'shrink': 345, 'similarity': 'cosine', 'normalize': False, 'feature_weighting': 'TF-IDF'}
| UserKNNCF_asym | 0.0341739 | {'topK': 321, 'shrink': 914, 'similarity': 'asymmetric', 'normalize': True, 'asymmetric_alpha': 0.4695015961698127, 'feature_weighting': 'TF-IDF'}
| **UserKNNCF_cosine** | 0.0344068 | {'topK': 342, 'shrink': 1, 'similarity': 'cosine', 'normalize': True, 'feature_weighting': 'none'}
| UserKNNCF_jaccard | 0.0340966 | {'topK': 393, 'shrink': 18, 'similarity': 'jaccard', 'normalize': False}
| UserKNNCF_dice | 0.0341211 |{'topK': 282, 'shrink': 8, 'similarity': 'dice', 'normalize': True}
| **RP3Beta** | 0.0345242 | {'topK': 121, 'alpha': 0.20195468746532336, 'beta': 0.004377103282444673, 'normalize_similarity': False}
| **P3alpha** | 0.0357172 | {'topK': 344, 'alpha': 0.4233795425391197, 'normalize_similarity': False}
| **ItemKNN_CBF_CF_asym** | 0.0466542 | {'topK': 651, 'shrink': 858, 'similarity': 'asymmetric', 'normalize': True, 'asymmetric_alpha': 0.01038549339003936, 'feature_weighting': 'TF-IDF', 'icm_weight': 1.0091171832447683}
| ItemKNN_CBF_CF_cosine | 0.0449181 | {'topK': 1000, 'shrink': 1000, 'similarity': 'cosine', 'normalize': True, 'feature_weighting': 'TF-IDF', 'icm_weight': 1.0}
| ItemKNN_CBF_CF_dice | 0.0383243 | {'topK': 103, 'shrink': 996, 'similarity': 'dice', 'normalize': True, 'icm_weight': 19.882204659440184}
| ItemKNN_CBF_CF_jaccard | 0.0382352 | {'topK': 100, 'shrink': 694, 'similarity': 'jaccard', 'normalize': False, 'icm_weight': 15.566263173944517}
| ItemKNN_CBF_CF_twersky | 0.0384311 | {'topK': 106, 'shrink': 979, 'similarity': 'tversky', 'normalize': True, 'tversky_alpha': 0.09608567445574437, 'tversky_beta': 0.0872747385379682, 'icm_weight': 18.89627667671209}
| SLIM_BPR | 0.0314483 | {'topK': 989, 'epochs': 1500, 'symmetric': False, 'sgd_mode': 'adam', 'lambda_i': 1.7432198130463203e-05, 'lambda_j': 0.0016819750046109673, 'learning_rate': 0.00031293205801039345, 'epochs0': 120, 'epochs1': 90, 'epochs2': 80, 'epochs3': 20, 'epochs4': 105}


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
| S-SLIM_EN | 0.0464815 | {'beta': 0.4849594591575789, 'topK': 1000, 'l1_ratio': 1e-05, 'alpha': 0.001}

## Combo performance

* Linear Combination

| Combined Recs | MAP on **validation_hybrid** | optimal parameters |
| ------ | ------| ------|
| P3alpha, ItemKNNCF, UserKNNCF| 0.0542544 | 'alpha': 0.5630173292967398, 'l1_ratio': 0.92733589638295
| RP3Beta, ItemKNNCF, UserKNNCF| 0.0533618 | 'alpha': 0.7416313012430469, 'l1_ratio': 0.8122593875086325
| RP3Beta, P3alpha, ItemKNNCF| 0.0535307 | 'alpha': 0.367698407319822, 'l1_ratio': 0.5878133798647788
| RP3Beta, P3alpha, UserKNNCF| 0.0541693 | 'alpha': 0.6405838432360388, 'l1_ratio': 0.4188312253799342
| ItemKNNCF, ItemKNNCBF, SLIM_EN | 0.0528577 | {'alpha': 0.21686976560272436, 'l1_ratio': 0.4598014054291886}
| ItemKNNCF, SLIM_EN, SLIM_BPR | 0.0514854 | {'alpha': 0.6342517822083686, 'l1_ratio': 0.38051846734018036}
| ItemKNNCF, UserKNNCF, ItemKNNCBF | 0.0551535 | {'alpha': 0.6094266148134767, 'l1_ratio': 0.46668601356447076}
| ItemKNNCF, UserKNNCF, SLIM_EN | 0.0518492 | {'alpha': 0.8356363650152253, 'l1_ratio': 0.9163620505237737}
| P3alpha, ItemKNNCBF, SLIM_EN | 0.0548520 | {'alpha': 0.9999772418587548, 'l1_ratio': 0.28511052552468436}
| P3alpha, ItemKNNCF, ItemKNNCBF | 0.0553892 | {'alpha': 0.4878994539449091, 'l1_ratio': 0.4667353090819475}
| P3alpha, ItemKNNCF, SLIM_EN | 0.0529564 | {'alpha': 1.0, 'l1_ratio': 1.0}
| P3alpha, SLIM_EN, SLIM_BPR | 0.0529564 | {'alpha': 1.0, 'l1_ratio': 1.0}
| P3alpha, UserKNNCF, ItemKNNCBF | 0.0556535 | {'alpha': 0.3987236515679141, 'l1_ratio': 0.15489605895390016}
| P3alpha, UserKNNCF, SLIM_EN | 0.0532338 | {'alpha': 0.626160652050037, 'l1_ratio': 0.5469215188707677}
| RP3beta, ItemKNNCBF, SLIM_EN | 0.0559213 | {'alpha': 1.0, 'l1_ratio': 0.3951763029766836}
| RP3beta, ItemKNNCF, ItemKNNCBF | 0.0561934 | {'alpha': 0.4577946628581237, 'l1_ratio': 0.7434539743766688}
| RP3beta, ItemKNNCF, SLIM_EN | 0.0525393 |{'alpha': 0.8999667813934427, 'l1_ratio': 0.3877035588220962}
| RP3beta, P3alpha, ItemKNNCBF | 0.0557777 | {'alpha': 0.3619673282977996, 'l1_ratio': 0.997620008978927}
| RP3beta, SLIM_EN, SLIM_BPR | 0.0523304 | {'alpha': 0.45684305746620557, 'l1_ratio': 0.6484637757274762}
| RP3beta, UserKNNCF, ItemKNNCBF | 0.0561958 |{'alpha': 0.3787194374490951, 'l1_ratio': 0.706212775218188}
| RP3beta, UserKNNCF, SLIM_EN | 0.0524121 | {'alpha': 0.4629787518361874, 'l1_ratio': 0.7137640115869015}
| UserKNNCF, ItemKNNCBF, SLIM_EN | 0.0557388 | {'alpha': 0.33535858857401674, 'l1_ratio': 0.4046400351885727}
| UserKNNCF, SLIM_EN, SLIM_BPR | 0.0519567 | {'alpha': 0.7231977321772529, 'l1_ratio': 0.09639275029166919}
| ItemKNNCBF, SLIM_EN, SLIM_BPR | 0.0542334 | {'alpha': 0.7321778261479165, 'l1_ratio': 0.15333729621089734}
| ItemKNNCF, ItemKNNCBF, SLIM_BPR | 0.0516746 | {'alpha': 0.1452330659544545, 'l1_ratio': 0.505309437774802}
| P3alpha, ItemKNNCBF, SLIM_BPR | 0.0551299 | {'alpha': 0.6336877202461957, 'l1_ratio': 0.2923784696799847}
| RP3beta, ItemKNNCBF, SLIM_BPR | 0.0560030 | {'alpha': 0.9902553208856146, 'l1_ratio': 0.36299727894312356}
| UserKNNCF, ItemKNNCBF, SLIM_BPR | 0.0546817 | {'alpha': 0.7647242911516163, 'l1_ratio': 0.4701015482243481}


| **ERR** ItemKNNCF, UserKNNCF, ItemKNNCBF **ERR** | 0.0659249 | 'alpha': 0.767469300493861, 'l1_ratio': 0.7325725081659069
| **ERR** P3alpha, ItemKNNCF, ItemKNNCBF **ERR** | 0.0677491 | 'alpha': 0.4066665999396494, 'l1_ratio': 0.7594645794234393
| **ERR** P3alpha, UserKNNCF, ItemKNNCBF **ERR** | 0.0681188 | 'alpha': 0.37776131907747645, 'l1_ratio': 0.44018901104481
| **ERR** RP3Beta, ItemKNNCF, ItemKNNCBF **ERR** | 0.0671482 | 'alpha': 0.40426999639005445, 'l1_ratio': 1.0
| **ERR** RP3Beta, P3alpha, ItemKNNCBF **ERR** | 0.0677085 | 'alpha': 0.3553383791480798, 'l1_ratio': 0.000435281815357902
| **ERR** RP3Beta, UserKNNCF, ItemKNNCBF **ERR** | 0.0680765 | 'alpha': 0.4648716125499346, 'l1_ratio': 0.292302921903516
| **ERR** ItemKNNCF, ItemKNNCBF, SLIM_EN **ERR** | 0.0668203 | 'alpha': 0.7783657178315921, 'l1_ratio': 0.9570845000744118
| **ERR** P3alpha, ItemKNNCBF, SLIM_EN **ERR** | 0.0676024 | 'alpha': 1.0, 'l1_ratio': 0.38257019222950617
| **ERR** RP3beta, ItemKNNCBF, SLMI_EN **ERR** | 0.0669610 | {'alpha': 0.9986952651067782, 'l1_ratio': 0.40273040559834994}
| **ERR** UserKNNCF, ItemKNNCBF, SLIM_EN **ERR** | 0.0668025 | {'alpha': 0.4960538848298478, 'l1_ratio': 0.3805091314133038}
| **ERR** P3alpha, ItemKNNCBF, SLIM_BPR **ERR** | 0.0681493 | 'alpha':0.5521014101582482, 'l1_ratio': 0.33884991586467056
| **ERR** RP3beta, ItemKNNCBF, SLMI_BPR **ERR** | 0.0676888 | {'alpha': 0.3592184105265153, 'l1_ratio': 0.2874563071761684}
| **ERR** UserKNNCF, ItemKNNCBF, SLIM_BPR **ERR** | 0.0666102 | {'alpha': 0.36976685355295563, 'l1_ratio': 0.46692198040584476}
| **ERR** ItemKNNCBF, SLIM_EN, SLIM_BPR **ERR** | 0.0667062 | {'alpha': 0.9651828293963296, 'l1_ratio': 0.14049121822460078}

* Merge Combination

| Combined Recs | MAP on **validation_hybrid** | optimal parameters |
| ------ | ------| ------|
| ItemKNNCBF, ItemKNNCF, P3alpha | 0.0655122 | {'alpha': 0.9004575733942728, 'l1_ratio': 0.26510663025884135, 'topK': 888}
| ItemKNNCBF, ItemKNNCF, RP3beta | 0.0648048 | {'alpha': 0.6299182764826382, 'l1_ratio': 0.9675762628242017, 'topK': 864}
| ItemKNNCBF, ItemKNNCF, SLIM_EN | 0.0634502 | {'alpha': 0.3791657333411357, 'l1_ratio': 0.23126551664267164, 'topK': 990}
| ItemKNNCBF, P3alpha, RP3beta | 0.0621390 | {'alpha': 0.6299182764826382, 'l1_ratio': 0.9675762628242017, 'topK': 864}
| ItemKNNCBF, P3alpha, SLIM_EN | 0.0651354 | {'alpha': 0.9712321721669441, 'l1_ratio': 0.6879306727642494, 'topK': 990}
| ItemKNNCBF, RP3beta, SLIM_EN | 0.0652897 | {'alpha': 0.6355738550417837, 'l1_ratio': 0.6617849709204384, 'topK': 538}
| ItemKNNCF, P3alpha, RP3beta | 0.0541159 | {'alpha': 0.44112400857241485, 'l1_ratio': 0.4636970676140909, 'topK': 489}
| ItemKNNCF, P3alpha, SLIM_EN | 0.0529576 | {'alpha': 1.0, 'l1_ratio': 0.0, 'topK': 1000} --> only P3alpha
| ItemKNNCF, RP3beta, SLIM_EN | 0.0540153 | {'alpha': 1.0, 'l1_ratio': 0.0, 'topK': 373} ---> only RP3beta
| P3alpha, RP3beta, SLIM_EN | 0.0530004 | {'alpha': 0.8589000153868548, 'l1_ratio': 0.4263692698842382, 'topK': 636}

* Linear Combination with **NORMALIZATION**

| Combined Recs | MAP on **validation_hybrid** | optimal parameters |
| ------ | ------| ------|
| ItemKNNCBF, ItemKNNCF, P3alpha | 0.0640789 | {'alpha': 0.27094237120805925, 'l1_ratio': 0.9998550263180496}
| ItemKNNCBF, ItemKNNCF, RP3beta | 0.0630379 | {'alpha': 0.30215925285279965, 'l1_ratio': 0.48033878522040957}
| ItemKNNCBF, ItemKNNCF, SLIM_BPR | 0.0610412 | {'alpha': 0.8124123649444593, 'l1_ratio': 0.25648942001711633}
| ItemKNNCBF, ItemKNNCF, SLIM_EN | 0.0617823 | {'alpha': 0.5011523958099686, 'l1_ratio': 0.6835202748204333}
| ItemKNNCBF, ItemKNNCF, UserKNNCF | 0.0626716 | {'alpha': 0.23127746676269062, 'l1_ratio': 0.47037997538194565}
| ItemKNNCBF, P3alpha, RP3beta | 0.0638838 | {'alpha': 0.32630431880229166, 'l1_ratio': 0.4598175570701317}
| ItemKNNCBF, P3alpha, SLIM_BPR | 0.0638735 | {'alpha': 0.9966101776460795, 'l1_ratio': 0.24103276386735034}
| ItemKNNCBF, P3alpha, SLIM_EN | 0.0639945 | {'alpha': 0.9678477202248401, 'l1_ratio': 0.26287752701640094}
| ItemKNNCBF, RP3beta, SLIM_BPR | 0.0615707 | {'alpha': 0.7781420779249448, 'l1_ratio': 0.27134602211348197}
| ItemKNNCBF, RP3beta, SLIM_EN | 0.0624512 | {'alpha': 0.7389620492322015, 'l1_ratio': 0.3355244273594045}
| ItemKNNCBF, SLIM_EN, SLIM_BPR | 0.0623007 | {'alpha': 0.7753869940322178, 'l1_ratio': 0.4380906766055395}
| ItemKNNCBF, UserKNNCF, P3alpha | 0.0647686 | {'alpha': 0.4840277797811048, 'l1_ratio': 0.356265002578487}
| ItemKNNCBF, UserKNNCF, RP3beta | 0.0640256 | {'alpha': 0.39585017788453875, 'l1_ratio': 0.2969289208656836}
| ItemKNNCBF, UserKNNCF, SLIM_BPR | 0.0615470 | {'alpha': 0.8170261573317669, 'l1_ratio': 0.20761151334998523}
| ItemKNNCBF, UserKNNCF, SLIM_EN | 0.0625221 | {'alpha': 0.696288087190842, 'l1_ratio': 0.3051487184937027}
| ItemKNNCF, P3alpha, RP3beta | 0.0540847 | {'alpha': 0.13211292014716505, 'l1_ratio': 0.9841972871240037}
| ItemKNNCF, P3alpha, SLIM_BPR | 0.0532125 | {'alpha': 0.7843155132242284, 'l1_ratio': 0.20247012940631276}
| ItemKNNCF, P3alpha, SLIM_EN | 0.0529564 | {'alpha': 1.0, 'l1_ratio': 0.0} --> **pure P3alpha**
| ItemKNNCF, RP3beta, SLIM_BPR | 0.0529982 | {'alpha': 0.9873729786283573, 'l1_ratio': 0.11357727617112759}
| ItemKNNCF, RP3beta, SLIM_EN | 0.0523226 | {'alpha': 0.9949623682515907, 'l1_ratio': 0.007879399002699851}
| ItemKNNCF, SLIM_EN, SLIM_BPR | 0.0512052 | {'alpha': 0.8434066208554849, 'l1_ratio': 0.4628304123637582}
| ItemKNNCF, UserKNNCF, P3alpha | 0.0532817 | {'alpha': 0.4318367153026247, 'l1_ratio': 0.06753630242082177}
| ItemKNNCF, UserKNNCF, RP3beta | 0.0539684 | {'alpha': 0.19390580434445875, 'l1_ratio': 0.2522643001286576}
| ItemKNNCF, UserKNNCF, SLIM_BPR | 0.0519019 | {'alpha': 0.9083484410214647, 'l1_ratio': 0.27023920102996396}
| ItemKNNCF, UserKNNCF, SLIM_EN | 0.0515942 | {'alpha': 0.8854496569140459, 'l1_ratio': 0.17103030574896252}
| P3alpha, RP3beta, SLIM_BPR | 0.0530645 | {'alpha': 0.9595683171219453, 'l1_ratio': 0.8747916676441719}
| P3alpha, RP3beta, SLIM_EN | 0.0529564 | {'alpha': 1.0, 'l1_ratio': 1.0} --> **pure P3alpha**
| P3alpha, SLIM_EN, SLIM_BPR |  | {'alpha': 1.0, 'l1_ratio': 1.0} --> **pure P3alpha**
| RP3beta, SLIM_EN, SLIM_BPR |  | {'alpha': 1.0, 'l1_ratio': 1.0} --> **pure RP3beta**
| UserKNNCF, P3alpha, RP3beta | 0.0540508 | {'alpha': 0.9394644725243001, 'l1_ratio': 0.4569374184464653}
| UserKNNCF, P3alpha, SLIM_EN | 0.0532232 | {'alpha': 0.9914065625485552, 'l1_ratio': 0.626411568207146}
| UserKNNCF, P3alpha, SLIM_BPR |  | {'alpha': 1.0, 'l1_ratio': 0.0} --> **pure P3alpha**
| UserKNNCF, RP3beta, SLIM_EN | 0.0526332 | {'alpha': 0.9928115184812, 'l1_ratio': 0.4394977708983866}
| UserKNNCF, RP3beta, SLIM_BPR | 0.0523734 | {'alpha': 0.9824185519731099, 'l1_ratio': 0.0049365500554859896}
| UserKNNCF, SLIM_EN, SLIM_BPR | 0.0511976 | {'alpha': 0.9979141772440999, 'l1_ratio': 0.9695756615044695}
| S-SLIM_EN, ItemKNNCBF, ItemKNNCF | 0.0662454 | {'alpha': 0.996772013761913, 'l1_ratio': 0.7831508517025596}
| S-SLIM_EN, ItemKNNCBF, UserKNNCF | 0.0672512 | {'alpha': 0.6461624491197696, 'l1_ratio': 0.7617220099582368}
| S-SLIM_EN, ItemKNNCF, UserKNNCF | 0.0601189 | {'alpha': 0.6418228871731989, 'l1_ratio': 1.0}
| S-SLIM_EN, P3alpha, ItemKNNCBF | 0.0664154 | {'alpha': 0.8416340030829476, 'l1_ratio': 0.6651408407090509}
| S-SLIM_EN, P3alpha, ItemKNNCF | 0.0598208 | {'alpha': 0.9847198829156348, 'l1_ratio': 0.9996962519963414}
| S-SLIM_EN, P3alpha, RP3beta | 0.0598264 | {'alpha': 0.9924434125443558, 'l1_ratio': 0.9904105385647505}
| S-SLIM_EN, P3alpha, UserKNNCF | 0.0601001 | {'alpha': 0.7021255257436378, 'l1_ratio': 0.8799646154792433}
| S-SLIM_EN, RP3beta, ItemKNNCBF | 0.0669431 | {'alpha': 0.8416340030829476, 'l1_ratio': 0.6651408407090509}
| S-SLIM_EN, RP3beta, ItemKNNCF | 0.0597954 | {'alpha': 0.99818657042913, 'l1_ratio': 0.9852822057143448}
| S-SLIM_EN, RP3beta, UserKNNCF | 0.0602100 | {'alpha': 0.6287606296404341, 'l1_ratio': 0.9991721235523667}


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
