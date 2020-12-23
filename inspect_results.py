import os


def get_best_config(file_content):
    complete = errors = best_found = False
    complete_metadata = {None}
    best_metadata = {}
    for l in file_content:
        if 'Search complete' in l:
            complete = True
            # get complete metadata ? Only for cross-checking

        elif 'New best config found' in l:
            complete = False
            best_found = True
            # get best_config metadata
            config_s = l.find('{')
            config_e = l.find('}')
            best_metadata['config'] = l[config_s + 1:config_e]
            results_i = l.find('MAP') + 5
            results_e = results_i + 8
            best_metadata['results'] = float(l[results_i:results_e + 1])

    errors = not complete or not best_found
    return best_metadata, {'complete': complete, 'best_found': best_found, 'errors': errors}


def append_sorted(container, content):
    # print(content)
    container = container.copy()
    value = content['content'][0]['results']
    index = None
    for pos, dict_ in enumerate(container):
        if dict_['content'][0]['results'] <= value:
            index = pos
            break
    # print('#',index, value)
    if index is None:
        container.append(content)
    elif index > 0:
        container.insert(index, content)
    else:
        container.insert(0, content)
    return container


def print_res(all_rec, type_, treshold, at=None):
    size = len(all_rec) if at is None else at
    for i in range(size):
        rec=all_rec[i]
        if treshold is not None and rec['content'][0]['results'] < treshold:
            continue
        if not rec['content'][1]['errors']:
            print(f"{i+1}# {rec['rec_name']} - MAP: {rec['content'][0]['results']}"+", config: {"+f"{rec['content'][0]['config']}"+"}")
        else:
            try:
                print(f"{i+1}# {rec['rec_name']} - MAP: {rec['content'][0]['results']}"+", config: {"+f"{rec['content'][0]['config']}"+"} [Warning: Something went wrong]")
            except:
                print(f"{i+1}# {rec['rec_name']} [Warning: Something went wrong]")


def convert_name(rec_name: str):
    f_words = {
        'SLIM_BPR_Recommender': 'SLIMBPRRecommender',
        'ItemKNN_CBF_CF_Recommender': 'ItemKNNCBFCFRecommender'
    }
    dict_name = {
        'Special-ItemKNNCBFRec': 'icbsup',
        'ItemKNNCBFCFRecommender': 'icfcb',
        'P3alphaRecommender': 'p3a',
        'SLIMBPRRecommender': 'sbpr',
        'S-SLIMElasticNet': 'sslim',
        'ItemKNNCFRecommender': 'icf',
        'RP3betaRecommender': 'rp3b',
        'UserKNNCFRecommender': 'ucf',
        'ItemKNNCBFRecommender': 'icb',
        'IALSRecommender': 'ials',
        'PureSVDRecommender': 'psvd',
    }
    for k in f_words.keys():
        if k in rec_name:
            rec_name = rec_name.replace(k, f_words[k])
    names = rec_name.split('_')
    final_str = ''
    for rec in names:
        final_str += dict_name[rec]+', '
    return final_str[:-2]


def print_res2(all_rec, type_, treshold, at=None):
    size = len(all_rec) if at is None else at
    for i in range(size):
        rec = all_rec[i]
        if treshold is not None and rec['content'][0]['results'] < treshold:
            continue
        str_ = convert_name(rec['rec_name'])
        try:
            print("('"+str_+"', '"+type_+"', ["+str_+"], {"+rec['content'][0]['config']+"}),")
        except:
            continue


if __name__ == '__main__':
    folder = 'result_experiments_CV2/seed_1666/linear_v2'
    filename = 'HybridCombinationSearchCV_SearchBayesianSkopt-CV.txt'
    constraint = ''
    type_ = 'linear'
    treshold = 0.071
    sub_folders = list(os.walk(folder))[0][1]

    all_rec = []
    for sub_folder in sub_folders:
        if constraint not in sub_folder:
            continue
        dict_ = {}
        path = os.path.join(folder, sub_folder, filename)
        with open(path) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        dict_['rec_name'] = sub_folder
        dict_['content'] = get_best_config(content)
        all_rec = append_sorted(all_rec, dict_)
    print_res2(all_rec, type_, treshold)
