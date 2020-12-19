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


def print_res(all_rec, at=None):
    size = len(all_rec) if at is None else at
    for i in range(size):
        rec=all_rec[i]
        if not rec['content'][1]['errors']:
            print(f"{i+1}# {rec['rec_name']} - MAP: {rec['content'][0]['results']}"+", config: {"+f"{rec['content'][0]['config']}"+"}")
        else:
            try:
                print(f"{i+1}# {rec['rec_name']} - MAP: {rec['content'][0]['results']}"+", config: {"+f"{rec['content'][0]['config']}"+"} [Warning: Something went wrong]")
            except:
                print(f"{i+1}# {rec['rec_name']} [Warning: Something went wrong]")


if __name__ == '__main__':
    folder = 'result_experiments_CV2/seed_1666/linear_v2'
    filename = 'HybridCombinationSearchCV_SearchBayesianSkopt-CV.txt'
    sub_folders = list(os.walk(folder))[0][1]

    all_rec = []
    for sub_folder in sub_folders:
        dict_ = {}
        path = os.path.join(folder, sub_folder, filename)
        with open(path) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        dict_['rec_name'] = sub_folder
        dict_['content'] = get_best_config(content)
        all_rec = append_sorted(all_rec, dict_)
    print_res(all_rec)
