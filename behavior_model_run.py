from pathlib import Path

import pickle

import re

from clustering import *

if __name__ == '__main__':
    top_n_models = 50
    repeat_per_model = 1
    only_within = False

    data_root = Path('data/')
    pickle_root = Path('ga_out/')

    csv_list = list(data_root.glob('*.csv'))
    csv_list = sorted(csv_list, key=lambda x: int(re.findall('(?<=_n)[0-9]*', x.name)[0]))
    print(csv_list)

    graph_dict = {csv.name: Graph(csv) for csv in csv_list}
    models_dict = {csv.name: pickle.load((pickle_root / '{}.pickle'.format(csv.name)).open('rb')) for csv in csv_list}
    result_lines = []

    for target_csv in csv_list:
        start_time = time.time()
        result_list = []
        result_list.append(target_csv.name)
        target_graph = graph_dict[target_csv.name]

        for training_csv in csv_list:
            models = models_dict[training_csv.name][:top_n_models]
            best_mq = 0

            if only_within and target_csv != training_csv:
                continue

            for model in models:
                model_best_mq = max([model.generate_cluster(target_graph).mq for _ in range(repeat_per_model)])
                best_mq = max(best_mq, model_best_mq)

            result_list.append(str(best_mq))

        result_list.append(str(time.time() - start_time))
        result_lines.append(', '.join(result_list) + '\n')
        print(result_lines[-1])

    with (pickle_root / 'result.csv').open('w') as f:
        f.writelines(result_lines)
