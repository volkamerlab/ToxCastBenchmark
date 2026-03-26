from datasail.sail import datasail
import pandas as pd
import os
import re
import sys

def main(delta_and_epsilon = 0.05):
    with open('datasail_inputs.txt', 'r') as inputs:
        assay_filenames = inputs.read().splitlines()

    for assay_filname in assay_filenames:

        df = pd.read_csv(assay_filname, sep='\t')
        assay_name = assay_filname.split('/')[-1]
        match = re.match(r"^(.*?)(-\d{4}-\d{2}-\d{2})", assay_name)
        subfolder = assay_filname.split('/')[5]
        if not subfolder in ['steroidal', 'androgens', 'estrogens', 'progestagens', 'glucocorticoids']:
            continue
        if match:
            assay_name = match.group(1)
        try:
            e_splits, _, _ = datasail(
                # we have one dimension and want a distance-based split and we use the "e entity"
                techniques=["C1e"],
                # 5 splits of size 20% of the data
                splits=[int(round(len(df)/5, 0)) for _ in range(5)],
                names=[f"fold{i}" for i in range(5)],
                runs=1,  # i guess since we do CV it's not necessary to rerun
                solver="SCIP",
                e_type="M",  # indicates we have a molecule
                e_data=dict(df[["compound", "smiles"]].values.tolist()),
                e_strat=dict(df[["compound", "response"]].values.tolist()),
                epsilon = delta_and_epsilon,
                delta = delta_and_epsilon
            )
            res = e_splits['C1e'][0]
            if not len(res.keys()) == len(df):
                with open('weird_output.txt', 'a') as not_feasible_file:
                    not_feasible_file.write(
                        f'{assay_name}\t{abs(len(res.keys()) - len(df))}\n')
        except:
            with open('not_feasible_03.txt', 'a') as not_feasible_file:
                not_feasible_file.write(f'{assay_name}\n')
            continue
        for f in range(5):
            os.makedirs(
                f'../model_inputs/{subfolder}//{assay_name}//fold{f}/', exist_ok=True)
            with open(f'../model_inputs/{subfolder}//{assay_name}//fold{f}/train.txt', 'w') as train:
                with open(f'../model_inputs/{subfolder}//{assay_name}//fold{f}/test.txt', 'w') as test:
                    for compound in res.keys():

                        fold = res[compound]

                        if fold == f'fold{f}':
                            test.write(f'{compound}\n')
                        else:
                            train.write(f'{compound}\n')
            ht_fold_index = 0
            for ht_fold in range(5):
                if ht_fold == f:
                    continue
                os.makedirs(
                    f'../model_inputs/{subfolder}//{assay_name}//fold{f}/hyperparameter_tuning/fold{ht_fold_index}/', exist_ok=True)
                with open(f'../model_inputs/{subfolder}//{assay_name}//fold{f}/hyperparameter_tuning/fold{ht_fold_index}/train.txt', 'w') as train:
                    with open(f'../model_inputs/{subfolder}//{assay_name}//fold{f}/hyperparameter_tuning/fold{ht_fold_index}/test.txt', 'w') as test:
                        for compound in res.keys():
                            fold = res[compound]
                            if fold == f'fold{ht_fold}':
                                test.write(f'{compound}\n')
                            elif fold == f'fold{f}':
                                continue
                            else:
                                train.write(f'{compound}\n')
                ht_fold_index += 1
        with open('considered_hormone_assay_names.txt', 'a') as assays_done:
            assays_done.write(f'{assay_name}\n')
        with open(f'considered_hormone_assays_epsilon_and_delta_{delta_and_epsilon}.txt', 'a') as only_feasible_now:
            only_feasible_now.write(f'{assay_name}\n')


if __name__ == '__main__':
    main(sys.argv[1])
