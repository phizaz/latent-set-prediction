from data.mimic_cxr import *
import pickle
from tqdm import tqdm
import json


def post_process(x):
    return x.replace(' .', '.').replace('. ', '.\n').split('\n')


root_dir = 'baselines/RM_MCLN'

with open(f'{root_dir}/data/annotation_ours2.json') as f:
    dataset = json.load(f)

mimic_data = MimicTextDatasetConfig(report='with_finding').make_dataset()
df = mimic_data.test.report_df

with open(f'{root_dir}/results/mimic_cxr2_aug/gen_test.pkl', 'rb') as f:
    # load the generation results
    data_r2 = pickle.load(f)
    data_r2['study_id'] = []
    data_gt = []
    by_study_id = {}
    # NOTE: we don't use the ground truths created by RM_MCLN library (it dropped a few characters)
    # we used the original texts and ground truths
    for i in range(len(data_r2['pred'])):
        # assign the corresponding study_id to the prediction
        data_r2['study_id'].append(dataset['test'][i]['study_id'])
        data_r2['pred'][i] = post_process(data_r2['pred'][i])
        data_r2['gt'][i] = post_process(data_r2['gt'][i])
        by_study_id[dataset['test'][i]['study_id']] = {
            'pred': data_r2['pred'][i],
            'gt': data_r2['gt'][i],
        }
    # create the corresponding grund truths
    for study_id in tqdm(data_r2['study_id']):
        text = df[df['study_id'] == study_id].iloc[0]['text']
        if text != text:
            text = ''
        text = text.lower()
        data_gt.append(text.split('\n'))

# print(data_gt[123])
# print(data_r2['gt'][123])

assert len(data_r2['pred']) == len(data_gt) == len(mimic_data.test)

# take a few minutes...
bleu = MimicTextCombinedDataset.evaluate(data_r2['pred'],
                                         data_gt,
                                         progress=True)
print(bleu)