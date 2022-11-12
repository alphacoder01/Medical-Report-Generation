from pycocoevalcap.eval import calculate_metrics
import numpy as np
import json
import argparse


def create_dataset(array):
    dataset = {'annotations': []}

    for i, caption in enumerate(array):
        dataset['annotations'].append({
            'image_id': i,
            'caption': caption
        })
    return dataset


def load_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--result_path', type=str,
                        default='../ckpt1/result.json')

    parser.add_argument('--gt_path', type=str,
                        default='./test_set_cap.json')
    
    args = parser.parse_args()

    test = load_json(args.result_path)
    gt = load_json(args.gt_path)

    datasetGTS = {'annotations': []}
    datasetRES = {'annotations': []}

    i = 0
    for image_id, image_id_gt in zip(test,gt):
        array = []
        for each in test[image_id]:
            array.append(test[image_id][each])
        pred_sent = '. '.join(array)
    
        array = []
        array.append(gt[image_id_gt] if not type(gt[image_id_gt])==float else '.')
        real_sent = '. '.join(array)



        datasetGTS['annotations'].append({
            'image_id': i,
            'caption': real_sent
        })
        datasetRES['annotations'].append({
            'image_id': i,
            'caption': pred_sent
        })
        i+=1

    rng = range(len(test))
    print(calculate_metrics(rng, datasetGTS, datasetRES))
