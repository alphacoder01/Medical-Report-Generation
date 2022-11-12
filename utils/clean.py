import os.path as osp
import os
import csv
import numpy as np
import json

data_dir = '/home/sweta/scratch/datasets/IUX_DATA/'
caption_out_file= '/home/sweta/adv_cv_project/Medical-Report-Generation/data/new_data/subset_captions_full.json'

def get_sorted(img_path):
        img_fnames = np.array(os.listdir(img_path))
        img_fnames_tmp = np.array([fname.split("_")[0]  for fname in img_fnames], dtype= np.int32)
        img_idx = np.argsort(img_fnames_tmp)
        img_fnames = img_fnames[img_idx]
        return img_fnames

def convert_caption(data_dir):
    img_path0= osp.join(data_dir, "Frontal")
    img_path1=osp.join(data_dir, "Lateral")
    img_fnames0 = get_sorted(img_path0)
    img_fnames1 = get_sorted(img_path1)

    report_path = osp.join(data_dir, "report.csv")
    captions = {}

    with open(report_path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for img_fname0, img_fname1, line in zip(img_fnames0, img_fnames1, reader):
            print(line[1], img_fname0.split("_")[0], img_fname1.split("_")[0], img_fname0, img_fname1)
            # assert line[1]==img_fname.split("_")[0]
            single_caption = ".".join([line[-1], line[-2]])
            captions[img_fname0]= single_caption
            captions[img_fname1]= single_caption

    return captions
    

cap1 = convert_caption(data_dir+'Train')
cap2 = convert_caption(data_dir+'Val')

captions = {**cap1, **cap2}


with open(caption_out_file, 'w') as f:
    json.dump(captions, f)
print(captions)