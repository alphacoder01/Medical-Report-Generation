import os
import pandas as pd
from tqdm.auto import tqdm


TRAIN_PATH = '../../../scratch/datasets/IUX_DATA/Train'
VAL_PATH = '../../../scratch/datasets/IUX_DATA/Val'

train_report = pd.read_csv(os.path.join(TRAIN_PATH, 'report.csv'))
val_report = pd.read_csv(os.path.join(VAL_PATH, 'report.csv'))

train_proj = pd.read_csv(os.path.join(TRAIN_PATH, 'projections.csv'))
val_proj = pd.read_csv(os.path.join(VAL_PATH, 'projections.csv'))


train_report.fillna('', inplace=True)
val_report.fillna('', inplace=True)

train = pd.merge(train_proj,train_report,on='0')
val = pd.merge(val_proj,val_report,on='0')


with open('../data/new_data/train_data.txt','r') as f:
    trainlst = f.readlines()

with open('../data/new_data/test_data.txt','r') as f:
    testlst = f.readlines()

with open('../data/new_data/val_data.txt','r') as f:
    vallst = f.readlines()



def search_lst(lst,query):
    for i in range(len(lst)):
        if query in lst[i].split(' ')[0]:
            return lst[i]
    return -1



def main_func(df):
    imgfound = []
    notFound = []
    for i in tqdm(range(len(df))):
        query = df['1_x'].iloc[i].replace('.dcm.png','').split('_')[1]
        inTrain = search_lst(trainlst,query)
        inTest = search_lst(testlst, query)
        inVal = search_lst(vallst, query)
        # print(inTrain, inTest, inVal)
        if inTrain == -1 and inTest==-1 and inVal==-1:
            # print(inTrain, inTest, inVal)
            imgfound.append(-1)
            # print(train['1_x'].iloc[i])
            notFound.append(query)
        else:
            if inTrain != -1:
                name = inTrain.split(' ')[0]
                imgfound.append(inTrain.replace(name, df['1_x'].iloc[i].replace('.dcm.png','')))
            elif inTest != -1:
                name = inTest.split(' ')[0]
                imgfound.append(inTest.replace(name, df['1_x'].iloc[i].replace('.dcm.png','')))
            else:
                name = inVal.split(' ')[0]
                imgfound.append(inVal.replace(name, df['1_x'].iloc[i].replace('.dcm.png','')))
    return imgfound


train_subset = main_func(train)
val_subset = main_func(val)


with open('../data/new_data/subset_train_data.txt','w') as f:
    f.writelines(train_subset)

with open('../data/new_data/subset_val_data.txt','w') as f:
    f.writelines(val_subset)