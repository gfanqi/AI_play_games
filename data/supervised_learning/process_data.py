import pandas as pd
import os
import cv2

index = 54549
csv_root = './{}/label'.format(index)
csv_paths = [os.path.join(csv_root, item) for item in os.listdir(csv_root)]
keys = pd.read_csv('../keys', names=['keys'])
keys = keys['keys'].tolist()
data = [pd.read_csv(csv_path, index_col=[0]) for csv_path in csv_paths]
data = pd.concat(data, axis=0, )
data = data.sort_values(by='time')
data = data.reset_index(drop=True)
# print(data)


# data.to_csv('./{}/label.csv'.format(index))

# data = pd.read_csv('./supervised_learning/72255/label.csv', index_col=[0])
data['keys'] = data['keys'].map(lambda x: eval(x))



for key in set(keys):
    data[key] = data['keys'].map(lambda x: int(key in x))

data['path'] = data['image_name'].map(
    lambda x: os.path.abspath(os.path.join('../supervised_learning/{}/image'.format(index), x)))

data.to_csv('modifyed_label.csv',)
path_list = data.path.tolist()

for index, path in enumerate(path_list):
    if index < 66: continue
    img = cv2.imread(path)
    cv2.imshow('windows', img)
    cv2.waitKey(0)
    print('\r{}'.format(index))
