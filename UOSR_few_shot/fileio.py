# from progress_bar import print_progress_bar
from torchvision.io import read_image

import cv2 as cv
import os, pickle, torch
import pandas as pd
import torchvision

NUM_CLASSES = 200
IMGS_PER_CLASS = 500

"""
Suggested workflow for preparing data:
get_train_data() # construct the training set
pickle_data()    # pickle the dataset for fast reading
get_val_data()   # construct the validation set
pickle_data()    # pickle the dataset for fast reading
"""

def get_label_mapping():
    object_mapping = pd.read_csv('/mnt/data-nas/data/tiny-imagenet-200/tiny-imagenet-200/words.txt', sep='\t', index_col=0, names=['label'])
    labels_str = [f.name for f in os.scandir('train') if f.is_dir()]
    labels = pd.DataFrame(labels_str, columns=['id'])
    labels['label'] = [object_mapping.loc[ids].item() for ids in labels['id']]

    return labels

def get_train_data(one_hot=False):
    train_data = torch.Tensor().type(torch.ByteTensor)
    labels_str = [f.name for f in os.scandir('/mnt/data-nas/data/tiny-imagenet-200/tiny-imagenet-200/train') if f.is_dir()]
    labels = []
    i = 1
    for root, dirs, files in os.walk('/mnt/data-nas/data/tiny-imagenet-200/tiny-imagenet-200/train'):
        if root.find('images') != -1:
            one_class = torch.Tensor().type(torch.ByteTensor)
            for name in files:
                img = read_image(root + os.sep + name)
                if img.shape[0] == 1:
                    img = torch.tensor(cv.cvtColor(img.permute(1,2,0).numpy(), cv.COLOR_GRAY2RGB)).permute(2,0,1)
                one_class = torch.cat((one_class, img), 0)
                labels.append(i-1)
                first_image = False
        
            one_class = one_class.reshape(-1, 3, 32, 32)
            print(one_class.shape)
            # print_progress_bar(i, NUM_CLASSES, prefix = 'Progress:', suffix = 'Complete')
            print(i, NUM_CLASSES)
            i+=1
            train_data = torch.cat((train_data, one_class), 0)

    return train_data, torch.Tensor(labels)

def get_val_data(one_hot=False):
    val_data = torch.Tensor().type(torch.ByteTensor)
    labels_str = [f.name for f in os.scandir('/mnt/data-nas/data/tiny-imagenet-200/tiny-imagenet-200/train') if f.is_dir()]
    labels = []
    val_annotations = pd.read_csv('/mnt/data-nas/data/tiny-imagenet-200/tiny-imagenet-200/val/val_annotations.txt', sep='\t', names=['filename', 'label_str', 'x_min', 'y_min', 'x_max', 'y_max'])
    num_imgs = len(os.listdir('/mnt/data-nas/data/tiny-imagenet-200/tiny-imagenet-200/val/images'))
    
    i = 1
    for name in os.listdir('/mnt/data-nas/data/tiny-imagenet-200/tiny-imagenet-200/val/images'):
        img = read_image('/mnt/data-nas/data/tiny-imagenet-200/tiny-imagenet-200/val/images' + os.sep + name)
        print(i, img.shape)
        if img.shape[0] == 1:
            img = torch.tensor(cv.cvtColor(img.permute(1,2,0).numpy(), cv.COLOR_GRAY2RGB)).permute(2,0,1)
        img = img.unsqueeze(0)
        val_data = torch.cat((val_data, img), 0)
        class_name = val_annotations.loc[val_annotations['filename'] == name]['label_str'].item()
        labels.append(labels_str.index(class_name))
        # print_progress_bar(i, num_imgs, prefix = 'Progress:', suffix = 'Complete')
        i+=1
        rs = torchvision.transforms.Resize(32)
        print(val_data.shape, rs(val_data).shape)

    return rs(val_data), torch.Tensor(labels)

def pickle_data(data, label, filename):
    outfile = open(filename, 'wb')
    pickle.dump((data, label), outfile)
    outfile.close()

if __name__ == '__main__':
    # data, labels = get_train_data()
    # print(data.shape, labels.shape)
    # pickle_data(data, labels, '/mnt/data-nas/data/tiny-imagenet-200/tiny-imagenet-200/train/train_dataset_32.pkl')
    data, labels = get_val_data()
    print(data.shape, labels.shape)
    pickle_data(data, labels, '/mnt/data-nas/data/tiny-imagenet-200/tiny-imagenet-200/val/val_dataset_32.pkl')