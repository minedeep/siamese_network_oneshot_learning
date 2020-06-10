import os
import glob
import numpy as np
import tensorflow as tf
from PIL import Image

from .image_augmentor import ImageAugmentor

WIDTH=105
HEIGHT=105


class DataLoader(object):
    """
    omniglot dataset has multiple alphabets, each alphabet has multiple characters
    each character has n_examples = 20 examples (images)
    Data Loader to split the datasets into train/val/test and building training batches out 
    of the training set depending on the n_way parameter
    n_way = 2 if we want to compare a pair of images
    """
    def __init__(self, data, batch_size, n_classes, n_way):
        self.data=data
        self.batch_size = batch_size
        self.n_way=n_way
        self.n_classes=n_classes

    def get_next_episode(self):
        
        # each character set has n_examples = 20 images
        n_examples = 20

        # self.n_wway ** 2: the number of pairs we need to compare from n_way classes
        # WIDTH, HEIGHT, 1: the size of each image (channel = 1)
        support_batches = np.zeros([self.batch_size, self.n_way**2, WIDTH, HEIGHT, 1], dtype=np.float32) 
        query_batches = np.zeros([self.batch_size, self.n_way**2, WIDTH, HEIGHT, 1], dtype=np.float32) 
        labels_batches = np.zeros([self.batch_size, self.n_way**2])
        classes_to_compare = np.random.permutation(self.n_classes)[:self.n_way]

        for i_batch in range(self.batch_size):
          
            support = np.zeros([self.n_way, WIDTH, HEIGHT, 1])
            query = np.zeros([self.n_way, WIDTH, HEIGHT, 1])

            # go through each class to compare for this episode
            # for each class, get 2 random examples
            for i, i_class in enumerate(classes_to_compare):
                # get 2 random examples from each class
                # each character class has n_examples
                selected = np.random.permutation(n_examples)[:2]
                support[i] = self.data[i_class, selected[0]]
                query[i] = self.data[i_class, selected[1]]

            # 
            support_batch = np.take(support, [i//self.n_way for i in range(self.n_way**2)], axis=0)
            query_batch = tf.tile(query, [self.n_wqy, 1, 1, 1])
            
            support_batches[i_batch] = support_batch
            query_batches[i_batch] = query_batch

        support_batches = np.vstack(support_batches)
        query_batches = np.vstack(query_batches)
        labels_batches[i_batch] = [i==j for i in range(self.n_way) for j in range(self.n_way)]

        return support_batches, query_batches, labels_batches


def data_to_dic(data_dir, splits):
    """
    Load all alphabets into dictionaries
    """

    res = {}
    if 'train' in splits:
        train_path = os.path.join(data_dir, 'images_background')

        train_dic = {}
        for alphabet in os.listdir(train_path):
            alphabet_path = os.path.join(train_path, alphabet)
            cur_alphabet_dic = {}
            for char in os.listdir(alphabet_path):
                char_path = os.path.join(alphabet_path, char)
                cur_alphabet_dic[char] = os.listdir(char_path)
            train_dic[alphabet] = cur_alphabet_dic
        res['train'] = train_dic

    if 'test' in splits:
        test_path = os.path.join(data_dir, 'images_evaluation')

        test_dic = {}
        for alphabet in os.listdir(test_path):
            alphabet_path = os.path.join(test_path, alphabet)
            cur_alphabet_dic = {}
            for char in os.listdir(alphabet_path):
                char_path = os.path.join(alphabet_path, char)
                cur_alphabet_dic[char] = os.listdir(char_path)
            test_dic[alphabet] = cur_alphabet_dic
        res['test'] = test_dic
    return res

def load_and_preprocess_image(img_path, image_augmentor):

    """
    img_path: path to the image
    use_augmentation: T/F: perform random image augmentation if T
    """
    img = Image.open(img_path).resize((WIDTH, HEIGHT))
    if image_augmentor:
        img = image_augmentor.perform_random_rotation()
    img = np.asarray(img).astype(np.float64)
    img /= 255
    return np.expand_dims(img, -1)


def load_omniglot(data_dir, config, splits):
    """
    splits = ['train', 'val', 'test']
    config: user-defined system parameters dic
    Split == 'train': split the data in the 'images_background' into train set in train and validation
    divide 30 train alphabets in train and validation 
    80% - 20%

    create the DataLoader for each split
    """
    
    # get all available alphabet in image_background
    data_dic = data_to_dic(data_dir, splits)
    
    # if for training, need to split the data into training and validation sets
    # train_alphabets, val_alphabets = [], []

    if 'train' in splits:
        train_dic = data_dic['train']

        all_alphabets = list(train_dic.keys())
        n_alphabets = len(all_alphabets)

        train_indexes = np.random.permutation(range(n_alphabets))[:0.8*n_alphabets]

        # sort the indexes in reverse order so we can pop the alphabet from the all alphabet list
        # the rest of the all_alphabet will be for validation
        train_indexes.sort(reverse=True)
        train_alphabets = []
        for index in train_indexes:
            train_alphabets.append(all_alphabets[index])
            all_alphabets.pop(index)
        val_alphabets = all_alphabets

    if config['use_augmentation']:
        image_augmentor = create_augmentor()
    else:
        image_augmentor = None
    res = {}
    for split in splits:
        if split in ['train', 'val']:
            split_dic = data_dic['train']
            if split == 'train':
                n_way = config['data.train_way'] 
                split_alphabets = train_alphabets 
            else:
                n_way = config['data.val_way'] 
                split_alphabets = val_alphabets 

        elif split in ['test']:
            split_dic = data_dic['test']
            n_way = config['data.test_way'] 
            split_alphabets = list(data_dic[split].keys())

        n_images_per_char = len(split_dic[split_alphabets[0]][list(split_dic[split_alphabets[0]].keys())[0]])
        data = np.zeros([len(split_alphabets) * len(split_dic[split_alphabets[0]]), 
                          n_images_per_char,
                           WIDTH, HEIGHT, 1])
        for i_alphabet, alphabet in enumerate(split_alphabets):
            for char in list(split_dic[alphabet].keys()): 
                for i_img, img_path in enumerate(split_dic[alphabet][char]):
                    data[i_class, i_img, :, :, :] = load_and_preprocess_image(img_path, image_augmentor)

        data_loader = DataLoader(data, 
                         batch_size=config['batch_size'],
                         n_classes = data.shape[0],
                         n_way = n_way)

        res[split] = data_loader

    return res

def create_augmentor(self):

    """
    Create Image Augmentor object for image augmentation
    rotation: -15 to 15 degress
    shear range : -0.3 to 0.3 radians
    zoom : 0.8 to 2
    shift range: +/- 5 pixels
    """

    rotation = [-15, 15]
    #shear_range = [-0.3*100/math.pi, 0.3*100/math.pi]
    #zoom_range=[0.8,2]
    #shift_range=[5,5]

    return ImageAugmentor(0.5, rotation_range)



     
    




