from tqdm import tqdm
import numpy as np
import tensorflow as tf

from siamesenet.models import SiameseNetwork
from siamesenet.data import load


def eval(config):
    
    np.random.seed(2020)
    tf.random.set_seed(2020)

    data_dir = config['data.dataset_path']
    res = load(data_dir, config, ['test'])

    test_loader = res['test']

    # Determine device
    if config['data.cuda']:
        cuda_num = config['data.gpu']
        device_name = f'GPU:{cuda_num}'
    else:
        device_name = 'CPU:0'

    # Setup training operations
    w, h, c = list(map(int, config['model.x_dim'].split(',')))
    n_way = config['data.test_way']
    model = Siamesenetwork(w, h, c, n_way)
    model.load(config['model.save_dir'])

    # Metrics to gather
    test_loss = tf.metrics.Mean(name='test_loss')
    test_acc = tf.metrics.Mean(name='test_accuracy')


    def calc_loss(support, query, labels):
        loss, acc = model(support, query, labels)
        return loss, acc
    
    with tf.device(device_name):
        for i_episode in tqdm(range(config['data.episodes'])):
            support, query, labels = test_loader.get_next_episode()
            loss, acc = calc_loss(support, query, labels)
            test_loss(loss)
            test_acc(acc)
    print("Loss: ", test_loss.result().numpy())
    print("Accuracy: ", test_accuracy.result().numpy())


if __name__ == "__main__":
    eval_config = {
        'data.dataset_path': 'data/omniglot',
        'data.dataset': 'omniglot',
        'data.test_way': 20,
        'data.split': 'vinyals',
        'data.batch': 1,
        'data.episodes': 100,
        'data.cuda': 1,
        'data.gpu': 0,

        'train.lr': 0.001,
        'train.patience': 100,

        'model.x_dim': '105,105,1',
        'model.save_dir': 'results/models/test'
    }
    eval(eval_config)
