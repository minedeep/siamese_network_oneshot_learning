import argparse
import configparser

from eval_setup import eval

def preprocess_config(c):
    conf_dic = {}

    int_params = ['data.train_way', 
                  'data.test_way',
                  'data.batch_size',
                  'data.episodes',
                  'data.cuda',
                  'data.gpu',
                  'data.gpu',
                  'train.epochs',
                  'train.patience']
    float_params = ['train.lr']

    for param in c:
        if param in int_params:
            config_dict[param] = int(c[param])
        elif param in float_params:
            config_dict[param] = float(c[param])
        else:
            config_dict[param] = c[param]
    return conf_dict

parser = argparse.ArgumentParser(description='Run training')
parser.add_argument("--config", type=str, default="./training/omniglot.conf"),
                     help = "Path to the config file.")

# Run training
args = vars(parser.parse_args())
config = configparser.ConfigParser()
config.read(args['config'])
config = preprocess_config(config['EVAL'])
eval(config)
    

