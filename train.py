import argparse
import ast
import datetime

from Processor import *
from models import *

def get_parser():

    parser = argparse.ArgumentParser(
        description='Dual-mode LSTM')
    parser.add_argument(
        '--base_dir',default='.',
        help='Base directory including these scrits.')
    parser.add_argument(
        '--save_base_dir',default='./savedata/',
        help='Directory for saving caches and models.')
    parser.add_argument(
        '--first_train_model', default='LSTM', 
        help='Your model name')
    parser.add_argument(
        '--second_train_model', default='LSTMresidualgated',  
        help='Your model name')
    parser.add_argument(
        '--phase',default='train',type=str)

    
    parser.add_argument(
        '--loss', default='weight')  
    parser.add_argument(
        '--lossweight', default=9)  


    ######################################

    parser.add_argument(
        '--dataset',default='Sample') 
    parser.add_argument(
        '--seed', default=5, type=int)
    parser.add_argument(
        '--save_dir')
    parser.add_argument(
        '--model_dir')


    # Model parameters

    # Perprocess
    parser.add_argument(
        '--batch_size', default=16, type=int)
    parser.add_argument(
        '--num_epochs', default=200, type=int)
    parser.add_argument(
        '--learning_rate', default=2e-3, type=float)

    #LSTM

    parser.add_argument(
        '--seq_length', default=10, type=int)
    parser.add_argument(
        '--input_size', default=20, type=int)
    parser.add_argument(
        '--output_size',default=1,type=int)
    parser.add_argument(
        '--n_layers', default=1, type=int)
    parser.add_argument(
        '--rnn_size',default=16,type=int)
    parser.add_argument(
        '--std_in',default=0.2,type=float)
    parser.add_argument(
        '--std_out',default=0.1,type=float)

    return parser

def load_arg(p):
    # save arg
    if  os.path.exists(p.config):
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                try:
                    assert (k in key)
                except:
                    s=1
        parser.set_defaults(**default_arg)
        return parser.parse_args()
    else:
        return False

def save_arg(args):
    # save arg
    arg_dict = vars(args)
    with open(args.config, 'w') as f:
        yaml.dump(arg_dict, f)

if __name__ == '__main__':

    parser = get_parser()
    p = parser.parse_args()

    p.save_dir = os.path.join(p.save_base_dir, str(p.dataset))
    p.model_dir = os.path.join(p.save_base_dir, str(p.dataset), p.first_train_model+p.second_train_model)
    if not os.path.exists(p.model_dir):
        os.makedirs(p.model_dir)

    save_arg(p)
    args = load_arg(p)
    print(args.program, args.dataset, ' args ', args)

    processor = Processor(args)
