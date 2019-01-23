from seq2seq import Seq2seq
from autoencoder import AutoEncoder
import yaml
from argparse import ArgumentParser
import sys

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-config', '-c', default='config/config.yaml')
    parser.add_argument('-model', '-m', default='seq2seq', 
                        choices=['seq2seq','autoencoder'])
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    if args.load_model:
        if args.model=='seq2seq':
            model = Seq2seq(config, load_model=True)
        elif args.model=='autoencoder':
            model = AutoEncoder(config, load_model=True)
    else:
        if args.model=='seq2seq':
            model = Seq2seq(config, load_model=False)
        elif args.model=='autoencoder':
            model = AutoEncoder(config, load_model=False)
    
    if args.test:
        if args.train:
            state_dict, wer = model.train()
            model.test(state_dict)
        else:
            model.test()

