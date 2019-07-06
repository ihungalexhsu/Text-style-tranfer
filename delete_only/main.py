import yaml
from argparse import ArgumentParser
from train import Delete_only
import sys

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-config', '-c', default='config/config.yaml')
    parser.add_argument('-model', '-m', default='delete', 
                        choices=['delete'])
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.load(f)

    if args.load_model:
        if args.model=='delete':
            model = Delete_only(config, load_model=True)
    else:
        if args.model=='delete':
            model = Delete_only(config, load_model=False)
    state_dict = None
    if args.train:
        state_dict, score = model.train()
    if args.test:
        model.test(state_dict)

