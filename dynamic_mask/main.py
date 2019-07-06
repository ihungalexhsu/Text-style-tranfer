import yaml
from argparse import ArgumentParser
from train_dynamic_mask import Dynamic_mask
import sys

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-config', '-c', default='config/config.yaml')
    parser.add_argument('-model', '-m', default='dynamic', 
                        choices=['dynamic'])
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--beta', dest='beta', action='store', type=str)
    parser.add_argument('--gamma', dest='gamma', action='store', type=str)

    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.load(f)

    if args.load_model:
        if args.model=='dynamic':
            model = Dynamic_mask(config, beta=args.beta, gamma=args.gamma, load_model=True)
    else:
        if args.model=='dynamic':
            model = Dynamic_mask(config, beta=args.beta, gamma=args.gamma, load_model=False)
    state_dict = None
    if args.train:
        state_dict, score = model.train()
    if args.test:
        model.test(state_dict)

