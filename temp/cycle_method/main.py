import yaml
from argparse import ArgumentParser
from train_base_cycle import Base_cycle
import sys

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-config', '-c', default='config/config.yaml')
    parser.add_argument('-model', '-m', default='cycle', 
                        choices=['cycle'])
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.load(f)

    if args.load_model:
        if args.model=='cycle':
            model = Base_cycle(config, load_model=True)
    else:
        if args.model=='cycle':
            model = Base_cycle(config, load_model=False)
    state_dict = None
    if args.train:
        state_dict, wer = model.train()
    if args.test:
        model.test(state_dict)

