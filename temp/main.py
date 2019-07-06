import yaml
from argparse import ArgumentParser
from train_base import Base_model
from train_proposed import Proposed_model
from train_base_dynamic import Base_model_dynamic
#from train_base_dynamic_withdis import Base_dynamic_with_dis
from train_cycle import Cycle_model
import sys

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-config', '-c', default='config/config.yaml')
    parser.add_argument('-model', '-m', default='base', 
                        choices=['base', 'proposed', 'base_dynamic', 'cycle'])
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--beta', dest='beta', action='store', type=str)
    parser.add_argument('--gamma', dest='gamma', action='store', type=str)

    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.load(f)

    if args.load_model:
        if args.model=='base':
            model = Base_model(config, load_model=True)
        elif args.model=='proposed':
            model = Proposed_model(config, load_model=True)
        elif args.model=='base_dynamic':
            model = Base_model_dynamic(config, beta=args.beta, gamma=args.gamma, load_model=True)
            #model = Base_dynamic_with_dis(config, beta=args.beta, gamma=args.gamma, load_model=True)
        elif args.model=='cycle':
            model = Cycle_model(config, load_model=True)
    else:
        if args.model=='base':
            model = Base_model(config, load_model=False)
        elif args.model=='proposed':
            model = Proposed_model(config, load_model=False)
        elif args.model=='base_dynamic':
            model = Base_model_dynamic(config, beta=args.beta, gamma=args.gamma, load_model=False)
            #model = Base_dynamic_with_dis(config, beta=args.beta, gamma=args.gamma, load_model=False)
        elif args.model=='cycle':
            model = Cycle_model(config, load_model=False)
    state_dict = None
    if args.train:
        state_dict, wer = model.train()
    if args.test:
        model.test(state_dict)

