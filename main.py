from style_transfer_cycle import Style_transfer
from style_transfer_fader import Style_transfer_fader
from style_transfer_proposed import Style_transfer_proposed
from autoencoder import AutoEncoder
import yaml
from argparse import ArgumentParser
import sys

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-config', '-c', default='config/config.yaml')
    parser.add_argument('-model', '-m', default='style_proposed', 
                        choices=['style_cycle','autoencoder',
                                 'style_fader','style_proposed'])
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--alpha', dest='alpha', action='store', type=float) 
    parser.add_argument('--beta', dest='beta', action='store', type=float) 
    parser.add_argument('--gamma', dest='gamma', action='store', type=float) 
    parser.add_argument('--delta', dest='delta', action='store', type=float) 
    parser.add_argument('--zeta', dest='zeta', action='store', type=float) 

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    if args.load_model:
        if args.model=='style_cycle':
            model = Style_transfer(config, load_model=True)
        elif args.model=='style_fader':
            model = Style_transfer_fader(config, load_model=True)
        elif args.model=='autoencoder':
            model = AutoEncoder(config, load_model=True)
        elif args.model=='style_proposed':
            model = Style_transfer_proposed(config, args.alpha, args.beta, 
                                            args.gamma, args.delta, args.zeta, 
                                            load_model=True)
    else:
        if args.model=='style_cycle':
            model = Style_transfer(config, load_model=False)
        elif args.model=='style_fader':
            model = Style_transfer_fader(config, load_model=False)
        elif args.model=='autoencoder':
            model = AutoEncoder(config, load_model=False)
        elif args.model=='style_proposed':
            model = Style_transfer_proposed(config, args.alpha, args.beta, 
                                            args.gamma, args.delta, args.zeta, 
                                            load_model=False)
    
    if args.test:
        if args.train:
            state_dict, wer = model.train()
            model.test(state_dict)
        else:
            model.test()

