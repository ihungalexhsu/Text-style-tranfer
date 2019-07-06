from run import rcnn_classifier
import yaml
from argparse import ArgumentParser
import sys

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-config', '-c', default='config/config.yaml')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.load(f)

    if args.load_model:
        model = rcnn_classifier(config, load_model=True)
    else:
        model = rcnn_classifier(config, load_model=False)
    
    state_dict = None
    
    if args.train:
        state_dict, best_dev_acc = model.train()
    if args.test:    
        model.test(state_dict)

