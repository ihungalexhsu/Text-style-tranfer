from pretrain_selfatt_classifier import Pretrain_selfatt_classifier
import yaml
from argparse import ArgumentParser
import sys

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-config', '-c', default='config/config.yaml')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test_ontxt', action='store_true')
    parser.add_argument('--testfilepath', dest='testfilepath', action='store', type=str)
    parser.add_argument('--testlabel', dest='testlabel', action='store', type=int,
                        default=1, help='the label for your input test file, 1 or 0')
    parser.add_argument('--write_testfile', dest='write_testfile', action='store', 
                        type=bool, default=False)

    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.load(f)

    if args.load_model:
        model = Pretrain_selfatt_classifier(config, load_model=True)
    else:
        model = Pretrain_selfatt_classifier(config, load_model=False)
    state_dict = None
    if args.train:
        state_dict, wer = model.train()
    if args.test:    
        model.test(state_dict)
    if args.test_ontxt:
        assert args.testfilepath is not None
        model.test_ontxt(args.testfilepath, args.testlabel, args.write_testfile, state_dict)
        

