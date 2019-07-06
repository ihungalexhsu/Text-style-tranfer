import torch 
import torch.nn as nn
import numpy as np
from model import RCNN
from dataloader import get_data_loader
from dataset import PickleDataset
from utils import *
import copy
import yaml
import os
import pickle

class rcnn_classifier(object):
    def __init__(self, config, load_model=False):
        self.config = config
        print(self.config)
        
        # logger
        self.logger = Logger(config['logdir'])

        # load vocab and non lang syms
        self.load_vocab()
       
        # get data loader
        self.get_data_loaders()
        
        # build model and optimizer
        self.build_model(load_model=load_model)
    
        self.train_target = self.config['train_target']
        self.test_target = self.config['test_target']

    def load_vocab(self):
        with open(self.config['vocab_path'], 'rb') as f:
            self.vocab = pickle.load(f) # a dict; word to index
        with open(self.config['non_lang_syms_path'], 'rb') as f:
            self.non_lang_syms = pickle.load(f) # an array
        return
    
    def get_data_loaders(self):
        root_dir = self.config['dataset_root_dir']       
        # get train dataset
        train_pos_set = os.path.join(root_dir, f'{self.config["train_pos"]}.p')
        train_neg_set = os.path.join(root_dir, f'{self.config["train_neg"]}.p')
        self.train_dataset = PickleDataset(train_pos_set,
                                           train_neg_set,
                                           config=self.config,
                                           sort=self.config['sort_dataset'])
        print("train data size: ", len(self.train_dataset))
        self.train_loader = get_data_loader(self.train_dataset,
                                            batch_size=self.config['batch_size'],
                                            shuffle=True)
        # get dev dataset
        dev_pos_set = os.path.join(root_dir, f'{self.config["dev_pos"]}.p')
        dev_neg_set = os.path.join(root_dir, f'{self.config["dev_neg"]}.p')
        self.dev_dataset = PickleDataset(dev_pos_set,
                                         dev_neg_set,
                                         sort=True)
        print("development data size: ", len(self.dev_dataset))
        self.dev_loader = get_data_loader(self.dev_dataset, 
                                          batch_size=self.config['batch_size'], 
                                          shuffle=False)
        return

    def build_model(self, load_model=False):
        pretrain_w2v_path = self.config['pretrain_w2v_path']
        print("original vocabulary size:", len(self.vocab))
        pretrain_w2v, self.vocab = loadw2vweight(pretrain_w2v_path, self.vocab,
                                                 self.vocab['<PAD>'], '<MASK>')
        print("vocabulary size:", len(self.vocab))

        self.classifier=cc_model(RCNN(vocab_size=len(self.vocab),
                                      embedding_dim=self.config['embedding_dim'],
                                      pre_embedding=pretrain_w2v,
                                      rnn_hidden_dim=self.config['rnn_hidden_dim'],
                                      linear_dim=self.config['linear_dim'],
                                      output_dim=2,
                                      pad_idx=self.vocab['<PAD>']))
        print(self.classifier)
        self.classifier.float()
        self.optimizer =\
            torch.optim.Adam(list(self.classifier.parameters()), 
                             lr=self.config['learning_rate'],
                             weight_decay=float(self.config['weight_decay']))
        if load_model:
            self.load_model(self.config['load_model_path'], self.config['load_optimizer'])
        return

    def load_model(self, model_path, load_optimizer):
        print(f'Load model from {model_path}')
        self.classifier.load_state_dict(torch.load(f'{model_path}.ckpt'))
        if load_optimizer:
            print(f'Load optmizer from {model_path}')
            self.optimizer.load_state_dict(torch.load(f'{model_path}.opt'))
            if self.config['adjust_lr']:
                adjust_learning_rate(self.optimizer, self.config['retrieve_lr']) 
        return

    def save_model(self, model_path):
        torch.save(self.classifier.state_dict(), f'{model_path}.ckpt')
        torch.save(self.optimizer.state_dict(), f'{model_path}.opt')
        return
    
    def train(self):
        best_acc = 0.
        best_model = None
        early_stop_counter = 0
        print('------start training-------')
        for epoch in range(self.config['epochs']):
            # train one epoch
            train_loss, train_acc = self.train_one_epoch(epoch)
            
            # save model in every epoch
            if not os.path.exists(self.config['model_dir']):
                os.makedirs(self.config['model_dir'])
            model_path = os.path.join(self.config['model_dir'], self.config['model_name'])
            self.save_model(f'{model_path}-{epoch:02d}')
            self.save_model(f'{model_path}_latest')
            
            # validation
            val_loss, val_acc = self.validation()
            print(f'epoch:{epoch}, train_loss:{train_loss:.4f}, train_acc:{train_acc:.4f} ,'
                  f'val_loss:{val_loss:.4f}, val_acc:{val_acc:.4f}')
            # add to tensorboard
            tag = self.config['tag']
            self.logger.scalar_summary(f'{tag}/val/val_loss', val_loss, epoch)
            self.logger.scalar_summary(f'{tag}/val/val_acc', val_acc, epoch)
            # save best
            if val_acc > best_acc: 
                model_path = os.path.join(self.config['model_dir'], self.config['model_name']+'_best')
                best_acc = copy.deepcopy(val_acc)
                self.save_model(model_path)
                best_model = copy.deepcopy(self.classifier.state_dict())
                print(f'Save #{epoch} model, val_loss={val_loss:.4f}, Accuracy={val_acc:.4f}')
                print('-----------------')
                early_stop_counter=0
            if epoch >= self.config['early_stop_start_epoch']:
                early_stop_counter += 1
                if early_stop_counter > self.config['early_stop_patience']:
                    break
        print('-------finish training--------')
        return best_model, best_acc
    
    def train_one_epoch(self, epoch):
        total_steps = len(self.train_loader)
        total_loss = 0.
        total_instance = 0.
        correct = 0.
        for train_steps, data in enumerate(self.train_loader):
            total_loss, correct, total_instance =\
                self._train_model_onetime(data, total_loss, epoch, 
                                          train_steps, total_steps,
                                          correct, total_instance)
        print()
        return (total_loss/total_steps), (correct/total_instance)
    
    def _train_model_onetime(self, data, total_loss, epoch, train_steps, 
                             total_steps, correct, total_instance):
        loss, correct, total_instance = self._gothrough_onetime(data, correct, total_instance)
        total_loss += loss.item()
        # calculate gradients 
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.classifier.parameters()), 
                                       max_norm=self.config['max_grad_norm'])
        self.optimizer.step()
        self._log_model(epoch, train_steps, total_steps, loss.item(),
                        epoch*total_steps+train_steps+1)
        return total_loss, correct, total_instance

    def _gothrough_onetime(self, data, correct, total_instance):
        pad = self.vocab['<PAD>']
        inputs, ilens, labels, aligns = to_gpu(data, pad)
        if self.train_target == 'remain':
            inputs=inputs.masked_fill_(aligns, self.vocab['<MASK>'])
        elif self.train_target == 'mask':
            reverse_aligns = torch.ones(aligns.size())
            reverse_aligns[aligns] = 0
            reverse_aligns = cc(torch.ByteTensor(reverse_aligns.byte()))
            inputs=inputs.masked_fill_(reverse_aligns, self.vocab['<MASK>'])
        log_probs, prediction = self.classifier(inputs, ilens)
        true_log_probs = torch.gather(log_probs, dim=1, index=labels.unsqueeze(1)).squeeze(1)
        loss = -torch.mean(true_log_probs)
        total_instance += len(prediction.view(-1))
        correct += (labels.eq(prediction.view(-1).long()).sum()).item()
        return loss, correct, total_instance

    def _log_model(self, epoch, train_steps, total_steps, loss, log_steps): 
        print(f'epoch: {epoch}, [{train_steps+1}/{total_steps}], loss: {loss:.3f}', end='\r')
        
        tag = self.config['tag']
        self.logger.scalar_summary(tag=f'{tag}/train/loss', 
                                   value=loss,
                                   step=log_steps)
        return
    
    def _valid_feed_data(self, dataloader, total_loss, correct, total_instance):
        
        for data in dataloader:
            pad = self.vocab['<PAD>']
            inputs, ilens, labels, aligns = to_gpu(data, pad)
            if self.test_target == 'remain':
                inputs=inputs.masked_fill_(aligns, self.vocab['<MASK>'])
            elif self.test_target == 'mask':
                reverse_aligns = torch.ones(aligns.size())
                reverse_aligns[aligns] = 0
                reverse_aligns = cc(torch.ByteTensor(reverse_aligns.byte()))
                inputs=inputs.masked_fill_(reverse_aligns, self.vocab['<MASK>'])
            log_probs, prediction = self.classifier(inputs, ilens)
            true_log_probs = torch.gather(log_probs, dim=1, index=prediction.view(-1).unsqueeze(1)).squeeze(1)
            loss = -torch.mean(true_log_probs)
            total_instance += len(prediction.view(-1))
            correct += (labels.eq(prediction.view(-1).long()).sum()).item()
            total_loss += loss.item()

        return correct, total_instance, total_loss

    def validation(self):
        self.classifier.eval()
        
        total_loss = 0.
        correct = 0.
        total_instance = 0.
        
        correct, total_instance, total_loss =\
            self._valid_feed_data(self.dev_loader, total_loss, correct, total_instance)
        
        self.classifier.train()
        
        # evaluation
        avg_loss = (total_loss / len(self.dev_loader))
        avg_acc = correct/total_instance
        return avg_loss, avg_acc

    def test(self, state_dict=None):
        # load model
        if state_dict is None:
            self.load_model(self.config['load_model_path'], self.config['load_optimizer'])
        else:
            self.classifier.load_state_dict(state_dict)

        # get test dataset
        root_dir = self.config['dataset_root_dir']
        test_pos_set = os.path.join(root_dir, f'{self.config["test_pos"]}.p')
        test_neg_set = os.path.join(root_dir, f'{self.config["test_neg"]}.p')
        test_dataset = PickleDataset(test_pos_set,
                                     test_neg_set,
                                     sort=True)
        print("test data size: ", len(test_dataset))
        test_loader = get_data_loader(test_dataset,
                                      batch_size=self.config['batch_size'], 
                                      shuffle=False)

        self.classifier.eval()
        total_loss = 0.
        correct = 0.
        total_instance = 0.
        correct, total_instance, total_loss =\
            self._valid_feed_data(test_loader, total_loss, correct, total_instance)
        self.classifier.train()
        
        # evaluation
        avg_acc = correct/total_instance
        print(f"------finish testing------, Testing Accuracy: {avg_acc:.4f}")
        tag = self.config['tag']
        self.logger.scalar_summary(f'{tag}/test/test_acc', avg_acc, 0)
        return avg_acc
