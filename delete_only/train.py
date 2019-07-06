import sys
sys.path.append('..')
import torch 
import torch.nn.functional as F
import numpy as np
from delete_only.model import E2E
from delete_only.dataloader import get_data_loader
from delete_only.dataset import PickleDataset
from delete_only.utils import *
from evaluation.calculate_bleu import BLEU
from evaluation.calculate_transfer import Transferability 
from evaluation.calculate_fluency_bert import calculate_fluency 
import yaml
import os
import copy
import pickle

class Delete_only(object):
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
        dev_pos_set = self.config['dev_pos']
        dev_neg_set = self.config['dev_neg']
        self.dev_pos_dataset = PickleDataset(os.path.join(root_dir, f'{dev_pos_set}.p'), 
                                             sort=True)
        print("positive dev data size: ", len(self.dev_pos_dataset))
        self.dev_neg_dataset = PickleDataset(os.path.join(root_dir, f'{dev_neg_set}.p'), 
                                             sort=True)
        print("negative dev data size: ", len(self.dev_neg_dataset))
        self.dev_pos_loader = get_data_loader(self.dev_pos_dataset, 
                                              batch_size=self.config['batch_size'], 
                                              shuffle=False)
        self.dev_neg_loader = get_data_loader(self.dev_neg_dataset, 
                                              batch_size=self.config['batch_size'], 
                                              shuffle=False)
        return
    
    def build_model(self, load_model=False):
        pretrain_w2v_path = self.config['pretrain_w2v_path']
        pretrain_w2v, self.vocab = loadw2vweight(pretrain_w2v_path, self.vocab)
        self.model = cc_model(E2E(vocab_size=len(self.vocab),
                                  embedding_dim=self.config['embedding_dim'],
                                  enc_hidden_dim=self.config['enc_hidden_dim'],
                                  enc_layers=self.config['enc_n_layers'],
                                  enc_dropout=self.config['enc_dropout_p'],
                                  bidirectional=self.config['bidirectional'],
                                  dec_hidden_dim=self.config['dec_hidden_dim'],
                                  dec_dropout=self.config['dec_dropout_p'],
                                  bos=self.vocab['<BOS>'],
                                  eos=self.vocab['<EOS>'],
                                  pad=self.vocab['<PAD>'],
                                  n_styles=self.config['n_style_type'],
                                  style_emb_dim=self.config['style_emb_dim'],
                                  att_dim=self.config['att_dim'],
                                  pre_embedding=pretrain_w2v,
                                  update_embedding=self.config['update_embedding']))
        print(self.model)
        self.model.float()
        self.params=list(self.model.parameters())
        self.optimizer=\
            torch.optim.Adam(self.params, 
                             lr=self.config['learning_rate'],
                             weight_decay=float(self.config['weight_decay']))
        if load_model:
            self.load_model(self.config['load_model_path'], self.config['load_optimizer'])
        return
 
    def load_model(self, model_path, load_optimizer):
        if os.path.exists(model_path+'.ckpt'):
            print(f'Load model from {model_path}')
            self.model.load_state_dict(torch.load(f'{model_path}.ckpt'))
            if load_optimizer:
                print(f'Load optmizer from {model_path}')
                self.optimizer.load_state_dict(torch.load(f'{model_path}.opt'))
                if self.config['adjust_lr']:
                    adjust_learning_rate(self.optimizer, self.config['retrieve_lr']) 
        return
  
    def save_model(self, model_path):
        torch.save(self.model.state_dict(), f'{model_path}.ckpt')
        torch.save(self.optimizer.state_dict(), f'{model_path}.opt')
        return
    
    def train(self):
        best_score = 0.
        best_model = None
        early_stop_counter = 0
        print('------start training-------')
        for epoch in range(self.config['epochs']):
            # train one epoch
            avg_train_loss = self.train_one_epoch(epoch)
            # save model in every epoch
            if not os.path.exists(self.config['model_dir']):
                os.makedirs(self.config['model_dir'])
            model_path = os.path.join(self.config['model_dir'], self.config['model_name'])
            self.save_model(f'{model_path}-{epoch:02d}')
            self.save_model(f'{model_path}_latest')
            
            # validation
            val_loss, transfer_acc, selfbleu, fluency = self.validation(epoch)
            print(f'epoch: {epoch}, val_loss: {val_loss:.4f}, '
                  f'trans_acc: {transfer_acc:.4f}, self_bleu: {selfbleu:.4f}, '
                  f'fluency: {fluency:.4f}')
            # add to tensorboard
            tag = self.config['tag']
            self.logger.scalar_summary(f'{tag}/val/selfbleu', selfbleu, epoch)
            self.logger.scalar_summary(f'{tag}/val/fluency', fluency, epoch)
            self.logger.scalar_summary(f'{tag}/val/transfer_acc', transfer_acc, epoch)
            score = selfbleu*transfer_acc/fluency
            # save best
            if score > best_score: 
                # save model
                model_path = os.path.join(self.config['model_dir'], self.config['model_name']+'_best')
                best_score = score
                self.save_model(model_path)
                best_model = copy.deepcopy(self.model.state_dict())
                print(f'Save #{epoch} model, val_loss={val_loss:.4f}, score={score:.4f}')
                print('-----------------')
                early_stop_counter=0
            if epoch >= self.config['early_stop_start_epoch']:
                early_stop_counter += 1
                if early_stop_counter > self.config['early_stop_patience']:
                    break
        print('---------------finish training----------------')
        print(f'-----get best score: {best_score:.4f}------')
        return best_model, best_score
    
    def _get_reverse_style(self, styles):
        reverse_styles = styles.cpu().new_zeros(styles.size())
        for idx, ele in enumerate(styles.cpu().tolist()):
            if not(ele):
                reverse_styles[idx]=1
        reverse_styles=cc(torch.LongTensor(reverse_styles))
        return reverse_styles
    
    def _get_loss(self, data, training=True):
        bos = self.vocab['<BOS>']
        eos = self.vocab['<EOS>']
        pad = self.vocab['<PAD>']
        xs, ys, ys_in, ys_out, ilens, styles, aligns = to_gpu(data, bos, eos, pad)
        # use aligns to mask xs
        xs = xs.masked_fill_(aligns, self.vocab['<MASK>'])
        if training:
            recon_log_probs, predictions, attns =\
                self.model(xs, ilens, styles, (ys_in, ys_out))
        else:
            recon_log_probs, predictions, attns =\
                self.model(xs, ilens, styles)
        
        # Reconstruction loss
        recon_loss = -torch.mean(recon_log_probs)
        return recon_loss
    
    def _log_train(self, epoch, train_steps, total_steps, loss):
        print(f'epoch: {epoch}, [{train_steps + 1}/{total_steps}], '
              f'train_loss: {loss:.3f}', end='\r')
        
        # add to logger
        tag = self.config['tag']
        self.logger.scalar_summary(tag=f'{tag}/train/reconstruct loss', 
                                   value=loss,
                                   step=(epoch*total_steps+train_steps+1))
        return

    def _train_onetime(self, data, total_loss, train_steps, total_steps, epoch):
        loss = self._get_loss(data)
        #calcuate gradients
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, max_norm=self.config['max_grad_norm'])
        self.optimizer.step()
        total_loss += loss.item()
        self._log_train(epoch, train_steps, total_steps, loss)
        return total_loss

    def train_one_epoch(self, epoch):
        total_steps = len(self.train_loader)
        total_loss = 0.
        for train_steps, data in enumerate(self.train_loader):
            total_loss = self._train_onetime(data, total_loss, train_steps, total_steps, epoch)
        print()
        return (total_loss/total_steps)

    def _valid_reverse_style(self, data_loader):
        all_prediction = []
        all_inputs = []
        for step, data in enumerate(data_loader):
            bos = self.vocab['<BOS>']
            eos = self.vocab['<EOS>']
            pad = self.vocab['<PAD>']
            xs, ys, ys_in, ys_out, ilens, styles, aligns = to_gpu(data, bos, eos, pad)
            reverse_styles = self._get_reverse_style(styles)
            xs = xs.masked_fill_(aligns, self.vocab['<MASK>'])
            _, predictions, _ =\
                self.model(xs, ilens, reverse_styles, dec_input=None,
                           max_dec_timesteps=self.config['max_dec_timesteps'])
            all_prediction = all_prediction + predictions.cpu().tolist()
            all_inputs = all_inputs + [y.cpu().tolist() for y in ys]
        return all_prediction, all_inputs
    
    def _log_valid(self, epoch, steps, total_steps, loss):
        print(f'epoch: {epoch}, [{steps + 1}/{total_steps}], '
              f'valid_loss: {loss:.3f}', end='\r')
        # add to logger
        tag = self.config['tag']
        self.logger.scalar_summary(tag=f'{tag}/val/recontruct loss', 
                                   value=loss, 
                                   step=(epoch*total_steps+steps+1))
        return
  
    def _valid_recons(self, epoch, data_loader, total_loss, step_start, total_steps):
        for step, data in enumerate(data_loader):
            loss  = self._get_loss(data, False)
            total_loss += loss.item()
            self._log_valid(epoch, (step+step_start), total_steps, loss)
        return total_loss

    def validation(self, epoch):
        self.model.eval()
        total_loss_recon = 0.
        # positive input
        total_steps = len(self.dev_pos_loader)+len(self.dev_neg_loader)
        step_start = 0
        total_loss_recon = self._valid_recons(epoch, self.dev_pos_loader, 
                                              total_loss_recon,
                                              step_start,
                                              total_steps)
        # negative input
        step_start = len(self.dev_pos_loader)
        total_loss_recon = self._valid_recons(epoch, self.dev_neg_loader, 
                                              total_loss_recon,
                                              step_start,
                                              total_steps)
        avg_recon_loss = (total_loss_recon/float(total_steps))

        posdata_pred, posdata_input =\
            self._valid_reverse_style(self.dev_pos_loader)
        # get sentence
        posdata_pred, posdata_input = self.idx2sent(posdata_pred, posdata_input)
        # write file
        write_file_path = self.config['dev_file_path']
        if not os.path.exists(write_file_path):
            os.makedirs(write_file_path)
        file_path_pos = os.path.join(write_file_path, 
                                     f'dev.1to0.pred')
        file_path_gtpos = os.path.join(write_file_path, 
                                       f'dev.1to0.input')
        writefile(posdata_pred, file_path_pos)
        writefile(posdata_input, file_path_gtpos)
        # negative input
        negdata_pred, negdata_input =\
            self._valid_reverse_style(self.dev_neg_loader)
        # get sentence
        negdata_pred, negdata_input = self.idx2sent(negdata_pred, negdata_input)
        # write file
        file_path_neg = os.path.join(write_file_path, 
                                     f'dev.0to1.pred')
        file_path_gtneg = os.path.join(write_file_path, 
                                       f'dev.0to1.input')
        writefile(negdata_pred, file_path_neg)
        writefile(negdata_input, file_path_gtneg)
        self.model.train()
        
        # evaluation
        pos_acc = Transferability(file_path_pos, 
                                  self.config['style_classifier_path'],
                                  '__label__0')
        neg_acc = Transferability(file_path_neg, 
                                  self.config['style_classifier_path'],
                                  '__label__1')
        avg_acc = (pos_acc+neg_acc)/2.0
        
        selfbleu_pos, _ = BLEU(file_path_pos, file_path_gtpos)
        selfbleu_neg, _ = BLEU(file_path_neg, file_path_gtneg)
        avg_selfbleu = (selfbleu_pos + selfbleu_neg)/2.0
        fluency_pos = calculate_fluency(file_path_pos)
        fluency_neg = calculate_fluency(file_path_neg)
        avg_fluency = (fluency_pos+fluency_neg)/2.0
        return avg_recon_loss, avg_acc, avg_selfbleu, avg_fluency
    
    def idx2sent(self, all_prediction, all_ys):
        # remove eos and pad
        prediction_til_eos = remove_pad_eos(all_prediction, eos=self.vocab['<EOS>'])
        # indexes to sentences
        ground_truth_sents = to_sents(all_ys, self.vocab, self.non_lang_syms)
        prediction_sents = to_sents(prediction_til_eos, self.vocab, self.non_lang_syms)

        return prediction_sents, ground_truth_sents

    def test(self, state_dict=None):
        # load model
        if not state_dict:
            self.load_model(self.config['load_model_path'], self.config['load_optimizer'])
        else:
            self.model.load_state_dict(state_dict)

        # get test dataset
        root_dir = self.config['dataset_root_dir']
        test_pos_set = self.config['test_pos']
        test_neg_set = self.config['test_neg']
        test_pos_dataset = PickleDataset(os.path.join(root_dir, f'{test_pos_set}.p'), 
                                         config=None, sort=True)
        test_pos_loader = get_data_loader(test_pos_dataset, batch_size=32, shuffle=False)
        test_neg_dataset = PickleDataset(os.path.join(root_dir, f'{test_neg_set}.p'), 
                                         config=None, sort=True)
        test_neg_loader = get_data_loader(test_neg_dataset, batch_size=32, shuffle=False)
        
        self.model.eval()
        # positive input
        posdata_pred, posdata_input =\
            self._valid_reverse_style(test_pos_loader)
        # get sentence
        posdata_pred, posdata_input = self.idx2sent(posdata_pred, posdata_input)
        # write file
        write_file_path = self.config['test_file_path']
        if not os.path.exists(write_file_path):
            os.makedirs(write_file_path)
        file_path_pos = os.path.join(write_file_path, 
                                     f'test.1to0.pred')
        file_path_gtpos = os.path.join(write_file_path, 
                                       f'test.1to0.input')
        writefile(posdata_pred, file_path_pos)
        writefile(posdata_input, file_path_gtpos)
        # negative input
        negdata_pred, negdata_input =\
            self._valid_reverse_style(test_neg_loader)
        # get sentence
        negdata_pred, negdata_input = self.idx2sent(negdata_pred, negdata_input)
        # write file
        file_path_neg = os.path.join(write_file_path, 
                                     f'test.0to1.pred')
        file_path_gtneg = os.path.join(write_file_path, 
                                       f'test.0to1.input')
        writefile(negdata_pred, file_path_neg)
        writefile(negdata_input, file_path_gtneg)       

        self.model.train()
        # evaluation
        pos_acc = Transferability(file_path_pos, 
                                  self.config['style_classifier_path'],
                                  '__label__0')
        neg_acc = Transferability(file_path_neg, 
                                  self.config['style_classifier_path'],
                                  '__label__1')
        avg_acc = (pos_acc+neg_acc)/2.0
        print(f'Average style accuracy: {avg_acc:.4f}')
        selfbleu_pos, _ = BLEU(file_path_pos, file_path_gtpos)
        selfbleu_neg, _ = BLEU(file_path_neg, file_path_gtneg)
        avg_selfbleu = (selfbleu_pos + selfbleu_neg)/2.0
        print(f'Average self bleu score: {avg_selfbleu:.4f}')
        fluency_pos = calculate_fluency(file_path_pos)
        fluency_neg = calculate_fluency(file_path_neg)
        avg_fluency = (fluency_pos+fluency_neg)/2.0
        print(f'Averge fluency score: {avg_fluency:.4f}')
        return avg_acc*avg_selfbleu/avg_fluency
