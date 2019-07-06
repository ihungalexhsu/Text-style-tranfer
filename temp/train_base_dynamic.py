import torch 
import torch.nn.functional as F
import numpy as np
from model_dynamicmask import E2E
from dataloader import get_data_loader
from dataset import PickleDataset
from utils import *
from utils import _seq_mask
from evaluation.calculate_bleu import BLEU
from evaluation.calculate_transfer import Transferability 
from evaluation.calculate_fluency_bert import calculate_fluency 
import yaml
import os
import copy
import pickle

class Base_model_dynamic(object):
    def __init__(self, config, beta=10, gamma=10, load_model=False):
        self.config = config
        print(self.config)
        self.beta = float(beta)
        self.gamma = float(gamma)
        # logger
        self.logger = Logger(config['logdir']+'b'+str(self.beta)+'g'+str(self.gamma))

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
        train_pos_set = self.config['train_pos']
        train_neg_set = self.config['train_neg']
        self.train_pos_dataset = PickleDataset(os.path.join(root_dir,f'{train_pos_set}.p'),
                                               config=self.config,
                                               sort=self.config['sort_dataset'])
        print("pos train data size: ", len(self.train_pos_dataset))
        self.train_pos_loader = get_data_loader(self.train_pos_dataset, 
                                                batch_size=self.config['batch_size'], 
                                                shuffle=self.config['shuffle'])
        print("pos train data # of batch: ", len(self.train_pos_loader))
        self.train_neg_dataset = PickleDataset(os.path.join(root_dir,f'{train_neg_set}.p'),
                                               config=self.config,
                                               sort=self.config['sort_dataset'])
        print("neg train data size: ", len(self.train_neg_dataset))
        self.train_neg_loader = get_data_loader(self.train_neg_dataset, 
                                                batch_size=self.config['batch_size'], 
                                                shuffle=self.config['shuffle'])
        print("neg train data # of batch: ", len(self.train_neg_loader))
        # get dev dataset
        dev_pos_set = self.config['dev_pos']
        dev_neg_set = self.config['dev_neg']
        self.dev_pos_dataset = PickleDataset(os.path.join(root_dir, f'{dev_pos_set}.p'), 
                                             sort=True)
        self.dev_neg_dataset = PickleDataset(os.path.join(root_dir, f'{dev_neg_set}.p'), 
                                             sort=True)
        self.dev_pos_loader = get_data_loader(self.dev_pos_dataset, 
                                              batch_size=self.config['batch_size'], 
                                              shuffle=False)
        self.dev_neg_loader = get_data_loader(self.dev_neg_dataset, 
                                              batch_size=self.config['batch_size'], 
                                              shuffle=False)
        return
    
    def build_model(self, load_model=False):
        pretrain_w2v_path = self.config['pretrain_w2v_path']
        if pretrain_w2v_path is None:
            pretrain_w2v = None
        else:
            pretrain_w2v = getw2v(pretrain_w2v_path, self.vocab, self.vocab['<PAD>'], False)
        self.model = cc_model(E2E(vocab_size=len(self.vocab),
                                  embedding_dim=self.config['embedding_dim'],
                                  enc_hidden_dim=self.config['enc_hidden_dim'],
                                  enc_layers=self.config['enc_n_layers'],
                                  enc_dropout=self.config['enc_dropout_p'],
                                  bidirectional=self.config['bidirectional'],
                                  maskgen_dim=self.config['maskgen_dim'],
                                  dec_hidden_dim=self.config['dec_hidden_dim'],
                                  dec_dropout=self.config['dec_dropout_p'],
                                  bos=self.vocab['<BOS>'],
                                  eos=self.vocab['<EOS>'],
                                  pad=self.vocab['<PAD>'],
                                  n_styles=self.config['n_style_type'],
                                  style_emb_dim=self.config['style_emb_dim'],
                                  att_dim=self.config['att_dim'],
                                  classifier_dim=self.config['cls_dim'],
                                  cls_dropout=self.config['cls_dropout_p'],
                                  cls_att_dim=self.config['cls_att_dim'],
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
        model_path = model_path+'_b'+str(self.beta)+'g'+str(self.gamma)
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
        model_path = model_path+'_b'+str(self.beta)+'g'+str(self.gamma)
        torch.save(self.model.state_dict(), f'{model_path}.ckpt')
        torch.save(self.optimizer.state_dict(), f'{model_path}.opt')
        return
    
    def train(self):
        best_score = 0.
        best_model = None
        early_stop_counter = 0
        # tf_rate
        init_tf = self.config['init_tf_rate']
        tf_start_decay_e = self.config['tf_start_decay_epochs']
        tf_decay_e = self.config['tf_decay_epochs']
        tf_lb = self.config['tf_rate_lowerbound']
        
        print('------start training-------')
        for epoch in range(self.config['epochs']):
            if epoch > tf_start_decay_e:
                if epoch <= tf_decay_e:
                    tf_rate = init_tf-(init_tf-tf_lb)*((epoch-tf_start_decay_e)/(tf_decay_e-tf_start_decay_e))
                else:
                    tf_rate = tf_lb
            else:
                tf_rate = init_tf
            # train one epoch
            avg_train_loss = self.train_one_epoch(epoch, tf_rate)
            # save model in every epoch
            if not os.path.exists(self.config['model_dir']):
                os.makedirs(self.config['model_dir'])
            model_path = os.path.join(self.config['model_dir'], self.config['model_name'])
            self.save_model(f'{model_path}-{epoch:02d}')
            self.save_model(f'{model_path}_latest')
            
            # validation
            val_loss, transfer_acc, selfbleu, fluency = self.validation(epoch)
            print(f'epoch: {epoch}, tf_rate: {tf_rate:.3f}, val_loss: {val_loss:.4f}, '
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
    
    def _get_loss(self, data, tf_rate):
        bos = self.vocab['<BOS>']
        eos = self.vocab['<EOS>']
        pad = self.vocab['<PAD>']
        xs, ys, ys_in, ys_out, ilens, styles, aligns = to_gpu(data, bos, eos, pad)
        mask, cls_log_probs, cls_predicts, recon_log_probs, predictions, attns =\
            self.model(xs, ilens, styles, (ys_in, ys_out), tf_rate)
        
        # mask regularizer
        #mask_loss = -torch.mean(mask**2+(1-mask)**2)
        mask_loss = 0
        # Reconstruction loss
        recon_loss = -torch.mean(recon_log_probs)
        # Coverage loss
        coverage = torch.sum(attns,dim=1)
        ones = coverage.new_ones(coverage.size())
        cover_loss = torch.mean((coverage-ones)**2)
        # Classifier loss
        true_log_probs = torch.gather(cls_log_probs, dim=1,
                                      index=styles.unsqueeze(1)).squeeze(1)
        cls_loss = -torch.mean(true_log_probs)*self.beta
        correct = styles.eq(cls_predicts.view(-1).long()).sum().item()
        acc = float(correct)/float(styles.size(0))
        # Dis loss
        #true_dislog_probs = torch.gather(dis_log_probs, dim=1,
        #                                 index=styles.unsqueeze(1)).squeeze(1)
        #dis_loss = -torch.mean(true_dislog_probs)*self.gamma
        #loss = mask_loss+recon_loss+cover_loss+cls_loss+dis_loss
        loss = mask_loss+recon_loss+cover_loss+cls_loss
        return loss, mask_loss, recon_loss, cover_loss, cls_loss, acc
    
    def _log_train(self, epoch, train_steps, total_steps, loss, mask_loss,
                   recon_loss, cover_loss, cls_loss, acc):
        print(f'epoch: {epoch}, [{train_steps + 1}/{total_steps}], '
              f'train_loss: {loss:.3f}', end='\r')
        
        # add to logger
        tag = self.config['tag']
        self.logger.scalar_summary(tag=f'{tag}/train/total loss', 
                                   value=loss, 
                                   step=(epoch*total_steps+train_steps+1))
        '''
        self.logger.scalar_summary(tag=f'{tag}/train/mask loss', 
                                   value=mask_loss,
                                   step=(epoch*total_steps+train_steps+1))
        '''
        self.logger.scalar_summary(tag=f'{tag}/train/reconstruct loss', 
                                   value=recon_loss,
                                   step=(epoch*total_steps+train_steps+1))
        self.logger.scalar_summary(tag=f'{tag}/train/coverage loss', 
                                   value=cover_loss,
                                   step=(epoch*total_steps+train_steps+1))
        self.logger.scalar_summary(tag=f'{tag}/train/classify loss', 
                                   value=cls_loss.item()/self.beta,
                                   step=(epoch*total_steps+train_steps+1))
        '''
        self.logger.scalar_summary(tag=f'{tag}/train/discriminator loss', 
                                   value=dis_loss.item()/self.gamma,
                                   step=(epoch*total_steps+train_steps+1))
        '''
        self.logger.scalar_summary(tag=f'{tag}/train/classify acc', 
                                   value=acc,
                                   step=(epoch*total_steps+train_steps+1))
        return

    def _train_onetime(self, data, total_loss, tf_rate, train_steps, total_steps, epoch):
        loss, mask_loss, recon_loss, cover_loss, cls_loss, acc =\
            self._get_loss(data, tf_rate)
        #calcuate gradients
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, max_norm=self.config['max_grad_norm'])
        self.optimizer.step()
        total_loss += loss.item()
        self._log_train(epoch, train_steps, total_steps, loss, mask_loss, recon_loss,
                        cover_loss, cls_loss, acc)
        train_steps +=1
        return (train_steps, total_loss)

    def train_one_epoch(self, epoch, tf_rate):
        total_steps = len(self.train_pos_loader)*2
        total_loss = 0.
        assert len(self.train_pos_loader) >= len(self.train_neg_loader)
        pos_data_iterator = iter(self.train_pos_loader)
        neg_data_iterator = iter(self.train_neg_loader)
        train_steps = 0
        for i in range(len(self.train_pos_loader)):
            try:
                data = next(pos_data_iterator)
            except StopIteration:
                print('StopIteration in pos part')
                pass
            train_steps, total_loss = self._train_onetime(data, total_loss, tf_rate,
                                                          train_steps, total_steps, epoch)
            try:
                data = next(neg_data_iterator)
            except StopIteration:
                neg_data_iterator = iter(self.train_neg_loader)
                data = next(neg_data_iterator)
            train_steps, total_loss = self._train_onetime(data, total_loss, tf_rate,
                                                          train_steps, total_steps, epoch)

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
            mask, _, _, _, predictions,_ =\
                self.model(xs, ilens, reverse_styles, dec_input=None,
                           max_dec_timesteps=self.config['max_dec_timesteps'])
            all_prediction = all_prediction + predictions.cpu().tolist()
            all_inputs = all_inputs + [y.cpu().tolist() for y in ys]
        return all_prediction, all_inputs
    
    def _log_valid(self, epoch, steps, total_steps, loss,
                   mask_loss, recon_loss, cover_loss, cls_loss, acc):
        print(f'epoch: {epoch}, [{steps + 1}/{total_steps}], '
              f'valid_loss: {loss:.3f}', end='\r')
        # add to logger
        tag = self.config['tag']
        self.logger.scalar_summary(tag=f'{tag}/val/total loss', 
                                   value=loss, 
                                   step=(epoch*total_steps+steps+1))
        '''
        self.logger.scalar_summary(tag=f'{tag}/val/mask loss', 
                                   value=mask_loss, 
                                   step=(epoch*total_steps+steps+1))
        '''
        self.logger.scalar_summary(tag=f'{tag}/val/recon loss', 
                                   value=recon_loss, 
                                   step=(epoch*total_steps+steps+1))
        self.logger.scalar_summary(tag=f'{tag}/val/cover loss', 
                                   value=cover_loss, 
                                   step=(epoch*total_steps+steps+1))
        self.logger.scalar_summary(tag=f'{tag}/val/classifier loss', 
                                   value=cls_loss.item()/self.beta, 
                                   step=(epoch*total_steps+steps+1))
        self.logger.scalar_summary(tag=f'{tag}/val/classifier acc', 
                                   value=acc, 
                                   step=(epoch*total_steps+steps+1))
        '''
        self.logger.scalar_summary(tag=f'{tag}/val/discriminator loss', 
                                   value=dis_loss.item()/self.gamma, 
                                   step=(epoch*total_steps+steps+1))
        '''
        return
  
    def _valid_recons(self, epoch, data_loader, total_loss, step_start, total_steps):
        for step, data in enumerate(data_loader):
            loss, mask_loss, recon_loss, cover_loss, cls_loss, acc =\
                self._get_loss(data, 0.)
            total_loss += loss.item()
            self._log_valid(epoch, (step+step_start), total_steps, loss,
                            mask_loss, recon_loss, cover_loss, cls_loss, acc)
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
        write_file_path = self.config['dev_file_path']+'b'+str(self.beta)+'g'+str(self.gamma)
        if not os.path.exists(write_file_path):
            os.makedirs(write_file_path)
        file_path_pos = os.path.join(write_file_path, 
                                     f'base.dev.1to0.pred')
        file_path_gtpos = os.path.join(write_file_path, 
                                       f'base.dev.1.input')
        writefile(posdata_pred, file_path_pos)
        writefile(posdata_input, file_path_gtpos)
        # negative input
        negdata_pred, negdata_input =\
            self._valid_reverse_style(self.dev_neg_loader)
        # get sentence
        negdata_pred, negdata_input = self.idx2sent(negdata_pred, negdata_input)
        # write file
        file_path_neg = os.path.join(write_file_path, 
                                     f'base.dev.0to1.pred')
        file_path_gtneg = os.path.join(write_file_path, 
                                       f'base.dev.0.input')
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
                                         config=None, sort=False)
        test_pos_loader = get_data_loader(test_pos_dataset, batch_size=2, shuffle=False)
        test_neg_dataset = PickleDataset(os.path.join(root_dir, f'{test_neg_set}.p'), 
                                         config=None, sort=False)
        test_neg_loader = get_data_loader(test_neg_dataset, batch_size=2, shuffle=False)
        self.model.eval()
        # positive input
        posdata_pred, posdata_input =\
            self._valid_reverse_style(test_pos_loader)
        # get sentence
        posdata_pred, posdata_input = self.idx2sent(posdata_pred, posdata_input)
        # write file
        write_file_path = self.config['test_file_path']+'b'+str(self.beta)+'g'+str(self.gamma)
        if not os.path.exists(write_file_path):
            os.makedirs(write_file_path)
        file_path_pos = os.path.join(write_file_path, 
                                     f'base.test.1to0.pred')
        file_path_gtpos = os.path.join(write_file_path, 
                                       f'base.test.1.input')
        writefile(posdata_pred, file_path_pos)
        writefile(posdata_input, file_path_gtpos)
        # negative input
        negdata_pred, negdata_input =\
            self._valid_reverse_style(test_neg_loader)
        # get sentence
        negdata_pred, negdata_input = self.idx2sent(negdata_pred, negdata_input)
        # write file
        file_path_neg = os.path.join(write_file_path, 
                                     f'base.test.0to1.pred')
        file_path_gtneg = os.path.join(write_file_path, 
                                       f'base.test.0.input')
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
        bleu_pos, _ = BLEU(file_path_pos, self.config['humanref_path_pos']) 
        bleu_neg, _ = BLEU(file_path_neg, self.config['humanref_path_neg']) 
        avg_bleu = (bleu_pos + bleu_neg)/2.0
        print(f'Average bleu score comparing with human: {avg_bleu:.4f}')
        fluency_pos = calculate_fluency(file_path_pos)
        fluency_neg = calculate_fluency(file_path_neg)
        avg_fluency = (fluency_pos+fluency_neg)/2.0
        print(f'Averge fluency score: {avg_fluency:.4f}')
        return avg_acc*avg_selfbleu/avg_fluency