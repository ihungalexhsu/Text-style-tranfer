import torch 
import torch.nn.functional as F
import numpy as np
from model import Encoder, DecWithAux
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
import time

class Proposed_model(object):
    def __init__(self, config, load_model=False):
        self.config = config
        print(self.config)

        # logger
        self.logger = Logger(config['logdir'])

        # load vocab and non lang syms
        self.load_vocab()
       
        # set up sentiment word list
        self.setup_wordlist()

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
    
    def setup_wordlist(self):
        pos_word_path = self.config['pos_word_list']
        neg_word_path = self.config['neg_word_list']
        self.pos_word_list = pickle.load(open(f'{pos_word_path}.p', 'rb'))
        self.neg_word_list = pickle.load(open(f'{neg_word_path}.p', 'rb'))
        pretrain_w2v_path = self.config['pretrain_w2v_path']
        embwordlist = getembwordlist(pretrain_w2v_path)
        temp = list()
        for w in self.pos_word_list:
            if w in self.vocab.keys() and w in embwordlist:
                temp.append(self.vocab[w])
        self.pos_word_list = cc(torch.LongTensor(temp))
        temp = list()
        for w in self.neg_word_list:
            if w in self.vocab.keys() and w in embwordlist:
                temp.append(self.vocab[w])
        self.neg_word_list = cc(torch.LongTensor(temp))
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
            #pretrain_w2v, self.vocab = mergew2v(pretrain_w2v_path, self.vocab,
            #                                    self.vocab['<PAD>'], False)
            pretrain_w2v = None
            Smodule_emb, _ = mergew2v(pretrain_w2v_path, copy.deepcopy(self.vocab), 
                                      self.vocab['<PAD>'], False)

        self.encoder = cc_model(Encoder(vocab_size=len(self.vocab),
                                        embedding_dim=self.config['embedding_dim'],
                                        hidden_dim=self.config['enc_hidden_dim'],
                                        n_layers=self.config['enc_n_layers'],
                                        dropout_rate=self.config['enc_dropout_p'],
                                        pad_idx=self.vocab['<PAD>'],
                                        bidirectional=self.config['bidir_enc'],
                                        pre_embedding=pretrain_w2v,
                                        update_embedding=self.config['update_embedding']))
        print(self.encoder)
        self.encoder.float()
        if self.config['bidir_enc']:
            enc_out_dim=2*self.config['enc_hidden_dim']
        else:
            enc_out_dim=self.config['enc_hidden_dim']
        self.decoder = cc_model(DecWithAux(output_dim=len(self.vocab),
                                           embedding_dim=self.config['embedding_dim'],
                                           hidden_dim=self.config['dec_hidden_dim'],
                                           dropout_rate=self.config['dec_dropout_p'],
                                           bos=self.vocab['<BOS>'],
                                           eos=self.vocab['<EOS>'],
                                           pad=self.vocab['<PAD>'],
                                           enc_out_dim=enc_out_dim,
                                           n_styles=self.config['n_style_type'],
                                           style_emb_dim=self.config['style_emb_dim'],
                                           att_dim=self.config['att_dim'],
                                           pos_word_list=self.pos_word_list,
                                           neg_word_list=self.neg_word_list,
                                           Smodule_emb=cc(Smodule_emb),
                                           pre_embedding=pretrain_w2v,
                                           update_embedding=self.config['update_embedding']))
        print(self.decoder)
        self.decoder.float()
        self.params=list(self.encoder.parameters())+list(self.decoder.parameters())
        self.optimizer=\
            torch.optim.Adam(self.params, 
                             lr=self.config['learning_rate'],
                             weight_decay=float(self.config['weight_decay']))
        if load_model:
            self.load_model(self.config['load_model_path'], self.config['load_optimizer'])
        return
 
    def load_model(self, model_path, load_optimizer):
        if os.path.exists(model_path+'_enc.ckpt'):
            print(f'Load model from {model_path}')
            encoder_path = model_path+'_enc'
            decoder_path = model_path+'_dec'
            opt_path = model_path+'_opt'
            self.encoder.load_state_dict(torch.load(f'{encoder_path}.ckpt'))
            state = self.decoder.state_dict()
            state.update(torch.load(f'{decoder_path}.ckpt'))
            self.decoder.load_state_dict(state)
            if not load_optimizer:
                print(f'Load optmizer from {opt_path}')
                self.optimizer.load_state_dict(torch.load(f'{opt_path}.opt'))
                if self.config['adjust_lr']:
                    adjust_learning_rate(self.optimizer, self.config['retrieve_lr']) 
        return
  
    def save_model(self, model_path):
        encoder_path = model_path+'_enc'
        decoder_path = model_path+'_dec'
        opt_path = model_path+'_opt'
        torch.save(self.encoder.state_dict(), f'{encoder_path}.ckpt')
        torch.save(self.decoder.state_dict(), f'{decoder_path}.ckpt')
        torch.save(self.optimizer.state_dict(), f'{opt_path}.opt')
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
                best_model_enc = copy.deepcopy(self.encoder.state_dict())
                best_model_dec = copy.deepcopy(self.decoder.state_dict())
                print(f'Save #{epoch} model, val_loss={val_loss:.4f}, score={score:.4f}')
                print('-----------------')
                early_stop_counter=0
            if epoch >= self.config['early_stop_start_epoch']:
                early_stop_counter += 1
                if early_stop_counter > self.config['early_stop_patience']:
                    break
        best_model = (best_model_enc, best_model_dec)
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
    
    def _get_loss(self, data, tf_rate, senti):
        bos = self.vocab['<BOS>']
        eos = self.vocab['<EOS>']
        pad = self.vocab['<PAD>']
        xs, ys, ys_in, ys_out, ilens, styles, aligns = to_gpu(data, bos, eos, pad)
        masked = True if np.random.random_sample() <= 1.0 else False
        if masked:
            xs_i=xs.clone().masked_fill_(aligns, pad)
        else:
            xs_i=xs
        enc_output, enc_lens = self.encoder(xs_i, ilens)
        switchs, recon_log_probs, predictions, attns =\
            self.decoder(enc_output, enc_lens, styles, aligns, 
                         senti, xs, (ys_in, ys_out), tf_rate)
        # Reconstruction loss
        recon_loss = -torch.mean(recon_log_probs)
        # Coverage loss
        coverage = torch.sum(attns, dim=1)
        ones = coverage.new_ones(coverage.size())
        cover_loss = torch.mean((coverage-ones)**2)
        loss = recon_loss+cover_loss
        return loss
    
    def _log_train(self, epoch, train_steps, total_steps, loss):
        print(f'epoch: {epoch}, [{train_steps + 1}/{total_steps}], '
              f'train_loss: {loss:.3f}', end='\r')
        
        # add to logger
        tag = self.config['tag']
        self.logger.scalar_summary(tag=f'{tag}/train/loss', 
                                   value=loss, 
                                   step=(epoch*total_steps+train_steps+1))
        return

    def _train_onetime(self, data, total_loss, tf_rate, senti, train_steps, total_steps, epoch):
        loss = self._get_loss(data, tf_rate, senti)
        #calcuate gradients
        time0 = time.time()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, max_norm=self.config['max_grad_norm'])
        self.optimizer.step()
        #print('backward', time.time()-time0)
        total_loss += loss.item()
        self._log_train(epoch, train_steps, total_steps, loss)
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
            train_steps, total_loss = self._train_onetime(data, total_loss, tf_rate, 1,
                                                          train_steps, total_steps, epoch)
            try:
                data = next(neg_data_iterator)
            except StopIteration:
                neg_data_iterator = iter(self.train_neg_loader)
                data = next(neg_data_iterator)
            train_steps, total_loss = self._train_onetime(data, total_loss, tf_rate, 0,
                                                          train_steps, total_steps, epoch)

        print()
        return (total_loss/total_steps)

    def _valid_reverse_style(self, data_loader, senti):
        all_prediction = []
        all_inputs = []
        for step, data in enumerate(data_loader):
            bos = self.vocab['<BOS>']
            eos = self.vocab['<EOS>']
            pad = self.vocab['<PAD>']
            xs, ys, ys_in, ys_out, ilens, styles, aligns = to_gpu(data, bos, eos, pad)
            reverse_styles = self._get_reverse_style(styles)
            enc_output, enc_lens = self.encoder(xs, ilens)
            switchs, _, predictions, attns =\
                self.decoder(enc_output, enc_lens, reverse_styles, aligns, 
                             senti, xs, None)
            all_prediction = all_prediction + predictions.cpu().tolist()
            all_inputs = all_inputs + [y.cpu().tolist() for y in ys]
        return all_prediction, all_inputs
    
    def _log_valid(self, epoch, steps, total_steps, loss):
        print(f'epoch: {epoch}, [{steps + 1}/{total_steps}], '
              f'valid_loss: {loss:.3f}', end='\r')
        # add to logger
        tag = self.config['tag']
        self.logger.scalar_summary(tag=f'{tag}/val/loss', 
                                   value=loss, 
                                   step=(epoch*total_steps+steps+1))
        return
  
    def _valid_recons(self, epoch, data_loader, total_loss, step_start, total_steps, senti):
        for step, data in enumerate(data_loader):
            loss = self._get_loss(data, 0., senti)
            total_loss += loss.item()
            self._log_valid(epoch, (step+step_start), total_steps, loss)
        return total_loss

    def validation(self, epoch):
        self.encoder.eval()
        self.decoder.eval()
        total_loss_recon = 0.
        # positive input
        total_steps = len(self.dev_pos_loader)+len(self.dev_neg_loader)
        step_start = 0
        total_loss_recon = self._valid_recons(epoch, self.dev_pos_loader, 
                                              total_loss_recon,
                                              step_start,
                                              total_steps,
                                              1)
        # negative input
        step_start = len(self.dev_pos_loader)
        total_loss_recon = self._valid_recons(epoch, self.dev_neg_loader, 
                                              total_loss_recon,
                                              step_start,
                                              total_steps,
                                              0)
        avg_recon_loss = (total_loss_recon/float(total_steps))
        posdata_pred, posdata_input =\
            self._valid_reverse_style(self.dev_pos_loader, 0)
        # get sentence
        posdata_pred, posdata_input = self.idx2sent(posdata_pred, posdata_input)
        # write file
        if not os.path.exists(self.config['dev_file_path']):
            os.makedirs(self.config['dev_file_path'])
        file_path_pos = os.path.join(self.config['dev_file_path'], 
                                     f'base.dev.1to0.pred')
        file_path_gtpos = os.path.join(self.config['dev_file_path'], 
                                       f'base.dev.1.input')
        writefile(posdata_pred, file_path_pos)
        writefile(posdata_input, file_path_gtpos)
        # negative input
        negdata_pred, negdata_input =\
            self._valid_reverse_style(self.dev_neg_loader, 1)
        # get sentence
        negdata_pred, negdata_input = self.idx2sent(negdata_pred, negdata_input)
        # write file
        file_path_neg = os.path.join(self.config['dev_file_path'], 
                                     f'base.dev.0to1.pred')
        file_path_gtneg = os.path.join(self.config['dev_file_path'], 
                                       f'base.dev.0.input')
        writefile(negdata_pred, file_path_neg)
        writefile(negdata_input, file_path_gtneg)
        self.encoder.train()
        self.decoder.train()
        
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
        prediction_sents = to_sents(prediction_til_eos, self.vocab, self.non_lang_syms)
        ground_truth_sents = to_sents(all_ys, self.vocab, self.non_lang_syms)

        return prediction_sents, ground_truth_sents

    def test(self, state_dict=None):
        # load model
        if not state_dict:
            self.load_model(self.config['load_model_path'], self.config['load_optimizer'])
        else:
            self.encoder.load_state_dict(state_dict[0])
            self.decoder.load_state_dict(state_dict[1])

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
        self.encoder.eval()
        self.decoder.eval()
        # positive input
        posdata_pred, posdata_input =\
            self._valid_reverse_style(test_pos_loader, 0)
        # get sentence
        posdata_pred, posdata_input = self.idx2sent(posdata_pred, posdata_input)
        # write file
        if not os.path.exists(self.config['test_file_path']):
            os.makedirs(self.config['test_file_path'])
        file_path_pos = os.path.join(self.config['test_file_path'], 
                                     f'base.test.1to0.pred')
        file_path_gtpos = os.path.join(self.config['test_file_path'], 
                                       f'base.test.1.input')
        writefile(posdata_pred, file_path_pos)
        writefile(posdata_input, file_path_gtpos)
        # negative input
        negdata_pred, negdata_input =\
            self._valid_reverse_style(test_neg_loader, 1)
        # get sentence
        negdata_pred, negdata_input = self.idx2sent(negdata_pred, negdata_input)
        # write file
        file_path_neg = os.path.join(self.config['test_file_path'], 
                                     f'base.test.0to1.input')
        file_path_gtneg = os.path.join(self.config['test_file_path'], 
                                       f'base.test.0.input')
        writefile(negdata_pred, file_path_neg)
        writefile(negdata_input, file_path_gtneg)       

        self.encoder.train()
        self.decoder.train()
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
