import torch 
import torch.nn.functional as F
import numpy as np
from model import E2E
from model import Encoder, Decoder
from dataloader import get_data_loader
from dataset import PickleDataset
from utils import *
from utils import _seq_mask
import yaml
import os
import pickle

class AutoEncoder(object):
    def __init__(self, config, load_model=False):

        self.config = config
        print(self.config)

        # logger
        self.logger = Logger(config['logdir'])

        # load vocab and non lang syms
        self.load_vocab()
       
        # get data loader
        self.get_data_loaders()

        # get label distribution
        self.get_label_dist(self.train_lab_dataset)

        # build model and optimizer
        self.build_model(load_model=load_model)

    def save_model(self, model_path):
        encoder_path = model_path+'_encoder'
        decoder_path = model_path+'_decoder'
        opt1_path = model_path+'_opt1'
        torch.save(self.encoder.state_dict(), f'{encoder_path}.ckpt')
        torch.save(self.decoder.state_dict(), f'{decoder_path}.ckpt')
        torch.save(self.optimizer_m1.state_dict(), f'{opt1_path}.opt')
        return

    def load_vocab(self):
        with open(self.config['vocab_path'], 'rb') as f:
            self.vocab = pickle.load(f) # a dict; word to index
        with open(self.config['non_lang_syms_path'], 'rb') as f:
            self.non_lang_syms = pickle.load(f)
        return

    def load_model(self, model_path, load_optimizer):
        print(f'Load model from {model_path}')
        encoder_path = model_path+'_encoder'
        decoder_path = model_path+'_decoder'
        opt1_path = model_path+'_opt1'
        self.encoder.load_state_dict(torch.load(f'{encoder_path}.ckpt'))
        self.decoder.load_state_dict(torch.load(f'{decoder_path}.ckpt'))
        if load_optimizer:
            print(f'Load optmizer from {model_path}')
            self.optimizer_m1.load_state_dict(torch.load(f'{opt1_path}.opt'))
            if self.config['adjust_lr']:
                adjust_learning_rate(self.optimizer_m1, self.config['retrieve_lr_m1']) 
        return

    def get_data_loaders(self):
        root_dir = self.config['dataset_root_dir']
        train_set = self.config['train_set']
        self.train_lab_dataset = PickleDataset(os.path.join(root_dir, 
                                                            f'{train_set}.p'),
                                               config=self.config, 
                                               sort=True)
        self.train_lab_loader = get_data_loader(self.train_lab_dataset, 
                batch_size=self.config['batch_size'], 
                shuffle=self.config['shuffle'])

        # get dev dataset
        dev_set = self.config['dev_set']
        self.dev_dataset = PickleDataset(os.path.join(root_dir, f'{dev_set}.p'), 
                                         sort=True)
        self.dev_loader = get_data_loader(self.dev_dataset, 
                batch_size=self.config['batch_size'], 
                shuffle=False)
        return

    def get_label_dist(self, dataset):
        labelcount = np.zeros(len(self.vocab))
        for token_ids,_ in dataset:
            for ind in token_ids:
                labelcount[ind] += 1.
        labelcount[self.vocab['<EOS>']] += len(dataset)
        labelcount[self.vocab['<PAD>']] = 0
        labelcount[self.vocab['<BOS>']] = 0
        self.labeldist = labelcount / np.sum(labelcount)
        return
             
    def build_model(self, load_model=False):
        labeldist = self.labeldist
        pretrain_w2v_path = self.config['pretrain_w2v_path']
        if pretrain_w2v_path is None:
            pretrain_w2v = None
        else:
            with open(pretrain_w2v_path, 'rb') as f:
                pretrain_w2v = pickle.load(f)

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
        self.decoder = cc_model(Decoder(output_dim=len(self.vocab),
                                        embedding_dim=self.config['embedding_dim'],
                                        hidden_dim=self.config['dec_hidden_dim'],
                                        dropout_rate=self.config['dec_dropout_p'],
                                        bos=self.vocab['<BOS>'],
                                        eos=self.vocab['<EOS>'],
                                        pad=self.vocab['<PAD>'],
                                        enc_out_dim=enc_out_dim,
                                        n_styles=self.config['n_style_type'],
                                        style_emb_dim=self.config['style_emb_dim'],
                                        use_enc_init=self.config['use_enc_init'],
                                        use_attention=self.config['use_attention'],
                                        ls_weight=self.config['ls_weight'],
                                        labeldist=labeldist,
                                        use_style_embedding=False))
        print(self.decoder)
        self.decoder.float()
        self.params_m1=list(self.encoder.parameters())+list(self.decoder.parameters())
        self.optimizer_m1 =\
            torch.optim.Adam(self.params_m1, 
                             lr=self.config['learning_rate_m1'], 
                             weight_decay=self.config['weight_decay_m1'])

        if load_model:
            self.load_model(self.config['load_model_path'], self.config['load_optimizer'])

        return

    def validation(self):

        self.encoder.eval()
        self.decoder.eval()
        all_prediction, all_ys = [], []
        total_loss = 0.
        for step, data in enumerate(self.dev_loader):

            bos = self.vocab['<BOS>']
            eos = self.vocab['<EOS>']
            pad = self.vocab['<PAD>']
            xs, ys, ys_in, ys_out, ilens, styles = to_gpu(data, bos, eos, pad)
            
            # input the encoder
            enc_outputs, enc_lens = self.encoder(xs, ilens)
            logits, log_probs, prediction, attns=\
                self.decoder(enc_outputs, enc_lens, styles, None,
                             max_dec_timesteps=self.config['max_dec_timesteps'])

           
            seq_len = [y.size(0) + 1 for y in ys]
            mask = cc(_seq_mask(seq_len=seq_len, max_len=log_probs.size(1)))
            loss = (-torch.sum(log_probs*mask))/sum(seq_len)
            total_loss += loss.item()

            all_prediction = all_prediction + prediction.cpu().numpy().tolist()
            all_ys = all_ys + [y.cpu().numpy().tolist() for y in ys]

        self.encoder.train()
        self.decoder.train()
        # calculate loss
        avg_loss = total_loss / len(self.dev_loader)

        wer, prediction_sents, ground_truth_sents = self.idx2sent(all_prediction, all_ys)

        return avg_loss, wer, prediction_sents, ground_truth_sents
    
    def idx2sent(self, all_prediction, all_ys):
        # remove eos and pad
        prediction_til_eos = remove_pad_eos(all_prediction, eos=self.vocab['<EOS>'])

        # indexes to sentences
        prediction_sents = to_sents(prediction_til_eos, self.vocab, self.non_lang_syms)
        ground_truth_sents = to_sents(all_ys, self.vocab, self.non_lang_syms)
        wer = calculate_wer(prediction_til_eos, all_ys)
        return wer, prediction_sents, ground_truth_sents

    def test(self, state_dict=None):

        # load model
        if not state_dict:
            self.load_model(self.config['load_model_path'], self.config['load_optimizer'])
        else:
            self.encoder.load_state_dict(state_dict[0])
            self.decoder.load_state_dict(state_dict[1])

        # get test dataset
        root_dir = self.config['dataset_root_dir']
        test_set = self.config['test_set']
        test_file_name = self.config['test_file_name']

        test_dataset = PickleDataset(os.path.join(root_dir, f'{test_set}.p'), 
                                     config=None, sort=False)

        test_loader = get_data_loader(test_dataset, 
                                      batch_size=2, 
                                      shuffle=False)

        self.encoder.eval()
        self.decoder.eval()
        all_prediction, all_ys = [], []
        for step, data in enumerate(test_loader):
            bos = self.vocab['<BOS>']
            eos = self.vocab['<EOS>']
            pad = self.vocab['<PAD>']
            xs, ys, ys_in, ys_out, ilens, styles = to_gpu(data, bos, eos, pad)
            
            # input the encoder
            enc_outputs, enc_lens = self.encoder(xs, ilens)
            logits, log_probs, prediction, attns=\
                self.decoder(enc_outputs, enc_lens, styles, None,
                             max_dec_timesteps=self.config['max_dec_timesteps'])

            all_prediction = all_prediction + prediction.cpu().numpy().tolist()
            all_ys = all_ys + [y.cpu().numpy().tolist() for y in ys]

        self.encoder.train()
        self.decoder.train()

        wer, prediction_sents, ground_truth_sents = self.idx2sent(all_prediction, all_ys)

        with open(f'{test_file_name}.txt', 'w') as f:
            for p in prediction_sents:
                f.write(f'{p}\n')

        print(f'{test_file_name}: {len(prediction_sents)} utterances, WER={wer:.4f}')
        return wer

    def _normal_target(self, x):
        out = x.new_zeros(x.size())
        out = out.fill_(1/int(x.size(-1)))
        return out

    def train_one_epoch(self, epoch, tf_rate):

        total_steps = len(self.train_lab_loader)
        total_loss = 0.
        
        for train_steps, data in enumerate(self.train_lab_loader):
            bos = self.vocab['<BOS>']
            eos = self.vocab['<EOS>']
            pad = self.vocab['<PAD>']
            xs, ys, ys_in, ys_out, ilens, styles = to_gpu(data, bos, eos, pad)

            # input the encoder
            enc_outputs, enc_lens = self.encoder(xs, ilens)
            logits, log_probs, prediction, attns=\
                self.decoder(enc_outputs, enc_lens, styles, (ys_in, ys_out),
                             tf_rate=tf_rate, sample=False,
                             max_dec_timesteps=self.config['max_dec_timesteps'])

            loss = -torch.mean(log_probs)*self.config['m1_loss_ratio']
            total_loss += loss.item()
            
            # calculate gradients 
            self.optimizer_m1.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.params_m1, max_norm=self.config['max_grad_norm'])
            self.optimizer_m1.step()
            # print message
            print(f'epoch: {epoch}, [{train_steps + 1}/{total_steps}], loss: {loss:.3f}', end='\r')
            # add to logger
            tag = self.config['tag']
            self.logger.scalar_summary(tag=f'{tag}/train/loss', 
                                       value=loss.item()/self.config['m1_loss_ratio'], 
                                       step=epoch * total_steps + train_steps + 1)
        print()
        return total_loss/total_steps

    def train(self):

        best_wer = 200
        best_model = None
        early_stop_counter = 0

        # tf_rate
        init_tf_rate = self.config['init_tf_rate']
        tf_start_decay_epochs = self.config['tf_start_decay_epochs']
        tf_decay_epochs = self.config['tf_decay_epochs']
        tf_rate_lowerbound = self.config['tf_rate_lowerbound']

	    # lr scheduler
        if self.config['change_lr_epoch'] is not None:
            milestone = [int(num) for num in str(self.config['change_lr_epoch']).split(',')]
        else:
            milestone = []
        scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_m1,
                                                         milestones=milestone,
                                                         gamma=self.config['lr_gamma'])

        print('------start training-------')
        for epoch in range(self.config['epochs']):
            scheduler.step()
            if epoch > tf_start_decay_epochs:
                if epoch <= tf_decay_epochs:
                    tf_rate = init_tf_rate-(init_tf_rate-tf_rate_lowerbound)*((epoch-tf_start_decay_epochs)/(tf_decay_epochs-tf_start_decay_epochs))
                else:
                    tf_rate = tf_rate_lowerbound
            else:
                tf_rate = init_tf_rate

            # train one epoch
            avg_train_loss = self.train_one_epoch(epoch, tf_rate)
            # validation
            avg_valid_loss, wer, prediction_sents, ground_truth_sents = self.validation()

            print(f'Epoch: {epoch}, tf_rate={tf_rate:.3f}, train_loss={avg_train_loss:.4f}, '
                    f'valid_loss={avg_valid_loss:.4f}, val_WER={wer:.4f}')

            # add to tensorboard
            tag = self.config['tag']
            self.logger.scalar_summary(f'{tag}/val/wer', wer, epoch)
            self.logger.scalar_summary(f'{tag}/val/loss', avg_valid_loss, epoch)

            # only add first n samples
            lead_n = self.config['sample_num']
            print('-----------------')
            for i, (p, gt) in enumerate(zip(prediction_sents[:lead_n], ground_truth_sents[:lead_n])):
                self.logger.text_summary(f'{tag}/sample/prediction-{i}', p, epoch)
                self.logger.text_summary(f'{tag}/sample/ground_truth-{i}', gt, epoch)
                print(f'prediction-{i+1}: {p}')
                print(f'reference-{i+1}: {gt}')
            print('-----------------')

            # save model in every epoch
            if not os.path.exists(self.config['model_dir']):
                os.makedirs(self.config['model_dir'])
            model_path = os.path.join(self.config['model_dir'], self.config['model_name'])
            self.save_model(f'{model_path}-{epoch:03d}')
            if wer < best_wer: 
                # save model
                model_path = os.path.join(self.config['model_dir'], self.config['model_name']+'_best')
                best_wer = wer
                self.save_model(model_path)
                best_model_enc = self.encoder.state_dict()
                best_model_dec = self.decoder.state_dict()
                print(f'Save #{epoch} model, val_loss={avg_valid_loss:.4f}, WER={wer:.4f}')
                print('-----------------')
                early_stop_counter=0
            if epoch >= self.config['early_stop_start_epoch']:
                early_stop_counter += 1
                if early_stop_counter > self.config['early_stop_patience']:
                    break
        best_model = (best_model_enc, best_model_dec)
        return best_model, best_wer
