import torch 
import torch.nn.functional as F
import numpy as np
from model import E2E
from model import Encoder, Decoder, Style_classifier
from dataloader import get_data_loader
from dataset import PickleDataset
from utils import *
from utils import _seq_mask
import yaml
import os
import pickle

class Style_transfer(object):
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
        s_classifier_path = model_path+'_sclr'
        opt1_path = model_path+'_opt1'
        opt2_path = model_path+'_opt2'
        torch.save(self.encoder.state_dict(), f'{encoder_path}.ckpt')
        torch.save(self.decoder.state_dict(), f'{decoder_path}.ckpt')
        torch.save(self.s_classifier.state_dict(), f'{s_classifier_path}.ckpt')
        torch.save(self.optimizer_m1.state_dict(), f'{opt1_path}.opt')
        torch.save(self.optimizer_m2.state_dict(), f'{opt2_path}.opt')
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
        s_classifier_path = model_path+'_sclr'
        opt1_path = model_path+'_opt1'
        opt2_path = model_path+'_opt2'
        self.encoder.load_state_dict(torch.load(f'{encoder_path}.ckpt'))
        self.decoder.load_state_dict(torch.load(f'{decoder_path}.ckpt'))
        self.s_classifier.load_state_dict(torch.load(f'{s_classifier_path}.ckpt'))
        if load_optimizer:
            print(f'Load optmizer from {model_path}')
            self.optimizer_m1.load_state_dict(torch.load(f'{opt1_path}.opt'))
            self.optimizer_m2.load_state_dict(torch.load(f'{opt2_path}.opt'))
            if self.config['adjust_lr']:
                adjust_learning_rate(self.optimizer_m1, self.config['retrieve_lr_m1']) 
                adjust_learning_rate(self.optimizer_m2, self.config['retrieve_lr_m2']) 
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
                                        labeldist=labeldist))
        print(self.decoder)
        self.decoder.float()
        self.s_classifier=\
            cc_model(Style_classifier(enc_out_dim=enc_out_dim,
                                      hidden_dim=self.config['s_classifier_hidden_dim'],
                                      n_layers=self.config['s_classifier_n_layers'],
                                      out_dim=self.config['n_style_type']))
        print(self.s_classifier)
        self.s_classifier.float()
        self.params_m1=list(self.encoder.parameters())+list(self.decoder.parameters())
        self.params_m2=list(self.s_classifier.parameters())
        self.optimizer_m1 =\
            torch.optim.Adam(self.params_m1, 
                             lr=self.config['learning_rate_m1'], 
                             weight_decay=self.config['weight_decay_m1'])
        self.optimizer_m2 =\
            torch.optim.Adam(self.params_m2, 
                             lr=self.config['learning_rate_m2'], 
                             weight_decay=self.config['weight_decay_m2'])

        if load_model:
            self.load_model(self.config['load_model_path'], self.config['load_optimizer'])

        return

    def validation(self):

        self.encoder.eval()
        self.decoder.eval()
        self.s_classifier.eval()
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
        self.s_classifier.train()
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
        self.s_classifier.eval()
        all_prediction, all_ys = [], []
        for step, data in enumerate(test_loader):
            bos = self.vocab['<BOS>']
            eos = self.vocab['<EOS>']
            pad = self.vocab['<PAD>']
            xs, ys, ys_in, ys_out, ilens, styles = to_gpu(data, bos, eos, pad)
            reverse_styles = styles.cpu().new_zeros(styles.size())
            for idx, ele in enumerate(styles.cpu().tolist()):
                if not(ele):
                    reverse_styles[idx]=1
            reverse_styles=cc(torch.LongTensor(reverse_styles))
            # input the encoder
            enc_outputs, enc_lens = self.encoder(xs, ilens)
            logits, log_probs, prediction, attns=\
                self.decoder(enc_outputs, enc_lens, reverse_styles, None,
                             max_dec_timesteps=self.config['max_dec_timesteps'])

            all_prediction = all_prediction + prediction.cpu().numpy().tolist()
            all_ys = all_ys + [y.cpu().numpy().tolist() for y in ys]

        self.encoder.train()
        self.decoder.train()
        self.s_classifier.train()

        wer, prediction_sents, ground_truth_sents = self.idx2sent(all_prediction, all_ys)

        with open(f'{test_file_name}.txt', 'w') as f:
            for idx, p in enumerate(prediction_sents):
                f.write(f'Predictions :{p}\n')
                f.write(f'OriSentence :{ground_truth_sents[idx]}\n')

        print(f'{test_file_name}: {len(prediction_sents)} utterances, WER={wer:.4f}')
        return wer

    def _normal_target(self, x):
        out = x.new_zeros(x.size())
        out = out.fill_(1/int(x.size(-1)))
        return out

    def train_one_epoch(self, epoch, tf_rate):

        total_steps = len(self.train_lab_loader)
        total_loss = 0.
        total_discri_loss = 0.
        total_cheat_loss = 0.
        total_cycle_loss = 0.

        for cnt in range(self.config['m2_train_freq']):
            for train_steps, data in enumerate(self.train_lab_loader):
                bos = self.vocab['<BOS>']
                eos = self.vocab['<EOS>']
                pad = self.vocab['<PAD>']
                xs, ys, ys_in, ys_out, ilens, styles = to_gpu(data, bos, eos, pad)
                # input the encoder
                enc_outputs, enc_lens = self.encoder(xs, ilens)
                enc_representation = get_enc_context(enc_outputs, enc_lens)
                s_logits, s_log_probs, s_pred = self.s_classifier(enc_representation)
                #s_logits, s_log_probs, s_pred = self.s_classifier(enc_outputs, enc_lens)
                true_label_log_probs = torch.gather(s_log_probs, dim=1, index= styles.unsqueeze(1)).squeeze(1)
                s_loss = -torch.mean(true_label_log_probs)*self.config['m2_loss_ratio']
                total_discri_loss += s_loss.item()

                # calculate gradients 
                self.optimizer_m2.zero_grad()
                s_loss.backward()
                self.optimizer_m2.step()
                # print message
                print(f'epoch: {epoch}, [{train_steps + 1}/{total_steps}],loss:{s_loss.item():.3f}', end='\r')
                # add to logger
                tag = self.config['tag']
                self.logger.scalar_summary(tag=f'{tag}/train/style_classifier_loss', 
                                           value=s_loss.item()/self.config['m2_loss_ratio'], 
                                           step=(epoch*(self.config['m2_train_freq'])+cnt)*total_steps+train_steps+1)
            print ()
            
        for train_steps, data in enumerate(self.train_lab_loader):
            bos = self.vocab['<BOS>']
            eos = self.vocab['<EOS>']
            pad = self.vocab['<PAD>']
            xs, ys, ys_in, ys_out, ilens, styles = to_gpu(data, bos, eos, pad)
            reverse_styles = styles.cpu().new_zeros(styles.size())
            for idx, ele in enumerate(styles.cpu().tolist()):
                if not(ele):
                    reverse_styles[idx]=1
            reverse_styles=cc(torch.LongTensor(reverse_styles))

            # Reconstruct Loss
            enc_outputs, enc_lens = self.encoder(xs, ilens)
            logits, log_probs, prediction, attns=\
                self.decoder(enc_outputs, enc_lens, styles, (ys_in, ys_out),
                             tf_rate=tf_rate, sample=False,
                             max_dec_timesteps=self.config['max_dec_timesteps'])

            loss = -torch.mean(log_probs)*self.config['m1_loss_ratio']
            total_loss += loss.item()
            
            # Adversarial Loss
            enc_representation = get_enc_context(enc_outputs, enc_lens)
            s_logits, s_log_probs, s_pred = self.s_classifier(enc_representation)
            true_label_log_probs = torch.gather(s_log_probs, dim=1, index= styles.unsqueeze(1)).squeeze(1)
            s_loss = torch.mean(true_label_log_probs)*self.config['m2_loss_ratio']            
            total_cheat_loss += s_loss.item()

            # Cycle Loss
            _,_,prediction,_ = self.decoder(enc_outputs, enc_lens, reverse_styles, None, 
                                            max_dec_timesteps=self.config['max_dec_timesteps'])
            fake_ilens = get_prediction_length(prediction, eos=self.vocab['<EOS>'])
            fake_enc_outputs, fake_enc_lens = self.encoder(prediction, fake_ilens)
            fake_logits, _, _, _=\
                self.decoder(fake_enc_outputs, fake_enc_lens, styles, (ys_in, ys_out),
                             tf_rate=tf_rate, sample=False,
                             max_dec_timesteps=self.config['max_dec_timesteps'])
            cycle_loss = -torch.mean(fake_logits)*self.config['cycle_loss_ratio']
            total_cycle_loss += cycle_loss.item()

            # calculate gradients 
            self.optimizer_m1.zero_grad()
            (loss+s_loss+cycle_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.params_m1, max_norm=self.config['max_grad_norm'])
            self.optimizer_m1.step()
            # print message
            print(f'epoch: {epoch}, [{train_steps + 1}/{total_steps}], loss: {loss.item():.3f},'
                  f'cycle_loss: {cycle_loss.item():.3f}, classifier_loss: {s_loss.item():.3f}', end='\r')
            # add to logger
            tag = self.config['tag']
            self.logger.scalar_summary(tag=f'{tag}/train/loss', 
                                       value=loss.item()/self.config['m1_loss_ratio'], 
                                       step=epoch * total_steps + train_steps + 1)
            self.logger.scalar_summary(tag=f'{tag}/train/adversarial_loss', 
                                       value=s_loss.item()/self.config['m2_loss_ratio'], 
                                       step=epoch * total_steps + train_steps + 1)
            self.logger.scalar_summary(tag=f'{tag}/train/cycle_loss', 
                                       value=loss.item()/self.config['cycle_loss_ratio'], 
                                       step=epoch * total_steps + train_steps + 1)
        print()
        return (total_loss/total_steps),(total_cheat_loss/total_steps),(total_cycle_loss/total_steps),\
            ((total_discri_loss/total_steps)/self.config['m2_train_freq'])

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
            avg_train_loss, avg_distri_loss, avg_cycle_loss, avg_discri_loss = self.train_one_epoch(epoch, tf_rate)
            # validation
            avg_valid_loss, wer, prediction_sents, ground_truth_sents = self.validation()

            print(f'Epoch: {epoch}, tf_rate={tf_rate:.3f}, train_loss={avg_train_loss:.4f},'
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
