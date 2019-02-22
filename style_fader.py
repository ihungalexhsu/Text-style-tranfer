import torch 
import torch.nn.functional as F
import numpy as np
from model import Encoder, Decoder, Domain_discri, Dense_classifier
from dataloader import get_data_loader
from dataset import PickleDataset
from utils import *
from utils import _seq_mask
from evaluation.calculate_bleu import BLEU
from evaluation.calculate_transfer import Transferability 
import yaml
import os
import pickle

class Style_transfer_fader(object):
    def __init__(self, config, alpha=10, load_model=False):
        self.config = config
        print(self.config)
        
        self.alpha = alpha
        
        # logger
        self.logger = Logger(config['logdir']+'_s'+str(alpha))

        # load vocab and non lang syms
        self.load_vocab()
       
        # get data loader
        self.get_data_loaders()

        # get label distribution
        self.get_label_dist()

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

    def get_label_dist(self):
        labelcount = np.zeros(len(self.vocab))
        for token_ids,_ in self.train_pos_dataset:
            for ind in token_ids:
                labelcount[ind] += 1.
        for token_ids,_ in self.train_neg_dataset:
            for ind in token_ids:
                labelcount[ind] += 1.

        labelcount[self.vocab['<EOS>']]+=\
            len(self.train_pos_dataset)+len(self.train_neg_dataset)
        labelcount[self.vocab['<PAD>']]=0
        labelcount[self.vocab['<BOS>']]=0
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
                                        ls_weight=self.config['ls_weight'],
                                        labeldist=labeldist))
        print(self.decoder)
        self.decoder.float()
        self.style_discri=\
            cc_model(Dense_classifier(input_dim=enc_out_dim,
                                      output_dim=self.config['n_style_type'],
                                      hidden_dim_vec=self.config['style_discri_hidden']))
        print(self.style_discri)
        self.style_discri.float()
        self.params_m1=list(self.encoder.parameters())+list(self.decoder.parameters())
        self.params_m2=list(self.style_discri.parameters())
        self.optimizer_m1 =\
            torch.optim.Adam(self.params_m1, 
                             lr=self.config['learning_rate_m1'],
                             betas=(0.5,0.9),
                             weight_decay=float(self.config['weight_decay_m1']))
        self.optimizer_m2 =\
            torch.optim.Adam(self.params_m2, 
                             lr=self.config['learning_rate_m2'], 
                             weight_decay=float(self.config['weight_decay_m2']))
        if load_model:
            self.load_model(self.config['load_model_path'], self.config['load_optimizer'])
        return

    def load_model(self, model_path, load_optimizer):
        model_path = model_path+"_s"+str(self.alpha) 
        print(f'Load model from {model_path}')
        encoder_path = model_path+'_encoder'
        decoder_path = model_path+'_decoder'
        style_discri_path = model_path+'_sdis'
        opt1_path = model_path+'_opt1'
        opt2_path = model_path+'_opt2'
        self.encoder.load_state_dict(torch.load(f'{encoder_path}.ckpt'))
        self.decoder.load_state_dict(torch.load(f'{decoder_path}.ckpt'))
        self.style_discri.load_state_dict(torch.load(f'{style_discri_path}.ckpt'))
        if load_optimizer:
            print(f'Load optmizer from {model_path}')
            self.optimizer_m1.load_state_dict(torch.load(f'{opt1_path}.opt'))
            self.optimizer_m2.load_state_dict(torch.load(f'{opt2_path}.opt'))
            if self.config['adjust_lr']:
                adjust_learning_rate(self.optimizer_m1, self.config['retrieve_lr_m1']) 
                adjust_learning_rate(self.optimizer_m2, self.config['retrieve_lr_m2']) 
        return

    def save_model(self, model_path):
        model_path = model_path+"_s"+str(self.alpha) 
        encoder_path = model_path+'_encoder'
        decoder_path = model_path+'_decoder'
        style_discri_path = model_path+'_sdis'
        opt1_path = model_path+'_opt1'
        opt2_path = model_path+'_opt2'
        torch.save(self.encoder.state_dict(), f'{encoder_path}.ckpt')
        torch.save(self.decoder.state_dict(), f'{decoder_path}.ckpt')
        torch.save(self.style_discri.state_dict(), f'{style_discri_path}.ckpt')
        torch.save(self.optimizer_m1.state_dict(), f'{opt1_path}.opt')
        torch.save(self.optimizer_m2.state_dict(), f'{opt2_path}.opt')
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
            loss_recon, loss_adv, loss_discri = self.train_one_epoch(epoch, tf_rate)
            
            # save model in every epoch
            if not os.path.exists(self.config['model_dir']):
                os.makedirs(self.config['model_dir'])
            model_path = os.path.join(self.config['model_dir'], self.config['model_name'])
            self.save_model(f'{model_path}-{epoch:03d}')
            self.save_model(f'{model_path}_latest')
            
            # validation
            val_loss, transfer_acc, selfbleu = self.validation()
            print(f'epoch:{epoch}, tf_rate:{tf_rate:.3f}, loss_recon:{loss_recon:.4f}, '
                  f'loss_adver:{loss_adv:.4f}, loss_discri:{loss_discri:.4f}, '
                  f'val_loss:{val_loss:.4f}, transfer_acc:{transfer_acc:.4f}, '
                  f'self_bleu:{selfbleu:.4f}')
            # add to tensorboard
            tag = self.config['tag']
            self.logger.scalar_summary(f'{tag}/val/selfbleu', selfbleu, epoch)
            self.logger.scalar_summary(f'{tag}/val/val_loss', val_loss, epoch)
            self.logger.scalar_summary(f'{tag}/val/transfer_acc', transfer_acc, epoch)

            score = selfbleu*transfer_acc
            # save best
            if score > best_score: 
                model_path = os.path.join(self.config['model_dir'], self.config['model_name']+'_best')
                best_score = score
                self.save_model(model_path)
                best_model_enc = self.encoder.state_dict()
                best_model_dec = self.decoder.state_dict()
                print(f'Save #{epoch} model, val_loss={val_loss:.4f}, Score={score:.4f}')
                print('-----------------')
                early_stop_counter=0
            if epoch >= self.config['early_stop_start_epoch']:
                early_stop_counter += 1
                if early_stop_counter > self.config['early_stop_patience']:
                    break
        best_model = (best_model_enc, best_model_dec)
        print('-------finish training--------')
        print(f'--get best score: {best_score:.4f}--')
        return best_model, best_score
    
    def _random_target(self, x, num_class=2):
        out = cc(torch.LongTensor(x.size()).random_(0,num_class))
        return out

    def _train_discriminator(self, data):
        bos = self.vocab['<BOS>']
        eos = self.vocab['<EOS>']
        pad = self.vocab['<PAD>']
        xs, ys, ys_in, ys_out, ilens, styles = to_gpu(data, bos, eos, pad)
        enc_outputs, enc_lens = self.encoder(xs, ilens)
        enc_representation = get_enc_context(enc_outputs, enc_lens)
        _, s_log_probs, _ = self.style_discri(enc_representation)
        true_s_log_probs = torch.gather(s_log_probs,dim=1,
                                        index=styles.unsqueeze(1)).squeeze(1)
        loss = -torch.mean(true_s_log_probs)
        return loss

    def _log_discriminator(self, epoch, train_steps, total_steps, loss, 
                           loss_name, log_steps):
        print(f'epoch: {epoch}, [{train_steps + 1}/{total_steps}], '
              f'{loss_name}: {loss:.3f},', end='\r')
        # add to logger
        tag = self.config['tag']
        self.logger.scalar_summary(tag=f'{tag}/train/{loss_name}', 
                                   value=loss, 
                                   step=log_steps)
        return
    
    def _train_m2_onetime(self, data, total_discri_loss, epoch, train_steps,
                          total_steps, cnt):
        discri_loss = self._train_discriminator(data)            
        total_discri_loss += discri_loss.item()           
        # calculate gradients 
        self.optimizer_m2.zero_grad()
        discri_loss.backward()
        self.optimizer_m2.step()
        # print message
        self._log_discriminator(epoch, train_steps, total_steps,
                                discri_loss.item(), "discri_loss",
                                (epoch*(self.config['m2_train_freq'])+cnt)*total_steps+train_steps+1)
        train_steps+=1
        return train_steps, total_discri_loss

    def train_one_epoch(self, epoch, tf_rate):
        total_steps = len(self.train_pos_loader)*2
        total_loss = 0.
        total_discri_loss = 0.
        total_adv_discri_loss = 0.
        assert len(self.train_pos_loader) >= len(self.train_neg_loader)
        for cnt in range(self.config['m2_train_freq']):
            pos_data_iterator = iter(self.train_pos_loader)
            neg_data_iterator = iter(self.train_neg_loader)
            train_steps = 0
            for i in range(len(self.train_pos_loader)):
                try:
                    data = next(pos_data_iterator)
                except StopIteration:
                    print('StopIteration in pos part')
                    pass
                train_steps, total_discri_loss =\
                    self._train_m2_onetime(data, total_discri_loss, epoch, 
                                           train_steps, total_steps, cnt)
                try:
                    data = next(neg_data_iterator)
                except StopIteration:
                    neg_data_iterator = iter(self.train_neg_loader)
                    data = next(neg_data_iterator)

                train_steps, total_discri_loss =\
                    self._train_m2_onetime(data, total_discri_loss, epoch, 
                                           train_steps, total_steps, cnt)
            print ()
       
        pos_data_iterator = iter(self.train_pos_loader)
        neg_data_iterator = iter(self.train_neg_loader)
        train_steps = 0
        for i in range(len(self.train_pos_loader)):
            try:
                data = next(pos_data_iterator)
            except StopIteration:
                print('StopIteration in pos part')
                pass
            train_steps, total_loss, total_adv_discri_loss =\
                self._train_m1_onetime(data, tf_rate, total_loss,
                                       total_adv_discri_loss, epoch, train_steps,
                                       total_steps)
            try:
                data = next(neg_data_iterator)
            except StopIteration:
                neg_data_iterator = iter(self.train_neg_loader)
                data = next(neg_data_iterator)
            train_steps, total_loss, total_adv_discri_loss =\
                self._train_m1_onetime(data, tf_rate, total_loss,
                                       total_adv_discri_loss, epoch, train_steps,
                                       total_steps)

        print()
        return ((total_loss/total_steps),(total_adv_discri_loss/total_steps),\
                (total_discri_loss/total_steps)/self.config['m2_train_freq'])
    
    def _train_m1_onetime(self, data, tf_rate, total_loss, total_adv_discri_loss,
                          epoch, train_steps, total_steps):
        
        loss, adv_loss = self._train_reconstructor(data, tf_rate)
        total_loss += loss.item()
        total_adv_discri_loss += adv_loss.item()
        # calculate gradients 
        self.optimizer_m1.zero_grad()
        tloss = loss+adv_loss
        tloss.backward()
        torch.nn.utils.clip_grad_norm_(self.params_m1, max_norm=self.config['max_grad_norm'])
        self.optimizer_m1.step()
        self._log_reconstructor(epoch, train_steps, total_steps, loss.item(),
                                adv_loss.item(), "adv_dis_loss",
                                epoch*total_steps+train_steps+1)
        train_steps +=1
        return train_steps, total_loss, total_adv_discri_loss

    def _train_reconstructor(self, data, tf_rate):
        bos = self.vocab['<BOS>']
        eos = self.vocab['<EOS>']
        pad = self.vocab['<PAD>']
        xs, ys, ys_in, ys_out, ilens, styles = to_gpu(data, bos, eos, pad)
        enc_outputs, enc_lens = self.encoder(xs, ilens)
        _, recon_log_probs, _, _=\
            self.decoder(enc_outputs, enc_lens, styles, (ys_in, ys_out),
                         tf_rate=tf_rate, sample=False,
                         max_dec_timesteps=self.config['max_dec_timesteps'])

        loss = -torch.mean(recon_log_probs)
        
        # Adversarial Loss
        enc_representation = get_enc_context(enc_outputs, enc_lens)
        _, s_log_probs, _= self.style_discri(enc_representation)
        random_s_log_probs =\
            torch.gather(s_log_probs,dim=1,
                         index=self._random_target(styles, self.config['n_style_type']).unsqueeze(1)).squeeze(1)
        s_loss = -torch.mean(random_s_log_probs)*self.alpha
        return loss, s_loss

    def _log_reconstructor(self, epoch, train_steps, total_steps, recons_loss, 
                           adv_loss, adv_loss_name, log_steps):
        
        print(f'epoch: {epoch}, [{train_steps+1}/{total_steps}], loss: {recons_loss:.3f}, '
              f'{adv_loss_name}: {adv_loss:.3f},', end='\r')
        
        tag = self.config['tag']
        self.logger.scalar_summary(tag=f'{tag}/train/reconloss', 
                                   value=recons_loss, 
                                   step=log_steps)
        self.logger.scalar_summary(tag=f'{tag}/train/{adv_loss_name}', 
                                   value=adv_loss/self.alpha, 
                                   step=log_steps)
        return
    
    def _valid_feed_data(self, dataloader, total_loss):
        all_prediction = []
        all_inputs = []
        for step, data in enumerate(dataloader):
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
            seq_len = [y.size(0) + 1 for y in ys]
            mask = cc(_seq_mask(seq_len=seq_len, max_len=log_probs.size(1)))
            loss = (-torch.sum(log_probs*mask))/sum(seq_len)
            total_loss += loss.item()

            all_prediction = all_prediction + prediction.cpu().tolist()
            all_inputs = all_inputs + [y.cpu().tolist() for y in ys]
        
        return all_prediction, all_inputs, total_loss       

    def validation(self):
        self.encoder.eval()
        self.decoder.eval()
        total_loss = 0.
        # positive input
        posdata_pred, posdata_input, total_loss =\
            self._valid_feed_data(self.dev_pos_loader, total_loss)
        # get sentence
        posdata_pred, posdata_input = self.idx2sent(posdata_pred, posdata_input)
        # write file
        if not os.path.exists(self.config['dev_file_path']):
            os.makedirs(self.config['dev_file_path'])
        file_path_pos = os.path.join(self.config['dev_file_path'], 
                                     f'pred.fader.s{str(self.alpha)}.dev.0.temp')
        file_path_gtpos = os.path.join(self.config['dev_file_path'], 
                                       f'gf.fader.s{str(self.alpha)}.dev.1.temp')
        writefile(posdata_pred, file_path_pos)
        writefile(posdata_input, file_path_gtpos)
        # negative input
        posdata_pred, posdata_input, total_loss =\
            self._valid_feed_data(self.dev_neg_loader, total_loss)
        # get sentence
        posdata_pred, posdata_input = self.idx2sent(posdata_pred, posdata_input)
        # write file
        file_path_neg = os.path.join(self.config['dev_file_path'], 
                                     f'pred.fader.s{str(self.alpha)}.dev.1.temp')
        file_path_gtneg = os.path.join(self.config['dev_file_path'], 
                                       f'gf.fader.s{str(self.alpha)}.dev.0.temp')
        writefile(posdata_pred, file_path_neg)
        writefile(posdata_input, file_path_gtneg)
        self.encoder.train()
        self.decoder.train()
        
        # evaluation
        avg_loss = total_loss / (len(self.dev_pos_loader)+len(self.dev_neg_loader))
        pos_acc = Transferability(file_path_pos, 
                                  self.config['style_classifier_path'],
                                  '__label__0')
        neg_acc = Transferability(file_path_neg, 
                                  self.config['style_classifier_path'],
                                  '__label__1')
        avg_acc = (pos_acc+neg_acc)/2
        selfbleu_pos, _ = BLEU(file_path_pos, file_path_gtpos)
        selfbleu_neg, _ = BLEU(file_path_neg, file_path_gtneg)
        avg_selfbleu = (selfbleu_pos + selfbleu_neg)/2
        return avg_loss, avg_acc, avg_selfbleu
    
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
        test_pos_loader = get_data_loader(test_pos_dataset, 
                                          batch_size=2, 
                                          shuffle=False)
        test_neg_dataset = PickleDataset(os.path.join(root_dir, f'{test_neg_set}.p'), 
                                         config=None, sort=False)
        test_neg_loader = get_data_loader(test_neg_dataset, 
                                          batch_size=2, 
                                          shuffle=False)
        self.encoder.eval()
        self.decoder.eval()
        total_loss = 0.
        # positive input
        posdata_pred, posdata_input, total_loss =\
            self._valid_feed_data(test_pos_loader, total_loss)
        # get sentence
        posdata_pred, posdata_input = self.idx2sent(posdata_pred, posdata_input)
        # write file
        if not os.path.exists(self.config['test_file_path']):
            os.makedirs(self.config['test_file_path'])
        file_path_pos = os.path.join(self.config['test_file_path'], 
                                     f'pred.fader.s{str(self.alpha)}.test.0(input1)')
        file_path_gtpos = os.path.join(self.config['test_file_path'], 
                                       f'gt.fader.s{str(self.alpha)}.test.input1')
        writefile(posdata_pred, file_path_pos)
        writefile(posdata_input, file_path_gtpos)
        # negative input
        posdata_pred, posdata_input, total_loss =\
            self._valid_feed_data(test_neg_loader, total_loss)
        # get sentence
        posdata_pred, posdata_input = self.idx2sent(posdata_pred, posdata_input)
        # write file
        file_path_neg = os.path.join(self.config['test_file_path'], 
                                     f'pred.fader.s{str(self.alpha)}.test.1(input0)')
        file_path_gtneg = os.path.join(self.config['test_file_path'], 
                                       f'gf.fader.s{str(self.alpha)}.test.input0')
        writefile(posdata_pred, file_path_neg)
        writefile(posdata_input, file_path_gtneg)       
        self.encoder.train()
        self.decoder.train()
        # evaluation
        avg_loss = total_loss / (len(test_pos_loader)+len(test_neg_loader))
        pos_acc = Transferability(file_path_pos, 
                                  self.config['style_classifier_path'],
                                  '__label__0')
        neg_acc = Transferability(file_path_neg, 
                                  self.config['style_classifier_path'],
                                  '__label__1')
        avg_acc = (pos_acc+neg_acc)/2
        selfbleu_pos, _ = BLEU(file_path_pos, file_path_gtpos)
        selfbleu_neg, _ = BLEU(file_path_neg, file_path_gtneg)
        avg_selfbleu = (selfbleu_pos + selfbleu_neg)/2
        print(f'Average style accuracy: {avg_acc:.4f}')
        print(f'Average self bleu score: {avg_selfbleu:.4f}')
        bleu_pos, _ = BLEU(file_path_pos, self.config['humanref_path_pos']) 
        bleu_neg, _ = BLEU(file_path_neg, self.config['humanref_path_neg']) 
        avg_bleu = (bleu_pos + bleu_neg)/2
        print(f'Average bleu score comparing with human: {avg_bleu:.4f}')
        return avg_acc*avg_selfbleu
