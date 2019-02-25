import torch 
import torch.nn.functional as F
import numpy as np
from model import Encoder, Decoder, Domain_discri, Dense_classifier, dotAttn, BiRNN_discri, DenseNet
from dataloader import get_data_loader
from dataset import PickleDataset
from utils import *
from utils import _seq_mask
from evaluation.calculate_bleu import BLEU
from evaluation.calculate_transfer import Transferability 
import yaml
import os
import pickle

class Style_proposed_att_adver(object):
    def __init__(self, config, alpha=1, beta=1, gamma=100, delta=10, 
                 zeta=10, load_model=False):
        self.config = config
        print(self.config)
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.zeta = zeta

        # logger
        self.logger = Logger(config['logdir']+'_a'+str(alpha)+'b'+str(beta)+'g'+\
                             str(gamma)+'d'+str(delta)+'z'+str(zeta))

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

        self.encS = cc_model(Encoder(vocab_size=len(self.vocab),
                                     embedding_dim=self.config['embedding_dim'],
                                     hidden_dim=self.config['encS_hidden_dim'],
                                     n_layers=self.config['encS_n_layers'],
                                     dropout_rate=self.config['enc_dropout_p'],
                                     pad_idx=self.vocab['<PAD>'],
                                     bidirectional=self.config['bidir_enc'],
                                     pre_embedding=pretrain_w2v,
                                     update_embedding=self.config['update_embedding']))
        print(self.encS)
        self.encS.float()
        self.encC = cc_model(Encoder(vocab_size=len(self.vocab),
                                     embedding_dim=self.config['embedding_dim'],
                                     hidden_dim=self.config['encC_hidden_dim'],
                                     n_layers=self.config['encC_n_layers'],
                                     dropout_rate=self.config['enc_dropout_p'],
                                     pad_idx=self.vocab['<PAD>'],
                                     bidirectional=self.config['bidir_enc'],
                                     pre_embedding=pretrain_w2v,
                                     update_embedding=self.config['update_embedding']))
        print(self.encC)
        self.encC.float()
        if self.config['bidir_enc']:
            encS_out_dim=2*self.config['encS_hidden_dim']
            encC_out_dim=2*self.config['encC_hidden_dim']
        else:
            encS_out_dim=self.config['encS_hidden_dim']
            encC_out_dim=self.config['encC_hidden_dim']
        self.s_classifier = cc_model(Dense_classifier(input_dim=encS_out_dim,
                                                      output_dim=self.config['n_style_type'],
                                                      hidden_dim_vec=self.config['style_cls_hidden']))
        print(self.s_classifier)
        self.s_classifier.float()
        self.disenC = cc_model(BiRNN_discri(input_dim=encC_out_dim,
                                            rnn_hidden_dim=self.config['disen_c_hidden_dim'],
                                            dropout_rate=0.1,
                                            dnn_hidden_dim=self.config['disen_c_dnn_dim'],
                                            output_dim=encS_out_dim))
        print(self.disenC)
        self.disenC.float()
        self.disenS = cc_model(Decoder(output_dim=len(self.vocab),
                                       embedding_dim=self.config['embedding_dim'],
                                       hidden_dim=self.config['disen_s_hidden_dim'],
                                       dropout_rate=0.1,
                                       bos=self.vocab['<BOS>'],
                                       eos=self.vocab['<EOS>'],
                                       pad=self.vocab['<PAD>'],
                                       enc_out_dim=encS_out_dim,
                                       n_styles=self.config['n_style_type'],
                                       use_enc_init=self.config['use_enc_init'],
                                       use_style_embedding=False,
                                       give_context_directly=True))
        print(self.disenS)
        self.disenS.float()

        '''
        self.style_mimicker = cc_model(DenseNet(input_dim=(encC_out_dim+1),
                                                output_dim=encS_out_dim,
                                                hidden_dim_vec=self.config['mimicker_hidden_dim']))
        '''
        self.style_mimicker = cc_model(DenseNet(input_dim=1,
                                                output_dim=encS_out_dim,
                                                hidden_dim_vec=self.config['mimicker_hidden_dim']))
        print(self.style_mimicker)
        self.style_mimicker.float()
        self.attention = cc_model(dotAttn(query_dim=self.config['dec_hidden_dim'],
                                          key_dim=encC_out_dim,
                                          att_dim=self.config['attention_dim']))
        print(self.attention)
        self.attention.float()
        self.decoder = cc_model(Decoder(output_dim=len(self.vocab),
                                        embedding_dim=self.config['embedding_dim'],
                                        hidden_dim=self.config['dec_hidden_dim'],
                                        dropout_rate=self.config['dec_dropout_p'],
                                        bos=self.vocab['<BOS>'],
                                        eos=self.vocab['<EOS>'],
                                        pad=self.vocab['<PAD>'],
                                        enc_out_dim=encC_out_dim,
                                        n_styles=self.config['n_style_type'],
                                        style_emb_dim=encS_out_dim,
                                        use_enc_init=self.config['use_enc_init'],
                                        use_attention=True,
                                        attention=self.attention,
                                        use_style_embedding=False,
                                        ls_weight=self.config['ls_weight'],
                                        labeldist=labeldist,
                                        give_context_directly=False,
                                        give_style_repre_directly=True))
        print(self.decoder)
        self.decoder.float()
        self.pos_domain_discri =\
            cc_model(Domain_discri(vocab_size=len(self.vocab),
                                   embedding_dim=self.config['embedding_dim'],
                                   rnn_hidden_dim=self.config['domain_discri_dim'],
                                   dropout_rate=self.config['enc_dropout_p'],
                                   dnn_hidden_dim=self.config['domain_discri_dim'],
                                   pad_idx=self.vocab['<PAD>'],
                                   pre_embedding=pretrain_w2v,
                                   update_embedding=self.config['update_embedding']))
        print(self.pos_domain_discri)
        self.pos_domain_discri.float()
        self.neg_domain_discri =\
            cc_model(Domain_discri(vocab_size=len(self.vocab),
                                   embedding_dim=self.config['embedding_dim'],
                                   rnn_hidden_dim=self.config['domain_discri_dim'],
                                   dropout_rate=self.config['enc_dropout_p'],
                                   dnn_hidden_dim=self.config['domain_discri_dim'],
                                   pad_idx=self.vocab['<PAD>'],
                                   pre_embedding=pretrain_w2v,
                                   update_embedding=self.config['update_embedding']))
        print(self.neg_domain_discri)
        self.neg_domain_discri.float()
        self.params_m1=list(self.encS.parameters())+list(self.encC.parameters())+\
            list(self.s_classifier.parameters())+list(self.style_mimicker.parameters())+\
            list(self.decoder.parameters())+list(self.attention.parameters())
        self.params_m2=list(self.disenC.parameters())+list(self.disenS.parameters())
        self.params_m3=list(self.pos_domain_discri.parameters())+\
            list(self.neg_domain_discri.parameters())
        self.optimizer_m1 =\
            torch.optim.Adam(self.params_m1, 
                             lr=self.config['learning_rate_m1'],
                             weight_decay=float(self.config['weight_decay_m1']))
        self.optimizer_m2 =\
            torch.optim.Adam(self.params_m2, 
                             lr=self.config['learning_rate_m2'], 
                             weight_decay=float(self.config['weight_decay_m2']))
        self.optimizer_m3 =\
            torch.optim.Adam(self.params_m3, 
                             lr=self.config['learning_rate_m3'], 
                             weight_decay=float(self.config['weight_decay_m3']))
        if load_model:
            self.load_model(self.config['load_model_path'], self.config['load_optimizer'])
        return
 
    def load_model(self, model_path, load_optimizer):
        model_path = (model_path+'_a'+str(self.alpha)+'b'+str(self.beta)+'g'+\
            str(self.gamma)+'d'+str(self.delta)+'z'+str(self.zeta))
        if os.path.exists(model_path+'_encS.ckpt'):
            print(f'Load model from {model_path}')
            encS_path = model_path+'_encS'
            encC_path = model_path+'_encC'
            attention_path = model_path+'_att'
            decoder_path = model_path+'_decoder'
            s_classifier_path = model_path+'_scls'
            style_mimicker_path = model_path+'_smimic'
            disenC_path = model_path+'_disenC'
            disenS_path = model_path+'_disenS'
            pos_discri_path = model_path+'_posdiscri'
            neg_discri_path = model_path+'_negdiscri'
            #mean_style_pos_path = model_path+'_stylevec_pos'
            #mean_style_neg_path = model_path+'_stylevec_neg'
            opt1_path = model_path+'_opt1'
            opt2_path = model_path+'_opt2'
            opt3_path = model_path+'_opt3'
            self.encS.load_state_dict(torch.load(f'{encS_path}.ckpt'))
            self.encC.load_state_dict(torch.load(f'{encC_path}.ckpt'))
            self.decoder.load_state_dict(torch.load(f'{decoder_path}.ckpt'))
            self.attention.load_state_dict(torch.load(f'{attention_path}.ckpt'))
            self.s_classifier.load_state_dict(torch.load(f'{s_classifier_path}.ckpt'))
            self.style_mimicker.load_state_dict(torch.load(f'{style_mimicker_path}.ckpt'))
            self.disenC.load_state_dict(torch.load(f'{disenC_path}.ckpt'))
            self.pos_domain_discri.load_state_dict(torch.load(f'{pos_discri_path}.ckpt'))
            self.neg_domain_discri.load_state_dict(torch.load(f'{neg_discri_path}.ckpt'))
            #self.mean_style_pos = torch.load(f'{mean_style_pos_path}.pt')
            #self.mean_style_neg = torch.load(f'{mean_style_neg_path}.pt')
            if load_optimizer:
                print(f'Load optmizer from {model_path}')
                self.optimizer_m1.load_state_dict(torch.load(f'{opt1_path}.opt'))
                self.optimizer_m2.load_state_dict(torch.load(f'{opt2_path}.opt'))
                self.optimizer_m3.load_state_dict(torch.load(f'{opt3_path}.opt'))
                if self.config['adjust_lr']:
                    adjust_learning_rate(self.optimizer_m1, self.config['retrieve_lr_m1']) 
                    adjust_learning_rate(self.optimizer_m2, self.config['retrieve_lr_m2']) 
                    adjust_learning_rate(self.optimizer_m3, self.config['retrieve_lr_m3']) 
        return
  
    def save_model(self, model_path):
        model_path = (model_path+'_a'+str(self.alpha)+'b'+str(self.beta)+'g'+\
            str(self.gamma)+'d'+str(self.delta)+'z'+str(self.zeta))
        encS_path = model_path+'_encS'
        encC_path = model_path+'_encC'
        decoder_path = model_path+'_decoder'
        attention_path = model_path+'_att'
        s_classifier_path = model_path+'_scls'
        style_mimicker_path = model_path+'_smimic'
        disenC_path = model_path+'_disenC'
        disenS_path = model_path+'_disenS'
        pos_discri_path = model_path+'_posdiscri'
        neg_discri_path = model_path+'_negdiscri'
        #mean_style_pos_path = model_path+'_stylevec_pos'
        #mean_style_neg_path = model_path+'_stylevec_neg'
        opt1_path = model_path+'_opt1'
        opt2_path = model_path+'_opt2'
        opt3_path = model_path+'_opt3'
        torch.save(self.encS.state_dict(), f'{encS_path}.ckpt')
        torch.save(self.encC.state_dict(), f'{encC_path}.ckpt')
        torch.save(self.decoder.state_dict(), f'{decoder_path}.ckpt')
        torch.save(self.attention.state_dict(), f'{attention_path}.ckpt')
        torch.save(self.s_classifier.state_dict(), f'{s_classifier_path}.ckpt')
        torch.save(self.style_mimicker.state_dict(), f'{style_mimicker_path}.ckpt')
        torch.save(self.disenC.state_dict(), f'{disenC_path}.ckpt')
        torch.save(self.disenS.state_dict(), f'{disenS_path}.ckpt')
        torch.save(self.pos_domain_discri.state_dict(), f'{pos_discri_path}.ckpt')
        torch.save(self.neg_domain_discri.state_dict(), f'{neg_discri_path}.ckpt')
        #torch.save(self.mean_style_pos, f'{mean_style_pos_path}.pt')
        #torch.save(self.mean_style_neg, f'{mean_style_neg_path}.pt')
        torch.save(self.optimizer_m1.state_dict(), f'{opt1_path}.opt')
        torch.save(self.optimizer_m2.state_dict(), f'{opt2_path}.opt')
        torch.save(self.optimizer_m3.state_dict(), f'{opt3_path}.opt')
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
            self.train_one_epoch(epoch, tf_rate)
            # save model in every epoch
            if not os.path.exists(self.config['model_dir']):
                os.makedirs(self.config['model_dir'])
            model_path = os.path.join(self.config['model_dir'], self.config['model_name'])
            self.save_model(f'{model_path}-{epoch:03d}')
            self.save_model(f'{model_path}_latest')
            
            # validation
            val_loss, transfer_acc, selfbleu = self.validation()
            print(f'epoch: {epoch}, tf_rate: {tf_rate:.3f}, val_loss: {val_loss:.4f}, '
                  f'transfer_acc: {transfer_acc:.4f}, self_bleu: {selfbleu:.4f}')
            # add to tensorboard
            tag = self.config['tag']
            self.logger.scalar_summary(f'{tag}/val/selfbleu', selfbleu, epoch)
            self.logger.scalar_summary(f'{tag}/val/val_loss', val_loss, epoch)
            self.logger.scalar_summary(f'{tag}/val/transfer_acc', transfer_acc, epoch)
            score = selfbleu*transfer_acc
            # save best
            if score > best_score: 
                # save model
                model_path = os.path.join(self.config['model_dir'], self.config['model_name']+'_best')
                best_score = score
                self.save_model(model_path)
                best_model_enc = self.encC.state_dict()
                best_model_dec = self.decoder.state_dict()
                bset_model_att = self.attention.state_dict()
                best_model_mimicker = self.style_mimicker.state_dict()
                print(f'Save #{epoch} model, val_loss={val_loss:.4f}, score={score:.4f}')
                print('-----------------')
                early_stop_counter=0
            if epoch >= self.config['early_stop_start_epoch']:
                early_stop_counter += 1
                if early_stop_counter > self.config['early_stop_patience']:
                    break
        best_model = (best_model_enc, best_model_dec, best_model_att, best_model_mimicker)
        print('---------------finish training----------------')
        print(f'-----get best score: {best_score:.4f}------')
        return best_model, best_score
    
    def _random_target(self, x, num_class=2):
        out = cc(torch.LongTensor(x.size()).random_(0,num_class))
        return out

    def _random_vector(self, x, embedding_activation):
        range_low = -1 if embedding_activation == 'tanh' else 0
        range_high = 1
        tfake = torch.FloatTensor(x.size()).uniform_(range_low, range_high)
        return cc(tfake)
    
    def _get_reverse_style(self, styles):
        reverse_styles = styles.cpu().new_zeros(styles.size())
        for idx, ele in enumerate(styles.cpu().tolist()):
            if not(ele):
                reverse_styles[idx]=1
        reverse_styles=cc(torch.LongTensor(reverse_styles))
        return reverse_styles
    
    def _get_decoder_pred(self, want_styles, content_outputs, content_lens, 
                          tf_rate=1.0, ground_truth=None):
        '''
        mimicked_style_vector =\
            self.style_mimicker(torch.cat([want_styles.float().view(-1,1),
                                           content_vector],dim=1))
        '''
        mimicked_style_vector = self.style_mimicker(want_styles.float().view(-1,1))
        _, recon_log_probs, predicts, _ =\
            self.decoder(content_outputs, content_lens, mimicked_style_vector, 
                         ground_truth, tf_rate=tf_rate, 
                         max_dec_timesteps=self.config['max_dec_timesteps'])
        pred_ilens = get_prediction_length(predicts, eos=self.vocab['<EOS>'])
        return recon_log_probs, predicts, pred_ilens, mimicked_style_vector
    
    def _pass2encs(self, xs, ilens):
        enc_outputs, enc_lens = self.encS(xs, ilens)
        style_vector = get_enc_context(enc_outputs, enc_lens)
        content_outputs, content_lens = self.encC(xs, ilens)
        return style_vector, content_outputs, content_lens

    def _train_domain_discri(self, xs, ilens, predicts, pred_ilens,
                             ori_domain_dis, opposite_domain_dis):
        # For real and matched sentiment data, let discri learn True
        _, RM_log_probs, _ = ori_domain_dis(xs, ilens)
        # For fake and non-matched sentiment data , let discri learn False               
        _, FNM_log_probs, _ = ori_domain_dis(predicts, pred_ilens, need_sort=True)
        # For fake and matched sentiment data , let discri learn False               
        _, FM_log_probs, _ = opposite_domain_dis(predicts, pred_ilens, need_sort=True)
        # For real but non-matched sentiment data, let discri learn False
        _, RNM_log_probs, _ = opposite_domain_dis(xs, ilens)
        return RM_log_probs, FNM_log_probs, FM_log_probs, RNM_log_probs

    def _get_m2_loss(self, data, sty_type):
        bos = self.vocab['<BOS>']
        eos = self.vocab['<EOS>']
        pad = self.vocab['<PAD>']
        xs, ys, ys_in, ys_out, ilens, styles = to_gpu(data, bos, eos, pad)
        reverse_styles = self._get_reverse_style(styles)
        # disentenglement
        style_vector, content_outputs, content_lens = self._pass2encs(xs, ilens)
        predict_style = self.disenC(content_outputs.detach(), content_lens.detach())
        _, disenS_log_probs, _, _ = self.disenS(style_vector.detach(), ilens, None, 
                                                (ys_in, ys_out), tf_rate=1.0,
                                                max_dec_timesteps=self.config['max_dec_timesteps'])
        # calculate loss
        loss_disencC = torch.mean((predict_style-style_vector.detach())**2)
        loss_disencS = -torch.mean(disenS_log_probs)
        if self.delta > 0:
            # domain discriminator
            _, predicts, pred_ilens,_ = self._get_decoder_pred(reverse_styles, 
                                                               content_outputs,
                                                               content_lens)
            if sty_type == 'pos':
                RM_log_probs, FNM_log_probs, FM_log_probs, RNM_log_probs =\
                    self._train_domain_discri(xs, ilens, predicts.detach(), pred_ilens.detach(),
                                              self.pos_domain_discri, self.neg_domain_discri)
            else:
                RM_log_probs, FNM_log_probs, FM_log_probs, RNM_log_probs =\
                    self._train_domain_discri(xs, ilens, predicts.detach(), pred_ilens.detach(),
                                              self.neg_domain_discri, self.pos_domain_discri)
            all_one = cc(styles.cpu().new_ones(styles.size()))
            all_zero = cc(styles.cpu().new_zeros(styles.size()))
            # calculate loss
            RM_log_probs = torch.gather(RM_log_probs,dim=1,
                                        index=all_one.unsqueeze(1)).squeeze(1) 
            FNM_log_probs = torch.gather(FNM_log_probs,dim=1,
                                         index=all_zero.unsqueeze(1)).squeeze(1) 
            FM_log_probs = torch.gather(FM_log_probs,dim=1,
                                        index=all_zero.unsqueeze(1)).squeeze(1) 
            RNM_log_probs = torch.gather(RNM_log_probs,dim=1,
                                         index=all_zero.unsqueeze(1)).squeeze(1)
            RM_dis_loss = -torch.mean(RM_log_probs)
            FNM_dis_loss = -torch.mean(FNM_log_probs)
            FM_dis_loss = -torch.mean(FM_log_probs)
            RNM_dis_loss = -torch.mean(RNM_log_probs)
        else:
            RM_dis_loss = torch.Tensor([0])
            FNM_dis_loss =  torch.Tensor([0])
            FM_dis_loss =  torch.Tensor([0])
            RNM_dis_loss =  torch.Tensor([0])

        return loss_disencC, loss_disencS, RM_dis_loss, FNM_dis_loss, FM_dis_loss, RNM_dis_loss
        
    def _log_m2(self, epoch, train_steps, total_steps, l_posdiscri, l_negdiscri,
                l_disencC, l_disencS, log_steps):
        print(f'epoch: {epoch}, [{train_steps + 1}/{total_steps}], '
              f'disencC: {l_disencC:.3f}, disencS: {l_disencS:.3f}, '
              f'domain_dis_pos: {l_posdiscri:.3f}, '
              f'domain_dis_neg: {l_negdiscri:.3f}', end='\r')
        # add to logger
        tag = self.config['tag']
        self.logger.scalar_summary(tag=f'{tag}/train/disen_encC_loss', 
                                   value=l_disencC, 
                                   step=log_steps)
        self.logger.scalar_summary(tag=f'{tag}/train/disen_encS_loss', 
                                   value=l_disencS, 
                                   step=log_steps)
        if self.delta > 0.:
            self.logger.scalar_summary(tag=f'{tag}/train/domain_discri_pos_loss', 
                                       value=l_posdiscri, 
                                       step=log_steps)
            self.logger.scalar_summary(tag=f'{tag}/train/domain_discri_neg_loss', 
                                       value=l_negdiscri, 
                                       step=log_steps)
        return

    def _train_m2_onetime(self, data, total_disencC, total_disencS, 
                          total_posdiscri, total_negdiscri, epoch, 
                          train_steps, total_steps, cnt, sty_type):
        l_disencC, l_disencS, RM_dis_loss, FNM_dis_loss, FM_dis_loss, RNM_dis_loss =\
            self._get_m2_loss(data, sty_type)
        # calculate gradients 
        self.optimizer_m2.zero_grad()
        (l_disencC+l_disencS).backward()
        torch.nn.utils.clip_grad_norm_(self.params_m2,
                                       max_norm=self.config['max_grad_norm'])
        self.optimizer_m2.step()
        if self.delta > 0:
            self.optimizer_m3.zero_grad()
            (RM_dis_loss+FNM_dis_loss+FM_dis_loss+RNM_dis_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.params_m3, 
                                           max_norm=self.config['max_grad_norm'])
            self.optimizer_m3.step()
        
        total_disencC += l_disencC.item()
        total_disencS += l_disencS.item()
        if sty_type=='pos':
            l_posdiscri = RM_dis_loss.item()+FNM_dis_loss.item()
            l_negdiscri = FM_dis_loss.item()+RNM_dis_loss.item()
        else:
            l_negdiscri = RM_dis_loss.item()+FNM_dis_loss.item()
            l_posdiscri = FM_dis_loss.item()+RNM_dis_loss.item()
        total_posdiscri += l_posdiscri
        total_negdiscri += l_negdiscri
        self._log_m2(epoch, train_steps, total_steps, l_posdiscri, 
                     l_negdiscri, l_disencC.item(), l_disencS.item(),
                     (epoch*(self.config['m2_train_freq'])+cnt)*total_steps+train_steps+1)
        train_steps+=1
        return train_steps, total_posdiscri, total_negdiscri, total_disencC, total_disencS
    
    def _get_m1_loss(self, data, domain_discriminator, tf_rate, sty_type):
        bos = self.vocab['<BOS>']
        eos = self.vocab['<EOS>']
        pad = self.vocab['<PAD>']
        #CosSim = nn.CosineSimilarity(dim=1)
        xs, ys, ys_in, ys_out, ilens, styles = to_gpu(data, bos, eos, pad)
        reverse_styles = self._get_reverse_style(styles)
        style_vector, content_outputs, content_lens = self._pass2encs(xs, ilens)
        # Style Loss
        _, s_log_probs, _= self.s_classifier(style_vector)
        true_s_log_probs = torch.gather(s_log_probs,dim=1,
                                        index=styles.unsqueeze(1)).squeeze(1)
        s_loss = -torch.mean(true_s_log_probs)*self.alpha
        # Disentangle Loss
        predict_style = self.disenC(content_outputs, content_lens)
        rand_vecS = self._random_vector(style_vector, 'tanh')
        loss_adv_disencC = torch.mean((predict_style-rand_vecS)**2)*self.gamma
        # Disentangle encS
        _, disenS_log_probs, _, _ = self.disenS(style_vector, ilens, None, 
                                                (ys_in, ys_out), tf_rate=1.0,
                                                max_dec_timesteps=self.config['max_dec_timesteps'])
        loss_adv_disencS = torch.mean(disenS_log_probs)*self.zeta
        # Mimick loss
        recon_log_probs, _, _, mimicked_style =\
            self._get_decoder_pred(styles, content_outputs, content_lens, 
                                   tf_rate, (ys_in, ys_out))
        loss_mimic = torch.mean((mimicked_style-style_vector)**2)*self.beta
        # Reconstruction loss
        loss_recon = -torch.mean(recon_log_probs)*1
        # Domain discriminator
        if self.delta > 0:
            _, predicts, pred_ilens, _ =\
                self._get_decoder_pred(reverse_styles, content_outputs, content_lens)
            # For fake data, try to cheat domain discriminator
            _, discri_log_probs,_ = domain_discriminator(predicts, pred_ilens, need_sort=True)
            all_one = cc(styles.cpu().new_ones(styles.size()))
            discri_log_probs = torch.gather(discri_log_probs,dim=1,
                                            index=all_one.unsqueeze(1)).squeeze(1)
            loss_adv_domain_discri = -torch.mean(discri_log_probs)*self.delta
        else:
            loss_adv_domain_discri = torch.Tensor([0])
        '''
        # directinal loss & cluster loss
        if sty_type == 'pos':
            loss_directional = torch.mean(CosSim(style_vector,
                                                 self.mean_style_neg.expand_as(style_vector))+1)*self.zeta*5
            self.mean_style_pos = torch.mean(style_vector, dim=0)
            loss_cluster = torch.mean((style_vector-self.mean_style_pos.expand_as(style_vector))**2)*self.zeta
        else:
            loss_directional = torch.mean(CosSim(style_vector,
                                                 self.mean_style_pos.expand_as(style_vector))+1)*self.zeta*5
            self.mean_style_neg = torch.mean(style_vector, dim=0)
            loss_cluster = torch.mean((style_vector-self.mean_style_neg.expand_as(style_vector))**2)*self.zeta
        '''
        return s_loss, loss_adv_disencC, loss_mimic, loss_recon, \
            loss_adv_domain_discri, loss_adv_disencS
    
    def _log_m1(self, epoch, train_steps, total_steps, s_loss, l_recon, 
                l_adv_disencC, l_mimic, l_adv_domain_discri, l_adv_disencS, log_steps):
        print(f'epoch: {epoch}, [{train_steps + 1}/{total_steps}], '
              f'recon_loss: {l_recon:.3f}, style: {s_loss:.3f}, '
              f'adv_disencC: {l_adv_disencC:.3f}, '
              f'adv_disencS: {-l_adv_disencS:.3f} '
              f'mimic: {l_mimic:.3f}, adv_domain: {l_adv_domain_discri:.3f}', end='\r')
        
        # add to logger
        tag = self.config['tag']
        self.logger.scalar_summary(tag=f'{tag}/train/reconstructionloss', 
                                   value=l_recon, 
                                   step=log_steps)
        self.logger.scalar_summary(tag=f'{tag}/train/stylepredictionloss', 
                                   value=s_loss/self.alpha, 
                                   step=log_steps)
        self.logger.scalar_summary(tag=f'{tag}/train/adver_disencC_loss', 
                                   value=l_adv_disencC/self.gamma, 
                                   step=log_steps)
        self.logger.scalar_summary(tag=f'{tag}/train/stylemimic_loss', 
                                   value=l_mimic/self.beta, 
                                   step=log_steps)
        if self.delta > 0:
            self.logger.scalar_summary(tag=f'{tag}/train/adver_domain_loss', 
                                       value=l_adv_domain_discri/self.delta, 
                                       step=log_steps)
        '''
        self.logger.scalar_summary(tag=f'{tag}/train/style_cosine_loss', 
                                   value=(l_dir/5)/self.zeta, 
                                   step=log_steps)
        self.logger.scalar_summary(tag=f'{tag}/train/style_cluster_loss', 
                                   value=l_clu/self.zeta, 
                                   step=log_steps)
        '''
        self.logger.scalar_summary(tag=f'{tag}/train/adver_disencS_loss', 
                                   value=-l_adv_disencS/self.zeta, 
                                   step=log_steps)
        return

    def _train_m1_onetime(self, data, total_adv_disencS, 
                          total_adv_disencC, total_sloss, total_mimic, 
                          total_recon, total_adv_domain, epoch, train_steps, 
                          total_steps, domain_discriminator, tf_rate, sty_type):
        s_loss, l_adv_disencC, l_mimic, l_recon, l_adv_domain_discri, l_adv_disencS=\
            self._get_m1_loss(data, domain_discriminator, tf_rate, sty_type)
        #calcuate gradients
        self.optimizer_m1.zero_grad()
        if self.delta > 0:
            total_loss = s_loss+l_adv_disencC+l_mimic+l_recon+l_adv_domain_discri+l_adv_disencS
        else:
            total_loss = s_loss+l_adv_disencC+l_mimic+l_recon+l_adv_disencS
        #total_loss.backward(retain_graph=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params_m1, max_norm=self.config['max_grad_norm'])
        self.optimizer_m1.step()
        total_adv_domain += l_adv_domain_discri.item()
        total_recon += l_recon.item()
        total_sloss += s_loss.item()
        total_adv_disencC += l_adv_disencC.item()
        total_mimic += l_mimic.item()
        #total_directional += l_dir.item()
        #total_cluster += l_clu.item()
        total_adv_disencS += l_adv_disencS.item()
        self._log_m1(epoch, train_steps, total_steps, s_loss.item(), l_recon.item(),
                     l_adv_disencC.item(), l_mimic.item(), 
                     l_adv_domain_discri.item(), l_adv_disencS.item(),
                     epoch*total_steps+train_steps+1)
        train_steps +=1
        return (train_steps, total_adv_domain, total_recon, total_sloss,\
                total_adv_disencS, total_adv_disencC, total_mimic)

    def train_one_epoch(self, epoch, tf_rate):
        total_steps = len(self.train_pos_loader)*2
        total_disencC = 0.
        total_adv_disencC = 0.
        total_posdiscri = 0.
        total_negdiscri = 0.
        total_adv_posdiscri = 0.
        total_adv_negdiscri = 0.
        total_sloss = 0.
        total_mimic = 0.
        total_recon = 0.
        total_disencS = 0.
        total_adv_disencS = 0.
        '''
        total_directional = 0.
        total_cluster = 0.
        # initial mean style vector
        try:
            self.mean_style_pos
        except AttributeError:
            self.mean_style_pos = cc(torch.ones(self.config['encS_hidden_dim']*2))
        try:
            self.mean_style_neg
        except AttributeError:
            self.mean_style_neg = cc(torch.ones(self.config['encS_hidden_dim']*2).fill_(-1.))
        '''
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
            (train_steps, total_adv_posdiscri, total_recon, total_sloss, 
             total_adv_disencS, total_adv_disencC, total_mimic)=\
                self._train_m1_onetime(data, total_adv_disencS, 
                                       total_adv_disencC, total_sloss, total_mimic, 
                                       total_recon, total_adv_negdiscri, epoch, 
                                       train_steps, total_steps, self.neg_domain_discri, 
                                       tf_rate, 'pos')
            try:
                data = next(neg_data_iterator)
            except StopIteration:
                neg_data_iterator = iter(self.train_neg_loader)
                data = next(neg_data_iterator)
            (train_steps, total_adv_posdiscri, total_recon, total_sloss, 
             total_adv_disencS, total_adv_disencC, total_mimic)=\
                self._train_m1_onetime(data, total_adv_disencS, 
                                       total_adv_disencC, total_sloss, total_mimic, 
                                       total_recon, total_adv_posdiscri, epoch, 
                                       train_steps, total_steps, self.pos_domain_discri, 
                                       tf_rate, 'neg')
        print()

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
                
                train_steps, total_posdiscri, total_negdiscri, total_disencC, total_disencS=\
                    self._train_m2_onetime(data, total_disencC, total_disencS,
                                           total_posdiscri, total_negdiscri, epoch,
                                           train_steps, total_steps, cnt, 'pos')
                try:
                    data = next(neg_data_iterator)
                except StopIteration:
                    neg_data_iterator = iter(self.train_neg_loader)
                    data = next(neg_data_iterator)
                
                train_steps, total_posdiscri, total_negdiscri, total_disencC, total_disencS=\
                    self._train_m2_onetime(data, total_disencC, total_disencS,
                                           total_posdiscri, total_negdiscri, epoch,
                                           train_steps, total_steps, cnt, 'neg')
            print()
        
        return ((total_recon/total_steps),(total_sloss/total_steps),\
                (total_adv_disencC/total_steps),(total_adv_disencS/total_steps),\
                (total_mimic/total_steps),(total_adv_posdiscri*2/total_steps),\
                (total_adv_negdiscri*2/total_steps),\
                ((total_adv_posdiscri/total_steps)/self.config['m2_train_freq']),\
                ((total_adv_negdiscri/total_steps)/self.config['m2_train_freq']),\
                ((total_disencC/total_steps)/self.config['m2_train_freq']))

    def _valid_feed_data(self, data_loader, total_loss):
        all_prediction = []
        all_inputs = []
        for step, data in enumerate(data_loader):
            bos = self.vocab['<BOS>']
            eos = self.vocab['<EOS>']
            pad = self.vocab['<PAD>']
            xs, ys, ys_in, ys_out, ilens, styles = to_gpu(data, bos, eos, pad)
            reverse_styles = self._get_reverse_style(styles)
            _, content_outputs, content_lens = self._pass2encs(xs, ilens)
            recon_log_probs, prediction, pred_ilens, _ =\
                self._get_decoder_pred(reverse_styles, content_outputs, content_lens)
            seq_len = [y.size(0) + 1 for y in ys]
            mask = cc(_seq_mask(seq_len=seq_len, max_len=recon_log_probs.size(1)))
            loss = (-torch.sum(recon_log_probs*mask))/sum(seq_len)
            total_loss += loss.item()
            all_prediction = all_prediction + prediction.cpu().tolist()
            all_inputs = all_inputs + [y.cpu().tolist() for y in ys]
        return all_prediction, all_inputs, total_loss

    def validation(self):
        self.encS.eval()
        self.encC.eval()
        self.s_classifier.eval()
        self.style_mimicker.eval()
        self.decoder.eval()
        self.attention.eval()
        self.disenC.eval()
        self.disenS.eval()
        total_loss = 0.
        # positive input
        posdata_pred, posdata_input, total_loss =\
            self._valid_feed_data(self.dev_pos_loader, total_loss)
        # get sentence
        posdata_pred, posdata_input = self.idx2sent(posdata_pred, posdata_input)
        # write file
        if not os.path.exists(self.config['dev_file_path']):
            os.makedirs(self.config['dev_file_path'])
        file_prefix = ('a'+str(self.alpha)+'b'+str(self.beta)+'g'+\
            str(self.gamma)+'d'+str(self.delta)+'z'+str(self.zeta))
        file_path_pos = os.path.join(self.config['dev_file_path'], 
                                     f'pred.fader.{file_prefix}.dev.0.temp')
        file_path_gtpos = os.path.join(self.config['dev_file_path'], 
                                       f'gf.fader.{file_prefix}.dev.1.temp')
        writefile(posdata_pred, file_path_pos)
        writefile(posdata_input, file_path_gtpos)
        # negative input
        posdata_pred, posdata_input, total_loss =\
            self._valid_feed_data(self.dev_neg_loader, total_loss)
        # get sentence
        posdata_pred, posdata_input = self.idx2sent(posdata_pred, posdata_input)
        # write file
        file_path_neg = os.path.join(self.config['dev_file_path'], 
                                     f'pred.fader.{file_prefix}.dev.1.temp')
        file_path_gtneg = os.path.join(self.config['dev_file_path'], 
                                       f'gf.fader.{file_prefix}.dev.0.temp')
        writefile(posdata_pred, file_path_neg)
        writefile(posdata_input, file_path_gtneg)
        self.encS.train()
        self.encC.train()
        self.s_classifier.train()
        self.style_mimicker.train()
        self.decoder.train()
        self.attention.train()
        self.disenC.train()
        self.disenS.train()
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
            self.encC.load_state_dict(state_dict[0])
            self.decoder.load_state_dict(state_dict[1])
            self.attention.load_state_dict(state_dict[2])
            self.style_mimicker.load_state_dict(state_dict[3])

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
        self.encS.eval()
        self.encC.eval()
        self.s_classifier.eval()
        self.style_mimicker.eval()
        self.decoder.eval()
        self.attention.eval()
        self.disenC.eval()
        self.disenS.eval()
        total_loss = 0.
        # positive input
        posdata_pred, posdata_input, total_loss =\
            self._valid_feed_data(test_pos_loader, total_loss)
        # get sentence
        posdata_pred, posdata_input = self.idx2sent(posdata_pred, posdata_input)
        # write file
        if not os.path.exists(self.config['test_file_path']):
            os.makedirs(self.config['test_file_path'])
        file_prefix = ('a'+str(self.alpha)+'b'+str(self.beta)+'g'+\
            str(self.gamma)+'d'+str(self.delta)+'z'+str(self.zeta))
        file_path_pos = os.path.join(self.config['test_file_path'], 
                                     f'pred.fader.{file_prefix}.test.0(input1)')
        file_path_gtpos = os.path.join(self.config['test_file_path'], 
                                       f'gt.fader.{file_prefix}.test.input1')
        writefile(posdata_pred, file_path_pos)
        writefile(posdata_input, file_path_gtpos)
        # negative input
        posdata_pred, posdata_input, total_loss =\
            self._valid_feed_data(test_neg_loader, total_loss)
        # get sentence
        posdata_pred, posdata_input = self.idx2sent(posdata_pred, posdata_input)
        # write file
        file_path_neg = os.path.join(self.config['test_file_path'], 
                                     f'pred.fader.{file_prefix}.test.1(input0)')
        file_path_gtneg = os.path.join(self.config['test_file_path'], 
                                       f'gf.fader.{file_prefix}.test.input0')
        writefile(posdata_pred, file_path_neg)
        writefile(posdata_input, file_path_gtneg)       

        self.encS.train()
        self.encC.train()
        self.s_classifier.train()
        self.style_mimicker.train()
        self.decoder.train()
        self.attention.train()
        self.disenC.train()
        self.disenS.train()
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
