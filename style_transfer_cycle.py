import torch 
import torch.nn.functional as F
import numpy as np
from model import Encoder, Decoder, Style_classifier, Domain_discri
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
        self.get_label_dist()

        # build model and optimizer
        self.build_model(load_model=load_model)

    def save_model(self, model_path):
        encoder_path = model_path+'_encoder'
        decoder_path = model_path+'_decoder'
        s_classifier_path = model_path+'_sclr'
        pos_discri_path = model_path+'_posdis'
        neg_discri_path = model_path+'_negdis'
        opt1_path = model_path+'_opt1'
        opt2_path = model_path+'_opt2'
        opt3_path = model_path+'_opt3'
        torch.save(self.encoder.state_dict(), f'{encoder_path}.ckpt')
        torch.save(self.decoder.state_dict(), f'{decoder_path}.ckpt')
        torch.save(self.s_classifier.state_dict(), f'{s_classifier_path}.ckpt')
        torch.save(self.pos_discri.state_dict(), f'{pos_discri_path}.ckpt')
        torch.save(self.neg_discri.state_dict(), f'{neg_discri_path}.ckpt')
        torch.save(self.optimizer_m1.state_dict(), f'{opt1_path}.opt')
        torch.save(self.optimizer_m2.state_dict(), f'{opt2_path}.opt')
        torch.save(self.optimizer_m3.state_dict(), f'{opt3_path}.opt')
        return

    def load_vocab(self):
        with open(self.config['vocab_path'], 'rb') as f:
            self.vocab = pickle.load(f) # a dict; word to index
        with open(self.config['non_lang_syms_path'], 'rb') as f:
            self.non_lang_syms = pickle.load(f) # an array
        return

    def load_model(self, model_path, load_optimizer):
        print(f'Load model from {model_path}')
        encoder_path = model_path+'_encoder'
        decoder_path = model_path+'_decoder'
        s_classifier_path = model_path+'_sclr'
        pos_discri_path = model_path+'_posdis'
        neg_discri_path = model_path+'_negdis'
        opt1_path = model_path+'_opt1'
        opt2_path = model_path+'_opt2'
        opt3_path = model_path+'_opt3'
        self.encoder.load_state_dict(torch.load(f'{encoder_path}.ckpt'))
        self.decoder.load_state_dict(torch.load(f'{decoder_path}.ckpt'))
        self.s_classifier.load_state_dict(torch.load(f'{s_classifier_path}.ckpt'))
        self.pos_discri.load_state_dict(torch.load(f'{pos_discri_path}.ckpt'))
        self.neg_discri.load_state_dict(torch.load(f'{neg_discri_path}.ckpt'))
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

    def get_data_loaders(self):
        root_dir = self.config['dataset_root_dir']
        
        # get train dataset
        train_pos_set = self.config['pos_train_set']
        train_neg_set = self.config['neg_train_set']
        self.train_pos_dataset = PickleDataset(os.path.join(root_dir,f'{train_pos_set}.p'),
                                               config=self.config,
                                               sort=self.config['sort_dataset'])
        self.train_pos_loader = get_data_loader(self.train_pos_dataset, 
                                                batch_size=self.config['batch_size'], 
                                                shuffle=self.config['shuffle'])
        self.train_neg_dataset = PickleDataset(os.path.join(root_dir,f'{train_neg_set}.p'),
                                               config=self.config,
                                               sort=self.config['sort_dataset'])
        self.train_neg_loader = get_data_loader(self.train_neg_dataset, 
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

    def get_label_dist(self):
        labelcount = np.zeros(len(self.vocab))
        for token_ids,_ in self.train_pos_dataset:
            for ind in token_ids:
                labelcount[ind] += 1.
        for token_ids,_ in self.train_neg_dataset:
            for ind in token_ids:
                labelcount[ind] += 1.
        labelcount[self.vocab['<EOS>']]+=len(self.train_pos_dataset)+len(self.train_neg_dataset)
        labelcount[self.vocab['<PAD>']]=0
        labelcount[self.vocab['<BOS>']]=0
        self.labeldist = labelcount / np.sum(labelcount)
        return
    
    def load_automatic_style_classifier(self, model_path, train=False):
        pretrain_w2v_path = self.config['pretrain_w2v_path']
        if pretrain_w2v_path is None:
            pretrain_w2v = None
        else:
            with open(pretrain_w2v_path, 'rb') as f:
                pretrain_w2v = pickle.load(f)
        self.textrnn = cc_model(Domain_discri(vocab_size=len(self.vocab),
                                              embedding_dim=100,
                                              rnn_hidden_dim=256,
                                              dropout_rate=0.2,
                                              dnn_hidden_dim=200,
                                              pad_idx=self.vocab['<PAD>'],
                                              pre_embedding=pretrain_w2v,
                                              update_embedding=self.config['update_embedding']))
        print(self.textrnn)
        self.textrnn.float()
        if (model_path is None) or (train):
            if train:
                best_model = self.train_automatic_style_classifier(train, model_path)
            else:
                best_model = self.train_automatic_style_classifier(False, model_path)
            self.textrnn.load_state_dict(best_model)
        else:
            self.textrnn.load_state_dict(torch.load(f'{model_path}.ckpt'))
        self.textrnn.eval()
        return

    def train_automatic_style_classifier(self, load=False, model_path=None): 
        total_steps = len(self.train_pos_loader)
        total_val_steps = len(self.dev_loader)
        best_acc = 0.
        early_stop_counter = 0
        best_model = None
        opt_pretrain_sc = torch.optim.Adam(list(self.textrnn.parameters()), 
                                           lr=0.001, weight_decay=1e-7)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_pretrain_sc,
                                                               mode='min',
                                                               factor=0.5,
                                                               patience=3,
                                                               verbose=True,
                                                               min_lr=1e-7) 
        if load:
            self.textrnn.load_state_dict(torch.load(f'{model_path}.ckpt'))
        print('------------------------------------------')
        print('start training pretrained style classifier')
        for epoch in range(20):
            total_train_loss = 0.
            for train_steps, data in enumerate(zip(self.train_pos_loader,self.train_neg_loader)):
                bos = self.vocab['<BOS>']
                eos = self.vocab['<EOS>']
                pad = self.vocab['<PAD>']
                xs, _, _, _, ilens, styles = to_gpu(data[0], bos, eos, pad)
                _, log_probs, prediction = self.textrnn(xs, ilens)
                true_log_probs=torch.gather(log_probs,dim=1,index=styles.unsqueeze(1)).squeeze(1)
                train_loss = -torch.mean(true_log_probs)
                xs, _, _, _, ilens, styles = to_gpu(data[1], bos, eos, pad)
                _, log_probs, prediction = self.textrnn(xs, ilens)
                true_log_probs=torch.gather(log_probs,dim=1,index=styles.unsqueeze(1)).squeeze(1)
                train_loss2 = -torch.mean(true_log_probs)
                total_train_loss += train_loss2.item()
                
                opt_pretrain_sc.zero_grad()
                (train_loss+train_loss2).backward()
                torch.nn.utils.clip_grad_norm_(list(self.textrnn.parameters()),5)
                opt_pretrain_sc.step()
                print(f'epoch: {epoch}, [{train_steps+1}/{total_steps}], '
                      f'loss: {train_loss.item()+train_loss2.item():.4f}', end='\r')
            
            # validation
            self.textrnn.eval()
            total_val_loss = 0.
            total_val_num = 0.
            total_correct_num = 0.
            for step, data in enumerate(self.dev_loader):
                bos = self.vocab['<BOS>']
                eos = self.vocab['<EOS>']
                pad = self.vocab['<PAD>']
                xs, _, _, _, ilens, styles = to_gpu(data, bos, eos, pad)
                _, log_probs, prediction = self.textrnn(xs, ilens)
                true_log_probs=torch.gather(log_probs,dim=1,index=styles.unsqueeze(1)).squeeze(1)
                val_loss = -torch.mean(true_log_probs)
                total_val_loss += val_loss.item()
                total_val_num += styles.size(0)
                correct = prediction.view(-1).eq(styles).sum().item()
                total_correct_num += correct
            
            val_acc = total_correct_num/total_val_num
            print(f'epoch: {epoch}, avg_train_loss: {total_train_loss/total_steps:.4f} '
                  f'avg_val_loss: {total_val_loss/total_val_steps:.4f}, val_acc: {val_acc:.4f}')
            # save model in every epoch
            if not os.path.exists('./models/pretrained_cnn_style_classifier'):
                os.makedirs('./models/pretrained_cnn_style_classifier')
            model_path = './models/pretrained_cnn_style_classifier/pretrained_style_classifier'
            torch.save(self.textrnn.state_dict(), f'{model_path}-{epoch:03d}.ckpt')
            self.textrnn.train()
            if val_acc > best_acc: 
                # save model
                torch.save(self.textrnn.state_dict(), f'{model_path}_best.ckpt')
                best_acc = val_acc
                best_model = self.textrnn.state_dict()
                print('-----------------')
                print(f'Save #{epoch} model, val_acc={val_acc:.4f}')
                print('-----------------')
                early_stop_counter=0
            early_stop_counter+=1
            if early_stop_counter > 5:
                break
            scheduler.step(val_acc)
        print('------------------------------------------')
        print('finish training pretrained style classifier')
        print(f'best validation acc :{best_acc:.4f}')
        return best_model

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
        self.pos_discri=\
            cc_model(Domain_discri(vocab_size=len(self.vocab),
                                   embedding_dim=self.config['embedding_dim'],
                                   rnn_hidden_dim=self.config['discri_hidden_dim'],
                                   dropout_rate=self.config['discri_dropout_p'],
                                   dnn_hidden_dim=self.config['discri_hidden_dim'],
                                   pad_idx=self.vocab['<PAD>'],
                                   pre_embedding=pretrain_w2v,
                                   update_embedding=self.config['update_embedding']))
        self.neg_discri=\
            cc_model(Domain_discri(vocab_size=len(self.vocab),
                                   embedding_dim=self.config['embedding_dim'],
                                   rnn_hidden_dim=self.config['discri_hidden_dim'],
                                   dropout_rate=self.config['discri_dropout_p'],
                                   dnn_hidden_dim=self.config['discri_hidden_dim'],
                                   pad_idx=self.vocab['<PAD>'],
                                   pre_embedding=pretrain_w2v,
                                   update_embedding=self.config['update_embedding']))
        self.pos_discri.float()
        self.neg_discri.float()
        print(self.pos_discri)
        print(self.neg_discri)
        self.params_m1=list(self.encoder.parameters())+list(self.decoder.parameters())
        self.params_m2=list(self.s_classifier.parameters())
        self.params_m3=list(self.pos_discri.parameters())+list(self.neg_discri.parameters())
        self.optimizer_m1 =\
            torch.optim.Adam(self.params_m1, 
                             lr=self.config['learning_rate_m1'], 
                             weight_decay=self.config['weight_decay_m1'])
        self.optimizer_m2 =\
            torch.optim.Adam(self.params_m2, 
                             lr=self.config['learning_rate_m2'], 
                             weight_decay=self.config['weight_decay_m2'])
        self.optimizer_m3 =\
            torch.optim.Adam(self.params_m3, 
                             lr=self.config['learning_rate_m3'], 
                             weight_decay=self.config['weight_decay_m3'])

        if load_model:
            self.load_model(self.config['load_model_path'], self.config['load_optimizer'])
        self.load_automatic_style_classifier(self.config['load_pretrained_style_classifier_path'],
                                             self.config['train_pretrained_style_classifier'])

        return

    def validation(self):

        self.encoder.eval()
        self.decoder.eval()
        all_prediction, all_ys = [], []
        total_loss = 0.
        total_val_num = 0.
        total_correct_num = 0.
        for step, data in enumerate(self.dev_loader):

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
           
            # pass the prediction to the pretrained style classifier
            pred_ilens = get_prediction_length(prediction, eos=self.vocab['<EOS>'])
            _, pr_style_probs, pr_style_prediction = self.textrnn(prediction, pred_ilens, need_sort=True)
            correct = pr_style_prediction.cpu().view(-1).eq(reverse_styles.cpu()).sum().item()
            total_correct_num += correct
            total_val_num += reverse_styles.size(0)
            
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
        avg_style_correct = total_correct_num / total_val_num

        wer, prediction_sents, ground_truth_sents = self.idx2sent(all_prediction, all_ys)

        return avg_loss, wer, prediction_sents, ground_truth_sents, avg_style_correct
    
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
        total_val_num = 0.
        total_correct_num = 0.
        all_prediction,all_ys,all_styles,all_reverse_styles,all_style_predict=[],[],[],[],[]
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

            # pass the prediction to the pretrained style classifier
            pred_ilens = get_prediction_length(prediction, eos=self.vocab['<EOS>'])
            #pr_style_probs, pr_style_prediction = self.textcnn(prediction)
            _, pr_style_probs, pr_style_prediction = self.textrnn(prediction, pred_ilens, need_sort=True)
            correct = pr_style_prediction.cpu().view(-1).eq(reverse_styles.cpu()).sum().item()
            total_val_num +=reverse_styles.size(0)
            total_correct_num +=correct
            
            all_style_predict = all_style_predict + pr_style_prediction.cpu().view(-1).tolist()
            all_prediction = all_prediction + prediction.cpu().numpy().tolist()
            all_ys = all_ys + [y.cpu().numpy().tolist() for y in ys]
            all_styles = all_styles + styles.cpu().tolist()
            all_reverse_styles = all_reverse_styles + reverse_styles.cpu().tolist()

        self.encoder.train()
        self.decoder.train()
        avg_style_correct = total_correct_num / total_val_num
        wer, prediction_sents, ground_truth_sents = self.idx2sent(all_prediction, all_ys)

        with open(f'{test_file_name}.txt', 'w') as f:
            f.write(f'Total sentences: {len(prediction_sents)}, WER={wer:.4f}\n')
            f.write(f'Average style accuracy: {avg_style_correct:.4f} \n')
            for idx, p in enumerate(prediction_sents):
                f.write(f'Pred (style:{all_reverse_styles[idx]},pred_s:{all_style_predict[idx]}) :{p}\n')
                f.write(f'Original Sent (style:{all_styles[idx]}) :{ground_truth_sents[idx]}\n')
                f.write('----------------------------------------\n')

        print(f'{test_file_name}: {len(prediction_sents)} utterances, WER={wer:.4f}')
        print(f'Average style accuracy: {avg_style_correct:.4f}')
        return wer

    def _random_target(self, x, num_class=2):
        out = cc(torch.LongTensor(x.size()).random_(0,num_class))
        return out

    def train_one_epoch(self, epoch, tf_rate):

        total_steps = len(self.train_pos_loader)
        total_loss = 0.
        total_style_loss = 0.
        total_inverse_style_loss = 0.
        total_cycle_loss = 0.
        total_discri_loss = 0.
        total_inverse_discri_loss = 0.
        
        for train_steps, data in enumerate(zip(self.train_pos_loader,self.train_neg_loader)):
            bos = self.vocab['<BOS>']
            eos = self.vocab['<EOS>']
            pad = self.vocab['<PAD>']
            xs_pos, ys_pos, ys_in_pos, ys_out_pos, ilens_pos, pos_styles = to_gpu(data[0], bos, eos, pad)
            xs_neg, ys_neg, ys_in_neg, ys_out_neg, ilens_neg, neg_styles = to_gpu(data[1], bos, eos, pad)
            reverse_pos_styles = cc(torch.LongTensor(pos_styles.cpu().new_zeros(pos_styles.size()))) 
            reverse_neg_styles = cc(torch.LongTensor(neg_styles.cpu().new_ones(neg_styles.size()))) 
            
            # Positive Part
            # Reconstruct Loss
            enc_outputs, enc_lens = self.encoder(xs_pos, ilens_pos)
            _, recon_log_probs, _, _=\
                self.decoder(enc_outputs, enc_lens, pos_styles, (ys_in_pos, ys_out_pos),
                             tf_rate=tf_rate, sample=False,
                             max_dec_timesteps=self.config['max_dec_timesteps'])

            loss_pos = -torch.mean(recon_log_probs)*self.config['recons_loss_ratio']
            total_loss += loss_pos.item()
            # Style Loss
            enc_representation = get_enc_context(enc_outputs, enc_lens)
            _, s_log_probs, _= self.s_classifier(enc_representation)
            random_s_log_probs =\
                torch.gather(s_log_probs,dim=1,
                             index=self._random_target(pos_styles, self.config['n_style_type']).unsqueeze(1)).squeeze(1)
            s_loss_pos = -torch.mean(random_s_log_probs)*self.config['style_loss_ratio']            
            total_style_loss += s_loss_pos.item()           
            # Cycle Loss
            _,_,predict_neg,_ = self.decoder(enc_outputs, enc_lens, reverse_pos_styles, None, 
                                             max_dec_timesteps=self.config['max_dec_timesteps'])
            cycle_ilens = get_prediction_length(predict_neg, eos=self.vocab['<EOS>'])
            cycle_enc_outputs, cycle_enc_lens = self.encoder(predict_neg, cycle_ilens, need_sort=True)
            _, cycle_log_probs, _, _ =\
                self.decoder(cycle_enc_outputs, cycle_enc_lens, pos_styles, (ys_in_pos, ys_out_pos),
                             tf_rate=tf_rate, sample=False,
                             max_dec_timesteps=self.config['max_dec_timesteps'])
            cycle_loss_pos = -torch.mean(cycle_log_probs)*self.config['cycle_loss_ratio']
            total_cycle_loss += cycle_loss_pos.item()            
            # Discriminator Loss
            # For fake negative data, try to cheat neg_discri
            _, neg_discri_log_probs, _ = self.neg_discri(predict_neg, cycle_ilens, need_sort=True)
            rand_neg_discri_log_probs =\
                torch.gather(neg_discri_log_probs,dim=1,
                             index=self._random_target(pos_styles).unsqueeze(1)).squeeze(1)
            neg_discri_loss = -torch.mean(rand_neg_discri_log_probs)*self.config['discri_loss_ratio']
            total_discri_loss += neg_discri_loss.item()
           
            # Negative Part
            # Reconstruct Loss
            enc_outputs, enc_lens = self.encoder(xs_neg, ilens_neg)
            _, recon_log_probs, _, _=\
                self.decoder(enc_outputs, enc_lens, neg_styles, (ys_in_neg, ys_out_neg),
                             tf_rate=tf_rate, sample=False,
                             max_dec_timesteps=self.config['max_dec_timesteps'])

            loss_neg = -torch.mean(recon_log_probs)*self.config['recons_loss_ratio']
            total_loss += loss_neg.item()
            # Style Loss
            enc_representation = get_enc_context(enc_outputs, enc_lens)
            _, s_log_probs, _ = self.s_classifier(enc_representation)
            random_s_log_probs =\
                torch.gather(s_log_probs,dim=1,
                             index=self._random_target(neg_styles, self.config['n_style_type']).unsqueeze(1)).squeeze(1)
            s_loss_neg = -torch.mean(random_s_log_probs)*self.config['style_loss_ratio']            
            total_style_loss += s_loss_neg.item()

            # Cycle Loss
            _,_,predict_pos,_ = self.decoder(enc_outputs, enc_lens, reverse_neg_styles, None, 
                                             max_dec_timesteps=self.config['max_dec_timesteps'])
            cycle_ilens = get_prediction_length(predict_pos, eos=self.vocab['<EOS>'])
            cycle_enc_outputs, cycle_enc_lens = self.encoder(predict_pos, cycle_ilens, need_sort=True)
            cycle_logits, cycle_log_probs, _, _ =\
                self.decoder(cycle_enc_outputs, cycle_enc_lens, neg_styles, (ys_in_neg, ys_out_neg),
                             tf_rate=tf_rate, sample=False,
                             max_dec_timesteps=self.config['max_dec_timesteps'])
            cycle_loss_neg = -torch.mean(cycle_log_probs)*self.config['cycle_loss_ratio']
            total_cycle_loss += cycle_loss_neg.item()
            
            # Discriminator Loss
            # For fake positive data, try to cheat pos_discri
            _, pos_discri_log_probs, _ = self.pos_discri(predict_pos, cycle_ilens, need_sort=True)
            rand_pos_discri_log_probs =\
                torch.gather(pos_discri_log_probs,dim=1,
                             index=self._random_target(neg_styles).unsqueeze(1)).squeeze(1)
            pos_discri_loss = -torch.mean(rand_pos_discri_log_probs)*self.config['discri_loss_ratio']
            total_discri_loss += pos_discri_loss.item()
           
            # calculate gradients 
            self.optimizer_m1.zero_grad()
            tloss = loss_pos+loss_neg+s_loss_pos+s_loss_neg+cycle_loss_pos
            tloss += cycle_loss_neg+neg_discri_loss+pos_discri_loss
            tloss.backward()
            torch.nn.utils.clip_grad_norm_(self.params_m1, max_norm=self.config['max_grad_norm'])
            self.optimizer_m1.step()
            # print message
            print(f'epoch: {epoch}, [{train_steps + 1}/{total_steps}], loss: {(loss_pos.item()+loss_neg.item())/2:.3f}, '
                  f'cycle_loss: {(cycle_loss_pos.item()+cycle_loss_neg.item())/2:.3f}, '
                  f'style_loss: {(s_loss_pos.item()+s_loss_neg.item())/2:.3f}, '
                  f'discri_loss: {(pos_discri_loss.item()+neg_discri_loss.item())/2:.3f}', end='\r')
            
            # add to logger
            tag = self.config['tag']
            self.logger.scalar_summary(tag=f'{tag}/train/reconloss_pos', 
                                       value=loss_pos.item()/self.config['recons_loss_ratio'], 
                                       step=epoch*total_steps+train_steps+1)
            self.logger.scalar_summary(tag=f'{tag}/train/reconloss_neg', 
                                       value=loss_neg.item()/self.config['recons_loss_ratio'], 
                                       step=epoch*total_steps+train_steps+1)
            self.logger.scalar_summary(tag=f'{tag}/train/adv_style_l_pos', 
                                       value=s_loss_pos.item()/self.config['style_loss_ratio'], 
                                       step=epoch*total_steps+train_steps+1)
            self.logger.scalar_summary(tag=f'{tag}/train/adv_style_l__neg', 
                                       value=s_loss_neg.item()/self.config['style_loss_ratio'], 
                                       step=epoch*total_steps+train_steps+1)
            self.logger.scalar_summary(tag=f'{tag}/train/cycleloss_pos', 
                                       value=cycle_loss_pos.item()/self.config['cycle_loss_ratio'], 
                                       step=epoch*total_steps+train_steps+1)
            self.logger.scalar_summary(tag=f'{tag}/train/cycleloss_neg', 
                                       value=cycle_loss_neg.item()/self.config['cycle_loss_ratio'], 
                                       step=epoch*total_steps+train_steps+1)
            self.logger.scalar_summary(tag=f'{tag}/train/adv_domain_pos', 
                                       value=pos_discri_loss.item()/self.config['discri_loss_ratio'], 
                                       step=epoch*total_steps+train_steps+1)
            self.logger.scalar_summary(tag=f'{tag}/train/adv_domain_neg', 
                                       value=neg_discri_loss.item()/self.config['discri_loss_ratio'], 
                                       step=epoch*total_steps+train_steps+1)
        print()
        
        for cnt in range(self.config['m2_train_freq']):
            for train_steps, data in enumerate(zip(self.train_pos_loader,self.train_neg_loader)):
                bos = self.vocab['<BOS>']
                eos = self.vocab['<EOS>']
                pad = self.vocab['<PAD>']
                xs_pos, ys_pos, ys_in_pos, ys_out_pos, ilens_pos, pos_styles = to_gpu(data[0], bos, eos, pad)
                xs_neg, ys_neg, ys_in_neg, ys_out_neg, ilens_neg, neg_styles = to_gpu(data[1], bos, eos, pad)
                reverse_pos_styles = cc(torch.LongTensor(pos_styles.cpu().new_zeros(pos_styles.size()))) 
                reverse_neg_styles = cc(torch.LongTensor(neg_styles.cpu().new_ones(neg_styles.size()))) 
                # Positive Part
                enc_outputs, enc_lens = self.encoder(xs_pos, ilens_pos)
                # Style Loss
                enc_representation = get_enc_context(enc_outputs, enc_lens)
                s_logits, s_log_probs, s_pred = self.s_classifier(enc_representation)
                true_s_log_probs =\
                    torch.gather(s_log_probs,dim=1,
                                 index=pos_styles.unsqueeze(1)).squeeze(1)
                s_loss_pos = -torch.mean(true_s_log_probs)*self.config['style_loss_ratio']            
                total_inverse_style_loss += s_loss_pos.item()           
                # Discriminator Loss
                _,_,predict_neg,_ = self.decoder(enc_outputs, enc_lens, reverse_pos_styles, None, 
                                                max_dec_timesteps=self.config['max_dec_timesteps'])
                cycle_ilens = get_prediction_length(predict_neg, eos=self.vocab['<EOS>'])
                # For real positive xs_pos, pos_discri should learn True
                _, realpos_discri_log_probs, _ = self.pos_discri(xs_pos, ilens_pos)
                # For fake negative predict_neg, neg_discri should learn False
                _, fakeneg_discri_log_probs, _ = self.neg_discri(predict_neg, cycle_ilens, need_sort=True)
                #here, use pos_styles since it's an all one tensor, neg_styles: all zero tensor
                true_realpos_discri_log_probs =\
                    torch.gather(realpos_discri_log_probs,dim=1,
                                 index=pos_styles.unsqueeze(1)).squeeze(1) 
                true_fakeneg_discri_log_probs =\
                    torch.gather(fakeneg_discri_log_probs,dim=1,
                                 index=reverse_pos_styles.unsqueeze(1)).squeeze(1) 
                realpos_discri_loss = -torch.mean(true_realpos_discri_log_probs)*self.config['discri_loss_ratio']
                fakeneg_discri_loss = -torch.mean(true_fakeneg_discri_log_probs)*self.config['discri_loss_ratio']
                total_inverse_discri_loss += realpos_discri_loss.item()
                total_inverse_discri_loss += fakeneg_discri_loss.item()
               
                # Negative Part
                enc_outputs, enc_lens = self.encoder(xs_neg, ilens_neg)
                # Style Loss
                enc_representation = get_enc_context(enc_outputs, enc_lens)
                s_logits, s_log_probs, s_pred = self.s_classifier(enc_representation)
                true_s_log_probs =\
                    torch.gather(s_log_probs,dim=1,
                                 index=neg_styles.unsqueeze(1)).squeeze(1)
                s_loss_neg = -torch.mean(true_s_log_probs)*self.config['style_loss_ratio']            
                total_inverse_style_loss += s_loss_neg.item()           
                # Discriminator Loss
                _,_,predict_pos,_ = self.decoder(enc_outputs, enc_lens, reverse_neg_styles, None, 
                                                max_dec_timesteps=self.config['max_dec_timesteps'])
                cycle_ilens = get_prediction_length(predict_pos, eos=self.vocab['<EOS>'])
                # For real negative xs_neg, neg_discri should learn True
                _, realneg_discri_log_probs, _ = self.neg_discri(xs_neg, ilens_neg)
                # For fake positive predict_pos, pos_discri should learn False
                _, fakepos_discri_log_probs, _ = self.pos_discri(predict_pos, cycle_ilens, need_sort=True)
                #here, use pos_styles since it's an all one tensor, neg_styles: all zero tensor
                true_realneg_discri_log_probs =\
                    torch.gather(realneg_discri_log_probs,dim=1,
                                 index=reverse_neg_styles.unsqueeze(1)).squeeze(1) 
                true_fakepos_discri_log_probs =\
                    torch.gather(fakepos_discri_log_probs,dim=1,
                                 index=neg_styles.unsqueeze(1)).squeeze(1) 
                realneg_discri_loss = -torch.mean(true_realneg_discri_log_probs)*self.config['discri_loss_ratio']
                fakepos_discri_loss = -torch.mean(true_fakepos_discri_log_probs)*self.config['discri_loss_ratio']
                total_inverse_discri_loss += realneg_discri_loss.item()
                total_inverse_discri_loss += fakepos_discri_loss.item()
     
                # calculate gradients 
                self.optimizer_m2.zero_grad()
                (s_loss_pos+s_loss_neg).backward()
                self.optimizer_m2.step()
                self.optimizer_m3.zero_grad()
                (realpos_discri_loss+fakeneg_discri_loss+realneg_discri_loss+fakepos_discri_loss).backward()
                self.optimizer_m3.step()
                # print message
                print(f'epoch: {epoch}, [{train_steps + 1}/{total_steps}], '
                      f'style_loss: {(s_loss_pos.item()+s_loss_neg.item())/2:.3f}, '
                      f'pos_d_loss: {(realpos_discri_loss.item()+fakepos_discri_loss.item())/2:.3f}, '
                      f'neg_d_loss: {(realneg_discri_loss.item()+fakeneg_discri_loss.item())/2:.3f}', end='\r')
                # add to logger
                tag = self.config['tag']
                self.logger.scalar_summary(tag=f'{tag}/train/style_classify_pos', 
                                           value=s_loss_pos.item()/self.config['style_loss_ratio'], 
                                           step=(epoch*(self.config['m2_train_freq'])+cnt)*total_steps+train_steps+1)
                self.logger.scalar_summary(tag=f'{tag}/train/style_classify_neg', 
                                           value=s_loss_neg.item()/self.config['style_loss_ratio'], 
                                           step=(epoch*(self.config['m2_train_freq'])+cnt)*total_steps+train_steps+1)
                self.logger.scalar_summary(tag=f'{tag}/train/realneg_dloss', 
                                           value=realneg_discri_loss.item()/self.config['discri_loss_ratio'], 
                                           step=(epoch*(self.config['m2_train_freq'])+cnt)*total_steps+train_steps+1)
                self.logger.scalar_summary(tag=f'{tag}/train/realpos_dloss', 
                                           value=realpos_discri_loss.item()/self.config['discri_loss_ratio'], 
                                           step=(epoch*(self.config['m2_train_freq'])+cnt)*total_steps+train_steps+1)
                self.logger.scalar_summary(tag=f'{tag}/train/fakepos_dloss', 
                                           value=fakepos_discri_loss.item()/self.config['discri_loss_ratio'], 
                                           step=(epoch*(self.config['m2_train_freq'])+cnt)*total_steps+train_steps+1)
                self.logger.scalar_summary(tag=f'{tag}/train/fakeneg_dloss', 
                                           value=fakeneg_discri_loss.item()/self.config['discri_loss_ratio'], 
                                           step=(epoch*(self.config['m2_train_freq'])+cnt)*total_steps+train_steps+1)
            print ()
           
        return ((total_loss/total_steps),(total_style_loss/total_steps),(total_cycle_loss/total_steps),\
                (total_discri_loss/total_steps),((total_inverse_style_loss/total_steps)/self.config['m2_train_freq']),\
                ((total_inverse_discri_loss/total_steps))/self.config['m2_train_freq'])

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
            avgl_re,avgl_s,avgl_c,avgl_dis,avgl_invs,avgl_invdis = self.train_one_epoch(epoch, tf_rate)
            # validation
            avg_valid_loss, wer, prediction_sents, ground_truth_sents, avg_style_acc = self.validation()

            print(f'Epoch: {epoch}, tf_rate={tf_rate:.3f}, train_reconl={avgl_re:.4f},'
                  f'train_stylel:{avgl_s:.4f}, train_disl:{avgl_dis:.4f},'
                  f'train_inv_stylel:{avgl_invs:.4f}, train_inv_disl:{avgl_invdis:.4f},'
                  f'valid_loss={avg_valid_loss:.4f}, val_WER={wer:.4f}, Val_style_acc={avg_style_acc:.4f}')

            # add to tensorboard
            tag = self.config['tag']
            self.logger.scalar_summary(f'{tag}/val/wer', wer, epoch)
            self.logger.scalar_summary(f'{tag}/val/loss', avg_valid_loss, epoch)
            self.logger.scalar_summary(f'{tag}/val/style_acc', avg_style_acc, epoch)

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
