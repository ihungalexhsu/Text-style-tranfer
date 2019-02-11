import torch 
import torch.nn.functional as F
import numpy as np
from model import Encoder, Decoder, Style_classifier, DenseNet, Domain_discri
from dataloader import get_data_loader
from dataset import PickleDataset
from utils import *
from utils import _seq_mask
import yaml
import os
import pickle

class Style_transfer_proposed(object):
    def __init__(self, config, alpha=10, beta=1, gamma=10, delta=100, zeta=100, load_model=False):

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

    def save_model(self, model_path):
        model_path = (model_path+'_a'+str(self.alpha)+'b'+str(self.beta)+'g'+\
            str(self.gamma)+'d'+str(self.delta)+'z'+str(self.zeta))
        encS_path = model_path+'_encS'
        encC_path = model_path+'_encC'
        decoder_path = model_path+'_decoder'
        s_classifier_path = model_path+'_scler'
        style_mimicker_path = model_path+'_smmer'
        disenS_path = model_path+'_disenS'
        disenC_path = model_path+'_disenC'
        opt1_path = model_path+'_opt1'
        opt2_path = model_path+'_opt2'
        torch.save(self.encS.state_dict(), f'{encS_path}.ckpt')
        torch.save(self.encC.state_dict(), f'{encC_path}.ckpt')
        torch.save(self.decoder.state_dict(), f'{decoder_path}.ckpt')
        torch.save(self.s_classifier.state_dict(), f'{s_classifier_path}.ckpt')
        torch.save(self.style_mimicker.state_dict(), f'{style_mimicker_path}.ckpt')
        torch.save(self.disenS.state_dict(), f'{disenS_path}.ckpt')
        torch.save(self.disenC.state_dict(), f'{disenC_path}.ckpt')
        torch.save(self.optimizer_m1.state_dict(), f'{opt1_path}.opt')
        torch.save(self.optimizer_m2.state_dict(), f'{opt2_path}.opt')
        return

    def load_vocab(self):
        with open(self.config['vocab_path'], 'rb') as f:
            self.vocab = pickle.load(f) # a dict; word to index
        with open(self.config['non_lang_syms_path'], 'rb') as f:
            self.non_lang_syms = pickle.load(f) # an array
        return

    def load_model(self, model_path, load_optimizer):
        model_path = (model_path+'_a'+str(self.alpha)+'b'+str(self.beta)+'g'+\
            str(self.gamma)+'d'+str(self.delta)+'z'+str(self.zeta))
        print(f'Load model from {model_path}')
        encS_path = model_path+'_encS'
        encC_path = model_path+'_encC'
        decoder_path = model_path+'_decoder'
        s_classifier_path = model_path+'_scler'
        style_mimicker_path = model_path+'_smmer'
        disenS_path = model_path+'_disenS'
        disenC_path = model_path+'_disenC'
        opt1_path = model_path+'_opt1'
        opt2_path = model_path+'_opt2'
        self.encS.load_state_dict(torch.load(f'{encS_path}.ckpt'))
        self.encC.load_state_dict(torch.load(f'{encC_path}.ckpt'))
        self.decoder.load_state_dict(torch.load(f'{decoder_path}.ckpt'))
        self.s_classifier.load_state_dict(torch.load(f'{s_classifier_path}.ckpt'))
        self.style_mimicker.load_state_dict(torch.load(f'{style_mimicker_path}.ckpt'))
        self.disenS.load_state_dict(torch.load(f'{disenS_path}.ckpt'))
        self.disenC.load_state_dict(torch.load(f'{disenC_path}.ckpt'))
        if load_optimizer:
            print(f'Load optmizer from {model_path}')
            self.optimizer_m1.load_state_dict(torch.load(f'{opt1_path}.opt'))
            self.optimizer_m2.load_state_dict(torch.load(f'{opt2_path}.opt'))
            if self.config['adjust_lr']:
                adjust_learning_rate(self.optimizer_m1, self.config['retrieve_lr_m1']) 
                adjust_learning_rate(self.optimizer_m2, self.config['retrieve_lr_m2']) 
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
        total_steps = len(self.train_lab_loader)
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
            for train_steps, data in enumerate(self.train_lab_loader):
                bos = self.vocab['<BOS>']
                eos = self.vocab['<EOS>']
                pad = self.vocab['<PAD>']
                xs, _, _, _, ilens, styles = to_gpu(data, bos, eos, pad)
                _, log_probs, prediction = self.textrnn(xs, ilens)
                true_log_probs=torch.gather(log_probs,dim=1,index=styles.unsqueeze(1)).squeeze(1)
                train_loss = -torch.mean(true_log_probs)
                total_train_loss += train_loss.item()
                opt_pretrain_sc.zero_grad()
                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.textrnn.parameters()),5)
                opt_pretrain_sc.step()
                print(f'epoch: {epoch}, [{train_steps+1}/{total_steps}], loss: {train_loss:.4f}', end='\r')
            
            # validation
            #self.textcnn.eval()
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

    def get_data_loaders(self):
        root_dir = self.config['dataset_root_dir']
        
        # get train dataset
        train_set = self.config['train_set']
        self.train_lab_dataset = PickleDataset(os.path.join(root_dir,f'{train_set}.p'),
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

    def get_label_dist(self):
        labelcount = np.zeros(len(self.vocab))
        for token_ids,_ in self.train_lab_dataset:
            for ind in token_ids:
                labelcount[ind] += 1.
        labelcount[self.vocab['<EOS>']]+=len(self.train_lab_dataset)
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
        self.s_classifier = cc_model(Style_classifier(enc_out_dim=encS_out_dim,
                                                      hidden_dim=self.config['s_classifier_hidden_dim'],
                                                      n_layers=self.config['s_classifier_n_layers'],
                                                      out_dim=self.config['n_style_type']))
        print(self.s_classifier)
        self.s_classifier.float()
        self.disenS = cc_model(DenseNet(input_dim=encS_out_dim,
                                        output_dim=encC_out_dim,
                                        hidden_dim_vec=self.config['disen_s_hidden_dim']))
        print(self.disenS)
        self.disenS.float()
        self.disenC = cc_model(DenseNet(input_dim=encC_out_dim,
                                        output_dim=encS_out_dim,
                                        hidden_dim_vec=self.config['disen_c_hidden_dim']))
        print(self.disenC)
        self.disenC.float()
        self.style_mimicker = cc_model(DenseNet(input_dim=(encC_out_dim+1),
                                                output_dim=encS_out_dim,
                                                hidden_dim_vec=self.config['mimicker_hidden_dim']))
        print(self.style_mimicker)
        self.style_mimicker.float()
        self.decoder = cc_model(Decoder(output_dim=len(self.vocab),
                                        embedding_dim=self.config['embedding_dim'],
                                        hidden_dim=self.config['dec_hidden_dim'],
                                        dropout_rate=self.config['dec_dropout_p'],
                                        bos=self.vocab['<BOS>'],
                                        eos=self.vocab['<EOS>'],
                                        pad=self.vocab['<PAD>'],
                                        enc_out_dim=(encS_out_dim+encC_out_dim),
                                        n_styles=self.config['n_style_type'],
                                        style_emb_dim=self.config['style_emb_dim'],
                                        use_enc_init=self.config['use_enc_init'],
                                        use_attention=self.config['use_attention'],
                                        use_style_embedding=False,
                                        ls_weight=self.config['ls_weight'],
                                        labeldist=labeldist,
                                        give_context_directly=True))
        print(self.decoder)
        self.decoder.float()
        self.params_m1=list(self.encS.parameters())+list(self.encC.parameters())+\
            list(self.s_classifier.parameters())+list(self.style_mimicker.parameters())+\
            list(self.decoder.parameters())
        self.params_m2=list(self.disenS.parameters())+list(self.disenC.parameters())
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
        self.load_automatic_style_classifier(self.config['load_pretrained_style_classifier_path'], 
                                             self.config['train_pretrained_style_classifier'])
        return

    def validation(self):
        self.encS.eval()
        self.encC.eval()
        self.s_classifier.eval()
        self.style_mimicker.eval()
        self.decoder.eval()
        self.disenS.eval()
        self.disenC.eval()
        all_prediction, all_ys = [], []
        total_loss = 0.
        total_val_num = 0.
        total_correct_num = 0.
        for step, data in enumerate(self.dev_loader):
            bos = self.vocab['<BOS>']
            eos = self.vocab['<EOS>']
            pad = self.vocab['<PAD>']
            xs, ys, ys_in, ys_out, ilens, styles = to_gpu(data, bos, eos, pad)
            # input the encoder
            enc_outputs, enc_lens = self.encC(xs, ilens)
            reverse_styles = styles.cpu().new_zeros(styles.size())
            for idx, ele in enumerate(styles.cpu().tolist()):
                if not(ele):
                    reverse_styles[idx]=1
            reverse_styles=cc(torch.LongTensor(reverse_styles))
            content_vector = get_enc_context(enc_outputs, enc_lens)
            mimicked_style_vector = self.style_mimicker(torch.cat([reverse_styles.float().view(-1,1),content_vector],dim=1))
            decoder_input = torch.cat([mimicked_style_vector, content_vector], dim=1)
            _, recon_log_probs, prediction, _=\
                self.decoder(decoder_input, enc_lens, reverse_styles, None,
                             max_dec_timesteps=self.config['max_dec_timesteps'])
            seq_len = [y.size(0) + 1 for y in ys]
            mask = cc(_seq_mask(seq_len=seq_len, max_len=recon_log_probs.size(1)))
            loss = (-torch.sum(recon_log_probs*mask))/sum(seq_len)
            total_loss += loss.item()

            # pass the prediction to the pretrained style classifier
            pred_ilens = get_prediction_length(prediction, eos=self.vocab['<EOS>'])
            _, pr_style_probs, pr_style_prediction = self.textrnn(prediction, pred_ilens, need_sort=True)
            correct = pr_style_prediction.cpu().view(-1).eq(reverse_styles.cpu()).sum().item()
            total_val_num +=reverse_styles.size(0)
            total_correct_num +=correct

            all_prediction = all_prediction + prediction.cpu().numpy().tolist()
            all_ys = all_ys + [y.cpu().numpy().tolist() for y in ys]

        self.encS.train()
        self.encC.train()
        self.s_classifier.train()
        self.style_mimicker.train()
        self.decoder.train()
        self.disenS.train()
        self.disenC.train()
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
            self.encC.load_state_dict(state_dict[0])
            self.decoder.load_state_dict(state_dict[1])
            self.style_mimicker.load_state_dict(state_dict[2])

        # get test dataset
        root_dir = self.config['dataset_root_dir']
        test_set = self.config['test_set']
        test_file_name = (self.config['test_file_name']+'_a'+str(self.alpha)+'b'+str(self.beta)+'g'+\
            str(self.gamma)+'d'+str(self.delta)+'z'+str(self.zeta))

        test_dataset = PickleDataset(os.path.join(root_dir, f'{test_set}.p'), 
                                     config=None, sort=False)

        test_loader = get_data_loader(test_dataset, 
                                      batch_size=2, 
                                      shuffle=False)
        self.encS.eval()
        self.encC.eval()
        self.s_classifier.eval()
        self.style_mimicker.eval()
        self.decoder.eval()
        self.disenS.eval()
        self.disenC.eval()
        total_val_num = 0.
        total_correct_num = 0.
        all_prediction,all_ys,all_styles,all_reverse_styles,all_style_predict,\
            all_half_styles,all_half_prediction,all_original_prediction=[],[],[],[],[],[],[],[]
        for step, data in enumerate(test_loader):
            bos = self.vocab['<BOS>']
            eos = self.vocab['<EOS>']
            pad = self.vocab['<PAD>']
            xs, ys, ys_in, ys_out, ilens, styles = to_gpu(data, bos, eos, pad)
            reverse_styles = styles.cpu().new_zeros(styles.size())
            half_styles = styles.cpu().new_zeros(styles.size())
            half_styles = half_styles.float().fill_(0.5)
            for idx, ele in enumerate(styles.cpu().tolist()):
                if not(ele):
                    reverse_styles[idx]=1
            reverse_styles=cc(torch.LongTensor(reverse_styles))
            half_styles=cc(torch.Tensor(half_styles))
            # input the encoder
            enc_outputs, enc_lens = self.encC(xs, ilens)
            content_vector = get_enc_context(enc_outputs, enc_lens)
            mimicked_style_vector = self.style_mimicker(torch.cat([reverse_styles.float().view(-1,1),content_vector],dim=1))
            decoder_input = torch.cat([mimicked_style_vector, content_vector], dim=1)
            _, recon_log_probs, prediction, _=\
                self.decoder(decoder_input, enc_lens, reverse_styles, None,
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
            
            # input the encoder
            enc_outputs, enc_lens = self.encC(xs, ilens)
            content_vector = get_enc_context(enc_outputs, enc_lens)
            mimicked_style_vector = self.style_mimicker(torch.cat([half_styles.float().view(-1,1),content_vector],dim=1))
            decoder_input = torch.cat([mimicked_style_vector, content_vector], dim=1)
            _, recon_log_probs, prediction, _=\
                self.decoder(decoder_input, enc_lens, reverse_styles, None,
                             max_dec_timesteps=self.config['max_dec_timesteps'])
            all_half_styles = all_half_styles + half_styles.cpu().tolist()
            all_half_prediction = all_half_prediction + prediction.cpu().numpy().tolist()

            enc_outputs, enc_lens = self.encC(xs, ilens)
            content_vector = get_enc_context(enc_outputs, enc_lens)
            mimicked_style_vector = self.style_mimicker(torch.cat([styles.float().view(-1,1),content_vector],dim=1))
            decoder_input = torch.cat([mimicked_style_vector, content_vector], dim=1)
            _, recon_log_probs, prediction, _=\
                self.decoder(decoder_input, enc_lens, reverse_styles, None,
                             max_dec_timesteps=self.config['max_dec_timesteps'])
            all_original_prediction = all_original_prediction + prediction.cpu().numpy().tolist()

        self.encS.train()
        self.encC.train()
        self.s_classifier.train()
        self.style_mimicker.train()
        self.decoder.train()
        self.disenS.train()
        self.disenC.train()
        avg_style_correct = total_correct_num / total_val_num
        wer, prediction_sents, ground_truth_sents = self.idx2sent(all_prediction, all_ys)
        _, prediction_sents_half, _ = self.idx2sent(all_half_prediction, all_ys)
        _, prediction_sents_ori, _ = self.idx2sent(all_original_prediction, all_ys)

        with open(f'{test_file_name}.txt', 'w') as f:
            f.write(f'Total sentences: {len(prediction_sents)}, WER={wer:.4f}\n')
            f.write(f'Average style accuracy: {avg_style_correct:.4f} \n')
            for idx, p in enumerate(prediction_sents):
                f.write(f'Pred (style:{all_reverse_styles[idx]},pred_s:{all_style_predict[idx]}) :{p}\n')
                f.write(f'Neutral Pred (style:{all_half_styles[idx]}) :{prediction_sents_half[idx]}\n')
                f.write(f'OriStyle Pred (style:{all_styles[idx]}) :{prediction_sents_ori[idx]}\n')
                f.write(f'Original Sent (style:{all_styles[idx]}) :{ground_truth_sents[idx]}\n')
                f.write('----------------------------------------\n')

        print(f'{test_file_name}: {len(prediction_sents)} utterances, WER={wer:.4f}')
        print(f'Average style accuracy: {avg_style_correct:.4f}')
        return wer

    def _random_target(self, x, num_class=2):
        out = cc(torch.LongTensor(x.size()).random_(0,num_class))
        return out

    def _random_vector(self, x, embedding_activation):
        range_low = -1 if embedding_activation == 'tanh' else 0
        range_high = 1
        tfake = torch.FloatTensor(x.size()).uniform_(range_low, range_high)
        return cc(tfake)

    def train_one_epoch(self, epoch, tf_rate):

        total_steps = len(self.train_lab_loader)
        total_disencS = 0.
        total_disencC = 0.
        total_sloss = 0.
        total_adv_disencS = 0.
        total_adv_disencC = 0.
        total_mimic = 0.
        total_recon = 0.
        if epoch == 0:
            train_counter = 0
        for train_steps, data in enumerate(self.train_lab_loader):
            '''
            if epoch == 0:
                train_counter += 1 
                if train_counter > 2500:
                    train_counter = 0
                    break
            '''
            bos = self.vocab['<BOS>']
            eos = self.vocab['<EOS>']
            pad = self.vocab['<PAD>']
            xs, ys, ys_in, ys_out, ilens, styles = to_gpu(data, bos, eos, pad)
            
            enc_outputs, enc_lens = self.encS(xs, ilens)
            style_vector = get_enc_context(enc_outputs, enc_lens)
            # Style Loss
            _, s_log_probs, _= self.s_classifier(style_vector)
            true_s_log_probs = torch.gather(s_log_probs,dim=1,
                                            index=styles.unsqueeze(1)).squeeze(1)
            s_loss = -torch.mean(true_s_log_probs)*self.beta
            total_sloss += s_loss.item()
            # Disentangle Loss
            predict_content = self.disenS(style_vector)
            enc_outputs, enc_lens = self.encC(xs, ilens)
            content_vector = get_enc_context(enc_outputs, enc_lens)
            predict_style = self.disenC(content_vector)
            rand_vec1 = self._random_vector(style_vector, 'tanh')
            rand_vec2 = self._random_vector(content_vector, 'tanh')
            loss_adv_disencS = torch.mean((predict_content-rand_vec2)**2)*self.delta
            #loss_adv_disencS = -torch.mean((predict_content-content_vector)**2)*self.delta
            loss_adv_disencC = torch.mean((predict_style-rand_vec1)**2)*self.zeta
            #loss_adv_disencC = -torch.mean((predict_style-style_vector)**2)*self.zeta
            total_disencS += loss_adv_disencS.item()
            total_disencC += loss_adv_disencC.item()
            # Mimick loss
            mimicked_style_vector = self.style_mimicker(torch.cat([styles.float().view(-1,1),content_vector],dim=1))
            loss_mimic = torch.mean((mimicked_style_vector-style_vector)**2)*self.gamma
            total_mimic += loss_mimic.item()
            # Reconstruction loss
            decoder_input = torch.cat([mimicked_style_vector, content_vector], dim=1)
            _, recon_log_probs, _, _=\
                self.decoder(decoder_input, enc_lens, styles, (ys_in, ys_out),
                             tf_rate=tf_rate, sample=False,
                             max_dec_timesteps=self.config['max_dec_timesteps'])

            loss_recon = -torch.mean(recon_log_probs)*self.alpha
            total_recon += loss_recon.item()
            # calculate gradients 
            self.optimizer_m1.zero_grad()
            tloss = s_loss+loss_adv_disencS+loss_adv_disencC+loss_mimic+loss_recon
            tloss.backward()
            torch.nn.utils.clip_grad_norm_(self.params_m1, max_norm=self.config['max_grad_norm'])
            self.optimizer_m1.step()
            # print message
            print(f'epoch: {epoch}, [{train_steps + 1}/{total_steps}], recon_loss: {loss_recon:.3f}, '
                  f'style_loss: {s_loss:.3f}, adver_disencS: {-loss_adv_disencS:.3f}, '
                  f'adver_disencC: {-loss_adv_disencC:.3f}, mimic_loss: {loss_mimic:.3f}', end='\r')
            
            # add to logger
            tag = self.config['tag']
            self.logger.scalar_summary(tag=f'{tag}/train/reconloss', 
                                       value=loss_recon.item()/self.alpha, 
                                       step=epoch*total_steps+train_steps+1)
            self.logger.scalar_summary(tag=f'{tag}/train/styleloss', 
                                       value=s_loss.item()/self.beta, 
                                       step=epoch*total_steps+train_steps+1)
            self.logger.scalar_summary(tag=f'{tag}/train/adver_disencS_loss', 
                                       value=loss_adv_disencS.item()/self.delta, 
                                       step=epoch*total_steps+train_steps+1)
            self.logger.scalar_summary(tag=f'{tag}/train/adver_disencC_loss', 
                                       value=loss_adv_disencC.item()/self.zeta, 
                                       step=epoch*total_steps+train_steps+1)
            self.logger.scalar_summary(tag=f'{tag}/train/mimic_loss', 
                                       value=loss_mimic.item()/self.gamma, 
                                       step=epoch*total_steps+train_steps+1)
        print()
        for cnt in range(self.config['m2_train_freq']):
            '''
            if epoch == 0:
                train_counter += 1 
                if train_counter > 3:
                    break
            '''
            for train_steps, data in enumerate(self.train_lab_loader):
                bos = self.vocab['<BOS>']
                eos = self.vocab['<EOS>']
                pad = self.vocab['<PAD>']
                xs, ys, ys_in, ys_out, ilens, styles = to_gpu(data, bos, eos, pad)
                
                enc_outputs, enc_lens = self.encS(xs, ilens)
                style_vector = get_enc_context(enc_outputs, enc_lens)
                predict_content = self.disenS(style_vector)
                enc_outputs, enc_lens = self.encC(xs, ilens)
                content_vector = get_enc_context(enc_outputs, enc_lens)
                predict_style = self.disenC(content_vector)
                #loss_disencS = torch.mean((predict_content-content_vector)**2)*self.delta
                loss_disencS = torch.mean((predict_content-content_vector)**2)
                total_disencS += loss_disencS.item()
                #loss_disencC = torch.mean((predict_style-style_vector)**2)*self.zeta
                loss_disencC = torch.mean((predict_style-style_vector)**2)
                total_disencC += loss_disencC.item()
                # calculate gradients 
                self.optimizer_m2.zero_grad()
                (loss_disencS+loss_disencC).backward()
                self.optimizer_m2.step()
                # print message
                print(f'epoch: {epoch}, [{train_steps + 1}/{total_steps}], '
                      f'disencS_loss: {loss_disencS:.3f}, disencC_loss: {loss_disencC:.3f}', end='\r')
                # add to logger
                tag = self.config['tag']
                self.logger.scalar_summary(tag=f'{tag}/train/disentangle_encS_loss', 
                                           #value=loss_disencS.item()/self.delta, 
                                           value=loss_disencS.item(), 
                                           step=(epoch*(self.config['m2_train_freq'])+cnt)*total_steps+train_steps+1)
                self.logger.scalar_summary(tag=f'{tag}/train/disentangle_encC_loss', 
                                           #value=loss_disencC.item()/self.zeta, 
                                           value=loss_disencC.item(), 
                                           step=(epoch*(self.config['m2_train_freq'])+cnt)*total_steps+train_steps+1)
            print ()

        return ((total_recon/total_steps),(total_sloss/total_steps),\
                (total_adv_disencS/total_steps),(total_adv_disencC/total_steps),\
                (total_mimic/total_steps),((total_disencS/total_steps)/self.config['m2_train_freq']),\
                ((total_disencC/total_steps)/self.config['m2_train_freq']))

    def train(self):

        best_acc = 0.
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
            avgl_re,avgl_s,avg_invencS,avg_invencC,avg_mimic,avg_encS,avg_encC = self.train_one_epoch(epoch, tf_rate)
            # validation
            avg_valid_loss, wer, prediction_sents, ground_truth_sents, avg_style_acc = self.validation()

            print(f'Epoch:{epoch}, tf_rate:{tf_rate:.3f}, Recon_l:{avgl_re:.3f}, '
                  f'Adver_encS:{avg_invencS:.3f}, Adver_encC:{avg_invencC:.3f}, '
                  f'Style_l:{avgl_s:.3f}, Mimic_l:{avg_mimic:.3f}, EncS_l:{avg_encS:.3f}, '
                  f'EncC_l:{avg_encC:.3f}, Valid_loss={avg_valid_loss:.4f}, '
                  f'Val_WER={wer:.4f}, Val_style_acc={avg_style_acc:.4f}')

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
            self.save_model(f'{model_path}_latest')
            if avg_style_acc > best_acc: 
                # save model
                model_path = os.path.join(self.config['model_dir'], self.config['model_name']+'_best')
                best_acc = avg_style_acc
                self.save_model(model_path)
                best_model_enc = self.encC.state_dict()
                best_model_dec = self.decoder.state_dict()
                best_model_mimicker = self.style_mimicker.state_dict()
                print(f'Save #{epoch} model, val_loss={avg_valid_loss:.4f}, '
                      f'WER={wer:.4f}, style_acc={avg_style_acc:.4f}')
                print('-----------------')
                early_stop_counter=0
            if epoch >= self.config['early_stop_start_epoch']:
                early_stop_counter += 1
                if early_stop_counter > self.config['early_stop_patience']:
                    break
        best_model = (best_model_enc, best_model_dec, best_model_mimicker)
        return best_model, best_acc
