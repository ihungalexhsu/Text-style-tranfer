import torch 
import torch.nn as nn
import numpy as np
from model import LSTMAttClassifier, StructureSelfAtt
from dataloader import get_data_loader
from dataset import PickleDataset
from utils import *
import copy
import yaml
import os
from collections import Counter
import pickle
from nltk.corpus import stopwords

class Pretrain_selfatt_classifier(object):
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
            #pretrain_w2v = loadw2vweight(pretrain_w2v_path, self.vocab,
            #                             self.vocab['<PAD>'], False)
            pretrain_w2v, self.vocab = mergew2v(pretrain_w2v_path, self.vocab,
                                                self.vocab['<PAD>'], False)
            print("vocabulary size:", len(self.vocab))

        self.classifier=cc_model(LSTMAttClassifier(vocab_size=len(self.vocab),
                                        embedding_dim=self.config['embedding_dim'],
                                        rnn_hidden_dim=self.config['rnn_hidden_dim'],
                                        dropout_rate=self.config['dropout_p'],
                                        dnn_hidden_dim=self.config['dnn_hidden_dim'],
                                        attention_dim=self.config['att_dim'],
                                        pad_idx=self.vocab['<PAD>'],
                                        pre_embedding=pretrain_w2v,
                                        update_embedding=self.config['update_embedding']))
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
            self.save_model(f'{model_path}-{epoch:03d}')
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
                best_acc = val_acc
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
        total_steps = len(self.train_pos_loader)*2
        total_loss = 0.
        assert len(self.train_pos_loader) >= len(self.train_neg_loader)
        pos_data_iterator = iter(self.train_pos_loader)
        neg_data_iterator = iter(self.train_neg_loader)
        train_steps = 0
        total_instance = 0.
        correct = 0.
        for i in range(len(self.train_pos_loader)):
            try:
                data = next(pos_data_iterator)
            except StopIteration:
                print('StopIteration in pos part')
                pass
            train_steps, total_loss, correct, total_instance =\
                self._train_model_onetime(data, total_loss, epoch, 
                                          train_steps, total_steps,
                                          correct, total_instance)
            try:
                data = next(neg_data_iterator)
            except StopIteration:
                neg_data_iterator = iter(self.train_neg_loader)
                data = next(neg_data_iterator)
            train_steps, total_loss, correct, total_instance =\
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
        train_steps +=1
        return train_steps, total_loss, correct, total_instance

    def _gothrough_onetime(self, data, correct, total_instance):
        bos = self.vocab['<BOS>']
        eos = self.vocab['<EOS>']
        pad = self.vocab['<PAD>']
        xs, ys, ys_in, ys_out, ilens, styles = to_gpu(data, bos, eos, pad)
        logits, log_probs, prediction, att_energy = self.classifier(xs, ilens)
        true_log_probs = torch.gather(log_probs, dim=1,
                                      index=styles.unsqueeze(1)).squeeze(1)
        loss = -torch.mean(true_log_probs)
        total_instance += len(prediction.view(-1))
        correct += (styles.eq(prediction.view(-1).long()).sum()).item()
        return loss, correct, total_instance

    def _log_model(self, epoch, train_steps, total_steps, loss, log_steps): 
        print(f'epoch: {epoch}, [{train_steps+1}/{total_steps}], loss: {loss:.3f}', end='\r')
        
        tag = self.config['tag']
        self.logger.scalar_summary(tag=f'{tag}/train/loss', 
                                   value=loss,
                                   step=log_steps)
        return

    def _get_mean_att(self, att_energy):
        return torch.mean(att_energy, dim=2)

    def get_important_word(self, input_words, att_energy):
        means = self._get_mean_att(att_energy).unsqueeze(2).expand_as(att_energy)
        important_idx = (att_energy >= means)
        emotion_alignment = (torch.sum(important_idx, dim=1) > 0) # (batch x seq_len) 
        important_words = list()
        emotion_alignment = self.tune_alignment(input_words, emotion_alignment)
        for (ws, mask) in zip(input_words, emotion_alignment):
            important_words.append(torch.masked_select(ws, mask).cpu().tolist())
        return important_words, emotion_alignment
    
    def tune_alignment(self, input_words, emotion_alignment):
        if self.stop_words is None:
            stop_words = set(stopwords.words('english'))
            stop_words = stop_words - set('not')
            symbol_list = ['.','!',"'s",'&','...',"'ll",'-','?','``','<UNK>',':',
                           '_num_',"''",'(',')',';','--', ',', "'ve",'$', "'m", "'d"]
            stop_words = stop_words.union(set(symbol_list))
            self.stop_words = stop_words
            idx_stop = list()
            for w in self.stop_words:
                if w in self.vocab.keys():
                    idx_stop.append(self.vocab[w])
            self.stop_words_ids = set(idx_stop)
        new_alignment = emotion_alignment.new_zeros(emotion_alignment.size())
        for i,(ws, masks) in enumerate(zip(input_words, emotion_alignment)):
            for j,(w, m) in enumerate(zip(ws, masks)):
                if (m) and (w.cpu().int().item() in self.stop_words_ids):
                    new_alignment[i,j]=0
                else:
                    new_alignment[i,j]=m
        return new_alignment

    def _valid_feed_data(self, dataloader, total_loss, correct, total_instance):
        all_inputs = []
        all_important_word = []
        for data in dataloader:
            bos = self.vocab['<BOS>']
            eos = self.vocab['<EOS>']
            pad = self.vocab['<PAD>']
            xs, ys, ys_in, ys_out, ilens, styles = to_gpu(data, bos, eos, pad)
            logits, log_probs, prediction, att_energy = self.classifier(xs, ilens)
            true_log_probs = torch.gather(log_probs, dim=1,
                                          index=prediction).squeeze(1)
            loss = -torch.mean(true_log_probs)
            total_instance += len(prediction.view(-1))
            correct += (styles.eq(prediction.view(-1).long()).sum()).item()
            all_inputs = all_inputs + [y.cpu().tolist() for y in ys]
            important_words,_  = self.get_important_word(xs, att_energy)
            all_important_word = all_important_word + important_words
            total_loss += loss.item()

        return all_inputs, all_important_word, correct, total_instance, total_loss

    def validation(self):
        self.classifier.eval()
        total_loss = 0.
        correct = 0.
        total_instance = 0.
        # positive input
        pos_inputs, pos_important_word, correct, total_instance, total_loss =\
            self._valid_feed_data(self.dev_pos_loader, total_loss, correct, total_instance)
        # get sentence
        pos_important_word, pos_inputs = self.idx2sent(pos_important_word, pos_inputs)
        # write file
        if not os.path.exists(self.config['dev_file_path']):
            os.makedirs(self.config['dev_file_path'])
        file_path_pos = os.path.join(self.config['dev_file_path'], 
                                     f'important_words.dev.1.temp')
        file_path_input_pos = os.path.join(self.config['dev_file_path'], 
                                       f'input_sentences.dev.1.temp')
        writefile(pos_important_word, file_path_pos)
        writefile(pos_inputs, file_path_input_pos)
        # negative input
        neg_inputs, neg_important_word, correct, total_instance, total_loss =\
            self._valid_feed_data(self.dev_neg_loader, total_loss, correct, total_instance)
        # get sentence
        neg_important_word, neg_inputs = self.idx2sent(neg_important_word, neg_inputs)
        # write file
        file_path_neg = os.path.join(self.config['dev_file_path'], 
                                     f'important_words.dev.0.temp')
        file_path_input_neg = os.path.join(self.config['dev_file_path'], 
                                       f'input_sentences.dev.0.temp')
        writefile(neg_important_word, file_path_neg)
        writefile(neg_inputs, file_path_input_neg)
        self.classifier.train()
        
        # evaluation
        avg_loss = total_loss / (len(self.dev_pos_loader)+len(self.dev_neg_loader))
        avg_acc = correct/total_instance
        return avg_loss, avg_acc

    def idx2sent(self, all_important_words, all_ys):
        # indexes to sentences
        important_words = to_sents(all_important_words, self.vocab, self.non_lang_syms)
        input_words = to_sents(all_ys, self.vocab, self.non_lang_syms)
        
        return important_words, input_words

    def test(self, state_dict=None):
        # load model
        if state_dict is None:
            self.load_model(self.config['load_model_path'], self.config['load_optimizer'])
        else:
            self.classifier.load_state_dict(state_dict)

        # get test dataset
        root_dir = self.config['dataset_root_dir']
        test_pos_set = self.config['test_pos']
        test_neg_set = self.config['test_neg']
        test_pos_dataset = PickleDataset(os.path.join(root_dir, f'{test_pos_set}.p'), 
                                         config=None, sort=False)
        test_pos_loader = get_data_loader(test_pos_dataset, 
                                          batch_size=1, 
                                          shuffle=False)
        test_neg_dataset = PickleDataset(os.path.join(root_dir, f'{test_neg_set}.p'), 
                                         config=None, sort=False)
        test_neg_loader = get_data_loader(test_neg_dataset, 
                                          batch_size=1, 
                                          shuffle=False)
        self.classifier.eval()
        total_loss = 0.
        correct = 0.
        total_instance = 0.
        # positive input
        pos_inputs, pos_important_word, correct, total_instance, total_loss =\
            self._valid_feed_data(test_pos_loader, total_loss, correct, total_instance)
        # get sentence
        pos_important_word, pos_inputs = self.idx2sent(pos_important_word, pos_inputs)
        # write file
        if not os.path.exists(self.config['test_file_path']):
            os.makedirs(self.config['test_file_path'])
        file_path_pos = os.path.join(self.config['test_file_path'], 
                                     f'important_words.test.1.temp')
        file_path_input_pos = os.path.join(self.config['test_file_path'], 
                                       f'input_sentences.test.1.temp')
        writefile(pos_important_word, file_path_pos)
        writefile(pos_inputs, file_path_input_pos)
        # negative input
        neg_inputs, neg_important_word, correct, total_instance, total_loss =\
            self._valid_feed_data(test_neg_loader, total_loss, correct, total_instance)
        # get sentence
        neg_important_word, neg_inputs = self.idx2sent(neg_important_word, neg_inputs)
        # write file
        file_path_neg = os.path.join(self.config['test_file_path'], 
                                     f'important_words.test.0.temp')
        file_path_input_neg = os.path.join(self.config['test_file_path'], 
                                       f'input_sentences.test.0.temp')
        writefile(neg_important_word, file_path_neg)
        writefile(neg_inputs, file_path_input_neg)
        self.classifier.train()
        
        # evaluation
        avg_acc = correct/total_instance
        print(f"------finish testing------, Testing Accuracy: {avg_acc:.4f}")
        return avg_acc

    def test_ontxt(self, testfilepath, label, write=False, state_dict=None):
        if not os.path.exists('./temp'):
            os.makedirs('./temp')
        if state_dict is None:
            self.load_model(self.config['load_model_path'], self.config['load_optimizer'])
        else:
            self.classifier.load_state_dict(state_dict)
        picklefilepath = transfer_txt2pickle(testfilepath, self.vocab, label)
        test_dataset = PickleDataset(picklefilepath, 
                                     config=None, sort=False)
        test_loader = get_data_loader(test_dataset, 
                                      batch_size=1, 
                                      shuffle=False)
        self.classifier.eval()
        total_loss = 0.
        correct = 0.
        total_instance = 0.
        inputs, important_word, correct, total_instance, total_loss =\
            self._valid_feed_data(test_loader, total_loss, correct, total_instance)
        important_word, inputs = self.idx2sent(important_word, inputs)
        # write file
        if write:
            if not os.path.exists(self.config['test_file_path']):
                os.makedirs(self.config['test_file_path'])
            file_path = os.path.join(self.config['test_file_path'], 
                                    f'important_words.testontxt.{label}.temp')
            file_path_input = os.path.join(self.config['test_file_path'], 
                                               f'input_sentences.testontxt.{label}.temp')
            writefile(important_word, file_path)
            writefile(inputs, file_path_input)
        self.classifier.train()
        # evaluation
        avg_acc = correct/total_instance
        print(f"------finish testing------, Testing Accuracy: {avg_acc:.4f}")
        return avg_acc

    def collect_alignment(self, data_loader):
        self.classifier.eval()
        emo_word_collect = Counter()
        pickle_data = list()
        for data in data_loader:
            bos = self.vocab['<BOS>']
            eos = self.vocab['<EOS>']
            pad = self.vocab['<PAD>']
            xs, ys, ys_in, ys_out, ilens, styles = to_gpu(data, bos, eos, pad)
            logits, log_probs, prediction, att_energy = self.classifier(xs, ilens)
            emo_words, emotion_alignment = self.get_important_word(xs, att_energy)
            emotion_alignment = trim_emotion_alignment(emotion_alignment, ilens)
            emo_words = to_sents(emo_words, self.vocab, self.non_lang_syms)
            for sen in emo_words:
                for w in sen.split(' '):
                    if w != '':
                        emo_word_collect[w]+=1
            ori_data, ori_style = reverse_to_dataformat(ys, styles)
            for i in range(len(ori_data)):
                temp_data={
                    'data':ori_data[i],
                    'label':ori_style[i],
                    'align':emotion_alignment[i].cpu()
                }
                pickle_data.append(temp_data)
        self.classifier.train()
        return pickle_data, emo_word_collect

    def get_alignmentoutput(self):
        self.stop_words = None
        train_pos_data, train_pos_emo = self.collect_alignment(self.train_pos_loader)
        store_root = self.config['store_data_path']
        pos_emo_path = os.path.join(store_root, 'pos_emotion_word.p')
        pickle.dump(train_pos_emo, open(pos_emo_path,'wb'))
        pos_data_path = os.path.join(store_root, 'pos_train_withA.p')
        pickle.dump(list_todictformat(train_pos_data), open(pos_data_path,'wb'))
        del train_pos_data
        del train_pos_emo
        
        train_neg_data, train_neg_emo = self.collect_alignment(self.train_neg_loader)
        neg_emo_path = os.path.join(store_root, 'neg_emotion_word.p')
        pickle.dump(train_neg_emo, open(neg_emo_path,'wb'))
        neg_data_path = os.path.join(store_root, 'neg_train_withA.p')
        pickle.dump(list_todictformat(train_neg_data), open(neg_data_path,'wb'))
        del train_neg_data
        del train_neg_emo
                
        dev_pos_data, _ = self.collect_alignment(self.dev_pos_loader)
        pos_data_path = os.path.join(store_root, 'pos_dev_withA.p')
        pickle.dump(list_todictformat(dev_pos_data), open(pos_data_path,'wb'))
        del dev_pos_data
        
        dev_neg_data, _ = self.collect_alignment(self.dev_neg_loader)
        neg_data_path = os.path.join(store_root, 'neg_dev_withA.p')
        pickle.dump(list_todictformat(dev_neg_data), open(neg_data_path,'wb'))
        del dev_neg_data
        
        root_dir = self.config['dataset_root_dir']
        test_pos_set = self.config['test_pos']
        test_neg_set = self.config['test_neg']
        test_pos_dataset = PickleDataset(os.path.join(root_dir, f'{test_pos_set}.p'), 
                                         config=None, sort=False)
        test_pos_loader = get_data_loader(test_pos_dataset, 
                                          batch_size=1, 
                                          shuffle=False)
        test_neg_dataset = PickleDataset(os.path.join(root_dir, f'{test_neg_set}.p'), 
                                         config=None, sort=False)
        test_neg_loader = get_data_loader(test_neg_dataset, 
                                          batch_size=1, 
                                          shuffle=False)
        test_pos_data, _ = self.collect_alignment(test_pos_loader)
        pos_data_path = os.path.join(store_root, 'pos_test_withA.p')
        pickle.dump(list_todictformat(test_pos_data), open(pos_data_path,'wb'))
        del test_pos_data
        
        test_neg_data, _ = self.collect_alignment(test_neg_loader)
        neg_data_path = os.path.join(store_root, 'neg_test_withA.p')
        pickle.dump(list_todictformat(test_neg_data), open(neg_data_path,'wb'))
        del test_neg_data
        return
