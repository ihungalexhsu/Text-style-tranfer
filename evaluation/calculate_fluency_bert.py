import sys
import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer,BertForMaskedLM
import copy

MASKED_TOKEN = '[MASK]'

def calculate_fluency(candidate_path):
    candidates = readfile2list(candidate_path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    model.eval()
    perplexities=list()
    if torch.cuda.is_available():
        model.to('cuda')
    for c in candidates:
        features, tokens = convert_example2idx(c, tokenizer)
        ppl = torch.exp(do_prediction(model, features, tokenizer)).item()
        perplexities.append(ppl)
    sorted_ppls = np.sort(perplexities)
    idx_25 = len(sorted_ppls)//4
    idx_75 = idx_25*3
    #return np.average(perplexities)
    return np.average(sorted_ppls[idx_25:idx_75])

def readfile2list(file_path):
    sentences = list()
    with open(file_path, 'r', encoding="utf8", errors='ignore') as f:
        for line in f.readlines():
            sentences.append(line.strip('\n'))
            #sentences.append(line.split('\t')[1].strip('\n'))
    return sentences

def convert_example2idx(example, tokenizer):
    '''
    example is a string
    '''
    tokens = tokenizer.tokenize(example)
    input_tokens = []
    segment_ids = []
    input_tokens.append('[CLS]')
    segment_ids.append(0)
    for token in tokens:
        input_tokens.append(token)
        segment_ids.append(0)
    input_tokens.append("[SEP]")
    segment_ids.append(0)
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    features = create_sequential_mask(input_tokens, input_ids, segment_ids, tokenizer)
    return features, input_tokens

def create_masked_lm_prediction(input_ids, mask_id, mask_position, mask_count=1):
    new_input_ids = copy.deepcopy(list(input_ids))
    masked_lm_labels = []
    masked_lm_positions = list(range(mask_position, mask_position+mask_count))
    for i in masked_lm_positions:
        new_input_ids[i] = mask_id
        masked_lm_labels.append(input_ids[i])
    return new_input_ids, masked_lm_positions, masked_lm_labels

class InputFeatures(object):
    def __init__(self, input_ids, segment_ids, masked_lm_positions, masked_lm_ids):
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_ids = masked_lm_ids

def is_subtoken(x):
    return x.startswith('##')

def create_sequential_mask(input_tokens, ori_input_ids, segment_ids, tokenizer):
    #print(input_tokens)
    features = []
    i = 1
    mask_id = tokenizer.convert_tokens_to_ids([MASKED_TOKEN])[0]
    while i < len(input_tokens) - 1:
        mask_count = 1
        while is_subtoken(input_tokens[i+mask_count]):
            mask_count += 1
        input_ids, masked_lm_positions, masked_lm_labels =\
            create_masked_lm_prediction(ori_input_ids, mask_id, i, mask_count)
        input_ids = torch.LongTensor(input_ids)
        segment_ids = torch.LongTensor(segment_ids)
        feature = InputFeatures(input_ids, segment_ids, 
                                masked_lm_positions, masked_lm_labels)
        features.append(feature)
        i+=mask_count
    return features

def do_prediction(bert_model, features, tokenizer):
    with torch.no_grad():
        token_tensors = []
        segment_tensors = []
        masked_ids = []
        masked_positions = []
        for feature in features:
             token_tensors.append(feature.input_ids)
             segment_tensors.append(feature.segment_ids)
             masked_ids.append(feature.masked_lm_ids[0])
             masked_positions.append(feature.masked_lm_positions[0])
        token_tensors = torch.stack(token_tensors,dim=0)
        segment_tensors = torch.stack(segment_tensors,dim=0)
        if torch.cuda.is_available():
            token_tensors = token_tensors.to('cuda')
            segment_tensors = segment_tensors.to('cuda')
        predictions = bert_model(token_tensors, segment_tensors)
        lsm = torch.nn.LogSoftmax(dim=2)
        prob_pred = lsm(predictions)
    accumulate_prob = 0.
    for i, masked_id in enumerate(masked_ids):
        print("original word:")
        print(tokenizer.convert_ids_to_tokens([masked_id])[0])
        print(torch.exp(prob_pred[i,masked_positions[i], masked_id]).item())
        top5_prob, top5_idx = torch.topk(prob_pred[i,masked_positions[i],:], 5)
        print("top5 word:")
        for cnt in range(len(top5_prob)):
            print(tokenizer.convert_ids_to_tokens([top5_idx[cnt].item()])[0])
            print(torch.exp(top5_prob[cnt]).item())
        accumulate_prob += prob_pred[i, masked_positions[i], masked_id]
    regularized_prob = (-1*accumulate_prob) / float(len(masked_ids))
    return regularized_prob

if __name__=="__main__":
    # main function for testing
    avg_ppl = calculate_fluency(sys.argv[1])
    print(avg_ppl)
