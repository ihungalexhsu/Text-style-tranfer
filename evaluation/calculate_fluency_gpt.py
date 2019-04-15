import sys
import numpy as np
import torch
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
import copy

def calculate_fluency(candidate_path):
    candidates = readfile2list(candidate_path)
    tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
    model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
    model.eval()
    perplexities=list()
    if torch.cuda.is_available():
        model.to('cuda')
    for c in candidates:
        input_ids, calculate_index, ori_idx = convert_example2idx(c, tokenizer)
        ppl = torch.exp(do_prediction(model, input_ids, calculate_index, ori_idx, tokenizer)).item()
        #print(c)
        #print(ppl)
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
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    calculate_index = list(range(len(tokens)-1))
    ori_idx = input_ids[1:]
    return input_ids, calculate_index, ori_idx

def do_prediction(model, input_ids, calculate_index, ori_idx, tokenizer):
    with torch.no_grad():
        tokens_tensor = torch.tensor([input_ids])
        if torch.cuda.is_available():
            tokens_tensor = tokens_tensor.to('cuda')
        predictions = model(tokens_tensor)
        lsm = torch.nn.LogSoftmax(dim=2)
        prob_pred = lsm(predictions)
    accumulate_prob = 0.
    for i in calculate_index:
        '''
        print("original word:")
        print(tokenizer.convert_ids_to_tokens([ori_idx[i]])[0])
        print(torch.exp(prob_pred[0, i, ori_idx[i]]).item())
        top5_prob, top5_idx = torch.topk(prob_pred[0,i,:], 5)
        print("top5 word:")
        for cnt in range(len(top5_prob)):
            print(tokenizer.convert_ids_to_tokens([top5_idx[cnt].item()])[0])
            print(torch.exp(top5_prob[cnt]).item())
        print()
        '''
        accumulate_prob += prob_pred[0,i,ori_idx[i]]
    regularized_prob = (-1*accumulate_prob) / float(len(calculate_index))
    return regularized_prob

if __name__=="__main__":
    # main function for testing
    avg_ppl = calculate_fluency(sys.argv[1])
    print(avg_ppl)
