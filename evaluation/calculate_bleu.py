from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
import numpy as np
import sys

def BLEU(candidate_path, reference_path):
    candidates = readfile2list(candidate_path)
    references = readfile2list(reference_path)
    #micro bleu
    BleuPerSen = list()
    for c,r in zip(candidates, references):
        ref = [r.lower().strip().split()]
        #can = (c.lower().split('\t')[1]).strip().split()
        can = c.lower().strip().split()
        cc = SmoothingFunction()
        score = sentence_bleu(ref, can, smoothing_function=cc.method2)
        BleuPerSen.append(score)
    
    #corpus bleu
    ref = list()
    can = list()
    for c,r in zip(candidates, references):
        ref.append([r.lower().strip().split()])
        #can.append((c.lower().split('\t')[1]).strip().split())
        can.append(c.lower().strip().split())
    cc = SmoothingFunction()
    score = corpus_bleu(ref, can, smoothing_function=cc.method2)
    return score*100, np.mean(np.asarray(BleuPerSen))*100

def readfile2list(file_path):
    sentences = list()
    with open(file_path, 'r', encoding="utf8", errors='ignore') as f:
        for line in f.readlines():
            sentences.append(line)
    return sentences

if __name__=="__main__":
    # main function for testing
    corpus_score, micro_score = BLEU(sys.argv[1], sys.argv[2])
    print(corpus_score)
    print(micro_score)
