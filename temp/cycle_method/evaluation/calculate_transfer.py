import fastText
import sys
from nltk.tokenize import word_tokenize

def Transferability(candidate_path, model_path, true_label):
    '''
    true_label is either "__label__1" or "__label__0"
    '''
    m = fastText.load_model(model_path)
    candidates = readfile2list(candidate_path)
    correct = 0.
    for c in candidates:
        label, confidence = m.predict(' '.join(word_tokenize(c.strip('\n').strip())))
        if label[0]==true_label:
            correct += 1
    return correct*100/len(candidates)

def readfile2list(file_path):
    sentences = list()
    with open(file_path, 'r', encoding="utf8", errors='ignore') as f:
        for line in f.readlines():
            sentences.append(line)
            #sentences.append(line.split('\t')[1])
    return sentences

if __name__=="__main__":
    # main function for testing
    acc = Transferability(sys.argv[1], sys.argv[2], sys.argv[3])
    print(acc)
