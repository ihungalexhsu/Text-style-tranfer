import pickle

def process_generated(generated_path, ori_path, input_label, output_label, vocab):
    inputs = list()
    with open(generated_path, 'r') as f:
        for l in f.readlines():
            inputs.append(l.strip('\n').strip())
    outputs = list()
    with open(ori_path, 'r') as f:
        for l in f.readlines():
            outputs.append(l.strip('\n').strip())
    assert len(inputs) == len(outputs)
    inputs, outputs = eliminate_same(inputs, outputs)
    data = list()
    for i,o in zip(inputs, outputs):
        datum={
            'input':[vocab[w] for w in i.split()],
            'output':[vocab[w] for w in o.split()],
            'input_style': input_label,
            'output_style': output_label
        }
        data.append(datum)
    return data

def eliminate_same(inputs, outputs):
    new_in = list()
    new_ou = list()
    for i,o in zip(inputs, outputs):
        if i==o:
            pass
        else:
            new_in.append(i)
            new_ou.append(o)
    return new_in, new_ou

def process_ori(ori_pickle):
    output = list()
    ori = pickle.load(open(ori_pickle,'rb'))
    for _,v in ori.items():
        datum={
            'input':v['data'],
            'output':v['data'],
            'input_style': v['label'],
            'output_style': v['label']
        }
        output.append(datum)
    return output

def merge(data_ori, data_generated):
    output = dict()
    i = 0
    for d in data_ori:
        output[i]=d
        i+=1
    if data_generated is not None:
        for d in data_generated:
            output[i]=d
            i+=1
    return output

vocab = pickle.load(open('ori_data/vocab_dict.p','rb'))
gen_train_1 = process_generated('generated_data/base.test.0to1.pred',
                                'generated_data/base.test.0.input',
                                1,0, vocab)
gen_train_0 = process_generated('generated_data/base.test.1.input',
                                'generated_data/base.test.1to0.pred',
                                1,0, vocab)
#gen_train_1 = gen_train_1 + gen_train_0
train_1 = process_ori('ori_data/pos_train.p')
pos_train = merge(train_1, gen_train_1)
print(len(pos_train))
pickle.dump(pos_train, open('data/pos_train.p','wb'))


gen_train_0 = process_generated('generated_data/base.test.1to0.pred',
                                'generated_data/base.test.1.input',
                                0,1, vocab)
gen_train_1 = process_generated('generated_data/base.test.0.input',
                                'generated_data/base.test.0to1.pred',
                                0,1, vocab)
#gen_train_0 = gen_train_0+gen_train_1
train_0 = process_ori('ori_data/neg_train.p')
neg_train = merge(train_0, gen_train_0)
print(len(neg_train))
pickle.dump(neg_train, open('data/neg_train.p','wb'))

dev_1 = process_ori('ori_data/pos_dev.p')
pos_dev = merge(dev_1, None)
pickle.dump(pos_dev, open('data/pos_dev.p','wb'))

dev_0 = process_ori('ori_data/neg_dev.p')
neg_dev = merge(dev_0, None)
pickle.dump(neg_dev, open('data/neg_dev.p','wb'))

test_1 = process_ori('ori_data/reference.1.p')
pos_test = merge(test_1, None)
pickle.dump(pos_test, open('data/pos_test.p','wb'))

test_0 = process_ori('ori_data/reference.0.p')
neg_test = merge(test_0, None)
pickle.dump(neg_test, open('data/neg_test.p','wb'))
