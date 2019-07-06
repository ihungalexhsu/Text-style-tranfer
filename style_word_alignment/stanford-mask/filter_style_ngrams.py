import sys
import random
import pickle

DATASET='yelp'
data_prefix='../../data/'+DATASET+'/proc.sentiment.train.'
result_prefix=DATASET+'_stanford/'+DATASET+'.salience.'
data_file_num=2

def load_data(file_name):
    f=open(file_name,'r')
    tmp=[]
    for line in f:
        if('url' in line):
            continue
        line=line.strip()
        tmp.append(line)
    random.shuffle(tmp)
    return tmp[:min(len(tmp),5000000000)]

def get_dict(sen_array):
    # Get the n-gram statistics of the file. Here, n is from 1 to 5.
    tmp_dict={}
    for i in sen_array:
        sens=i.strip().split(' ')
        for n in range(1,5):
            for l in range(0,len(sens)-n+1):
                tmp=' '.join(sens[l:l+n])
                if(tmp_dict.get(tmp)!=None):
                    tmp_dict[tmp]+=1
                else:
                    tmp_dict[tmp]=1
    return tmp_dict

name_array=[] # store the "name of data file" to process
for i in range(0,data_file_num):
    name_array.append(data_prefix+str(i))

num=0
for tag in name_array:
    negative_array=[] # relative negative
    positive_array=load_data(tag) # load relative positive data
    for i in name_array:
        if tag!=i:
            negative_array+=load_data(i) # load relative negative data
    neg_dict=get_dict(negative_array) # n-gram from neg
    pos_dict=get_dict(positive_array) # n-gram from pos
    tf_idf={}
    for i in pos_dict.keys(): # calculate relative score for each n-gram with smoothing
        if(neg_dict.get(i)!=None):
            tf_idf[i]=(pos_dict[i]+1.0)/(neg_dict[i]+1.0)
        else:
            tf_idf[i]=(pos_dict[i]+1.0)/(1.0)
    tf_dif1=sorted(tf_idf.items(), key=lambda x:x[1], reverse=True) # sort the result
    fw=open(result_prefix+'tf_idf.'+str(num),'w') # write file name
    for i in tf_dif1:
        fw.write(i[0]+'\t'+str(i[1])+'\n')
    pickle.dump(tf_dif1, open(result_prefix+'tf_idf.'+str(num)+'.p','wb'))
    num+=1
