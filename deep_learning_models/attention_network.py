import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import numpy as np
import os

import pandas as pd 

#load data
import pickle
import numpy as np
from scipy.stats import entropy
import os
#%matplotlib inline
import sys
sys.path.insert(1, '../')
import utilities

torch.manual_seed(1)


r= os.getcwd()
print(os.getcwd())
data_path = '../data_info/loaded_pickles_nips19/'


reviewer_representation = pickle.load( open( data_path+ 'dict_all_reviewer_lda_vectors.pickle', "rb" ))
paper_representation = pickle.load( open( data_path + 'dict_paper_lda_vectors.pickle', "rb" ))


#df= pd.read_csv('../../neurips19_anon/anon_bids_file')

bds_path='~/arcopy/neurips19_anon/anon_bids_file'
df= utilities.get_bids_data_filtered(bds_path)


size= len(df.index)
print('data size is ', size)
df= df[:int(0.01 * size)]
print(df.sample(4))

def prepare_data(submitter, reviewer, df, gpu_flag=False):
    train_data_sub = []
    train_data_rev = []
    submit = submitter.keys()
    submitter_ids = []
    reviewer_ids = []
    rev = reviewer.keys()
    labels = []
    for i in range(len(df)):
        pid_curr= str(df.iloc[i]['pid'])
        rev_curr=    str(df.iloc[i]['anon_id']) 
        if pid_curr  in submit and rev_curr in reviewer:
            train_data_sub.append(torch.tensor(submitter[pid_curr],requires_grad=True))#.cuda()
            train_data_rev.append(torch.tensor(reviewer[rev_curr], requires_grad=True))#.cuda()
            idx = int(df.iloc[i]['bid'])
            temp = torch.LongTensor([0, 0, 0, 0])#.cuda()
            for i in range(4):
                if i == idx:
                    temp[i] = 1
            labels.append(temp)
            submitter_ids.append(df.iloc[i]['pid'])
            reviewer_ids.append(df.iloc[i]['anon_id'])
    return train_data_sub, train_data_rev, labels, submitter_ids, reviewer_ids

def get_batch_eval(paper_emb, rev_emb, trg_value, idx, batch_size):
   # paper_lines = Variable(torch.stack(paper_emb[idx:idx+batch_size]).squeeze(), requires_grad=True)
    paper_lines = Variable(torch.stack(paper_emb[idx:idx+batch_size]), requires_grad=True).permute(1,0)
    #review_emb = Variable(torch.stack(rev_emb[idx:idx+batch_size]).squeeze(), requires_grad=True)
    #review_emb = Variable(torch.cat(rev_emb[idx:idx+batch_size],dim=0), requires_grad=True)
    #torch.cat(data,dim=0)
    reviewer_paper= torch.tensor(rev_emb[idx:idx+batch_size][-1],requires_grad=True).squeeze().T#permute(1,0)

    trg = torch.stack(trg_value[idx:idx+batch_size]).squeeze()
  #  return paper_lines, review_emb, trg
    return paper_lines.float(), reviewer_paper.float(), trg    




class Match_Classify(nn.Module):
    def __init__(self,submitter_emb_dim, reviewer_emb_dim,
                 batch_size, n_classes,):
        super(Match_Classify, self).__init__()
        
        self.submitter_emb_dim = submitter_emb_dim
        self.reviewer_emb_dim = reviewer_emb_dim
        
        #you need self.num_topics for now. You can call it in semantic embedding space  in future
        self.num_topics= 25
        self.attention_matrix_size= 40

        self.n_classes = n_classes
        self.batch_size = batch_size        
        

        self.W_Q= Variable(torch.rand(self.attention_matrix_size, self.num_topics), requires_grad=True)
        self.W_K= Variable(torch.rand(self.attention_matrix_size, self.num_topics),requires_grad=True)
        self.W_V = Variable(torch.rand(self.attention_matrix_size, self.num_topics),requires_grad=True)
        
        self.combined = nn.Linear(self.submitter_emb_dim, 25)
        self.w_submitted= nn.Linear(self.num_topics,self.num_topics)
        self.w_out= nn.Linear(self.num_topics, n_classes)

    #forward is actually defining the equation
    def forward(self, submitter_emb, reviewer_emb):    

        Q= torch.matmul (self.W_Q,submitter_emb)

        K= torch.matmul (self.W_K, reviewer_emb)
        
        V= torch.matmul( self.W_V, reviewer_emb)

        s= torch.matmul(Q.T,K).squeeze()
        softm= torch.nn.Softmax()
        z=(softm(s))
        z= torch.sum(z * V,dim=0)

        try:
            x= torch.matmul (z,reviewer_emb.T).unsqueeze(dim=0)
            
        except RuntimeError:
            x= (z*reviewer_emb.T).unsqueeze(dim=0)       
    
        combine= x + self.w_submitted(submitter_emb.T)
        out= self.w_out(combine)
        op = F.softmax(out,dim=1)
        return op


#wonder what the shapes of submitter hid and reviewer hid can be::
data_sub, data_rev, data_y, submitter_ids, reviewer_ids = prepare_data(paper_representation, reviewer_representation, df)
train_ratio = int(0.8*len(data_sub))
test_ratio = len(data_sub) - train_ratio

train_sub = data_sub[:train_ratio]
test_sub = data_sub[train_ratio:]

train_rev = data_rev[:train_ratio]
test_rev = data_rev[train_ratio:]

y_train = data_y[:train_ratio]
y_test = data_y[train_ratio:]


"CHANGING BATCH SIZE TO 1 FOR NOW . CAREFUL. IT IS PURE SGD"
batch_size=1
model = Match_Classify(25,25,batch_size,4) 
  
criterion = torch.nn.MSELoss(size_average = False) 
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01) 
  

epochs=10
#TRAINING MODULE
losses= []
for e_num in range(epochs):
    loss_ep = 0
    correct=0
    wrong=0
    for i in range(0, len(y_train), batch_size):
        mini_batch_submitted_paper, mini_batch_reviewer_paper, y = get_batch_eval(train_sub, train_rev, y_train, i, batch_size) 
        optimizer.zero_grad()
        prediction = model(mini_batch_submitted_paper, mini_batch_reviewer_paper).float()
    
        loss = criterion(prediction, y.float())
        loss_ep += loss.item()
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()

        class_label = prediction.argmax(dim=1)
        trg_label = y.argmax()
        if class_label == trg_label:
            correct += 1
        else:
            wrong += 1
    
    losses.append(loss_ep/batch_size)
    print("Epoch:", e_num, " Loss:", losses[-1], ": Train Accuracy:", correct/len(y_train))


#code for evaluation part 
with torch.no_grad():
    model.eval()
    correct=0
    wrong=0
    loss_test=0
    for i in range(0, len(y_test)):
        prediction = model(test_sub[i].float(), test_rev[i].T.float()).float()
        loss = criterion(prediction, y_test[i].float())
        loss_test += loss.item()
    
        class_label = prediction.argmax(dim=1)
        trg_label = y_test[i].argmax()
        if class_label == trg_label:
            correct += 1
        else:
            wrong += 1

    print("Test Loss:", loss_test/len(y_test), ": Test Accuracy:", correct/len(y_test))




