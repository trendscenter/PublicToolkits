'''

This is the source code for the journal paper: 

"Interpretable LSTM Model Reveals Transiently-Realized Patterns of Dynamic Brain Connectivity that Predict Patient Deterioration or Recovery from Very Mild Cognitive Impairment"
Yutong Gao, Noah Lewis, Vince D. Calhoun and Robyn L. Miller

Computers in Biology and Medicine

'''
import torch
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import math
import os, sys
from fastai.layers import *
import numpy as np
import seaborn as sns
from torch import optim
from torch.autograd import Variable
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import fdrcorrection
from scipy.stats import ttest_ind

import numpy as np
import seaborn as sns
import torch
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import math
import os, sys
from fastai.layers import *
from torch.autograd import Variable
import os, sys
from fastai.layers import *
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from sklearn.model_selection import train_test_split

from torch import optim
import torch.backends.cudnn as cudnn
torch.cuda.empty_cache()
cudnn.benchmark = True



class model_LSTMwAtttime_simulation(Module):

    '''
    TA-LSTM
    This model is used for simulation training
    
    Simulation Parameter: 
    f = 20 
    t = 30
    '''
    def __init__(self,t,f):
        super(model_LSTMwAtttime_simulation,self).__init__()
        self.input_size = f
        self.hidden_dim = 16
        self.num_layers = 1

        self.lstm = nn.LSTM(self.input_size,self.hidden_dim,self.num_layers,bidirectional=False,batch_first=True)
        self.out = nn.Linear(t,2)
        self.dropout = nn.Dropout(0.5)
      
    def attention_net(self,x,query,mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)  
        alpha_n = F.softmax(scores, dim=-1) 
        context = torch.matmul(alpha_n, x).sum(2)
        return context,alpha_n
    def forward(self, x):
        r_out,(h_n,h_c) = self.lstm(x,None)
        query = self.dropout(r_out)
        attn_out, alpha_n = self.attention_net(r_out,query)
        out = self.out(attn_out)
        return out
    
    
class model_LSTMwAtttime_dFNC(Module):
    '''
    TA-LSTM
    This model is used for dFNC training for predicting qMCI progression vs recovery 
    
    num_classes: 2
    '''
    def __init__(self,num_classes):
        super(model_LSTMwAtttime_dFNC,self).__init__()
        self.input_size = 1378
        self.hidden_dim = 64
        self.num_layers = 3

        self.lstm = nn.LSTM(self.input_size,self.hidden_dim,self.num_layers,bidirectional=False,batch_first=True)
        self.out = nn.Linear(159,num_classes)
        self.dropout = nn.Dropout(0.5)
        self.gradients = None
        
        
    def activations_hook(self, grad):
        self.gradients = grad
        
    def attention_net(self,x,query,mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)  

        alpha_n = F.softmax(scores, dim=-1)  

        context = torch.matmul(alpha_n, x).sum(2)
        return context,alpha_n
    def forward(self, x):
        
        r_out,(h_n,h_c) = self.lstm(x,None)
        query = self.dropout(r_out)
        attn_out, alpha_n = self.attention_net(r_out,query)
        out = self.out(attn_out)
        return out


class model_m_cnn(Module):
    '''
    M-CNN
    This model is used for dFNC training for predicting qMCI progression vs recovery 
    
    num_classes: 2
    num_tp: 159 when training WWdFNC
    num_tp: 130 when training SWCdFNC
    '''
    def __init__(self,num_classes,num_tp):
        super(model_m_cnn, self).__init__()
        self.conv1 = nn.Conv1d(1378,32,3,padding='same')
        self.conv2 = nn.Conv1d(32,32,3,padding='same')
        self.conv3 = nn.Conv1d(32,32,3,padding='same')

        self.fc1 = nn.Linear(32*num_tp,32)
        self.fc2 = nn.Linear(32,num_classes)

    
    def forward(self , x):
        
        x = F.relu(self.conv1(x))   
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.shape[0],-1) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    

def save_model(epoch, model, optimizer, PATH):

    '''
        save model weight
    '''
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, PATH)


def train_function(model, num_epochs, train_loader, test_loader, optimizer, loss_fn, loop_number, best_acc):
    
    '''
    Deep learning model training
    '''    
    torch.cuda.empty_cache()

    for param_group in optimizer.param_groups:
        param_group["lr"] = 0.01
    for epoch in range(num_epochs):
        
        train_acc = 0.0
        train_loss = 0.0
        for _, (images, labels) in enumerate(train_loader):
            model.train()
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = loss_fn(outputs,labels)
            loss.backward()

            optimizer.step()

            train_loss += loss.item()
            _, prediction = torch.max(outputs.data, 1)

            train_acc += (prediction == labels).sum().item()



        train_acc = train_acc / len(train_loader.dataset)
        train_loss = train_loss / len(train_loader.dataset)


        test_acc = 0.0
        model.eval()

        for i, (images, labels) in enumerate(test_loader):

            images = Variable(images).to(device)
            labels = Variable(labels).to(device)

            outputs = model(images)
            _,prediction = torch.max(outputs.data, 1)
            test_acc += (prediction == labels).sum().item()
        test_acc = test_acc / len(test_loader.dataset)

        Weight_PATH = log_path + "/path_to_save.pth"
        
        if test_acc >= best_acc:
            save_model(epoch,model,optimizer,Weight_PATH)
            best_acc = test_acc


        print("Epoch {}, Train Accuracy: {} , TrainLoss: {} , Test Accuracy: {}".format(epoch, train_acc, train_loss,test_acc))
    print("testing best_acc is ", best_acc)
    print("Checkpoint saved")
    return best_acc


def main_grad(X,y, model, weight_path, random, optimizer):

    '''

    TA-LSTM return saliency map

    '''
    grads = []
    outputs_all = []
    predictions = []
    
    torch_X = torch.Tensor(X)
    torch_y = torch.Tensor(y).to(dtype=torch.long)


    checkpoint = torch.load(weight_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()
    test_acc = 0.0

    X_sali = torch_X
    y_sali = torch_y
    images = Variable(X_sali)
    labels = Variable(y_sali)
    outputs = model(images)
    _,prediction = torch.max(outputs.data, 1)
    test_acc += (prediction == labels).sum().item()
    m = nn.Softmax(dim=1)
    probs = m(outputs).detach().numpy()
    preds = probs[:,1]
    fpr_keras, tpr_keras, _ = metrics.roc_curve(y_sali, preds)
    roc_auc = metrics.auc(fpr_keras, tpr_keras)


    for sub in range(y_sali.shape[0]):
        images = Variable(X_sali[sub:sub+1])
        labels = Variable(y_sali[sub:sub+1])
        images.requires_grad = True
        outputs = model(images)
        pred = outputs.argmax(dim = 1)
        _,prediction = torch.max(outputs, 1)
        predictions.append(prediction)
        outputs_all.append(outputs[0])
        if pred == labels:
            correct_logit = outputs[:,outputs.argmax(dim = 1)]
            grad = torch.autograd.grad(correct_logit, images, retain_graph=True)[0].data.detach().numpy()
            grads.append(grad)
        else:
            print(sub,'predicted wrong, pred',pred,'real label',labels)
    return X_sali, y_sali, outputs_all, grads,predictions



def main(path_X,path_y,path_weight,epoch):

    device = torch.device("cuda:0")

    X = np.load(path_X,allow_pickle=True)
    y = np.load(path_y,allow_pickle=True)

    random_state_ = 42

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state_)

    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train).to(dtype=torch.long)

    X_test = torch.Tensor(X_test)
    y_test = torch.Tensor(y_test).to(dtype=torch.long)

    train_set = TensorDataset(X_train, y_train)
    test_set = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_set, batch_size = 8,shuffle=True, num_workers = 4)
    test_loader = DataLoader(test_set, batch_size = 8, shuffle=False, num_workers = 4)

    model = model_LSTMwAtttime(t,f)
    model.to(device)

    opti = optim.Adam(model.parameters(),lr=0.05, weight_decay=0.0001)
    loss_fn = nn.CrossEntropyLoss()
    log_path = path_weight
    best_acc = train_function(model, epoch, train_loader, test_loader, opti, loss_fn, log_path, 0)



    '''
    TEAM saliency map acquired from the saving trained model
    '''
    model = model_LSTMwAtttime(t,f) 

    weight_path = 'path_to_saved_model'
    optimizer = optim.Adam(model.parameters(),lr=0.1, weight_decay=0.0001)

    '''
    grads return is the saliency map
    '''
    images, labels, outputs_all, grads,predictions = main_grad(X,y, model, weight_path, random_state_, optimizer)


if __name__ == "__main__":
    main()