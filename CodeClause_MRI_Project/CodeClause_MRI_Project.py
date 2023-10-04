#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install torch')


# In[2]:


get_ipython().system('pip install opencv-python')


# In[3]:


# importing packages 

import numpy as np 
import torch 
from torch.utils.data import Dataset, DataLoader
import glob
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix, accuracy_score
import cv2
import sys


# In[4]:


tumor = []
path = './downloads/brain_tumor_dataset/yes/*.jpg'
for f in glob.iglob(path):
    img = cv2.imread(f)
    img = cv2.resize(img, (128,128))
    b, g, r = cv2.split(img) 
    img = cv2.merge([r, g, b])
    tumor.append(img)


# In[5]:


len(tumor)


# In[6]:


for img in tumor:
    print(img.shape)


# In[7]:


healthy = []
path = './downloads/brain_tumor_dataset/no/*.jpg'
for f in glob.iglob(path):
    img = cv2.imread(f)
    img = cv2.resize(img, (128,128))
    b, g, r = cv2.split(img) 
    img = cv2.merge([r, g, b])
    healthy.append(img)


# In[8]:


len(healthy)


# In[9]:


healthy = np.array(healthy)
tumor = np.array(tumor)


# In[10]:


tumor.shape


# In[11]:


healthy.shape


# In[12]:


All = np.concatenate((healthy, tumor))


# In[13]:


All.shape


# In[14]:


plt.imshow(healthy[4])


# In[15]:


def random_plot(healthy, tumor, num=5):
    healthy_imgs = healthy[np.random.choice(healthy.shape[0], num, replace=False)]
    tumor_imgs = tumor[np.random.choice(tumor.shape[0], num, replace=False)]
    
    plt.figure(figsize=(16,9))
    for i in range(num):
        plt.subplot(1, num, i+1)
        plt.title('healthy')
        plt.imshow(healthy_imgs[i])
        
    plt.figure(figsize=(16,9))
    for i in range(num):
        plt.subplot(1, num, i+1)
        plt.title('tumor')
        plt.imshow(tumor_imgs[i])


# In[ ]:





# In[17]:


class Dataset(object):
    
    def __getitem__(self, index):
        raise NotImplementedError
        
        def __len__(self):
            raise NotImplementedError
            
        def __add__(self,other):
            return ConcatDataset([self, other])


# In[18]:


# MRI custom dataset class

class MRI(Dataset):
   
   def __init__(self):
       
       tumor = []
       healthy = []
      
       for f in glob.iglob('./downloads/brain_tumor_dataset/yes/*.jpg'):
           img = cv2.imread(f)
           img = cv2.resize(img, (128,128))
           b, g, r = cv2.split(img) 
           img = cv2.merge([r, g, b])
           tumor.append(img)
   
   
       for f in glob.iglob('./downloads/brain_tumor_dataset/no/*.jpg'):
           img = cv2.imread(f)
           img = cv2.resize(img, (128,128))
           b, g, r = cv2.split(img) 
           img = cv2.merge([r, g, b])
           healthy.append(img)
       
       # images
       healthy = np.array(healthy, dtype= np.float32)
       tumor = np.array(tumor, dtype= np.float32)
       
       #labels (one hot encoding)
       tumor_labels = np.ones(tumor.shape[0], dtype=np.float32)
       healthy_labels = np.zeros(healthy.shape[0], dtype=np.float32)
       
       # concatenating tumour and healthy images as a numpy array
       self.images = np.concatenate((tumor, healthy), axis=0)
       self.labels = np.concatenate((tumor_labels, healthy_labels),axis=0)
        
       
   def __len__(self):
       return self.images.shape[0]
   
   def __getitem__(self, index):
       sample = {'image': self.images[index], 'label': self.labels[index]}
       return sample
       
   # images grey scale so we normalising for CNN 
   
   def normalize(self):
       self .images = self.images / 255.0
       


# In[19]:


mri = MRI()


# In[20]:


mri.normalize()


# In[21]:


random_plot(healthy, tumor)


# In[22]:


mri[5]


# In[23]:


dataloader = DataLoader(mri, shuffle = True)   


# In[25]:


for sample in dataloader:
    img = sample['image'].squeeze()
    #img = img.reshape(img.shape[1], img.shape[2], img.shape[0])
    print(img.shape)
    plt.imshow(img)
    plt.show()
    


# In[26]:


import torch.nn as nn 
import torch.nn.functional as F


# In[128]:


# subclassed from pytorch .Module class to inherit functions
# 2d CNN because we are handling images

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.cnn_model = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(5,5)),
        nn.Tanh(),
        nn.AvgPool2d(kernel_size=2, stride=5),
        nn.Conv2d(in_channels=6, out_channels=16,kernel_size=5),
        nn.Tanh(),
        nn.AvgPool2d(kernel_size=2,stride=5),
        )
        
        self.fc_model = nn.Sequential(
        nn.Linear(in_features=256, out_features=120),
        nn.Tanh(),
        nn.Linear(in_features=120, out_features=48),
        nn.Tanh(),
        nn.Linear(in_features=48, out_features=1))    # we need only 1 neuron 
        
        
    # fwd propagation overrirde 
    
    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1) # flattens 2d arrays
        x = self.fc_models(x)
        
        # sigmoid unit (binary classification)
        x = F.sigmoid(x) 
        
        return x


# In[129]:


# looking into the parameters of the model

model = CNN()


# In[130]:


model


# In[131]:


model.cnn_model


# In[132]:


model.cnn_model[0]


# In[133]:


model.cnn_model[0].weight


# In[134]:


model.cnn_model[0].weight.shape


# In[135]:


model.cnn_model[0].weight[0][0]


# In[136]:


# linear layer 

model.fc_model


# In[137]:


model.fc_model[0].weight


# In[138]:


model.fc_model[0].weight.shape


# In[139]:


# new neural network


# In[140]:


print(torch.cuda.is_available())
print(torch.version.cuda)


# In[141]:


mri_dataset = MRI()
mri_dataset.normalize()
device = torch.device('cpu')
model = CNN().to(device)


# In[142]:


dataloader = DataLoader(mri_dataset,  shuffle=False)


# In[143]:


model.eval()
outputs = []
y_true = []
with torch.no_grad():
    for d in dataloader:
        image = d['image'].to(device)
        label = d['label'].to(device)
        y_hat = model(image)
        outputs.append(y_hat.cpu().detach().numpy())
        y_true.append(label.cpu().detach().numpy())
    


# In[ ]:





# In[ ]:





# In[ ]:




