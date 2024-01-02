from typing import List
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pytorch_lightning as pl
import time
import torchmetrics
import utils
from torchvision import transforms
import numpy as np
import datetime
from plot_utils import plot_confusion_matrix
class CarActionModel(pl.LightningModule): 
    def __init__(self,  number_actions: int,action_names: List[str] = None, action_labels:List[int] = None,fc_lr: float = 0.0, cnn_lr: float = 0.0,fc_wd: float = 0.0, cnn_wd: float = 0.0, fc_dropout: float = 0.0, cf_matrix_filename: str = "", 
                 conv1_out_dim = 0,conv1_kernel_dim = 0, conv1_stride_dim = 0, pool1_kernel_dim = 0, pool1_stride_dim =0, conv2_out_dim = 0,conv2_kernel_dim = 0, conv2_stride_dim = 0, pool2_kernel_dim = 0, pool2_stride_dim =0,conv3_out_dim = 0,conv3_kernel_dim = 0, conv3_stride_dim = 0, pool3_kernel_dim = 0, pool3_stride_dim =0) -> None:
        """Car action model init function

        Args:
            number_actions (int): Number of actions
            action_names(List[str]): Action names list of string. Optional
            action_labels(List[int]): List of action labels in integer values
            fc_lr (float, optional): Linear layer learning rate. Defaults to 0.0.
            cnn_lr (float, optional): CNN learning rate. Defaults to 0.0.
            fc_wd (float, optional): Linear layer weight decay. Defaults to 0.0.
            cnn_wd (float, optional): CNN weight decay. Defaults to 0.0.
            fc_dropout (float, optional): Linear layer dropout . Defaults to 0.0.
            cnn_dropout (float, optional): CNN dropout. Defaults to 0.0.
        """
        super().__init__()
        self.number_actions = number_actions
        self.action_names = action_names
        self.action_labels = action_labels
        self.cf_matrix_filename = cf_matrix_filename

        conv1_pad = utils.calculate_padding(96,conv1_kernel_dim,conv1_stride_dim,None)
        self.conv1 = nn.Conv2d(3, conv1_out_dim, kernel_size=conv1_kernel_dim, stride=conv1_stride_dim, padding=conv1_pad)
        
        out = utils.convolution_output_dimension(96,conv1_kernel_dim,conv1_pad,conv1_stride_dim)
        #print(out)
        self.relu = nn.ReLU()
        pool1_pad = utils.calculate_padding(out,pool1_kernel_dim,pool1_stride_dim,None)
        self.pool1 = nn.MaxPool2d(kernel_size=pool1_kernel_dim, stride=pool1_stride_dim,padding=pool1_pad)
        out = utils.convolution_output_dimension(out,pool1_kernel_dim,pool1_pad,pool1_stride_dim)
        #print(out)
        #print("ELLE")
        
        conv2_pad = utils.calculate_padding(out,conv2_kernel_dim,conv2_stride_dim,None)
        self.conv2 = nn.Conv2d(conv1_out_dim, conv2_out_dim, kernel_size=conv2_kernel_dim, stride=conv2_stride_dim, padding=conv2_pad)
        out = utils.convolution_output_dimension(out,conv2_kernel_dim,conv2_pad,conv2_stride_dim)
        #print(out)
        pool2_pad = utils.calculate_padding(out,pool2_kernel_dim,pool2_stride_dim,None)
        self.pool2 = nn.MaxPool2d(kernel_size=pool2_kernel_dim, stride=pool2_stride_dim,padding= pool2_pad)
        out = utils.convolution_output_dimension(out,pool2_kernel_dim,pool2_pad,pool2_stride_dim)
        #print(out)

        conv3_pad = utils.calculate_padding(out,conv3_kernel_dim,conv3_stride_dim,None)
        
        self.conv3 = nn.Conv2d(conv2_out_dim, conv3_out_dim, kernel_size=conv3_kernel_dim, stride=conv3_stride_dim, padding=conv3_pad)
        out = utils.convolution_output_dimension(out,conv3_kernel_dim,conv3_pad,conv3_stride_dim)
        #print(out)
        pool3_pad = utils.calculate_padding(out,pool3_kernel_dim,pool3_stride_dim,None)
        
        self.pool3 = nn.MaxPool2d(kernel_size=pool3_kernel_dim, stride=pool3_stride_dim,padding=pool3_pad)
        out = utils.convolution_output_dimension(out,pool3_kernel_dim,pool3_pad,pool3_stride_dim)
        #print(out)

        self.flatten = nn.Flatten()
        
        # Fully connected layers for classification
        self.fc1 = nn.Linear(out*out*conv3_out_dim, out*out*conv3_out_dim//2)
        self.fc2 = nn.Linear(out*out*conv3_out_dim//2, out*out*conv3_out_dim//2)
        self.fc3 = nn.Linear(out*out*conv3_out_dim//2, 5)  # Adjust the output size to 5 for 5 classes
        #self.fc4 = nn.Linear(2048, 5)  # Adjust the output size to 5 for 5 classes

        self.fc_dropout = nn.Dropout(fc_dropout)

        
        self.fc_lr = fc_lr
        self.fc_wd = fc_wd
        self.cnn_lr = cnn_lr
        self.cnn_wd = cnn_wd
        
        self.val_f1  = torchmetrics.F1Score(task="multiclass", num_classes=number_actions, average='macro')
        self.val_accuracy = torchmetrics.Accuracy(task = "multiclass", num_classes = number_actions)
        
        
        self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=number_actions, average='macro')
        self.test_accuracy = torchmetrics.Accuracy(task = "multiclass", num_classes = number_actions)
        self.y_pred = None
        self.test_labels = None
        
        
        self.save_hyperparameters()
        
    def forward(self, x):
        #x = self.bn1(x)

        x = self.conv1(x)
        #print("conv1",x.shape)
        x = self.relu(x)
        x = self.pool1(x)
        #print("pool",x.shape)
        
        x = self.conv2(x)
        #print("conv2",x.shape)
        x = self.relu(x)
        x = self.pool2(x)
        #print("pool2",x.shape)
       
        x = self.conv3(x)
        #print("conv3",x.shape)
        #x = self.bn(x)
        x = self.relu(x)
        
        x = self.pool3(x)
        #print("pool3",x.shape)
        #time.sleep(1000)

        # Flatten the output
        x = self.flatten(x)
        
        # Fully connected layers for classification
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc_dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc_dropout(x)

        x = self.fc3(x)
        #x = self.relu(x)
        #x = self.fc_dropout(x)

        #x = self.fc4(x)


        return x

    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(self.parameters(),lr=self.fc_lr)
               
        return optimizer
    
    def training_step(self,train_batch,batch_idx):
        image,labels = train_batch
        outputs = self(image)
        
        loss = F.cross_entropy(outputs,labels)
        
        

        self.log_dict({'train_loss':loss},on_epoch=True, batch_size=utils.BATCH_SIZE,on_step=False,prog_bar=True)
        return loss
        

    def validation_step(self, val_batch,idx):
        image, labels = val_batch
        outputs = self(image)
        
        y_pred = outputs.argmax(dim = 1)
       
        loss = F.cross_entropy(outputs,labels)
        self.val_f1(y_pred,labels)
        self.val_accuracy(y_pred,labels)

        self.log_dict({'val_loss':loss,'valid_f1': self.val_f1, 'valid_acc':self.val_accuracy },batch_size=utils.BATCH_SIZE,on_epoch=True, on_step=False,prog_bar=True,enable_graph=False)
        
    def test_step(self, test_batch):
        image, labels = test_batch
        outputs = self(image)
        y_pred = outputs.argmax(dim = 1)
        self.y_pred=torch.cat((self.y_pred,y_pred),dim=0).detach()
        
        self.test_labels= torch.cat((self.test_labels,labels),dim=0).detach()
        loss = F.cross_entropy(outputs,labels)
        self.test_f1(y_pred,labels)
        self.test_accuracy(y_pred,labels)
        
        self.log_dict({'test_loss':loss,'test_f1': self.test_f1, 'test_acc':self.test_accuracy},batch_size=utils.BATCH_SIZE,on_epoch=True, on_step=False,prog_bar=True)
    
    def on_test_end(self) -> None:
        
        plot_confusion_matrix(self.test_labels.cpu().numpy(),self.y_pred.cpu().numpy(),"Car action",0,str(utils.ROOT_FOOLDER)+"/Saves/conf_mat/",False,True,self.action_names,self.action_labels,cf_matrix_filename=self.cf_matrix_filename)
        
    def predict(self,to_predict):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((96, 96)),
            transforms.ToTensor()
        ])
        to_predict = transform(to_predict).unsqueeze(0).cpu()
        
        p = self(to_predict)
        
        _,action = torch.max(p,1)
        return int(action)
        