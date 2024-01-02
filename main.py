import utils
import model

from itertools import product
from car_action_datamodule import CarActionDataModule
import tqdm
from model import CarActionModel
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from termcolor import colored
import os
import subprocess
import threading
import torch
from pytorch_lightning.profilers import PyTorchProfiler

import pytorch_lightning as pl



hyp_comb = list(product(utils.FC_LR, utils.FC_WD, utils.FC_DROPOUT, utils.CNN_LR,utils.CNN_WD,utils.NUM_EPOCHS,
                        utils.conv1_out_dim,utils.conv1_kernel_dim,utils.conv1_stride_dim,
                        utils.POOL1_KERNEL_DIM,utils.POOL1_STRIDE_DIM,
                        utils.conv2_out_dim,utils.conv2_kernel_dim,utils.conv2_stride_dim,
                        utils.POOL2_KERNEL_DIM,utils.POOL2_STRIDE_DIM,
                        utils.conv3_out_dim,utils.conv3_kernel_dim,utils.conv3_stride_dim,
                        utils.POOL3_KERNEL_DIM,utils.POOL3_STRIDE_DIM))


car_action_datamodule = CarActionDataModule(utils.ROOT_FOOLDER/"train_crop.csv",utils.ROOT_FOOLDER/"test_crop.csv")


for hyperparameter in tqdm.tqdm(hyp_comb,colour="yellow", desc="Tried combinations"):
    
    fc_lr = hyperparameter[0]
    fc_wd = hyperparameter[1]
    fc_dropout = hyperparameter[2]

    cnn_lr = hyperparameter[3]
    cnn_wd = hyperparameter[4]
    num_epochs = hyperparameter[5]
    
    conv1_out_dim = hyperparameter[6]
    conv1_kernel_dim = hyperparameter[7]
    conv1_stride_dim = hyperparameter[8]

    pool1_kernel_dim = hyperparameter[9]
    pool1_stride_dim = hyperparameter[10]

    conv2_out_dim = hyperparameter[11]
    conv2_kernel_dim = hyperparameter[12]
    conv2_stride_dim = hyperparameter[13]

    pool2_kernel_dim = hyperparameter[14]
    pool2_stride_dim = hyperparameter[15]

    conv3_out_dim = hyperparameter[16]
    conv3_kernel_dim = hyperparameter[17]
    conv3_stride_dim = hyperparameter[18]

    pool3_kernel_dim = hyperparameter[19]
    pool3_stride_dim = hyperparameter[20]


    filename = str(conv1_out_dim) + ", " + str(conv1_kernel_dim) + ", "+ str(conv1_stride_dim) + ", "+ str(pool1_kernel_dim)+ ", "+ str(pool1_stride_dim) + ", " + str(conv2_out_dim) + ", " + str(conv2_kernel_dim) + ", "+ str(conv2_stride_dim) + ", "+ str(pool2_kernel_dim)+ ", "+ str(pool2_stride_dim) + ", "+str(conv3_out_dim) + ", " + str(conv3_kernel_dim) + ", "+ str(conv3_stride_dim) + ", "+ str(pool3_kernel_dim)+ ", "+ str(pool3_stride_dim) + ", " + str(fc_dropout)
    print(filename)
    if(filename+".ckpt" in os.listdir(utils.CKPT_SAVE_DIR_NAME)):
        print(colored("SKIPPO","red"))
        continue
    
    
    print(colored("Built data","green"))

    logger = TensorBoardLogger(save_dir=str(utils.LOG_SAVE_DIR_NAME),name= filename)
    #profiler = PyTorchProfiler(on_trace_ready = torch.profiler.tensorboard_trace_handler("/home/bruno/Desktop/ML_EX2/Saves/tb_logs/profiler0"),trace_memory = True)

    trainer = pl.Trainer(log_every_n_steps=140,max_epochs = num_epochs,callbacks=[EarlyStopping(monitor="val_loss", patience=3,mode='min'), ModelCheckpoint(filename= filename,monitor='val_loss',save_top_k=1,every_n_epochs=1,mode='min',save_weights_only=False,verbose=True,dirpath=utils.CKPT_SAVE_DIR_NAME)],logger=logger)#,accelerator='cuda')
    print(colored("Built logger and trainer","green"))
    labels_name = ["Nothing", "Left", "Right", "Gas", "Brake"]
    car_action_model = CarActionModel(5,labels_name, [0,1,2,3,4],fc_lr,cnn_lr,fc_wd,cnn_wd,fc_dropout,cf_matrix_filename= filename,
                                      conv1_out_dim=conv1_out_dim,conv1_kernel_dim=conv1_kernel_dim, conv1_stride_dim=conv1_stride_dim,
                                      pool1_kernel_dim=pool1_kernel_dim, pool1_stride_dim=pool1_stride_dim,
                                      
                                      conv2_out_dim=conv2_out_dim,conv2_kernel_dim=conv2_kernel_dim, conv2_stride_dim=conv2_stride_dim,
                                      pool2_kernel_dim=pool2_kernel_dim, pool2_stride_dim=pool2_stride_dim,
                                      
                                      conv3_out_dim=conv3_out_dim,conv3_kernel_dim=conv3_kernel_dim, conv3_stride_dim=conv3_stride_dim,
                                      pool3_kernel_dim=pool3_kernel_dim, pool3_stride_dim=pool3_stride_dim)
    
    print(colored("Starting training...","green"))
    try:

        trainer.fit(car_action_model,datamodule = car_action_datamodule)
        
    except torch.cuda.OutOfMemoryError:
        print(colored("Out of memory detected, skipping","red"))
        continue
    
    print(colored("Starting testing...","green"))
   
    #trainer.test(car_action_model,datamodule = car_action_datamodule)#,ckpt_path="best")
    
    #original_path = str(utils.CKPT_SAVE_DIR_NAME/str(filename))

    #utils.save_last_ckpt_path(original_path)
    #command_thread = threading.Thread(target=subprocess.Popen(['python', "play_policy_template.py"]))
    
    
#subprocess.run(['bash',"alert.sh"])
print(colored("Pipeline over","red"))
