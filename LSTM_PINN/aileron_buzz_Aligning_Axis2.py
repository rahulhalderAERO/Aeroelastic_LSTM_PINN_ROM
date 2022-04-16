#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 11:17:20 2020

@author: rahul
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 02:27:35 2019

@author: rahul
"""



from keras.models import Sequential 
from keras.layers import Dense
import numpy as np
from matplotlib import pyplot as plt
from keras.layers import LSTM
#from keras.layers import Dropout
import pandas as pd
from keras import backend as K
import tensorflow as tf
###############################################################################
output_data = pd.read_csv("output_hadot_P1.csv",skiprows = None , header = None )
input_data= pd.read_csv("input_clcm_P1.csv",skiprows = None , header = None )
output_data1 = pd.read_csv("output_hadot_P1.csv",skiprows = None , header = None )
input_data1 = pd.read_csv("input_clcm_P1.csv",skiprows = None , header = None )

test_step = 25
x_train,x_train_cl,x_train_cm, y_train = [],[],[],[]
x_train1,x_train1_cl,x_train1_cm,y_train1 = [],[],[],[]

output_data = output_data.values
input_data = input_data.values
output_data1 = output_data1.values
input_data1 = input_data1.values
###############################################################################
def max( data_Rahul ):
	max_value = data_Rahul[0]
	for x in data_Rahul:
		if max_value < x:
			max_value = x
	return max_value
###############################################################################
    

for i in range(test_step,len(input_data)):
    x_train_cl.append(input_data[i-test_step:i,0].reshape(-1,1))
x_train_cl = np.array(x_train_cl)  

for i in range(test_step,len(input_data1)):
    x_train1_cl.append(input_data1[i-test_step:i,0].reshape(-1,1))
x_train1_cl = np.array(x_train1_cl) 


for i in range(test_step,len(input_data)):
    x_train_cm.append(input_data[i-test_step:i,1].reshape(-1,1))
x_train_cm = np.array(x_train_cm)  

for i in range(test_step,len(input_data1)):
    x_train1_cm.append(input_data1[i-test_step:i,1].reshape(-1,1))
x_train1_cm = np.array(x_train1_cm)

x_train =  np.concatenate((x_train_cl,x_train_cm),axis=2)
x_train1 =  np.concatenate((x_train1_cl,x_train1_cm),axis=2)  


y_train = output_data[test_step:len(output_data)]
y_train1 = output_data1[test_step:len(output_data1)]
###############################################################################
output_data_P2 = pd.read_csv("output_hadot_P3.csv",skiprows = None , header = None )
input_data_P2= pd.read_csv("input_clcm_P2.csv",skiprows = None , header = None )
output_data1_P2 = pd.read_csv("output_hadot_P3.csv",skiprows = None , header = None )
input_data1_P2 = pd.read_csv("input_clcm_P2.csv",skiprows = None , header = None )

test_step_P2 = 25
x_train_P2,x_train_cl_P2,x_train_cm_P2, y_train_P2 = [],[],[],[]
x_train1_P2,x_train1_cl_P2,x_train1_cm_P2,y_train1_P2 = [],[],[],[]

output_data_P2 = output_data_P2.values
input_data_P2 = input_data_P2.values
output_data1_P2 = output_data1_P2.values
input_data1_P2 = input_data1_P2.values
###############################################################################
  

for i in range(test_step_P2,len(input_data_P2)):
    x_train_cl_P2.append(input_data_P2[i-test_step_P2:i,0].reshape(-1,1))
x_train_cl_P2 = np.array(x_train_cl_P2)  

for i in range(test_step_P2,len(input_data1_P2)):
    x_train1_cl_P2.append(input_data1_P2[i-test_step_P2:i,0].reshape(-1,1))
x_train1_cl_P2 = np.array(x_train1_cl_P2) 


for i in range(test_step_P2,len(input_data_P2)):
    x_train_cm_P2.append(input_data_P2[i-test_step_P2:i,1].reshape(-1,1))
x_train_cm_P2 = np.array(x_train_cm_P2)  

for i in range(test_step_P2,len(input_data1_P2)):
    x_train1_cm_P2.append(input_data1_P2[i-test_step_P2:i,1].reshape(-1,1))
x_train1_cm_P2 = np.array(x_train1_cm_P2)

x_train_P2 =  np.concatenate((x_train_cl_P2,x_train_cm_P2),axis=2)
x_train1_P2 =  np.concatenate((x_train1_cl_P2,x_train1_cm_P2),axis=2)  


y_train_P2 = output_data_P2[test_step_P2:len(output_data_P2)]
y_train1_P2 = output_data1_P2[test_step_P2:len(output_data1_P2)]
###############################################################################

x_train_tot = np.concatenate((x_train,x_train_P2),axis=0)
x_train1_tot = np.concatenate((x_train1,x_train1_P2),axis=0)

###############################################################################

y_train_tot = np.concatenate((y_train,y_train_P2),axis=0)
y_train1_tot = np.concatenate((y_train1,y_train1_P2),axis=0)

###############################################################################
#Traindata_align = (abs(y_train[:,:-2])).reshape(-1,1)
###############################################################################
import random
list=[]
for i in range(500):
    r=random.randint(1,y_train_tot.shape[0])
    if r not in list: list.append(r)
    list_array = np.array(list)

for k in range(len(list_array)):
    val_row = list_array[k]
    y_train_tot[val_row,:] = 0

Traindata_align = (abs(y_train_tot[:,:-2])).reshape(-1,1)
max_train = max(Traindata_align)
y_train_tot = (1/max_train)*y_train_tot 

###############################################################################
model = Sequential()

#Initialise the RNN
#regressor = Sequential()

# Adding the first LSTM layer
model.add(LSTM(units = 50,return_sequences = True, input_shape = (x_train_tot.shape[1], x_train_tot.shape[2])))    
#model.add(Dropout(0.2))
# Adding the Second LSTM layer
model.add(LSTM(units =50,return_sequences = True ))
#model.add(Dropout(0.2))

# Adding the Third LSTM layer
model.add(LSTM(units =50,return_sequences = True ))
#model.add(Dropout(0.2))
#
## Adding the Fourth LSTM layer
model.add(LSTM(units = 50))
#model.add(Dropout(0.2))
#
## Adding the output layer
model.add(Dense(units = 10))
#pred = model(x_train)
dt = np.pi/18;

M_2dof = np.zeros((2,2))
K_2dof = np.zeros((2,2))

M_2dof[0,0] = 1;
M_2dof[0,1] = 1.8;
M_2dof[1,0] = 1.8;
M_2dof[1,1] = 3.48;

K_2dof[0,0] = 1.0;
K_2dof[0,1] = 0.0;
K_2dof[1,0] = 0.0;
K_2dof[1,1] = 3.48;

def PINN_LOSS(y_true, y_pred):
    
    def DIFF_FILTERED(x, y):
        y_true_new = x+0.00001
        y_true_div = tf.math.divide(y, y_true_new)
        difference = 1-y_true_div
        y_true_mult = tf.math.multiply(x, difference)
        return y_true_mult
    
    
    
    loss1 = (K.square(DIFF_FILTERED(y_true[:,0], y_pred[:,0]))) 
    loss2 = (K.square(DIFF_FILTERED(y_true[:,1], y_pred[:,1])))
    loss3 = (K.square(DIFF_FILTERED(y_true[:,2], y_pred[:,2])))
    loss4 = (K.square(DIFF_FILTERED(y_true[:,3], y_pred[:,3]))) 
    loss5 = (K.square(DIFF_FILTERED(y_true[:,4], y_pred[:,4])))
    loss6 = (K.square(DIFF_FILTERED(y_true[:,5], y_pred[:,5])))
    loss7 = (K.square(DIFF_FILTERED(y_true[:,6], y_pred[:,6])))
    loss8 = (K.square(DIFF_FILTERED(y_true[:,7], y_pred[:,7])))
    h_double_dot = (3*y_pred[:,0]-4*y_pred[:,1]+y_pred[:,2])/(2*dt)
    a_double_dot = (3*y_pred[:,3]-4*y_pred[:,4]+y_pred[:,5])/(2*dt)
    loss9 = (K.square(M_2dof[0,0]*h_double_dot+M_2dof[0,1]*a_double_dot+K_2dof[0,0]*y_pred[:,6]+K_2dof[0,1]*y_pred[:,7]-y_true[:,8]))
    loss10 = (K.square(M_2dof[1,0]*h_double_dot+M_2dof[1,1]*a_double_dot+K_2dof[1,0]*y_pred[:,6]+K_2dof[1,1]*y_pred[:,7]-y_true[:,9]))

#    loss4 = (K.square((3/dt)*y_pred[:,0]-(4/dt)*y_pred[:,1]+(1/dt)*y_pred[:,2]-y_pred[:,3]))
    loss_mid = loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8 + loss9 + loss10
#    loss_mid = tf.transpose(loss_mid)
    loss = K.mean((loss_mid),axis=-1)
    return loss

##Compile the RNN
#opt = K.optimizers.SGD(lr=0.01, nesterov=True)
model.compile(optimizer = 'adam' , loss = PINN_LOSS)
#
history = model.fit(x_train_tot,y_train_tot, epochs = 100 , batch_size = 10,shuffle=True)
#predicted_y = model.predict(x_train1)
#predicted_y = max_train*predicted_y
predicted_y = max_train*(model.predict(x_train1_tot))
###############################################################################
output_data_P3 = pd.read_csv("output_hadot_P3.csv",skiprows = None , header = None )
output_data_P3 = output_data_P3.values
y_train_P3 = output_data_P3[test_step_P2:len(output_data_P3)]
y_train2_tot = np.concatenate((y_train1,y_train_P3),axis=0)



plt.plot(np.arange(0,len(y_train2_tot)),y_train2_tot[:,0],label='Actual')
plt.plot(np.arange(0,len(y_train2_tot)),predicted_y[:,0] , label ='Predicted')