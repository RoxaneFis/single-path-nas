import pandas as pd
import numpy as np
from numpy import loadtxt
import tensorflow as tf
           

class TreatNeuralNetwork_searchphase():
    " This class is used to preprocess the NN list for the predictor"
    def __init__(self, NN,std):
        self.list = NN
        self.columns_names = ['exp', 'c_out', 's', 'k', 'skip', 'convtype','use_bias']
        self.conv_types = ['conv', 'dw', 'inv']
        self.values_to_keep = ['FLOPS', 'weights', 'tensor_in', 'tensor_out', 'hidden_dim', 'k2', 'skip']
        self.max_blocks=37  # FIXME : hardcoded
        self.std=std

    def build_list(self):
        biases ={'conv': [False,], 'conv_norelu': [False,], "dw":[False,], 'inv':[False,False, False]}
        NN_advanced = self.list.copy()
        for ii in range (len(NN_advanced)):
            NN_advanced[ii].append(biases[NN_advanced[ii][-1]])
        self.list = NN_advanced


    @staticmethod
    def conv_flops(hout,cin,cout,k,tensor_in,skip=0):
        "Calculate the number of FLOPS in a convolution"
        flops = 2*k*k*cin*hout*hout*cout
        if skip ==1:
            flops += tensor_in
        return flops

    @staticmethod
    def dw_flops(hout,c,k,tensor_in,skip=0, mult=1):
        "Calculate the number of FLOPS in a depthwise convolution"
        flops = 2*k*k*c*hout*hout
        if skip == 1:
            flops +=tensor_in
        return flops

    def calculate_flop(self, convtype, hin,hout,cin,cout,k,exp,tensor_in, skip=0, mult=1):
        """Calculate the number of FLOPS in one layer
        Args :
            -convtype : ['conv', 'dw', 'inv']
            -hin/hout : size of the input and output height (=width)
            -cin/cout : size of the input and ouput number og layers
            -k : kernel size
            -exp : expansion ration
            -tensor_in : size of the input tensor
            -skip : whether there is a skip connection (input added to the output)
            -mult : multiplication of the block (FIXME :NOT USED)
        """

        if convtype=='conv':
            return self.conv_flops(hout,cin,cout,k, tensor_in, skip)
        elif convtype=='dw':
            return self.dw_flops(hout,cin,k,tensor_in,skip)
        elif convtype=='inv':
            hout_pointwise = hin
            return self.conv_flops(hout=hout_pointwise, cin=cin,cout=cin*exp,k=1, tensor_in=tensor_in, skip=skip) + \
        self.dw_flops(hout=hout,c=cin*exp,k=k,tensor_in=0,skip=0) + \
        self.conv_flops(hout=hout,cin=cin*exp,cout=cout,k=1, tensor_in=0, skip=0)

    @staticmethod
    def conv_weights(cin,cout,k,use_bias=True):
        "Calculate the number of weights in a convolution"
        if not use_bias :
            return k*k*cin*cout
        return (k*k*cin+1)*cout

    @staticmethod
    def dw_weights(cin,k,mult=1, use_bias=True):
        "Calculate the number of weights in a depthwise convolution"
        if not use_bias :
            return k*k*cin
        return (k*k+1)*cin

    def calculate_weights(self, convtype,cin,cout,k,exp,use_bias, mult=1):
        """Calculate the number of weights in one layer
        Args :
            -convtype : ['conv', 'dw', 'inv']
            -cin/cout : size of the input and ouput number og layers
            -k : kernel size
            -exp : expansion ration
            -use_bias : bool if we add some bias weights
            -mult : multiplication of the block (FIXME NOT USED)
        """
        if convtype=='conv':
            return self.conv_weights(cin,cout,k, use_bias[0])
        elif convtype=='dw':
            return self.dw_weights(cin,k, use_bias[0])
        elif convtype=='inv':
            if len(use_bias) == 2 : #expand ratio ==1 : there is no expand layer (tot : 2 layers)
                return self.dw_weights(cin*exp,k,mult,use_bias[0]) + self.conv_weights(cin*exp,cout,1,use_bias[1])  
            elif len(use_bias)==3 : #expand ratio !=1 : there is  some expand layer (tot : 3 layers)
                return self.conv_weights(cin, cin*exp, 1, use_bias[0])+ self.dw_weights(cin*exp,k,mult,use_bias[1]) + self.conv_weights(cin*exp,cout,1,use_bias[2])



    def treat_NN(self, NN_frame):
        "Calculate the usefull features to keep as input parameters for the predictor "
        NN_frame['convtype']= NN_frame['convtype'].apply(lambda x : 'conv' if x=='conv_norelu' else x) 
        NN_frame['c_in'] = np.roll(NN_frame['c_out'], 1)
        NN_frame.loc[0,'c_in']=3 # RGB channels
        NN_frame['k2']=NN_frame['k'].apply(lambda x : x*x)
        NN_frame['hidden_dim']=NN_frame['exp']*NN_frame['c_in']
        NN_frame['h_in'] = np.nan
        NN_frame.loc[0,'h_in']=224 # FIXME Size imagenet - hardcoded
        for i in range(1, len(NN_frame)):
            if NN_frame.loc[i-1, 's']==1:
                NN_frame.loc[i, 'h_in']=NN_frame.loc[i-1, 'h_in']
            elif NN_frame.loc[i-1, 's']==2:
                NN_frame.loc[i, 'h_in']=NN_frame.loc[i-1, 'h_in']/2
            else : 
                raise NameError('Dont know the padding')
        NN_frame['h_out'] = np.roll(NN_frame['h_in'], -1)
        NN_frame.loc[NN_frame.last_valid_index(), 'h_out']=NN_frame['h_in'][NN_frame.last_valid_index()]/NN_frame.loc[NN_frame.last_valid_index(), 's']
        NN_frame['tensor_in']=NN_frame['c_in']*NN_frame['h_in']*NN_frame['h_in']
        NN_frame['tensor_out']=NN_frame['c_out']*NN_frame['h_out']*NN_frame['h_out']
        NN_frame['weights']=NN_frame.apply(lambda x : self.calculate_weights(x.convtype,x.c_in,x.c_out,x.k,x.exp,x.use_bias), axis=1)
        NN_frame['FLOPS']= NN_frame.apply(lambda x : self.calculate_flop(x.convtype, x.h_in, x.h_out, x.c_in, x.c_out, x.k, x.exp, x.tensor_in, x.skip), axis=1)
        NN_frame =NN_frame.loc[:, self.values_to_keep]
        #print(f"SUM WEIGHTS CALCUL : {sum( NN_frame['weights'])}")
        return  NN_frame


    def array_to_tensor(self, nn_array):
        "Convert a np array of tf.tensors and floats to a full tf.tensor"
        for i in range(nn_array.shape[0]):
            for j in range(nn_array.shape[1]):
                nn_array[i][j] = tf.convert_to_tensor((nn_array[i][j],), dtype=tf.float32)
                nn_array[i][j] = tf.reshape(nn_array[i][j], [1])
        blocks = []
        for i in range(nn_array.shape[0]):
          blocks.append(tf.stack(list(nn_array[i]), axis=0))
        nn_array = tf.stack(blocks)
        nn_array = tf.reshape(nn_array,[nn_array.shape[0],nn_array.shape[1]])
        return nn_array
    
    def skip_blocks(self, NN_tensor):
        "Keep the blocks which have not been skiped"
        return tf.boolean_mask(NN_tensor, self._blocks_to_keep)


    def add_zero_blocks(self, NN_tensor):
        "Complete the tensor with zeros from a (?,7) to a (max_blocks, 7) shape in order to have the same predictor inputs shape "
        nb_blocks = tf.shape(NN_tensor)[0]
        nb_features =  tf.shape(NN_tensor)[1]
        nb_zero_blocks = self.max_blocks-nb_blocks
        zero_blocks = tf.zeros((nb_zero_blocks,nb_features), dtype=tf.float32)
        tensor = tf.concat((NN_tensor, zero_blocks),axis=0)
        return tensor


    def normalize(self, NN_tensor):
        "Normalize the inputs with the training dataset standard deviations"
        normalized_columns =[]
        for jj in range(NN_tensor.shape[1]): 
            normalizing_std = tf.cast(1/self.std[jj], dtype=tf.float32)
            normalized_column = tf.math.scalar_mul(normalizing_std, NN_tensor[:,jj])
            normalized_columns.append(normalized_column)
        NN_normalized_tensor = tf.stack(normalized_columns, axis=1)
        return NN_normalized_tensor



    def NN_to_input(self):
        "Realize all the preprocessing steps"
        self.build_list()
        NN_frame = pd.DataFrame(self.list, columns = self.columns_names)
        NN_numpy_array = self.treat_NN(NN_frame).to_numpy()
       # NN_tensor = self.array_to_tensor(NN_numpy_array)
        #NN_tensor = self.skip_blocks(NN_tensor) 
        NN_tensor = self.add_zero_blocks(NN_numpy_array)
        NN_tensor_normalize = self.normalize(NN_tensor)
        return NN_tensor_normalize

