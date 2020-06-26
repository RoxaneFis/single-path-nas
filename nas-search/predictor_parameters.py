import pandas as pd
import numpy as np
from numpy import loadtxt
import tensorflow as tf


# def convert_indicators(i5x5, i50c, i100c):
#     "Convert indicators to kernel, expand size"
#     import pdb
#     pdb.set_trace()
#     zero = tf.constant(0.0)
#     if  tf.math.equal(i5x5, zero):

#     inds_row = [i5x5, i50c, i100c]

#     if inds_row ==[0.0, 0.0, 0.0]:
#       k =None; exp = None; skip = True # skip 
#     elif inds_row == [0.0, 0.0, 1.0]:
#       k =None; exp = None; skip = True # skip
#     elif inds_row == [0.0, 1.0, 0.0]:
#       k =3; exp = 3; skip = False # 3x3-3
#     elif inds_row == [0.0, 1.0, 1.0]:
#       k =3; exp = 6; skip = False # 3x3-6
#     elif inds_row == [1.0, 0.0, 0.0]:
#       k =None; exp = None; skip = True # skip
#     elif inds_row == [1.0, 0.0, 1.0]:
#       k =None; exp = None; skip = True  # skip
#     elif inds_row == [1.0, 1.0, 0.0]:
#       k =5; exp = 3; skip = False # 5x5-3
#     elif inds_row == [1.0, 1.0, 1.0]:
#       k =5; exp = 6; skip = False # 5x5-6
#     else:
#       print("Wrong indicators values")
#       assert 0 == 1 # will crash
#     return k, exp, skip

def convert_indicators(i5x5, i50c, i100c):
    "Convert indicators to kernel, expand size"
    if type(i50c)==float:
        zero = tf.constant(0.0)
    else:
        zero = tf.fill(i50c.shape, 0.0)
    # one = tf.constant(1.0) 

    skip = tf.cond(tf.reduce_mean(i50c - zero) <= 0,lambda: tf.constant(True), lambda:tf.constant(False)) #shape ()
    k = 3 + 2* i5x5 #shape 1
    exp = 3 + 3* i100c
    # skip = False
    # k = 5
    # exp = 6

    return k, exp, skip



class ModelToList():
    def __init__(self, conv_stem, blocks, conv_head, fc):
        self._conv_stem = conv_stem
        self._blocks = blocks
        self._conv_head = conv_head
        self._fc = fc
        self.list  = []
        self.convtypes = [ "conv", "dw", "inv"]
        # !!!! FIXME
        self.std = loadtxt('single-path-nas/HAS/params/std.csv', delimiter=',')
    '''
    (t, c, s, k, skip, convtype)
    [ 
    [1, 16, 2, 3, 0, 'conv'], 
    [1, 16, 1, 3, 0, 'dw'], 
    [1, 8, 1, 1, 0, 'conv_norelu'], 
    [6, 8, 2, 3, 0, 'inv'],
    [6, 8, 1, 3, 1, 'inv'],   
    [6, 16, 2, 3, 0, 'inv'],   
    [6, 16, 1, 3, 1, 'inv'],   
    [6, 56, 1, 3, 1, 'inv'],
    [6, 56, 1, 3, 1, 'inv'], 
    [6, 112, 1, 3, 0, 'inv'], 
    [1, 1280, 1, 1, 0, 'conv'], 
]
    '''
    #Use_bias :                   

    def _add_conv_stem(self):
        conv_steam_layer = [1, self._conv_stem.filters, self._conv_stem.strides[0],self._conv_stem.kernel_size[0],0,'conv', [self._conv_stem.use_bias]]
        self.list.append(conv_steam_layer)
    
    # FIXME call_se excitation layer
    # FIXME skip ? if k = 0  
    def _add_blocks(self):
        for block in self._blocks:
            #args=block._block_args
            args=block
            trues = {False : 0, True : 1}
            if args.expand_ratio != 1 :
                #block_layer = [args.expand_ratio, args.output_filters, args.strides[0], args.kernel_size, trues[args.id_skip],'inv', [block._expand_conv.use_bias, block._depthwise_conv.use_bias , block._project_conv.use_bias]]
                block_layer = [args.expand_ratio, args.output_filters, args.strides[0], args.kernel_size, trues[args.id_skip],'inv', [False, False, False]]
            else :
                #no expand layer
                #block_layer = [args.expand_ratio, args.output_filters, args.strides[0], args.kernel_size, trues[args.id_skip],'inv', [block._depthwise_conv.use_bias , block._project_conv.use_bias]]
                block_layer = [args.expand_ratio, args.output_filters, args.strides[0], args.kernel_size, trues[args.id_skip],'inv', [False, False]]

            self.list.append(block_layer)
    
    def _add_conv_head(self):
        head = self._conv_head
        conv_head_layer = [1, head.filters, head.strides[0], head.kernel_size[0], 0, 'conv', [head.use_bias] ]
        self.list.append(conv_head_layer)

    def _add_fc(self):
        fc_layer = [1, self._fc, 1,1,0, 'conv', [True]]
        self.list.append(fc_layer)

    
    def _build(self):
        self._add_conv_stem()
        self._add_blocks()
        self._add_conv_head()
        self._add_fc()

            
class TreatNeuralNetwork():
    def __init__(self, conv_stem, blocks, conv_head, num_classes, blocks_to_delete, model_path):
        self.Model_to_List =  ModelToList(conv_stem, blocks, conv_head, num_classes)
        self._blocks_to_keep = blocks_to_delete
        self.columns_names = ['exp', 'c_out', 's', 'k', 'skip', 'convtype','use_bias']
        self.conv_types = ['conv', 'dw', 'inv']
        self.values_to_keep = ['FLOPS', 'weights', 'tensor_in', 'tensor_out', 'hidden_dim', 'k2', 'skip']
        self.separate_types = False
        self.max_blocks=37  # FIXME : hardcoded
        try :
            self.std = np.loadtxt(f'{model_path}/std.csv', delimiter=',')
        except Exception as err :
            print("The standard deviation values used for normalising the predictor inputs are not found")


    def build_list(self):
        self.Model_to_List._build()
        self.list = self.Model_to_List.list


    @staticmethod
    def conv_flops(hout,cin,cout,k,tensor_in,skip=0):
        flops = 2*k*k*cin*hout*hout*cout
        if skip ==1:
            flops += tensor_in
        return flops

    @staticmethod
    def dw_flops(hout,c,k,tensor_in,skip=0, mult=1):
        flops = 2*k*k*c*hout*hout
        if skip == 1:
            flops +=tensor_in
        return flops

    def calculate_flop(self, convtype,hin,hout,cin,cout,k,exp,tensor_in, skip=0, mult=1,skip_op=False):
        if skip_op==True:
            return 0
        else:
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
        if not use_bias :
            return k*k*cin*cout
        return (k*k*cin+1)*cout

    @staticmethod
    def dw_weights(cin,k,mult=1, use_bias=True):
        if not use_bias :
            return k*k*cin
        return (k*k+1)*cin

    def calculate_weights(self, convtype,cin,cout,k,exp,use_bias, mult=1):
        if convtype=='conv':
            return self.conv_weights(cin,cout,k, use_bias[0])
        elif convtype=='dw':
            return self.dw_weights(cin,k, use_bias[0])
        elif convtype=='inv':
            if len(use_bias) == 2 : #expand ratio ==1 : no expand 
                return self.dw_weights(cin*exp,k,mult,use_bias[0]) + self.conv_weights(cin*exp,cout,1,use_bias[1])
                
            elif len(use_bias)==3 :
                return self.conv_weights(cin, cin*exp, 1, use_bias[0])+ self.dw_weights(cin*exp,k,mult,use_bias[1]) + self.conv_weights(cin*exp,cout,1,use_bias[2])



    def treat_NN(self, NN_frame):
        NN_frame['convtype']= NN_frame['convtype'].apply(lambda x : 'conv' if x=='conv_norelu' else x)

        NN_frame['c_in'] = np.roll(NN_frame['c_out'], 1)
        NN_frame.loc[0,'c_in']=3 # RGB channels

        NN_frame['k2']=NN_frame['k'].apply(lambda x : x*x)
        NN_frame['hidden_dim']=NN_frame['exp']*NN_frame['c_in']

        NN_frame['h_in'] = np.nan
        #NN_frame['h_in'][0]=224 # Size imagenet - hardcoded
        NN_frame.loc[0,'h_in']=224
        for i in range(1, len(NN_frame)):
            if NN_frame.loc[i-1, 's']==1:
                NN_frame.loc[i, 'h_in']=NN_frame.loc[i-1, 'h_in']
            elif NN_frame.loc[i-1, 's']==2:
                NN_frame.loc[i, 'h_in']=NN_frame.loc[i-1, 'h_in']/2
            else : 
                raise NameError('Dont know the paddind')

        NN_frame['h_out'] = np.roll(NN_frame['h_in'], -1)
        NN_frame.loc[NN_frame.last_valid_index(), 'h_out']=NN_frame['h_in'][NN_frame.last_valid_index()]/NN_frame.loc[NN_frame.last_valid_index(), 's']
       # NN_frame['h_out'][NN_frame.last_valid_index()]=NN_frame['h_in'][NN_frame.last_valid_index()]/NN_frame.loc[NN_frame.last_valid_index(), 's']

        NN_frame['tensor_in']=NN_frame['c_in']*NN_frame['h_in']*NN_frame['h_in']
        NN_frame['tensor_out']=NN_frame['c_out']*NN_frame['h_out']*NN_frame['h_out']

        NN_frame['weights']=NN_frame.apply(lambda x : self.calculate_weights(x.convtype,x.c_in,x.c_out,x.k,x.exp,x.use_bias), axis=1)
        NN_frame['FLOPS']= NN_frame.apply(lambda x : self.calculate_flop(x.convtype,x.h_in,x.h_out,x.c_in,x.c_out,x.k,x.exp, x.tensor_in,x.skip), axis=1)
        NN_frame =NN_frame.loc[:, self.values_to_keep]
        print(f"SUM WEIGHTS CALCUL : {sum( NN_frame['weights'])}")
        import pdb
        #pdb.set_trace()
        return  NN_frame



   
   
    # def complete_boolean_mask(self):
    #     nb_blocks = self.max_blocks - len(self._blocks_to_keep) # 22
    #     for i in 

    
    def array_to_tensor(self, nn_array):
       # nn_array = self.NN_to_array()
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
        #import pdb
       # pdb.set_trace()
        return tf.boolean_mask(NN_tensor, self._blocks_to_keep)


    def add_zero_blocks(self, NN_tensor):
       # NN_tensor = self.array_to_tensor()
       # nb_zero_blocks = self.max_blocks-NN_tensor.shape[0]
        nb_blocks = tf.shape(NN_tensor)[0]
        nb_features =  tf.shape(NN_tensor)[1]
        nb_zero_blocks = self.max_blocks-nb_blocks

        zero_blocks = tf.zeros((nb_zero_blocks,nb_features), dtype=tf.float32)
        tensor = tf.concat((NN_tensor, zero_blocks),axis=0)
        return tensor


    def normalize(self, NN_tensor):
        normalized_columns =[]
        for jj in range(NN_tensor.shape[1]): #nb of parameters
            normalizing_std = tf.cast(1/self.std[jj], dtype=tf.float32)
            normalized_column = tf.math.scalar_mul(normalizing_std, NN_tensor[:,jj])
            normalized_columns.append(normalized_column)
        NN_normalized_tensor = tf.stack(normalized_columns, axis=1)
        return NN_normalized_tensor



    def NN_to_input(self):
        self.build_list()
        NN_frame = pd.DataFrame(self.list, columns = self.columns_names)
        NN_numpy_array = self.treat_NN(NN_frame).to_numpy()
        NN_tensor = self.array_to_tensor(NN_numpy_array)
        NN_tensor = self.skip_blocks(NN_tensor) #taille plus petite
        
        NN_tensor = self.add_zero_blocks(NN_tensor)
        NN_tensor_normalize = self.normalize(NN_tensor)
        return NN_tensor_normalize
        #return NN_tensor
        
       # numpy_array = self.add_zero_blocks(numpy_array)
       # numpy_array = self.normalize(numpy_array)
       # return numpy_array