import pandas as pd
import numpy as np

class ModelToList():
    def __init__(self, conv_stem, blocks, conv_head, fc):
        self._conv_stem = conv_stem
        self._blocks = blocks
        self._conv_head = conv_head
        self._fc = fc
        self.list  = []
        self.convtypes = [ "conv", "dw", "inv"]
        # !!!! FIXME
        self.std = [282669167.2068437,
                    474753.5435396601,
                    135528.50830774553,
                    137473.53306652652,
                    310.4712383514654,
                    12.199977367222479,
                    0.39403139516178176]
                        

    def _add_conv_stem(self):
        conv_steam_layer = [1, self._conv_stem.filters, self._conv_stem.strides[0],self._conv_stem.kernel_size[0],0,'conv']
        self.list.append(conv_steam_layer)
    
    # FIXME call_se excitation layer
    # FIXME skip ? if k = 0  
    def _add_blocks(self):
        for block in self._blocks:
            args=block._block_args
            trues = {False : 0, True : 1}
            block_layer = [args.expand_ratio, args.output_filters, args.strides[0], args.kernel_size, trues[args.id_skip],'inv']
            self.list.append(block_layer)
    
    def _add_conv_head(self):
        head = self._conv_head
        conv_head_layer = [1, head.filters, head.strides[0], head.kernel_size[0], 0, 'conv' ]
        self.list.append(conv_head_layer)

    def _add_fc(self):
        fc_layer = [1, self._fc, 1,1,0, 'conv']
        self.list.append(fc_layer)

    
    def _build(self):
        self._add_conv_stem()
        self._add_blocks()
        self._add_conv_head()
        self._add_fc()

            
class TreatNeuralNetwork():
    def __init__(self, conv_stem, blocks, conv_head, num_classes):
        self.Model_to_List =  ModelToList(conv_stem, blocks, conv_head, num_classes)
        self.columns_names = ['exp', 'c_out', 's', 'k', 'skip']
        self.conv_types = ['conv', 'dw', 'inv']
        self.values_to_keep = ['FLOPS', 'weights', 'tensor_in', 'tensor_out', 'hidden_dim', 'k2', 'skip']
        self.separate_types = False
        self.max_blocks=37  # FIXME : hardcoded
        self.std = [290269479.2486268, 502976.69976542174,134091.3240227188,135896.62164816633,313.5131629286966,12.21604009918021, 0.39038205807298065]


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
    def conv_weights(cin,cout,k):
        return (k*k*cin+1)*cout

    @staticmethod
    def dw_weights(cin,k,mult=1):
        return (k*k+1)*cin

    def calculate_weights(self, convtype,cin,cout,k,exp,mult=1):
        if convtype=='conv':
            return self.conv_weights(cin,cout,k)
        elif convtype=='dw':
            return self.dw_weights(cin,k)
        elif convtype=='inv':
            return self.conv_weights(cin, cin*exp, 1)+ self.dw_weights(cin*exp,k,mult) + self.conv_weights(cin*exp,cout,1)



    def treat_NN(self, NN_frame):
        NN_frame['convtype']= NN_frame['convtype'].apply(lambda x : 'conv' if x=='conv_norelu' else x)

        NN_frame['c_in'] = np.roll(NN_frame['c_out'], 1)
        NN_frame['c_in'][0]=3 # RGB channels

        NN_frame['k2']=NN_frame['k'].apply(lambda x : x*x)
        NN_frame['hidden_dim']=NN_frame['exp']*NN_frame['c_in']

        NN_frame['h_in'] = np.nan
        NN_frame['h_in'][0]=224 # Size imagenet - hardcoded
        for i in range(1, len(NN_frame)):
            if NN_frame.loc[i-1, 's']==1:
                NN_frame.loc[i, 'h_in']=NN_frame.loc[i-1, 'h_in']
            elif NN_frame.loc[i-1, 's']==2:
                NN_frame.loc[i, 'h_in']=NN_frame.loc[i-1, 'h_in']/2
            else : 
                raise NameError('Dont know the paddind')


        NN_frame['h_out'] = np.roll(NN_frame['h_in'], -1)
        NN_frame['h_out'][NN_frame.last_valid_index()]=NN_frame['h_in'][NN_frame.last_valid_index()]/NN_frame.loc[NN_frame.last_valid_index(), 's']

        NN_frame['tensor_in']=NN_frame['c_in']*NN_frame['h_in']*NN_frame['h_in']
        NN_frame['tensor_out']=NN_frame['c_out']*NN_frame['h_out']*NN_frame['h_out']

        NN_frame['weights']=NN_frame.apply(lambda x : self.calculate_weights(x.convtype,x.c_in,x.c_out,x.k,x.exp), axis=1)
        NN_frame['FLOPS']= NN_frame.apply(lambda x : self.calculate_flop(x.convtype,x.h_in,x.h_out,x.c_in,x.c_out,x.k,x.exp, x.tensor_in,x.skip), axis=1)
        NN_frame =NN_frame.loc[:, self.values_to_keep]
        return  NN_frame



    def add_zero_blocks(self,arr):
        zero_blocks = np.zeros((self.max_blocks-arr.shape[0],arr.shape[1]))
        return np.append(arr, zero_blocks, axis=0)

    def normalize(self, arr):
        for i in range(arr.shape[1]): #nb of parameters
            arr[:,i]/= self.std[i]
        return arr

    def NN_to_array(self):
        self.build_list()
        NN_frame = pd.DataFrame(self.list, columns = self.columns_names+ ['convtype',])
        numpy_array = self.treat_NN(NN_frame).to_numpy()
        numpy_array = self.add_zero_blocks(numpy_array)
        numpy_array = self.normalize(numpy_array)
        return numpy_array