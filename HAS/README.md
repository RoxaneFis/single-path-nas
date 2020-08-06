
This markdown explains the different steps and expirements to train a power predictor and find the best hardware parameters.


# process_data.ipynb

This notebook pre-processes two kinds of data :
+ **"nets_*.json"** : files which describe the neural networks (identified by a number) as a sequence of blocks.
+ **"train_*.txt" or "val_*.txt"** : training or validation files. Contains some hardware parameters, the name of the used networks, and the predicted power.

The pre-processing consits of :

1) Encompass all training data (resp. val) in one csv file, encompass all networks description in a dictionnary.
2) Normalize some columns of the csv files in order to ease researchs between the networks description and the training (resp val) files. Also retrieves some usefull information.
3) Calculate some usefull information about the neural networks (number of FLOPS, weights...)
4) Apply all the preprocessing steps in order to save a training (resp val) csv file which contains the treated neural networks. 

!!! In the already treated data : The validation file contains all the data (ie the hardware parameters, the processed neural network arrays and the predicted power) whereas the training file only contains the processed neural networks arrays (because of memory issues). In practise, we only need to have the preprocessed neural network arrays and will concatanete the rest of the information in another notebook. Better, we could have only saved a dictionnary with the processed networks instead of having saved all of them to their corresponding lines in the training (resp val) files (the saved *.csv files are highly redundant as the networks are used several times). !!!

# treatment_multiple_inputs.ipynb

This notebook partially continues the processing of the data and defines the dataset. It then creates the predictor's model, trains it (or load the already trained weights). Several tests are then conducted, notably on the prediction of the hardware parameters (the trained predictor is freezed, and only the hardware parameters are updtated in a regression). Some cells also allow to make some tests on the layer sizes/depths, and test if more data is needed.

1) Load the processed networks (obtained from process_data.ipynb) and complete the arrays with zero blocks.
2) Load the hardware (HW) parameters and normalize them (save the Scaler)
3) Normalize the Neural Network parameters (save the standard deviations used for normalization).
4) Define two datasets and two losses: a single and a multiple outputs one (the predictions correspond to power P or to (P_core, P_bw)).
5) Define the model "recover" layer in order to regress the hardware parameters after the training of the predictor.
6) Defines and trains the model (uses "plot_figures.py" in order to save the checkpoints and other information)
7) Expirement on the HW parameters : 
   + Retrieves some neural networks from the dataset and treats them in order to use them as inputs to the predictor (uses "get_network_search_phase.py" to do so)
   + Predict the HW parameters (regression)
8) More tests
   + Evaluates a multiple outputs model
   + Tests some models randomly (different sizes of layers)
   + Tests if we need more data for the model


# stats_data.ipynb

This notebook makes some several test on the repartition of the data. Notably :
1) The power repartion (historigrams)
2) Some PCA of the HW params (not used)
3) k-means on the HW params (used in treatment_multiple_inputs.ipynb as initial HW parameters).

# stats_by_network.ipynb

This notebook makes some statistics over the different networks (we want to study the influence of the size, number of FLOPS... with the performances of the predictor). Notably it:
1) Describes the repartition of Pbw and Pcore powers depending of the different networks in the train and val datasets.
2) Evaluates the model performances by networks of the val dataset
3) Make some performance tests depending of different parameters (#FLOPS, #weights...)