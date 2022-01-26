# monkey-identity-expression

This repository includes the codes, documents and network weights of training and validating the best-performing multi-task deep neural network. It contains two packages, one is named "maincode" and another is "weight".

In "maincode", multitasktrain.py define the architecture of the multi-task deep neural network, and train the network weights. 
idonehot.npy and exponehot.npy are one-hot encoding of face images' expression and identity categories, respectively; 
train_id_exp.xls records the labels of each image in the large macaque monkey face dataset for training; 
expressionvalidate.py was to calculate the monkey facial expression classifiation accuracy;
identityvalidate.py was to calculate the monkey facial identity classification accuracy, which need to get information of identity within-category pairs from identityvalidate.txt. branchc51.json records the best-performing deep neural network architecture, sharing the layers from conv1_1 to pool4, and separating the layers starting from conv5_1 to fc8.

In "weight",  result.hdf5 saves the weights of the best-performing multi-task deep neural network, which correspond to branchc51.json.
