# monkey-identity-expression

There are two branches in this project, one named "maincode" and another named "weight".
In the "maincode" branch, multitasktrain.py was used to define and train the multi-task network. idonehot.npy and exponehot.npy were one-hot encoding for identity and expression, respectively. train_id_exp.xls records the label corresponding to each picture used for training. expressionvalidate.py was used to get expression recognition accuracy, and identityvalidate.py was used to get identity recognition accuracy, which needed to get ideneity pairs from identityvalidate.txt. branchc51.json records the best-performance multitask network structure, which shared layers from conv1_1 to pool4, and separated from conv5_1.
In the "weight" branch, result.hdf5 was the best-performance weight corresponding to branchc51.json.
