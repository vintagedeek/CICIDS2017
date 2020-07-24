# CICIDS2017

### cicids2017_2020_07_23_97_val_acc.ipynb (97.1394% val accuracy)
- Val accuracy 97.1394% at epoch 19
- Standardized features, 1 hidden layer (26 neurons)
- Linear, Sigmoid, BatchNorm1d, Linear, Sigmoid, 
- Cross Entropy Cost function

#### cicids2017_2020_07_23.ipynb
- Best run: validation accuracy 80%
- Fully connected neural network with one hidden layer (26 hidden neurons)
- Both activation functions are sigmoid
- Cross entropy cost function
- Best run: learning_rate=0.0001, batch_size=500, weight_decay=0, epochs: ~250


#### cicids2017_2020_07_23_96.ipynb
- Best run: validation accuracy 96.8%
- Used MinMaxScaler(), but incorrectly because validation data was included in the scaling. Need to rescale using only training data then transform val data based on training data.
