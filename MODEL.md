# Our Model

A more detailed look into our model architecture and how the model is trained, evaluated, and prepared for deployment

## An important note

Any given model is only as valuable as the dataset on which it is trained. Consult the README and data pre-processing documentation for more insight
regarding our data.

## Data Split

Training, validation, and testing all perform a valuable role in making sure the model is actually learning appropriately.
To make sure each of the three sets does not have any leakage (individual cases having data in overlapping sets, such as some data in training and some in validation),
the `GroupShuffleSplit` method is used. Each of the sets can be described as follows:
- Training: a majority of the dataset, used to adjust model parameters and help the model learn
- Validation: used alongside the training set to evaluate model performance on unseen data during training, tune hyperparameters, and prevent 
    overfitting without changing model parameters
- Testing: used to provide unbiased estimates of the fully-tuned model's performance on unseen data and helps evaluate how well the model generalizes
    to real-world applications

Our training/validation/testing split is 64/16/20. This is not the only appropriate split but works for our use case, showing the model plenty of 
data to learn patterns without overfitting.

## Scaling

Scaling transforms numerical features to have a similar range or distribution. This keeps features with larger magnitudes from dominating the model. During preprocessing, we use Standard scaling (using sklearn's `StandardScaler`), which scales the data to have a mean of 0 and standard deviation of 1.
Normalization (min-max scaling) is another acceptable technique but is not used here due to its sensitivity to outliers.

## Model Architecture

Our model follows a multi-layer perceptron architecture, a type of feed-forward model in which the outputs of one layer become the inputs to the next. 
Specifically, our model consists of an input layer with the dimensions of our input features, four hidden layers (128-64-32-16), and an output layer.
Most of the hidden layers undergo batch normalization and dropout. Batch normalization keeps the values at the layer centered around zero for faster, reliable training. Dropout turns off a small percentage of neurons in a layer to prevent heavy reliance on a single neuron and reduce overfitting.
The hidden layers have ReLU activation, which outputs the input value if it is positive and zero otherwise, introducing non-linearity to help the model learn complex patterns.
The output layer has linear activation, which means the final value is passed without any transformations to output a continuous value.
Our model uses the Adam optimizer with a small learning rate for stable training and a loss function based on the MAE. It also tracks both training and validation MAE and MAPE during training.

## Training

The model is currently trained over 60 epochs with a batch size of 32. Callback functions `EarlyStopping` and `ReduceLROnPlateau` adjust training when
performance stagnates. 

## Validation

The validation set is passed as a parameter to the `fit()` function, adjusting model hyperparameters during training.

## Testing and Evaluation

Once training is complete, the model can be tested. This is done by feeding the model sample inputs it has not yet seen (from the testing set)
and having it generate predictions. These predictions can then be evaluated using various statistical metrics, including:
- Coefficient of determination (`R^2`): represents the proportion of variance in the dependent variable explained by the independent variable and
    how well the model fits the data
- Mean absolute error (MAE): measures the average absolute magnitude of errors in original units 
- Mean absolute percentage error (MAPE): calculates error relative to actual values as a percentage

Another important evaluation metric is the Clarke Error Grid Analysis (CEGA), which quantifies the accuracy of blood glucose estimations against
known values. Each estimation is put into one of five zones, based on the following criteria:
- Zone A: clinically accurate, the estimate falls within 20% of the actual value
- Zone B: clinically acceptable, the estimate is beyond 20% of the actual value, but treatment to regulate BG based on the estimate would not be inappropriate
- Zone C: the estimate suggests treatment, while the actual value suggests treatment is unnecessary
- Zone D: the estimate fails to properly detect hypoglycemia or hyperglycemia, which can be dangerous
- Zone E: the estimate fails to properly detect hypoglycemia or hyperglycemia and even mistakes one condition for the other

## Quantization and Deployment

To fit within the hardware constraints of the ESP-32 microcontroller, the model must be quantized. This is done through:
- Creating a converter using the `tf.lite.TFLiteConverter.from_keras_model(model)` method
- Performing optimizations
- Designating input and output inference types as int8
- Providing a representative dataset yielding validation samples for calibration
- Returning the quantized tflite model using the `converter.convert()` method

Once the model has been quantized and saved, it must be converted to C for integration into the embedded code for on-device deployment.
This can be done using the command:
```bash
xxd -i {TFLITE_MODEL} > mlp.h
```
TFLITE_MODEL is the name of the saved .tflite file.
The returned .h file can then be called in the main ESP-32 embedded code and used to define input and output tensors.