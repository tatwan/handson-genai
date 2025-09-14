# Predicting Employees Attrition

In this activity, you will apply your knowledge of deep learning models to predict whether an employee is likely to depart from a company.

## Instructions

The steps for this activity are divided into the following sections:

* Preprocess the data.

* Create a neural network model to predict employee attrition.

* Train and evaluate the neural network model.

#### Preprocess the Data 

1. Read the `HR-Employee-Attrition.csv` file from the Resources folder and create a DataFrame.

2. Review the resulting DataFrame. Check the data type associated with each column to identify categorical and non-categorical variables.

  **Hint:** Remember that categorical variables have an `object` data type.

3. Create a list of categorical variables.

  **Hint:** The Pandas DataFrame `dtypes` property returns a Pandas Series that displays the data type of each column. The column name is the index. You can create the list of categorical variables manually. Or, you can use Pandas filtering. For more detail about the filtering process, review the [`dtypes` documentation at Pandas API reference](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dtypes.html).

4. Encode the dataset's categorical variables using the OneHotEncoder module from scikit-learn.

5. Create a new DataFrame named `attrition_df` that contains the encoded categorical variables and the numerical variables from the original dataset.

 **Note:** To complete this step, you will employ the Pandas `concat()` function, which was introduced earlier in the course. Recall that the `concat` function combines data from different DataFrames into a new DataFrame. Here, you’ll use `concat` to create a new DataFrame containing the preprocessed data. You can learn more about the `concat` function in [this section of the Pandas documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html).

6. Use `attrition_df` to create the features (`X`) and target (`y`) sets.

    * Your assignment is to build a model that predicts whether a person is at risk for attrition. So, you'll need to separate the attrition columns from the rest of the input data. Because the attrition data is binary (two possible values), you only need to keep the “Attrition_Yes” column and can ignore the “Attrition_No” column because it is redundant. So, the “Attrition_Yes” column will become the target set `y`.

    * The rest of the DataFrame columns will become the features set `X`.
    
7. Create the training and testing sets using the `train_test_split` function from scikit-learn.

8. Use StandardScaler to standardize the numerical features.

#### Create a Neural Network Model to Predict Employee Attrition

1. Use the optimization techniques presented in this lesson to create a deep neural network model with two hidden layers.

    * Set a variable `number_input_features` equal to the number of input features.

    * Set a variable `hidden_nodes_layer1` equal to an integer number close to the mean of the number of input features and the number of neurons in the output layer:

* Number of hidden nodes = (# of input features + output layer size) / 2
* Hint: Use “//” instead of “/” to divide and round down to the nearest integer.

    * Set a variable `hidden_nodes_layer2` equal to an integer number close to the mean of the number of hidden nodes defined for the first layer and the number of neurons in the output layer.

	* Note: in the second hidden layer, we use the size of the first hidden layer, rather than the size of the input layer (i.e., the original data coming in).

    * Add the model’s layers, and then use the `relu` activation function for each hidden layer.

    * This model will predict a binary output. Add an output layer with one neuron and use the `sigmoid` activation function.

  **Hint:** To set the number of input features, you can fetch the length of the first element, like this: `number_input_features = len(X_train.iloc[0])`.

2. Display the structure of your model using the `summary` function.

3. Compile the model. Use the `binary_crossentropy` loss function, the `adam` optimizer, and the `accuracy` metric.

#### Train and Evaluate the Neural Network Model

1. Train (fit) the neural network model with the training data for 100 epochs.

2. Evaluate the model using the test data to determine its loss and accuracy.

## References

[Keras Sequential model](https://keras.io/api/models/sequential/)

[Keras Dense module](https://keras.io/api/layers/core_layers/dense/)

[Keras evaluate](https://keras.io/api/models/model_training_apis/)

[SKLearn OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)

