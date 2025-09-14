# Detecting Myopia

In this activity, you'll use a deep learning model to predict if a person has myopia.

**Instructions:**

* Upload the [starter notebook](Unsolved/DetectingMyopia.ipynb) to Google Colab and run the code to import the dependencies and load the myopia dataset.

* Separate the myopia **MYOPIC** target from the other features in the dataset.

* Split the features and target into training and test datasets.

* Preprocess the input data accordingly:

  * If preprocessing categorical data, use Scikit-learn's **OneHotEncoder** module.

  * If preprocessing numerical data, use Scikit-learn's **StandardScaler** module.

* Define a deep learning model with the following features:

  * A first Dense layer with eight inputs and the "ReLU" activation function

  * A second Dense layer with at least eight neurons and the "ReLU" activation function

  * An output layer with one neuron and the "sigmoid" activation function

* Compile and train the model across no more than 100 epochs.

* Evaluate the performance of the deep learning model by calculating the test loss and predictive accuracy.

## Reference

Reduced dataset from [Orinda Longitudinal Study of Myopia conducted by the US National Eye Institute.](https://clinicaltrials.gov/ct2/show/NCT00000169)

