# Finding the Features from the Trees

In this activity, you'll use a Random Forests model to find the most important features for predicting arrhythmia in heartbeats.

## Instructions

* Use the starter file [RandomForest-Feature-Selection.ipynb](Unsolved/RandomForest-Feature-Selection.ipynb) and [arrhythmia.csv](Resources/arrhythmia.csv) dataset for this activity.

* Import the arrhythmia data, and then fit a Random Forests model to the scaled and split data.

* Import `SelectModel` to extract the best features from the Random Forests model.

* Fit a logistic regression to the original dataset, and then print its score.

* Fit a logistic regression to the selected dataset, and then print its score.

* Compare the scores of the two logistic regression models.

## Reference

Guvenir, H., Acar, Burak & Muderrisoglu, Haldun. (1997). Cardiac Arrhythmia Database. UCI Machine Learning Repository. ​​https://archive.ics.uci.edu/ml/datasets/arrhythmia

---

© 2022 edX Boot Camps LLC. Confidential and Proprietary. All Rights Reserved.
