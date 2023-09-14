import pandas as pd 
from sklearn.model_selection import train_test_split
from prepro import preprocessing_data
from train_data import read_data 
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, confusion_matrix
from shapely import shap_plots

encoder = LabelEncoder()

df = read_data()

df_new  = preprocessing_data(df)


def model():

    random_forest = RandomForestClassifier(
        n_estimators=400,  # Number of decision trees in the forest
        criterion='gini',  # Splitting criterion ('gini' or 'entropy')
        max_depth=None,  # Maximum depth of the trees (None means nodes expand until they contain < min_samples_split samples)
        min_samples_split=2,  # Minimum samples required to split an internal node
        min_samples_leaf=1,  # Minimum samples required to be at a leaf node
        max_features='auto',  # Number of features to consider when looking for the best split
        bootstrap=True,  # Whether to bootstrap samples when building trees
        random_state=42,  # Random seed for reproducibility
        n_jobs=None,  # Number of CPU cores to use for parallelism (None uses all available cores)
        class_weight=None,  # Weights associated with classes (None means all classes have equal weight)
        oob_score=False,  # Whether to use out-of-bag samples to estimate the generalization accuracy
        verbose=0  # Controls the verbosity of the tree-building process (0: silent, 1: warning, 2: info)
        )

    xgboost = XGBClassifier(
        n_estimators=400,  # Number of boosting rounds (trees)
        learning_rate=0.1,  # Step size shrinkage used in update to prevent overfitting
        max_depth=5,  # Maximum depth of a tree
        min_child_weight=1,  # Minimum sum of instance weight (hessian) needed in a child
        gamma=0.1,  # Minimum loss reduction required to make a further partition on a leaf node
        subsample=1.0,  # Fraction of samples used for fitting the trees
        colsample_bytree=1.0,  # Fraction of features used for fitting the trees
        objective='binary:logistic',  # Objective function for binary classification
        random_state=43,  # Random seed for reproducibility
        verbosity=1  # Controls the level of verbosity (0: silent, 1: warning, 2: info)
    )

    return random_forest,xgboost

random_forest,xgboost = model()

def training(df,random_forest,xgboost):
    x = df.drop(columns=['y'])
    y = df['y']

    y = encoder.fit_transform(y)
    
    X_train ,X_test , y_train ,y_test = train_test_split(x , y , test_size=0.2 ,random_state=42)
    
    random_forest.fit(X_train,y_train)
    xgboost.fit(X_train,y_train)

    y_pred_xgb = xgboost.predict_proba(X_test)[:, 1]
    new_threshold = 0.4 
    y_pred_adjusted = (y_pred_xgb >= new_threshold).astype(int)

    y_prob_random = random_forest.predict_proba(X_test)[:, 1]
    y_pred_adjusted_1 = (y_prob_random >= new_threshold).astype(int)


    return y_pred_adjusted ,y_pred_adjusted_1 , y_test , X_test


y_pred_adjusted ,y_pred_adjusted_1 ,y_test , X_test = training(df_new,random_forest,xgboost)


def evaluation(y1,y2):
        precision = precision_score(y1, y2)


        # Calculate recall
        recall = recall_score(y1, y2)

        accuracy = accuracy_score(y1,y2)
        # Calculate F1-score
        f1 = f1_score(y1, y2)


        # Calculate confusion matrix
        conf_matrix = confusion_matrix(y1, y2)

        return precision,recall,f1,conf_matrix,accuracy


precision_1,recall_1 ,f1_1 ,conf_matrix_1 ,accuracy_1 = evaluation(y_test,y_pred_adjusted)

precision_2 ,recall_2 ,f1_2 , conf_matrix_2 , accuracy_2 = evaluation(y_test,y_pred_adjusted_1)

print("Precision:", precision_1)
print("Recall (Sensitivity):", recall_1)
print("F1-Score:", f1_1)
print("Confusion Matrix:\n", conf_matrix_1)
print("accuracy:\n", accuracy_1)

print(120 * "**")

print("Precision:", precision_2)
print("Recall (Sensitivity):", recall_2)
print("F1-Score:", f1_2)
print("Confusion Matrix:\n", conf_matrix_2)
print("accuracy:\n", accuracy_2)


shap_plots(random_forest,X_test)
