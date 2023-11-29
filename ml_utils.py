import pandas as pd
import chardet

with open("./Datasets/merged_cleaned.csv", "rb") as f:
    result = chardet.detect(f.read())
    
matches = pd.read_csv("./Datasets/merged_cleaned.csv", index_col=0, encoding=result["encoding"])

matches["Date"] = pd.to_datetime(matches["Date"])

matches["htr_code"] = matches["HTR"].astype("category").cat.codes

matches["ftr_code"] = matches["FTR"].astype("category").cat.codes

matches["opp_code"] = matches["AwayTeam"].astype("category").cat.codes

matches["Time"] = matches["Time"].astype(str).str.replace(":.+", "", regex=True).astype(int)

matches["day_code"] = matches["Date"].dt.day_of_week

matches["target"] = (matches["ftr_code"] == 2)

predictors = ["FTHG", "FTAG", "HTHG","HTAG","HS","AS","HST","AST","HF","AF","HC","AC","HY","AY","HR","AR","B365H","B365D","B365A","BWH","BWD","BWA","IWH","IWD","IWA","PSH","PSD","PSA","WHH","WHD","WHA","VCH","VCD","VCA","MaxH","MaxD","MaxA","AvgH","AvgD","AvgA","B365>2.5","B365<2.5","P>2.5","P<2.5","Max>2.5","Max<2.5","Avg>2.5","Avg<2.5","AHh","B365AHH","B365AHA","PAHH","PAHA","MaxAHH","MaxAHA","AvgAHH","AvgAHA","B365CH","B365CD","B365CA","BWCH","BWCD","BWCA","IWCH","IWCD","IWCA","PSCH","PSCD","PSCA","WHCH","WHCD","WHCA","VCCH","VCCD","VCCA","MaxCH","MaxCD","MaxCA","AvgCH","AvgCD","AvgCA","B365C>2.5","B365C<2.5","PC>2.5","PC<2.5","MaxC>2.5","MaxC<2.5","AvgC>2.5","AvgC<2.5","AHCh","B365CAHH","B365CAHA","PCAHH","PCAHA","MaxCAHH","MaxCAHA","AvgCAHH","AvgCAHA", "opp_code", "day_code"]

matches.head()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

if all(col in matches.columns for col in predictors):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(matches[predictors], matches["target"], test_size=0.2, random_state=42)

    # Create a Gradient Boosting Classifier and train it on the data
    gbm = GradientBoostingClassifier(random_state=42)
    gbm.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = gbm.predict(X_test)

    # Evaluate the model
    acc = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    acc, class_report
else:
    "Some predictor columns are missing in the dataset."

X_train, X_test, y_train, y_test = train_test_split(matches[predictors], matches["target"], test_size=0.2, random_state=42)

# Create a Gradient Boosting Classifier and train it on the cleaned data
gbm = GradientBoostingClassifier(random_state=42)

gbm.fit(X_train, y_train)

y_pred = gbm.predict(X_test)

# Evaluate the model
acc = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

acc, class_report

from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np

# 1. Feature Importance
feature_importance = gbm.feature_importances_
sorted_idx = np.argsort(feature_importance)[::-1]

# 2. Cross-Validation
cross_val_scores = cross_val_score(gbm, X_train, y_train, cv=5)

# 3. Data Inspection: Already done during the data cleaning and transformation steps.

feature_importance[:10], cross_val_scores, np.mean(cross_val_scores)

# 1. Correlation Analysis
# Calculate the correlation between the target variable and the highly important features
highly_important_features = np.array(predictors)[sorted_idx[:2]]
correlation_matrix = matches[["target"] + list(highly_important_features)].corr()

# 2. Model Exclusion of Suspect Features
# Remove the highly important features and retrain the model
reduced_predictors = [p for p in predictors if p not in highly_important_features]
X_train_reduced, X_test_reduced = X_train[reduced_predictors], X_test[reduced_predictors]

# Create a new Gradient Boosting Classifier and train it on the reduced data
gbm_reduced = GradientBoostingClassifier(random_state=42)
gbm_reduced.fit(X_train_reduced, y_train)

# Make predictions on the reduced test set
y_pred_reduced = gbm_reduced.predict(X_test_reduced)

# Evaluate the new model
acc_reduced = accuracy_score(y_test, y_pred_reduced)
class_report_reduced = classification_report(y_test, y_pred_reduced)

correlation_matrix, acc_reduced, class_report_reduced

from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold
import numpy as np

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize lists to store metrics
auc_scores = []
f1_scores = []

# Initialize predictors and target
X = matches[reduced_predictors]
y = matches['target']

# Perform Stratified K-Fold cross-validation
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Train the model
    gbm_reduced.fit(X_train, y_train)
    
    # Make predictions
    y_pred_proba = gbm_reduced.predict_proba(X_test)[:, 1]
    y_pred = gbm_reduced.predict(X_test)
    
    # Calculate metrics and append to lists
    auc_scores.append(roc_auc_score(y_test, y_pred_proba))
    f1_scores.append(f1_score(y_test, y_pred))

# Calculate average metrics
avg_auc = np.mean(auc_scores)
avg_f1 = np.mean(f1_scores)

avg_auc, avg_f1

# Filter the data to only include matches where either Bayern Munich or Leipzig were playing
bayern_leipzig_matches = matches[(matches['HomeTeam'].str.contains('Bayern Munich|Leipzig', case=False)) |
                                 (matches['AwayTeam'].str.contains('Bayern Munich|Leipzig', case=False))]

# Display the first few rows of the filtered data
bayern_leipzig_matches.head()

# Check if all predictor columns are present in the dataset
if all(col in bayern_leipzig_matches.columns for col in predictors):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        bayern_leipzig_matches[predictors], 
        bayern_leipzig_matches["target"], 
        test_size=0.2, 
        random_state=42
    )

    # Create a Gradient Boosting Classifier and train it on the filtered data
    gbm_filtered = GradientBoostingClassifier(random_state=42)
    gbm_filtered.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred_filtered = gbm_filtered.predict(X_test)

    # Evaluate the model
    acc_filtered = accuracy_score(y_test, y_pred_filtered)
    class_report_filtered = classification_report(y_test, y_pred_filtered)
else:
    acc_filtered = "Some predictor columns are missing in the dataset."
    class_report_filtered = "Some predictor columns are missing in the dataset."

acc_filtered, class_report_filtered

# Create a function to prepare the input features for prediction
def prepare_features(home_team, away_team, model_features, data):
    # Filter the most recent match between the two teams
    last_match = data[((data['HomeTeam'] == home_team) & (data['AwayTeam'] == away_team)) |
                      ((data['HomeTeam'] == away_team) & (data['AwayTeam'] == home_team))].iloc[-1]
    
    # Prepare the feature vector based on the model's features
    feature_vector = last_match[model_features].values.reshape(1, -1)
    
    return feature_vector


