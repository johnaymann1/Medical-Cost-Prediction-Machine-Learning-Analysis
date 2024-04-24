import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt



######################################################################################################
######################################################################################################
#Phase 1: Load the dataset and Prepare train-validation-test split
######################################################################################################
######################################################################################################

# Load the dataset
insurance_filepath = 'insurance.csv'
insurance_data = pd.read_csv(insurance_filepath)

features = insurance_data.drop("charges", axis=1)
target = insurance_data["charges"]

# Train-validation-test split
X_train_val, X_test, y_train_val, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)


######################################################################################################
######################################################################################################
#Phase 2: Apply any preprocessing for features
######################################################################################################
######################################################################################################
# Identify numerical and categorical features
numerical_features = X_train.select_dtypes(include=['float64', 'int64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

# Create transformers
numerical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')),('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Create preprocessor
preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, numerical_features),('cat', categorical_transformer, categorical_features)])

# Apply preprocessing to training, validation, and test sets
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_val_preprocessed = preprocessor.transform(X_val)
X_test_preprocessed = preprocessor.transform(X_test)



######################################################################################################
######################################################################################################
#Phase 3: Apply 3 different models and compare between them
######################################################################################################
######################################################################################################
# Model building and evaluation
models = [
    ("Linear Regression", LinearRegression()),
    ("Random Forest", RandomForestRegressor()),
    ("Support Vector Regression", SVR())
]

# Hyperparameter tuning and model evaluation
for model_name, model in models:
    if model_name == "Support Vector Regression":
        # Define SVR model with hyperparameter grid for tuning
        svr = SVR()
        param_grid = {'C': [0.1, 1, 10, 100],
                      'kernel': ['linear','poly'],
                      'degree': [2, 3, 4],
                      'gamma': ['scale', 'auto']}
        
        # Create a GridSearchCV object
        grid_search = GridSearchCV(svr, param_grid, scoring='neg_mean_squared_error', cv=5)
        
        # Fit the model with the preprocessed training data
        grid_search.fit(X_train_preprocessed, y_train)
        
        # Get the best parameters and best model
        best_params = grid_search.best_params_
        best_svr_model = grid_search.best_estimator_
        
        # Predictions on validation set
        y_val_pred = best_svr_model.predict(X_val_preprocessed)
    else:
        # For other models, use the original pipeline
        model_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        model_pipeline.fit(X_train, y_train)
        y_val_pred = model_pipeline.predict(X_val)

    # Evaluate the model
    mae = mean_absolute_error(y_val, y_val_pred)
    mse = mean_squared_error(y_val, y_val_pred)
    r2 = r2_score(y_val, y_val_pred)
    
    print(f"Model: {model_name}")
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")
    print("=" * 30)
    
    
    # Reporting and insights
    # Scatter plot of predicted vs actual values
    plt.scatter(y_val, y_val_pred)
    plt.xlabel("Actual Charges")
    plt.ylabel("Predicted Charges")
    plt.title(f"Actual vs Predicted Charges on {model_name}")
    plt.show()

######################################################################################################
######################################################################################################
# Phase 4: Final Model Comparison and Error Analysis
#######################################################################################################
#######################################################################################################
#Choose the best model and evaluate on the test set
final_model_name = "Random Forest"
final_model = RandomForestRegressor()

# Train the final model on the entire training set
final_model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),('model', final_model)])
final_model_pipeline.fit(X_train, y_train)

# Predictions on the test set
y_test_pred = final_model_pipeline.predict(X_test)

# Evaluate the final model on the test set
final_mae = mean_absolute_error(y_test, y_test_pred)
final_mse = mean_squared_error(y_test, y_test_pred)
final_r2 = r2_score(y_test, y_test_pred)

print(f"Final Model on the entire training set: {final_model_name}")
print(f"Mean Absolute Error on Test Set: {final_mae}")
print(f"Mean Squared Error on Test Set: {final_mse}")
print(f"R-squared on Test Set: {final_r2}")



######################################################################################################
######################################################################################################
# Phase 5: Reporting
#######################################################################################################
######################################################################################################

print("\nError Analysis:")

# Calculate error (difference between predicted and actual values)
error = y_test - y_test_pred

# Identify cases where the model performed well (small error)
well_predicted_indices = error.abs().argsort()[:5]

# Identify cases where the model struggled (large error)
struggled_indices = error.abs().argsort()[-5:]

# Print insights for well-predicted cases
print("Well-predicted cases:")
print(insurance_data.iloc[well_predicted_indices])

# Print insights for struggled cases
print("\nStruggled cases:")
print(insurance_data.iloc[struggled_indices])

