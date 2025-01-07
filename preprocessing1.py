import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
from ensemble_model import EnsembleModel

# Load the dataset
data = pd.read_csv('data/demo.csv')

# Preprocessing
data['Formatted Date'] = pd.to_datetime(data['Formatted Date'])
data = data.drop(columns=['Formatted Date', 'Loud Cover', 'Daily Summary'])  # Drop irrelevant columns

# Handle missing values
data = data.dropna()

# Ensure all required columns are present
required_columns = ['Humidity', 'Wind Speed (km/h)', 'Wind Bearing (degrees)', 
                    'Visibility (km)', 'Pressure (millibars)', 'Summary', 'Precip Type']
missing_columns = [col for col in required_columns if col not in data.columns]

if missing_columns:
    raise ValueError(f"The dataset is missing required columns: {missing_columns}")

# Separate features and target
target = 'Temperature (C)'
X = data[required_columns]
y = data[target]

# Identify categorical and numerical features
categorical_features = ['Summary', 'Precip Type']
numerical_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
numerical_features = [col for col in numerical_features if col not in categorical_features]

# Preprocessing pipeline
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler()), 
    ('pca', PCA(n_components=0.95))  # Keep 95% of variance with PCA
])
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'LinearRegression': LinearRegression(),
    'DecisionTree': DecisionTreeRegressor(random_state=42),
    'RandomForest': RandomForestRegressor(random_state=42),
    'GradientBoosting': GradientBoostingRegressor(random_state=42)
}

# Train and evaluate models
trained_models = {}
model_r2_scores = {}
model_mae = {}
model_rmse = {}

for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    model_r2_scores[name] = r2
    model_mae[name] = mae
    model_rmse[name] = rmse
    
    print(f'{name} - MSE: {mse}, R²: {r2}, MAE: {mae}, RMSE: {rmse}')
    trained_models[name] = pipeline

# Train the ensemble model
ensemble_model = EnsembleModel(trained_models)

# Evaluate the ensemble model
ensemble_pred = ensemble_model.predict(X_test)
ensemble_r2 = r2_score(y_test, ensemble_pred)
ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))

# Add ensemble model metrics to the dictionary
model_r2_scores['Ensemble'] = ensemble_r2
model_mae['Ensemble'] = ensemble_mae
model_rmse['Ensemble'] = ensemble_rmse

# Save models
for name, model in trained_models.items():
    joblib.dump(model, f'{name}_model.joblib')

joblib.dump(ensemble_model, 'Ensemble_model.joblib')

# Save the model metrics
joblib.dump(model_r2_scores, 'model_r2_scores.joblib')
joblib.dump(model_mae, 'model_mae_scores.joblib')
joblib.dump(model_rmse, 'model_rmse_scores.joblib')