import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
import os
import sys
import time
import joblib

def main():
    try:
        print("Loading dataset...")
        # Check if the file exists
        file_path = "mosquito_genomic_data.csv"
        if not os.path.exists(file_path):
            print("\nTop 10 Most Important Features:")
            print(feature_importance.head(10))
            
            # Save the model and feature names
            model_path = 'mosquito_model.joblib'
            model_info = {
                'model': best_rf,
                'feature_names': X.columns.tolist(),
                'species_labels': sorted(y.unique().tolist()),
                'feature_importance': feature_importance
            }
            joblib.dump(model_info, model_path)
            print(f"\nModel saved to {os.path.abspath(model_path)}")
            
            return 0
        # Load the dataset
        data = pd.read_csv(file_path)
        
        # Verify the dataset has the required column
        if "species" not in data.columns:
            print("Error: The dataset does not contain a 'species' column.")
            print(f"Available columns: {', '.join(data.columns)}")
            return 1
            
        print(f"Dataset loaded successfully with {data.shape[0]} samples and {data.shape[1]} features.")
        
        # Prepare features and target
        X = data.drop("species", axis=1)
        y = data["species"]
        
        print(f"Target variable distribution:\n{y.value_counts()}")
        
        # Split into training and testing sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
        
        # Initialize the random forest classifier
        rf = RandomForestClassifier(random_state=42)
        
        # Define the grid of parameters to search over
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
        
        print("Starting Grid Search with 5-fold cross-validation...")
        print(f"Parameter grid: {param_grid}")
        start_time = time.time()
        
        try:
            # Set up the grid search with 5-fold cross-validation
            grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1)
            grid_search.fit(X_train, y_train)
            
            # Calculate time taken
            elapsed_time = time.time() - start_time
            print(f"Grid search completed in {elapsed_time:.2f} seconds.")
            
            # Print out the best parameters and the model accuracy on the test set
            print("\nBest Parameters:", grid_search.best_params_)
            print("Cross-validation score (accuracy):", grid_search.best_score_)
            print("Model Accuracy on test set:", grid_search.score(X_test, y_test))
            
            # Get feature importance
            best_rf = grid_search.best_estimator_
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': best_rf.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            print("\nTop 10 Most Important Features:")
            print(feature_importance.head(10))
            
            return 0
            
        except Exception as e:
            print(f"An error occurred during model training: {str(e)}")
            return 1
            
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

