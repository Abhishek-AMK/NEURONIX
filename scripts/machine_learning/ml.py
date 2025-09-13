import os
import sys
import json
import pickle
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, accuracy_score, 
    classification_report, silhouette_score, davies_bouldin_score, 
    calinski_harabasz_score, adjusted_rand_score, normalized_mutual_info_score
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_data(file_path):
    """Read data from various file formats."""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        return pd.read_json(file_path)
    elif file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def impute_missing_values(df):
    """Impute missing values in the dataset."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    
    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    
    for col in cat_cols:
        if df[col].isnull().any():
            mode = df[col].mode()
            if not mode.empty:
                df[col] = df[col].fillna(mode[0])
            else:
                df[col] = df[col].fillna('missing')
    
    return df

def run_supervised_learning(df, config):
    """Execute supervised learning tasks."""
    target_column = config['target_column']
    features = config.get('features', [col for col in df.columns if col != target_column])
    module = config['module']  # 'regression' or 'classification'
    model_type = config['model_type']
    test_size = config.get('test_size', 0.2)
    scale = config.get('scale', True)
    n_jobs = config.get('n_jobs', -1)
    
    X = df[features]
    y = df[target_column]
    
    # Validation checks
    if module == "classification":
        if np.issubdtype(y.dtype, np.number) and y.nunique() > 20:
            raise ValueError("Target appears to be continuous. Use regression instead.")
    
    if module == "regression":
        if y.dtype == 'O' or y.nunique() <= 10:
            raise ValueError("Target appears to be categorical. Use classification instead.")
    
    # Train-test split
    if module == "classification":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42)
    
    # Encode categorical targets for classification
    target_mapping = None
    inverse_target_mapping = None
    if module == "classification":
        if y_train.dtype == 'O' or not np.issubdtype(y_train.dtype, np.number):
            unique_vals = sorted(y_train.unique())
            target_mapping = {val: idx for idx, val in enumerate(unique_vals)}
            inverse_target_mapping = {idx: val for val, idx in target_mapping.items()}
            y_train = y_train.map(target_mapping)
            y_test = y_test.map(target_mapping)
    
    # Preprocessing
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = X_train.select_dtypes(exclude=['object', 'category']).columns.tolist()
    
    transformers = []
    if numeric_cols:
        if scale:
            transformers.append(('num', StandardScaler(), numeric_cols))
        else:
            transformers.append(('num', 'passthrough', numeric_cols))
    
    if categorical_cols:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols))
    
    preprocessor = ColumnTransformer(transformers=transformers)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Model selection and parameter grids
    model_configs = {
        'regression': {
            'linear': (LinearRegression(), {"fit_intercept": [True, False]}),
            'ridge': (Ridge(), {'alpha': [0.01, 0.1, 1, 10, 100]}),
            'lasso': (Lasso(max_iter=10000), {'alpha': [0.01, 0.1, 1, 10, 100]}),
            'xgboost': (xgb.XGBRegressor(eval_metric='rmse', use_label_encoder=False), 
                       {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.05, 0.1]}),
            'randomforest': (RandomForestRegressor(random_state=42), 
                           {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]})
        },
        'classification': {
            'logistic': (LogisticRegression(max_iter=2000), 
                        {'C': [0.01, 0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']}),
            'xgboost': (xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                       {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.05, 0.1]}),
            'randomforest': (RandomForestClassifier(random_state=42),
                           {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]})
        }
    }
    
    model, param_grid = model_configs[module][model_type]
    scoring = 'neg_root_mean_squared_error' if module == 'regression' else 'accuracy'
    
    # Training with hyperparameter tuning
    if param_grid:
        gs = GridSearchCV(
            model, param_grid, scoring=scoring, 
            cv=StratifiedKFold(n_splits=5) if module == "classification" else 5,
            n_jobs=n_jobs, verbose=0
        )
        gs.fit(X_train_processed, y_train)
        best_model = gs.best_estimator_
        best_params = gs.best_params_
    else:
        model.fit(X_train_processed, y_train)
        best_model = model
        best_params = "N/A"
    
    # Predictions and evaluation
    preds = best_model.predict(X_test_processed)
    
    if module == "regression":
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        results = {
            "task_type": "supervised",
            "module": module,
            "model_type": model_type,
            "best_parameters": best_params,
            "metrics": {
                "RMSE": rmse,
                "MAE": mae,
                "R2": r2
            },
            "predictions": {
                "actual": y_test.tolist(),
                "predicted": preds.tolist()
            }
        }
    else:
        acc = accuracy_score(y_test, preds)
        class_report = classification_report(y_test, preds, output_dict=True)
        
        # Decode predictions if needed
        if inverse_target_mapping:
            decoded_actual = [inverse_target_mapping.get(val, val) for val in y_test]
            decoded_pred = [inverse_target_mapping.get(val, val) for val in preds]
        else:
            decoded_actual = y_test.tolist()
            decoded_pred = preds.tolist()
        
        results = {
            "task_type": "supervised",
            "module": module,
            "model_type": model_type,
            "best_parameters": best_params,
            "metrics": {
                "accuracy": acc,
                "classification_report": class_report
            },
            "predictions": {
                "actual": decoded_actual,
                "predicted": decoded_pred
            }
        }
    
    return results, best_model, preprocessor

def run_unsupervised_learning(df, config):
    """Execute unsupervised learning tasks."""
    features = config.get('features', df.select_dtypes(include=[np.number]).columns.tolist())
    unsup_module = config['unsup_module']  # 'clustering', 'dimensionality_reduction', 'anomaly_detection'
    unsup_model = config['unsup_model']
    scale = config.get('scale', True)
    
    X = df[features]
    
    if scale and len(X) > 0:
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)
    else:
        X_scaled = X
        scaler = None
    
    results = {
        "task_type": "unsupervised",
        "module": unsup_module,
        "model_type": unsup_model,
        "features_used": features
    }
    
    if unsup_module == "clustering":
        if unsup_model == "kmeans":
            n_clusters = config.get('n_clusters', 3)
            model = KMeans(n_clusters=n_clusters, random_state=42)
            labels = model.fit_predict(X_scaled)
            
            if len(set(labels)) > 1:
                sil = silhouette_score(X_scaled, labels)
                db = davies_bouldin_score(X_scaled, labels)
                ch = calinski_harabasz_score(X_scaled, labels)
                
                results.update({
                    "n_clusters": n_clusters,
                    "labels": labels.tolist(),
                    "metrics": {
                        "silhouette_score": sil,
                        "davies_bouldin_index": db,
                        "calinski_harabasz_index": ch
                    }
                })
            else:
                results.update({
                    "n_clusters": n_clusters,
                    "labels": labels.tolist(),
                    "warning": "All data assigned to one cluster"
                })
                
        elif unsup_model == "dbscan":
            model = DBSCAN()
            labels = model.fit_predict(X_scaled)
            
            core_mask = labels != -1
            if sum(core_mask) > 1 and len(set(labels[core_mask])) > 1:
                sil = silhouette_score(X_scaled[core_mask], labels[core_mask])
                db = davies_bouldin_score(X_scaled[core_mask], labels[core_mask])
                ch = calinski_harabasz_score(X_scaled[core_mask], labels[core_mask])
                
                results.update({
                    "labels": labels.tolist(),
                    "metrics": {
                        "silhouette_score": sil,
                        "davies_bouldin_index": db,
                        "calinski_harabasz_index": ch
                    }
                })
            else:
                results.update({
                    "labels": labels.tolist(),
                    "warning": "Most points assigned as noise or single cluster"
                })
    
    elif unsup_module == "dimensionality_reduction":
        if unsup_model == "pca":
            n_components = config.get('n_components', 2)
            if n_components > min(X_scaled.shape):
                n_components = min(X_scaled.shape)
            
            model = PCA(n_components=n_components, random_state=42)
            X_pca = model.fit_transform(X_scaled)
            
            results.update({
                "n_components": n_components,
                "explained_variance_ratio": model.explained_variance_ratio_.tolist(),
                "transformed_data": X_pca.tolist()
            })
    
    elif unsup_module == "anomaly_detection":
        if unsup_model == "isolation_forest":
            contamination = config.get('contamination', 0.05)
            model = IsolationForest(contamination=contamination, random_state=42)
            preds = model.fit_predict(X_scaled)
            
            anomaly_labels = ['Anomaly' if p == -1 else 'Normal' for p in preds]
            anomaly_count = sum(1 for p in preds if p == -1)
            
            results.update({
                "contamination": contamination,
                "predictions": anomaly_labels,
                "anomaly_count": anomaly_count,
                "normal_count": len(preds) - anomaly_count
            })
    
    return results, model, scaler

def main():
    """Main function for pipeline integration."""
    try:
        # Get parameters from environment variables
        file_path = os.environ.get('file_path', '')
        task_type = os.environ.get('task_type', 'supervised')  # 'supervised' or 'unsupervised'
        
        if not file_path:
            print("Error: No file path provided")
            sys.exit(1)
        
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            sys.exit(1)
        
        print(f"Processing ML task on file: {file_path}")
        
        # Set up output directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, '..', '..', 'ml_output')
        os.makedirs(output_dir, exist_ok=True)
        
        # Read and preprocess data
        df = read_data(file_path)
        df = impute_missing_values(df)
        
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Parse configuration from environment variables
        if task_type == 'supervised':
            config = {
                'target_column': os.environ.get('target_column', df.columns[-1]),
                'module': os.environ.get('module', 'regression'),  # 'regression' or 'classification'
                'model_type': os.environ.get('model_type', 'linear'),
                'test_size': float(os.environ.get('test_size', '0.2')),
                'scale': os.environ.get('scale', 'true').lower() == 'true',
                'n_jobs': int(os.environ.get('n_jobs', '-1'))
            }
            
            results, model, preprocessor = run_supervised_learning(df, config)
            
            # Save model and preprocessor
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            model_file = os.path.join(output_dir, f"model_{base_name}.pkl")
            preprocessor_file = os.path.join(output_dir, f"preprocessor_{base_name}.pkl")
            
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            with open(preprocessor_file, 'wb') as f:
                pickle.dump(preprocessor, f)
            
            results['model_file'] = model_file
            results['preprocessor_file'] = preprocessor_file
            
        else:  # unsupervised
            config = {
                'unsup_module': os.environ.get('unsup_module', 'clustering'),
                'unsup_model': os.environ.get('unsup_model', 'kmeans'),
                'scale': os.environ.get('scale', 'true').lower() == 'true',
                'n_clusters': int(os.environ.get('n_clusters', '3')),
                'n_components': int(os.environ.get('n_components', '2')),
                'contamination': float(os.environ.get('contamination', '0.05'))
            }
            
            results, model, scaler = run_unsupervised_learning(df, config)
            
            # Save model and scaler
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            model_file = os.path.join(output_dir, f"model_{base_name}.pkl")
            
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            
            if scaler:
                scaler_file = os.path.join(output_dir, f"scaler_{base_name}.pkl")
                with open(scaler_file, 'wb') as f:
                    pickle.dump(scaler, f)
                results['scaler_file'] = scaler_file
            
            results['model_file'] = model_file
        
        # Save results
        results_file = os.path.join(output_dir, f"ml_results_{os.path.splitext(os.path.basename(file_path))[0]}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Terminal output
        print(f"\nML task completed successfully!")
        print(f"Task type: {task_type}")
        print(f"File processed: {os.path.basename(file_path)}")
        
        if task_type == 'supervised':
            print(f"Module: {config['module']}")
            print(f"Model: {config['model_type']}")
            if 'metrics' in results:
                print("Metrics:")
                for metric, value in results['metrics'].items():
                    if isinstance(value, dict):
                        print(f"  {metric}: {json.dumps(value, indent=4)}")
                    else:
                        print(f"  {metric}: {value:.4f}")
        else:
            print(f"Module: {config['unsup_module']}")
            print(f"Model: {config['unsup_model']}")
            if 'metrics' in results:
                print("Metrics:")
                for metric, value in results['metrics'].items():
                    print(f"  {metric}: {value:.4f}")
        
        print(f"\nResults saved to: {results_file}")
        
    except Exception as e:
        print(f"ML task failed: {e}")
        logger.error(f"ML task failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
