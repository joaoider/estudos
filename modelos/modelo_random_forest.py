"""
Random Forest Regressor Model Implementation

This module provides a comprehensive Random Forest regression model implementation
with hyperparameter optimization, model evaluation, and visualization capabilities.

Author: [Your Name]
Date: [Current Date]
Version: 1.0.0
"""

import logging
import warnings
from typing import Dict, Tuple, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.exceptions import NotFittedError
from scipy.stats import randint

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Constants
RANDOM_STATE = 42
DEFAULT_N_ESTIMATORS = 100
N_ESTIMATORS_RANGE = [10, 50, 100, 200, 300]
GRID_SEARCH_PARAMS = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
RANDOM_SEARCH_PARAMS = {
    'n_estimators': randint(50, 300),
    'max_depth': [None] + list(range(5, 21)),
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 5)
}
CV_FOLDS = 5
RANDOM_SEARCH_ITERATIONS = 20


class RandomForestModel:
    """
    A professional Random Forest regression model implementation.
    
    This class encapsulates all Random Forest functionality including training,
    hyperparameter optimization, evaluation, and visualization.
    """
    
    def __init__(self, random_state: int = RANDOM_STATE):
        """
        Initialize the Random Forest model.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.baseline_model = None
        self.grid_search_model = None
        self.random_search_model = None
        self.best_model = None
        self.feature_importances = None
        
    def train_baseline_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
        """
        Train a baseline Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Trained Random Forest model
        """
        try:
            self.baseline_model = RandomForestRegressor(
                n_estimators=DEFAULT_N_ESTIMATORS,
                random_state=self.random_state
            )
            self.baseline_model.fit(X_train, y_train)
            self.feature_importances = pd.Series(
                self.baseline_model.feature_importances_,
                index=X_train.columns
            )
            logger.info("Baseline model trained successfully")
            return self.baseline_model
        except Exception as e:
            logger.error(f"Error training baseline model: {e}")
            raise
    
    def evaluate_model(self, model: RandomForestRegressor, X_test: pd.DataFrame, 
                      y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate a trained model using multiple metrics.
        
        Args:
            model: Trained model to evaluate
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary containing evaluation metrics
        """
        try:
            y_pred = model.predict(X_test)
            metrics = {
                'MAE': mean_absolute_error(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'R²': r2_score(y_test, y_pred)
            }
            return metrics
        except NotFittedError:
            logger.error("Model must be fitted before evaluation")
            raise
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise
    
    def optimize_hyperparameters_grid(self, X_train: pd.DataFrame, 
                                    y_train: pd.Series) -> GridSearchCV:
        """
        Perform grid search hyperparameter optimization.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Fitted GridSearchCV object
        """
        try:
            grid_search = GridSearchCV(
                RandomForestRegressor(random_state=self.random_state),
                GRID_SEARCH_PARAMS,
                cv=CV_FOLDS,
                scoring='neg_root_mean_squared_error',
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X_train, y_train)
            self.grid_search_model = grid_search
            logger.info("Grid search completed successfully")
            return grid_search
        except Exception as e:
            logger.error(f"Error in grid search: {e}")
            raise
    
    def optimize_hyperparameters_random(self, X_train: pd.DataFrame, 
                                     y_train: pd.Series) -> RandomizedSearchCV:
        """
        Perform randomized search hyperparameter optimization.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Fitted RandomizedSearchCV object
        """
        try:
            random_search = RandomizedSearchCV(
                RandomForestRegressor(random_state=self.random_state),
                RANDOM_SEARCH_PARAMS,
                n_iter=RANDOM_SEARCH_ITERATIONS,
                cv=CV_FOLDS,
                scoring='neg_root_mean_squared_error',
                n_jobs=-1,
                verbose=0,
                random_state=self.random_state
            )
            random_search.fit(X_train, y_train)
            self.random_search_model = random_search
            logger.info("Random search completed successfully")
            return random_search
        except Exception as e:
            logger.error(f"Error in random search: {e}")
            raise
    
    def plot_feature_importance(self, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Create a feature importance visualization.
        
        Args:
            figsize: Figure size as (width, height)
            
        Returns:
            Matplotlib figure object
        """
        if self.feature_importances is None:
            raise ValueError("Model must be trained before plotting feature importance")
        
        fig, ax = plt.subplots(figsize=figsize)
        self.feature_importances.sort_values().plot(kind='barh', ax=ax)
        ax.set_title("Feature Importance - Random Forest", fontsize=14, fontweight='bold')
        ax.set_xlabel("Importance Score", fontsize=12)
        ax.set_ylabel("Features", fontsize=12)
        plt.tight_layout()
        return fig
    
    def plot_n_estimators_analysis(self, X_train: pd.DataFrame, y_train: pd.Series,
                                  X_test: pd.DataFrame, y_test: pd.Series) -> plt.Figure:
        """
        Analyze the effect of n_estimators on model performance.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            
        Returns:
            Matplotlib figure object
        """
        r2_scores, rmse_scores = [], []
        
        for n in N_ESTIMATORS_RANGE:
            rf = RandomForestRegressor(n_estimators=n, random_state=self.random_state)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            r2_scores.append(r2_score(y_test, y_pred))
            rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # R² plot
        ax1.plot(N_ESTIMATORS_RANGE, r2_scores, marker='o', linewidth=2, markersize=8)
        ax1.set_title("R² vs Number of Trees", fontsize=14, fontweight='bold')
        ax1.set_xlabel("n_estimators", fontsize=12)
        ax1.set_ylabel("R² Score", fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # RMSE plot
        ax2.plot(N_ESTIMATORS_RANGE, rmse_scores, marker='s', color='orange', 
                linewidth=2, markersize=8)
        ax2.set_title("RMSE vs Number of Trees", fontsize=14, fontweight='bold')
        ax2.set_xlabel("n_estimators", fontsize=12)
        ax2.set_ylabel("RMSE", fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_prediction_scatter(self, y_test: pd.Series, y_pred_grid: np.ndarray,
                               y_pred_random: np.ndarray) -> plt.Figure:
        """
        Create scatter plots comparing predicted vs actual values.
        
        Args:
            y_test: Actual test values
            y_pred_grid: Grid search predictions
            y_pred_random: Random search predictions
            
        Returns:
            Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Grid search scatter plot
        ax1.scatter(y_test, y_pred_grid, alpha=0.6, s=50)
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', linewidth=2, label='Perfect Prediction')
        ax1.set_title("GridSearchCV: Actual vs Predicted", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Actual Values", fontsize=12)
        ax1.set_ylabel("Predicted Values", fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Random search scatter plot
        ax2.scatter(y_test, y_pred_random, alpha=0.6, s=50, color='orange')
        ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', linewidth=2, label='Perfect Prediction')
        ax2.set_title("RandomizedSearchCV: Actual vs Predicted", fontsize=14, fontweight='bold')
        ax2.set_xlabel("Actual Values", fontsize=12)
        ax2.set_ylabel("Predicted Values", fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


def run_random_forest_analysis(X_train: pd.DataFrame, X_test: pd.DataFrame,
                              y_train: pd.Series, y_test: pd.Series,
                              X: pd.DataFrame, st) -> Dict[str, Any]:
    """
    Main function to run complete Random Forest analysis.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target
        X: Full feature dataset for column names
        st: Streamlit object for UI
        
    Returns:
        Dictionary containing the best model results
    """
    st.subheader("🌲 Random Forest Regressor Analysis")
    
    try:
        # Initialize model
        rf_model = RandomForestModel(random_state=RANDOM_STATE)
        
        # Train baseline model
        baseline_model = rf_model.train_baseline_model(X_train, y_train)
        baseline_metrics = rf_model.evaluate_model(baseline_model, X_test, y_test)
        
        # Display baseline results
        st.markdown("### 🎯 Baseline Model Performance")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("MAE", f"{baseline_metrics['MAE']:.3f}")
        with col2:
            st.metric("RMSE", f"{baseline_metrics['RMSE']:.3f}")
        with col3:
            st.metric("R²", f"{baseline_metrics['R²']:.3f}")
        
        # Feature importance
        st.markdown("### 🔍 Feature Importance Analysis")
        feature_fig = rf_model.plot_feature_importance()
        st.pyplot(feature_fig)
        plt.close(feature_fig)
        
        # N_estimators analysis
        st.markdown("### 📈 Hyperparameter Analysis: n_estimators")
        n_estimators_fig = rf_model.plot_n_estimators_analysis(
            X_train, y_train, X_test, y_test
        )
        st.pyplot(n_estimators_fig)
        plt.close(n_estimators_fig)
        
        # Hyperparameter optimization
        st.markdown("### 🔬 Hyperparameter Optimization")
        
        with st.spinner("Performing Grid Search..."):
            grid_search = rf_model.optimize_hyperparameters_grid(X_train, y_train)
        
        with st.spinner("Performing Randomized Search..."):
            random_search = rf_model.optimize_hyperparameters_random(X_train, y_train)
        
        # Display optimization results
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**GridSearchCV Results:**")
            st.json(grid_search.best_params_)
            st.write(f"Best RMSE: {-grid_search.best_score_:.3f}")
        
        with col2:
            st.markdown("**RandomizedSearchCV Results:**")
            st.json(random_search.best_params_)
            st.write(f"Best RMSE: {-random_search.best_score_:.3f}")
        
        # Evaluate optimized models
        st.markdown("### 📊 Optimized Models Performance")
        
        y_pred_grid = grid_search.predict(X_test)
        y_pred_random = random_search.predict(X_test)
        
        grid_metrics = rf_model.evaluate_model(grid_search.best_estimator_, X_test, y_test)
        random_metrics = rf_model.evaluate_model(random_search.best_estimator_, X_test, y_test)
        
        # Display metrics comparison
        comparison_data = {
            "Metric": ["MAE", "RMSE", "R²"],
            "GridSearchCV": [grid_metrics["MAE"], grid_metrics["RMSE"], grid_metrics["R²"]],
            "RandomizedSearchCV": [random_metrics["MAE"], random_metrics["RMSE"], random_metrics["R²"]]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Prediction scatter plots
        st.markdown("### 📈 Prediction vs Actual Comparison")
        scatter_fig = rf_model.plot_prediction_scatter(y_test, y_pred_grid, y_pred_random)
        st.pyplot(scatter_fig)
        plt.close(scatter_fig)
        
        # Determine best model
        if grid_metrics["RMSE"] <= random_metrics["RMSE"]:
            best_model_name = "Random Forest (GridSearch)"
            best_metrics = grid_metrics
            best_params = grid_search.best_params_
            best_estimator = grid_search.best_estimator_
        else:
            best_model_name = "Random Forest (RandomizedSearch)"
            best_metrics = random_metrics
            best_params = random_search.best_params_
            best_estimator = random_search.best_estimator_
        
        # Final summary
        st.markdown("### 🏆 Best Model Summary")
        st.success(f"**Selected Model:** {best_model_name}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Best MAE", f"{best_metrics['MAE']:.3f}")
        with col2:
            st.metric("Best RMSE", f"{best_metrics['RMSE']:.3f}")
        with col3:
            st.metric("Best R²", f"{best_metrics['R²']:.3f}")
        
        return {
            "Modelo": best_model_name,
            "MAE": best_metrics["MAE"],
            "RMSE": best_metrics["RMSE"],
            "R²": best_metrics["R²"],
            "Parâmetros": best_params,
            "Estimator": best_estimator
        }
        
    except Exception as e:
        st.error(f"❌ Error during Random Forest analysis: {str(e)}")
        logger.error(f"Random Forest analysis failed: {e}")
        raise


# Legacy function for backward compatibility
def rodar_random_forest(X_train, X_test, y_train, y_test, X, st):
    """
    Legacy function - use run_random_forest_analysis instead.
    
    This function is maintained for backward compatibility but is deprecated.
    """
    import warnings
    warnings.warn(
        "rodar_random_forest is deprecated. Use run_random_forest_analysis instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return run_random_forest_analysis(X_train, X_test, y_train, y_test, X, st)