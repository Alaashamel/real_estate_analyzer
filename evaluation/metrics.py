"""
Evaluation metrics for multimodal price prediction
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

class ModelEvaluator:
    """Evaluate and compare different model configurations"""
    
    def __init__(self):
        self.results = {}
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate all evaluation metrics"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # Calculate percentage error
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'MAPE': mape
        }
    
    def compare_models(self, predictions_dict, y_true):
        """
        Compare multiple models
        predictions_dict: {'model_name': predictions_array}
        """
        results = {}
        
        for model_name, y_pred in predictions_dict.items():
            metrics = self.calculate_metrics(y_true, y_pred)
            results[model_name] = metrics
            print(f"\n📊 {model_name}:")
            for metric, value in metrics.items():
                print(f"   {metric}: {value:.2f}")
        
        self.results = results
        return results
    
    def plot_comparison(self, save_path=None):
        """Plot comparison of models"""
        if not self.results:
            print("No results to plot. Run compare_models first.")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        model_names = list(self.results.keys())
        
        # MAE comparison
        mae_values = [self.results[m]['MAE'] for m in model_names]
        axes[0].bar(model_names, mae_values, color='skyblue')
        axes[0].set_title('Mean Absolute Error (MAE)')
        axes[0].set_ylabel('EGP')
        axes[0].tick_params(axis='x', rotation=45)
        
        # RMSE comparison
        rmse_values = [self.results[m]['RMSE'] for m in model_names]
        axes[1].bar(model_names, rmse_values, color='lightcoral')
        axes[1].set_title('Root Mean Square Error (RMSE)')
        axes[1].set_ylabel('EGP')
        axes[1].tick_params(axis='x', rotation=45)
        
        # R² comparison
        r2_values = [self.results[m]['R²'] for m in model_names]
        axes[2].bar(model_names, r2_values, color='lightgreen')
        axes[2].set_title('R² Score')
        axes[2].set_ylabel('Score')
        axes[2].set_ylim(0, 1)
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"📊 Plot saved to {save_path}")
        
        plt.show()
    
    def plot_predictions_vs_actual(self, y_true, y_pred, model_name, save_path=None):
        """Scatter plot of predictions vs actual"""
        plt.figure(figsize=(8, 8))
        
        plt.scatter(y_true, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
        
        # Add perfect prediction line
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction')
        
        plt.xlabel('Actual Price (EGP)')
        plt.ylabel('Predicted Price (EGP)')
        plt.title(f'{model_name}: Predictions vs Actual')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            print(f"📊 Plot saved to {save_path}")
        
        plt.show()
    
    def plot_error_distribution(self, y_true, y_pred, model_name, save_path=None):
        """Plot distribution of prediction errors"""
        errors = y_pred - y_true
        percentage_errors = (errors / np.maximum(y_true, 1)) * 100
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Absolute error distribution
        axes[0].hist(np.abs(errors), bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Absolute Error (EGP)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'{model_name}: Absolute Error Distribution')
        axes[0].axvline(np.median(np.abs(errors)), color='r', linestyle='--', label=f'Median: {np.median(np.abs(errors)):,.0f}')
        axes[0].legend()
        
        # Percentage error distribution
        axes[1].hist(percentage_errors, bins=50, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Percentage Error (%)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'{model_name}: Percentage Error Distribution')
        axes[1].axvline(0, color='g', linestyle='-')
        axes[1].axvline(np.median(percentage_errors), color='r', linestyle='--', 
                        label=f'Median: {np.median(percentage_errors):.1f}%')
        axes[1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()


def compare_modalities(text_predictions, image_predictions, multimodal_predictions, y_true):
    """
    Compare performance of different modality combinations
    """
    evaluator = ModelEvaluator()
    
    predictions = {
        'Text Only': text_predictions,
        'Image Only': image_predictions,
        'Multimodal (Text + Image)': multimodal_predictions
    }
    
    results = evaluator.compare_models(predictions, y_true)
    
    # Print improvement summary
    print("\n" + "="*50)
    print("📈 IMPROVEMENT SUMMARY")
    print("="*50)
    
    multimodal_mae = results['Multimodal (Text + Image)']['MAE']
    text_mae = results['Text Only']['MAE']
    image_mae = results['Image Only']['MAE']
    
    improvement_vs_text = ((text_mae - multimodal_mae) / text_mae) * 100
    improvement_vs_image = ((image_mae - multimodal_mae) / image_mae) * 100
    
    print(f"📊 Multimodal model improves over Text-only by: {improvement_vs_text:.1f}%")
    print(f"📊 Multimodal model improves over Image-only by: {improvement_vs_image:.1f}%")
    
    return results