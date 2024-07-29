import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import hydra
from omegaconf import DictConfig

def precision_recall_f1(true_counts, predicted_counts):
    """
    Calculates precision, recall, and F1 score based on true and predicted counts.

    Args:
        true_counts (list): List of true counts.
        predicted_counts (list): List of predicted counts.

    Returns:
        tuple: Precision, recall, and F1 score rounded to 4 decimal places.
    """
    true_positives = sum(min(gt, pred) for gt, pred in zip(true_counts, predicted_counts))
    false_positives = sum(max(0, pred - gt) for gt, pred in zip(true_counts, predicted_counts))
    false_negatives = sum(max(0, gt - pred) for gt, pred in zip(true_counts, predicted_counts))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Round to 4 decimal places
    precision = round(precision, 4)
    recall = round(recall, 4)
    f1 = round(f1, 4)
    
    return precision, recall, f1

def evaluate_metrics(base_path: str, ground_truth_csv: str, output_csv: str, model_names: list):
    """
    Evaluates metrics for people count predictions and saves results to a CSV file.

    Args:
        base_path (str): Base path where CSV files are located.
        ground_truth_csv (str): Filename of the ground truth CSV.
        output_csv (str): Filename for the output CSV to save results.
        model_names (list): List of model names to evaluate.
    """
    # Construct the full path for the ground truth CSV file
    ground_truth_csv_path = os.path.join(base_path, ground_truth_csv)
    
    # Read the ground truth CSV file
    gt_df = pd.read_csv(ground_truth_csv_path)
    
    metrics = []
    
    for model_name in model_names:
        result_csv_name = f"out_result_{model_name}.csv"
        result_csv_path = os.path.join(base_path, result_csv_name)
        
        if not os.path.exists(result_csv_path):
            print(f"Result file for {model_name} does not exist. Skipping...")
            continue
        
        # Read the result CSV file
        result_df = pd.read_csv(result_csv_path)
        true_counts = gt_df['People Count'].values
        predicted_counts = result_df['People Count'].values
        
        # Calculate MSE and MAE
        mse = mean_squared_error(true_counts, predicted_counts)
        mae = mean_absolute_error(true_counts, predicted_counts)
        
        # Calculate Precision, Recall, and F1 score
        precision, recall, f1 = precision_recall_f1(true_counts, predicted_counts)
        
        # Prepare the results
        results = {
            'Model Name': model_name,
            'Mean Squared Error': mse,
            'Mean Absolute Error': mae,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }
        
        metrics.append(results)
    
    # Write the results to the output CSV file
    results_df = pd.DataFrame(metrics)
    output_csv_path = os.path.join(base_path, output_csv)
    results_df.to_csv(output_csv_path, index=False)
    print(f"Metrics saved to {output_csv_path}")
    print(results_df)

@hydra.main(version_base=None, config_path="./cfg", config_name="eval_config")
def main(cfg: DictConfig) -> None:
    """
    Main function to read configuration and evaluate metrics.

    Args:
        cfg (DictConfig): Configuration dictionary containing evaluation settings.
    """
    evaluate_metrics(cfg.base_path, cfg.ground_truth_csv, cfg.output_csv, cfg.model_names)

if __name__ == "__main__":
    main()