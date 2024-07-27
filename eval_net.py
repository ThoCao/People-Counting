import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import hydra
from omegaconf import DictConfig

# Function to calculate precision, recall, and F1 score
def precision_recall_f1(true_counts, predicted_counts):
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

# Function to calculate MSE, MAE, Precision, Recall, and F1 score
def evaluate_metrics(base_path: str, ground_truth_csv: str, output_csv: str, model_names: list):
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

# Hydra main function to read configuration
@hydra.main(version_base=None, config_path=".", config_name="eval_config")
def main(cfg: DictConfig) -> None:
    evaluate_metrics(cfg.base_path, cfg.ground_truth_csv, cfg.output_csv, cfg.model_names)

if __name__ == "__main__":
    main()