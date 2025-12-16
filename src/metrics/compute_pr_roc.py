import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score


def calculate_detection_labels_and_scores(all_particles: Dict[int, Dict[str, List]],
                                          gt_track: Dict[str, Any],
                                          distance_threshold: float = 20.0) -> Tuple[np.ndarray, np.ndarray]:

    labels = []
    scores = []
    
    gt_frames = set(gt_track['frames'])
    gt_positions = np.array(gt_track['positions'])
    gt_frame_to_idx = {frame: idx for idx, frame in enumerate(gt_track['frames'])}
    
    # Track which GT frames were successfully detected
    detected_gt_frames = set()
    
    # Process all detections
    for frame, particles in all_particles.items():
        if not particles['positions']:
            continue
        
        det_positions = np.array(particles['positions'])
        det_scores = np.array(particles['scores'])
        
        if frame in gt_frames:
            # Frame has ground truth - find best match
            gt_pos = gt_positions[gt_frame_to_idx[frame]]
            distances = np.sqrt(np.sum((det_positions - gt_pos)**2, axis=1))
            
            # Find the BEST (closest) detection
            best_idx = np.argmin(distances)
            best_dist = distances[best_idx]
            
            # Label the best match
            if best_dist <= distance_threshold:
                labels.append(1)  # True Positive
                scores.append(det_scores[best_idx])
                detected_gt_frames.add(frame)
            else:
                labels.append(0)  # False Positive (detected but too far)
                scores.append(det_scores[best_idx])
            
            # All OTHER detections in this frame are False Positives
            for i in range(len(det_positions)):
                if i != best_idx:
                    labels.append(0)
                    scores.append(det_scores[i])
        else:
            # No ground truth in this frame - all detections are False Positives
            for score in det_scores:
                labels.append(0)
                scores.append(score)
    
    # CRITICAL FIX: Add False Negatives as lowest-confidence "detections"
    # These are GT frames that were either not detected or detected too far away
    for frame in gt_frames:
        if frame not in detected_gt_frames:
            labels.append(1)  # This is a positive sample we failed to detect
            scores.append(0.0)  # Assign lowest possible score
    
    return np.array(labels), np.array(scores)


def calculate_pr_roc_curves(all_particles: Dict[int, Dict[str, List]],
                            gt_track: Dict[str, Any],
                            distance_threshold: float = 20.0) -> Dict[str, Any]:
    """
    Calculate Precision-Recall and ROC curves for detection performance.
    
    FIXED VERSION: Properly handles False Negatives
    """
    # Get labels and scores (now includes FNs)
    labels, scores = calculate_detection_labels_and_scores(all_particles, gt_track, distance_threshold)
    
    if len(labels) == 0 or np.sum(labels) == 0:
        print("Warning: No valid detections found for PR/ROC analysis")
        return None
    
    # Calculate PR curve
    precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
    avg_precision = average_precision_score(labels, scores)
    
    # Calculate ROC curve
    fpr, tpr, roc_thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    # Find optimal thresholds
    # For PR: maximize F1 score
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    optimal_pr_idx = np.argmax(f1_scores)
    optimal_pr_threshold = pr_thresholds[optimal_pr_idx]
    optimal_f1 = f1_scores[optimal_pr_idx]
    
    # For ROC: maximize Youden's J statistic (TPR - FPR)
    j_scores = tpr - fpr
    optimal_roc_idx = np.argmax(j_scores)
    optimal_roc_threshold = roc_thresholds[optimal_roc_idx]
    optimal_j = j_scores[optimal_roc_idx]
    
    return {
        'precision': precision,
        'recall': recall,
        'pr_thresholds': pr_thresholds,
        'avg_precision': avg_precision,
        'optimal_pr_threshold': optimal_pr_threshold,
        'optimal_f1': optimal_f1,
        'fpr': fpr,
        'tpr': tpr,
        'roc_thresholds': roc_thresholds,
        'roc_auc': roc_auc,
        'optimal_roc_threshold': optimal_roc_threshold,
        'optimal_j': optimal_j,
        'num_positives': np.sum(labels),
        'num_negatives': len(labels) - np.sum(labels),
        'total_detections': len(labels),
        # Additional diagnostic info
        'num_fn_added': len(gt_track['frames']) - len([f for f in all_particles.keys() if f in gt_track['frames']])
    }


def evaluate_at_multiple_thresholds(all_particles: Dict[int, Dict[str, List]],
                                    gt_track: Dict[str, Any],
                                    distance_threshold: float = 20.0,
                                    score_thresholds: List[float] = None) -> pd.DataFrame:
    """
    Evaluate detection performance at multiple confidence score thresholds.
    
    FIXED VERSION: Properly calculates FN at each threshold
    """
    if score_thresholds is None:
        score_thresholds = np.linspace(0, 1, 21)  # 0.0, 0.05, 0.10, ..., 1.0
    
    results = []
    gt_frames = set(gt_track['frames'])
    
    for score_thresh in score_thresholds:
        # Filter particles by score threshold
        detected_gt_frames = set()
        fp_count = 0
        
        for frame, particles in all_particles.items():
            # Filter by score
            high_score_positions = []
            high_score_values = []
            for pos, score in zip(particles['positions'], particles['scores']):
                if score >= score_thresh:
                    high_score_positions.append(pos)
                    high_score_values.append(score)
            
            if not high_score_positions:
                continue
            
            if frame in gt_frames:
                # Check if any detection matches GT
                gt_idx = np.where(gt_track['frames'] == frame)[0][0]
                gt_pos = np.array(gt_track['positions'])[gt_idx]
                
                distances = np.sqrt(np.sum((np.array(high_score_positions) - gt_pos)**2, axis=1))
                best_dist = np.min(distances)
                
                if best_dist <= distance_threshold:
                    detected_gt_frames.add(frame)
                    # Count extra detections as FP
                    fp_count += len(high_score_positions) - 1
                else:
                    # All detections are FP (none close enough)
                    fp_count += len(high_score_positions)
            else:
                # No GT in this frame - all detections are FP
                fp_count += len(high_score_positions)
        
        # Calculate metrics
        tp = len(detected_gt_frames)
        fn = len(gt_frames) - tp
        fp = fp_count
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            'score_threshold': score_thresh,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'num_detections': tp + fp
        })
    
    return pd.DataFrame(results)


def plot_pr_roc_curves(pr_roc_data: Dict[str, Any], output_dir: str) -> str:
    """
    Create visualization of PR and ROC curves.
    """
    import os
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # PR Curve
    ax1 = axes[0]
    ax1.plot(pr_roc_data['recall'], pr_roc_data['precision'], 'b-', linewidth=2,
             label=f'AP = {pr_roc_data["avg_precision"]:.3f}')
    ax1.scatter(pr_roc_data['recall'][:-1][np.argmax(2 * (pr_roc_data['precision'][:-1] * pr_roc_data['recall'][:-1]) / 
                                                     (pr_roc_data['precision'][:-1] + pr_roc_data['recall'][:-1] + 1e-10))],
               pr_roc_data['precision'][:-1][np.argmax(2 * (pr_roc_data['precision'][:-1] * pr_roc_data['recall'][:-1]) / 
                                                       (pr_roc_data['precision'][:-1] + pr_roc_data['recall'][:-1] + 1e-10))],
               color='red', s=100, zorder=5,
               label=f'Optimal (F1={pr_roc_data["optimal_f1"]:.3f}, thr={pr_roc_data["optimal_pr_threshold"]:.3f})')
    ax1.set_xlabel('Recall', fontsize=12)
    ax1.set_ylabel('Precision', fontsize=12)
    ax1.set_title('Precision-Recall Curve (FIXED)', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1.05])
    ax1.set_ylim([0, 1.05])
    
    # ROC Curve
    ax2 = axes[1]
    ax2.plot(pr_roc_data['fpr'], pr_roc_data['tpr'], 'b-', linewidth=2,
             label=f'AUC = {pr_roc_data["roc_auc"]:.3f}')
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax2.scatter(pr_roc_data['fpr'][np.argmax(pr_roc_data['tpr'] - pr_roc_data['fpr'])],
               pr_roc_data['tpr'][np.argmax(pr_roc_data['tpr'] - pr_roc_data['fpr'])],
               color='red', s=100, zorder=5,
               label=f'Optimal (J={pr_roc_data["optimal_j"]:.3f}, thr={pr_roc_data["optimal_roc_threshold"]:.3f})')
    ax2.set_xlabel('False Positive Rate', fontsize=12)
    ax2.set_ylabel('True Positive Rate', fontsize=12)
    ax2.set_title('ROC Curve (FIXED)', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1.05])
    ax2.set_ylim([0, 1.05])
    
    # Add summary text
    summary_text = (f"Total Detections: {pr_roc_data['total_detections']}\n"
                   f"True Positives: {pr_roc_data['num_positives']}\n"
                   f"False Positives: {pr_roc_data['num_negatives']}")
    fig.text(0.5, 0.02, summary_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    path = os.path.join(output_dir, "pr_roc_curves.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return path


def plot_threshold_analysis(threshold_df: pd.DataFrame, output_dir: str) -> str:
    """
    Visualize performance metrics across different thresholds.
    """
    import os
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Precision, Recall, F1 vs Threshold
    ax1 = axes[0, 0]
    ax1.plot(threshold_df['score_threshold'], threshold_df['precision'], 'b-o', label='Precision', markersize=4)
    ax1.plot(threshold_df['score_threshold'], threshold_df['recall'], 'g-s', label='Recall', markersize=4)
    ax1.plot(threshold_df['score_threshold'], threshold_df['f1_score'], 'r-^', label='F1 Score', markersize=4)
    ax1.set_xlabel('Score Threshold')
    ax1.set_ylabel('Score')
    ax1.set_title('Performance Metrics vs Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    
    # Plot 2: TP, FP, FN counts
    ax2 = axes[0, 1]
    ax2.plot(threshold_df['score_threshold'], threshold_df['true_positives'], 'g-o', label='True Positives', markersize=4)
    ax2.plot(threshold_df['score_threshold'], threshold_df['false_positives'], 'r-s', label='False Positives', markersize=4)
    ax2.plot(threshold_df['score_threshold'], threshold_df['false_negatives'], 'orange', marker='^', label='False Negatives', markersize=4)
    ax2.set_xlabel('Score Threshold')
    ax2.set_ylabel('Count')
    ax2.set_title('Detection Counts vs Threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Number of detections
    ax3 = axes[1, 0]
    ax3.plot(threshold_df['score_threshold'], threshold_df['num_detections'], 'b-o', markersize=4)
    ax3.set_xlabel('Score Threshold')
    ax3.set_ylabel('Number of Detections')
    ax3.set_title('Total Detections vs Threshold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Optimal threshold
    ax4 = axes[1, 1]
    optimal_idx = threshold_df['f1_score'].idxmax()
    optimal_thresh = threshold_df.loc[optimal_idx, 'score_threshold']
    optimal_f1 = threshold_df.loc[optimal_idx, 'f1_score']
    
    ax4.bar(['Precision', 'Recall', 'F1'], 
            [threshold_df.loc[optimal_idx, 'precision'],
             threshold_df.loc[optimal_idx, 'recall'],
             threshold_df.loc[optimal_idx, 'f1_score']],
            color=['blue', 'green', 'red'], alpha=0.7)
    ax4.set_ylabel('Score')
    ax4.set_title(f'Optimal Performance\n(threshold = {optimal_thresh:.3f})')
    ax4.set_ylim([0, 1.05])
    ax4.grid(True, alpha=0.3, axis='y')
    
    for i, (label, val) in enumerate(zip(['Precision', 'Recall', 'F1'],
                                         [threshold_df.loc[optimal_idx, 'precision'],
                                          threshold_df.loc[optimal_idx, 'recall'],
                                          threshold_df.loc[optimal_idx, 'f1_score']])):
        ax4.text(i, val, f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    path = os.path.join(output_dir, "threshold_analysis.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return path


def evaluate_with_pr_roc(all_particles: Dict[int, Dict[str, List]],
                        ground_truth_csv: str,
                        output_dir: str,
                        distance_threshold: float = 20.0) -> Dict[str, Any]:
    """
    Complete PR/ROC analysis wrapper.
    
    FIXED VERSION: Uses corrected label computation
    """
    import os
    from src.metrics.detection_metrics import load_ground_truth_track
    
    print("\nCalculating PR and ROC curves (FIXED VERSION)...")
    
    # Load ground truth
    gt_track = load_ground_truth_track(ground_truth_csv)
    
    # Calculate PR/ROC curves
    pr_roc_data = calculate_pr_roc_curves(all_particles, gt_track, distance_threshold)
    
    if pr_roc_data is None:
        print("Could not generate PR/ROC curves (insufficient data)")
        return None
    
    # Generate plots
    pr_roc_path = plot_pr_roc_curves(pr_roc_data, output_dir)
    
    # Threshold analysis with fixed FN calculation
    threshold_df = evaluate_at_multiple_thresholds(all_particles, gt_track, distance_threshold)
    threshold_path = plot_threshold_analysis(threshold_df, output_dir)
    
    # Save threshold data
    csv_path = os.path.join(output_dir, "threshold_analysis.csv")
    threshold_df.to_csv(csv_path, index=False)
    
    print(f"\nPR/ROC Analysis Results (FIXED):")
    print(f"  Average Precision (AP): {pr_roc_data['avg_precision']:.3f}")
    print(f"  ROC AUC:                {pr_roc_data['roc_auc']:.3f}")
    print(f"  Optimal F1 Score:       {pr_roc_data['optimal_f1']:.3f}")
    print(f"  Optimal Threshold (F1): {pr_roc_data['optimal_pr_threshold']:.3f}")
    if 'num_fn_added' in pr_roc_data:
        print(f"  False Negatives added:  {pr_roc_data['num_fn_added']}")
    print(f"  Saved plots to:         {pr_roc_path}")
    print(f"  Saved analysis to:      {csv_path}")
    
    return {
        'pr_roc_data': pr_roc_data,
        'threshold_analysis': threshold_df,
        'pr_roc_path': pr_roc_path,
        'threshold_path': threshold_path,
        'csv_path': csv_path
    }