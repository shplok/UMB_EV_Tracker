import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import cv2
import os
from typing import Dict, List, Any, Optional


def create_comprehensive_track_report(image_stack: np.ndarray,
                                     tracks: Dict[int, Dict[str, Any]],
                                     all_particles: Dict[int, Dict[str, List]],
                                     gt_track: Dict[str, Any],
                                     frame_metrics: Dict[str, Any],
                                     output_dir: str,
                                     top_n_tracks: int = 5) -> str:
    
    # Sort tracks by quality
    sorted_tracks = sorted(tracks.items(),
                          key=lambda x: (x[1]['avg_detection_score'], len(x[1]['frames'])),
                          reverse=True)
    top_tracks = sorted_tracks[:top_n_tracks]
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # === TOP ROW: Track overlay image (spans 2 columns) + Summary metrics ===
    ax_overlay = fig.add_subplot(gs[0, :2])
    ax_summary = fig.add_subplot(gs[0, 2])
    
    # Create track overlay on mid-frame
    mid_frame = len(image_stack) // 2
    base_frame = image_stack[mid_frame]
    frame_norm = cv2.normalize(base_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    frame_rgb = cv2.cvtColor(frame_norm, cv2.COLOR_GRAY2RGB)
    
    # Color palette for tracks
    colors = plt.cm.tab10(np.linspace(0, 1, top_n_tracks))
    
    # Draw ground truth if available
    if gt_track:
        gt_positions = np.array(gt_track['positions'])
        ax_overlay.plot(gt_positions[:, 0], gt_positions[:, 1], 
                       'w--', linewidth=3, alpha=0.7, label='Ground Truth')
        ax_overlay.scatter(gt_positions[0, 0], gt_positions[0, 1], 
                          c='white', s=200, marker='*', edgecolors='black', linewidths=2)
    
    # Draw top tracks
    for idx, (track_id, track) in enumerate(top_tracks):
        positions = np.array(track['positions'])
        color = colors[idx]
        
        # Draw path
        ax_overlay.plot(positions[:, 0], positions[:, 1], 
                       color=color, linewidth=2, alpha=0.8,
                       label=f"Track #{track_id} (score={track['avg_detection_score']:.2f})")
        
        # Start marker
        ax_overlay.scatter(positions[0, 0], positions[0, 1], 
                          c=[color], s=150, marker='o', edgecolors='black', linewidths=2)
        
        # End marker
        ax_overlay.scatter(positions[-1, 0], positions[-1, 1], 
                          c=[color], s=150, marker='X', edgecolors='black', linewidths=2)
    
    ax_overlay.imshow(frame_rgb, cmap='gray', alpha=0.5)
    ax_overlay.set_title('Track Overlay (○=start, X=end)', fontsize=14, fontweight='bold')
    ax_overlay.legend(loc='upper right', fontsize=9)
    ax_overlay.axis('equal')
    ax_overlay.set_xlabel('X Position (pixels)')
    ax_overlay.set_ylabel('Y Position (pixels)')
    
    # Summary metrics bar chart - Get overall metrics from results if available
    # Check if we have track metrics from the matched track
    track_recall = 0
    track_precision = 0
    track_f1 = 0
    
    # Look for matched track in tracks
    matched_track_id = None
    for track_id, track in tracks.items():
        # Simple heuristic: track with most overlap with ground truth frames
        if gt_track:
            gt_frames_set = set(gt_track['frames'])
            track_frames_set = set(track['frames'])
            overlap = len(gt_frames_set.intersection(track_frames_set))
            if overlap >= len(gt_frames_set) * 0.5:  # At least 50% overlap
                matched_track_id = track_id
                # Calculate metrics for this track
                track_recall = overlap / len(gt_frames_set)
                track_precision = overlap / len(track_frames_set)
                if track_precision + track_recall > 0:
                    track_f1 = 2 * (track_precision * track_recall) / (track_precision + track_recall)
                break
    
    metrics_names = ['Detection\nRate', 'Overall\nAccuracy', 'Track\nRecall', 'Track\nPrecision', 'F1\nScore']
    metrics_values = [
        frame_metrics.get('detection_rate', 0),
        frame_metrics.get('overall_frame_accuracy', 0),
        track_recall,
        track_precision,
        track_f1
    ]

    
    bars = ax_summary.bar(metrics_names, metrics_values, 
        color=['#2E86AB', '#6C63FF', '#A23B72', '#F18F01', '#06A77D'],
        alpha=0.8, edgecolor='black', linewidth=1.5)

    
    # Add value labels on bars
    for bar, val in zip(bars, metrics_values):
        height = bar.get_height()
        ax_summary.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}', ha='center', va='bottom', 
                       fontweight='bold', fontsize=11)
    
    ax_summary.set_ylim([0, 1.1])
    ax_summary.set_ylabel('Score', fontsize=12)
    ax_summary.set_title('Performance Summary', fontsize=14, fontweight='bold')
    ax_summary.grid(axis='y', alpha=0.3)
    ax_summary.axhline(y=0.75, color='green', linestyle='--', alpha=0.5, label='Good (>0.75)')
    ax_summary.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Fair (>0.50)')
    
    # === MIDDLE & BOTTOM ROWS: Per-track detailed metrics ===
    # ONLY show tracks that have meaningful overlap with ground truth
    
    # Filter tracks to only those with ground truth overlap
    tracks_with_gt = []
    if gt_track:
        gt_frames_set = set(gt_track['frames'])
        
        for track_id, track in top_tracks:
            track_frames_set = set(track['frames'])
            overlap = len(gt_frames_set.intersection(track_frames_set))
            
            # Only include if track has at least 5 overlapping frames
            if overlap >= 5:
                tracks_with_gt.append((track_id, track, overlap))
        
        # Sort by overlap (best matches first)
        tracks_with_gt.sort(key=lambda x: x[2], reverse=True)
    
    # Determine how many tracks we can actually show (max 5)
    num_tracks_to_show = min(len(tracks_with_gt), 5)
    
    if num_tracks_to_show == 0:
        # No tracks with GT overlap - add a message
        ax_msg = fig.add_subplot(gs[1:, :])
        ax_msg.text(0.5, 0.5, 'No tracks with sufficient ground truth overlap found.\nMinimum 5 overlapping frames required.',
                   ha='center', va='center', fontsize=14, 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax_msg.axis('off')
    else:
        row_positions = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]
        
        for idx in range(num_tracks_to_show):
            track_id, track, overlap_count = tracks_with_gt[idx]
            
            row, col = row_positions[idx]
            ax = fig.add_subplot(gs[row, col])
        
            # Get ground truth metrics for this track
            track_frames = track['frames']
            gt_frames_set = set(gt_track['frames']) if gt_track else set()
            
            # Calculate position errors and detection status for frames
            position_errors = []
            detection_scores = []
            frame_numbers = []
            colors_per_frame = []
            
            for i, frame in enumerate(track_frames):
                score = track['scores'][i]
                
                if frame in gt_frames_set:
                    # Calculate position error
                    gt_idx = list(gt_track['frames']).index(frame)
                    gt_pos = np.array(gt_track['positions'][gt_idx])
                    track_pos = np.array(track['positions'][i])
                    error = np.linalg.norm(gt_pos - track_pos)
                    
                    position_errors.append(error)
                    detection_scores.append(score)
                    frame_numbers.append(frame)
                    
                    # Color by accuracy
                    if error < 15:
                        colors_per_frame.append('#06A77D')  # Green = good
                    elif error < 25:
                        colors_per_frame.append('#F18F01')  # Orange = ok
                    else:
                        colors_per_frame.append('#A23B72')  # Red = poor
            
            # Only plot if we have data
            if not position_errors:
                continue
            
            # Create twin axes for position error and detection score
            ax2 = ax.twinx()
            
            # Plot position errors as bars
            bars = ax.bar(frame_numbers, position_errors, 
                         color=colors_per_frame, alpha=0.6, 
                         label='Position Error', width=0.8)
            ax.axhline(y=15, color='green', linestyle='--', alpha=0.4, linewidth=1)
            ax.axhline(y=25, color='orange', linestyle='--', alpha=0.4, linewidth=1)
            
            # Plot detection scores as line
            ax2.plot(frame_numbers, detection_scores, 
                    'ko-', linewidth=2, markersize=4, 
                    label='Detection Score')
            
            ax.set_xlabel('Frame Number', fontsize=10)
            ax.set_ylabel('Position Error (px)', fontsize=10, color='#2E86AB')
            ax2.set_ylabel('Detection Score', fontsize=10, color='black')
            ax.set_title(f'Track #{track_id} Performance ({overlap_count} GT frames)', 
                        fontsize=11, fontweight='bold')
            ax.tick_params(axis='y', labelcolor='#2E86AB')
            ax2.tick_params(axis='y', labelcolor='black')
            ax.grid(axis='x', alpha=0.3)
            
            # Add track info text
            avg_error = np.mean(position_errors)
            avg_score = np.mean(detection_scores)
            info_text = f'Avg Error: {avg_error:.1f}px\nAvg Score: {avg_score:.2f}\nLength: {len(track_frames)} frames'
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Overall title
    fig.suptitle('Comprehensive Tracking Analysis Report', 
                fontsize=16, fontweight='bold', y=0.995)
    
    # Save
    output_path = os.path.join(output_dir, 'comprehensive_track_report.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Comprehensive track report saved: {output_path}")
    return output_path


def create_multi_dataset_summary(all_results: List[Dict[str, Any]], 
                                output_dir: str) -> str:
    """
    Create summary visualization across multiple datasets
    
    Args:
        all_results: List of results dictionaries from multiple TIFF files
        output_dir: Output directory for summary
    """
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Extract metrics from all datasets
    dataset_names = []
    detection_rates = []
    precisions = []
    recalls = []
    f1_scores = []
    avg_position_errors = []
    overall_accuracies = []
    
    for i, result in enumerate(all_results):
        dataset_names.append(result.get('dataset_name', f'Dataset {i+1}'))
        
        if 'metrics' in result['stage_results']:
            metrics = result['stage_results']['metrics']
            detection_rates.append(metrics.get('frame_detection_rate', 0))
            overall_accuracies.append(metrics.get('overall_frame_accuracy', 0))
            
            if metrics.get('track_metrics'):
                tm = metrics['track_metrics']
                precisions.append(tm.get('track_precision', 0))
                recalls.append(tm.get('track_recall', 0))
                f1_scores.append(tm.get('track_f1', 0))
            else:
                precisions.append(0)
                recalls.append(0)
                f1_scores.append(0)
            
            avg_position_errors.append(metrics.get('avg_position_error', 0))
    
    # 1. Detection Rate comparison
    ax1 = axes[0, 0]
    bars = ax1.bar(range(len(dataset_names)), detection_rates, 
                   color='#2E86AB', alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Detection Rate', fontsize=12)
    ax1.set_title('Detection Rate Across Datasets', fontsize=13, fontweight='bold')
    ax1.set_xticks(range(len(dataset_names)))
    ax1.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax1.set_ylim([0, 1.1])
    ax1.axhline(y=np.mean(detection_rates), color='red', linestyle='--', 
                label=f'Mean: {np.mean(detection_rates):.3f}')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, detection_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Precision comparison
    ax2 = axes[0, 1]
    bars = ax2.bar(range(len(dataset_names)), precisions, 
                   color='#A23B72', alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision Across Datasets', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(len(dataset_names)))
    ax2.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax2.set_ylim([0, 1.1])
    ax2.axhline(y=np.mean(precisions), color='red', linestyle='--',
                label=f'Mean: {np.mean(precisions):.3f}')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, precisions):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Recall comparison
    ax3 = axes[0, 2]
    bars = ax3.bar(range(len(dataset_names)), recalls, 
                   color='#F18F01', alpha=0.8, edgecolor='black')
    ax3.set_ylabel('Recall', fontsize=12)
    ax3.set_title('Recall Across Datasets', fontsize=13, fontweight='bold')
    ax3.set_xticks(range(len(dataset_names)))
    ax3.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax3.set_ylim([0, 1.1])
    ax3.axhline(y=np.mean(recalls), color='red', linestyle='--',
                label=f'Mean: {np.mean(recalls):.3f}')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, recalls):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 4. F1 Score comparison
    ax4 = axes[1, 0]
    bars = ax4.bar(range(len(dataset_names)), f1_scores, 
                   color='#06A77D', alpha=0.8, edgecolor='black')
    ax4.set_ylabel('F1 Score', fontsize=12)
    ax4.set_title('F1 Score Across Datasets', fontsize=13, fontweight='bold')
    ax4.set_xticks(range(len(dataset_names)))
    ax4.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax4.set_ylim([0, 1.1])
    ax4.axhline(y=np.mean(f1_scores), color='red', linestyle='--',
                label=f'Mean: {np.mean(f1_scores):.3f}')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, f1_scores):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 5. Position Error comparison
    ax5 = axes[1, 1]
    bars = ax5.bar(range(len(dataset_names)), avg_position_errors, 
                   color='#C73E1D', alpha=0.8, edgecolor='black')
    ax5.set_ylabel('Avg Position Error (px)', fontsize=12)
    ax5.set_title('Position Accuracy Across Datasets', fontsize=13, fontweight='bold')
    ax5.set_xticks(range(len(dataset_names)))
    ax5.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax5.axhline(y=np.mean(avg_position_errors), color='red', linestyle='--',
                label=f'Mean: {np.mean(avg_position_errors):.1f}px')
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, avg_position_errors):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 6. Overall performance radar/summary
    ax6 = axes[1, 2]
    
    # Calculate mean metrics
    mean_metrics = {
        'Detection Rate': np.mean(detection_rates),
        'Precision': np.mean(precisions),
        'Recall': np.mean(recalls),
        'F1 Score': np.mean(f1_scores)
    }
    
    # Bar chart of mean performance
    metrics = list(mean_metrics.keys())
    values = list(mean_metrics.values())
    colors_bar = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']
    
    bars = ax6.bar(metrics, values, color=colors_bar, alpha=0.8, edgecolor='black')
    ax6.set_ylabel('Score', fontsize=12)
    ax6.set_title('Mean Performance Across All Datasets', fontsize=13, fontweight='bold')
    ax6.set_ylim([0, 1.1])
    ax6.set_xticklabels(metrics, rotation=45, ha='right')
    ax6.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', 
                fontsize=10, fontweight='bold')
    
    # Add dataset count info
    info_text = f'Total Datasets: {len(all_results)}\n'
    info_text += f'Mean Detection Rate: {np.mean(detection_rates):.3f}\n'
    info_text += f'Mean F1 Score: {np.mean(f1_scores):.3f}\n'
    info_text += f'Mean Position Error: {np.mean(avg_position_errors):.1f}px'
    
    ax6.text(0.02, 0.02, info_text, transform=ax6.transAxes,
            verticalalignment='bottom', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.suptitle('Multi-Dataset Performance Summary', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'multi_dataset_summary.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Multi-dataset summary saved: {output_path}")
    return output_path


def create_combined_pr_curves(all_results: List[Dict[str, Any]], 
                              output_dir: str) -> str:
    """
    Create combined PR curves from all datasets
    
    Args:
        all_results: List of results from multiple datasets
        output_dir: Output directory
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
    
    all_aps = []
    all_aucs = []
    
    # Plot individual PR curves
    ax1 = axes[0]
    ax2 = axes[1]
    
    for i, result in enumerate(all_results):
        dataset_name = result.get('dataset_name', f'Dataset {i+1}')
        
        if 'pr_roc' in result['stage_results'] and result['stage_results']['pr_roc']:
            pr_roc_data = result['stage_results']['pr_roc']['pr_roc_data']
            
            # PR Curve
            ax1.plot(pr_roc_data['recall'], pr_roc_data['precision'],
                    color=colors[i], linewidth=2, alpha=0.7,
                    label=f'{dataset_name} (AP={pr_roc_data["avg_precision"]:.3f})')
            
            # ROC Curve
            ax2.plot(pr_roc_data['fpr'], pr_roc_data['tpr'],
                    color=colors[i], linewidth=2, alpha=0.7,
                    label=f'{dataset_name} (AUC={pr_roc_data["roc_auc"]:.3f})')
            
            all_aps.append(pr_roc_data['avg_precision'])
            all_aucs.append(pr_roc_data['roc_auc'])
    
    # Format PR curve
    ax1.set_xlabel('Recall', fontsize=12)
    ax1.set_ylabel('Precision', fontsize=12)
    ax1.set_title(f'Precision-Recall Curves\nmAP = {np.mean(all_aps):.3f}', 
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1.05])
    ax1.set_ylim([0, 1.05])
    
    # Format ROC curve
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
    ax2.set_xlabel('False Positive Rate', fontsize=12)
    ax2.set_ylabel('True Positive Rate', fontsize=12)
    ax2.set_title(f'ROC Curves\nMean AUC = {np.mean(all_aucs):.3f}', 
                 fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1.05])
    ax2.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'combined_pr_roc_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Combined PR/ROC curves saved: {output_path}")
    return output_path