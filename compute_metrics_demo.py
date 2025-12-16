import os
from pathlib import Path
from src.ev_tracker import EVTracker

DATASET_LIST = [
   
    (
        r"data\tiff\xslot_HCC1954_02_1500uLhr_z40um_mov_2_MMStack_Pos0.ome.tif",
        r"data\csv\Infocus_xslot_HCC1954_02_1500uLhr_z40um_mov_2.csv"
    ),
    (
        r"data\tiff\xslot_BT747_01_1500uLhr_z35um_mov_flush_adj_8_MMStack_Pos0.ome.tif",
        r"data\csv\xslot_BT747_01_1500uLhr_z35um_mov_flush_adj_8_MMStack_Pos0.ome.csv"
    ),
    # is this right? (below)
    (
        r"data\tiff\xslot_BT747_00_1500uLhr_z40um_mov_flush_adj_9_MMStack_Pos0.ome.tif",
        r"data\csv\xslot_BT747_01_1500uLhr_z40um_mov_flush_adj_9_MMStack_Pos0.ome.csv"
    ),
    (
        r"data\tiff\xslot_BT747_03_1000uLhr_z35um_adjSP_mov_2_MMStack_Pos0.ome.tif",
        r"data\csv\xslot_BT747_03_1000uLhr_z35um_adjSP_mov_2.csv"
    ),
    (
        r"data\tiff\xslot_HCC1954_01_500uLhr_z35um_mov_1_MMStack_Pos0.ome.tif",
        r"data\csv\xslot_HCC1954_01_500uLhr_z35um_mov_1.csv"
    ),
    (
        r"data\tiff\xslot_HCC1954_01_500uLhr_z40um_mov_1_MMStack_Pos0.ome.tif",
        r"data\csv\xslot_HCC1954_01_500uLhr_z40um_mov_1.csv"
    ),
    (
        r"data\tiff\xslot_HCC1954_PT03_xp4_1500uLhr_z35um_mov_adjSP_9_MMStack_Pos0.ome.tif",
        r"data\csv\xslot_HCC1954_PT03_xp4_SP9.csv"
    ),
    (
        r"data\tiff\new\xslot_BT747_PT00_xp1_1500uLhr_z40um_mov_6_flush_adj_MMStack_Pos0.ome.tif",
        r"data\csv\new\InfocusEVs_xslot_BT747_PT00_xp1_1500uLhr_z40um_mov_6_flush_adj_MMStack_Pos0.ome.csv"
    ),
    (
        r"data\tiff\new\xslot_HCC1954_PT00_xp1_750uLhr_z35um_mov_1_MMStack_Pos0.ome.tif",
        r"data\csv\new\InfocusEVs_xslot_HCC1954_PT00_xp1_750uLhr_z35um_mov_1.csv"
    ),
    (
        r"data\tiff\new\xslot_HCC1954_PT03_xp4_1250uLhr_z40um_mov_1_MMStack_Pos0.ome.tif",
        r"data\csv\new\InfocusEVs_xslot_HCC1954_PT03_xp4_1250uLhr_z40um_mov_1_MMStack_Pos0.ome.csv"
    ),
    (
        r"data\tiff\new\xslot_BT747_PT03_xp4_750uLhr_z35um_mov_2_MMStack_Pos0.ome.tif",
        r"data\csv\new\xslot_BT747_PT03_xp4_750uLhr_z35um_mov_2.csv"
    ),
]

def validate_dataset_list(dataset_list):

    valid_datasets = []
    
    for tiff_path, csv_path in dataset_list:
        tiff_exists = os.path.exists(tiff_path)
        csv_exists = os.path.exists(csv_path)
        
        if tiff_exists and csv_exists:
            valid_datasets.append((tiff_path, csv_path))  
    
    return valid_datasets

def run_batch_metrics_analysis(dataset_list, parameters=None):

    if not dataset_list:
        print("No datasets found to process")
        return None
    

    tracker = EVTracker()
    
    if parameters:
        tracker.set_params(**parameters)
    else:
        tracker.set_params(
            threshold=0.55,
            min_distance=30,
            filter_radius=10,
            filter_size=41,
            filter_sigma=2.0,
            bg_window_size=15,
            blur_kernel_size=7,
            clahe_clip_limit=2.0,
            clahe_grid_size=(8, 8),
            max_distance=25,
            min_track_length=5,
            max_frame_gap=3,
            distance_threshold=30.0
        )
    
    tracker.print_params()
    
    # Run batch analysis
    print(f"\nStarting batch analysis ({len(dataset_list)} files)...")
    print("Note: Using threshold=0.1 for comprehensive PR curves\n")
    
    results = tracker.run_batch(dataset_list)
    
    return results

def main():

    print("\nEV TRACKER - BATCH METRICS COMPUTATION")
    print("="*70)
    
    # Validate dataset list
    if not DATASET_LIST:
        return
    
    valid_datasets = validate_dataset_list(DATASET_LIST)
    
    if not valid_datasets:
        return


    response = input("\nProceed with batch analysis? (y/n): ").lower().strip()
    
    if response != 'y':
        print("Cancelled by user.")
        return
    
    # Run batch analysis
    custom_params = {
        'threshold': 0.55,
        'min_distance': 30,
        'max_distance': 25,
        'min_track_length': 5,
        'distance_threshold': 30.0
    }
    
    results = run_batch_metrics_analysis(valid_datasets, parameters=custom_params)

    if results:
        print("\nBatch analysis complete.")
        print(f"View detailed results in: {results['output_dir']}")


if __name__ == "__main__":
    main()