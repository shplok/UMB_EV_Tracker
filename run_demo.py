from src.ev_tracker import EVTracker

tracker = EVTracker()
tracker.set_params(
    threshold=0.55, min_distance=10, filter_radius=10,
    bg_window_size=15, blur_kernel_size=7,
    max_distance=35, min_track_length=5
)

tracker.print_params()

results = tracker.run(
    tiff_file=r"c:\Users\sawye\OneDrive - University of Massachusetts Boston\Desktop\xslot_HCC1954_01_500uLhr_z40um_mov_1_MMStack_Pos0.ome.tif",
    ground_truth_csv=r"c:\Users\sawye\OneDrive - University of Massachusetts Boston\Desktop\xslot_HCC1954_01_500uLhr_z40um_mov_1.csv"
)

if results['success']:
    print(f"Average Precision: {results['global_ap']:.3f}")
    print(f"ROC AUC: {results['global_auc']:.3f}")
    print(f"Output saved to: {results['output_dir']}")
