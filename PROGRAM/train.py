import os
import sys
import torch
from ultralytics import YOLO
import pandas as pd
from pathlib import Path

# Configuration to reduce CUDA memory fragmentation
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

def get_input(message, default_value=None, expected_type=str):
    """
    Utility function to request user input with a default value.
    """
    if default_value is not None:
        prompt = f"{message} [Default: {default_value}]: "
    else:
        prompt = f"{message}: "

    while True:
        user_input = input(prompt).strip()

        if not user_input and default_value is not None:
            return default_value
        
        if not user_input and default_value is None:
            print("Error: This field is required.")
            continue

        try:
            return expected_type(user_input)
        except ValueError:
            print(f"Error: Please enter a valid value of type {expected_type.__name__}.")

def verify_file(path):
    """Checks if file exists."""
    if not os.path.isfile(path):
        print(f"Error: The file '{path}' not found.")
        return False
    return True

def export_statistics_to_excel(results_dir):
    """
    Parses per-class results file and exports to Excel.
    
    Args:
        results_dir: Path to training results folder
    """
    try:
        # Search for per-class results file
        results_dir = Path(results_dir)
        
        # Possible paths for results
        possible_paths = [
            results_dir / "weights" / "results class",
            results_dir / "results class",
            results_dir / "results_class.txt",
        ]
        
        results_file = None
        for path in possible_paths:
            if path.exists():
                results_file = path
                break
        
        if results_file is None:
            print("⚠️  Per-class results file not found. Excel export cancelled.")
            return
        
        print(f"\n📊 Reading statistics from : {results_file}")
        
        # Read the file
        with open(results_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Parse the data
        data = []
        header_found = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect header
            if 'Class' in line and 'Images' in line:
                header_found = True
                continue
            
            if not header_found:
                continue
            
            # Parse data lines
            # Format: Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
            parts = line.split()
            
            if len(parts) < 7:
                continue
            
            # Extract data
            # Class name can contain multiple words
            # The last 6 elements are always numeric metrics
            class_name = ' '.join(parts[:-6])
            
            try:
                images = int(parts[-6])
                instances = int(parts[-5])
                precision = float(parts[-4])
                recall = float(parts[-3])
                map50 = float(parts[-2])
                map50_95 = float(parts[-1])
                
                data.append({
                    'Class': class_name,
                    'Images': images,
                    'Instances': instances,
                    'Precision': precision,
                    'Recall': recall,
                    'mAP50': map50,
                    'mAP50-95': map50_95
                })
            except (ValueError, IndexError):
                # Ignore lines that cannot be parsed
                continue
        
        if not data:
            print("⚠️  No class data found in results file.")
            return
        
        # Create pandas DataFrame
        df = pd.DataFrame(data)
        
        # Sort by mAP50-95 descending
        df = df.sort_values('mAP50-95', ascending=False)
        
        # Create Excel file
        excel_path = results_dir / "class_statistics.xlsx"
        
        # Export to Excel with formatting
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Class Statistics', index=False)
            
            # Get worksheet for formatting
            worksheet = writer.sheets['Class Statistics']
            
            # Adjust column widths
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
            
            # Format numbers with 3 decimals
            from openpyxl.styles import numbers
            for row in worksheet.iter_rows(min_row=2, min_col=4, max_col=7):
                for cell in row:
                    cell.number_format = '0.000'
        
        print(f"✅ Statistics exported to Excel : {excel_path}")
        print(f"   📈 {len(df)} classes analyzed")
        
        # Display summary
        print("\n📊 Top 5 classes (par mAP50-95):")
        print(df.head(5)[['Class', 'mAP50-95', 'Precision', 'Recall']].to_string(index=False))
        
        print("\n📉 Bottom 5 classes (par mAP50-95):")
        print(df.tail(5)[['Class', 'mAP50-95', 'Precision', 'Recall']].to_string(index=False))
        
    except Exception as e:
        print(f"❌ Error during Excel export : {e}")
        import traceback
        traceback.print_exc()

def main():
    print("="*60)
    print("   YOLOv8 TRAINING ASSISTANT (FINE-TUNING)")
    print("="*60)

    # 1. GPU Verification
    device_name = 'cpu'
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU detected : {device_name}")
        device_arg = 0 # 0 correspond au premier GPU
    else:
        print("⚠️  Aucun GPU detected. L'entraînement se fera sur CPU (cela peut être lent).")
        device_arg = 'cpu'
    print("-" * 60)

    # 2. Base Model Selection
    print("\n--- Configuration du Model ---")
    print("Available models : yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium), etc.")
    model_name = get_input("Model name or path to weights", "yolov8n.pt")

    # 3. Chemin vers les données
    print("\n--- Data Configuration ---")
    while True:
        data_path = get_input("Path to dataset.yaml file")
        if verify_file(data_path):
            break
    
    # 4. Hyperparamètres de base
    print("\n--- Basic Hyperparameters ---")
    epochs = get_input("Number of epochs", 50, int)
    imgsz = get_input("Image size (pixels)", 640, int)
    batch_size = get_input("Batch size (-1 for auto)", 8, int)
    
    # Option for gradient accumulation (simulates larger batch)
    # use_accumulate = input("Use gradient accumulation? (o/n) [Default: n]: ").strip().lower()
    # accumulate = None
    # if use_accumulate == 'o':
    #     accumulate = get_input("Number of batches to accumulate", 2, int)
    
    # 5. Hyperparamètres d'augmentation de données
    print("\n--- Data Augmentation Hyperparameters ---")
    print("💡 Press Enter to use default values")
    print()
    
    # Demander si l'utilisateur veut configurer les augmentations
    config_augmentation = input("Do you want to configure data augmentations? (o/n) [Default: n]: ").strip().lower()
    
    augmentation_params = {}
    
    if config_augmentation == 'o':
        print("\n📊 HSV augmentation configuration (Hue, Saturation, Value):")
        augmentation_params['hsv_h'] = get_input("  hsv_h (Hue variation, 0.0-1.0)", 0.015, float)
        augmentation_params['hsv_s'] = get_input("  hsv_s (Saturation variation, 0.0-1.0)", 0.7, float)
        augmentation_params['hsv_v'] = get_input("  hsv_v (Brightness variation, 0.0-1.0)", 0.4, float)
        
        print("\n🔄 Geometric transformations configuration:")
        augmentation_params['degrees'] = get_input("  degrees (Rotation in degrees, 0.0-180.0)", 0.0, float)
        augmentation_params['translate'] = get_input("  translate (Translation, 0.0-1.0)", 0.1, float)
        augmentation_params['scale'] = get_input("  scale (Scale, 0.0-1.0)", 0.5, float)
        augmentation_params['fliplr'] = get_input("  fliplr (Horizontal flip probability, 0.0-1.0)", 0.5, float)
        augmentation_params['flipud'] = get_input("  flipud (Vertical flip probability, 0.0-1.0)", 0.0, float)
        
        print("\n🎨 Mosaic and Mixup configuration:")
        augmentation_params['mosaic'] = get_input("  mosaic (Mosaic probability, 0.0-1.0)", 1.0, float)
        augmentation_params['mixup'] = get_input("  mixup (Mixup probability, 0.0-1.0)", 0.0, float)
        
        print("\n⏱️  Training configuration:")
        augmentation_params['close_mosaic'] = get_input("  close_mosaic (Disable mosaic after N epochs)", 10, int)
        augmentation_params['patience'] = get_input("  patience (Early stopping patience)", 50, int)
        
        print(f"\n✅ {len(augmentation_params)} augmentation hyperparameters configured")
    else:
        print("ℹ️  Using YOLO default values for augmentations")
    
    # 6. Configuration du projet
    print("\n--- Project Configuration ---")
    project_name = get_input("Project folder name (to save results)", "runs/detect")
    exp_name = get_input("Experiment name", "mon_entrainement")

    # 7. Confirmation and Launch
    print("\n" + "="*60)
    print("CONFIGURATION SUMMARY :")
    print(f"Model      : {model_name}")
    print(f"Data     : {data_path}")
    print(f"Epochs     : {epochs}")
    print(f"Image Size  : {imgsz}")
    print(f"Batch Size  : {batch_size}")
    print(f"Device      : {device_name}")
    
    if augmentation_params:
        print("\nCustom augmentations :")
        for key, value in augmentation_params.items():
            print(f"  {key:<15} : {value}")
    
    print("="*60)

    confirm = input("Do you want to start training? (o/n) : ").lower()
    if confirm != 'o':
        print("Operation cancelled.")
        sys.exit()

    print("\n🚀 Launching training...\n")

    try:
        # Loading model
        model = YOLO(model_name)

        # Prepare training arguments
        train_args = {
            'data': data_path,
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch_size,
            'device': device_arg,
            'project': project_name,
            'name': exp_name,
            'exist_ok': True,  # Overwrite if folder already exists
            'plots': True,
            'workers': 4,  # Reduce number of workers to save memory
            'amp': True  # Use Automatic Mixed Precision to save memory
        }
        
        # Add gradient accumulation if configured
        # if accumulate is not None:
        #     train_args['accumulate'] = accumulate
        
        # Add augmentation parameters if configured
        if augmentation_params:
            train_args.update(augmentation_params)
        
        # Launching training
        results = model.train(**train_args)

        # Path to results folder
        results_dir = Path(project_name) / exp_name

        print("\n" + "="*60)
        print("✅ TRAINING COMPLETED SUCCESSFULLY !")
        print(f"Final weights are saved in : {results_dir / 'weights'}")
        print("="*60)
        
        # Export des statistiques vers Excel
        print("\n📊 Exporting per-class statistics to Excel...")
        export_statistics_to_excel(results_dir)

    except Exception as e:
        print(f"\n❌ An error occurred during training : {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
