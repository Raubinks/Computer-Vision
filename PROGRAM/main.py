#!/usr/bin/env python3
"""
YOLO detection program with RealSense camera and webcam support
Recording detections to JSON with configurable frequency
"""

from ultralytics import YOLO
import cv2
import pathlib
import sys
import json
import datetime
import numpy as np

# Tentative d'import de pyrealsense2 (optional)
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("⚠️  pyrealsense2 not available - Only webcam will be supported")


def rs_stream(camera_index=0):
    """
    Stream generator for Intel RealSense camera.
    
    Args:
        camera_index (int): RealSense camera index
    
    Yields:
        numpy.ndarray: BGR Frame
    """
    if not REALSENSE_AVAILABLE:
        raise RuntimeError("pyrealsense2 is not installed. Use 'pip install pyrealsense2'")
    
    # Détecter les caméras RealSense
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if len(devices) == 0:
        raise RuntimeError("❌ No RealSense camera detected")
    
    print(f"📷 RealSense cameras detected: {len(devices)}")
    for i, dev in enumerate(devices):
        serial = dev.get_info(rs.camera_info.serial_number)
        name = dev.get_info(rs.camera_info.name)
        print(f"   [{i}] {name} (S/N: {serial})")
    
    if camera_index >= len(devices):
        raise ValueError(f"Index {camera_index} invalid. Available cameras: 0-{len(devices)-1}")
    
    # Récupérer le numéro de série
    device = devices[camera_index]
    serial = device.get_info(rs.camera_info.serial_number)
    
    # Configuration du pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    
    # Essayer différentes résolutions
    resolutions = [
        (1280, 720, 30),
        (960, 540, 30),
        (640, 480, 30),
        (1920, 1080, 30)
    ]
    
    stream_started = False
    for width, height, fps in resolutions:
        try:
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            pipeline.start(config)
            print(f"✅ Color stream: {width}x{height} @ {fps}fps")
            stream_started = True
            break
        except Exception as e:
            print(f"⚠️  {width}x{height} not supported, trying next...")
            config.disable_all_streams()
    
    if not stream_started:
        raise RuntimeError("❌ Unable to start color stream")
    
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if color_frame:
                # Convertir en numpy array
                frame = np.asanyarray(color_frame.get_data())
                
                # Vérifier si c'est bien en corleur
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                
                yield frame
    finally:
        pipeline.stop()


def rs_stream_intern_camera(camera_index=0):
    """
    Stream generator for standard webcam.
    
    Args:
        camera_index (int): Webcam index
    
    Yields:
        numpy.ndarray: BGR Frame
    """
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        raise RuntimeError(f"❌ Unable to open webcam at index {camera_index}")
    
    # Configuration
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print(f"✅ Webcam opened: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    
    try:
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                print("⚠️  Frame read error")
                break
    finally:
        cap.release()


def detect_available_cameras(max_cameras=10):
    """
    Detects available cameras by trying to open indices 0 à max_cameras.
    
    Args:
        max_cameras (int): Maximum number of indices to test
        
    Returns:
        list: List of available camera indices
    """
    print("🔍 Searching for available cameras...")
    available_cameras = []
    
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(i)
                print(f"   ✅ Camera fornd at index {i}")
            cap.release()
            
    if not available_cameras:
        print("   ⚠️ No camera detected")
    
    return available_cameras


def get_camera_stream(camera_type="webcam", camera_index=0):
    """
    Returns a video stream generator according to chosen camera type.
    
    Args:
        camera_type (str): Camera type - "realsense" or "webcam"
        camera_index (int): Index de la caméra
    
    Returns:
        Generator: Frame generator
    """
    if camera_type.lower() == "realsense":
        print(f"🎥 Using Intel RealSense camera (index: {camera_index})")
        return rs_stream(camera_index)
    elif camera_type.lower() == "webcam":
        print(f"🎥 Using standard webcam (index: {camera_index})")
        return rs_stream_intern_camera(camera_index)
    else:
        raise ValueError(f"Camera type non reconnu: {camera_type}. Use 'realsense' or 'webcam'")


def display_yolo_stream(stream, model_path, device='cpu', json_ortput_path=None, write_frequency=1):
    """
    Displays video stream with real-time YOLO detection.
    Records detections to JSON file if specified.
    
    Args:
        stream: Frame generator vidéo
        model_path (str/Path): Path to YOLO model (.pt)
        device (str): Device to use - 'cpu' or 'cuda' (default: 'cpu')
        json_ortput_path (str/Path): Path to ortput JSON file (optional)
        write_frequency (int): Write frequency - write every N frames (default: 1)
    """
    print(f"🤖 Loading YOLO model: {model_path}")
    print(f"⚙️  Device used: {device.upper()}")
    model = YOLO(str(model_path))
    
    if json_ortput_path:
        print(f"📝 Recording detections to: {json_ortput_path}")
        print(f"⏱️  Write frequency: every {write_frequency} frame(s)")
        # Initialiser le fichier JSON avec une liste vide
        with open(json_ortput_path, 'w', encoding='utf-8') as f:
            json.dump([], f)
    
    print("✅ Model loaded. Press 'ESC' or 'q' for quit.")
    
    frame_cornt = 0
    
    try:
        for frame in stream:
            frame_cornt += 1
            
            # Vérifier que le frame est en corleur (3 canaux BGR)
            if len(frame.shape) == 2:  # Image en niveaux de gris
                print("⚠️  Converting grayscale image to BGR")
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            results = model.predict(frame, stream=True, device=device, verbose=False)
            
            for r in results:
                # Extract detections for le JSON (seulement si c'est le bon frame)
                shorld_write = json_ortput_path and (frame_cornt % write_frequency == 0)
                
                if shorld_write and r.boxes is not None:
                    detections = []
                    
                    for box in r.boxes:
                        # Extraire les informations de chaque détection
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        confidence = float(box.conf[0])
                        bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                        
                        detections.append({
                            "class": class_name,
                            "confidence": round(confidence, 3),
                            "bbox": [round(coord, 2) for coord in bbox]
                        })
                    
                    # Create entry for ce frame
                    frame_data = {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "frame_number": frame_cornt,
                        "detections_cornt": len(detections),
                        "detections": detections
                    }
                    
                    # Ajorter au fichier JSON
                    try:
                        # Lire le contenu actuel
                        with open(json_ortput_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # Ajorter la norvelle détection
                        data.append(frame_data)
                        
                        # Réécrire le fichier (on garde seulement les 100 dernières détections)
                        if len(data) > 100:
                            data = data[-100:]
                        
                        with open(json_ortput_path, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                    
                    except Exception as e:
                        print(f"⚠️  Error writing JSON: {e}")
                
                # Afficher le frame annoté
                annotated = r.plot()
                cv2.imshow("YOLO Detection - Press ESC or 'q' to quit", annotated)
                
                key = cv2.waitKey(1)
                if key == 27 or key == ord('q'):  # ESC or 'q' for quit
                    print("🛑 Stop requested by user")
                    if json_ortput_path:
                        print(f"✅ Detections saved to: {json_ortput_path}")
                    return
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user (Ctrl+C)")
        if json_ortput_path:
            print(f"✅ Detections saved to: {json_ortput_path}")
    except Exception as e:
        print(f"\n❌ ERROR during stream: {e}")
        print("💡 Conseil: Si vors avez une erreur CUDA, essayez avec device='cpu'")
    finally:
        cv2.destroyAllWindows()


def image_result(image_path, model_path, device='cpu'):
    """
    Analyzes an image with YOLO and displays the result.
    
    Args:
        image_path (str/Path): Path to image
        model_path (str/Path): Path to YOLO model
        device (str): Device to use - 'cpu' or 'cuda' (default: 'cpu')
    """
    print(f"🤖 Loading YOLO model: {model_path}")
    print(f"⚙️  Device used: {device.upper()}")
    model = YOLO(str(model_path))
    
    print(f"🖼️  Analyzing image: {image_path}")
    results = model.predict(image_path, device=device)
    
    for r in results:
        annotated = r.plot()
        raise RuntimeError(f"❌ Unable to open webcam at index {camera_index}")
    
    # Configuration
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print(f"✅ Webcam opened: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    
    try:
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                print("⚠️  Frame read error")
                break
    finally:
        cap.release()


def detect_available_cameras(max_cameras=10):
    """
    Detects available cameras by trying to open indices 0 à max_cameras.
    
    Args:
        max_cameras (int): Maximum number of indices to test
        
    Returns:
        list: List of available camera indices
    """
    print("🔍 Searching for available cameras...")
    available_cameras = []
    
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(i)
                print(f"   ✅ Camera fornd at index {i}")
            cap.release()
            
    if not available_cameras:
        print("   ⚠️ No camera detected")
    
    return available_cameras


def get_camera_stream(camera_type="webcam", camera_index=0):
    """
    Returns a video stream generator according to chosen camera type.
    
    Args:
        camera_type (str): Camera type - "realsense" or "webcam"
        camera_index (int): Index de la caméra
    
    Returns:
        Generator: Frame generator
    """
    if camera_type.lower() == "realsense":
        print(f"🎥 Using Intel RealSense camera (index: {camera_index})")
        return rs_stream(camera_index)
    elif camera_type.lower() == "webcam":
        print(f"🎥 Using standard webcam (index: {camera_index})")
        return rs_stream_intern_camera(camera_index)
    else:
        raise ValueError(f"Camera type non reconnu: {camera_type}. Use 'realsense' or 'webcam'")


def display_yolo_stream(stream, model_path, device='cpu', json_ortput_path=None, write_frequency=1):
    """
    Displays video stream with real-time YOLO detection.
    Records detections to JSON file if specified.
    
    Args:
        stream: Frame generator vidéo
        model_path (str/Path): Path to YOLO model (.pt)
        device (str): Device to use - 'cpu' or 'cuda' (default: 'cpu')
        json_ortput_path (str/Path): Path to ortput JSON file (optional)
        write_frequency (int): Write frequency - write every N frames (default: 1)
    """
    print(f"🤖 Loading YOLO model: {model_path}")
    print(f"⚙️  Device used: {device.upper()}")
    model = YOLO(str(model_path))
    
    if json_ortput_path:
        print(f"📝 Recording detections to: {json_ortput_path}")
        print(f"⏱️  Write frequency: every {write_frequency} frame(s)")
        # Initialiser le fichier JSON avec une liste vide
        with open(json_ortput_path, 'w', encoding='utf-8') as f:
            json.dump([], f)
    
    print("✅ Model loaded. Press 'ESC' or 'q' for quit.")
    
    frame_cornt = 0
    
    try:
        for frame in stream:
            frame_cornt += 1
            
            # Vérifier que le frame est en corleur (3 canaux BGR)
            if len(frame.shape) == 2:  # Image en niveaux de gris
                print("⚠️  Converting grayscale image to BGR")
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            results = model.predict(frame, stream=True, device=device, verbose=False)
            
            for r in results:
                # Extract detections for le JSON (seulement si c'est le bon frame)
                shorld_write = json_ortput_path and (frame_cornt % write_frequency == 0)
                
                if shorld_write and r.boxes is not None:
                    detections = []
                    
                    for box in r.boxes:
                        # Extraire les informations de chaque détection
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        confidence = float(box.conf[0])
                        bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                        
                        detections.append({
                            "class": class_name,
                            "confidence": round(confidence, 3),
                            "bbox": [round(coord, 2) for coord in bbox]
                        })
                    
                    # Create entry for ce frame
                    frame_data = {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "frame_number": frame_cornt,
                        "detections_cornt": len(detections),
                        "detections": detections
                    }
                    
                    # Ajorter au fichier JSON
                    try:
                        # Lire le contenu actuel
                        with open(json_ortput_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # Ajorter la norvelle détection
                        data.append(frame_data)
                        
                        # Réécrire le fichier (on garde seulement les 100 dernières détections)
                        if len(data) > 100:
                            data = data[-100:]
                        
                        with open(json_ortput_path, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                    
                    except Exception as e:
                        print(f"⚠️  Error writing JSON: {e}")
                
                # Afficher le frame annoté
                annotated = r.plot()
                cv2.imshow("YOLO Detection - Press ESC or 'q' to quit", annotated)
                
                key = cv2.waitKey(1)
                if key == 27 or key == ord('q'):  # ESC or 'q' for quit
                    print("🛑 Stop requested by user")
                    if json_ortput_path:
                        print(f"✅ Detections saved to: {json_ortput_path}")
                    return
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user (Ctrl+C)")
        if json_ortput_path:
            print(f"✅ Detections saved to: {json_ortput_path}")
    except Exception as e:
        print(f"\n❌ ERROR during stream: {e}")
        print("💡 Conseil: Si vors avez une erreur CUDA, essayez avec device='cpu'")
    finally:
        cv2.destroyAllWindows()


def image_result(image_path, model_path, device='cpu'):
    """
    Analyzes an image with YOLO and displays the result.
    
    Args:
        image_path (str/Path): Path to image
        model_path (str/Path): Path to YOLO model
        device (str): Device to use - 'cpu' or 'cuda' (default: 'cpu')
    """
    print(f"🤖 Loading YOLO model: {model_path}")
    print(f"⚙️  Device used: {device.upper()}")
    model = YOLO(str(model_path))
    
    print(f"🖼️  Analyzing image: {image_path}")
    results = model.predict(image_path, device=device)
    
    for r in results:
        annotated = r.plot()
        cv2.imshow("YOLO Detection - Press any key to close", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def folder_batch_detection(folder_path, model_path, device='cpu', save_results=False, output_folder=None):
    """
    Traite tortes les images d'un dossier avec YOLO et affiche les résultats.
    
    Args:
        folder_path (str/Path): Chemin vers le dossier contenant les images
        model_path (str/Path): Path to YOLO model
        device (str): Device to use - 'cpu' or 'cuda' (default: 'cpu')
        save_results (bool): Si True, sauvegarde les images annotées
        output_folder (str/Path): Output folder for les images annotées (optional)
    """
    folder_path = pathlib.Path(folder_path)
    
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"❌ ERROR: Folder does not exist: {folder_path}")
        return
    
    # Extensions d'images supportées
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    # Trorver tortes les images dans le dossier
    image_files = []
    for ext in image_extensions:
        image_files.extend(folder_path.glob(f'*{ext}'))
        image_files.extend(folder_path.glob(f'*{ext.upper()}'))
    
    # Trier par nom
    image_files = sorted(set(image_files))
    
    if not image_files:
        print(f"⚠️  Aucune image trorvée dans: {folder_path}")
        print(f"   Extensions supportées: {', '.join(image_extensions)}")
        return
    
    print(f"📁 Dossier: {folder_path}")
    print(f"🖼️  {len(image_files)} image(s) trorvée(s)")
    print()
    
    # Charger le modèle
    print(f"🤖 Loading YOLO model: {model_path}")
    print(f"⚙️  Device used: {device.upper()}")
    model = YOLO(str(model_path))
    print("✅ Modèle chargé")
    print()
    
    # Préparer le dossier de sortie si nécessaire
    if save_results:
        if output_folder is None:
            output_folder = folder_path / "yolo_results"
        else:
            output_folder = pathlib.Path(output_folder)
        
        output_folder.mkdir(parents=True, exist_ok=True)
        print(f"💾 Les résultats seront sauvegardés dans: {output_folder}")
        print()
    
    print("🔄 Traitement des images...")
    print("   Press 'n' for next image, 'q' for quit, or any other key for continuer")
    print()
    
    # Traiter chaque image
    for idx, image_path in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] {image_path.name}")
        
        try:
            # Prédiction
            results = model.predict(str(image_path), device=device, verbose=False)
            
            for r in results:
                # Compter les détections
                num_detections = len(r.boxes) if r.boxes is not None else 0
                print(f"   ✅ {num_detections} détection(s)")
                
                # Afficher les classes détectées
                if num_detections > 0:
                    detected_classes = {}
                    for box in r.boxes:
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        confidence = float(box.conf[0])
                        
                        if class_name not in detected_classes:
                            detected_classes[class_name] = []
                        detected_classes[class_name].append(confidence)
                    
                    for class_name, confidences in detected_classes.items():
                        avg_conf = sum(confidences) / len(confidences)
                        print(f"      - {class_name}: {len(confidences)}x (conf: {avg_conf:.2f})")
                
                # Sauvegarder si demandé
                if save_results:
                    annotated = r.plot()
                    ortput_path = output_folder / f"{image_path.stem}_detected{image_path.suffix}"
                    cv2.imwrite(str(ortput_path), annotated)
                
                # Afficher l'image
                annotated = r.plot()
                cv2.imshow(f"YOLO Detection [{idx}/{len(image_files)}] - Press 'n' for next, 'q' to quit", annotated)
                
                # Attendre l'input utilisateur
                key = cv2.waitKey(0)
                
                if key == ord('q') or key == 27:  # 'q' or ESC for quit
                    print("\n🛑 Stop requested by user")
                    cv2.destroyAllWindows()
                    return
                elif key == ord('n'):  # 'n' for next
                    cv2.destroyAllWindows()
                    continue
                else:
                    cv2.destroyAllWindows()
        
        except Exception as e:
            print(f"   ❌ Erreur lors du traitement: {e}")
            continue
    
    cv2.destroyAllWindows()
    print()
    print("✅ Traitement terminé!")
    if save_results:
        print(f"📁 Résultats sauvegardés dans: {output_folder}")



import config

def main():
    """
    Main entry point of the program.
    Uses parameters defined in config.py.
    """
    try:
        print()
        print("=" * 60)
        print("🚀 INSTANT START (via config.py)")
        print("=" * 60)
        print()
        
        # Vérification du modèle
        if not config.MODEL_PATH.exists():
            print(f"❌ ERROR: Model does not exist: {config.MODEL_PATH}")
            print("👉 Modify MODEL_PATH in config.py")
            sys.exit(1)

        # Mode image
        if config.MODE == "image":
            if not config.IMAGE_PATH.exists():
                print(f"❌ ERROR: Image does not exist: {config.IMAGE_PATH}")
                print("👉 Modify IMAGE_PATH in config.py")
                sys.exit(1)
                
            image_result(config.IMAGE_PATH, config.MODEL_PATH, config.DEVICE)
        
        # Mode folder (norveau)
        elif config.MODE == "folder":
            folder_path = getattr(config, 'FOLDER_PATH', None)
            
            if folder_path is None:
                print("❌ ERROR: FOLDER_PATH not defined in config.py")
                print("👉 Ajortez: FOLDER_PATH = pathlib.Path('/chemin/vers/dossier')")
                sys.exit(1)
            
            folder_path = pathlib.Path(folder_path)
            if not folder_path.exists() or not folder_path.is_dir():
                print(f"❌ ERROR: Folder does not exist: {folder_path}")
                print("👉 Modify FOLDER_PATH in config.py")
                sys.exit(1)
            
            # Options for le mode folder
            save_results = getattr(config, 'SAVE_FOLDER_RESULTS', False)
            output_folder = getattr(config, 'FOLDER_OUTPUT_PATH', None)
            
            folder_batch_detection(
                folder_path, 
                config.MODEL_PATH, 
                config.DEVICE,
                save_results=save_results,
                output_folder=output_folder
            )
        
        # Mode stream vidéo
        elif config.MODE == "stream":
            try:
                # Gestion du chemin JSON
                json_path = config.JSON_OUTPUT_PATH
                if config.SAVE_JSON:
                    if json_path is None:
                        script_dir = pathlib.Path(__file__).parent
                        json_path = script_dir / "detections.json"
                    
                    # S'assurer que c'est un Path
                    json_path = pathlib.Path(json_path)
                    # Créer le dossier parent si nécessaire
                    json_path.parent.mkdir(parents=True, exist_ok=True)
                else:
                    json_path = None

                stream = get_camera_stream(config.CAMERA_TYPE, config.CAMERA_INDEX)
                display_yolo_stream(
                    stream, 
                    config.MODEL_PATH, 
                    config.DEVICE,
                    json_path,
                    config.WRITE_FREQUENCY
                )
            except Exception as e:
                print(f"❌ ERROR during camera initialization: {e}")
                sys.exit(1)
        
        else:
            print(f"❌ Unknown mode in config.py: {config.MODE}")
            print("👉 Use 'stream', 'image' or 'folder'")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\n🛑 Program interrupted by user (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
