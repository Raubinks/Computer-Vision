#!/usr/bin/env python3
"""
Script de configuration interactive pour YOLO Detection.
Génère le fichier config.py.
"""

import pathlib
import cv2
import sys

def detect_available_cameras(max_cameras=10):
    """Détecte les caméras disponibles."""
    print("🔍 Recherche des caméras disponibles...")
    available_cameras = []
    
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(i)
                print(f"   ✅ Caméra trouvée à l'index {i}")
            cap.release()
            
    if not available_cameras:
        print("   ⚠️ Aucune caméra détectée")
    
    return available_cameras

def get_user_input():
    """Demande les paramètres à l'utilisateur."""
    print("=" * 60)
    print("🛠️  CONFIGURATION YOLO DETECTION")
    print("=" * 60)
    print()
    
    # Choix du mode
    print("📋 MODE D'UTILISATION:")
    print("  1. Stream vidéo en temps réel")
    print("  2. Analyse d'une image")
    print("  3. Traitement d'un dossier d'images")
    print()
    
    while True:
        mode_choice = input("Choisissez le mode (1, 2 ou 3): ").strip()
        if mode_choice in ["1", "2", "3"]:
            break
        print("❌ Choix invalide.")
    
    if mode_choice == "1":
        mode = "stream"
    elif mode_choice == "2":
        mode = "image"
    else:
        mode = "folder"
    print()
    
    # Modèle
    print("🤖 MODÈLE YOLO:")
    print("  Entrez le chemin vers votre modèle YOLO (.pt)")
    
    while True:
        model_path_str = input("Chemin du modèle: ").strip()
        # Enlever les guillemets si l'utilisateur en a mis
        model_path_str = model_path_str.strip("'\"")
        model_path = pathlib.Path(model_path_str)
        
        if model_path.exists():
            print(f"✅ Modèle trouvé: {model_path}")
            break
        else:
            print(f"❌ Le fichier '{model_path}' n'existe pas.")
            retry = input("Voulez-vous réessayer ? (o/n): ").lower()
            if retry != 'o':
                print("⚠️  Attention: Chemin enregistré tel quel.")
                break
    
    print()
    
    # Device
    print("⚙️  DEVICE DE CALCUL:")
    print("  1. CPU")
    print("  2. GPU (CUDA)")
    
    device_choice = input("Choix (1/2) [défaut: 1]: ").strip()
    device = "cuda" if device_choice == "2" else "cpu"
    print()
    
    # Paramètres spécifiques au mode
    image_path = "test.jpg"
    folder_path = "/home/adrolab/Desktop/ROBIN/IA/test_images"
    save_folder_results = True
    folder_output_path = None
    camera_type = "webcam"
    camera_index = 0
    save_json = True
    json_path = "detections.json"
    write_freq = 5
    
    if mode == "image":
        print("🖼️  IMAGE À ANALYSER:")
        while True:
            img_str = input("Chemin de l'image: ").strip().strip("'\"")
            img_path = pathlib.Path(img_str)
            if img_path.exists():
                image_path = str(img_path)
                break
            print("❌ Fichier introuvable.")
            if input("Forcer ce chemin ? (o/n): ").lower() == 'o':
                image_path = str(img_path)
                break
    
    elif mode == "folder":
        print("📁 DOSSIER D'IMAGES:")
        while True:
            folder_str = input("Chemin du dossier: ").strip().strip("'\"")
            folder_p = pathlib.Path(folder_str)
            if folder_p.exists() and folder_p.is_dir():
                folder_path = str(folder_p)
                print(f"✅ Dossier trouvé: {folder_path}")
                break
            print("❌ Dossier introuvable.")
            if input("Forcer ce chemin ? (o/n): ").lower() == 'o':
                folder_path = str(folder_p)
                break
        
        print()
        print("💾 SAUVEGARDE DES RÉSULTATS:")
        save_input = input("Sauvegarder les images annotées ? (o/n) [défaut: o]: ").lower()
        save_folder_results = (save_input == "" or save_input == "o")
        
        if save_folder_results:
            output_input = input("Dossier de sortie [défaut: auto]: ").strip().strip("'\"")
            if output_input:
                folder_output_path = output_input
    
    else:
        # Stream
        print("🎥 TYPE DE CAMÉRA:")
        print("  1. Intel RealSense")
        print("  2. Webcam standard")
        
        cam_choice = input("Choix (1/2) [défaut: 2]: ").strip()
        camera_type = "realsense" if cam_choice == "1" else "webcam"
        
        print()
        print("📹 INDEX DE LA CAMÉRA:")
        if camera_type == "webcam":
            available = detect_available_cameras()
            if available:
                print(f"  Caméras détectées: {available}")
                default_cam = available[0]
            else:
                default_cam = 0
        else:
            default_cam = 0
            
        idx_input = input(f"Index [défaut: {default_cam}]: ").strip()
        try:
            camera_index = int(idx_input) if idx_input else default_cam
        except ValueError:
            camera_index = 0
            
        print()
        print("📝 ENREGISTREMENT JSON:")
        save_input = input("Activer ? (o/n) [défaut: o]: ").lower()
        save_json = (save_input == "" or save_input == "o")
        
        if save_json:
            json_input = input("Chemin JSON [défaut: detections.json]: ").strip().strip("'\"")
            if json_input:
                json_path = json_input
            
            freq_input = input("Fréquence d'écriture (frames) [défaut: 5]: ").strip()
            try:
                write_freq = int(freq_input) if freq_input else 5
            except ValueError:
                write_freq = 5

    return {
        "MODE": mode,
        "MODEL_PATH": str(model_path),
        "DEVICE": device,
        "IMAGE_PATH": str(image_path),
        "FOLDER_PATH": str(folder_path),
        "SAVE_FOLDER_RESULTS": save_folder_results,
        "FOLDER_OUTPUT_PATH": str(folder_output_path) if folder_output_path else None,
        "CAMERA_TYPE": camera_type,
        "CAMERA_INDEX": camera_index,
        "SAVE_JSON": save_json,
        "JSON_OUTPUT_PATH": str(json_path),
        "WRITE_FREQUENCY": write_freq
    }

def generate_config_file(params):
    """Génère le contenu de config.py."""
    content = f'''"""
Fichier de configuration généré par configure.py.
"""
import pathlib

# =============================================================================
# PARAMÈTRES GÉNÉRAUX
# =============================================================================

# Mode d'utilisation : "stream" (vidéo), "image" (fichier statique) ou "folder" (dossier d'images)
MODE = "{params['MODE']}"

# Chemin vers le modèle YOLO (.pt)
MODEL_PATH = pathlib.Path("{params['MODEL_PATH']}")

# Device de calcul : "cpu" ou "cuda" (GPU)
DEVICE = "{params['DEVICE']}"

# =============================================================================
# PARAMÈTRES STREAM VIDÉO (si MODE = "stream")
# =============================================================================

# Type de caméra : "webcam" ou "realsense"
CAMERA_TYPE = "{params['CAMERA_TYPE']}"

# Index de la caméra
CAMERA_INDEX = {params['CAMERA_INDEX']}

# =============================================================================
# PARAMÈTRES IMAGE (si MODE = "image")
# =============================================================================

# Chemin vers l'image à analyser
IMAGE_PATH = pathlib.Path("{params['IMAGE_PATH']}")

# =============================================================================
# PARAMÈTRES FOLDER (si MODE = "folder")
# =============================================================================

# Chemin vers le dossier contenant les images
FOLDER_PATH = pathlib.Path("{params['FOLDER_PATH']}")

# Sauvegarder les images annotées
SAVE_FOLDER_RESULTS = {params['SAVE_FOLDER_RESULTS']}

# Dossier de sortie pour les images annotées (None = créer sous-dossier "yolo_results")
FOLDER_OUTPUT_PATH = {f'pathlib.Path("{params["FOLDER_OUTPUT_PATH"]}")' if params["FOLDER_OUTPUT_PATH"] else 'None'}

# =============================================================================
# ENREGISTREMENT DES DÉTECTIONS (JSON)
# =============================================================================

# Activer l'enregistrement JSON
SAVE_JSON = {params['SAVE_JSON']}

# Chemin du fichier de sortie
JSON_OUTPUT_PATH = pathlib.Path("{params['JSON_OUTPUT_PATH']}")

# Fréquence d'écriture (tous les N frames)
WRITE_FREQUENCY = {params['WRITE_FREQUENCY']}
'''
    return content

def main():
    try:
        params = get_user_input()
        content = generate_config_file(params)
        
        config_path = pathlib.Path(__file__).parent / "config.py"
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        print()
        print("=" * 60)
        print(f"✅ CONFIGURATION SAUVEGARDÉE : {config_path}")
        print("=" * 60)
        print("👉 Vous pouvez maintenant lancer : python3 main.py")
        print()
        
    except KeyboardInterrupt:
        print("\n🛑 Configuration annulée.")
        sys.exit(0)

if __name__ == "__main__":
    main()
