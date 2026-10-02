"""
Fichier de configuration généré par configure.py.
"""
import pathlib

# =============================================================================
# PARAMÈTRES GÉNÉRAUX
# =============================================================================

# Mode d'utilisation : "stream" (vidéo), "image" (fichier statique) ou "folder" (dossier d'images)
MODE = "stream"

# Chemin vers le modèle YOLO (.pt)
MODEL_PATH = pathlib.Path("/home/adrolab/Desktop/ROBIN/IA/MODELE IA/ia 8.5.pt")

# Device de calcul : "cpu" ou "cuda" (GPU)
DEVICE = "cuda"

# =============================================================================
# PARAMÈTRES STREAM VIDÉO (si MODE = "stream")
# =============================================================================

# Type de caméra : "webcam" ou "realsense"
CAMERA_TYPE = "webcam"

# Index de la caméra
CAMERA_INDEX = 6

# =============================================================================
# PARAMÈTRES IMAGE (si MODE = "image")
# =============================================================================

# Chemin vers l'image à analyser
IMAGE_PATH = pathlib.Path("test.jpg")

# =============================================================================
# PARAMÈTRES FOLDER (si MODE = "folder")
# =============================================================================

# Chemin vers le dossier contenant les images
FOLDER_PATH = pathlib.Path("/home/adrolab/Desktop/ROBIN/IA/test_images")

# Sauvegarder les images annotées
SAVE_FOLDER_RESULTS = True

# Dossier de sortie pour les images annotées (None = créer sous-dossier "yolo_results")
FOLDER_OUTPUT_PATH = None

# =============================================================================
# ENREGISTREMENT DES DÉTECTIONS (JSON)
# =============================================================================

# Activer l'enregistrement JSON
SAVE_JSON = True

# Chemin du fichier de sortie
JSON_OUTPUT_PATH = pathlib.Path("detections.json")

# Fréquence d'écriture (tous les N frames)
WRITE_FREQUENCY = 5
