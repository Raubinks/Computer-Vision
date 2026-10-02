#!/usr/bin/env python3
"""
YOLO Dataset Configurator v12 - English Version
Interactive tool for managing and preparing YOLO datasets

Features:
- Import/Export YOLO datasets
- Class management (delete, merge, rename, normalize)
- Data augmentation
- Quality analysis
- Dataset balancing
- Interactive visualization with FiftyOne

Author: Antigravity AI Assistant
Version: 12.0 (English)
"""

import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
import fiftyone.utils.random as four
import pathlib
import shutil
import os
import yaml
import stat
import errno
import uuid 
import glob 
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageOps
import random
import re
from tqdm import tqdm

# Liste de mots-clés prioritaires (classes génériques cibles)
GENERIC_FOOD_KEYWORDS = [
    'pizza', 'burger', 'sandwich', 'salad', 'soup', 'steak', 'sushi', 'taco', 'burrito',
    'pasta', 'noodle', 'rice', 'bread', 'cake', 'cookie', 'pie', 'donut', 'muffin',
    'chocolate', 'candy', 'ice cream', 'yogurt', 'cheese', 'butter', 'egg', 'meat',
    'chicken', 'beef', 'pork', 'lamb', 'fish', 'salmon', 'tuna', 'shrimp', 'crab',
    'lobster', 'apple', 'banana', 'orange', 'grape', 'strawberry', 'blueberry',
    'raspberry', 'pear', 'peach', 'plum', 'mango', 'pineapple', 'watermelon',
    'melon', 'lemon', 'lime', 'avocado', 'tomato', 'potato', 'carrot', 'onion',
    'pepper', 'broccoli', 'cabbage', 'lettuce', 'cucumber', 'corn', 'mushroom',
    'beer', 'wine', 'cider', 'juice', 'soda', 'cola', 'tea', 'coffee', 'milk', 'water'
]

class YoloDatasetManager:
    """
    Class pour gérer, nettoyer et export des datasets pour YOLOv8/v5
    using FiftyOne.
    """

    def __init__(self, dataset_name, dataset_dir=None, dataset_type=fo.types.YOLOv5Dataset, force_reload=False):
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir
        self.dataset_type = dataset_type
        self.force_reload = force_reload
        
        self.dataset = self._load_or_create_dataset()
        
        if len(self.dataset) > 0:
            self.label_field = self._detect_label_field()
            print(f"Active detection field identified : '{self.label_field}'")
        else:
            self.label_field = "ground_truth"

    def _handle_remove_readonly(self, func, path, exc):
        """Gestionnaire d'error robuste pour Windows."""
        excvalue = exc[1]
        if func in (os.rmdir, os.remove, os.unlink) and excvalue.errno == errno.EACCES:
            os.chmod(path, stat.S_IWRITE)
            try:
                func(path)
            except Exception:
                pass
        else:
            raise

    def _extract_classes_from_yaml(self, yaml_path):
        """Retrieves only the class list from YAML."""
        try:
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f) or {}
            
            names = data.get('names', [])
            if isinstance(names, dict):
                return list(names.values())
            elif isinstance(names, list):
                return names
            return []
        except Exception as e:
            print(f"Error lecture classes YAML : {e}")
            return []

    def _parse_yolo_line(self, line, img_width, img_height, classes_list):
        """Converts a YOLO line (class x y w h) en Detection FiftyOne."""
        parts = line.strip().split()
        if len(parts) < 5: return None
        
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        
        x_top_left = x_center - (width / 2)
        y_top_left = y_center - (height / 2)
        
        x_top_left = max(0, min(1, x_top_left))
        y_top_left = max(0, min(1, y_top_left))
        
        label_name = str(class_id)
        if classes_list and class_id < len(classes_list):
            label_name = classes_list[class_id]
            
        return fo.Detection(
            label=label_name,
            bounding_box=[x_top_left, y_top_left, width, height]
        )

    def _import_from_disk_logic(self, name, clean_dir):
        """
        Logique d'import "MANUELLE".
        Reads images and TXT files directly to create samples.
        """
        clean_path = pathlib.Path(clean_dir)
        
        # 1. Retrieving classes (Data.yaml)
        target_yaml = clean_path / "data.yaml"
        if not target_yaml.exists():
            candidates = list(clean_path.glob("*.yaml"))
            target_yaml = candidates[0] if candidates else None
        
        classes_list = []
        if target_yaml:
            classes_list = self._extract_classes_from_yaml(target_yaml)
            print(f"Classes detected in YAML : {classes_list}")

        # 2. Creating empty dataset
        if fo.dataset_exists(name):
            fo.delete_dataset(name)
        
        dataset = fo.Dataset(name)
        dataset.persistent = True
        if classes_list:
            dataset.default_classes = classes_list
            dataset.save()

        # 3. Import manuelle
        splits_to_check = {
            'train': ['train'],
            'valid': ['valid', 'val'],
            'test': ['test']
        }

        total_imported = 0
        print("\n--- Import Manuelle (Scan fichiers) ---")

        samples = []
        
        for split_tag, folder_candidates in splits_to_check.items():
            found_folder = None
            for cand in folder_candidates:
                cand_path = clean_path / cand
                if cand_path.exists() and cand_path.is_dir() and (cand_path / "images").exists():
                    found_folder = cand_path
                    break
            
            if found_folder:
                images_dir = found_folder / "images"
                labels_dir = found_folder / "labels"
                
                image_files = []
                for ext in ["*.jpg", "*.png", "*.jpeg", "*.bmp", "*.tif"]:
                    image_files.extend(list(images_dir.glob(ext)))
                
                print(f" -> Split '{split_tag}' : {len(image_files)} images found in {images_dir.name}")
                
                for img_path in image_files:
                    sample = fo.Sample(filepath=str(img_path))
                    sample.tags = [split_tag]
                    
                    if labels_dir.exists():
                        label_path = labels_dir / f"{img_path.stem}.txt"
                        if label_path.exists():
                            detections = []
                            try:
                                with open(label_path, 'r') as f:
                                    lines = f.readlines()
                                    for line in lines:
                                        det = self._parse_yolo_line(line, 1, 1, classes_list)
                                        if det: detections.append(det)
                                
                                sample["ground_truth"] = fo.Detections(detections=detections)
                            except Exception as e:
                                print(f"Error lecture label {label_path.name} : {e}")
                    
                    samples.append(sample)
            else:
                pass

        if samples:
            print(f"Adding {len(samples)} samples to database...")
            dataset.add_samples(samples)
            dataset.compute_metadata()
            dataset.save()
            total_imported = len(dataset)
        
        if total_imported == 0:
            print("\n/!\\ CRITICAL ERROR : 0 image imported.")
            print("Check that your folders contain images (.jpg, .png...).")
        else:
            print(f"\nTotal success! {total_imported} images imported.")

        return dataset

    def _load_or_create_dataset(self):
        """Charge ou importe le dataset principal."""
        if self.force_reload and fo.dataset_exists(self.dataset_name):
            print(f"--- CLEANUP ---")
            fo.delete_dataset(self.dataset_name)
            print("Cache cleared.")

        if fo.dataset_exists(self.dataset_name):
            print(f"Loading du dataset depuis le cache FiftyOne : '{self.dataset_name}'...")
            dataset = fo.load_dataset(self.dataset_name)
            if len(dataset) == 0 and self.dataset_dir:
                print(f"/!\\ Empty cache dataset. Forced reload.")
                fo.delete_dataset(self.dataset_name)
            else:
                return dataset
        
        if self.dataset_dir:
            clean_dir = str(pathlib.Path(self.dataset_dir).resolve())
            print(f"Import fraîche depuis le disque : {clean_dir}")
            if not os.path.exists(clean_dir):
                raise FileNotFoundError(f"Folder not found : {clean_dir}")
            
            return self._import_from_disk_logic(self.dataset_name, clean_dir)
        
        raise ValueError(f"Dataset '{self.dataset_name}' not found.")

    def _detect_label_field(self):
        if "ground_truth" in self.dataset.get_field_schema(): return "ground_truth"
        schema = self.dataset.get_field_schema()
        for f, ft in schema.items():
            if hasattr(ft, 'document_type') and ft.document_type is fo.Detections: return f
        return "ground_truth"

    def get_summary(self):
        print("-" * 30)
        print(f"Dataset: {self.dataset.name} | Images: {len(self.dataset)}")
        
        # Compter les images background (vides ou sans champ)
        label_field = self.label_field
        n_background = self.dataset.match(
            (F(label_field) == None) | (F(f"{label_field}.detections").length() == 0)
        ).count()
        
        print(f"Classes: {self.dataset.default_classes}")
        print(f"Tags used: {self.dataset.count_sample_tags()}") 
        print(f"Backgrounds (empty images): {n_background}")
        print("-" * 30)

    def count_images_per_class(self):
        label_field = self.label_field
        classes = self.dataset.default_classes or self.dataset.distinct(f"{label_field}.detections.label")
        print(f"\n--- Statistics ({len(classes)} classes) ---")
        for c in classes:
            count = self.dataset.match(F(f"{label_field}.detections.label").contains(c)).count()
            print(f"- {c}: {count} images")


    def visualize_class_distribution(self):
        """Affiche un graphique et une liste triée de la distribution des classes."""
        label_field = self.label_field
        
        # Récupérer les classes de manière sécurisée
        try:
            classes = self.dataset.distinct(f"{label_field}.detections.label")
        except:
            # Fallback si distinct() échoue
            classes_set = set()
            for sample in self.dataset:
                if sample[label_field] and sample[label_field].detections:
                    for det in sample[label_field].detections:
                        if det.label:
                            classes_set.add(det.label)
            classes = sorted(list(classes_set))
        
        if not classes:
            print("❌ None class trouvée dans le dataset")
            return
        
        stats = []
        print(f"\n--- Class distribution ({len(classes)} classes) ---")
        
        # Compter les images par class
        for c in classes:
            try:
                # Essayer la méthode rapide avec match/contains
                count = self.dataset.match(F(f"{label_field}.detections.label").contains(c)).count()
            except:
                # Fallback: comptage manuel si match échoue
                count = 0
                for sample in self.dataset:
                    if sample[label_field] and sample[label_field].detections:
                        if any(d.label == c for d in sample[label_field].detections):
                            count += 1
            stats.append((c, count))
        
        # Tri par nombre d'images croissant
        stats.sort(key=lambda x: x[1])
        
        print(f"{'Class':<20} | {'Images':<10}")
        print("-" * 35)
        for c, count in stats:
            print(f"{c:<20} | {count:<10}")
            
        # Visualisation graphique
        names = [x[0] for x in stats]
        counts = [x[1] for x in stats]
        
        plt.figure(figsize=(10, 6))
        plt.barh(names, counts, color='skyblue')
        plt.xlabel('Nombre d\'images')
        plt.ylabel('Classes')
        plt.title('Class distribution (from least to most represented)')
        plt.tight_layout()
        plt.show()

    def resize_images(self, width, height):
        """Redimensionne toutes les images du dataset."""
        print(f"\n--- Resizing images to {width}x{height} ---")
        print("/!\\ WARNING: This operation will overwrite original files.")
        confirm = input("Confirm ? (oui/non) : ")
        if confirm.lower() != "oui":
            print("Cancelled.")
            return

        count = 0
        errors = 0
        
        for sample in self.dataset.iter_samples(autosave=True):
            try:
                filepath = sample.filepath
                with Image.open(filepath) as img:
                    img_resized = img.resize((width, height))
                    img_resized.save(filepath)
                
                # Mise à jour des métadonnées (optionnel mais recommandé)
                sample.metadata = fo.ImageMetadata(
                    width=width, 
                    height=height, 
                    num_channels=3, # Supposé RGB
                    mime_type=f"image/{pathlib.Path(filepath).suffix[1:]}"
                )
                count += 1
                if count % 100 == 0:
                    print(f"Processed : {count}...")
            except Exception as e:
                print(f"Error sur {filepath} : {e}")
                errors += 1
        
        self.dataset.compute_metadata()
        self.dataset.save()
        print(f"\nTerminé ! {count} images redimensionnées. {errors} errors.")

    def add_background_images(self, folder_path):
        """
        Importe toutes les images d'un folder comme images de background (sans détections).
        """
        path = pathlib.Path(folder_path).resolve()
        if not path.exists():
            print(f"❌ Folder not found : {path}")
            return

        print(f"\n📥 Import d'images background depuis : {path}")
        
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        image_files = []
        
        # Scan récursif
        for ext in image_extensions:
            image_files.extend(path.rglob(f"*{ext}"))
            image_files.extend(path.rglob(f"*{ext.upper()}"))
            
        if not image_files:
            print("❌ No images found.")
            return

        print(f"📊 {len(image_files)} images trouvées.")
        confirm = input("Import ces images ? (o/n) : ").strip().lower()
        if confirm != 'o':
            print("Cancelled.")
            return

        print("Import en cours...")
        samples = []
        for img_path in tqdm(image_files, desc="Adding samples"):
            sample = fo.Sample(filepath=str(img_path))
            # Créer un champ detections vide explicitement
            sample[self.label_field] = fo.Detections(detections=[])
            sample.tags = ["background"]
            samples.append(sample)
        
        self.dataset.add_samples(samples)
        self.dataset.save()
        print(f"✅ {len(samples)} images added to dataset.")



    def remove_class_annotations(self, class_name, keep_images=True):
        """
        Supprime uniquement les annotations d'une class spécifique dans les labels,
        sans delete les images. Utile pour convertir des images en background.
        
        Args:
            class_name (str): Nom de la class à delete des labels
            keep_images (bool): Si True, garde les images même si elles deviennent vides
        
        Exemple:
            # Delete toutes les annotations "plate" pour créer des backgrounds
            manager.remove_class_annotations("plate", keep_images=True)
        """
        print(f"\n{'='*70}")
        print(f"🗑️  ANNOTATION DELETION - Class: '{class_name}'")
        print(f"{'='*70}")
        print()
        
        label_field = self.label_field
        
        # Vérifier que la class existe dans le dataset
        try:
            if self.dataset.default_classes:
                all_classes = set(self.dataset.default_classes)
            else:
                print("⚠️  default_classes non défini, extraction depuis les annotations...")
                all_classes = set(self.dataset.distinct(f"{label_field}.detections.label"))
        except Exception as e:
            print(f"⚠️  Extraction manuelle des classes...")
            all_classes = set()
            for sample in self.dataset:
                if sample[label_field] and sample[label_field].detections:
                    for det in sample[label_field].detections:
                        if det.label:
                            all_classes.add(det.label)
        
        if class_name not in all_classes:
            print(f"❌ La class '{class_name}' does not exist in dataset")
            print(f"   Available classes: {sorted(list(all_classes))}")
            return
        
        # Compter les images contenant cette class manuellement
        images_with_class = 0
        for sample in self.dataset:
            if sample[label_field] and sample[label_field].detections:
                if any(d.label == class_name for d in sample[label_field].detections):
                    images_with_class += 1
        
        if images_with_class == 0:
            print(f"❌ No images found avec la class '{class_name}'")
            return
        
        print(f"📊 Images containing '{class_name}': {images_with_class}")
        print()
        
        # Demander confirmation
        if keep_images:
            print(f"⚠️  The annotations '{class_name}' will be deleted")
            print(f"✅ Images will be kept (may become empty)")
        else:
            print(f"⚠️  The annotations '{class_name}' will be deleted")
            print(f"❌ Les empty images will be deleted")
        
        print()
        confirm = input("Confirm la deletion ? (o/n): ").strip().lower()
        
        if confirm != 'o':
            print("❌ Operation cancelled")
            return
        
        print()
        print("🔄 Deletion des annotations...")
        
        # Compteurs
        images_modified = 0
        annotations_removed = 0
        images_became_empty = 0
        
        # Parcourir toutes les images avec barre de progression
        total_images = len(self.dataset)
        
        with tqdm(total=total_images, desc="Deletion annotations", unit="img") as pbar:
            for sample in self.dataset.iter_samples(autosave=True):
                if sample[label_field] and sample[label_field].detections:
                    # Compter les annotations de cette class
                    count_before = len(sample[label_field].detections)
                    class_annotations = sum(1 for d in sample[label_field].detections if d.label == class_name)
                    
                    # Delete les annotations de cette class
                    sample[label_field].detections = [
                        d for d in sample[label_field].detections 
                        if d.label != class_name
                    ]
                    
                    count_after = len(sample[label_field].detections)
                    
                    if class_annotations > 0:
                        images_modified += 1
                        annotations_removed += class_annotations
                        
                        if count_after == 0:
                            images_became_empty += 1
                
                pbar.update(1)
        
        # Recalculer les classes réelles après deletion
        print("\n🔄 Updating class list...")
        actual_classes = set()
        for sample in self.dataset:
            if sample[label_field] and sample[label_field].detections:
                for det in sample[label_field].detections:
                    if det.label:
                        actual_classes.add(det.label)
        
        # Mettre à jour default_classes avec les classes réellement présentes
        self.dataset.default_classes = sorted(list(actual_classes))
        self.dataset.save()
        
        print(f"\n✅ Deleted annotations:")
        print(f"   Modified images: {images_modified}")
        print(f"   Deleted annotations: {annotations_removed}")
        print(f"   Images became empty: {images_became_empty}")
        print(f"   Remaining classes: {len(actual_classes)}")
        
        # Gérer les empty images
        if not keep_images and images_became_empty > 0:
            print()
            print(f"🗑️  Deletion des {images_became_empty} empty images...")
            
            empty = self.dataset.match(
                (F(label_field) == None) | (F(f"{label_field}.detections").length() == 0)
            )
            
            if len(empty) > 0:
                self.dataset.delete_samples(empty)
                print(f"✅ {len(empty)} empty images supprimées")
        elif keep_images and images_became_empty > 0:
            print()
            print(f"ℹ️  {images_became_empty} images are now empty (kept)")
            print(f"   These images can serve as negative samples (background)")
        
        self.dataset.compute_metadata()
        self.dataset.save()
        
        print()
        print(f"{'='*70}")
        print(f"✅ Operation completed")
        print(f"{'='*70}")

    def convert_classes_to_lowercase(self):
        """Convertit toutes les classes en minuscules."""
        print(f"\n--- Converting classes to lowercase ---")
        confirm = input("Confirm ? (oui/non) : ")
        if confirm.lower() != "oui":
            print("Cancelled.")
            return

        label_field = self.label_field
        
        # 1. Mettre à jour les annotations
        for sample in self.dataset.iter_samples(autosave=True):
            if sample[label_field] and sample[label_field].detections:
                for det in sample[label_field].detections:
                    if det.label:
                        det.label = det.label.lower()
        
        # 2. Mettre à jour la liste des classes par défaut
        if self.dataset.default_classes:
            self.dataset.default_classes = sorted(list(set([c.lower() for c in self.dataset.default_classes])))
            self.dataset.save()
        
        print("Conversion completed.")

    def normalize_class_names(self):
        """Normalise les noms de classes : retire tirets et caractères spéciaux, passe en minuscules.
        Fusionne automatiquement les classes qui ont le même nom après normalization."""
        print(f"\n--- CLASS NAME NORMALIZATION ---")
        print("This operation will :")
        print("  - Convert dashes and underscores to spaces")
        print("  - Remove special characters")
        print("  - Convert all characters to lowercase")
        print("  - Merge les classes ayant le même nom après normalization")
        
        confirm = input("\nConfirmer ? (oui/non) : ")
        if confirm.lower() != "oui":
            print("Cancelled.")
            return

        label_field = self.label_field
        classes = self.dataset.default_classes or self.dataset.distinct(f"{label_field}.detections.label")
        
        # Fonction de normalization
        def normalize_name(name):
            """Normalise un nom de class."""
            # Passer en minuscules
            normalized = name.lower()
            # Remplacer les tirets et underscores par des espaces
            normalized = normalized.replace('-', ' ')
            normalized = normalized.replace('_', ' ')
            # Retirer tous les caractères spéciaux sauf lettres, chiffres et espaces
            normalized = re.sub(r'[^a-z0-9 ]', '', normalized)
            # Retirer les espaces multiples
            normalized = re.sub(r' +', ' ', normalized)
            # Retirer les espaces en début et fin
            normalized = normalized.strip()
            return normalized
        
        # Créer un mapping ancien nom -> nouveau nom
        name_mapping = {}
        normalized_to_originals = {}  # Pour détecter les merges
        
        print("\nAnalyzing classes...")
        for cls in classes:
            normalized = normalize_name(cls)
            name_mapping[cls] = normalized
            
            if normalized not in normalized_to_originals:
                normalized_to_originals[normalized] = []
            normalized_to_originals[normalized].append(cls)
        
        # Afficher les changements
        print("\nDetected changes :")
        changes_count = 0
        merges_count = 0
        
        for normalized, originals in sorted(normalized_to_originals.items()):
            if len(originals) > 1:
                print(f"  MERGE : {originals} -> '{normalized}'")
                merges_count += 1
            elif originals[0] != normalized:
                print(f"  RENAME : '{originals[0]}' -> '{normalized}'")
                changes_count += 1
        
        if changes_count == 0 and merges_count == 0:
            print("  No changes necessary.")
            return
        
        print(f"\nTotal : {changes_count} renames, {merges_count} merges")
        
        final_confirm = input("\nProceed with modifications ? (oui/non) : ")
        if final_confirm.lower() != "oui":
            print("Cancelled.")
            return
        
        # Appliquer les changements
        print("\nApplying modifications...")
        for sample in self.dataset.iter_samples(autosave=True):
            if sample[label_field] and sample[label_field].detections:
                for det in sample[label_field].detections:
                    if det.label and det.label in name_mapping:
                        old_name = det.label
                        new_name = name_mapping[old_name]
                        if old_name != new_name:
                            det.label = new_name
        
        # Mettre à jour la liste des classes par défaut
        if self.dataset.default_classes:
            # Créer la nouvelle liste de classes (sans doublons)
            new_classes = sorted(list(set(normalized_to_originals.keys())))
            self.dataset.default_classes = new_classes
            self.dataset.save()
        
        print("\n✓ Normalization terminée !")
        print(f"  - {changes_count} classes renamed")
        print(f"  - {merges_count} merges effectuées")
        print(f"  - {len(normalized_to_originals)} classes total")

    def compare_classes(self):
        """Compare une liste de classes fournie par l'utilisateur avec celles du dataset."""
        print(f"\n--- Class comparison ---")
        user_input = input("Enter the class list (separated by commas, ex: avocado, beef) : ")
        
        # Nettoyage et création du set utilisateur
        user_classes = {c.strip() for c in user_input.split(',') if c.strip()}
        
        # Retrieving classes du dataset
        label_field = self.label_field
        dataset_classes = set(self.dataset.default_classes or self.dataset.distinct(f"{label_field}.detections.label"))
        
        # Calcul des différences
        common = user_classes.intersection(dataset_classes)
        missing_in_dataset = user_classes - dataset_classes
        extra_in_dataset = dataset_classes - user_classes
        
        print("\n--- RESULTS ---")
        print(f"Common classes ({len(common)}) :")
        print(f"  {', '.join(sorted(list(common))) if common else '(None)'}")
        
        print(f"\nMissing in dataset ({len(missing_in_dataset)}) [Present in your list but not in dataset] :")
        print(f"  {', '.join(sorted(list(missing_in_dataset))) if missing_in_dataset else '(None)'}")
        
        print(f"\nExtra in dataset ({len(extra_in_dataset)}) [Present in dataset but not in your list] :")
        print(f"  {', '.join(sorted(list(extra_in_dataset))) if extra_in_dataset else '(None)'}")

    def merge_similar_classes(self):
        """
        Analysis les classes du dataset et propose des merges pour regrouper
        les classes similaires en classes génériques.
        Ex: 'pizza slice' -> 'pizza', 'apple red' -> 'apple', 'chicken breast' -> 'chicken'
        """
        print("\n" + "="*70)
        print("🔍 ANALYSE ET MERGE DE CLASSES SIMILAIRES".center(70))
        print("="*70)
        print()
        
        label_field = self.label_field
        classes = sorted(self.dataset.default_classes or self.dataset.distinct(f"{label_field}.detections.label"))
        
        if not classes:
            print("❌ None class trouvée dans le dataset")
            return
        
        print(f"📊 Nombre total de classes: {len(classes)}")
        print()
        
        # Analyser les classes et détecter les patterns
        print("🔎 Analysis des classes...")
        print()
        
        # Dictionnaire pour stocker les propositions de merge
        # Format: {classe_générique: [liste des classes spécifiques]}
        fusion_proposals = {}
        
        # Mots-clés à ignorer/delete pour trouver la class générique
        modifiers = [
            'red', 'green', 'yellow', 'white', 'black', 'brown', 'orange', 'purple',
            'fresh', 'frozen', 'cooked', 'raw', 'grilled', 'fried', 'baked', 'boiled',
            'slice', 'sliced', 'diced', 'chopped', 'whole', 'half', 'quarter',
            'piece', 'pieces', 'chunk', 'chunks',
            'breast', 'thigh', 'wing', 'leg', 'fillet',
            'large', 'small', 'medium', 'big', 'mini',
            'sweet', 'sour', 'spicy', 'hot', 'cold',
            'organic', 'natural', 'artificial'
        ]
        
        # Pour chaque class, essayer de trouver une version générique
        for class_name in classes:
            # Nettoyer et analyser le nom
            words = class_name.lower().replace('_', ' ').replace('-', ' ').split()
            
            # Si la class a plusieurs mots, chercher la version générique
            if len(words) > 1:
                # Essayer de retirer les modificateurs
                generic_words = [w for w in words if w not in modifiers]
                
                if generic_words and len(generic_words) < len(words):
                    # Créer le nom générique
                    generic_name = ' '.join(generic_words)
                    
                    # Vérifier si ce nom générique existe déjà dans les classes
                    if generic_name in classes or any(generic_name in c.lower() for c in classes):
                        # Ajouter à la proposition de merge
                        if generic_name not in fusion_proposals:
                            fusion_proposals[generic_name] = []
                        fusion_proposals[generic_name].append(class_name)
        
        # Afficher les propositions
        if not fusion_proposals:
            print("ℹ️  None merge automatique détectée.")
            print("   Les classes semblent déjà génériques ou uniques.")
            print()
            
            # Proposer une merge manuelle
            manual = input("Voulez-vous créer des merges manuellement ? (o/n): ").strip().lower()
            if manual == 'o':
                self._manual_class_merge()
            return
        
        print(f"✅ {len(fusion_proposals)} groupe(s) de merge détecté(s):")
        print()
        
        # Afficher les propositions
        for idx, (generic, specifics) in enumerate(sorted(fusion_proposals.items()), 1):
            if len(specifics) > 0:
                print(f"{idx}. '{generic}' ← {len(specifics)} class(s):")
                for spec in sorted(specifics):
                    # Compter les images
                    view = self.dataset.match(F(f"{label_field}.detections.label").contains(spec))
                    count = len(view)
                    print(f"     - {spec} ({count} images)")
                print()
        
        # Demander confirmation
        print("="*70)
        choice = input("Voulez-vous appliquer ces merges ? (o/n/m pour manuel): ").strip().lower()
        
        if choice == 'm':
            self._manual_class_merge()
            return
        elif choice != 'o':
            print("❌ Merge annulée")
            return
        
        # Appliquer les merges
        print()
        print("🔄 Application des merges...")
        print()
        
        total_merged = 0
        # Préparer le mapping global
        full_mapping = {}
        for generic, specifics in fusion_proposals.items():
            if specifics:
                for specific in specifics:
                    if specific != generic:
                        full_mapping[specific] = generic
                        print(f"  Mapping: '{specific}' → '{generic}'")
                        total_merged += 1
                        
        # Appliquer en batch
        if full_mapping:
            self._batch_merge_classes(full_mapping)
        
        self.dataset.save()
        
        print()
        print("="*70)
        print(f"✅ Merge terminée ! {total_merged} class(s) fusionnée(s)")
        print(f"📊 Nombre de classes après merge: {len(self.dataset.default_classes or self.dataset.distinct(f'{label_field}.detections.label'))}")
        print("="*70)
    
    def _manual_class_merge(self):
        """Permet à l'utilisateur de créer des merges de classes manuellement."""
        print("\n--- MERGE MANUELLE DE CLASSES ---")
        print("Format: classe_source1,classe_source2,... -> classe_cible")
        print("Exemple: pizza slice,pizza margherita,pizza pepperoni -> pizza")
        print()
        
        label_field = self.label_field
        classes = sorted(self.dataset.default_classes or self.dataset.distinct(f"{label_field}.detections.label"))
        
        print(f"Available classes ({len(classes)}):")
        for i, cls in enumerate(classes, 1):
            view = self.dataset.match(F(f"{label_field}.detections.label").contains(cls))
            count = len(view)
            print(f"  {i:3d}. {cls} ({count} images)")
        
        print()
        user_input = input("Entrez la merge (ou 'q' pour annuler): ").strip()
        
        if user_input.lower() == 'q':
            print("❌ Cancelled")
            return
        
        # Parser l'input
        if '->' not in user_input:
            print("❌ Format invalide. Utilisez: source1,source2 -> cible")
            return
        
        parts = user_input.split('->')
        if len(parts) != 2:
            print("❌ Format invalide")
            return
        
        sources = [s.strip() for s in parts[0].split(',') if s.strip()]
        target = parts[1].strip()
        
        if not sources or not target:
            print("❌ Classes source ou cible manquantes")
            return
        
        # Vérifier que les classes existent
        valid_sources = []
        for source in sources:
            if source not in classes:
                print(f"⚠️ Class '{source}' not found dans le dataset (ignorée)")
            else:
                valid_sources.append(source)
                
        if not valid_sources:
            print("❌ None class source valide.")
            return

        # Appliquer les merges en batch
        print()
        print(f"🔄 Merge de {len(valid_sources)} class(s) vers '{target}'...")
        
        mapping = {src: target for src in valid_sources if src != target}
        if mapping:
            self._batch_merge_classes(mapping)
        else:
             print("None merge nécessaire (sources identiques à la cible).")
        
        print(f"\n✅ Merge terminée !")

    def _batch_merge_classes(self, mapping):
        """
        Fusionne plusieurs classes en une seule passe optimisée.
        Arg: mapping = {source_label: target_label, ...}
        """
        if not mapping:
             return
             
        classes_to_change = list(mapping.keys())
        # Filtrer uniquement les samples concernés (énorme gain de performance)
        label_field = self.label_field
        view = self.dataset.match(
            F(f"{label_field}.detections.label").is_in(classes_to_change)
        )
        
        count = len(view)
        print(f"⚡ Application optimisée de {len(mapping)} règles sur {count} images concernées...")
        
        with tqdm(total=count, desc="Merge classes", unit="img") as pbar:
            for sample in view.iter_samples(autosave=False):
                if sample[label_field] and sample[label_field].detections:
                    modified = False
                    for d in sample[label_field].detections:
                        if d.label in mapping:
                            d.label = mapping[d.label]
                            modified = True
                    if modified:
                        sample.save()
                pbar.update(1)
        
        # Mise à jour des default_classes
        if self.dataset.default_classes:
            current = set(self.dataset.default_classes)
            for src, tgt in mapping.items():
                if src in current:
                     current.remove(src)
                current.add(tgt)
            self.dataset.default_classes = sorted(list(current))
            self.dataset.save()


    def rename_class(self, old_name, new_name, update_files=True):
        """
        Rename a class in the dataset.
        
        Args:
            old_name (str): Current name of the class to rename
            new_name (str): New name for the class
            update_files (bool): If True, update underlying YOLO label files (default: True)
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Validation
        if not old_name or not new_name:
            print("❌ Error: Class names cannot be empty.")
            return False
        
        if not isinstance(old_name, str) or not isinstance(new_name, str):
            print("❌ Error: Class names must be strings.")
            return False
        
        # Check if old class exists
        if self.dataset.default_classes and old_name not in self.dataset.default_classes:
            print(f"❌ Error: Class '{old_name}' does not exist in the dataset.")
            print(f"   Available classes: {', '.join(sorted(self.dataset.default_classes))}")
            return False
        
        # Check if new name already exists (potential merge)
        if self.dataset.default_classes and new_name in self.dataset.default_classes:
            print(f"⚠️  Warning: Class '{new_name}' already exists!")
            print(f"   This will merge '{old_name}' into '{new_name}'.")
            confirm = input("   Continue? (y/n): ").strip().lower()
            if confirm != 'y':
                print("   Operation cancelled.")
                return False
        
        print(f"\n🔄 Renaming class '{old_name}' → '{new_name}'...")
        
        # Count affected samples
        label_field = self.label_field
        
        # Use a more robust filtering method that handles special characters
        # Instead of using == which can fail with special chars, iterate and count
        affected_samples = []
        for sample in self.dataset.iter_samples():
            if sample[label_field] and sample[label_field].detections:
                for detection in sample[label_field].detections:
                    if detection.label == old_name:
                        affected_samples.append(sample)
                        break  # Count each sample only once
        
        affected_count = len(affected_samples)
        
        if affected_count == 0:
            print(f"⚠️  No samples found with class '{old_name}'.")
            return False
        
        print(f"   📊 Found {affected_count} images with class '{old_name}'")
        
        # Update FiftyOne dataset
        print(f"   🔧 Updating FiftyOne dataset...")
        updated_samples = 0
        
        with tqdm(total=affected_count, desc="Renaming in FiftyOne", unit="img") as pbar:
            for sample in affected_samples:
                if sample[label_field] and sample[label_field].detections:
                    modified = False
                    for detection in sample[label_field].detections:
                        if detection.label == old_name:
                            detection.label = new_name
                            modified = True
                    
                    if modified:
                        sample.save()
                        updated_samples += 1
                
                pbar.update(1)
        
        # Update default_classes
        if self.dataset.default_classes:
            current_classes = set(self.dataset.default_classes)
            if old_name in current_classes:
                current_classes.remove(old_name)
            current_classes.add(new_name)
            self.dataset.default_classes = sorted(list(current_classes))
            self.dataset.save()
        
        print(f"   ✅ Updated {updated_samples} samples in FiftyOne")
        
        # Update YOLO label files if requested
        if update_files:
            print(f"   📝 Updating YOLO label files...")
            
            # Get class ID mapping
            if not self.dataset.default_classes:
                print("   ⚠️  Warning: No default_classes found. Skipping file updates.")
                return True
            
            # Calculate old and new class indices
            try:
                # Get the OLD class list (before rename)
                old_classes_list = sorted(list(current_classes - {new_name} | {old_name}))
                # Get the NEW class list (after rename)
                new_classes_list = sorted(list(current_classes))
                
                # Find old and new indices
                old_class_id = old_classes_list.index(old_name) if old_name in old_classes_list else None
                new_class_id = new_classes_list.index(new_name) if new_name in new_classes_list else None
                
                if old_class_id is None or new_class_id is None:
                    print(f"   ⚠️  Warning: Could not determine class indices.")
                    print(f"   ℹ️  FiftyOne dataset was updated successfully.")
                    return True
                
                print(f"   🔢 Class index mapping: {old_class_id} ('{old_name}') → {new_class_id} ('{new_name}')")
                
                # Only update files if the index actually changed
                if old_class_id == new_class_id:
                    print(f"   ℹ️  Class index unchanged ({old_class_id}). No label file updates needed.")
                    return True
                
                # Build complete index mapping for ALL classes (in case multiple changed)
                # This handles the case where renaming shifts other class indices
                index_mapping = {}
                for old_idx, old_class in enumerate(old_classes_list):
                    if old_class == old_name:
                        # This class was renamed
                        index_mapping[old_idx] = new_class_id
                    else:
                        # Find where this class ended up in new list
                        new_idx = new_classes_list.index(old_class)
                        if old_idx != new_idx:
                            index_mapping[old_idx] = new_idx
                
                if not index_mapping:
                    print(f"   ℹ️  No index changes needed.")
                    return True
                
                print(f"   🔄 Index remapping needed for {len(index_mapping)} class(es)")
                
                files_updated = 0
                files_with_changes = 0
                
                with tqdm(total=len(self.dataset), desc="Updating label files", unit="file") as pbar:
                    for sample in self.dataset.iter_samples():
                        # Get the label file path
                        img_path = sample.filepath
                        
                        # Construct label file path (assuming YOLO structure)
                        label_path = img_path.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
                        label_path_obj = pathlib.Path(label_path)
                        
                        if not label_path_obj.exists():
                            # Try alternative paths
                            alt_paths = [
                                pathlib.Path(str(img_path).rsplit('.', 1)[0] + '.txt'),
                                pathlib.Path(str(img_path).replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'),
                            ]
                            
                            for alt_path in alt_paths:
                                if alt_path.exists():
                                    label_path_obj = alt_path
                                    break
                        
                        if label_path_obj.exists():
                            try:
                                # Read the label file
                                with open(label_path_obj, 'r') as f:
                                    lines = f.readlines()
                                
                                # Update class indices in the file
                                updated_lines = []
                                file_modified = False
                                
                                for line in lines:
                                    line = line.strip()
                                    if not line:
                                        updated_lines.append(line)
                                        continue
                                    
                                    parts = line.split()
                                    if len(parts) >= 5:
                                        try:
                                            class_id = int(parts[0])
                                            
                                            # Check if this class ID needs to be remapped
                                            if class_id in index_mapping:
                                                new_id = index_mapping[class_id]
                                                parts[0] = str(new_id)
                                                file_modified = True
                                            
                                            updated_lines.append(' '.join(parts))
                                        except ValueError:
                                            # Invalid class ID, keep line as-is
                                            updated_lines.append(line)
                                    else:
                                        updated_lines.append(line)
                                
                                # Write back if modified
                                if file_modified:
                                    with open(label_path_obj, 'w') as f:
                                        f.write('\n'.join(updated_lines))
                                        if updated_lines and updated_lines[-1]:  # Add final newline if needed
                                            f.write('\n')
                                    files_with_changes += 1
                                
                                files_updated += 1
                                
                            except Exception as e:
                                print(f"   ⚠️  Error updating {label_path_obj}: {e}")
                        
                        pbar.update(1)
                
                print(f"   ✅ Updated {files_with_changes} label files (checked {files_updated} total)")
                
            except Exception as e:
                print(f"   ⚠️  Error updating label files: {e}")
                print(f"   ℹ️  FiftyOne dataset was updated successfully.")
                import traceback
                traceback.print_exc()
        
        print(f"\n✅ Successfully renamed '{old_name}' → '{new_name}'")
        print(f"   📊 {updated_samples} images affected")
        
        return True



    def delete_class(self, class_name, delete_empty_images=True):
        label_field = self.label_field
        print(f"\nSuppression class '{class_name}'...")
        view = self.dataset.match(F(f"{label_field}.detections.label").contains(class_name))
        
        if len(view) == 0:
            print("Class not found.")
            return

        for s in view.iter_samples(autosave=True):
            if s[label_field]:
                s[label_field].detections = [d for d in s[label_field].detections if d.label != class_name]
        
        if self.dataset.default_classes and class_name in self.dataset.default_classes:
            self.dataset.default_classes.remove(class_name)
            self.dataset.save()
        print("Deleted annotations.")

        if delete_empty_images:
            empty = self.dataset.match((F(label_field) == None) | (F(f"{label_field}.detections").length() == 0))
            if len(empty) > 0:
                print(f"Deletion de {len(empty)} empty images.")
                self.dataset.delete_samples(empty)
        
        self.dataset.compute_metadata()

    def delete_classes_by_image_count(self, n, comparator='le', delete_empty_images=True):
        """Supprime toutes les classes selon un critère sur le nombre d'images.

        Args:
            n (int): seuil d'images.
            comparator (str): 'lt', 'le', 'eq', 'ge', 'gt' (par défaut 'le' = <=).
            delete_empty_images (bool): supprime les images devenues vides si True.
        """
        label_field = self.label_field
        classes = self.dataset.default_classes or self.dataset.distinct(f"{label_field}.detections.label")

        # Normalize comparator
        comp = comparator.lower()
        valid = {'lt', 'le', 'eq', 'ge', 'gt'}
        if comp not in valid:
            print(f"Comparator invalide '{comparator}', utilisation de 'le' par défaut.")
            comp = 'le'

        to_delete = []
        for cls in classes:
            cnt = self.dataset.match(F(f"{label_field}.detections.label").contains(cls)).count()
            keep = True
            if comp == 'lt' and cnt < n: keep = False
            if comp == 'le' and cnt <= n: keep = False
            if comp == 'eq' and cnt == n: keep = False
            if comp == 'ge' and cnt >= n: keep = False
            if comp == 'gt' and cnt > n: keep = False
            if not keep:
                to_delete.append((cls, cnt))

        if not to_delete:
            print(f"None class ne correspond au critère ({comparator} {n}).")
            return

        print("Classes sélectionnées pour deletion :")
        for cls, cnt in to_delete:
            print(f" - {cls}: {cnt} images")

        confirm = input("Confirm deletion de ces classes ? (oui/non) : ")
        if confirm.lower() != 'oui':
            print("Cancelled.")
            return

        classes_to_remove = {cls for cls, _ in to_delete}
        print(f"Deletion des annotations pour {len(classes_to_remove)} classes en un seul passage...")

        # Itérer sur tout le dataset UNE SEULE FOIS
        for sample in self.dataset.iter_samples(autosave=True):
            if sample[label_field] and sample[label_field].detections:
                # Filtrer toutes les classes à delete en une fois
                sample[label_field].detections = [
                    d for d in sample[label_field].detections 
                    if d.label not in classes_to_remove
                ]

        # Mettre à jour la liste des classes par défaut
        if self.dataset.default_classes:
            self.dataset.default_classes = [c for c in self.dataset.default_classes if c not in classes_to_remove]
        
        removed = list(classes_to_remove)

        if delete_empty_images:
            empty = self.dataset.match((F(label_field) == None) | (F(f"{label_field}.detections").length() == 0))
            if len(empty) > 0:
                print(f"Deletion de {len(empty)} empty images.")
                self.dataset.delete_samples(empty)

        self.dataset.save()
        self.dataset.compute_metadata()
        print(f"Deletion terminée. Classes supprimées : {removed}")

    def delete_multiple_classes(self, class_list, delete_empty_images=True):
        """Supprime plusieurs classes fournies en une fois.

        Args:
            class_list (list or str): liste de classes ou chaîne séparée par des virgules.
        """
        label_field = self.label_field
        # Accepter une chaîne séparée par des virgules
        if isinstance(class_list, str):
            class_list = [c.strip() for c in class_list.split(',') if c.strip()]

        if not isinstance(class_list, (list, tuple)) or not class_list:
            print("Liste de classes invalide ou vide.")
            return

        # Vérifier quelles classes existent réellement
        dataset_classes = set(self.dataset.default_classes or self.dataset.distinct(f"{label_field}.detections.label"))
        to_process = [c for c in class_list if c in dataset_classes]
        not_found = [c for c in class_list if c not in dataset_classes]

        if not to_process:
            print(f"None des classes fournies n'existe dans le dataset. Non trouvées : {not_found}")
            return

        print(f"Classes trouvées et prêtes à être supprimées : {to_process}")
        if not_found:
            print(f"Non trouvées (ignorées) : {not_found}")

        confirm = input("Confirm deletion ? (oui/non) : ")
        if confirm.lower() != 'oui':
            print("Cancelled.")
            return

        classes_to_remove = set(to_process)
        print(f"\n🔄 Deletion de {len(classes_to_remove)} classes...")
        
        total_images = len(self.dataset)
        print(f"📊 Processing de {total_images} images...")
        
        # Itérer sur tout le dataset avec barre de progression
        with tqdm(total=total_images, desc="Deletion des annotations", unit="img") as pbar:
            for sample in self.dataset.iter_samples(autosave=True):
                if sample[label_field] and sample[label_field].detections:
                    # Filtrer toutes les classes à delete en une fois
                    sample[label_field].detections = [
                        d for d in sample[label_field].detections 
                        if d.label not in classes_to_remove
                    ]
                pbar.update(1)

        # Mettre à jour la liste des classes par défaut
        if self.dataset.default_classes:
            self.dataset.default_classes = [c for c in self.dataset.default_classes if c not in classes_to_remove]
            
        removed = to_process

        if delete_empty_images:
            print("\n🔍 Recherche des empty images...")
            empty = self.dataset.match((F(label_field) == None) | (F(f"{label_field}.detections").length() == 0))
            if len(empty) > 0:
                print(f"🗑️  Deletion de {len(empty)} empty images...")
                self.dataset.delete_samples(empty)
            else:
                print("✅ None image vide")

        print("\n💾 Sauvegarde...")
        self.dataset.save()
        self.dataset.compute_metadata()
        print(f"✅ {len(removed)} class(s) supprimée(s) : {removed}")
        print(f"Completed. Classes supprimées : {removed}")

    def keep_only_classes(self, class_list, delete_empty_images=True):
        """Garde uniquement les classes spécifiées et supprime toutes les autres.

        Args:
            class_list (list or str): liste de classes à garder ou chaîne séparée par des virgules.
            delete_empty_images (bool): supprime les images devenues vides si True.
        """
        label_field = self.label_field
        # Accepter une chaîne séparée par des virgules
        if isinstance(class_list, str):
            class_list = [c.strip() for c in class_list.split(',') if c.strip()]

        if not isinstance(class_list, (list, tuple)) or not class_list:
            print("Liste de classes invalide ou vide.")
            return

        # Récupérer toutes les classes du dataset de manière sécurisée
        try:
            if self.dataset.default_classes:
                dataset_classes = set(self.dataset.default_classes)
            else:
                # Si default_classes n'existe pas, extraire depuis les détections
                print("⚠️  default_classes non défini, extraction depuis les annotations...")
                dataset_classes = set(self.dataset.distinct(f"{label_field}.detections.label"))
        except Exception as e:
            print(f"❌ Error lors de la récupération des classes: {e}")
            print("Tentative d'extraction manuelle...")
            dataset_classes = set()
            for sample in self.dataset:
                if sample[label_field] and sample[label_field].detections:
                    for det in sample[label_field].detections:
                        if det.label:
                            dataset_classes.add(det.label)
            print(f"✅ {len(dataset_classes)} classes trouvées")
        
        if not dataset_classes:
            print("❌ None class trouvée dans le dataset")
            return
        
        # Vérifier quelles classes existent et seront gardées
        classes_to_keep = set(c for c in class_list if c in dataset_classes)
        not_found = [c for c in class_list if c not in dataset_classes]

        if not classes_to_keep:
            print(f"None des classes fournies n'existe dans le dataset. Non trouvées : {not_found}")
            return

        # Calculer les classes à delete (toutes sauf celles à garder)
        classes_to_remove = dataset_classes - classes_to_keep

        if not classes_to_remove:
            print("Toutes les classes du dataset sont déjà dans votre liste. None deletion nécessaire.")
            return

        print(f"\n--- CONSERVATION DE CLASSES ---")
        print(f"Classes à GARDER ({len(classes_to_keep)}) : {sorted(list(classes_to_keep))}")
        if not_found:
            print(f"Non trouvées dans le dataset (ignorées) : {not_found}")
        print(f"\nClasses à SUPPRIMER ({len(classes_to_remove)}) : {sorted(list(classes_to_remove))}")

        confirm = input("\nConfirmer la deletion de toutes les classes NON listées ? (oui/non) : ")
        if confirm.lower() != 'oui':
            print("Cancelled.")
            return

        print(f"\n🔄 Deletion de {len(classes_to_remove)} classes...")
        
        # Compter le nombre total d'images pour la barre de progression
        total_images = len(self.dataset)
        
        # Itérer sur tout le dataset avec barre de progression
        print(f"📊 Processing de {total_images} images...")
        
        with tqdm(total=total_images, desc="Nettoyage des annotations", unit="img") as pbar:
            for sample in self.dataset.iter_samples(autosave=True):
                if sample[label_field] and sample[label_field].detections:
                    # Garder uniquement les détections des classes à conserver
                    sample[label_field].detections = [
                        d for d in sample[label_field].detections 
                        if d.label in classes_to_keep
                    ]
                pbar.update(1)

        # Mettre à jour la liste des classes par défaut
        self.dataset.default_classes = sorted(list(classes_to_keep))
            
        removed = sorted(list(classes_to_remove))

        if delete_empty_images:
            print("\n🔍 Recherche des empty images...")
            empty = self.dataset.match((F(label_field) == None) | (F(f"{label_field}.detections").length() == 0))
            if len(empty) > 0:
                print(f"🗑️  Deletion de {len(empty)} empty images...")
                self.dataset.delete_samples(empty)
            else:
                print("✅ None image vide à delete")

        print("\n💾 Sauvegarde du dataset...")
        self.dataset.save()
        self.dataset.compute_metadata()
        
        print(f"\n{'='*70}")
        print(f"✅ TERMINÉ !".center(70))
        print(f"{'='*70}")
        print(f"  📋 Classes kept ({len(classes_to_keep)}): {sorted(list(classes_to_keep))}")
        print(f"  🗑️  Classes supprimées ({len(removed)}): {removed}")
        print(f"  📊 Total d'images restantes: {len(self.dataset)}")
        print(f"{'='*70}")


    def clone_dataset(self, new_name):
        if fo.dataset_exists(new_name): fo.delete_dataset(new_name)
        print(f"Clonage vers '{new_name}'...")
        cloned = self.dataset.clone(new_name)
        cloned.persistent = True
        cloned.save()
        return cloned

    def merge_external_dataset(self, other_dataset_dirs):
        """Fusionne un ou plusieurs datasets externes.
        
        Args:
            other_dataset_dirs: Path unique ou chemins multiples séparés par des virgules ou point-virgules
        """
        # Gérer plusieurs chemins séparés par des virgules ou point-virgules
        if isinstance(other_dataset_dirs, str):
            # Séparer par virgule ou point-virgule et nettoyer
            separators = [',', ';']
            for sep in separators:
                if sep in other_dataset_dirs:
                    paths_list = [p.strip().replace('"', '').replace("'", '') for p in other_dataset_dirs.split(sep) if p.strip()]
                    break
            else:
                # Un seul path
                paths_list = [other_dataset_dirs.strip().replace('"', '').replace("'", '')]
        else:
            paths_list = other_dataset_dirs

        print(f"\n--- MERGE DE DATASETS ---")
        print(f"Nombre de datasets à merge : {len(paths_list)}")
        
        total_merged = 0
        total_images_added = 0
        all_new_classes = set()
        
        for idx, dataset_path in enumerate(paths_list, 1):
            path = pathlib.Path(dataset_path).resolve()
            
            if not path.exists():
                print(f"\n[{idx}/{len(paths_list)}] ❌ Error : Folder not found '{path}'")
                print("   -> Ignoré, passage au suivant...")
                continue

            print(f"\n[{idx}/{len(paths_list)}] Merge du dataset : {path.name}")
            temp_name = f"temp_merge_{uuid.uuid4().hex[:8]}"
            
            try:
                print(f"   1. Import...")
                temp_ds = self._import_from_disk_logic(temp_name, str(path))
                images_count = len(temp_ds)
                print(f"      -> {images_count} images chargées")
                
                print(f"   2. Merge des collections...")
                self.dataset.add_collection(temp_ds)
                
                # Récupérer les nouvelles classes
                if temp_ds.default_classes:
                    current_classes = set(self.dataset.default_classes or [])
                    new_classes = set(temp_ds.default_classes)
                    newly_added = new_classes - current_classes
                    
                    if newly_added:
                        print(f"      -> Nouvelles classes ajoutées : {sorted(list(newly_added))}")
                        all_new_classes.update(newly_added)
                    
                    merged_classes = sorted(list(current_classes.union(new_classes)))
                    self.dataset.default_classes = merged_classes
                
                total_merged += 1
                total_images_added += images_count
                print(f"   ✓ Merge réussie !")

            except Exception as e:
                print(f"   ❌ Error lors de la merge : {e}")
                import traceback
                traceback.print_exc()
            finally:
                if fo.dataset_exists(temp_name):
                    fo.delete_dataset(temp_name)
        
        # Sauvegarder le dataset final
        self.dataset.save()
        
        # Résumé final
        print(f"\n{'='*50}")
        print(f"RÉSUMÉ DE LA MERGE")
        print(f"{'='*50}")
        print(f"Datasets fusionnés avec success : {total_merged}/{len(paths_list)}")
        print(f"Total d'images ajoutées : {total_images_added}")
        print(f"Total d'images dans le dataset : {len(self.dataset)}")
        
        if all_new_classes:
            print(f"Nouvelles classes ajoutées ({len(all_new_classes)}) : {sorted(list(all_new_classes))}")
        
        final_classes = self.dataset.default_classes or []
        print(f"Total de classes : {len(final_classes)}")
        print(f"{'='*50}")
    def merge_datasets_from_folder(self, folder_path):
        """Fusionne tous les datasets détectés dans un folder."""
        path = pathlib.Path(folder_path).resolve()
        if not path.exists() or not path.is_dir():
            print(f"❌ Folder invalide : {path}")
            return
            
        print(f"🔍 Scan du folder : {path}")
        sub_datasets = []
        for child in path.iterdir():
            if child.is_dir():
                # Simple check for dataset validity
                if (child / "data.yaml").exists() or (child / "train").exists() or (child / "images").exists():
                    sub_datasets.append(str(child))
                    
        if not sub_datasets:
            print("❌ Aucun dataset trouvé dans ce folder.")
            return
            
        print(f"✅ {len(sub_datasets)} datasets trouvés.")
        confirm = input("Merge tous ces datasets ? (o/n) : ")
        if confirm.lower() != 'o':
            print("Cancelled.")
            return
            
        self.merge_external_dataset(sub_datasets)

    def split_dataset(self, train=None, val=None, test=None):
        """
        Divise le dataset aléatoirement. 
        Si les ratios ne sont pas fournis, demande à l'utilisateur de les saisir.
        """
        print("\n--- CONFIGURATION DES SPLITS ---")
        
        # Si aucun argument n'est fourni, on demande à l'utilisateur
        if train is None and val is None and test is None:
            while True:
                try:
                    print("Entrez les ratios (doivent sommer à 1.0, ex: 0.7, 0.2, 0.1)")
                    train = float(input("Ratio Train (ex: 0.7) : "))
                    val = float(input("Ratio Validation (ex: 0.2) : "))
                    test = float(input("Ratio Test (ex: 0.1) : "))
                    
                    total = train + val + test
                    if not (0.99 <= total <= 1.01):
                        print(f"Error : La somme fait {total:.2f}, elle doit être égale à 1.0. Réessayez.")
                        continue
                    break
                except ValueError:
                    print("Error : Veuillez entrer des nombres décimaux valides (ex: 0.1).")
        
        # Validation finale au cas où les args sont passés mais faux
        total = (train or 0) + (val or 0) + (test or 0)
        if not (0.99 <= total <= 1.01):
             print(f"Error : La somme des ratios ({total:.2f}) doit être égale à 1.0.")
             return

        print(f"Application des splits -> Train: {train}, Val: {val}, Test: {test}")
        
        # Nettoyage des anciens tags
        self.dataset.untag_samples(["train", "valid", "test", "val"])
        
        # Application aléatoire
        four.random_split(
            self.dataset,
            {"train": train, "valid": val, "test": test}
        )
        
        self.dataset.save()
        print("Splits appliqués avec success.")
        print(f"Répartition : {self.dataset.count_sample_tags()}")

    def reduce_class_samples(self, class_name, count_to_remove):
        """Supprime aléatoirement N images contenant une certaine class."""
        print(f"Réduction de la class '{class_name}' : deletion de {count_to_remove} images...")
        
        # Filtrer les images contenant la class
        view = self.dataset.match(F(f"{self.label_field}.detections.label").contains(class_name))
        
        total_available = len(view)
        if total_available == 0:
            print(f"No images found pour la class '{class_name}'.")
            return

        if count_to_remove >= total_available:
            print(f"Attention: Vous demandez de delete {count_to_remove} images mais il n'y en a que {total_available}.")
            print("Toutes les images de cette class will be deleted.")
            self.dataset.delete_samples(view)
        else:
            # Sélection aléatoire
            samples_to_delete = view.shuffle().take(count_to_remove)
            self.dataset.delete_samples(samples_to_delete)
            print(f"{count_to_remove} images supprimées avec success.")
        
        self.dataset.save()

    def balance_classes(self, max_samples_per_class):
        """Fixe un nombre maximum d'images pour toutes les classes (équilibrage)."""
        print(f"\n--- Équilibrage du Dataset (Max {max_samples_per_class} images/class) ---")
        
        classes = self.dataset.default_classes or self.dataset.distinct(f"{self.label_field}.detections.label")
        
        for cls in classes:
            view = self.dataset.match(F(f"{self.label_field}.detections.label").contains(cls))
            count = len(view)
            
            if count > max_samples_per_class:
                excess = count - max_samples_per_class
                print(f" -> Class '{cls}' : {count} images. Deletion de {excess} images excédentaires...")
                
                # Deletion aléatoire de l'excédent
                # Note : Cela peut affecter d'autres classes si les images sont multi-labels,
                # c'est un comportement normal pour un équilibrage simple.
                samples_to_delete = view.shuffle().take(excess)
                self.dataset.delete_samples(samples_to_delete)
            else:
                print(f" -> Class '{cls}' : {count} images (OK).")
        
        self.dataset.save()
        print("Équilibrage completed.")


    def augment_classes_to_target(self, target_count):
        """Augmente les classes sous-représentées par data augmentation jusqu'à atteindre le nombre cible."""
        print(f"\n--- AUGMENTATION DE DONNÉES (Cible: {target_count} images/class) ---")
        
        if not self.dataset_dir:
            print("/!\\ ERREUR : Cette fonction nécessite un dataset_dir défini.")
            print("Veuillez charger le dataset depuis le disque avec force_reload=True.")
            return
        
        label_field = self.label_field
        classes = self.dataset.default_classes or self.dataset.distinct(f"{label_field}.detections.label")
        
        # Déterminer le folder de sortie pour les images augmentées
        base_path = pathlib.Path(self.dataset_dir).resolve()
        augmented_dir = base_path / "augmented_temp"
        augmented_dir.mkdir(exist_ok=True)
        
        total_generated = 0
        
        for cls in classes:
            view = self.dataset.match(F(f"{label_field}.detections.label").contains(cls))
            current_count = len(view)
            
            if current_count >= target_count:
                print(f" -> Class '{cls}' : {current_count} images (OK, pas d'augmentation nécessaire)")
                continue
            
            needed = target_count - current_count
            print(f" -> Class '{cls}' : {current_count} images. Génération de {needed} images supplémentaires...")
            
            # Récupérer les samples de cette class
            samples_list = list(view)
            
            # Vérifier que la liste n'est pas vide
            if not samples_list:
                print(f"   /!\\ ATTENTION : Aucun échantillon trouvé pour la class '{cls}'. Impossible de générer des augmentations.")
                continue
            
            generated = 0
            while generated < needed:
                # Choisir un sample aléatoire à augmenter
                source_sample = random.choice(samples_list)
                source_path = pathlib.Path(source_sample.filepath)
                
                # Déterminer le folder de destination (train/valid/test)
                split_tag = source_sample.tags[0] if source_sample.tags else "train"
                
                # Créer les dossiers de destination
                dest_images_dir = augmented_dir / split_tag / "images"
                dest_labels_dir = augmented_dir / split_tag / "labels"
                dest_images_dir.mkdir(parents=True, exist_ok=True)
                dest_labels_dir.mkdir(parents=True, exist_ok=True)
                
                # Générer un nom unique pour l'image augmentée
                aug_filename = f"{source_path.stem}_aug_{generated}_{uuid.uuid4().hex[:4]}{source_path.suffix}"
                dest_img_path = dest_images_dir / aug_filename
                dest_label_path = dest_labels_dir / f"{source_path.stem}_aug_{generated}_{uuid.uuid4().hex[:4]}.txt"
                
                try:
                    # Charger et augmenter l'image
                    with Image.open(source_path) as img:
                        augmented_img = self._apply_augmentation(img)
                        augmented_img.save(dest_img_path)
                    
                    # Copier le file de labels
                    original_labels_dir = source_path.parent.parent / "labels"
                    source_label_path = original_labels_dir / f"{source_path.stem}.txt"

                    if source_label_path.exists():
                        shutil.copy(source_label_path, dest_label_path)
                    else:
                        with open(dest_label_path, 'w') as f:
                            pass
                    
                    # Créer un nouveau sample dans FiftyOne
                    new_sample = fo.Sample(filepath=str(dest_img_path))
                    new_sample.tags = [split_tag]
                    
                    # Copier les détections du sample source
                    if source_sample[label_field]:
                        new_sample[label_field] = source_sample[label_field].copy()
                    
                    self.dataset.add_sample(new_sample)
                    generated += 1
                    total_generated += 1
                    
                except Exception as e:
                    print(f"   Error lors de l'augmentation : {e}")
                    continue
            
            print(f"   -> {generated} images générées pour '{cls}'")
        
        self.dataset.compute_metadata()
        self.dataset.save()
        
        print(f"\nAugmentation terminée ! {total_generated} images générées au total.")
        print(f"Les images augmentées sont dans : {augmented_dir}")
        print("/!\\ IMPORTANT : Les images augmentées sont temporaires. Exportez le dataset pour les conserver.")
    
    def _apply_augmentation(self, img):
        """Applique des transformations aléatoires à une image."""
        # Convertir en RGB si nécessaire
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Choisir aléatoirement 1-3 transformations
        num_transforms = random.randint(1, 3)
        transforms = random.sample([
            'rotate', 'flip_h', 'flip_v', 'brightness', 'contrast', 'color'
        ], num_transforms)
        
        for transform in transforms:
            if transform == 'rotate':
                angle = random.uniform(-15, 15)
                img = img.rotate(angle, fillcolor=(0, 0, 0))
            elif transform == 'flip_h':
                img = ImageOps.mirror(img)
            elif transform == 'flip_v':
                img = ImageOps.flip(img)
            elif transform == 'brightness':
                enhancer = ImageEnhance.Brightness(img)
                factor = random.uniform(0.7, 1.3)
                img = enhancer.enhance(factor)
            elif transform == 'contrast':
                enhancer = ImageEnhance.Contrast(img)
                factor = random.uniform(0.8, 1.2)
                img = enhancer.enhance(factor)
            elif transform == 'color':
                enhancer = ImageEnhance.Color(img)
                factor = random.uniform(0.8, 1.2)
                img = enhancer.enhance(factor)
        
        return img




    def _singularize(self, word):
        """
        Convertit un mot au singulier (règles basiques anglais).
        """
        word = word.lower()
        
        # Check simple "s" removal against keywords first
        if word.endswith('s') and word[:-1] in GENERIC_FOOD_KEYWORDS:
             return word[:-1]

        if word.endswith('ies') and len(word) > 3:
             # Check if it's ie + s (cookie -> cookies)
             if word[:-1] in GENERIC_FOOD_KEYWORDS:
                 return word[:-1]
             return word[:-3] + 'y'
        elif word.endswith('ss'):
             return word
        elif word.endswith('es'):
            if word[:-2].endswith('s') or word[:-2].endswith('sh') or word[:-2].endswith('ch') or word[:-2].endswith('x') or word[:-2].endswith('z'):
                return word[:-2]
            elif word.endswith('oes'): # Potatoes, tomatoes
                return word[:-2]
            else: # Grapes, cakes -> just remove s
                 return word[:-1]
        elif word.endswith('s') and len(word) > 1:
            return word[:-1]
        return word

    def generalize_classes(self):
        """
        Généralise les noms des classes en utilisant une liste de mots-clés prédéfinis.
        Ex: 'long pepperoni pizza' -> 'pizza'
        Ex: 'adnams ghost ship beer' -> 'beer'
        Applique également une singulariation (apples -> apple).
        """
        print("\n" + "="*70)
        print("🔄 GÉNÉRALISATION DES CLASSES".center(70))
        print("="*70)
        
        # Utilise la liste globale GENERIC_FOOD_KEYWORDS
        keywords = GENERIC_FOOD_KEYWORDS
        
        label_field = self.label_field
        classes = self.dataset.default_classes or sorted(self.dataset.distinct(f"{label_field}.detections.label"))
        
        if not classes:
            print("❌ None class trouvée.")
            return

        print(f"📊 Analysis de {len(classes)} classes...")
        
        # Identifier les changements
        mapping = {}
        changes = []
        
        for cls in classes:
            old_name = cls.lower()
            new_name = None
            
            # 1. Chercher un mot-clé exact ou partiel
            for kw in keywords:
                if re.search(r'\b' + re.escape(kw) + r'\b', old_name):
                    new_name = kw
                    break
            
            # 2. Si pas de mot-clé, essayer de singulariser le dernier mot
            if not new_name:
                words = old_name.split()
                if words:
                    last_word = words[-1]
                    singular_last = self._singularize(last_word)
                    if singular_last != last_word:
                        # Reconstruire le nom avec le dernier mot singularisé
                        new_name = " ".join(words[:-1] + [singular_last])
                    else:
                        new_name = cls # Garder l'original si pas de changement
                else:
                    new_name = cls
            
            # 3. Appliquer singularisation sur le mot clé trouvé aussi (double check)
            if new_name:
                 new_name = self._singularize(new_name)

            mapping[cls] = new_name
            if cls != new_name:
                changes.append((cls, new_name))
        
        if not changes:
            print("\nℹ️  None généralisation possible détectée avec la liste actuelle.")
            return
        
        # Afficher les propositions
        print(f"\n✅ {len(changes)} généralisations proposées :")
        print(f"{'ANCIEN NOM':<40} -> {'NOUVEAU NOM'}")
        print("-" * 60)
        
        # Grouper pour l'affichage
        grouped = {}
        for old, new in changes:
            if new not in grouped: grouped[new] = []
            grouped[new].append(old)
            
        for new, olds in sorted(grouped.items()):
            for old in sorted(olds):
                print(f"{old:<40} -> {new}")
        
        print("-" * 60)
        
        confirm = input("\nAppliquer ces changements ? (o/n) : ").strip().lower()
        if confirm != 'o':
            print("❌ Cancelled.")
            return
        
        print("\n🔄 Application des changements...")
        this_mapping = {old: new for old, new in mapping.items() if old != new}
        
        # Appliquer les renames
        count = 0
        for sample in self.dataset.iter_samples(autosave=True):
            if sample[label_field] and sample[label_field].detections:
                modified = False
                for det in sample[label_field].detections:
                    if det.label in this_mapping:
                        det.label = this_mapping[det.label]
                        modified = True
                if modified:
                    count += 1
        
        # Mettre à jour la liste des classes par défaut
        new_classes = sorted(list(set(mapping.values())))
        self.dataset.default_classes = new_classes
        self.dataset.save()
        
        print(f"\n✅ Completed ! {count} images mises à jour.")
        print(f"📊 Nombre de classes final : {len(new_classes)}")

    def analyze_dataset_quality(self):
        """
        Analysis la qualité du dataset et de chaque class.
        Affiche des statistiques détaillées, scores de qualité et recommandations.
        """
        print("\n" + "="*80)
        print("📊 ANALYSE DE QUALITÉ DU DATASET".center(80))
        print("="*80)
        print()
        
        label_field = self.label_field
        classes = self.dataset.default_classes or self.dataset.distinct(f"{label_field}.detections.label")
        
        if not classes:
            print("❌ None class trouvée dans le dataset")
            return
        
        # Statistics globales
        total_images = len(self.dataset)
        total_detections = sum(
            len(sample[label_field].detections) if sample[label_field] else 0
            for sample in self.dataset
        )
        
        print(f"📈 STATISTIQUES GLOBALES")
        print(f"   Total d'images: {total_images}")
        print(f"   Total de détections: {total_detections}")
        print(f"   Nombre de classes: {len(classes)}")
        print(f"   Détections par image (moyenne): {total_detections/total_images if total_images > 0 else 0:.2f}")
        print()
        
        # Analyser chaque class
        class_stats = {}
        
        print(f"🔍 ANALYSE PAR CLASSE")
        print()
        
        for cls in classes:
            view = self.dataset.match(F(f"{label_field}.detections.label").contains(cls))
            num_images = len(view)
            
            # Compter les détections et calculer les tailles de boxes
            num_boxes = 0
            box_areas = []
            
            for sample in view:
                if sample[label_field] and sample[label_field].detections:
                    for det in sample[label_field].detections:
                        if det.label == cls:
                            num_boxes += 1
                            # Bounding box format: [x, y, width, height] (normalized)
                            bbox = det.bounding_box
                            width, height = bbox[2], bbox[3]
                            area = width * height
                            box_areas.append(area)
            
            # Calculer les statistiques
            avg_area = sum(box_areas) / len(box_areas) if box_areas else 0
            min_area = min(box_areas) if box_areas else 0
            max_area = max(box_areas) if box_areas else 0
            
            # Évaluer la qualité
            quality_score = 0
            issues = []
            
            # Critère 1: Number of images
            if num_images >= 300:
                quality_score += 40
                quality_level_images = "🌟 Excellent"
            elif num_images >= 100:
                quality_score += 30
                quality_level_images = "✅ Bon"
            elif num_images >= 50:
                quality_score += 20
                quality_level_images = "⚠️  Moyen"
                issues.append(f"Seulement {num_images} images (recommandé: ≥100)")
            else:
                quality_score += 10
                quality_level_images = "❌ Faible"
                issues.append(f"Très peu d'images ({num_images}, recommandé: ≥100)")
            
            # Critère 2: Taille moyenne des boxes
            if avg_area >= 0.1:  # >= 10% de l'image
                quality_score += 30
                quality_level_size = "🌟 Excellent"
            elif avg_area >= 0.05:  # >= 5%
                quality_score += 25
                quality_level_size = "✅ Bon"
            elif avg_area >= 0.01:  # >= 1%
                quality_score += 15
                quality_level_size = "⚠️  Moyen"
                issues.append(f"Boxes petites (moy: {avg_area*100:.1f}%)")
            else:
                quality_score += 5
                quality_level_size = "❌ Faible"
                issues.append(f"Boxes très petites (moy: {avg_area*100:.2f}%, recommandé: ≥5%)")
            
            # Critère 3: Nombre de boxes par image
            boxes_per_image = num_boxes / num_images if num_images > 0 else 0
            if boxes_per_image >= 1.5:
                quality_score += 15
                quality_level_density = "✅ Bon"
            elif boxes_per_image >= 1.0:
                quality_score += 10
                quality_level_density = "⚠️  Moyen"
            else:
                quality_score += 5
                quality_level_density = "❌ Faible"
                issues.append(f"Peu de boxes par image ({boxes_per_image:.1f})")
            
            # Critère 4: Variabilité des tailles
            if box_areas:
                std_area = (sum((a - avg_area)**2 for a in box_areas) / len(box_areas))**0.5
                if std_area > avg_area * 0.3:  # Bonne variabilité
                    quality_score += 15
                    quality_level_variety = "✅ Bon"
                else:
                    quality_score += 10
                    quality_level_variety = "⚠️  Moyen"
                    issues.append("Peu de variabilité dans les tailles")
            else:
                quality_level_variety = "❌ N/A"
            
            # Qualité globale
            if quality_score >= 80:
                overall_quality = "🌟 EXCELLENT"
            elif quality_score >= 60:
                overall_quality = "✅ BON"
            elif quality_score >= 40:
                overall_quality = "⚠️  MOYEN"
            else:
                overall_quality = "❌ FAIBLE"
            
            class_stats[cls] = {
                'num_images': num_images,
                'num_boxes': num_boxes,
                'avg_area': avg_area,
                'min_area': min_area,
                'max_area': max_area,
                'boxes_per_image': boxes_per_image,
                'quality_score': quality_score,
                'overall_quality': overall_quality,
                'issues': issues
            }
        
        # Trier par score de qualité (décroissant)
        sorted_classes = sorted(class_stats.items(), key=lambda x: x[1]['quality_score'], reverse=True)
        
        # Afficher le tableau
        print(f"{'Class':<30} {'Images':<10} {'Boxes':<10} {'Taille moy.':<15} {'Qualité'}")
        print("-" * 80)
        
        for cls, stats in sorted_classes:
            print(f"{cls:<30} {stats['num_images']:<10} {stats['num_boxes']:<10} "
                  f"{stats['avg_area']*100:>6.2f}%        {stats['overall_quality']}")
        
        print()
        print("="*80)
        print("💡 RECOMMANDATIONS".center(80))
        print("="*80)
        print()
        
        # Classes problématiques
        problematic = [(cls, stats) for cls, stats in class_stats.items() if stats['quality_score'] < 60]
        if problematic:
            print("⚠️  CLASSES À AMÉLIORER:")
            print()
            for cls, stats in sorted(problematic, key=lambda x: x[1]['quality_score']):
                print(f"   📌 {cls} (Score: {stats['quality_score']}/100)")
                for issue in stats['issues']:
                    print(f"      - {issue}")
                print()
        
        # Classes excellentes
        excellent = [(cls, stats) for cls, stats in class_stats.items() if stats['quality_score'] >= 80]
        if excellent:
            print("🌟 CLASSES EXCELLENTES:")
            for cls, stats in excellent:
                print(f"   ✅ {cls} - {stats['num_images']} images, {stats['avg_area']*100:.1f}% taille moyenne")
            print()
        
        # Recommandations générales
        print("📋 ACTIONS RECOMMANDÉES:")
        print()
        
        # Classes avec trop peu d'images
        low_images = [(cls, stats['num_images']) for cls, stats in class_stats.items() if stats['num_images'] < 100]
        if low_images:
            print("   1. Augmenter les données pour ces classes (< 100 images):")
            for cls, count in sorted(low_images, key=lambda x: x[1]):
                print(f"      - {cls}: {count} images → Recommandé: ≥100")
            print(f"      💡 Utilisez: manager.augment_classes_to_target(100)")
            print()
        
        # Classes avec boxes très petites
        small_boxes = [(cls, stats['avg_area']) for cls, stats in class_stats.items() if stats['avg_area'] < 0.01]
        if small_boxes:
            print("   2. Vérifier les annotations pour ces classes (boxes < 1%):")
            for cls, area in sorted(small_boxes, key=lambda x: x[1]):
                print(f"      - {cls}: {area*100:.2f}% en moyenne")
            print(f"      💡 Vérifiez la qualité des annotations ou supprimez si non pertinent")
            print()
        
        # Déséquilibre
        if class_stats:
            max_images = max(stats['num_images'] for stats in class_stats.values())
            min_images = min(stats['num_images'] for stats in class_stats.values())
            if max_images > min_images * 5:  # Ratio > 5
                print("   3. Dataset déséquilibré détecté:")
                print(f"      - Class max: {max_images} images")
                print(f"      - Class min: {min_images} images")
                print(f"      - Ratio: {max_images/min_images if min_images > 0 else 'N/A':.1f}x")
                print(f"      💡 Utilisez: manager.balance_classes(max_samples)")
                print()
        
        print("="*80)
        print()

    def export_yolo(self, export_dir):
        """Exporte en YOLOv8 avec structure APLATIE : train/images, valid/images."""
        path = pathlib.Path(export_dir).resolve()
        
        if self.dataset_dir:
            source_path = pathlib.Path(self.dataset_dir).resolve()
            if path == source_path or source_path.is_relative_to(path):
                print("\n/!\\ CRITICAL ERROR : Folder d'export identique au folder source /!\\")
                return

        if path.exists():
            print(f"Nettoyage du folder d'export existant : {path}")
            shutil.rmtree(path, onerror=self._handle_remove_readonly)
        
        tags = self.dataset.count_sample_tags()
        print(f"Tags trouvés : {tags}")
        
        valid_tag = "valid" if "valid" in tags and tags["valid"] > 0 else "val"
        splits_map = {
            "train": "train",
            valid_tag: "valid", 
            "test": "test"
        }

        classes = self.dataset.default_classes or sorted(self.dataset.distinct(f"{self.label_field}.detections.label"))
        classes = list(classes) 
        
        exported_count = 0
        print(f"Export structurée vers {path}...")
        
        for tag, folder_name in splits_map.items():
            if tag in tags and tags[tag] > 0:
                print(f" -> Export du split '{tag}' vers folder '{folder_name}'...")
                split_view = self.dataset.match_tags(tag)
                
                target_split_dir = path / folder_name
                
                # Export initial par FiftyOne (crée potentiellement images/train/)
                split_view.export(
                    export_dir=str(target_split_dir),
                    dataset_type=fo.types.YOLOv5Dataset,
                    label_field=self.label_field,
                    classes=classes,
                )
                
                # --- CORRECTION STRUCTURE (Aplatissement) ---
                for type_dir in ["images", "labels"]:
                    target_type_dir = target_split_dir / type_dir
                    if target_type_dir.exists():
                        subdirs = [d for d in target_type_dir.iterdir() if d.is_dir()]
                        for subdir in subdirs:
                            for file_path in subdir.iterdir():
                                dest_path = target_type_dir / file_path.name
                                if not dest_path.exists():
                                    shutil.move(str(file_path), str(dest_path))
                            try: subdir.rmdir()
                            except: pass

                sub_yaml = target_split_dir / "dataset.yaml"
                if sub_yaml.exists(): os.remove(sub_yaml)
                
                exported_count += 1

        if exported_count == 0:
            print("\n/!\\ ATTENTION : Aucun split détecté. Mode SECOURS -> Tout dans 'train/'.")
            target_split_dir = path / "train"
            self.dataset.export(
                export_dir=str(target_split_dir),
                dataset_type=fo.types.YOLOv5Dataset,
                label_field=self.label_field,
                classes=classes,
            )
            
            for type_dir in ["images", "labels"]:
                target_type_dir = target_split_dir / type_dir
                if target_type_dir.exists():
                    subdirs = [d for d in target_type_dir.iterdir() if d.is_dir()]
                    for subdir in subdirs:
                        for file_path in subdir.iterdir():
                            dest_path = target_type_dir / file_path.name
                            if not dest_path.exists(): shutil.move(str(file_path), str(dest_path))
                        try: subdir.rmdir()
                        except: pass

            sub_yaml = target_split_dir / "dataset.yaml"
            if sub_yaml.exists(): os.remove(sub_yaml)
            yaml_fallback = True
        else:
            yaml_fallback = False

        print("Génération du file 'data.yaml'...")
        yaml_content = {
            "train": "train/images",
            "val": "valid/images", 
            "test": "test/images",
            "nc": len(classes),
            "names": classes
        }
        
        if yaml_fallback:
            yaml_content["val"] = "train/images" 
            if "test" in yaml_content: del yaml_content["test"]
        else:
            if "test" not in tags or tags["test"] == 0:
                if "test" in yaml_content: del yaml_content["test"]
            if valid_tag not in tags or tags[valid_tag] == 0:
                if "val" in yaml_content: del yaml_content["val"]

        path.mkdir(parents=True, exist_ok=True)
        with open(path / "data.yaml", 'w') as f:
            yaml.dump(yaml_content, f, sort_keys=False, default_flow_style=None)

        print("-" * 40)
        print("Export terminée avec success !")
        print(f"File de configuration : {path / 'data.yaml'}")
        print("-" * 40)

    def launch_app(self):
        print("Lancement App...")
        session = fo.launch_app(self.dataset)
        input("Appuyez sur Entrée pour fermer...")
        session.close()

def handle_class_menu(manager):
    """Dedicated menu for class management."""
    while True:
        print("\n" + "="*70)
        print("🏷️ CLASS MANAGEMENT".center(70))
        print("="*70)
        print()
        print("  [1] Delete classes")
        print("  [2] Merge similar classes")
        print("  [3] Generalize names (ex: 'long pizza' -> 'pizza')")
        print("  [4] Remove annotations (backgrounds)")
        print("  [5] Normalize names (lowercase, no special chars)")
        print("  [6] Convert to lowercase")
        print("  [7] Compare with a list")
        print("  [8] Rename a class")
        print("  [0] Back to Main Menu")
        print()
        
        choice = input("👉 Choice: ").strip()
        
        if choice == "0":
            break
            
        elif choice == "1":
            print("\n" + "="*70)
            print("🗑️  CLASS DELETION".center(70))
            print("="*70)
            print("\n[1] Delete a single class")
            print("[2] Delete by image count (with comparator)")
            print("[3] Delete a list of classes")
            print("[4] KEEP only specific classes")
            print("[0] Back")
            
            sub = input("\n👉 Choice: ").strip()
            
            if sub == "1":
                cls = input("   Class name: ").strip()
                del_empty = input("   Delete empty images? (y/n) [y]: ").strip().lower()
                manager.delete_class(cls, delete_empty_images=(del_empty != 'n'))
            
            elif sub == "2":
                try:
                    n = int(input("   Image count threshold: "))
                    print("\n   Comparator:")
                    print("   [1] < (less than)")
                    print("   [2] <= (less than or equal)")
                    print("   [3] == (equal)")
                    print("   [4] >= (greater than or equal)")
                    print("   [5] > (greater than)")
                    comp_map = {'1':'lt', '2':'le', '3':'eq', '4':'ge', '5':'gt'}
                    comp_choice = input("   Choice [2]: ").strip()
                    comp = comp_map.get(comp_choice, 'le')
                    del_empty = input("   Delete empty images? (y/n) [y]: ").strip().lower()
                    manager.delete_classes_by_image_count(n, comparator=comp, delete_empty_images=(del_empty != 'n'))
                except ValueError:
                    print("❌ Error: Enter a valid integer")
            
            elif sub == "3":
                lst = input("   Class list (comma-separated): ").strip()
                del_empty = input("   Delete empty images? (y/n) [y]: ").strip().lower()
                manager.delete_multiple_classes(lst, delete_empty_images=(del_empty != 'n'))
            
            elif sub == "4":
                lst = input("   Classes to KEEP (comma-separated): ").strip()
                del_empty = input("   Delete empty images? (y/n) [y]: ").strip().lower()
                manager.keep_only_classes(lst, delete_empty_images=(del_empty != 'n'))
        
        elif choice == "2":
            print("\n🔄 Merging similar classes...")
            manager.merge_similar_classes()
            
        elif choice == "3":
            manager.generalize_classes()
            
        elif choice == "4":
            print("\n🗑️  Removing annotations")
            cls = input("   Class to remove: ").strip()
            keep = input("   Keep images? (y/n) [y]: ").strip().lower()
            manager.remove_class_annotations(cls, keep_images=(keep != 'n'))
            
        elif choice == "5":
            manager.normalize_class_names()
            
        elif choice == "6":
            manager.convert_classes_to_lowercase()
            
        elif choice == "7":
            manager.compare_classes()
            
        elif choice == "8":
            print("\n✏️  Rename a class")
            old_name = input("   Current class name: ").strip()
            new_name = input("   New class name: ").strip()
            
            if old_name and new_name:
                update_files = input("   Update label files? (y/n) [y]: ").strip().lower()
                manager.rename_class(old_name, new_name, update_files=(update_files != 'n'))
            else:
                print("❌ Error: Both names are required.")
            
        else:
            print("❌ Invalid choice")

def main():
    """Main menu for YOLO dataset configurator."""
    print("="*70)
    print("🎯 YOLOv8 DATASET CONFIGURATOR".center(70))
    print("="*70)
    
    # Initial dataset loading
    ds_name = input("\n📂 Dataset name (ex: 'food_v7'): ").strip()
    ds_dir = None
    force_reload = False

    if fo.dataset_exists(ds_name):
        print(f"✅ Dataset '{ds_name}' detected in cache.")
        reload_choice = input("   Use cache (c) or Reload from disk (r)? [c/r]: ").strip().lower()
        if reload_choice == 'r':
            force_reload = True
            ds_dir = input("   Source folder path: ").strip().replace('"', '')
    else:
        print(f"📁 New dataset '{ds_name}'")
        ds_dir = input("   Source folder path: ").strip().replace('"', '')

    try:
        manager = YoloDatasetManager(ds_name, ds_dir, force_reload=force_reload)
        print(f"\n✅ Dataset loaded: {len(manager.dataset)} images")
    except Exception as e:
        print(f"\n❌ Error during loading: {e}")
        return

    # Main menu loop
    while True:
        print("\n" + "="*70)
        print(f"📊 Current dataset: {manager.dataset_name}".center(70))
        print("="*70)
        print()
        print("┌─ 📊 VISUALIZATION")
        print("│  [1] Dataset summary")
        print("│  [2] Statistics per class")
        print("│  [3] Launch FiftyOne (interactive visualization)")
        print("│  [4] View class distribution")
        print("│  [5] Analyze quality")
        print("│")
        print("├─ ✏️  DATASET MODIFICATIONS")
        print("│  [10] Manage classes (delete, rename, generalize...)")
        print("│  [11] Clone dataset")
        print("│  [12] Split dataset (train/val/test)")
        print("│  [13] Merge external dataset")
        print("│  [14] Reduce / Balance data")
        print("│  [15] Augment data (data augmentation)")
        print("│  [16] Resize images")
        print("│  [17] Export YOLO dataset")
        print("│  [20] Add background images (negatives)")
        print("│  [21] Merge entire folder of datasets")
        print("│")
        print("├─ 🔄 SYSTEM")
        print("│  [30] Load another dataset")
        print("│  [31] Quit")
        print("└─")
        print()

        try:
            choice = input("👉 Your choice: ").strip()
            
            # ========== VISUALIZATION ==========
            if choice == "1":
                manager.get_summary()
            elif choice == "2":
                manager.count_images_per_class()
            elif choice == "3":
                print("\n🚀 Launching FiftyOne...")
                manager.launch_app()
            elif choice == "4":
                manager.visualize_class_distribution()
            elif choice == "5":
                print("\n📊 Analyzing dataset quality...")
                manager.analyze_dataset_quality()
            
            # ========== DATASET MODIFICATIONS ==========
            elif choice == "10":
                handle_class_menu(manager)
            
            elif choice == "11":
                nm = input("\n📋 Clone name: ").strip()
                switch = input("   Switch to clone? (y/n) [n]: ").strip().lower()
                new_ds = manager.clone_dataset(nm)
                if switch == 'y':
                    manager.dataset = new_ds
                    manager.dataset_name = nm
                    print(f"✅ Switched to '{nm}'")
            
            elif choice == "12":
                print("\n✂️  Splitting dataset...")
                manager.split_dataset()
            
            elif choice == "13":
                merge_path = input("\n📁 Path to dataset to merge: ").strip().replace('"', '')
                manager.merge_external_dataset(merge_path)
            
            elif choice == "14":
                print("\n" + "="*70)
                print("⚖️  REDUCTION / BALANCING".center(70))
                print("="*70)
                print("\n[1] Reduce a specific class")
                print("[2] Balance all classes (set a max)")
                print("[0] Back")
                
                sub_c = input("\n👉 Choice: ").strip()
                
                if sub_c == "1":
                    cls = input("   Target class: ").strip()
                    try:
                        nb = int(input("   Number of images to remove: "))
                        manager.reduce_class_samples(cls, nb)
                    except ValueError:
                        print("❌ Error: Enter an integer")
                
                elif sub_c == "2":
                    try:
                        max_nb = int(input("   MAX number of images per class: "))
                        manager.balance_classes(max_nb)
                    except ValueError:
                        print("❌ Error: Enter an integer")
                
                elif sub_c == "0":
                    continue
                else:
                    print("❌ Invalid choice")
            
            elif choice == "15":
                try:
                    target = int(input("\n📈 TARGET number of images per class: "))
                    manager.augment_classes_to_target(target)
                except ValueError:
                    print("❌ Error: Enter an integer")
            
            elif choice == "16":
                fmt = input("   Format (ex: 640x640): ").strip()
                try:
                    w, h = map(int, fmt.lower().split('x'))
                    manager.resize_images(w, h)
                except ValueError:
                    print("❌ Invalid format. Use 'WIDTHxHEIGHT' (ex: 640x640)")
            
            elif choice == "17":
                export_path = input("\n📦 Export folder: ").strip().replace('"', '')
                manager.export_yolo(export_path)

            elif choice == "20":
                path = input("\n📂 Folder containing background images: ").strip().replace('"', '')
                manager.add_background_images(path)

            elif choice == "21":
                path = input("\n📂 Folder containing datasets to merge: ").strip().replace('"', '')
                manager.merge_datasets_from_folder(path)
            
            # ========== SYSTEM ==========
            elif choice == "30":
                print("\n🔄 Loading another dataset...")
                new_name = input("   Dataset name: ").strip()
                
                if fo.dataset_exists(new_name):
                    print(f"   ✅ Dataset '{new_name}' found in cache")
                    reload = input("   Reload from disk? (y/n) [n]: ").strip().lower()
                    new_dir = None
                    if reload == 'y':
                        new_dir = input("   Source path: ").strip().replace('"', '')
                        force_reload = True
                    else:
                        force_reload = False
                else:
                    new_dir = input("   Source path: ").strip().replace('"', '')
                    force_reload = False
                
                try:
                    manager = YoloDatasetManager(new_name, new_dir, force_reload=force_reload)
                    print(f"\n✅ Dataset '{new_name}' loaded: {len(manager.dataset)} images")
                except Exception as e:
                    print(f"\n❌ Error: {e}")
            
            elif choice == "31":
                print("\n👋 Goodbye!")
                break
            
            else:
                print("\n❌ Invalid choice. Please enter a valid number.")
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Interruption detected")
            if input("Do you want to quit? (y/n): ").strip().lower() == 'y':
                break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Program continues...")


if __name__ == "__main__":
    main()