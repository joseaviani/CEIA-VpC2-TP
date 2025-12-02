import shutil
import argparse
import random
from pathlib import Path
from collections import Counter
from tqdm import tqdm


# ==========================================
# 1. Lógica de Conteo (Estadísticas)
# ==========================================
def count_classes_distribution(data_dir):
  """
  Recorre todos los labels y cuenta las ocurrencias de cada clase.
  """
  data_path = Path(data_dir)
  labels_dir = data_path / 'labels'

  if not labels_dir.exists():
    print(f"❌ Error: No existe la carpeta {labels_dir}")
    return

  print(f"📊 Analizando distribución de clases en: {labels_dir}")

  label_files = list(labels_dir.glob('*.txt'))
  class_counter = Counter()
  total_files_checked = 0

  for label_file in tqdm(label_files, desc="Leyendo etiquetas"):
    try:
      with open(label_file, 'r') as f:
        lines = f.readlines()
        if not lines: continue

        total_files_checked += 1
        for line in lines:
          parts = line.strip().split()
          if parts:
            class_id = int(parts[0])
            class_counter[class_id] += 1
    except Exception:
      continue

  print("\n" + "=" * 40)
  print(f"📑 REPORTE DEL DATASET ({total_files_checked} archivos no vacíos)")
  print("=" * 40)
  print(f"{'CLASE ID':<10} | {'CANTIDAD (Instancias)':<20}")
  print("-" * 35)

  # Ordenar por ID de clase
  for class_id in sorted(class_counter.keys()):
    count = class_counter[class_id]
    print(f"{class_id:<10} | {count:<20}")
  print("=" * 40 + "\n")


# ==========================================
# 2. Lógica de Detección (Solo Lectura)
# ==========================================
def find_samples_with_class(data_dir, class_id):
  """
  Devuelve lista de (imagen, label) que contienen la clase ID.
  """
  data_path = Path(data_dir)
  images_dir = data_path / 'images'
  labels_dir = data_path / 'labels'

  valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.webp'}
  found_pairs = []

  print(f"🔍 Buscando archivos con la clase {class_id}...")

  label_files = list(labels_dir.glob('*.txt'))

  for label_file in tqdm(label_files, desc="Escaneando"):
    has_class = False
    try:
      with open(label_file, 'r') as f:
        for line in f:
          parts = line.strip().split()
          if parts and int(parts[0]) == int(class_id):
            has_class = True
            break
    except:
      continue

    if has_class:
      stem = label_file.stem
      image_file = None
      for ext in valid_extensions:
        temp_img = images_dir / (stem + ext)
        if temp_img.exists():
          image_file = temp_img
          break

      if image_file:
        found_pairs.append((image_file, label_file))

  return found_pairs


# ==========================================
# 3. Lógica de Aumento por Factor
# ==========================================
def augment_samples(file_pairs, factor):
  """
  Aplica el factor de multiplicación.
  Factor 2.0 -> Crea 1 copia extra asegurada.
  Factor 2.5 -> Crea 1 copia asegurada + 50% chance de una 2da copia.
  """
  if factor <= 1.0:
    print("⚠️ El factor es <= 1.0. No se necesitan copias.")
    return

  # Calculamos cuántas copias adicionales necesitamos por archivo
  added_copies = factor - 1.0
  guaranteed_copies = int(added_copies)  # Parte entera (ej: 1.5 -> 1)
  probability_extra = added_copies % 1.0  # Parte decimal (ej: 1.5 -> 0.5)

  print(f"🚀 Iniciando Augmentation x{factor}")
  print(
    f"   Logic: {guaranteed_copies} copias aseguradas + {probability_extra * 100:.0f}% chance de copia extra por archivo.")

  created_count = 0

  for img_path, lbl_path in tqdm(file_pairs, desc="Generando copias"):

    # Determinar cuántas copias hacer para ESTE archivo
    num_copies_for_this_file = guaranteed_copies

    # Tirar los dados para la parte decimal
    if random.random() < probability_extra:
      num_copies_for_this_file += 1

    # Ejecutar las copias
    for i in range(num_copies_for_this_file):
      try:
        # Sufijo único para evitar colisiones: _aug_1, _aug_2, etc.
        suffix = f"_aug_{i + 1}"
        new_stem = f"{img_path.stem}{suffix}"

        new_img_path = img_path.parent / (new_stem + img_path.suffix)
        new_lbl_path = lbl_path.parent / (new_stem + lbl_path.suffix)

        shutil.copy2(img_path, new_img_path)
        shutil.copy2(lbl_path, new_lbl_path)
        created_count += 1
      except Exception as e:
        print(f"❌ Error copiando {img_path.stem}: {e}")

  print(f"\n✅ Proceso terminado. Se crearon {created_count} nuevos archivos.")


# ==========================================
# 4. Orquestador (Main)
# ==========================================
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Herramienta para gestionar Dataset YOLO.')

  # Subcomandos para que sea elegante: "count" o "augment"
  subparsers = parser.add_subparsers(dest='command', required=True, help='Comando a ejecutar')

  # --- Comando: COUNT ---
  parser_count = subparsers.add_parser('count', help='Cuenta cuántas instancias hay de cada clase')
  parser_count.add_argument('--path', type=str, required=True, help='Ruta a la carpeta train')

  # --- Comando: AUGMENT ---
  parser_aug = subparsers.add_parser('augment', help='Multiplica las muestras de una clase específica')
  parser_aug.add_argument('--path', type=str, required=True, help='Ruta a la carpeta train')
  parser_aug.add_argument('--class_id', type=int, required=True, help='ID de la clase a aumentar')
  parser_aug.add_argument('--factor', type=float, required=True,
                          help='Factor multiplicador (ej: 2.0 duplica, 1.5 aumenta 50%)')

  args = parser.parse_args()

  if args.command == 'count':
    count_classes_distribution(args.path)

  elif args.command == 'augment':
    # 1. Buscar
    samples = find_samples_with_class(args.path, args.class_id)
    print(f"📊 Se encontraron {len(samples)} imágenes base para la clase {args.class_id}.")

    # 2. Aumentar
    if len(samples) > 0:
      augment_samples(samples, args.factor)
    else:
      print("⚠️ No hay nada que aumentar.")