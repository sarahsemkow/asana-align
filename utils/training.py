import os
import random
import shutil


def split_train_test(directory, train_ratio=0.8):
  subdirectories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

  for subdir in subdirectories:
    subdir_path = os.path.join(directory, subdir)
    renamed_subdir = subdir.replace('_stick', '')

    files = [f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))]

    random.shuffle(files)

    split_index = int(len(files) * train_ratio)

    train_files = files[:split_index]
    test_files = files[split_index:]
    
    train_dir = os.path.join('data', 'train', renamed_subdir)
    test_dir = os.path.join('data', 'test', renamed_subdir)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for file in train_files:
      src = os.path.join(subdir_path, file)
      dst = os.path.join(train_dir, file)
      shutil.copy(src, dst)

    for file in test_files:
      src = os.path.join(subdir_path, file)
      dst = os.path.join(test_dir, file)
      shutil.copy(src, dst)

split_train_test('./sticks_dataset')