import json
import os
import imageio
import numpy as np

def load_lego(path, split, samples=None, skip=1):
  assert split in ['train', 'val', 'test']
  print(f"loading lego {split} split")
  with open(os.path.join(path, f"transforms_{split}.json"), "r") as f:
    transforms = json.load(f)
  dataset = {}

  if samples is None:
    samples = list(range(0, len(transforms['frames']), skip))
  
  imgs = []
  poses = []
  for sample in samples:
    frame = transforms['frames'][sample]
    img = imageio.imread(os.path.join(path, frame['file_path'] + ".png"))
    imgs.append(img[None, ...])
    poses.append(np.array(frame['transform_matrix'])[None, ...])

  imgs = np.concatenate(imgs, 0)
  poses = np.concatenate(poses, 0)

  H, W = imgs.shape[-3:-1]
  focal = 0.5 * W / np.tan(0.5 * float(transforms['camera_angle_x']))

  dataset['imgs'] = imgs
  dataset['poses'] = poses
  dataset['camera_setting'] = {'resolution': (H, W), 'focal': focal}

  print(f"loaded {imgs.shape[0]} samples of shape ({H}, {W}), focal={focal}")

  return dataset

  




