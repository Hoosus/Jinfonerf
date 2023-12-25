import torch

from data import load_lego
from utils import get_datetime_str



if __name__ == "__main__":

  lego_path = './data/lego'
  log_path = f"./logs/lego_{get_datetime_str()}"
  
  dataset = {
    'train' : load_lego(lego_path, 'train', samples=[26, 86, 2, 55]),
    'val' : load_lego(lego_path, 'val', skip=8),
    'test' : load_lego(lego_path, 'test', skip=8)
  }
  
