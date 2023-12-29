import torch
import yaml
from data import load_lego
from utils import get_datetime_str, parser_parse_args



if __name__ == "__main__":
  args = parser_parse_args()
  config_path = args.config
  config = yaml.safe_load(open(config_path))
  print(config)
  name = config["name"]
  assert name in ["lego"]
  data_path = config["data_path"]
  exp_name = config["exp_name"]
  log_path = f"./logs/{exp_name}_{get_datetime_str()}"

  if name == "lego":
    dataset = {
      'train' : load_lego(data_path, 'train', samples=[26, 86, 2, 55]),
      'val' : load_lego(data_path, 'val', skip=8),
      'test' : load_lego(data_path, 'test', skip=8)
    }
  
  
  
