import datetime
import argparse

def get_datetime_str():
  return datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

def parser_parse_args():
  parser = argparse.ArgumentParser(description="JInfoNerf")
  parser.add_argument('--config', nargs=1, type=str, default="./configs/lego.yaml")
  return parser.parse_args()