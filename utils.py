import datetime

def get_datetime_str():
  return datetime.datetime.now().strftime("date:%Y-%m-%d_%H:%M:%S")