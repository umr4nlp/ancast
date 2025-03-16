#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""misc. utils"""

import codecs
import json
import logging
import os

logger = logging.getLogger(__name__)


### I/O
def get_fpath(fpath_or_dir, fname=None):
  fpath = fpath_or_dir
  if fname is not None:
    if not os.path.isdir(fpath_or_dir):
      fpath_or_dir = os.path.dirname(fpath_or_dir)
    fpath = os.path.join(fpath_or_dir, fname)
  return fpath

# txt
def load_txt(fpath_or_dir, fname=None, encoding=None, delimiter=None):
  fpath = get_fpath(fpath_or_dir, fname)
  with codecs.open(fpath, encoding=encoding, errors='ignore') as f:
    data = f.read()
  if delimiter is not None:  # drop empty strings
    data = list(filter(None, data.split(delimiter)))
  return data

def save_txt(obj, fpath_or_dir, fname=None, delimiter=None):
  fpath = get_fpath(fpath_or_dir, fname)
  with open(fpath, 'w') as f:
    if delimiter is not None:
      obj = delimiter.join(obj)
    f.write(obj)
  return fpath

# json
def load_json(fpath_or_dir, fname=None):
  fpath = get_fpath(fpath_or_dir, fname)
  with open(fpath) as f:
    obj = json.loads(f.read())
  return obj

def save_json(obj, fpath_or_dir, fname=None, indent=4):
  fpath = get_fpath(fpath_or_dir, fname)
  with open(fpath, 'w') as f:
    json.dump(obj, f, indent=indent)
  return fpath

# csv
def csv_is_empty(fpath):
  if os.path.exists(fpath) and os.path.getsize(fpath) > 0:
    with open(fpath, 'r', encoding='utf-8') as f:
      first_line = f.readline()
      return first_line.strip() == ''
  return True

