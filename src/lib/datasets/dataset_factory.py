from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .dataset.sche import JointDataset_Sche

def get_dataset(dataset, task):
  if task == 'mot_sche':
    return JointDataset_Sche
  else:
    return None
  
