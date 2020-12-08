import os
import shutil
from tensorpack.callbacks import Callback
from tensorpack.utils import logger


import moxing as mox
mox.file.shift('os', 'mox')

class Sync(Callback):
    """
    Save the model once triggered.
    """

    def __init__(self, local_path,obs_path):
       self.local_path= local_path
       self.obs_path = obs_path

    def _trigger(self):
        shutil.copytree(self.local_path,self.obs_path)
        logger.info("Sync the files...")