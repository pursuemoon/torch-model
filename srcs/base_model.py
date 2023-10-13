# -- coding: utf-8 --

import torch.nn as nn
import signal

from utils.log import logger

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.is_being_stoped = False
        self.is_trained = False
        signal.signal(signal.SIGINT, self.__sigint_handler)

    def __sigint_handler(self, signum, frame):
        self.is_being_stoped = True
        logger.info('Training is being early stoped.')