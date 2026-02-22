from .BinFill import *
from .PickXtimes import *
from .SwingXtimes import *
from .ButtonUnmask import *
from .VideoUnmask import *
from .PickHighlight import *
from .VideoUnmaskSwap import *
from .VideoRepick import *
from .VideoPlaceButton import *
from .VideoPlaceOrder import *
from .ButtonUnmaskSwap import *
from .InsertPeg import *
from .MoveCube import *
from .PatternLock import *
from .StopCube import *
from .RouteStick import *



import warnings
import logging
import os

def suppress_warnings():
    # Suppress specific warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")
    warnings.filterwarnings("ignore", message=".*env.task_list.*")
    warnings.filterwarnings("ignore", message=".*env.elapsed_steps.*")
    warnings.filterwarnings("ignore", message=".*not in the task's list of supported robots.*")
    warnings.filterwarnings("ignore", message=".*No initial pose set for actor builder.*")

    warnings.filterwarnings("ignore", category=UserWarning, module="mani_skill")

    # Suppress ManiSkill warnings - comprehensive approach
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

    # Set up logging to suppress all warnings
    logging.basicConfig(level=logging.CRITICAL)
    logging.getLogger("mani_skill").setLevel(logging.CRITICAL)
    logging.getLogger("mani_skill").propagate = False
    
suppress_warnings()