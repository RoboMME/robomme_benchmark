from remote_evaluation.server import PolicyServer
from remote_evaluation.policy import DummyPolicy



###### Participant Parameters ######
# You will submit a json file including the following parameters at eval.ai 
ACTION_SPACE = "joint_angle"
USE_DEPTH = False
USE_CAMERA_PARAMS = False
HOST = "141.212.115.116"
PORT = 8012
CHUNK_SIZE = 8
####################################


policy = DummyPolicy()
server = PolicyServer(policy, host=HOST, port=PORT)
server.serve_forever()