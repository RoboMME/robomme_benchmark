"""
This is used by participants to serve their policy.
"""

from remote_evaluation.server import PolicyServer
from remote_evaluation.policy import DummyPolicy



###### Participant Parameters ######
# You will submit a json file including the following parameters at eval.ai 
HOST = "141.212.115.116"
PORT = 8001
####################################


policy = DummyPolicy()
server = PolicyServer(policy, host=HOST, port=PORT)
server.serve_forever()