REGISTRY = {}


from .rnn_agent import RNNAgent
from .rnn_agent import PairRNNAgent
from .rnn_agent_DIVA import RNNAgentDIVA

REGISTRY["rnn"] = RNNAgent
REGISTRY["pair_rnn"] = PairRNNAgent
REGISTRY["rnn_DIVA"] = RNNAgentDIVA
