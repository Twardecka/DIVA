from .q_learner import QLearner
from .q_learner_DIVA import QLearnerDIVA
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .casec_learner import CASECLearner


REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["q_learner_DIVA"] = QLearnerDIVA
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["casec_learner"] = CASECLearner
