from dataclasses import dataclass

from RL_CFR_MONOPOLYMODIFIED.game.action import BaseActionResolver, GreedyActionResolver
from RL_CFR_MONOPOLYMODIFIED.game.state import BaseStateAbstraction, IntentStateAbstraction
from RL_CFR_MONOPOLYMODIFIED.RL_models_1_CounterfactualRegretMinimization.cfr.cfr import CFR
from RL_CFR_MONOPOLYMODIFIED.RL_models_1_CounterfactualRegretMinimization.cfr.selector import CFRActionSelector
from RL_CFR_MONOPOLYMODIFIED.RL_models_1_CounterfactualRegretMinimization.gae.model import PolicyAndValueNetwork
from RL_CFR_MONOPOLYMODIFIED.RL_models_1_CounterfactualRegretMinimization.gae.selector import GAEActionSelector
from RL_CFR_MONOPOLYMODIFIED.RL_models_1_CounterfactualRegretMinimization.reinforce.model import NeuralNetworkReinforceModel, TabularReinforceModel
from RL_CFR_MONOPOLYMODIFIED.RL_models_1_CounterfactualRegretMinimization.reinforce.selector import ReinforceActionSelector
from RL_CFR_MONOPOLYMODIFIED.RL_models_1_CounterfactualRegretMinimization.selector import RandomSelector, RiskAwareSelector


@dataclass(frozen=True)
class EmptyBaselineModel:
    abstraction_cls: type[BaseStateAbstraction] = IntentStateAbstraction
    resolver_cls: type[BaseActionResolver] = GreedyActionResolver


MODEL_NAME_TO_CLS = {
    "CFR": CFR,
    "TabularReinforceModel": TabularReinforceModel,
    "NeuralNetworkReinforceModel": NeuralNetworkReinforceModel,
    "PolicyAndValueNetwork": PolicyAndValueNetwork,
    "RandomModel": EmptyBaselineModel,
    "RiskAwareModel": EmptyBaselineModel,
}

MODEL_NAME_TO_SELECTOR = {
    "CFR": CFRActionSelector,
    "TabularReinforceModel": ReinforceActionSelector,
    "NeuralNetworkReinforceModel": ReinforceActionSelector,
    "PolicyAndValueNetwork": GAEActionSelector,
    "RandomModel": RandomSelector,
    "RiskAwareModel": RiskAwareSelector,
}
