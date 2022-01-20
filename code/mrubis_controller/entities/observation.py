from dataclasses import dataclass
from typing import List
from entities.fixes import Fixes
from entities.components import Components
from entities.component_failure import ComponentFailure

@dataclass
class Component:
    component_name: Components
    utility: float

@dataclass
class AgentFix:
    component: Components
    fix: Fixes

@dataclass
class Fix:
    fix_type: Fixes
    fix_cost: float

@dataclass
class Issue(Component):
    failure_type: ComponentFailure
    fixes: List[Fix]

@dataclass
class AppliedFix:
    fix_type: Fixes
    fixed_component: Components
    worked: bool

@dataclass
class Observation:
    shop_name: str
    shop_utility: float
    issues: List[Issue]
    applied_fix: AppliedFix

@dataclass
class InitialState:
    shop_name: str
    shop_utility: float
    components: List[Component]
