import logging
import pandas as pd
import numpy as np
from typing import Dict, List
from entities.fixes import Fixes
import random
from entities.component_failure import ComponentFailure
from entities.components import Components
import json
import copy
import re
from copy import deepcopy

list_string_regex = re.compile(r"[\[(?:\,\s)]?([A-z]+)[\]\,]")
list_float_regex  = re.compile(r"[\[(?:\,\s)]?([\d\.]+)[\]\,]")

class FPHMM():
    def __init__(self, config_path: str ='transition_matrix.csv'):
        self.components = np.array(Components.list()).flat
        self.transition_matrix = pd.read_csv(config_path)
        self.transition_matrix = self.transition_matrix.set_index('Sources')
    
        with open('rule_costs.json', "r") as json_file:
            self.sample_params = json.load(json_file)

        self.current_state = None
        self.initial_shop_states = {}
        self.lastshopstates = {}
        self.lastshoppropagations = {}

    def add_initial_state(self, state):
        self.initial_shop_states[list(state.keys())[0]] = list(state.values())[0]

    def get_state(self):
        return self.current_state
        
    def update_state(self, state):
        logging.debug(state)
        # TODO update state
        pass

    def propagate_failures(self, failed_components: Dict[str, str]):
        matrix = self.transition_matrix
        all_failed_components = deepcopy(failed_components)
        for component, errorId in failed_components.items():
            probabilities = matrix.loc[component]
            for index, new_component in enumerate(self.components):
                choice = random.random()
                if choice < probabilities[index]:
                    all_failed_components[new_component] = errorId

        return all_failed_components

    def create_observation(self, issues):
        shop_name = list(issues.keys())[0]

        if (shop_name in self.lastshopstates) and (self.lastshopstates[shop_name] == issues):
            return self.lastshoppropagations[shop_name]

        self.lastshopstates[shop_name] = copy.deepcopy(issues)
        component_name = list(issues[shop_name].keys())[0]
        failure_type = issues[shop_name][component_name]["failure_name"]
        failed_components = {component_name: failure_type}
        failed_components = self.propagate_failures(failed_components)
        #print(failed_components)
        del failed_components[component_name]

        shop_utility = float(issues[shop_name][component_name]["shop_utility"])

        for failed_component, failure_type in failed_components.items():
            sample_params = self.sample_params[failed_component][failure_type]
            issues[shop_name][failed_component] = self.initial_shop_states[shop_name][failed_component]
            issues[shop_name][failed_component]["failure_name"] = failure_type
            shop_utility -= float(issues[shop_name][failed_component]["component_utility"])
            issues[shop_name][failed_component]["component_utility"] = np.random.normal(*sample_params["component_utility"])
            shop_utility += float(issues[shop_name][failed_component]["component_utility"])
            issues[shop_name][failed_component]["criticality"] = np.random.normal(*sample_params["criticality"])
            issues[shop_name][failed_component]["importance"] = np.random.normal(*sample_params["importance"])
            issues[shop_name][failed_component]["reliability"] = np.random.normal(*sample_params["reliability"])
            issues[shop_name][failed_component]["rule_names"] = list(sample_params.keys())[4:]
            issues[shop_name][failed_component]["rule_costs"] = [np.random.normal(*sample_params[rule_name]) for rule_name in issues[shop_name][failed_component]["rule_names"]]
        
        for failed_component in issues[shop_name].keys():
            issues[shop_name][failed_component]["shop_utility"] = shop_utility

        self.lastshoppropagations[shop_name] = issues
        for shop in issues.keys():
            for component in issues[shop].keys():
                fixes: List[str] = issues[shop][component]["rule_names"]
                costs: List[str] = issues[shop][component]["rule_costs"]
                if isinstance(fixes, str):
                    fixes: List[str] = list_string_regex.findall(fixes)
                    costs = list(map(float, list_float_regex.findall(costs)))
                logging.debug(fixes)
                if "ReplaceComponent" in fixes: # and component != Components.AUTHENTICATION_SERVICE.value:
                    logging.debug("Found ReplaceComponent")
                    index = fixes.index(Fixes.REPLACE_COMPONENT.value)
                    fixes.pop(index)
                    costs.pop(index)
                    logging.debug(fixes)
                issues[shop][component]["rule_names"] = str(fixes).replace("'", "")
                issues[shop][component]["rule_costs"] = str(costs).replace("'", "")
        return issues
