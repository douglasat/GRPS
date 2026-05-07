#!/usr/bin/env python3.9
from heuristic_estimation import HeuristicEstimation
from plan_recognizer import PlanRecognizer
import time
from pddl_parser import PDDLParser
from action import Action
import os
import numpy as np
import random
import matplotlib.pyplot as plt

import importlib
import logging
import os
import re
import subprocess
import sys
import time
import copy
import pickle
from tarski.io import PDDLReader
from tarski.grounding import LPGroundingStrategy
from fastdtw import fastdtw

import grounding, search, tools
from heuristics.landmarks import LandmarkHeuristic
from pddl.parser import Parser

import signature as sig_module
from tree import Tree
from tree import Node
import graphviz
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def parse_predicate(predicate_str):
    name, args = predicate_str.split('(', 1)
    args = args.rstrip(')').split(',')
    return (name, *args)


def nodes_traj_to_signature_traj(node_traj):
    traj = []
    traj.append((1,))
    for node in node_traj:
        if not node.identifier == (1,):
            traj.append(node.identifier)
    return traj 

class Validator:
    def __init__(self):
        self.Objects = []
        self.Predicates = []
        self.StatesLog = []

    def parse_plan(self, filename):
        with open(filename,'r') as f:
            plan = []
            for act in f.read().splitlines():
                act = act[1:-1].split()
                plan.append(Action(act[0], tuple(act[1:]), [], [], [], []))
            return plan

    def validate_file(self, domainfile, problemfile, planfile):
        self.Objects = []
        self.Predicates = []
        self.StatesLog = []
        return self.validate_plan(domainfile, problemfile, planfile)

    def validate_plan(self, domainfile, problemfile, plan):
        # Parser
        parser = PDDLParser()
        parser.parse_domain(domainfile)
        parser.parse_problem(problemfile)

        for a in parser.objects:
            self.Objects.extend(list(parser.objects.get(a)))

        self.Predicates = list(parser.predicates.keys())

        # Grounding process
        ground_actions = []

        for p in plan:
            p = str(p[1:-1]).rsplit(' ')
            for action in parser.actions:
                O = {}
                if action.name == p[0]:
                    n = 1
                    for par in action.parameters:
                        if O.get(par[1]):
                            b = O.get(par[1])
                            b.append(p[n])
                            O.update({par[1]: b})
                        else:
                            O.update({par[1]: [p[n]]})
                        n += 1

                    for act in action.groundify(O):
                        ground_actions.append(act)

        self.validate(ground_actions, parser.state, (parser.positive_goals, parser.negative_goals),
                      plan), parser.objects

    def validate(self, actions, initial_state, goals, plan):

        s = list(initial_state)
        self.StatesLog = []
        self.StatesLog.append(s)
        for p in plan:
            p = str(p[1:-1]).rsplit(' ')
            for a in actions:
                b = [a.name]
                b.extend(a.parameters)
                if b == p:
                    s = s + list(a.add_effects)
                    for c in list(a.del_effects):
                        if s.count(c) > 0:
                            s.remove(c)
                    self.StatesLog.append(s)
                    break
        
        # self.StatesLog = np.array(self.StatesLog)


class EstimationHRecognizer(PlanRecognizer):
    name = "vector_PS"

    def __init__(self, options, h = True, h_c = False, h_s = False):
        PlanRecognizer.__init__(self, options, h, h_c, h_s)
        self.uncertainty_ratio = 1
        self.states = []
        self.accepted = []

    def accept_hypothesis(self, h):
        return np.linalg.norm(h.score - self.unique_goal.score) <= 1e-6

    def get_score(self, h):
        return [h.h, h.obs_misses]

    def verify_hypothesis(self):
        if self.unique_goal:
            for h in self.hyps:
                if self.accept_hypothesis(h):
                    self.accepted_hypotheses.add(h)
        else:
            for h in self.hyps:
                self.accepted_hypotheses.add(h)
            print("All hypotheses failed.")
            print(self.options.exp_file)

    def calculate_uncertainty(self):
        self.uncertainty_ratio = 1

    def run_recognizer(self, obs, exp_file, merge, prune, method, topk):
        sig = sig_module.Signatures(device="cpu", depth=2)
        
        self.fd_time = 0.0
        self.lp_time = 0.0
        num_considered_plan = int(topk)

        v = Validator()
        heuristic = []

        # path signature trees
        trees = {}
        roots = {}
        sig_level_by_goal = {}
        sig_branch_by_goal = {}

        # offline stage vector inference
        self.offline_time = time.time()
        for hyp in self.hyps:
            # trajectory free definition
            tree = Tree(merge_threshold=float(merge), prune_threshold=float(prune))
            root = Node(identifier=(1,), data=[], level=0)
            tree.add_node(root)
            trees[str(hyp.index)] = tree
            roots[str(hyp.index)] = root
            sig_level_by_goal[str(hyp.index)] = []

            if float(merge) == 0 and float(prune) == 0:
                memorize = True
            else:
                memorize = False

            if memorize:
                hyp.evaluate(self.observations)
                os.system('~/symk/fast-downward.py  domain.pddl hyp_%d_problem.pddl --search "symk-fw(simple=true,plan_selection=top_k(num_plans=%d,dump_plans=true))"' %  (hyp.index, num_considered_plan))
                #os.system('~/symk/fast-downward.py  domain.pddl hyp_%d_problem.pddl --search "symq-fw(plan_selection=top_k(num_plans=%d,dump_plans=true),quality=1)"' %  (hyp.index, num_considered_plan))
                
                hyp.num_plan_call = 1
                hyp.score = 0

                if hyp.is_true:
                    v.validate_file('domain.pddl', 'hyp_%d_problem.pddl' % hyp.index, self.observations)
                    observations = v.StatesLog
                    self.obs_len = round(len(self.observations)*int(obs)/100)

                    # # Load the PDDL domain and problem
                    reader = PDDLReader()
                    reader.parse_domain('domain.pddl')
                    reader.parse_instance('hyp_%d_problem.pddl' % hyp.index)

                    # Get the problem object and its initial state
                    problem = reader.problem
        
                    grounding = LPGroundingStrategy(problem)
                    ground_states = grounding.ground_state_variables()
                    ground = {}

                    for i, atom in ground_states.enumerate():
                        ground[i] = parse_predicate(str(atom))

                    vec = [[0] * len(ground) for _ in range(len(v.StatesLog))]
                    for j in range(len(v.StatesLog)):
                        for i in sorted(ground.keys()):
                            if ground[i] in v.StatesLog[j]:
                                if i > 0:
                                    vec[j][i] =  1.0
                                else:
                                    vec[j][i] =  1.0
                            else:
                                if i > 0:
                                    vec[j][i] = 0.0
                                else:
                                    vec[j][i] = 0.0

                        
                    observations = np.array(vec)

                    with open(f"./Grid_Data/{method}/{exp_file.split('/')[0][0:-8]}/obs_states{exp_file[-20]}{exp_file[-14]}.pkl", "wb") as f:
                        pickle.dump(observations, f)

                hyp.state = []
                plan_number = 1
                while os.path.exists('./sas_plan.%d' % plan_number):
                    if num_considered_plan == 1:
                        hyp.load_plan('sas_plan')
                    else:
                        hyp.load_plan('sas_plan.%d' % plan_number)
    
                    
                    v.validate_file('domain.pddl', 'hyp_%d_problem.pddl' % hyp.index, hyp.plan)
                    if not heuristic:
                        heuristic = HeuristicEstimation(v.Objects, v.Predicates)

                    # Load the PDDL domain and problem
                    reader = PDDLReader()
                    reader.parse_domain('domain.pddl')
                    reader.parse_instance('hyp_%d_problem.pddl' % hyp.index)

                    # Get the problem object and its initial state
                    problem = reader.problem
    
                    grounding = LPGroundingStrategy(problem)
                    ground_states = grounding.ground_state_variables()
                    ground = {}

                    for i, atom in ground_states.enumerate():
                        ground[i] = parse_predicate(str(atom))

                    vec = [[0] * len(ground) for _ in range(len(v.StatesLog))]
                    for j in range(len(v.StatesLog)):
                        for i in sorted(ground.keys()):
                            if ground[i] in v.StatesLog[j]:
                                    if i > 0:
                                        vec[j][i] =  1.0
                                    else:
                                        vec[j][i] =  1.0
                            else:
                                if i > 0:
                                    vec[j][i] =  0.0
                                else:
                                    vec[j][i] =  0.0
                    
                    hyp.state.append(np.array(vec))

                    plan_number += 1
                    hyp.num_opt_plans = len(hyp.state)
                

                with open(f"./Grid_Data/{method}/{exp_file.split('/')[0][0:-8]}/data{exp_file[-20]}{exp_file[-14]}{hyp.index}.pkl", "wb") as f:
                    pickle.dump(hyp.state, f)

                with open(f"./Grid_Data/{method}/{exp_file.split('/')[0][0:-8]}/heuristic{exp_file[-20]}{exp_file[-14]}.pkl", "wb") as f:
                    pickle.dump(heuristic, f)

            else:
                with open(f"./Grid_Data/{method}/{exp_file.split('/')[0][0:-8]}/data{exp_file[-20]}{exp_file[-14]}{hyp.index}.pkl", "rb") as f:
                    hyp.state = pickle.load(f)
                hyp.num_opt_plans = len(hyp.state)
                hyp.num_plan_call = 1
                
                with open(f"./Grid_Data/{method}/{exp_file.split('/')[0][0:-8]}/heuristic{exp_file[-20]}{exp_file[-14]}.pkl", "rb") as f:
                    heuristic = pickle.load(f)
                
                with open(f"./Grid_Data/{method}/{exp_file.split('/')[0][0:-8]}/obs_states{exp_file[-20]}{exp_file[-14]}.pkl", "rb") as f:
                    observations = pickle.load(f)
                
                self.obs_len = round(len(self.observations)*int(obs)/100)

            #path signature trees
            for t, states in zip(range(num_considered_plan), hyp.state):
                signatures_byK = sig.get_all_signatures(states)
                previous = roots[str(hyp.index)]
                for signature in signatures_byK:
                    if not trees[str(hyp.index)].contains(signature):
                        node = Node(identifier=signature, data=[t])
                        trees[str(hyp.index)].add_node(node, None if signature == (1,) else previous)
                    else:
                        node = trees[str(hyp.index)].get_node(signature)
                        node.data.append(t)
                        node.data = list(set(node.data))

                    previous = node   

            trees[str(hyp.index)].naming()
            trees[str(hyp.index)].merge()
            trees[str(hyp.index)].prune()

            # Align trajectory by level
            k = 0
            last_node = 0
            max_data_legth = 0
            nodes = trees[str(hyp.index)].get_node_by_level(k)
            while len(nodes) > 0:
                level_k = []
                length_data = []
                for node in nodes:
                    level_k.append(node.identifier)
                    length_data.append(node.data)
                
                max_data_legth = max(max_data_legth, len(length_data))
                sig_level_by_goal[str(hyp.index)].append(level_k)
                last_node = k
                k = k + 1
                nodes = trees[str(hyp.index)].get_node_by_level(k)
            
            # Align trajectory by branch
            sig_branch_by_goal[str(hyp.index)] = [[None for _ in range(num_considered_plan)] for _ in range(k)]
            for i in range(k):
                nodes = trees[str(hyp.index)].get_node_by_level(i)
                for node in nodes:
                    for index in node.data:
                        sig_branch_by_goal[str(hyp.index)][i][index] = copy.deepcopy(node.identifier)

        self.offline_time = time.time() - self.offline_time

        # online stage vector inference
        self.online_time = time.time()
        for obs in range(1, self.obs_len + 1):
            sig_path_observations = sig.get_all_signatures(observations[0:obs+1])
            for hyp in self.hyps:
                sig_group = []
                signatures_hyp = sig_branch_by_goal[str(hyp.index)]
                if len(signatures_hyp)-1 >= obs:
                    for plan_number in range(num_considered_plan):
                        sig_vec = [signatures_hyp[o][plan_number] for o in range(obs+1) if signatures_hyp[o][plan_number] is not None]
                        if len(sig_vec) > 1:
                            if not sig_vec[1] == (1,):
                                if method == 'GRPS+DTW':
                                    _, path = fastdtw(np.array(sig_path_observations[1:]), sig_vec[1:], dist=euclidean)
                                    path = np.array(path)
                                    sig_vec = np.array(sig_vec[1:]) 
                                    obs_index = list(path[:, 0])
                                    obs_index = obs_index.index(obs - 1)
                                    val = np.linalg.norm(sig_path_observations[-1] - sig_vec[path[obs_index, 1]])
                                else:
                                    if obs <= len(sig_vec) - 1:
                                        val = np.linalg.norm(sig_path_observations[-1] - np.array(sig_vec[obs]))
                                    else:
                                        val = np.linalg.norm(sig_path_observations[-1] - np.array(sig_vec[-1]))

                                if val == 0:
                                    sig_group.append(1)
                                else:
                                    sig_group.append(1 - np.exp(-1/val))
                            else:
                                sig_group.append(0)
                        else:
                            sig_group.append(0)
                else:
                    sig_group.append(0)
                
                hyp.score = np.mean(sig_group)
        
            # Select unique goal (choose the goal with the greatest prop)
            max_prop = np.max([h.score for h in self.hyps])

            for h in self.hyps:
                if h.score >= max_prop:    
                    self.accepted.append(h)
        
        
        self.online_time = time.time() - self.online_time 
        self.total_time = self.online_time + self.offline_time
