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

import grounding, search, tools
from heuristics.landmarks import LandmarkHeuristic
from pddl.parser import Parser


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
    name = "vector_inference"

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

    def run_recognizer(self, obs, topk):
        self.fd_time = 0.0
        self.lp_time = 0.0
        num_considered_plan = int(topk)

        v = Validator()
        heuristic = []

        # offline stage vector inference
        self.offline_time = time.time()
        for hyp in self.hyps:
            hyp.evaluate(self.observations)
            os.system('~/symk/fast-downward.py  domain.pddl hyp_%d_problem.pddl --search "symk-fw(simple=true,plan_selection=top_k(num_plans=%d,dump_plans=true))"' %  (hyp.index, num_considered_plan))
            #os.system('~/symk/fast-downward.py  domain.pddl hyp_%d_problem.pddl --search "symq-fw(plan_selection=top_k(num_plans=%d,dump_plans=true),quality=1)"' %  (hyp.index, num_considered_plan))
           
            #os.system('~/symk/fast-downward.py domain.pddl hyp_%d_problem.pddl --search "astar(lmcut())"' % hyp.index)
            
            #os.system('pyperplan -H hff -s gbf domain.pddl hyp_%d_problem.pddl' % hyp.index)
            #hyp.load_plan('hyp_%d_problem.pddl.soln' % hyp.index)
            hyp.num_plan_call = 1
            hyp.score = 0

            if hyp.is_true:
                    v.validate_file('domain.pddl', 'hyp_%d_problem.pddl' % hyp.index, self.observations)
                    observations = v.StatesLog
                    self.obs_len = round(len(self.observations)*int(obs)/100)

            hyp.state = []
            plan_number = 1
            while os.path.exists('./sas_plan.%d' % plan_number):
            #for plan_number in range(1, num_considered_plan + 1):
                if num_considered_plan == 1:
                    hyp.load_plan('sas_plan')
                else:
                    hyp.load_plan('sas_plan.%d' % plan_number)
 
                v.validate_file('domain.pddl', 'hyp_%d_problem.pddl' % hyp.index, hyp.plan)
                if not heuristic:
                    heuristic = HeuristicEstimation(v.Objects, v.Predicates)
             
                hyp.state.append(v.StatesLog)
                #hyp.state.append(heuristic.generateMatrix(v.StatesLog))
                plan_number += 1

            hyp.num_opt_plans = len(hyp.state)
        
        self.offline_time = time.time() - self.offline_time

        # online stage vector inference
        self.online_time = time.time()
        for o in range(1, self.obs_len + 1):
            for hyp in self.hyps:
                hyp.score = []
                #state_vec = heuristic.generateMatrix(observations[0:o+1])
                for plan_number in range(hyp.num_opt_plans):
                    if len(hyp.state[0])-1 >= o:
                        #hyp.score.append(np.linalg.norm(state_vec[o] - hyp.state[plan_number][o]))
                        hyp.score.append(1-np.exp(-1 / np.sqrt(len((set(observations[o]) - set(hyp.state[plan_number][o])) 
                                                     | (set(hyp.state[plan_number][o]) - set(observations[o]))))))
                    else:
                        hyp.score.append(0)
                
                hyp.score = np.mean(hyp.score)
                #print(hyp.score, sum(hyp.score))
                #hyp.score = np.sum(hyp.score)
            #print('========================')
            # Select unique goal (choose the goal with the greatest prop)
            max_prop = np.max([h.score for h in self.hyps])
            # for h in self.hyps:
                # if not self.unique_goal or h.score > self.unique_goal.score:
                #     self.unique_goal = h

            for h in self.hyps:
                if h.score >= max_prop:    
                #if h.score >= self.unique_goal.score:
                    self.accepted.append(h)
        
        
        self.online_time = time.time() - self.online_time 
        self.total_time = self.online_time + self.offline_time
