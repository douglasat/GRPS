#!/usr/bin/env python3.10
from heuristic_estimation import HeuristicEstimation
from plan_recognizer import PlanRecognizer
import time
from pddl_parser import PDDLParser
from action import Action
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import copy

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

from teste_java2python import ExtractLandmarks


def find_key_by_value(dictionary, target_value):
    for key, values in dictionary.items():
        if target_value in values:
            return key
    return None

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

        self.validate(ground_actions, parser.state, (parser.positive_goals, parser.negative_goals), plan), parser.objects

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

        #self.StatesLog = np.array(self.StatesLog)


class EstimationHRecognizer(PlanRecognizer): # Mirroring with landmarks
    name = "mirroring_landmarks"

    def modify_problem(self):
        with open("template.pddl", "r") as file:
            pddl_lines = file.read()
            #line = file.readlines()

        pddl_lines = pddl_lines.replace("<HYPOTHESIS>", "")
        with open("modified_template.pddl", "w") as file:
            file.write(pddl_lines)

    def modify_hyps(self):
        stream = ""
        for a in self.hyps:
            s = str(a)
            stream += s.replace(")(", "),(") + '\n'

        instream = open("hyps.dat", 'w')
        instream.writelines(stream)
        instream.close()


    def __init__(self, options, h = True, h_c = False, h_s = False):
        PlanRecognizer.__init__(self, options, h, h_c, h_s)
        self.uncertainty_ratio = 1
        self.countObservations = 0
        self.accepted = []

    def run_recognizer(self, obs):
        # Get the directory of the script
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sas_plan')
        self.fd_time = 0.0
        self.lp_time = 0.0

        # offline stage Mirroring
        self.offline_time = time.time()
        #self.modify_hyps()
        self.modify_problem()

        v = Validator()
        lm = ExtractLandmarks()

        for hyp in self.hyps:
            hyp.evaluate(self.observations)
            #os.system('../fast-downward/fast-downward.py domain.pddl hyp_%d_problem.pddl --search "astar(lmcut())"' % hyp.index)
            os.system('~/symk/fast-downward.py  domain.pddl hyp_%d_problem.pddl --search "sym-fw()"' %  hyp.index)
            hyp.load_plan('sas_plan')
            hyp.costOptimal = len(hyp.plan)

            hyp.score = 0
            hyp.num_plan_call = 1

            if hyp.is_true:
                self.obs_len = round(len(self.observations)*int(obs)/100)
        
        self.offline_time = time.time() - self.offline_time

        # online stage Mirroring
        self.online_time = time.time()
        with open("obs.dat", "r") as file:
            lines = file.readlines()

        for o in range(self.obs_len):
            with open("mod_obs.dat", "w") as file:
                file.writelines(lines[0:o + 1])

            prop, active, count = lm.getProbability()

            i = 0
            for hyp in self.hyps:
                if max(prop) == prop[i]:
                    v.validate_file('domain.pddl', 'hyp_%d_problem.pddl' % hyp.index, self.observations[0:o + 1])

                    with open('sas', 'w') as file:
                        for position in self.observations[0:o + 1]:
                            file.write(position + '\n')


                    hyp.num_plan_call += 1

                    instream = open('hyp_%d_problem.pddl' % hyp.index, 'r')
                    instream1 = instream.readlines()
                    listStates = list(v.StatesLog[-1])
                    for k in range(len(instream1)):
                        line = instream1[k].strip()
                        if line == '(:init':
                            while not (instream1[k + 1].strip() == ')' or instream1[k + 1].strip() == '(:goal'):
                                instream1.pop(k + 1)

                            if not instream1[k + 1].strip() == ')':
                                instream1.insert(k + 1, ')\n')

                            while len(listStates) > 0:
                                vv = ''
                                for s in listStates[0]:
                                    vv = vv + str(s) + ' '
                                vv = '(' + vv[0:-1] + ')\n'
                                instream1.insert(k + 1, vv)
                                listStates.pop(0)
                            break

                    instream.close()
                    instream = open('templateMinus.pddl', 'w')
                    instream.writelines(list(instream1))
                    instream.close()

                    #os.system('../fast-downward/fast-downward.py domain.pddl templateMinus.pddl --search "astar(lmcut())"')
                    os.system('~/symk/fast-downward.py  domain.pddl hyp_%d_problem.pddl --search "sym-fw()"' %  hyp.index)
                    if os.path.isfile(file_path):
                        hyp.load_plan('sas_plan')
                        hyp.score = hyp.costOptimal/(o + 1 + len(hyp.plan))
                    else:
                        hyp.score = 0 

                else:
                    hyp.score = 0

                i += 1

            # Select unique goal (choose the goal with the highest hypotheses)
            for h in self.hyps:
                if not self.unique_goal or h.score > self.unique_goal.score:
                    self.unique_goal = h

            for h in self.hyps:
                if h.score >= self.unique_goal.score:
                    self.accepted.append(h)

        self.online_time = time.time() - self.online_time 
        self.total_time = self.online_time + self.offline_time

