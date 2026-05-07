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


class Validator:
    def __init__(self):
        self.Objects = []
        self.Predicates = []
        self.StatesLog = []

class EstimationHRecognizer(PlanRecognizer): # Mirroring with landmarks
    name = "landmarks"

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
            stream += s.replace(")(", "), (") + '\n'

        instream = open("hyps.dat", 'w')
        instream.writelines(stream)
        instream.close()

    def __init__(self, options, h = True, h_c = False, h_s = False):
        PlanRecognizer.__init__(self, options, h, h_c, h_s)
        self.uncertainty_ratio = 1
        self.countObservations = 0
        self.accepted = []

    def run_recognizer(self, obs):
        
        self.fd_time = 0.0
        self.lp_time = 0.0
        active_now = []

        self.obs_len = round(len(self.observations)*int(obs)/100)
        for hyp in self.hyps:
            hyp.evaluate(self.observations)
            hyp.num_plan_call = 0
            active_now.append(1)

        # offline stage Mirroring
        self.offline_time = time.time()
        #self.modify_hyps()
        self.modify_problem()


        lm = ExtractLandmarks()
        self.offline_time = time.time() - self.offline_time

        # online stage online landmarks
        self.online_time = time.time()
        with open("obs.dat", "r") as file:
            lines = file.readlines()

        count_prune = 0
        count_prune_all = 0
        for o in range(self.obs_len):
            with open("mod_obs.dat", "w") as file:
                file.writelines(lines[0:o+1])

            prop, active, count = lm.getProbability()

            i = 0
            for hyp in self.hyps:
                count_prune_all += 1
                if max(prop) == prop[i] and active[i] == 1:
                    count_prune += 1
                    hyp.score = prop[i]
                else:
                    active_now[i] = 0
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

