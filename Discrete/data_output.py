#!/usr/bin/env python2.7

##
## Generate method results summary tables (data-tables and data-charts folder).
##

## Uses:
# For especific domains:
# ./data_output.py "delta-cl delta-o-cl1" blocks-world-optimal depots-optimal [-fast]
# For all domains:
# ./data_output.py "delta-cl delta-o-cl1" all [-fast]
# For method groups:
# ./data_output.py lmc all [-fast]
##

import os, sys
import data_domain as dd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import numpy as np
import statistics


def cmp(a, b):
	if a < b:
		return -1
	if a == b:
		return 0
	if a > b:
		return 1


class ProblemOutput:

	#
	# The results for a single goal recognition task. 
	#

	def __init__(self, name, recognizer = None):
		self.name = name
		self.scores = {}
		self.lp_infos = {}
		self.score_roc = []
		self.hyp_roc = []
		if recognizer is None:
			return

		# Problem data
		self.hyp_atoms = [h.atoms for h in recognizer.hyps]

		# Get solution
		self.solution_set = [h.index for h in recognizer.accepted]
		exact_solution_set = [h.index for h in recognizer.hyps if h.is_true]
		real_hyp = recognizer.get_real_hypothesis()
		hyp = recognizer.unique_goal

		# Wrong hyp
		wrong_hyp = None

		# Time results
		self.lp_time = recognizer.lp_time
		self.fd_time = recognizer.fd_time
		self.online_time = recognizer.online_time
		self.offline_time = recognizer.offline_time
		self.total_time = self.online_time + self.offline_time

		# Domain info
		self.num_obs = recognizer.obs_len
		self.num_goals = len(recognizer.hyps)
		self.num_solutions = len(exact_solution_set)

		# Results
		#self.num_OP = np.mean([h.num_opt_plans for h in recognizer.hyps])
		self.num_plan = sum([h.num_plan_call for h in recognizer.hyps])
		total = len(self.solution_set)
		for exact in exact_solution_set:
			if exact not in self.solution_set:
				total += 1


		tp = 0
		for exact in exact_solution_set:
			tp += self.solution_set.count(exact)

		var = [1 if exact_solution_set.count(a) > 0 else 0 for a in self.solution_set]
		if len(var) > 2:
			self.var = statistics.variance(var)
		else:
			self.var = 0


		fp = len(self.solution_set) - tp
		fn = self.num_obs - tp
		tn = self.num_goals * self.num_obs - tp - fp - fn

		self.fpr = fp/(fp + tn)
		self.fnr = fn/(fn + tp)
		self.tpr = tp/(tp + fn)

		self.agreement = tp/(tp + fp)
		self.spread = len(self.solution_set)/self.num_obs
		self.accuracy = (tp + tn)/(tp + tn + fp + fn)
		self.perfect_agr = 1 if self.agreement == 1 else 0

		converged_count = 1
		for i in range(len(self.solution_set) - 1):
			if self.solution_set[i] == exact_solution_set[0] and self.solution_set[i+1] == exact_solution_set[0]:
				converged_count += 1

		self.cv = converged_count/self.num_obs



	def load(self, problem_data, raw_problem):
		# Problem data
		self.hyp_atoms = problem_data.hyps

		# Time results
		if len(raw_problem.times) > 0:
			self.total_time = raw_problem.times[0]
		if len(raw_problem.times) > 1:
			self.lp_time = raw_problem.times[1]
		if len(raw_problem.times) > 2:
			self.fd_time = raw_problem.times[2]

		# Domain info
		self.num_obs = len(problem_data.obs)
		self.num_goals = len(problem_data.hyps)
		self.num_solutions = len(problem_data.solution)

		# Get solution
		exact_solution_set = frozenset(problem_data.get_solution_indexes())
		wrong_solution_set = frozenset(raw_problem.goals) - exact_solution_set
		real_hyp = problem_data.get_true_hyp_index()
		self.solution_set = frozenset([i for i in raw_problem.goals if raw_problem.accepted[i] == True])
		min_score = float("inf")
		hyp = None # index
		for goal in raw_problem.goals:
			score = raw_problem.scores[goal][1] - raw_problem.scores[goal][0]
			if score < min_score:
				min_score = score
				hyp = goal
		min_score = float("inf")
		wrong_hyp = None # index
		for goal in wrong_solution_set:
			score = raw_problem.scores[goal][1] - raw_problem.scores[goal][0]
			if score < min_score:
				min_score = score
				wrong_hyp = goal

		# Results
		total = float(len(exact_solution_set | self.solution_set))
		fp = float(len(self.solution_set - exact_solution_set))
		fn = float(len(exact_solution_set - self.solution_set))
		agr = (total - fp - fn) / total
		self.fpr = fp / total
		self.fnr = fn / total
		self.agreement = agr
		self.spread = len(self.solution_set)
		self.accuracy = 1 if real_hyp in self.solution_set else 0
		self.perfect_agr = 1 if agr == 1 else 0




		# Hyp data
		self.scores = raw_problem.scores
		self.lp_infos = raw_problem.lp_infos
		# Chosen hyp
		if hyp:
			self.lp_info_best = raw_problem.lp_infos[hyp]
			self.h_value_best = raw_problem.scores[hyp][0]
			self.hc_value_best = raw_problem.scores[hyp][1]
		else:
			self.lp_info_best = [0, 0]
			self.h_value_best = 0
			self.hc_value_best = 0
		# Reference hyp
		if real_hyp in raw_problem.goals:
			self.lp_info_real = raw_problem.lp_infos[real_hyp]
			self.h_value_real = raw_problem.scores[real_hyp][0]
			self.hc_value_real = raw_problem.scores[real_hyp][1]
		else:
			self.lp_info_real = [0, 0]
			self.h_value_real = 0
			self.hc_value_real = 0
		# Wrong hyp
		if wrong_hyp:
			self.lp_info_wrong = raw_problem.lp_infos[wrong_hyp]
			self.h_value_wrong = raw_problem.scores[wrong_hyp][0]
			self.hc_value_wrong = raw_problem.scores[wrong_hyp][1]
		else:
			self.lp_info_wrong = [0, 0]
			self.h_value_wrong = 0
			self.hc_value_wrong = 0

	def print_content(self):
		content = self.name + ":" + str(self.total_time) + ":" + str(self.lp_time) + ":" + str(self.fd_time) + "\n"
		hyps = list(self.scores.keys())
		for h in hyps:
			#atoms = ','.join(self.hyp_atoms[h])
			atoms = str(h)
			score = ','.join([str(int(x)) for x in self.scores[h]])
			lp_info = ','.join([str(int(x)) for x in self.lp_infos[h]])
			accepted = str(h in self.solution_set)
			content += "> " + atoms + ":" + accepted + ":" + score + ":" + lp_info + "\n"
		return content


class ExperimentOutput:

	#
	# The results for the set of goal recognition tasks for a single domain and a single observatility level.
	#

	def __init__(self, obs):
		self.obs = obs
		self.online_time = 0
		self.offline_time = 0
		self.max_time = 0
		self.problem_outputs = dict()

	def load(self, domain_data, raw_experiment):
		if self.obs in raw_experiment.experiment_times:
			self.total_time = raw_experiment.experiment_times[self.obs]
			self.max_time = self.total_time
		for raw_problem in raw_experiment.problems[self.obs]:
			problem_data = domain_data.data[self.obs][raw_problem.file]
			problem = ProblemOutput(problem_data.name)
			problem.load(problem_data, raw_problem)
			self.add_problem(problem)

	def add_problem(self, problem):
		self.problem_outputs[problem.name] = problem
		self.online_time += problem.online_time
		self.offline_time += problem.offline_time
		#self.max_time = max(self.max_time, problem.total_time)

	def print_stats(self):
		problems = self.problem_outputs.values()
		n = float(len(problems))
		values = [\
		sum([p.num_obs for p in problems]) / n, \
		sum([p.num_goals for p in problems]) / n, \
		sum([p.num_solutions for p in problems]) / n, \
		sum([p.agreement for p in problems]) / n, \
		np.std([p.agreement for p in problems]), \
		sum([p.accuracy for p in problems]) / n, \
		np.std([p.accuracy for p in problems]), \
		sum([p.spread for p in problems]) / n, \
		sum([p.perfect_agr for p in problems]), \
		self.online_time / n, \
		self.offline_time / n, \
		sum([p.num_plan for p in problems]) / n]

		content = "%s\t%s\t" % (len(problems), self.obs)
		content += '\t'.join(["%2.4f" % x for x in values])
		content += '\n'
		return content 

	def print_hdata(self):
		content = self.obs + '\n'
		content += str([x.h_value_real for x in self.problem_outputs.values()]) + '\n'
		content += str([x.h_value_best for x in self.problem_outputs.values()]) + '\n'
		content += str([x.spread for x in self.problem_outputs.values()]) + '\n'
		content += str([x.hc_value_real for x in self.problem_outputs.values()]) + '\n'
		content += str([x.hc_value_best for x in self.problem_outputs.values()]) + '\n'
		content += str([x.fpr for x in self.problem_outputs.values()]) + '\n'
		content += str([x.fnr for x in self.problem_outputs.values()]) + '\n'
		content += str([x.agreement for x in self.problem_outputs.values()]) + '\n'
		content += str([x.lp_time for x in self.problem_outputs.values()]) + '\n'
		content += str([x.h_value_wrong for x in self.problem_outputs.values()]) + '\n'
		content += str([x.hc_value_wrong for x in self.problem_outputs.values()]) + '\n'
		return content


class MethodOutput:

	#
	# The results for the set of goal recognition tasks for a single domain.
	#

	def __init__(self, method, domain_data):
		self.name = method
		self.domain_data = domain_data
		self.experiments = []
		raw_experiment = RawExperiment(domain_data.name + "-" + self.name, domain_data.observabilities)
		for obs in domain_data.observabilities:
			experiment = ExperimentOutput(obs)
			experiment.load(domain_data, raw_experiment)
			self.experiments.append(experiment)

	def print_table(self):
		content = "#P\tO%\t|O|\t|G|\t|S|\tAR\tFPR\tFNR\tAcc\tSpread\tPER\tTime\tTimeLP\tTimeFD\tVars\tConsts\tH\tHC\n"
		for experiment in self.experiments:
			content += experiment.print_stats()
		return content


class RawProblem:

	#
	# Output data for a single goal recognition task.
	#

	def __init__(self, file):
		self.file = file
		self.goals = []
		self.accepted = {}
		self.scores = {}
		self.lp_infos = {}
		self.times = []

	def add_goal(self, line):
		line = line.strip().replace("> ", "").split(":")
		#hyp = frozenset([tok.strip().lower() for tok in line[0].split(',')])
		hyp = int(line[0])
		self.goals.append(hyp)
		if len(line) > 1:
			self.accepted[hyp] = line[1] == 'True'
		else:
			self.accepted[hyp] = True
		if len(line) > 2:
			self.scores[hyp] = [float(x) for x in line[2].split(',')]
		else:
			self.scores[hyp] = [0, 0]
		if len(line) > 3:
			self.lp_infos[hyp] = [float(x) for x in line[3].split(',')]
		else:
			self.lp_infos[hyp] = [0, 0]

	def add_times(self, line):
		self.times = [float(x) for x in line[1:]]


class RawExperiment:

	#
	# Output data for a single domain and a single observatility level.
	#

	def __init__(self, filename, observabilities):
		file = open("outputs/" + filename + ".output")
		current_problem = None
		problems = []
		experiment_times = []
		for line in file:
			# New goal for current problem
			if line.startswith(">"):
				if current_problem is not None:
					current_problem.add_goal(line)
				continue
			line = line.split(":")
			current_file = line[0].strip().replace("pb", "p")
			if dd.filter(current_file):
				current_file = None
				continue
			if current_file[0].isdigit():
				experiment_times.append(current_file)
				continue
			current_problem = RawProblem(current_file)
			problems.append(current_problem)
			if len(line) > 1 and line[1].strip()[0].isdigit():
				current_problem.add_times(line)
		# Separate by observability
		self.problems = dict()
		for obs in observabilities:
			self.problems[obs] = []
		for problem in problems:
			for obs in observabilities:
				if obs + "/" in problem.file:
					self.problems[obs].append(problem)
					break
		self.experiment_times = dict()
		for i in range(len(experiment_times)):
			self.experiment_times[observabilities[i]] = experiments[i]


def write_txt_files(domain_data, methods):
	for domain_data in all_domain_data.values():
		for method in methods:
			method_output = MethodOutput(method, domain_data)
			with open("data-tables/" + domain_data.name + "-" + method + ".txt", 'w') as f:
				f.write(method_output.print_table())
			with open("data-charts/" + domain_data.name + "-" + method + ".txt", 'w') as f:
				for experiment in method_output.experiments:
					f.write(experiment.print_hdata())

if __name__ == '__main__':
	observabilities = ['10', '30', '50', '70', '100']
	base_path = '../goal-plan-recognition-dataset/'

	# Flags
	test = False
	if '-fast' in sys.argv:
		set_filter(True)
		dd.set_filter(True)
		sys.argv.remove('-fast')
	if '-test' in sys.argv:
		test = True
		sys.argv.remove('-test')
		base_path = 'experiments/'

	# Domains
	domains = dd.parse_domains(sys.argv[2:], test)
	all_domain_data = {}
	for d in domains:
		domain_data = dd.DomainData(d, observabilities)
		if os.path.exists("data-domains/" + d + ".txt"):
			domain_data.read("data-domains/")
		else:
			domain_data.load(base_path)
		all_domain_data[d] = domain_data

	# Methods
	if 'basic' in sys.argv[1]:
		methods = ['delta-cl', 'delta-cp', 'delta-cs']
	elif 'lmc' in sys.argv[1]:
		methods = ['delta-cl', 'delta-o-cl', 'delta-o-cl3', 'delta-o-cl1']
	elif 'delr' in sys.argv[1]:
		methods = ['delta-o-cdt', 'delta-o-cdto', 'delta-o-cdtb5']
	elif 'flow' in sys.argv[1]:
		methods = ['delta-cf1', 'delta-cf1ab', 'delta-o-cf17', 'delta-o-cf16', 'delta-cf2']
	else:
		methods = sys.argv[1].split()
		sys.argv[1] = ""
	if 'f2' in sys.argv[1]:
		methods = [method + "-f2" for method in methods]

	write_txt_files(all_domain_data, methods)
