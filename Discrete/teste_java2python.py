import jpype
import os
import re

def find_top_down_branches(tree, start_node):
    def dfs(current_node, current_branch):
        current_branch.append(current_node)
        if current_node not in tree or len(tree[current_node]) == 0:
            branches.append(list(current_branch))
        else:
            for child_node in tree[current_node]:
                dfs(child_node, list(current_branch))

    branches = []
    dfs(start_node, [])
    return branches

def parse_dot_file(file_path):
    tree = {}
    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith("struct"):
                if "->" in line:
                    parts = line.split("->")
                    source_node = parts[0].split(':')[1].strip()
                    target_node = parts[1].split(':')[1].strip()[0:-1]
                    if source_node not in tree:
                        tree[source_node] = []
                    tree[source_node].append(target_node)
    return tree

def is_ascending(sequence):
    for i in range(1, len(sequence)):
        if sequence[i] != sequence[i - 1] + 1:
            return False
    return True


class ExtractLandmarks:

    def achieved_landmarks_fn1(self, obs):
        obs_now = set()
        for lm in self.L:
            if obs.issuperset(lm):
                obs_now = obs_now | set([lm])

        if len(obs_now - self.activeFL) > 0 and len(self.activeFL) > 0:
            self.FL = self.FL | (self.activeFL - obs_now)
            #self.FL = self.FL - obs_now
            self.activeFL = set()

        self.activeFL = set()
        for lm in self.L:
            if obs.issuperset(lm):
                self.activeFL = self.activeFL | set([lm])

    def achieved_landmarks_fn(self, obs):
        obs_now = set()
        #print([str(a) for a in self.L])
        for lm in self.L:
            #if obs.issuperset(lm):
            if lm in obs:
                obs_now = obs_now | set([lm])

        if len(obs_now - self.activeFL) > 0 and len(self.activeFL) > 0:
            self.FL = self.FL | (self.activeFL - obs_now)
            self.FL = self.FL - obs_now
            self.activeFL = set()

        self.activeFL = set()
        for lm in self.L:
            #if obs.issuperset(lm):
            if lm in obs:
                self.activeFL = self.activeFL | set([lm])

    def inferFactLandmarks(self, landmarkOrdering, observedFacts, observedLandmarks):
        inferredFactLandmarks = set()
        for obs in observedFacts:
            landmarkIndexOrdering = landmarkOrdering.getOrdering().index(obs) + 1
            for i in range(landmarkIndexOrdering - 1):
                if landmarkOrdering.getOrdering()[i] not in observedLandmarks:
                    inferredFactLandmarks.add(landmarkOrdering.getOrdering()[i])
                    #print("\t\t\tinf)", landmarkOrdering.getOrdering()[i])
        return inferredFactLandmarks

    def __init__(self):
        # Set the paths to your Java library JAR files
        lib_jar_files = ['planning-utils4.0.jar', 'jgrapht-jdk1.6.jar', 'planning-landmarks3.0.jar']

        # Get the current directory
        self.current_directory = os.path.dirname(os.path.abspath(__file__))

        # Construct the classpath to include the Java library JAR files
        classpath = os.pathsep.join([os.path.join(self.current_directory, jar_file) for jar_file in lib_jar_files])
        if not jpype.isJVMStarted():
            jpype.startJVM(classpath=classpath)


        # Load Java classes
        self.PDDLParser = jpype.JClass('parser.PDDLParser')
        self.PartialLandmarkGenerator = jpype.JClass('extracting.PartialLandmarkGenerator')
        File = jpype.JClass('java.io.File')
        HashMap = jpype.JClass('java.util.HashMap')

        # Instantiate Java objects
        domainFilePath = os.path.join(self.current_directory, 'domain.pddl')
        initialFilePath = os.path.join(self.current_directory, 'modified_template.pddl')
        goalsFilePath = os.path.join(self.current_directory, 'hyps.dat')


        #realGoalFilePath = os.path.join(current_directory, 'real_hyp.dat')


        self.groundProblem = self.PDDLParser.getGroundDomainProblem(domainFilePath, initialFilePath)


        self.goals = self.PDDLParser.getGoals(self.groundProblem, goalsFilePath)


        #realGoal = PDDLParser.getGoals(groundProblem, realGoalFilePath).get(0)
        self.initialState = self.groundProblem.getSTRIPSInitialState()
        self.goalsToLandmarkExtractor = HashMap()
        self.goalsToFactLandmarks = HashMap()
        self.goalsToLandmarks = HashMap()

        self.goalsToAchievedLandmarks = []

        self.L = set()
        #self.common_landmark = dict()
        #sets = dict()

        for goal in self.goals:
            #self.common_landmark.update({goal: []})
            #sets.update({goal: []})
            currentState = self.initialState
            landmarkExtractor = self.goalsToLandmarkExtractor.get(goal)
            if landmarkExtractor is None:
                landmarkExtractor = self.PartialLandmarkGenerator(self.initialState, goal.getFacts(),
                                                                  self.groundProblem.getActions())


                landmarkExtractor.extractLandmarks()
                for landmarkOrdering in landmarkExtractor.getLandmarksOrdering():
                    lm = landmarkOrdering.getOrdering()
                    #self.last_landmarks = self.last_landmarks | set(lm[-1])
                    for a in lm:
                        self.L = self.L | set(a)

                    #for a in b:
                    #    if a in sets.get(goal) and a not in self.common_landmark.get(goal):
                    #        c = self.common_landmark.get(goal)
                    #        c.append(a)
                    #        self.common_landmark[goal] = c
                    #    else:
                    #        c = sets.get(goal)
                    #        c.append(a)
                    #        sets[goal] = c


            self.goalsToLandmarkExtractor.put(goal, landmarkExtractor)
            self.goalsToLandmarks.put(goal, landmarkExtractor.getLandmarks())
            self.goalsToFactLandmarks.put(goal, landmarkExtractor.getFactLandmarks())

            self.goalsToAchievedLandmarks.append(0)

    def getStates(self):
        domainFilePath = os.path.join(self.current_directory, 'domain.pddl')
        initialFilePath = os.path.join(self.current_directory, 'modified_template.pddl')

        self.groundProblem = self.PDDLParser.getGroundDomainProblem(domainFilePath, initialFilePath)

        observationsFilePath = os.path.join(self.current_directory, 'sas')
        observations = self.PDDLParser.getObservationsAsFacts(self.groundProblem, observationsFilePath)

        currentState = self.initialState
        init = str(self.initialState)

        log = [currentState]
        for o in observations:
            observedFacts = set()
            observedFacts.update(o.getAddPropositions())
            observedFacts.update(o.getPreconditions())
            observedFacts.update(currentState.getFacts())
            currentState = currentState.apply(o)
            log.append(str(currentState))

        return log


    def getProbability(self):
        observationsFilePath = os.path.join(self.current_directory, 'mod_obs.dat')
        observations = self.PDDLParser.getObservations(self.groundProblem, observationsFilePath)
    

        self.activeFL = set()
        self.FL = set()
        active = [1] * len(self.goals)

        achievedFL = []
        currentState = self.initialState
        for o in observations:
            observedFacts = set()
            observedFacts.update(o.getAddPropositions())
            observedFacts.update(o.getPreconditions())
            observedFacts.update(currentState.getFacts())
            self.achieved_landmarks_fn(observedFacts)
            achievedFL.append(self.activeFL)

            currentState = currentState.apply(o)

            #print([str(a) for a in self.activeFL])

        k = 0
        for goal in self.goals:
            landmarkExtractor = self.goalsToLandmarkExtractor.get(goal)
            #landmarkExtractor.extractLandmarksWithDotGraph('graph.dot')
            #tree = parse_dot_file('graph.dot')
            for landmarkOrdering in landmarkExtractor.getLandmarksOrdering():
                lk = landmarkOrdering.getOrdering()
                #goal_node = str(lk[-1])
                #goal_node = goal_node[1:-1]
                #goal_node = "".join(goal_node.split())
                #branches = find_top_down_branches(tree, goal_node)

                #print(lk)
                #print(branches)
                eval = dict()
                if set(lk[-1]) & self.FL:
                    active[k] = 0
                    break

                #for i in range(len(achievedFL)):
                #    ack = []
                #    for a in achievedFL[i]:
                #        b = str(a)
                #        b = "".join(b.split())
                #        ack.append(b)



                    #print(ack)
                    #print([str(a) for a in achievedFL[i]])

                    #for j, branch in enumerate(branches): #and factsOrdering not in self.common_landmark.get(goal):
                        #rev = branch
                        #rev.reverse()
                    #    for n, br_part in enumerate(branch[::-1]):
                    #        br_part = br_part.replace('_', '-')
                            #print(br_part)
                            #print(ack)
                    #        if br_part in ack:
                                #print(br_part)
                    #            if n not in eval.get(str(j), []):
                    #                resul = eval.get(str(j), [])
                    #                resul.append(n)
                    #                eval[str(j)] = resul
                            #print(eval)

                    #        if not is_ascending(eval.get(str(j), [])):
                                #print(eval.get(str(j), []))
                    #            bla
                    #            active[k] = 10
                    #bla
                #for factsOrdering in lk:
                #    for i in range(len(achievedFL)):
                #        if factsOrdering in achievedFL[i]: #and factsOrdering not in self.common_landmark.get(goal):
                #            eval.append(i)
                #            break

            #print(active)
            #print('............')

            k += 1

        k = 0
        count_landmarks = []
        for goal in self.goals:
            currentState = self.initialState
            landmarkExtractor = self.goalsToLandmarkExtractor.get(goal)

            # Computing achieved landmarks from observations for a candidate goal
            achievedLandmarks = set()
            for o in observations:
                observedFacts = set()
                observedFacts.update(o.getAddPropositions())
                observedFacts.update(o.getPreconditions())
                observedFacts.update(currentState.getFacts())
                self.achieved_landmarks_fn(observedFacts)
                achievedFL.append(self.FL)

                for landmarkOrdering in landmarkExtractor.getLandmarksOrdering():
                    for factsOrdering in landmarkOrdering.getOrdering():
                        if observedFacts.issuperset(factsOrdering): # and factsOrdering not in achievedLandmarks:
                            achievedLandmarks.add(factsOrdering)
                            # inferredFacts = self.inferFactLandmarks(landmarkOrdering, observedFacts, achievedLandmarks)
                            # achievedLandmarks.update(inferredFacts)

                currentState = currentState.apply(o)
                #print('====')


            if active[k] == 1:
                self.goalsToAchievedLandmarks[k] = len(achievedLandmarks) / len(self.goalsToLandmarks[goal])
            else:
                self.goalsToAchievedLandmarks[k] = 0

            count_landmarks.append(len(achievedLandmarks))

            k += 1


        return self.goalsToAchievedLandmarks, active, count_landmarks
