import jpype
import os
import re


def is_ascending(sequence):
    # if len(sequence) == 1 and not sequence[0] == 0:
    #    return False

    for i in range(len(sequence) - 1):
        if sequence[i] > sequence[i + 1]:
            return False
    return True


class ExtractLandmarks:

    def achieved_landmarks_fn(self, obs):
        obs_now = set()
        for lm in self.L:
            if obs.issuperset(lm):
                obs_now = obs_now | set([lm])

        if len(obs_now - self.activeFL) > 0 and len(self.activeFL) > 0:
            self.FL = self.FL | (self.activeFL - obs_now)
            self.activeFL = set()

        self.activeFL = set()
        for lm in self.L:
            if obs.issuperset(lm):
                self.activeFL = self.activeFL | set([lm])

    def inferFactLandmarks(self, landmarkOrdering, observedFacts, observedLandmarks):
        inferredFactLandmarks = set()
        for obs in observedFacts:
            landmarkIndexOrdering = landmarkOrdering.getOrdering().index(obs) + 1
            for i in range(landmarkIndexOrdering - 1):
                if landmarkOrdering.getOrdering()[i] not in observedLandmarks:
                    inferredFactLandmarks.add(landmarkOrdering.getOrdering()[i])
                    # print("\t\t\tinf)", landmarkOrdering.getOrdering()[i])
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

        # realGoalFilePath = os.path.join(current_directory, 'real_hyp.dat')

        self.groundProblem = self.PDDLParser.getGroundDomainProblem(domainFilePath, initialFilePath)

        self.goals = self.PDDLParser.getGoals(self.groundProblem, goalsFilePath)

        # realGoal = PDDLParser.getGoals(groundProblem, realGoalFilePath).get(0)
        self.initialState = self.groundProblem.getSTRIPSInitialState()
        self.goalsToLandmarkExtractor = HashMap()
        self.goalsToFactLandmarks = HashMap()
        self.goalsToLandmarks = HashMap()
        self.goalsToOrderedLandmarks = HashMap()

        self.goalsToAchievedLandmarks = []

        self.L = set()
        self.currentState = self.initialState
        self.activeFL = set()
        self.FL = set()
        self.prune = [1] * len(self.goals)
        self.achievedLandmarks = dict()
        self.achieved = dict()

        for goal in self.goals:
            landmarkExtractor = self.goalsToLandmarkExtractor.get(goal)
            if landmarkExtractor is None:
                landmarkExtractor = self.PartialLandmarkGenerator(self.initialState, goal.getFacts(),
                                                                  self.groundProblem.getActions())
                landmarkExtractor.extractLandmarks()
                for landmarkOrdering in landmarkExtractor.getLandmarksOrdering():
                    lm = landmarkOrdering.getOrdering()
                    # self.last_landmarks = self.last_landmarks | set(lm[-1])
                    self.L = self.L | set(lm)
                    # for a in b:
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
            self.goalsToOrderedLandmarks.put(goal, landmarkExtractor.getLandmarksOrdering())

            self.achievedLandmarks[goal] = set()
            self.goalsToAchievedLandmarks.append(0)
            self.achieved[goal] = []

    def getStates(self):
        observationsFilePath = os.path.join(self.current_directory, 'mod_obs.dat')
        observations = self.PDDLParser.getObservations(self.groundProblem, observationsFilePath)

        currentState = self.initialState

        for o in observations:
            observedFacts = set()
            observedFacts.update(o.getAddPropositions())
            observedFacts.update(o.getPreconditions())
            observedFacts.update(currentState.getFacts())

            currentState = currentState.apply(o)

        return observedFacts

    def getProbability(self):
        observationsFilePath = os.path.join(self.current_directory, 'mod_obs.dat')
        observations = self.PDDLParser.getObservations(self.groundProblem, observationsFilePath)


        self.observedFacts = set()
        self.observedFacts.update(observations[-1].getAddPropositions())
        self.observedFacts.update(observations[-1].getPreconditions())
        self.observedFacts.update(self.currentState.getFacts())
        self.achieved_landmarks_fn(self.observedFacts)
        self.currentState = self.currentState.apply(observations[-1])



        k = 0
        count = []
        for goal in self.goals:
            self.achieved[goal].append(self.FL)
            count.append(0)
            # Computing achieved landmarks from observations for a candidate goal
            for landmarkOrdering in self.goalsToOrderedLandmarks.get(goal):
                for factsOrdering in landmarkOrdering.getOrdering():
                    if self.observedFacts.issuperset(factsOrdering) and factsOrdering not in self.achievedLandmarks[goal]:
                        self.achievedLandmarks[goal].add(factsOrdering)
                            # inferredFacts = self.inferFactLandmarks(landmarkOrdering, observedFacts, achievedLandmarks)
                            # achievedLandmarks.update(inferredFacts)



            for landmarkOrdering in self.goalsToOrderedLandmarks.get(goal):
                lk = landmarkOrdering.getOrdering()
                if self.FL & set([lk[-1]]):
                    self.prune[k] = 0

                eval = []
                for i in range(len(self.achieved[goal])):
                    for factsOrdering in lk:
                        if self.FL & set([factsOrdering]): #and factsOrdering not in self.common_landmark.get(goal):
                            eval.append(i)
                    # print(eval)
                    # print('............')
                    if not is_ascending(eval):
                        active[k] = 0

            if self.prune[k] == 1:
                self.goalsToAchievedLandmarks[k] = len(self.achievedLandmarks[goal]) / len(self.goalsToLandmarks[goal])
                count[k] = len(self.achievedLandmarks[goal])
            else:
                self.goalsToAchievedLandmarks[k] = 0
                count[k] = 0

            k += 1


        return self.goalsToAchievedLandmarks, self.prune, count
