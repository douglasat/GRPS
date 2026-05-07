import math
import ompl
import ompl.base as ob
import ompl.geometric as og

# Globals for the scenario map
scenario_map = []
scenario_step = 0.0


def createPlanner(si):
    return og.RRTstar(si)


def isValid(state):
    global scenario_map, scenario_step
    # Cast to SE2
    se2state = ob.SE2StateSpace()
    se2state = state

    x = se2state.getX()
    y = se2state.getY()

    # Check local neighborhood like in C++
    for step in range(4):
        radius = scenario_step * step
        for a in range(8):
            x_radius = round((math.cos(math.radians(45 * a)) * radius + x) / scenario_step)
            y_radius = round((math.sin(math.radians(45 * a)) * radius + y) / scenario_step)

            x_radius = max(0, min(511, int(x_radius)))
            y_radius = max(0, min(511, int(y_radius)))

            if (
                scenario_map[int(x_radius)][int(y_radius)] != "."
                or x_radius in [0, 511]
                or y_radius in [0, 511]
            ):
                return False
            if step == 0:
                break
    return True


def plan(start, goal, scenario_map1, scenario_step1, num_planners=1, seed=1):
    global scenario_map, scenario_step
    scenario_map = scenario_map1
    scenario_step = scenario_step1

    # Seed RNG
    ompl.util.RNG.setSeed(seed + 1)

    # Define SE(2) state space
    space = ob.SE2StateSpace()

    # Set bounds
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(0)
    bounds.setHigh(10)
    space.setBounds(bounds)

    # Space information
    si = ob.SpaceInformation(space)
    si.setStateValidityChecker(ob.StateValidityCheckerFn(isValid))

    # Problem definition
    pdef = ob.ProblemDefinition(si)

    # Start and goal states
    start_state = ob.State(space)
    start_state().setX(start[0])
    start_state().setY(start[1])

    goal_state = ob.State(space)
    goal_state().setX(goal[0])
    goal_state().setY(goal[1])

    pdef.setStartAndGoalStates(start_state, goal_state)

    # Planner
    planner = createPlanner(si)
    planner.setProblemDefinition(pdef)

    # Termination conditions
    # cond = ob.plannerOrTerminationCondition(
    #     ob.CostConvergenceTerminationCondition(pdef, 20, 0.01),
    #     ob.timedPlannerTerminationCondition(1),
    # )

    # Solve
    solved = planner.solve(ob.timedPlannerTerminationCondition(5))

    if solved:
        paths = pdef.getSolutions()
        output = []
        for solution in paths:
            path = solution.path_
            if path is not None:
                pg = og.PathGeometric(path)
                output.append(pg.printAsMatrix())
        return "\n".join(output)
    else:
        return "Path is null."