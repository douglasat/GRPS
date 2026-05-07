from math import sin, cos, tan
import generate_scenario
import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys as sys
import os
import time
import optimalTrajectory
import output
import concurrent.futures
from itertools import combinations_with_replacement
import signature
from tree import Tree
from tree import Node
import graphviz
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import pickle
import argparse

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")


def velMax(a, viaPoints, vmax, u, i):  # constrains of velocity
    np.random.seed(12345)
    u[i] = a
    for k in range(i, i + 1):
        if k == 0:
            velx = u[k][0]
            vely = u[k][1]
            tf = u[k][2]

            s = [viaPoints[0][0], viaPoints[0][1], viaPoints[1][0], viaPoints[1][1], 0, 0]

            coefX = optimalTrajectory.coefPoli5(s[0], s[2], s[4], velx, tf)
            qx, qdotx = optimalTrajectory.poli5(coefX, tf)

            coefY = optimalTrajectory.coefPoli5(s[1], s[3], s[5], vely, tf)
            qy, qdoty = optimalTrajectory.poli5(coefY, tf)

        else:
            velx = u[k][0]
            vely = u[k][1]
            tf = u[k][2]

            s = [viaPoints[k][0], viaPoints[k][1], viaPoints[k + 1][0], viaPoints[k + 1][1], u[k - 1][0], u[k - 1][1]]

            coefX = optimalTrajectory.coefPoli5(s[0], s[2], s[4], velx, tf)
            qx, qdotx = optimalTrajectory.poli5(coefX, tf)

            coefY = optimalTrajectory.coefPoli5(s[1], s[3], s[5], vely, tf)
            qy, qdoty = optimalTrajectory.poli5(coefY, tf)

    dot_max = np.max(np.linalg.norm(np.array([qdotx, qdoty]).T, axis=1))

    return 100 * (vmax - dot_max)


def rolloutViapoints(u, viaPoints):  # rollout using the via points to compute a full trajectory
    x = []
    y = []
    dotx = []
    doty = []
    for k in range(len(viaPoints) - 1):
        if k == 0:
            velx = u[k][0]
            vely = u[k][1]
            tf = u[k][2]

            s = [viaPoints[0][0], viaPoints[0][1], viaPoints[1][0], viaPoints[1][1], 0, 0]

            coefX = optimalTrajectory.coefPoli5(s[0], s[2], s[4], velx, tf)  # coeficientes do X
            qx, qdotx = optimalTrajectory.poli5(coefX, tf)

            coefY = optimalTrajectory.coefPoli5(s[1], s[3], s[5], vely, tf)  # coeficientes do Y
            qy, qdoty = optimalTrajectory.poli5(coefY, tf)

            x.extend(qx)
            y.extend(qy)
            dotx.extend(qdotx)
            doty.extend(qdoty)

        else:
            velx = u[k][0]
            vely = u[k][1]
            tf = u[k][2]

            s = [viaPoints[k][0], viaPoints[k][1], viaPoints[k + 1][0], viaPoints[k + 1][1], u[k - 1][0], u[k - 1][1]]

            coefX = optimalTrajectory.coefPoli5(s[0], s[2], s[4], velx, tf)  # coeficientes do X
            qx, qdotx = optimalTrajectory.poli5(coefX, tf)

            coefY = optimalTrajectory.coefPoli5(s[1], s[3], s[5], vely, tf)  # coeficientes do Y
            qy, qdoty = optimalTrajectory.poli5(coefY, tf)

            x.extend(qx[1:])
            y.extend(qy[1:])
            dotx.extend(qdotx[1:])
            doty.extend(qdoty[1:])

    return x, y, dotx, doty


def fun_Policy(u, viaPoints):  # cost function getEstimationPath
    u = np.array(np.split(u, len(viaPoints) - 1))
    return sum(u[:, 2])
    #return u[2]


def interpolate_path(viapoints, max_dist):
    # Generate a smooth set of points along the spline
    t = np.linspace(0, 1, len(viapoints))
    cs_x = sp.interpolate.interp1d(t, viapoints[:, 0], kind='linear')
    cs_y = sp.interpolate.interp1d(t, viapoints[:, 1], kind='linear')

    # Interpolated t values for smooth path
    t_smooth = np.linspace(0, 1, 200)
    inter_x = cs_x(t_smooth)
    inter_y = cs_y(t_smooth)

    viapoints_mod = [[inter_x[0], inter_y[0]]]
    for x, y in zip(inter_x, inter_y):
        if len(viapoints_mod) == 1 and np.linalg.norm(viapoints_mod[-1] - np.array([x, y])) >= 0.5:
            viapoints_mod.append([x, y])
            continue

        if np.linalg.norm(viapoints_mod[-1] - np.array([x, y])) >= max_dist or [x, y] == [inter_x[-1], inter_y[-1]]:
            viapoints_mod.append([x, y])
    
    viapoints_mod = np.array(viapoints_mod)

    return viapoints_mod


def velMax_group(u, viaPoints, vmax):
    vel = []
    u = np.split(u, len(viaPoints) - 1)
    for k in range(len(viaPoints) - 1):
        a = u[k]
        vel.append(velMax(a, viaPoints, vmax, u, k))

    return vel    


def compute_approximation(init, g, seed, data_memorize):
    #Memorize
    if memorize:
        data_problem = data_memorize.get((str(init), str(g), seed), None)
    else:
        data_problem = None

    if not data_problem == None:
        return data_problem
    else:
        np.random.seed(12345)
        dist_viapoints = 1
        # print("Computing an approximated trajectory to Goal:", gk)
        viaPoints = np.array(optimalTrajectory.geometric_plan(init, g, scenario, seed,'single')[0])  # compute the via points
        viaPoints = interpolate_path(viaPoints, dist_viapoints)
        
        bnd = []
        u = []
        for _ in range(len(viaPoints) - 1):
            bnd.extend([(-scenario.vmax, scenario.vmax), (-scenario.vmax, scenario.vmax), (0.2, 1.2*dist_viapoints/scenario.vmax)])  # bounds
            u.extend([0, 0, 1.2*dist_viapoints/scenario.vmax])
        
        ineq_cons1 = {'type': 'ineq', 'fun': lambda z: velMax_group(z, viaPoints, scenario.vmax)}  # constrains

        result = sp.optimize.minimize(fun_Policy, u, args=(viaPoints), options={'maxiter': 500}, bounds=bnd, 
                            constraints=ineq_cons1, method='SLSQP')

        u = np.array(np.split(result.x, len(viaPoints) - 1))
        qx, qy, qdotx, qdoty = rolloutViapoints(u, viaPoints)  # compute the full path trajectory with all viapoints terms
        
        path = np.array([qx, qy])
        sig_path = sig.get_all_signatures(path.T)

        return [[qx, qy, qdotx, qdoty, sig_path]]


def nodes_traj_to_signature_traj(node_traj):
    traj = []
    traj.append((1,))
    for node in node_traj:
        if not node.identifier == (1,):
            traj.append(node.identifier)
    return traj 


def getEstimationPath(init, goals, data_memorize):  # compute an estimation from init to each goal in the set
    id_init = np.where(goals==init)[0][0]
    
    # Time Control
    start_time = time.time()
    cumulative_time = 0
    
    # memorize
    data_problem = {}
    if time_memorize:
        time_dict = {}
        if not os.path.exists(f'./{scenario.name}/memorize/time.pkl'):
            with open('./%s/memorize/time.pkl' % (scenario.name), "wb") as pkl_file:
                pickle.dump(time_dict, pkl_file)
        else:
            with open('./%s/memorize/time.pkl'  % (scenario.name), "rb") as pkl_file:
                time_dict = pickle.load(pkl_file)
    
    # path signature trees
    trees = {}
    roots = {}
    for g in scenario.goalPoints:
        tree = Tree(merge_threshold=merge, prune_threshold=prune)
        root = Node(identifier=(1,), data=[], level=0)
        tree.add_node(root)
        trees[str(g)] = tree
        roots[str(g)] = root
    
    #sig_branch_by_goal = {}
    sig_level_by_goal = {}
    for gk, g in enumerate(goals):
        if str(init) != str(g):
            x_eval = []
            y_eval = []
            dotx_eval = []
            doty_eval = []
            sig_level_by_goal[str(g)] = []
            sig_path = {}

            tic = time.time()
            # Create a ProcessPoolExecutor
            with concurrent.futures.ProcessPoolExecutor(max_workers=parallel_topk) as executor:
                # Submit tasks to the executor
                futures = [executor.submit(compute_approximation, init, g, k, data_memorize) for k in range(top_k)]

                # Wait for all tasks to complete
                completed_futures, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)    

            # Memorize offline time
            tac = time.time() - tic
            if time_memorize:
                time_dict[(scenario.name, id_init, gk, top_k)] = tac
                with open('./%s/memorize/time.pkl' % (scenario.name), "wb") as pkl_file:
                    pickle.dump(time_dict, pkl_file)
            else:
                with open('./%s/memorize/time.pkl' % (scenario.name), "rb") as pkl_file:
                    time_dict = pickle.load(pkl_file)
                cumulative_time += time_dict.get((scenario.name, id_init, gk, top_k)) - tac

            # Memorize data update for each topk sulution
            if memorize_save:
                for future, k in zip(futures, range(len(futures))):
                    if data_problem.get((str(init), str(g), k), None) == None:
                        data_problem[(str(init), str(g), k)] = future.result()

            # Retrieve results in the order of task submission
            for future, k in zip(futures, range(len(futures))):
                for q in future.result():
                    x_eval.append(q[0])
                    y_eval.append(q[1])
                    dotx_eval.append(q[2])
                    doty_eval.append(q[3])
                    sig_path[str(k)] = q[4]
            
            for t, signatures in sig_path.items():
                previous = roots[str(g)]
                for signature in signatures:
                    if not trees[str(g)].contains(signature):
                        node = Node(identifier=signature, data=[t])
                        trees[str(g)].add_node(node, None if signature == (1,) else previous)
                    else:
                        node = trees[str(g)].get_node(signature)
                        node.data.append(t)
                        node.data = list(set(node.data))

                    previous = node
            
            trees[str(g)].naming()
            trees[str(g)].merge()
            trees[str(g)].prune()
            #trees[str(g)].render("output", "tree.gv")
            
            # Align trajectory by level
            k = 0
            last_node = 0
            max_data_legth = 0
            nodes = trees[str(g)].get_node_by_level(k)
            while len(nodes) > 0:
                level_k = []
                length_data = []
                for node in nodes:
                    level_k.append(node.identifier)
                    length_data.append(node.data)
                
                max_data_legth = max(max_data_legth, len(length_data))
                sig_level_by_goal[str(g)].append(level_k)
                last_node = k
                k = k + 1
                nodes = trees[str(g)].get_node_by_level(k)

            # Align trajectory by branch
            # max_iter = k
            # k = 0
            # sig_branch = []
            # visited_branches = []
            # while not len(sig_branch) == max_data_legth and k < max_iter: 
            #     nodes = trees[str(g)].get_node_by_level(last_node - k)
            #     for node in nodes:
            #         if not any(int(elem) in visited_branches for elem in node.data):
            #             visited_branches.append(int(node.data[0]))
            #             all_nodes = trees[str(g)].get_all_nodes_by_branch(int(node.data[0]))
            #             sig_branch.append(nodes_traj_to_signature_traj(all_nodes))
  
            #             if len(sig_branch) == max_data_legth:
            #                 break
                
            #     k = k + 1

            # sig_branch_by_goal[str(g)] = sig_branch     

    offline_time = time.time() - start_time + cumulative_time
    return sig_level_by_goal, data_problem, offline_time

  
def find_max_prop(vector):
    max_value = np.min(vector)
    max_prop = [index for index, value in enumerate(vector) if max_value == value]
    # max_prop = [index for index, value in enumerate(vector) if max_value-0.10 <= value <= max_value]

    return max_prop


def vector_inference_multi(initial, goal):
    # Memorize data creation for each problem
    if memorize_save:
         data_memorize = {}
         if not os.path.exists('./%s/memorize/data_memorize.pkl' % (scenario.name)):
            with open('./%s/memorize/data_memorize.pkl' % (scenario.name), "wb") as pkl_file:
                pickle.dump(data_memorize, pkl_file)
    else:
        with open('./%s/memorize/data_memorize.pkl' % (scenario.name), "rb") as pkl_file:
            data_memorize = pickle.load(pkl_file)

    result_list = []
    state_init = scenario.goalPoints[initial]
    goal_Hypothesis = np.delete(scenario.goalPoints, initial, axis=0)

    # Load the optimal observations computed with optimalTrajectory.py
    loaded_data = np.load('./%s/group%d/stateData%d%d.npz' % (scenario.name, group_number, initial, goal), allow_pickle=True)
    O_Optimal = loaded_data['O_Optimal']

    # chose the observations points
    sampled_obser = optimalTrajectory.sample_observations(O_Optimal, num_obser)

    # compute the offline part of the Estimation method
    print("Computing recognition inference problem:%d%d" % (initial, goal))
    # print('Group:', group_number)

    sig_by_goal, data_problem, offline_time = getEstimationPath(state_init, scenario.goalPoints, data_memorize)
    
    # compute the online part of the Estimation method + path signature
    sum_planner = len(scenario.goalPoints) - 1
    solution_set = []
    start_time = time.time()
    for obs in sampled_obser:
        sample_now = sampled_obser.index(obs) + 1
        # print('Evaluating observation %d of 6' % sample_now)
        
        # prob = []
        # sig_path_observations = sig.get_all_signatures(O_Optimal[0:2, :obs + 1].T)
        # for goal_Hypothese in goal_Hypothesis:
        #     group = []
        #     for traj in nodes_traj[str(goal_Hypothese)]:
        #         if len(traj) > 1:
        #             n = min(len(sig_path_observations[1:]), len(traj[1:obs + 1]))
        #             t = np.array(traj[1:obs + 1])
        #             s = sig_path_observations[1:]
        #             distance = np.mean(np.linalg.norm(s[-n:] - t[-n:], axis=1))
        #             group.append(distance)
        #         else:
        #             group.append(1e10)
            
        #     prob.append(np.min(group))

        prob = []
        sig_path_observations = np.array(sig.get_signature(O_Optimal[0:2, 0:obs + 1].T)[0])
        for goal_Hypothese in goal_Hypothesis:
            sig_group = []
            signatures_by_level = sig_by_goal[str(goal_Hypothese)]
            if len(signatures_by_level) - 1 >= obs:
                for sig_g in signatures_by_level[obs]:
                    if not sig_g == (1,): 
                        error = np.linalg.norm(sig_path_observations - np.array(sig_g)) # inference process
                        sig_group.append(error)
                    else:
                        sig_group.append(1e10)   
            else:
                for sig_g in signatures_by_level[-1]:
                    if not sig_g == (1,):
                        error = np.linalg.norm(sig_path_observations - np.array(sig_g)) # inference process
                        sig_group.append(error)
                    else:
                        sig_group.append(1e10)

            prob.append(np.min(sig_group))
        
        prob = np.insert(prob, initial, 1e6)    
        solution_set.append(find_max_prop(prob))

    online_time = time.time() - start_time
    result_list.append([initial, goal, solution_set, sum_planner, online_time, offline_time, data_problem])

    return result_list  


def process_input():
    """Main function to handle inputs and call processing logic."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Inputs")
    parser.add_argument(
        "-map", "--scenario_map", type=str, required=True, help="The first parameter (str)."
    )
    parser.add_argument(
        "-par", "--parallel", type=int, required=True, help="The first parameter (int)."
    )
    parser.add_argument(
        "-m", "--merge", type=float, required=True, help="The first parameter (float)."
    )
    parser.add_argument(
        "-p", "--prune", type=float, required=True, help="The second parameter (float)."
    )
    parser.add_argument(
        "-k", "--topk", type=int, required=True, help="The second parameter (float)."
    )
    
    args = parser.parse_args()
    return args.scenario_map, args.parallel, args.merge, args.prune, args.topk


if __name__ == "__main__":
    scenario_map, num_cores, merge, prune, top_k = process_input()

    # path signature class
    sig = signature.Signatures(device="cpu")

    # create scenario
    scenario = generate_scenario.Scenario(scenario_map)
    groupPoints = np.load('./%s/groupPoints.npy' % scenario.name, allow_pickle=True)

    #Memorize
    memorize = True
    if not os.path.exists('./%s/memorize' % scenario.name) and memorize:
        os.makedirs('./%s/memorize' % scenario.name)

    # output files
    output = output.OutputData(scenario.name, scenario.num_obser, len(groupPoints[0]) - 1, 'grps')

    # constrains and experiments definitions
    vmax = scenario.vmax  # velocity constrain
    omegamax = scenario.omegamax  # angular velocity constrain
    num_obser = scenario.num_obser  # number of observations to compare

    prop_data = []
    group_number = 0
    for points in groupPoints:
        scenario.goalPoints = points

        problem_number = [[initial, goal] for initial in range(0, len(scenario.goalPoints)) for goal in range(0, len(scenario.goalPoints)) if initial != goal]

        first_problem = [[0,1], [1,0], [2,0], [3,0], [4,0], [5,0], [6,0], [7,0]]
        parallel_topk = top_k
        memorize = False
        memorize_save = True
        time_memorize = True

        # obs = vector_inference_multi(2, 1)[0]
        # data_problem = obs[6]
        # output.save_probability(obs[0], obs[1], obs[2], obs[3], obs[4], obs[5])
        # bla

        for prop in first_problem:
            obs = vector_inference_multi(prop[0], prop[1])[0]   
            output.save_probability(obs[0], obs[1], obs[2], obs[3], obs[4], obs[5])
            
            # Memorize data update for each topk sulution
            if memorize_save:
                data_problem = obs[6]
                with open('./%s/memorize/data_memorize.pkl' % (scenario.name), "rb") as pkl_file:
                    data = pickle.load(pkl_file)
                    
                for d in data_problem:
                    data[d] = data_problem.get(d)

                with open("./%s/memorize/data_memorize.pkl" % (scenario.name), "wb") as pkl_file:
                    pickle.dump(data, pkl_file)
        
        parallel_topk = 1
        memorize = True
        memorize_save = False
        time_memorize = False
        problem_number = [p for p in problem_number if not p in first_problem]
        
        # Create a ProcessPoolExecutor
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
            # Submit tasks to the executor
            futures = [executor.submit(vector_inference_multi, prop[0], prop[1]) for prop in problem_number]

            # Wait for all tasks to complete
            completed_futures, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)

            # Retrieve results in the order of task submission
        for future in futures:
            for obs in future.result():
                output.save_probability(obs[0], obs[1], obs[2], obs[3], obs[4], obs[5])


        group_number += 1
        if group_number > 0:
            break
