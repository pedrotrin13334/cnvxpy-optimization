import pandas as pd 
import cvxpy as cp 
import numpy as np
from enum import Enum

import sys

def read_battery_list(file_name:str ='battery-list.csv') -> np.ndarray:
    battery_values = pd.read_csv(file_name)
    return battery_values.to_numpy()

def create_problem_variables(battery_list : list[int, float], 
                            number_of_bins : int) -> list[list[cp.Variable]]:

    #variables_for_each_bin = cp.Variable((number_of_bins,len(battery_list)), integer=True)
    variables_for_each_bin = cp.Variable((number_of_bins,len(battery_list)))

    return variables_for_each_bin

def create_objective_function(variable_matrix: cp.Variable,
                              battery_list: list[tuple, float]):
    voltages = cp.Parameter(len(battery_list), nonneg=True)
    voltages.value = np.array(battery_list[:, 1])
    
    objective = cp.Maximize(cp.sum(cp.log(variable_matrix @ voltages)))
    return objective

def create_constraints(variable_matrix: cp.Variable,
                       battery_list: list[tuple, float],
                       is_relaxed = False):
    constraints = []
    for row in variable_matrix:
        constraints.append(row >= 0)
        constraints.append(row <= 4)
        if(is_relaxed):
            constraints.append(cp.sum(row) <= 5)
        else:
            constraints.append(cp.sum(row) <= 4)
        constraints.append(cp.sum(row) >= 4)
    
    for idx, column in enumerate(variable_matrix.T):
        constraints.append(cp.sum(column) <= battery_list[idx,0])

    return constraints 

def create_problem(number_of_bins : int = 2,
                   is_relaxed : bool = False) -> cp.Problem:
    battery_list = read_battery_list()
    variables = create_problem_variables(battery_list, number_of_bins)
    obj_function = create_objective_function(variables, battery_list)
    constraints = create_constraints(variables, battery_list, is_relaxed=is_relaxed)
    problem = cp.Problem(obj_function, constraints)
    return problem

def solve_problem(opt_problem : cp.Problem):
    opt_problem.solve()
    #print("status:", opt_problem.status)

def check_battery_choice_constraint(rounded_solutions : list[list[int]]) -> bool:
    """
        It is necessary to double check if the value of batteries are 
        actually summing to four, otherwise we might have to solve a relaxed
        solution.
    """
    return all(np.sum(np.array(rounded_solutions), axis=1) == 4)

def get_rounded_solutions(var_matrix : cp.Variable):
    rounded_solutions = []
    for row in var_matrix.value:
        rounded_solution = np.round(row)
        rounded_solutions.append(rounded_solution)
    return rounded_solutions

class ERROR_CODE(Enum):
    NO_ERROR = 0
    INFEASIBLE = 2
    FAILED_BATTERY_SUM_COND = 3

def battery_problem_solve(num_of_bins : int = 2, is_relaxed : bool =False)-> tuple[bool,cp.Problem]:
    problem = create_problem(number_of_bins=num_of_bins,
                             is_relaxed=is_relaxed)
    solve_problem(problem)
    var_matrix = problem.variables()[0]

    ## If on first try everything is okay, return the problem,
    ## if feasibillity is not achieved, abort. 
    ## if only the solution doesn't meet the battery sum criteria, 
    ## relax the problem. 
    if(problem.status == cp.OPTIMAL):
        rounded_solutions = get_rounded_solutions(var_matrix)
        if(check_battery_choice_constraint(rounded_solutions) == False):
            return ERROR_CODE.FAILED_BATTERY_SUM_COND, problem
        else:
            return ERROR_CODE.NO_ERROR, problem
    else:
        return ERROR_CODE.INFEASIBLE, None

def print_solution(opt_problem : cp.Problem,
                   battery_list : tuple[int, float]):
    var_matrix = opt_problem.variables()[0]
    rounded_solutions = []
    for row in var_matrix.value:
        print(np.sum(row))
        rounded_solution = np.round(row)
        rounded_solutions.append(rounded_solution)
        print(rounded_solution)
    
    print("Total Voltage Per Bin: ")
    total_voltages = []
    individual_voltages = np.array(battery_list[:, 1])
    for row_sol in rounded_solutions: 
        total_voltage = np.sum(individual_voltages @ np.array(row_sol))
        total_voltages.append(total_voltage)
        print(total_voltage)

def str_2_bool(s:str) -> bool:
    return s.lower() in {'true', '1', 't', 'yes'}

if __name__ == '__main__':
    battery_list = read_battery_list()
    np.set_printoptions(precision=2)

    ## Initial values to be changed by arguments
    is_relaxed = False 
    num_of_bins = 2 
    
    args_num = len(sys.argv)
    if(not(args_num == 3 or args_num == 2)):
        print("Wrong Number of Arguments!")
    else:
        try:
            num_of_bins = int(sys.argv[1])
            if(args_num == 3):
                is_relaxed = str_2_bool(sys.argv[2])
        except Exception as error:
            print(f"Error: {error}")
            exit()

    error_code, result = battery_problem_solve(num_of_bins=num_of_bins, is_relaxed=is_relaxed)

    if(error_code == ERROR_CODE.INFEASIBLE):
        print("Could not solve the problem properly")
        print("Maybe the problem was not well posed")
    elif(error_code == ERROR_CODE.FAILED_BATTERY_SUM_COND):
        print_solution(result, battery_list)
        print("Could not get a solution which matched the battery sum for each bin")
        print("Try the relaxed version if not already done")
    elif(error_code == ERROR_CODE.NO_ERROR):
        print_solution(result, battery_list)
        print("\nEverything Went Well!")
    else:
        print("Something weird happened")
    

