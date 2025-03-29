import pandas as pd 
import cvxpy as cp 
import numpy as np

def read_battery_list(file_name:str ='battery-list.csv') -> np.ndarray:
    battery_values = pd.read_csv(file_name)
    return battery_values.to_numpy()

def create_problem_variables(battery_list : list[tuple, float], 
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
                       battery_list: list[tuple, float]):
    constraints = []
    for row in variable_matrix:
        constraints.append(row >= 0)
        constraints.append(row <= 4)
        constraints.append(cp.sum(row) <= 4)
    
    for idx, column in enumerate(variable_matrix.T):
        constraints.append(cp.sum(column) <= battery_list[idx,0])

    return constraints 

def create_problem(number_of_bins : int = 2) -> cp.Problem:
    battery_list = read_battery_list()
    variables = create_problem_variables(battery_list, number_of_bins)
    obj_function = create_objective_function(variables, battery_list)
    constraints = create_constraints(variables, battery_list)
    problem = cp.Problem(obj_function, constraints)
    return problem

def solve_problem(opt_problem : cp.Problem):
    problem.solve()
    print("status:", problem.status)

def print_solution(opt_problem : cp.Problem):
    var_matrix = opt_problem.variables()[0]
    for row in var_matrix.value:
        print(np.round(row))

if __name__ == '__main__':
    #battery_list = read_battery_list()
    #print(battery_list[:, 1])
    problem = create_problem()
    print(problem)
    print(problem.is_dcp())
    solve_problem(problem)
    print_solution(problem)

