import ortools
from ortools.linear_solver import pywraplp
import argparse
import time


def create_data(data_file):
    d = []
    s = []
    e = []
    with open(data_file, "r") as f:
        head = f.readline().split(" ")
        N, m, M = int(head[0]), float(head[1]), float(head[2])
        for i in range(N):
            field_info = f.readline().split(" ")
            d.append(float(field_info[0]))
            s.append(int(field_info[1]))
            e.append(int(field_info[2]))
        data = {"M": M, "N": N, "m": m, "d": d, "e": e, "s": s}
        return data


def IP_solver(data):
    min_start = min(data["s"])
    for i in range(data["N"]):
        data["s"][i] -= min_start
        data["e"][i] -= min_start
    D = max(data["e"])

    start = time.time()

    solver = pywraplp.Solver.CreateSolver("SCIP")

    a = {}
    b = {}
    # Create variable
    for i in range(0, data["N"]):
        for j in range(0, D + 1):
            a[i, j] = solver.IntVar(0, 1, "a[{}_{}]".format(i, j))
    max_day = solver.NumVar(lb=0, ub=data["M"], name="max_day")
    min_day = solver.NumVar(lb=0, ub=data["M"], name="min_day")
    # Add constraint
    for i in range(0, data["N"]):
        for j in range(0, data["s"][i]):
            solver.Add(a[i, j] == 0)
        solver.Add(solver.Sum([a[i, j] for j in range(
            data["s"][i], data["e"][i] + 1)]) == 1)
        for j in range(data["e"][i] + 1, D + 1):
            solver.Add(a[i, j] == 0)
    for j in range(0, D + 1):
        b[j] = solver.IntVar(0, 1, "b[{}]".format(j))
    for j in range(0, D + 1):
        for i in range(0, data["N"]):
            solver.Add(b[j] >= a[i, j])
        solver.Add(solver.Sum([a[i, j] for i in range(0, data["N"])]) >= b[j])
    for j in range(0, D + 1):
        solver.Add(solver.Sum([a[i, j] * data['d'][i] for i in range(0, data["N"])]) <= max_day)
        solver.Add(solver.Sum([a[i, j] * data['d'][i] for i in range(0, data["N"])]) >= min_day * b[j])
        solver.Add(solver.Sum([a[i, j] * data['d'][i] for i in range(0, data["N"])]) >= data["m"] * b[j])
    solver.Minimize(max_day - min_day)
    solver.set_time_limit(600 * 1000)
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        solution = {}
        solution["name"] = "%d-%d-%d" % (data["N"], data["m"], data["M"])
        solution["obj"] = solver.Objective().Value()
        solution["timetable"] = []
        solution["field"] = []
        solution["quantity"] = []
        solution["time"] = time.time() - start
        print(solution["time"])
        for i in range(data["N"]):
            for j in range(0, D + 1):
                if a[i, j].solution_value() == 1:
                    solution["field"].append(j + min_start)
                    break
        for j in range(0, D + 1):
            days = []
            for i in range(data["N"]):
                if a[i, j].solution_value() == 1:
                    days.append(i)
            solution["timetable"].append(days)
            solution["quantity"].append(
                sum([a[i, j].solution_value() * data["d"][i] for i in range(0, data["N"])]))
        return solution


def export(output_file, solution):
    if solution is not None:
        with open(output_file, "w+") as f:
            f.write("Name: %s\n" % solution["name"])
            f.write("Time: %fs\n" % solution["time"])
            f.write("Result: %d\n" % int(solution["obj"]))
            f.write("Solution: %s\n" %
                    (' '.join([str(x) for x in solution["field"]])))
            for i, (days, amount) in enumerate(zip(solution["timetable"], solution["quantity"])):
                f.write("%d: %s (%d)\n" %
                        (i, ' '.join([str(x) for x in days]), amount))
    else:
        with open(output_file, "w+") as f:
            f.write("No solution")


def process(input_file, output_file):
    data = create_data(input_file)
    solution = IP_solver(data)
    export(output_file, solution)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    obj = parser.parse_args()
    input_file = obj.input_file
    output_file = obj.output_file
    process(input_file, output_file)
