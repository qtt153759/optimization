import ortools
from ortools.sat.python import cp_model
import argparse
import time


def create_data(data_file):
    data = {}
    with open(data_file, "r") as f:
        line = f.readline().split(' ')
        data['N'] = int(line[0])
        data['m'] = int(line[1])
        data['M'] = int(line[2])

        data['d'] = {}
        data['s'] = {}
        data['e'] = {}
        for i in range(data['N']):
            line = f.readline().split(' ')
            data['d'][i] = int(line[0])
            data['s'][i] = int(line[1])
            data['e'][i] = int(line[2])
        data['start'] = min(data['s'].values())
        data['end'] = max(data['e'].values())
        return data


def CP_solver(data):
    model = cp_model.CpModel()
    start_time = time.time()

    # x[i,j] - cánh đồng i thu hoạch vào ngày j
    x = {}
    for i in range(data['N']):
        for j in range(data['start'], data['end']+1):
            x[i, j] = model.NewIntVar(0, 1, 'x['+str(i)+','+str(j)+']')
    max_day = model.NewIntVar(0, data['M'], 'max_day')
    min_day = model.NewIntVar(0, data['M'], 'min_day')

    # day[j] - sản lượng thu hoạch ngày j
    day = {}
    for j in range(data['start'], data['end']+1):
        day[j] = model.NewIntVar(0, data['M'], 'day['+str(j)+']')

    # mỗi cánh đồng chỉ thu hoạch trong một ngày
    for i in range(data['N']):
        model.Add(sum(x[i, j]
                  for j in range(data['start'], data['end']+1)) == 1)

    # cánh đồng i thu hoạch trong khoảng ngày [si, ei]
    for i in range(data['N']):
        model.AddLinearConstraint(sum(
            x[i, j]*j for j in range(data['start'], data['end']+1)), data['s'][i], data['e'][i])

    # mngày j có sản lượng day[j]
    for j in range(data['start'], data['end']+1):
        model.Add(day[j] == sum(x[i, j]*data['d'][i]
                  for i in range(data['N'])))

    # Nếu day[j] != 0 thi m <= day[j] <= M
    for j in range(data['start'], data['end']+1):
        b = model.NewBoolVar('b')
        model.Add(day[j] != 0).OnlyEnforceIf(b)
        model.Add(day[j] == 0).OnlyEnforceIf(b.Not())
        model.Add(day[j] >= data['m']).OnlyEnforceIf(b)

    model.AddMaxEquality(max_day, [day[j]
                         for j in range(data['start'], data['end']+1)])
    model.AddMinEquality(min_day, [day[j]
                         for j in range(data['start'], data['end']+1)])

    model.Minimize(max_day - min_day)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 600
    status = solver.Solve(model)

    result = []
    sum_day = []
    solution = []
    obj = 0
    name = str(data['N']) + '-' + str(data['m']) + '-' + str(data['M'])
    run_time = time.time() - start_time

    if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        obj = int(solver.ObjectiveValue())
        for j in range(data['start'], data['end']+1):
            result.append([])
            for i in range(data['N']):
                if solver.Value(x[i, j]) == 1:
                    result[j].append(i)
            sum_day.append(int(solver.Value(day[j])))
        for i in range(data['N']):
            for j in range(data['start'], data['end']+1):
                if solver.Value(x[i, j]) == 1:
                    solution.append(j)
        return name, obj, result, sum_day, solution, run_time
    return 1

def export(output_file, name, obj, result, sum_day, solution, time):
    with open(output_file, "w+") as f:
        f.write("Name: "+name+"\n")
        f.write("Time: {} \n".format(time))
        f.write("Result: {}\n".format(obj))
        f.write("Solution: ")
        for j in solution:
            f.write("{} ".format(j))
        f.write('\n')
        n = 0
        for k in result:
            f.write('{}: '.format(n))
            for i in k:
                f.write('{} '.format(i))
            f.write('({})\n'.format(sum_day[n]))
            n += 1

def process(input_file, output_file):
    data = create_data(input_file)
    if CP_solver(data) == 1:
        with open(output_file, "w+") as f:
            f.write("No solution")
    else:
        name, obj, result, sum_day, solution, run_time = CP_solver(data)
        export(output_file, name, obj, result, sum_day, solution, run_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    obj = parser.parse_args()
    input_file = obj.input_file
    output_file = obj.output_file
    process(input_file, output_file)
