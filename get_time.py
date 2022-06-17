import csv
import sys


def get_platform_name(platform):
    # encode platform
    if platform == 1:
        return "device_edge"
    elif platform == 2:
        return "cloud_edge"
    elif platform == 3:
        return "cloud"

    return "unknown"


def get_invoke_time(function, parameter, platform, cpu, memory):
    '''
    从文件中读取 invoke time, 如果不存在则返回 999
    '''
    if platform == 1 or platform == 2 or platform == 3:
        platform = get_platform_name(platform=platform)

    if platform == "device_edge":
        platform = "openfaas"
    elif platform == "cloud_edge":
        platform = "knative"
    elif platform == "cloud":
        platform = "scknative"

    path = '/home/jupyter/workspace/ygb/farad/latency_calculate/latency_results/' + \
        function + '/' + platform + '_' + \
        function + '_' + parameter + '_average.csv'
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['CPU'] == str(cpu) and row['memory'] == str(memory):
                return float(row['invoketime'])

    print("CPU: %s , Memory %s Unable to find invoke time" %
          (str(cpu), str(memory)))
    return 999


def get_runtime(function, parameter, platform, cpu, memory):
    '''
    从文件中读取 runtime time, 如果不存在则返回 999
    '''
    if platform == 1 or platform == 2 or platform == 3:
        platform = get_platform_name(platform=platform)

    if platform == "device_edge":
        platform = "openfaas"
    elif platform == "cloud_edge":
        platform = "knative"
    elif platform == "cloud":
        platform = "scknative"

    path = '/home/jupyter/workspace/ygb/farad/latency_calculate/latency_results/' + \
        function + '/' + platform + '_' + \
        function + '_' + parameter + '_average.csv'
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['CPU'] == str(cpu) and row['memory'] == str(memory):
                return float(row['runtime'])
    print("CPU: %s , Memory %s Unable to find runtime" %
          (str(cpu), str(memory)))
    return 999


def get_cpu(cpu_config):
    '''CPU Action'''
    return cpu_config * 25


def get_memory(memory_config):
    '''Memory Action'''
    return memory_config * 32


if __name__ == "__main__":
    platform = ["openfaas", "knative", "scknative"]
    fdps = [(100, 128), (250, 256), (500, 512), (1000, 1024), (2000, 2048)]
    function = "imageinception"
    parameter = "1351"

    result_file = function + ".csv"
    with open(result_file, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["platform", "fdp", "invoke_time",
                         "runtime", "cost"])

    with open(result_file, "a+", newline='') as csvfile:
        writer = csv.writer(csvfile)
        for platform_name in platform:
            for fdp in fdps:
                invoke_time = get_invoke_time(function=function, parameter=parameter,
                                              platform=platform_name, cpu=fdp[0], memory=fdp[1])

                runtime = get_runtime(function=function, parameter=parameter, platform=platform_name,
                                      cpu=fdp[0], memory=fdp[1])
                cost = runtime * (fdp[0] + fdp[1])
                writer.writerow(
                    [platform_name, fdp, invoke_time, runtime, cost])
