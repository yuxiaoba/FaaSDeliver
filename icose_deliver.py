from requests import request
import numpy as np
from get_time import *
from log import *
import logging
import datetime
import sys
import os
import copy
from bayes_opt import UtilityFunction
from bayes_opt import BayesianOptimization
from platform_resource import Platform_Resource
from generate_workload import *
np.set_printoptions(suppress=True)


penalty_factor = 100
knative_cost = 4
scknative_cost = 6

discrete_bo_release_list = []
function_list = {}
discribe_bo_plat_resource = Platform_Resource()


def consume_resource(platform_name, node, cpu, memory, runtime, request_time):
    # 消耗资源
    global discrete_bo_release_list
    if runtime == 998.0:
        release_time = request_time + 0.1
    else:
        release_time = request_time + runtime

    consume_success = discribe_bo_plat_resource.del_resource(
        platform_name, node, cpu, memory)
    if consume_success:
        discrete_bo_release_list.append({
            "release_time": release_time,
            "platform_name": platform_name,
            "node": node,
            "cpu":  cpu,
            "memory": memory
        })
    return consume_success


def release_resource(request_time):
    global discrete_bo_release_list
    if len(discrete_bo_release_list) > 0:
        discrete_bo_release_list = sorted(
            discrete_bo_release_list, key=lambda i: i['release_time'], reverse=False)

        while request_time > discrete_bo_release_list[0]['release_time']:
            discribe_bo_plat_resource.add_resource(
                discrete_bo_release_list[0]['platform_name'], discrete_bo_release_list[0]['node'], discrete_bo_release_list[0]['cpu'], discrete_bo_release_list[0]['memory'])
            discrete_bo_release_list.pop(0)

            if len(discrete_bo_release_list) == 0:
                break


def split_list_n_list(origin_list, n):
    if len(origin_list) % n == 0:
        cnt = len(origin_list) // n
    else:
        cnt = len(origin_list) // n + 1

    split_list = []
    for i in range(0, n):
        split_list.append(origin_list[i*cnt:(i+1)*cnt])

    return split_list


def reward_funtion(function, parameter, platform, cpu_config, memory_config, slo):
    cpu = get_cpu(cpu_config)
    memory = get_memory(memory_config)
    invoke_time = get_invoke_time(
        function=function, parameter=parameter, platform=platform, cpu=cpu, memory=memory)

    runtime = get_runtime(
        function=function, parameter=parameter, platform=platform, cpu=cpu, memory=memory)

    if platform == 1:
        if invoke_time > slo:
            return -1 * (cpu + memory) * runtime * runtime / slo * penalty_factor, runtime
        else:
            return -1 * (cpu + memory) * runtime, runtime

    elif platform == 2:
        if invoke_time > slo:
            return -1 * (cpu + memory) * runtime * runtime / slo * knative_cost * penalty_factor, runtime
        else:
            return -1 * (cpu + memory) * runtime * knative_cost, runtime

    elif platform == 3:
        if invoke_time > slo:
            return -1 * (cpu + memory) * runtime * runtime / slo * scknative_cost * penalty_factor, runtime
        else:
            return -1 * (cpu + memory) * runtime * scknative_cost, runtime

    return 99999, 0.1


def get_cost(function, parameter, platform, cpu_config, memory_config):
    cpu = get_cpu(cpu_config)
    memory = get_memory(memory_config)
    invoke_time = get_invoke_time(
        function=function, parameter=parameter, platform=platform, cpu=cpu, memory=memory)

    runtime = get_runtime(
        function=function, parameter=parameter, platform=platform, cpu=cpu, memory=memory)

    cost = 0.0
    if invoke_time == 999.0:
        # 如果服务为启动不成功，则立马返回
        runtime = 1

    if platform == 1:
        cost = (cpu + memory) * runtime
    elif platform == 2:
        cost = (cpu + memory) * runtime * knative_cost
    elif platform == 3:
        cost = (cpu + memory) * runtime * scknative_cost

    return cost


def out_resource_limitation(platform, cpu_config, memory_config):
    if platform == 1:
        if get_cpu(cpu_config) > 3000:
            return True
        if get_memory(memory_config) > 3072:
            return True

    elif int(platform) == 2:
        if get_cpu(cpu_config) > 6000:
            return True
        if get_memory(memory_config) > 6144:
            return True

    else:
        return False


def run_device_edge_bo(function, parameter, request_time, logger, function_list):
    function_name = function + "_" + parameter
    slo = function_list[function_name]["slo"]
    device_edge_max, _, _ = discribe_bo_plat_resource.get_resource()

    if function_list[function_name]["de_iteration"] == 0:
        pbounds = {'platform': (1.0, 1.999999), 'cpu_config': (
            1, device_edge_max['cpu']+0.999999), 'memory_config': (1, device_edge_max['memory']+0.999999)}

        optimizer = BayesianOptimization(
            f=None,
            pbounds=pbounds,
            verbose=2,
        )
        function_list[function_name]["device_edge_optimizer"] = optimizer

    utility = UtilityFunction(kind="ei", kappa=2.5, xi=0.0)

    release_resource(request_time)
    next_point = function_list[function_name]["device_edge_optimizer"].suggest(
        utility)

    platform = int(next_point['platform'])
    cpu = int(next_point['cpu_config'])
    memory = int(next_point['memory_config'])

    target, runtime = reward_funtion(function=function, parameter=parameter,
                                     platform=platform, cpu_config=cpu, memory_config=memory, slo=slo)
    function_list[function_name]["iteration"] += 1
    consume_success = consume_resource(
        "device_edge", device_edge_max["node"], cpu, memory, runtime, request_time)
    if next_point not in function_list[function_name]["observered"] and consume_success:
        function_list[function_name]["observered"].append(next_point)
        function_list[function_name]["device_edge_optimizer"].register(
            params=next_point, target=target)

        # cost += get_cost(function=function, parameter=parameter,
        #                  platform=platform, cpu_config=cpu, memory_config=memory)

    # logger.info("device edge bo result is %s" % (optimizer.max))
    # logger.info("platform: %s, cpu: %s, memory: %s" % (get_platform_name(int(optimizer.max['params']['platform'])), get_cpu(
    #     int(optimizer.max['params']['cpu_config'])), get_memory(int(optimizer.max['params']['memory_config']))))
    return function_list[function_name]["device_edge_optimizer"].max


def run_cloud_edge_bo(function, parameter, request_time, function_list, logger):
    function_name = function + "_" + parameter
    slo = function_list[function_name]["slo"]
    _, cloud_edge_max, _ = discribe_bo_plat_resource.get_resource()

    if function_list[function_name]["ce_iteration"] == 0:
        pbounds = {'platform': (2.0, 2.999999), 'cpu_config': (
            1, cloud_edge_max['cpu']+0.999999), 'memory_config': (1, cloud_edge_max['memory']+0.999999)}

        optimizer = BayesianOptimization(
            f=None,
            pbounds=pbounds,
            verbose=2,
        )
        function_list[function_name]["cloud_edge_optimizer"] = optimizer

    utility = UtilityFunction(kind="ei", kappa=2.5, xi=0.0)

    release_resource(request_time)
    next_point = function_list[function_name]["cloud_edge_optimizer"].suggest(
        utility)

    platform = int(next_point['platform'])
    cpu = int(next_point['cpu_config'])
    memory = int(next_point['memory_config'])

    target, runtime = reward_funtion(function=function, parameter=parameter,
                                     platform=platform, cpu_config=cpu, memory_config=memory, slo=slo)
    consume_success = consume_resource(
        "cloud_edge", cloud_edge_max["node"], cpu, memory, runtime, request_time)
    function_list[function_name]["iteration"] += 1

    if next_point not in function_list[function_name]["observered"] and consume_success:
        function_list[function_name]["observered"].append(next_point)
        function_list[function_name]["cloud_edge_optimizer"].register(
            params=next_point, target=target)

        # cost += get_cost(function=function, parameter=parameter,
        #                  platform=platform, cpu_config=cpu, memory_config=memory)

    # logger.info("cloud edge bo result is %s" % (function_list[function_name]["cloud_edge_optimizer"].max))
    # logger.info("platform: %s, cpu: %s, memory: %s" % (get_platform_name(int(function_list[function_name]["cloud_edge_optimizer"].max['params']['platform'])), get_cpu(
    #     int(function_list[function_name]["cloud_edge_optimizer"].max['params']['cpu_config'])), get_memory(int(function_list[function_name]["cloud_edge_optimizer"].max['params']['memory_config']))))
    return function_list[function_name]["cloud_edge_optimizer"].max


def run_cloud_bo(function, parameter, function_list, request_time, logger):
    function_name = function + "_" + parameter
    slo = function_list[function_name]["slo"]
    _, _, cloud_max = discribe_bo_plat_resource.get_resource()

    if function_list[function_name]["cloud_iteration"] == 0:
        pbounds = {'platform': (3.0, 3.999999), 'cpu_config': (
            1, cloud_max['cpu']+0.999999), 'memory_config': (1, cloud_max['memory']+0.999999)}

        optimizer = BayesianOptimization(
            f=None,
            pbounds=pbounds,
            verbose=2,
        )
        function_list[function_name]["cloud_optimizer"] = optimizer

    utility = UtilityFunction(kind="ei", kappa=2.5, xi=0.0)

    release_resource(request_time)
    next_point = function_list[function_name]["cloud_optimizer"].suggest(
        utility)

    platform = int(next_point['platform'])
    cpu = int(next_point['cpu_config'])
    memory = int(next_point['memory_config'])
    function_list[function_name]["iteration"] += 1
    target, runtime = reward_funtion(function=function, parameter=parameter,
                                     platform=platform, cpu_config=cpu, memory_config=memory, slo=slo)
    consume_success = consume_resource(
        "cloud", cloud_max["node"], cpu, memory, runtime, request_time)
    if next_point not in function_list[function_name]["observered"] and consume_success:
        function_list[function_name]["observered"].append(next_point)
        function_list[function_name]["cloud_optimizer"].register(
            params=next_point, target=target)

        # cost += get_cost(function=function, parameter=parameter,
        #                  platform=platform, cpu_config=cpu, memory_config=memory)

    # logger.info("cloud bo result is %s" % (optimizer.max))
    # logger.info("platform: %s, cpu: %s, memory: %s" % (get_platform_name(int(optimizer.max['params']['platform'])), get_cpu(
    #     int(optimizer.max['params']['cpu_config'])), get_memory(int(optimizer.max['params']['memory_config']))))
    return function_list[function_name]["cloud_optimizer"].max


def Discrete_BO(function, parameter, logger, function_list, request_time, iteration):
    function_name = function + "_" + parameter

    if function_list[function_name]["iteration"] == 0:
        function_list[function_name]["observered"] = []
        function_list[function_name]["de_iteration"] = 0
        function_list[function_name]["ce_iteration"] = 0
        function_list[function_name]["cloud_iteration"] = 0
        function_list[function_name]["max"] = []

    if function_list[function_name]["iteration"] % 3 == 0:
        max_point = run_device_edge_bo(
            function=function, parameter=parameter, request_time=request_time, function_list=function_list, logger=logger)
        function_list[function_name]["de_iteration"] += 1
    elif function_list[function_name]["iteration"] % 3 == 1:
        max_point = run_cloud_edge_bo(
            function=function, parameter=parameter, request_time=request_time, function_list=function_list, logger=logger)
        function_list[function_name]["ce_iteration"] += 1
    elif function_list[function_name]["iteration"] % 3 == 2:
        max_point = run_cloud_bo(
            function=function, parameter=parameter, request_time=request_time, function_list=function_list, logger=logger)
        function_list[function_name]["cloud_iteration"] += 1
    function_list[function_name]["max"].append(max_point['target'])

    if function_list[function_name]["iteration"] == iteration:
        logger.info("Discrete bo result is %s" %
                    (max(function_list[function_name]["max"])*-1))

    if len(function_list[function_name]["max"]) > 0:
        return max(function_list[function_name]["max"]) * -1

    return 0


def run_corunning_experiment(approach, logger, experiment, function_list, iteration, repeat):
    dir_name = './'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    result_file = dir_name + datetime.datetime.now().strftime('%m-%d') + "_" +\
        approach + '.csv'

    function_name_list = ["qrcode_250", "markdown_50", "sentiment_50",
                          "resizeimage_2576", "imageinception_1351", "pagerank_100"]

    with open(result_file, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(function_name_list)

    request_list = generate_workload_with_compete(repeat)

    # logger.info(request_list)
    for _, request_times in request_list.items():
        logger.info("iteration: %d" %
                    function_list["qrcode_250"]["iteration"])
        # print(request_times)
        stop_flag = 0
        result_list = []
        number = 0
        while stop_flag < 6:
            base = number * 1440 * 60
            number += 1

            for request_time in request_times:
                function_name = request_time[0] + "_" + request_time[1]
                if approach == "bo":
                    result = BO(function=request_time[0], parameter=request_time[1], request_time=request_time[2] + base,
                                iteration=iteration, function_list=function_list, logger=logger)
                else:
                    result = Discrete_BO(function=request_time[0], parameter=request_time[1], request_time=request_time[2] + base,
                                         iteration=iteration, function_list=function_list, logger=logger)
                if function_list[function_name]["iteration"] == iteration:
                    logger.info("iteration: %d, function name %s" %
                                (function_list[function_name]["iteration"], function_name))
                    stop_flag += 1
                    function_list[function_name]["result"] = result
                if stop_flag == 6:
                    break
        for key in function_name_list:
            result_list.append(('%.15f' % abs(function_list[key]["result"])))
            function_list[key]["iteration"] = 0
            logger.info("iteration: %d, function name %s" %
                        (function_list[key]["iteration"], key))

        with open(result_file, "a+", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(result_list)


def corunning_main(approach, iteration, repeat):
    function_list["qrcode_250"] = {}
    function_list["qrcode_250"]["slo"] = 0.2
    function_list["qrcode_250"]["iteration"] = 0

    function_list["markdown_50"] = {}
    function_list["markdown_50"]["slo"] = 0.1
    function_list["markdown_50"]["iteration"] = 0

    function_list["sentiment_50"] = {}
    function_list["sentiment_50"]["slo"] = 1
    function_list["sentiment_50"]["iteration"] = 0

    function_list["resizeimage_2576"] = {}
    function_list["resizeimage_2576"]["slo"] = 1
    function_list["resizeimage_2576"]["iteration"] = 0

    function_list["imageinception_1351"] = {}
    function_list["imageinception_1351"]["slo"] = 5
    function_list["imageinception_1351"]["iteration"] = 0

    function_list["pagerank_100"] = {}
    function_list["pagerank_100"]["slo"] = 30
    function_list["pagerank_100"]["iteration"] = 0

    experiment = "corunning"
    log_path = './logs/' + str(datetime.datetime.now().strftime(
        '%Y-%m-%d')) + "_" + experiment + '.log'
    logger = Logger(log_path, logging.DEBUG, __name__).getlog()
    run_corunning_experiment(approach, logger, experiment,
                             function_list, iteration, repeat)


if __name__ == "__main__":
    corunning_main("icose", 300, 25)
    # function = "qrcode"
    # parameter = "50"
    # slo = 0.2
    # function = "sentiment"
    # parameter = "50"
    # slo = 1

    # function = "resizeimage"
    # parameter = "2576"
    # slo = 1

    # function_list = {}

    # function_name = function + "_" + parameter
    # log_path = './logs/' + str(datetime.datetime.now().strftime(
    #     '%Y-%m-%d')) + "_" + function_name + '_.log'
    # logger = Logger(log_path, logging.DEBUG, __name__).getlog()
    # function_list[function_name] = {}
    # function_list[function_name]["slo"] = slo

    # request_list = generate_workload_without_compete(300)

    # # BO(function=function, parameter=parameter, function_list=function_list, logger=logger)
    # for request_time in request_list:
    #     # Discrete_BO(function=function, parameter=parameter, function_list=function_list,
    #     #             logger=logger, request_time=request_time, iteration=30)
    #     BO(function=function, parameter=parameter,
    #        function_list=function_list, logger=logger, request_time=request_time, iteration=len(request_list))

    # logger.info(bo_platform)
