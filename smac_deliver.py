from pickle import NONE
from skopt import forest_minimize
from skopt.space import Integer, Categorical
from skopt import Optimizer
from get_time import get_cpu
from get_time import get_invoke_time
from get_time import get_memory
from get_time import get_runtime
from platform_resource import *
import os
import logging
import csv
import datetime
from log import *
from generate_workload import *
penalty_factor = 100
knative_cost = 4
scknative_cost = 6


smac_release_list = []
smac_plat_resource = Platform_Resource()


def consume_resource(platform_name, node, cpu, memory, runtime, request_time):
    # 消耗资源
    global smac_release_list
    if runtime == 998.0:
        release_time = request_time + 0.1
    else:
        release_time = request_time + runtime

    consume_success = smac_plat_resource.del_resource(
        platform_name, node, cpu, memory)
    if consume_success:
        smac_release_list.append({
            "release_time": release_time,
            "platform_name": platform_name,
            "node": node,
            "cpu":  cpu,
            "memory": memory
        })
    return consume_success


def release_resource(request_time):
    global smac_release_list
    if len(smac_release_list) > 0:
        smac_release_list = sorted(
            smac_release_list, key=lambda i: i['release_time'], reverse=False)

        while request_time > smac_release_list[0]['release_time']:
            smac_plat_resource.add_resource(
                smac_release_list[0]['platform_name'], smac_release_list[0]['node'], smac_release_list[0]['cpu'], smac_release_list[0]['memory'])
            smac_release_list.pop(0)

            if len(smac_release_list) == 0:
                break


def reward_funtion(function, parameter, platform, cpu_config, memory_config, slo):
    cpu = get_cpu(cpu_config)
    memory = get_memory(memory_config)
    invoke_time = get_invoke_time(
        function=function, parameter=parameter, platform=platform, cpu=cpu, memory=memory)

    runtime = get_runtime(
        function=function, parameter=parameter, platform=platform, cpu=cpu, memory=memory)

    if platform == 1:
        if invoke_time > slo:
            return (cpu + memory) * runtime * runtime / slo * penalty_factor, runtime
        else:
            return (cpu + memory) * runtime, runtime

    elif platform == 2:
        if invoke_time > slo:
            return (cpu + memory) * runtime * runtime / slo * knative_cost * penalty_factor, runtime
        else:
            return (cpu + memory) * runtime * knative_cost, runtime

    elif platform == 3:
        if invoke_time > slo:
            return (cpu + memory) * runtime * runtime / slo * scknative_cost * penalty_factor, runtime
        else:
            return (cpu + memory) * runtime * scknative_cost, runtime

    return 9999999, 0.1


def get_cost(function, parameter, platform, cpu_config, memory_config):
    cpu = get_cpu(cpu_config)
    memory = get_memory(memory_config)
    invoke_time = get_invoke_time(
        function=function, parameter=parameter, platform=platform, cpu=cpu, memory=memory)

    runtime = get_runtime(
        function=function, parameter=parameter, platform=platform, cpu=cpu, memory=memory)

    cost = 0.0
    if invoke_time == 998.0:
        # 如果服务为启动不成功，则立马返回
        runtime = 0.1

    if platform == 1:
        cost = (cpu + memory) * runtime
    elif platform == 2:
        cost = (cpu + memory) * runtime * 4
    elif platform == 3:
        cost = (cpu + memory) * runtime * 6

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


def SMAC(function, parameter, iteration, function_list, request_time, logger, number):
    function_name = function + "_" + parameter
    slo = function_list[function_name]["slo"]

    if function_list[function_name]["iteration"] == 0:
        device_edge_max, cloud_edge_max, cloud_max = smac_plat_resource.get_resource()
        space = [Categorical([1, 2, 3], name='platform'),
                 Integer(1, cloud_max['cpu'], name="cpu"),
                 Integer(1, cloud_max['memory'], name="memory"), ]
        opt = Optimizer(dimensions=space, base_estimator="rf", acq_func="EI")
        function_list[function_name]["optimizer"] = opt

    release_resource(request_time)
    function_list[function_name]["iteration"] += 1

    suggested = function_list[function_name]["optimizer"].ask()

    platform = suggested[0]
    cpu = suggested[1]
    memory = suggested[2]
    # print(suggested)

    device_edge_max, cloud_edge_max, cloud_max = smac_plat_resource.get_resource()
    target, runtime = reward_funtion(function=function, parameter=parameter,
                                     platform=platform, cpu_config=cpu, memory_config=memory, slo=slo)

    if platform == 1:
        consume_success = consume_resource(
            "device_edge", device_edge_max["node"], cpu, memory, runtime, request_time)
    elif platform == 2:
        consume_success = consume_resource(
            "cloud_edge", cloud_edge_max["node"], cpu, memory, runtime, request_time)
    elif platform == 3:
        consume_success = consume_resource(
            "cloud", cloud_max["node"], cpu, memory, runtime, request_time)

    if consume_success:
        function_list[function_name]["res"] = function_list[function_name]["optimizer"].tell(
            suggested, target)

    if function_list[function_name]["iteration"] == iteration:
        logger.info("SMAC result")
        logger.info("Best params: %s" %
                    (function_list[function_name]["res"].x))
        logger.info("Best value: %s" %
                    (function_list[function_name]["res"].fun))

        return function_list[function_name]["res"].fun

    return 0


def run_corunning_experiment(logger, experiment, function_list, iteration, repeat):
    dir_name = './'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    result_file = dir_name + datetime.datetime.now().strftime('%m-%d') + \
        '_smac.csv'

    function_name_list = ["qrcode_250", "markdown_50", "sentiment_50",
                          "resizeimage_2576", "imageinception_1351", "pagerank_100"]

    with open(result_file, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(function_name_list)

    request_list = generate_workload_with_compete(repeat)

    # logger.info(request_list)
    repeat_number = 0
    for _, request_times in request_list.items():
        logger.info("iteration: %d" %
                    function_list["qrcode_250"]["iteration"])
        # print(request_times)
        stop_flag = 0
        result_list = []
        number = 0
        repeat_number += 1
        while stop_flag < 6:
            base = number * 1440 * 60
            number += 1

            for request_time in request_times:
                function_name = request_time[0] + "_" + request_time[1]
                result = SMAC(function=request_time[0], parameter=request_time[1], request_time=request_time[2] + base,
                              iteration=iteration, function_list=function_list, logger=logger, number=repeat_number)
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


def corunning_main(iteration, repeat):
    function_list = {}
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
    run_corunning_experiment(logger, experiment,
                             function_list, iteration, repeat)


if __name__ == "__main__":
    corunning_main(300, 25)
    # function = "qrcode"
    # parameter = "50"
    # slo = 0.2
    # function_list = {}
    # function_name = function + "_" + parameter
    # log_path = './logs/' + str(datetime.datetime.now().strftime(
    #     '%Y-%m-%d')) + "_" + function_name + '.log'
    # logger = Logger(log_path, logging.DEBUG, __name__).getlog()
    # function_list[function_name] = {}
    # function_list[function_name]["slo"] = slo

    # request_list = generate_workload_without_compete(300)

    # for request_time in request_list:
    #     SMAC(function=function, parameter=parameter,
    #          iteration=len(request_list), function_list=function_list, request_time=request_time, logger=logger)
