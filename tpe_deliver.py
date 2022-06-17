from importlib.resources import Resource
import logging
import time
import sys
import os
from log import *
import datetime
from requests import request
import numpy as np
import joblib
import optuna
import os
from get_time import *
from platform_resource import *
from generate_workload import *

# Add stream handler of stdout to show the messages
# optuna.logging.get_logger("optuna").addHandler(
#     logging.StreamHandler(sys.stdout))

cost = 0.0
penalty_factor = 100
knative_cost = 4
scknative_cost = 6
beta = 25

release_list = []
tpe_platform = [0, 0, 0]

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


plat_resource = Platform_Resource()
day = time.strftime("%Y-%m-%d", time.localtime())


def gamma(x: int) -> int:
    return min(int(np.ceil(0.1 * x)), beta)


def consume_resource(platform_name, node, cpu, memory, runtime, request_time):
    # 消耗资源
    global release_list
    if runtime == 998.0 or runtime == 999.0:
        release_time = request_time + 0.1
    else:
        release_time = request_time + runtime

    plat_resource.del_resource(
        platform_name, node, cpu, memory)
    release_list.append({
        "release_time": release_time,
        "platform_name": platform_name,
        "node": node,
        "cpu":  cpu,
        "memory": memory
    })


def release_resource(request_time):
    global release_list
    if len(release_list) > 0:
        release_list = sorted(
            release_list, key=lambda i: i['release_time'], reverse=False)

        while request_time > release_list[0]['release_time']:
            plat_resource.add_resource(
                release_list[0]['platform_name'], release_list[0]['node'], release_list[0]['cpu'], release_list[0]['memory'])

            print("add resource", request_time, release_list[0]['release_time'], release_list[0]['platform_name'], release_list[0]
                  ['node'], release_list[0]['cpu'], release_list[0]['memory'])
            release_list.pop(0)

            if len(release_list) == 0:
                break


def objective(trial, function, parameter, request_time, function_list):
    function_name = function + "_" + parameter
    min_memory_object = function_list[function_name]["min_memory"]
    slo = function_list[function_name]["slo"]

    device_edge_mem_min, cloud_edge_mem_min, cloud_mem_min = min_memory_object.get()
    device_edge_max, cloud_edge_max, cloud_max = plat_resource.get_resource()

    platform_list = ["device_edge", "cloud_edge", "cloud"]

    if device_edge_mem_min > device_edge_max["memory"]:
        platform_list.remove("device_edge")

    if cloud_edge_mem_min > cloud_edge_max["memory"]:
        platform_list.remove("cloud_edge")

    if cloud_mem_min > cloud_max["memory"]:
        cloud_mem_min = 1

    platform_name = trial.suggest_categorical(
        "platform", platform_list)
    if platform_name == "device_edge":
        tpe_platform[0] += 1
        cpu = trial.suggest_int(
            "cpu", 1, device_edge_max["cpu"])
        memory = trial.suggest_int(
            "memory", device_edge_mem_min, device_edge_max["memory"])

        invoke_time = get_invoke_time(function=function, parameter=parameter, platform=platform_name,
                                      cpu=get_cpu(cpu), memory=get_memory(memory))

        runtime = get_runtime(function=function, parameter=parameter, platform=platform_name,
                              cpu=get_cpu(cpu), memory=get_memory(memory))

        consume_resource(platform_name, device_edge_max["node"], cpu,
                         memory, runtime, request_time)

        if invoke_time > slo:
            if invoke_time == 998.0:
                # 如果函数没起来, 说明内存分配不足
                if memory >= device_edge_mem_min and memory < 96:
                    min_memory_object.change(platform_name, memory + 1)

                print("Launch failed! platfrom_name:", platform_name, "cpu:", get_cpu(
                    cpu), "memory:", get_memory(memory), device_edge_mem_min)

            # cost = cost + (get_cpu(cpu) + get_memory(memory)) * runtime
            return (get_cpu(cpu) + get_memory(memory)) * runtime * runtime / slo * penalty_factor

        # cost = cost + (get_cpu(cpu) + get_memory(memory)) * runtime
        return (get_cpu(cpu) + get_memory(memory)) * runtime

    elif platform_name == "cloud_edge":
        tpe_platform[1] += 1
        cpu = trial.suggest_int(
            "cpu", 1, cloud_edge_max['cpu'])
        memory = trial.suggest_int(
            "memory", cloud_edge_mem_min, cloud_edge_max['memory'])

        invoke_time = get_invoke_time(function=function, parameter=parameter, platform=platform_name,
                                      cpu=get_cpu(cpu), memory=get_memory(memory))

        runtime = get_runtime(function=function, parameter=parameter, platform=platform_name,
                              cpu=get_cpu(cpu), memory=get_memory(memory))

        # 消耗资源
        consume_resource(platform_name, cloud_edge_max["node"], cpu,
                         memory, runtime, request_time)

        if invoke_time > slo:
            if invoke_time == 998.0:
                # 如果函数没起来, 说明内存分配不足
                if memory >= cloud_edge_mem_min and memory < 192:
                    min_memory_object.change(platform_name, memory + 1)

                invoke_time = 0.1
                print("Launch failed! platfrom_name:", platform_name, "cpu:", get_cpu(cpu),
                      "memory:", get_memory(memory), cloud_edge_mem_min)

            # cost = cost + (get_cpu(cpu) + get_memory(memory)) * runtime * knative_cost
            return (get_cpu(cpu) + get_memory(memory)) * runtime * runtime / slo * penalty_factor * knative_cost

        # cost = cost + (get_cpu(cpu) + get_memory(memory)) * runtime * knative_cost
        return (get_cpu(cpu) + get_memory(memory)) * runtime * knative_cost

    else:
        tpe_platform[2] += 1
        cpu = trial.suggest_int(
            "cpu", 1, cloud_max['cpu'])
        memory = trial.suggest_int(
            "memory", cloud_mem_min, cloud_max['memory'])

        invoke_time = get_invoke_time(function=function, parameter=parameter,
                                      platform=platform_name, cpu=get_cpu(cpu), memory=get_memory(memory))

        runtime = get_runtime(function=function, parameter=parameter, platform=platform_name,
                              cpu=get_cpu(cpu), memory=get_memory(memory))

        # 消耗资源
        consume_resource(platform_name, cloud_max["node"], cpu,
                         memory, runtime, request_time)

        if invoke_time > slo:
            if invoke_time == 998.0:
                # 如果函数没起来, 说明内存分配不足
                if memory >= cloud_mem_min and memory < 320:
                    min_memory_object.change(platform_name, memory + 1)
                invoke_time = 0.1
                print("platfrom_name:", platform_name, "cpu:", get_cpu(
                    cpu), "memory:", get_memory(memory), cloud_mem_min)

            # cost = cost + (get_cpu(cpu) + get_memory(memory) ) * runtime * scknative_cost
            return (get_cpu(cpu) + get_memory(memory)) * runtime * runtime / slo * penalty_factor * scknative_cost

        # cost = cost + (get_cpu(cpu) + get_memory(memory)) * runtime * scknative_cost
        return (get_cpu(cpu) + get_memory(memory)) * runtime * scknative_cost


def run_tpe_once(function, parameter, ei, study, request_time, function_list):
    function_name = function + "_" + parameter
    storage_file = function_list[function_name]["storage_file"]

    def func(trial): return objective(trial, function,
                                      parameter, request_time, function_list)
    study.optimize(func=func, n_trials=1, n_jobs=ei)
    joblib.dump(study, storage_file)


'''
运行 TPE 算法
     function : 函数名
     parameter : 参数名
     slo: SLO
     random_start: 冷启动的次数
     ei: ei candidate 选取的个数
     experiment: 实验测试的类型
     request_time: 函数被触发的时间
'''


def TPE(function, parameter, experiment, iteration, request_time, function_list, logger, number=0, random_start=10, ei=24):
    release_resource(request_time)
    function_name = function + "_" + parameter

    if function_list[function_name]["iteration"] == 0:
        study_name = function + "_" + parameter
        dir_name = "./experiment_results/" + \
            experiment + "/" + day + "/" + function + "/"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        storage_file = dir_name + study_name + str(number) + ".pkl"
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(multivariate=True, gamma=gamma, n_startup_trials=random_start, n_ei_candidates=ei),
                                    direction='minimize', study_name=study_name)
        function_list[function_name]["storage_file"] = storage_file
        function_list[function_name]["min_memory"] = Min_memory()
        function_list[function_name]["iteration"] = 1
        run_tpe_once(function, parameter, ei, study,
                     request_time, function_list)

    else:
        storage_file = function_list[function_name]["storage_file"]
        study = joblib.load(storage_file)
        run_tpe_once(function, parameter, ei, study,
                     request_time, function_list)
        function_list[function_name]["iteration"] += 1

    if function_list[function_name]["iteration"] == iteration:
        storage_file = function_list[function_name]["storage_file"]
        study = joblib.load(storage_file)

        logger.info("TPE result")
        logger.info("Best params: %s" % (study.best_params))
        logger.info("Best value: %s" % (study.best_value))
        logger.info("Best Trial: %s" % (study.best_trial))
        logger.info(tpe_platform)

    return study.best_value


def run_corunning_experiment(logger, experiment, function_list, iteration, repeat):
    dir_name = './'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    result_file = dir_name + datetime.datetime.now().strftime('%m-%d') + \
        '_tpe.csv'

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
                result = TPE(function=request_time[0], parameter=request_time[1], request_time=request_time[2] + base, experiment=experiment,
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
    experiment = "corunning"
    log_path = './logs/' + str(datetime.datetime.now().strftime(
        '%Y-%m-%d')) + "_" + experiment + '.log'
    logger = Logger(log_path, logging.DEBUG, __name__).getlog()
    run_corunning_experiment(logger, experiment,
                             function_list, iteration, repeat)


def main():
    experiment = "iteration"

    function_name = function + "_" + parameter
    log_path = './logs/' + str(datetime.datetime.now().strftime(
        '%Y-%m-%d')) + "_" + function_name + '.log'
    logger = Logger(log_path, logging.DEBUG, __name__).getlog()

    request_list = generate_workload_without_compete(300)

    for request_time in request_list:
        TPE(function=function, parameter=parameter, experiment=experiment,
            iteration=len(request_list), function_list=function_list, request_time=request_time, logger=logger)


def n_random_main():
    experiment = "iteration"
    function_name_list = ["qrcode_250", "markdown_50", "sentiment_50",
                          "resizeimage_2576", "imageinception_1351", "pagerank_100"]
    random_list = [1, 5, 10, 15, 20, 25]

    dir_name = './'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    result_file = dir_name + datetime.datetime.now().strftime('%m-%d') + \
        '_random_number.csv'

    with open(result_file, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(random_list)

    log_path = './logs/' + str(datetime.datetime.now().strftime(
        '%Y-%m-%d')) + "_" + experiment + '.log'
    logger = Logger(log_path, logging.DEBUG, __name__).getlog()

    for faas_name in function_name_list:
        for random_number in random_list:
            result_list = []
            function = faas_name.split("_")[0]
            parameter = faas_name.split("_")[1]

            request_list = generate_workload_without_compete(300)

            for request_time in request_list:
                result = TPE(function=function, parameter=parameter, experiment=experiment, random_start=random_number,
                             iteration=len(request_list), function_list=function_list, request_time=request_time, logger=logger)
            result_list.append(('%.15f' % abs(result)))
            function_list[faas_name]["iteration"] = 0
        with open(result_file, "a+", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(result_list)


if __name__ == "__main__":
    # main()
    n_random_main()
    # corunning_main(300, 25)
