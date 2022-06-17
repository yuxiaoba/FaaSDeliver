from importlib.resources import Resource
import logging
import time
import sys
import os
from log import *
from requests import request
import numpy as np
import joblib
import optuna
import os
import datetime
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

function_list = {}
tpe_release_list = []
tpe_plat_resource = Platform_Resource()
day = time.strftime("%Y-%m-%d", time.localtime())


def gamma(x: int) -> int:
    return min(int(np.ceil(0.1 * x)), beta)


def consume_resource(platform_name, node, cpu, memory, runtime, request_time):
    # 消耗资源
    global tpe_release_list
    if runtime == 998.0:
        release_time = request_time + 0.1
    else:
        release_time = request_time + runtime

    tpe_plat_resource.del_resource(
        platform_name, node, cpu, memory)
    tpe_release_list.append({
        "release_time": release_time,
        "platform_name": platform_name,
        "node": node,
        "cpu":  cpu,
        "memory": memory
    })


def release_resource(request_time):
    global tpe_release_list
    if len(tpe_release_list) > 0:
        tpe_release_list = sorted(
            tpe_release_list, key=lambda i: i['release_time'], reverse=False)

        while request_time > tpe_release_list[0]['release_time']:
            tpe_plat_resource.add_resource(
                tpe_release_list[0]['platform_name'], tpe_release_list[0]['node'], tpe_release_list[0]['cpu'], tpe_release_list[0]['memory'])
            tpe_release_list.pop(0)

            if len(tpe_release_list) == 0:
                break


def objective(trial, function, parameter, request_time):
    function_name = function + "_" + parameter
    min_memory_object = function_list[function_name]["min_memory"]
    slo = function_list[function_name]["slo"]

    device_edge_mem_min, cloud_edge_mem_min, cloud_mem_min = min_memory_object.get()
    device_edge_max, cloud_edge_max, cloud_max = tpe_plat_resource.get_resource()

    platform_name = trial.suggest_categorical(
        "platform", ["device_edge", "cloud_edge", "cloud"])
    if platform_name == "device_edge":
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
            # cost = cost + (get_cpu(cpu) + get_memory(memory)) * runtime
            return (get_cpu(cpu) + get_memory(memory)) * runtime * runtime / slo * penalty_factor

        # cost = cost + (get_cpu(cpu) + get_memory(memory)) * runtime
        return (get_cpu(cpu) + get_memory(memory)) * runtime

    elif platform_name == "cloud_edge":
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
            # cost = cost + (get_cpu(cpu) + get_memory(memory)) * runtime * knative_cost
            return (get_cpu(cpu) + get_memory(memory)) * runtime * runtime / slo * penalty_factor * knative_cost

        # cost = cost + (get_cpu(cpu) + get_memory(memory)) * runtime * knative_cost
        return (get_cpu(cpu) + get_memory(memory)) * runtime * knative_cost

    else:
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
            # cost = cost + (get_cpu(cpu) + get_memory(memory) ) * runtime * scknative_cost
            return (get_cpu(cpu) + get_memory(memory)) * runtime * runtime / slo * penalty_factor * scknative_cost

        # cost = cost + (get_cpu(cpu) + get_memory(memory)) * runtime * scknative_cost
        return (get_cpu(cpu) + get_memory(memory)) * runtime * scknative_cost


def run_tpe_once(function, parameter, ei, study, request_time, function_list):
    function_name = function + "_" + parameter
    storage_file = function_list[function_name]["storage_file"]

    def func(trial): return objective(trial, function, parameter, request_time)
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


def TPE_Not_Prune(function, parameter, random_start, ei, experiment, iteration, request_time, function_list, logger):
    release_resource(request_time)
    function_name = function + "_" + parameter

    if function_list[function_name]["iteration"] == 0:
        study_name = function + "_" + parameter
        dir_name = "./experiment_results/" + \
            experiment + "/" + day + "/" + function + "/"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        storage_file = dir_name + study_name + ".pkl"
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

    return study.best_value


if __name__ == "__main__":
    experiment = "effective"
    function = "qrcode"
    parameter = "50"
    slo = 0.2
    function_list = {}

    function_name = function + "_" + parameter
    log_path = './logs/' + str(datetime.datetime.now().strftime(
        '%Y-%m-%d')) + "_" + function_name + '.log'
    logger = Logger(log_path, logging.DEBUG, __name__).getlog()
    function_list[function_name] = {}
    function_list[function_name]["slo"] = slo
    function_list[function_name]["iteration"] = 0

    request_list = generate_workload_without_compete(300)

    for request_time in request_list:
        TPE_Not_Prune(function=function, parameter=parameter, experiment=experiment, request_time=request_time,
                      iteration=len(request_list), random_start=10, ei=24, function_list=function_list, logger=logger)
