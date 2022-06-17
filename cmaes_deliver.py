import logging
import sys
import time
from wsgiref.util import request_uri
import joblib
import optuna
import os
from log import *
import datetime
from get_time import *
from platform_resource import *
from generate_workload import *
o_path = os.getcwd()
sys.path.append(o_path)

# Add stream handler of stdout to show the messages
# optuna.logging.get_logger("optuna").addHandler(
#     logging.StreamHandler(sys.stdout))

cost = 0.0
penalty_factor = 100
knative_cost = 4
scknative_cost = 6
cmaes_release_list = []
function_list = {}
cmaes_plat_resource = Platform_Resource()
cmaes_platform = [0, 0, 0]
day = time.strftime("%Y-%m-%d", time.localtime())


def consume_resource(platform_name, node, cpu, memory, runtime, request_time):
    # 消耗资源
    global cmaes_release_list
    if runtime == 998.0:
        release_time = request_time + 0.1
    else:
        release_time = request_time + runtime

    consume_success = cmaes_plat_resource.del_resource(
        platform_name, node, cpu, memory)
    if consume_success:
        cmaes_release_list.append({
            "release_time": release_time,
            "platform_name": platform_name,
            "node": node,
            "cpu":  cpu,
            "memory": memory
        })
    return consume_success


def release_resource(request_time):
    global cmaes_release_list
    if len(cmaes_release_list) > 0:
        cmaes_release_list = sorted(
            cmaes_release_list, key=lambda i: i['release_time'], reverse=False)

        while request_time > cmaes_release_list[0]['release_time']:
            cmaes_plat_resource.add_resource(
                cmaes_release_list[0]['platform_name'], cmaes_release_list[0]['node'], cmaes_release_list[0]['cpu'], cmaes_release_list[0]['memory'])
            cmaes_release_list.pop(0)

            if len(cmaes_release_list) == 0:
                break


def objective(trial, function, parameter, request_time):
    function_name = function + "_" + parameter
    slo = function_list[function_name]["slo"]

    device_edge_max, cloud_edge_max, cloud_max = cmaes_plat_resource.get_resource()

    platform = trial.suggest_int("platform", 1, 3)

    cpu = trial.suggest_int("cpu", 1, cloud_max['cpu'])

    memory = trial.suggest_int("memory", 1,  cloud_max['memory'])

    invoke_time = get_invoke_time(function=function, parameter=parameter,
                                  platform=platform, cpu=get_cpu(cpu), memory=get_memory(memory))

    runtime = get_runtime(function=function, parameter=parameter,
                          platform=platform, cpu=get_cpu(cpu), memory=get_memory(memory))

    if platform == 1:
        cmaes_platform[0] += 1
        consume_resource(
            "device_edge", device_edge_max["node"], cpu, memory, runtime, request_time)
    elif platform == 2:
        cmaes_platform[1] += 1
        consume_resource(
            "cloud_edge", cloud_edge_max["node"], cpu, memory, runtime, request_time)
    elif platform == 3:
        cmaes_platform[2] += 1
        consume_resource(
            "cloud", cloud_max["node"], cpu, memory, runtime, request_time)

    if invoke_time > slo:
        if platform == 1:
            # cost = cost + (get_cpu(cpu) + get_memory(memory)) * runtime
            return (get_cpu(cpu) + get_memory(memory)) * runtime * runtime / slo * penalty_factor

        elif platform == 2:
            #cost = cost + (get_cpu(cpu) + get_memory(memory)) * runtime * knative_cost
            return (get_cpu(cpu) + get_memory(memory)) * runtime * runtime / slo * knative_cost * penalty_factor

        else:
            # cost = cost + (get_cpu(cpu) + get_memory(memory)) * runtime * scknative_cost
            return (get_cpu(cpu) + get_memory(memory)) * runtime * runtime / slo * scknative_cost * penalty_factor

    else:
        if platform == 1:
            # cost = cost + (get_cpu(cpu) + get_memory(memory)) * runtime
            return (get_cpu(cpu) + get_memory(memory)) * runtime

        elif platform == 2:
            # cost = cost + (get_cpu(cpu) + get_memory(memory)) * runtime * knative_cost
            return (get_cpu(cpu) + get_memory(memory)) * runtime * knative_cost

        else:
            # cost = cost + (get_cpu(cpu) + get_memory(memory)) * runtime * scknative_cost
            return (get_cpu(cpu) + get_memory(memory)) * runtime * scknative_cost


def run_cmaes_once(function, parameter, study, request_time, function_list):
    function_name = function + "_" + parameter
    storage_file = function_list[function_name]["storage_file"]

    def func(trial): return objective(trial, function, parameter, request_time)
    study.optimize(func=func, n_trials=1)
    joblib.dump(study, storage_file)


def CMAES_deliver(function, parameter, experiment, request_time, function_list, iteration, logger):
    release_resource(request_time)
    function_name = function + "_" + parameter

    if function_list[function_name]["iteration"] == 0:
        study_name = function + "_" + parameter
        dir_name = "./experiment_results/" + \
            experiment + "/" + day + "/" + function + "/"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        storage_file = dir_name + study_name + "_cmaes.pkl"
        study = optuna.create_study(sampler=optuna.samplers.CmaEsSampler(
        ), direction='minimize', study_name=study_name)
        function_list[function_name]["storage_file"] = storage_file
        function_list[function_name]["iteration"] = 1
        run_cmaes_once(function, parameter, study, request_time, function_list)

    else:
        storage_file = function_list[function_name]["storage_file"]
        study = joblib.load(storage_file)
        run_cmaes_once(function, parameter, study, request_time, function_list)
        function_list[function_name]["iteration"] += 1

    if function_list[function_name]["iteration"] == iteration:
        storage_file = function_list[function_name]["storage_file"]
        study = joblib.load(storage_file)
        logger.info("CMAES result:")
        logger.info("Best params: %s" % (study.best_params))
        logger.info("Best value: %s" % (study.best_value))
        logger.info("Best Trial: %s" % (study.best_trial))
        logger.info(cmaes_platform)

    return study.best_value


def run_corunning_experiment(logger, experiment, function_list, iteration, repeat):
    dir_name = './'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    result_file = dir_name + datetime.datetime.now().strftime('%m-%d') + \
        '_cmaes.csv'

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
                result = CMAES_deliver(function=request_time[0], parameter=request_time[1], request_time=request_time[2] + base, experiment=experiment,
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


def corunning_main(iteration, repeat):
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
    # experiment = "effective"
    # # function = "qrcode"
    # # parameter = "250"
    # # slo = 0.2

    # # function = "sentiment"
    # # parameter = "50"
    # # slo = 1

    # function = "resizeimage"
    # parameter = "2576"
    # slo = 1

    # function_list = {}
    # function_name = function + "_" + parameter

    # log_path = './logs/' + str(datetime.datetime.now().strftime(
    #     '%Y-%m-%d')) + "_" + function_name + '.log'
    # logger = Logger(log_path, logging.DEBUG, __name__).getlog()
    # function_list[function_name] = {}
    # function_list[function_name]["slo"] = slo
    # function_list[function_name]["iteration"] = 0

    # request_list = generate_workload_without_compete(300)

    # for request_time in request_list:
    #     CMAES_deliver(function=function, parameter=parameter,
    #                   experiment=experiment, iteration=len(request_list), request_time=request_time, function_list=function_list, logger=logger)
