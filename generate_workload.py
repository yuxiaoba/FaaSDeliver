import pandas as pd


def generate_workload_without_compete(iteration):
    request_list = []
    for i in range(iteration):
        request_list.append(i*1000)

    return request_list


def generate_workload_with_compete(number):
    invocation_list = {}
    request_list = {}
    csv_data = pd.read_csv('workload.csv')  # 读取 workload 数据

    for i in range(number*6):
        row = csv_data.iloc[i]
        invocation_list[i] = []
        for second in range(4, len(row)):
            if row[second] > 0:
                for invocation in range(1, row[second]+1):
                    invocation_list[i].append((second-3) * 60 - 60/invocation)

    for i in range(number):
        request_list[i] = []
        for j in range(i*6, i*6+6):
            for reuqest_time in invocation_list[j]:
                if j % 6 == 0:
                    request_list[i].append(("qrcode", "250", reuqest_time))
                if j % 6 == 1:
                    request_list[i].append(("markdown", "50", reuqest_time))
                if j % 6 == 2:
                    request_list[i].append(("sentiment", "50", reuqest_time))
                if j % 6 == 3:
                    request_list[i].append(
                        ("resizeimage", "2576", reuqest_time))
                if j % 6 == 4:
                    request_list[i].append(
                        ("imageinception", "1351", reuqest_time))
                if j % 6 == 5:
                    request_list[i].append(
                        ("pagerank", "100", reuqest_time))
        request_list[i] = sorted(
            request_list[i], key=lambda request: request[2])

    return request_list


if __name__ == "__main__":
    request_list = generate_workload_with_compete(1)
    for _, request_times in request_list.items():
        print(request_times)
