class Platform_Resource():
    def __init__(self):
        # resource of platform
        self.device_dege = [{"node": 1, "cpu": 3000, "memory": 3072},
                            {"node": 2, "cpu": 3000, "memory": 3072},
                            {"node": 3, "cpu": 3000, "memory": 3072}]

        self.cloud_edge = [{"node": 1, "cpu": 6000, "memory": 6144},
                           {"node": 2, "cpu": 6000, "memory": 6144},
                           {"node": 3, "cpu": 6000, "memory": 6144},
                           {"node": 4, "cpu": 6000, "memory": 6144},
                           {"node": 5, "cpu": 6000, "memory": 6144}]

        self.cloud = [{"node": 1, "cpu": 10000, "memory": 10240},
                      {"node": 2, "cpu": 10000, "memory": 10240},
                      {"node": 3, "cpu": 10000, "memory": 10240},
                      {"node": 4, "cpu": 10000, "memory": 10240},
                      {"node": 5, "cpu": 10000, "memory": 10240}]

    def get_resource(self):
        # 资源降序排列，返回最大的资源节点
        self.device_dege.sort(key=lambda x: x['cpu'], reverse=True)
        self.cloud_edge.sort(key=lambda x: x['cpu'], reverse=True)
        self.cloud.sort(key=lambda x: x['cpu'], reverse=True)

        print()

        device_edge_max = {
            "node": self.device_dege[0]["node"],
            "cpu": self.device_dege[0]['cpu']/25,
            "memory": self.device_dege[0]['memory']/32,
        }

        cloud_edge_max = {
            "node": self.cloud_edge[0]["node"],
            "cpu": self.cloud_edge[0]['cpu']/25,
            "memory": self.cloud_edge[0]['memory']/32,
        }

        cloud_max = {
            "node": self.cloud[0]["node"],
            "cpu": self.cloud[0]['cpu']/25,
            "memory": self.cloud[0]['memory']/32,
        }

        # print(device_edge_max, cloud_edge_max, cloud_max )

        return device_edge_max, cloud_edge_max, cloud_max

    def add_resource(self, platform, node, cpu, memory):
        # 运行后回收资源
        if platform == "device_edge":
            for i in range(len(self.device_dege)):
                if self.device_dege[i]["node"] == node:
                    if self.device_dege[i]["cpu"] + cpu * 25 > 3000:
                        self.device_dege[i]["cpu"] = 3000
                    else:
                        self.device_dege[i]["cpu"] = self.device_dege[i]["cpu"] + cpu * 25

                    if self.device_dege[i]["memory"] + memory * 32 > 3072:
                        self.device_dege[i]["memory"] = 3072
                    else:
                        self.device_dege[i]["memory"] = self.device_dege[i]["memory"] + memory * 32

        elif platform == "cloud_edge":
            for i in range(len(self.cloud_edge)):
                if self.cloud_edge[i]["node"] == node:
                    if self.cloud_edge[i]["cpu"] + cpu * 25 > 6000:
                        self.cloud_edge[i]["cpu"] = 6000
                    else:
                        self.cloud_edge[i]["cpu"] = self.cloud_edge[i]["cpu"] + cpu * 25

                    if self.cloud_edge[i]["memory"] + memory * 32 > 6144:
                        self.cloud_edge[i]["memory"] = 6144
                    else:
                        self.cloud_edge[i]["memory"] = self.cloud_edge[i]["memory"] + memory * 32

        elif platform == "cloud":
            for i in range(len(self.cloud)):
                if self.cloud[i]["node"] == node:
                    if self.cloud[i]["cpu"] + cpu * 25 > 10000:
                        self.cloud[i]["cpu"] = 10000
                    else:
                        self.cloud[i]["cpu"] = self.cloud[i]["cpu"] + cpu * 25

                    if self.cloud[i]["memory"] + memory * 32 > 10240:
                        self.cloud[i]["memory"] = 10240
                    else:
                        self.cloud[i]["memory"] = self.cloud[i]["memory"] + \
                            memory * 32

    def del_resource(self, platform, node, cpu, memory):
        # 删除资源
        if platform == "device_edge":
            for i in range(len(self.device_dege)):
                if self.device_dege[i]["node"] == node:
                    if self.device_dege[i]["cpu"] - cpu * 25 > 0 and self.device_dege[i]["memory"] - memory * 32 > 0:
                        self.device_dege[i]["cpu"] = self.device_dege[i]["cpu"] - cpu * 25
                        self.device_dege[i]["memory"] = self.device_dege[i]["memory"] - memory * 32

                        return True
                    else:
                        return False

        elif platform == "cloud_edge":
            for i in range(len(self.cloud_edge)):
                if self.cloud_edge[i]["node"] == node:
                    if self.cloud_edge[i]["cpu"] - cpu * 25 > 0 and self.cloud_edge[i]["memory"] - memory * 32 > 0:
                        self.cloud_edge[i]["cpu"] = self.cloud_edge[i]["cpu"] - cpu * 25
                        self.cloud_edge[i]["memory"] = self.cloud_edge[i]["memory"] - memory * 32
                        return True
                    else:
                        return False

        elif platform == "cloud":
            for i in range(len(self.cloud)):
                if self.cloud[i]["node"] == node:
                    if self.cloud[i]["cpu"] - cpu * 25 > 0 and self.cloud[i]["memory"] - memory * 32 > 0:
                        self.cloud[i]["cpu"] = self.cloud[i]["cpu"] - cpu * 25
                        self.cloud[i]["memory"] = self.cloud[i]["memory"] - \
                            memory * 32
                        return True
                    else:
                        return False


class Min_memory():
    def __init__(self):
        # resource of platform
        self.device_dege = 1
        self.cloud_edge = 1
        self.cloud = 1

    def change(self, platform, memory):
        # change min allocatable memory
        if platform == "device_edge":
            self.device_dege = memory
        if platform == "cloud_edge":
            self.cloud_edge = memory
        if platform == "cloud":
            self.cloud = memory

    def get(self):
        # change min allocatable memory
        return self.device_dege, self.cloud_edge, self.cloud
