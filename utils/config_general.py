import json
import random
def generate_node_config():
    nodes = []
    for i in range(20):
        cost=random.randint(100, 1000)
        node = {
            "id": i,
            "cost":cost,
            "storage": {},
            "computing": {}
        }

        # 随机选择需求等级
        levels = ["低", "中", "高"]
        level = random.choice(levels)

        # 设置存储需求
        if level == "低":
            node["storage"]["capacity"] = str(random.randint(1, 5))+'TB'  # 1 TB 到 5 TB
            node["storage"]["performance"] = str(random.randint(100, 250))+'MB/s'  # 100 MB/s 到 250 MB/s
        elif level == "中":
            node["storage"]["capacity"] = str(random.randint(5, 15))+'TB'  # 5 TB 到 15 TB
            node["storage"]["performance"] = str(random.randint(250, 500))+'MB/s'  # 250 MB/s 到 500 MB/s
        else:  # 高
            node["storage"]["capacity"] = str(random.randint(15, 50))+'TB'  # 15 TB 到 50 TB
            node["storage"]["performance"] = str(random.randint(500, 1000))+'MB/s'  # 500 MB/s 到 1000 MB/s

        # 设置算力需求
        if level == "低":
            node["computing"]["cpu_power"] = f"{random.randint(1, 4)} cores {random.randint(1500, 2250)}MHz"
            node["computing"]["gpu_power"] = random.randint(80, 100)
            node["computing"]["gpu_Utilization"] = random.randint(50, 100)
        elif level == "中":
            node["computing"]["cpu_power"] = f"{random.randint(4, 12)} cores {random.randint(2000, 2750)}MHz"
            node["computing"]["gpu_power"] = random.randint(101, 500)
            node["computing"]["gpu_Utilization"] = random.randint(50, 100)
        else:  # 高
            node["computing"]["cpu_power"] = f"{random.randint(12, 24)} cores {random.randint(2500, 3250)}MHz"
            node["computing"]["gpu_power"] = random.randint(501, 1000)
            node["computing"]["gpu_Utilization"] = random.randint(50, 100)
        nodes.append(node)
    return nodes


if __name__ == "__main__":
    node_config = generate_node_config()

    with open('../api/node_config.json', 'w') as f:
        json.dump(node_config, f, indent=4)