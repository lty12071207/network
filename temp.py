import json

from utils.intent_handel import load_json_file, extract_numbers
from utils.intentroute import NetworkPlanner
import time
current_timestamp = int(time.time())
data=load_json_file('./api/intent/i3.json')
map=load_json_file('./api/address_function.json')
planner = NetworkPlanner.from_config_file('./api/topo.json','./api/node_config.json')
src=data['ip_info']['source_ip']
dest=data['ip_info']['destination_ip']
sid=map[src]
did=map[dest]
mtype=data['modality_info']['modality']
method=data['modality_info']['method']
gpu=data['computational_constraints']['gpu_power']
min_computing_power=extract_numbers(gpu)[0]
bw=data['network_qos_constraints']['bandwidth_requirement']
min_bandwidth=extract_numbers(bw)[0]
path, compute_nodes = planner.find_time_optimal_route(src=sid,dst=did,required_compute_nodes=2,min_computing_power=min_computing_power,min_bandwidth=min_bandwidth)
print(path)
print(compute_nodes)
# 将路径转换为边的集合
edges = []
for i in range(len(path) - 1):
    edges.append((path[i], path[i + 1]))
# 构建 JSON 数据
data = {
    'id':current_timestamp,
    "mname": mtype,
    "rname": method,
    "e": edges,
    "cn": compute_nodes
}
print(data)
# 从文件中读取现有的 JSON 数据
file_name = "./api/route.json"
try:
    with open(file_name, 'r', encoding='utf-8') as file:
        existing_data = json.load(file)
except FileNotFoundError:
    # 如果文件不存在，初始化为空列表
    existing_data = []
for item in existing_data:
    item['e'] = [tuple(edge) for edge in item['e']]
# 检查是否已有相同的 'e' 字段
existing_es = [item.get('e') for item in existing_data]

print(existing_es)
print(data['e'])

# 如果当前 data 的 'e' 不在 existing_es 中，则添加到 existing_data
if data['e'] not in existing_es:
    existing_data.append(data)

    # 将更新后的数据写回文件
    with open(file_name, 'w', encoding='utf-8') as file:
        json.dump(existing_data, file, ensure_ascii=False, indent=4)
else:
    print("该路径的边集合已经存在于文件中，未添加重复项。")


planner.visualize(path, compute_nodes)