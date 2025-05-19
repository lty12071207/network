import json

from utils.cost_first import cNetworkPlanner
from utils.topo_handle import convert_json_to_echarts_topology
from utils.utl_first import uNetworkPlanner
from utils.intent_handel import load_json_file, extract_numbers
from utils.intentroute import NetworkPlanner
import time

#intent=./api/intent/i2.json

def createroutejson(intent):
    current_timestamp = int(time.time())
    data = load_json_file(intent)
    map = load_json_file('./api/address_function.json')
    planner = NetworkPlanner.from_config_file('./api/topo.json', './api/node_config.json')
    #planner2 = cNetworkPlanner.from_config_file('./api/topo.json', './api/node_config.json')
    #planner3 = uNetworkPlanner.from_config_file('./api/topo.json', './api/node_config.json')
    src = data['ip_info']['source_ip']
    dest = data['ip_info']['destination_ip']
    sid = map[src]
    did = map[dest]
    mtype = data['modality_info']['modality']
    method = data['modality_info']['method']
    gpu = data['computational_constraints']['gpu_power']
    min_computing_power = extract_numbers(gpu)[0]
    bw = data['network_qos_constraints']['bandwidth_requirement']

    # 将带宽转换为Mbps
    if 'Gbps' in bw:
        min_bandwidth = extract_numbers(bw)[0] * 1000  # 1 Gbps = 1000 Mbps
    elif 'Mbps' in bw:
        min_bandwidth = extract_numbers(bw)[0]
    else:
        min_bandwidth = extract_numbers(bw)[0]  # 默认假设为Mbps
    if method=='1':
        path, compute_nodes = planner.find_time_optimal_route(src=sid, dst=did, required_compute_nodes=2, min_computing_power=min_computing_power, min_bandwidth=min_bandwidth)
    elif method=='2':
        # path1, compute_nodes1 = planner2.ant_colony_optimization(source=sid, destination=did, num_computing_nodes=2,min_computing_power=min_computing_power,min_bandwidth=min_bandwidth,packet_size=100, num_ants=20, max_iter=50,
        #                         alpha=1, beta=2, rho=0.5)
        path1, compute_nodes1 =planner.find_cost_optimal_route(source=sid, destination=did, num_computing_nodes=2,
                                        min_computing_power=min_computing_power, min_bandwidth=min_bandwidth,
                                        packet_size=100, num_ants=20, max_iter=50,
                                        alpha=1, beta=2, rho=0.5)
        path=[]
        compute_nodes=[]
        for i in path1:
            path.append(int(str(i)))
        for i in compute_nodes1:
            compute_nodes.append(int(str(i)))



    elif method=='3':
        print(sid,did,min_computing_power,min_bandwidth)
        path1, compute_nodes1 = planner.find_utl_optimal_route(source=sid, destination=did, num_computing_nodes=2,min_computing_power=min_computing_power,min_bandwidth=min_bandwidth,packet_size=100, num_ants=20, max_iter=50,
                                alpha=1, beta=2, rho=0.5)
        path = []
        compute_nodes = []
        for i in path1:
            path.append(int(str(i)))
        for i in compute_nodes1:
            compute_nodes.append(int(str(i)))

    elif method=='4':
        print(sid,did,min_computing_power,min_bandwidth)
        path1, compute_nodes1 = planner.find_loss_optimal_route(source=sid, destination=did, num_computing_nodes=2,min_computing_power=min_computing_power,min_bandwidth=min_bandwidth,packet_size=100, num_ants=20, max_iter=50,
                                alpha=1, beta=2, rho=0.5)
        path = []
        compute_nodes = []
        for i in path1:
            path.append(int(str(i)))
        for i in compute_nodes1:
            compute_nodes.append(int(str(i)))

    # temppath,tempcompute_nodes= planner.find_cost_optimal_route(source=sid, destination=did, num_computing_nodes=2,min_computing_power=min_computing_power,min_bandwidth=min_bandwidth,packet_size=100, num_ants=20, max_iter=50,
    #                              alpha=1, beta=2, rho=0.5)
    # print(temppath==path)
    # print(tempcompute_nodes==compute_nodes)
    # print(path)
    # print(compute_nodes)
    # 将路径转换为边的集合
    # for i in temppath:
    #     print(type(i))
    #     i=int(i)
    #     print(type(i))
    edges = []
    for i in range(len(path) - 1):
        edges.append((path[i], path[i + 1]))
    # 构建 JSON 数据
    data = {
        'id': current_timestamp,
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

    #print(existing_es)
    #print(data['e'])

    # 如果当前 data 的 'e' 不在 existing_es 中，则添加到 existing_data
    if data['e'] not in existing_es:
        existing_data.append(data)
        topo=load_json_file('./api/topo.json')
        for i in topo:
            sd=set()
            sd.add(int(i['src']))
            sd.add(int(i['dest']))
            for j in data['e']:
                # print(type(j))
                if j[0] in sd and j[1] in sd:
                    i['bw']-=min_bandwidth
        with open('./api/topo.json', 'w', encoding='utf-8') as file:
            json.dump(topo, file, ensure_ascii=False, indent=4)
        node = load_json_file('./api/node_config.json')
        for i in node:
            for j in data['cn']:
                if i['id']==j :
                    i['computing']['gpu_power']-=min_computing_power
        with open('./api/node_config.json', 'w', encoding='utf-8') as file:
            json.dump(node, file, ensure_ascii=False, indent=4)
        # 将更新后的数据写回文件
        #print(existing_data)
        with open(file_name, 'w', encoding='utf-8') as file:
            json.dump(existing_data, file, ensure_ascii=False, indent=4)

    else:
        print("该路径的边集合已经存在于文件中，未添加重复项。")
    frontend=convert_json_to_echarts_topology('./api/topo.json', './api/node_config.json')
    #print(frontend)

    #planner.visualize(path, compute_nodes)

if __name__ == "__main__":
    createroutejson('./api/intent/i3.json')