import json
from collections import defaultdict
import random

def load_json_file(file_path):
    """
    从 JSON 文件中加载数据。

    Args:
        file_path (str): JSON 文件的路径。

    Returns:
        dict: 文件中的 JSON 数据转换为字典格式。
             如果读取失败或文件不存在，则返回 None。
    """
    try:
        # 打开并读取文件
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file {file_path} contains invalid JSON.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None


def convert_txt_to_json(input_filename, output_filename='output.json'):
    """
    Converts a text file with link data into a JSON file.

    Args:
    input_filename: 此函数的输入是文本文件的路径
    output_filename: 此函数的输出是JSON文件的路径，默认为'output.json'
    """
    data = []
    with open(input_filename, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # 跳过第一行（标题行）
            if line.strip():  # 跳过空行
                label, src, dest, weight, bw, delay = line.strip().split()
                data.append({
                    "label": label,
                    "src": src,
                    "dest": dest,
                    "weight": random.randint(10, 100),
                    "bw": random.randint(9000, 10000),
                    "delay": round(random.random()*3,2),
                    "lost": random.randint(0,100),
                    "LinkUtilization":random.randint(50,100)
                })

    # 将转换后的数据保存到JSON文件中
    with open(output_filename, 'w') as f:
        json.dump(data, f, indent=4)

    return data


def convert_json_to_echarts_topology(input_json_file,config):
    """
    从 JSON 文件中读取数据，并转换为 ECharts 拓扑图所需的数据格式。

    Args:
        input_json_file: 输入的 JSON 文件路径

    Returns:
        topology_data: ECharts 拓扑图所需的数据格式
    """
    # 读取 JSON 文件
    with open(input_json_file, 'r') as f:
        data = json.load(f)

    degree_counter = defaultdict(int)
    for item in data:
        degree_counter[item["src"]] += 1
        degree_counter[item["dest"]] += 1

    # 提取节点名称和链接关系
    nodes = []
    links = []
    node_names = set()  # 用于记录已经添加过的节点名称



    for item in data:
        label = item.get("label")
        src = item.get("src")
        dest = item.get("dest")
        bw=item.get('bw')
        weight = item.get('weight')
        delay=item.get('delay')
        LinkUtilization=item.get('LinkUtilization')

        # 添加源节点（带度数判断）
        if src not in node_names:
            node_degree = degree_counter[src]
            nodes.append({
                "name": src,
                "value": [80, 80],
                "symbol": get_image_url(src, node_degree),
                "symbolSize": [80, 80],
                "degree": node_degree  # 可选：存储度数用于调试
            })
            node_names.add(src)

        # 添加目标节点（带度数判断）
        if dest not in node_names:
            node_degree = degree_counter[dest]
            nodes.append({
                "name": dest,
                "value": [80, 80],
                "symbol": get_image_url(src, node_degree),
                "symbolSize": [80, 80],
                "degree": node_degree  # 可选：存储度数用于调试
            })
            node_names.add(dest)

        # 添加链接
        links.append({"source": src, "target": dest,"bw":str(bw)+'Mbps', "weight":weight, "delay":str(delay)+'ms','utl':str(LinkUtilization)+'%'})

    # 构造拓扑数据
    with open(config, 'r') as f:
        data2 = json.load(f)
    for i in nodes:
        for j in data2:
            if str(i['name'])==str(j['id']):
                i['storage']={}
                i['computing'] = {}
                i['storage']['capacity']=j['storage']['capacity']
                i['storage']['performance'] = j['storage']['performance']
                i['computing']['cpu_power']= j['computing']['cpu_power']
                i['computing']['gpu_power'] = j['computing']['gpu_power']
                i['computing']['gpu_Utilization'] = str(j['computing']['gpu_Utilization'])+'%'

    topology_data = {
        "nodes": nodes,
        "links": links
    }

    return topology_data


def get_image_url(node_name, node_degree):
    """
    根据节点名称和度数返回图标URL
    规范图标说明：
    - 终端设备：使用工作站图标（度数=1）
    - 网络设备：使用路由器图标（度数>1）
    """
    # 专业图标资源（来自Flaticon商用授权图标）
    # image_url_map = {
    #     "terminal": "https://cdn-icons-png.flaticon.com/512/1698/1698522.png",  # 工作站图标
    #     "router": "https://cdn-icons-png.flaticon.com/512/4785/4785429.png",    # 路由器图标
    #     "default": "https://cdn-icons-png.flaticon.com/512/126/126083.png"      # 默认交换机图标
    # }
    image_url_map = {
        # 终端设备（度数为1）
        "terminal": "image://static/images/terminal.png",

        # 网络设备（度数>1）
        "router": "image://static/images/router.png",
        "switch": "https://img.icons8.com/ios-filled/100/000000/switch.png",
        "firewall": "https://img.icons8.com/ios-filled/100/000000/firewall.png",
        "server": "https://img.icons8.com/ios-filled/100/000000/server.png",

        # 默认设备
        "default": "https://img.icons8.com/ios-filled/100/000000/network-card.png"
    }

    # 根据度数选择图标类型
    if node_degree == 1:
        return image_url_map["terminal"]
    elif node_degree > 1:
        return image_url_map["router"]
    else:  # 理论上不会出现degree=0的情况
        return image_url_map["default"]


def drawroute(input_json_file, config, path_results):
    """
    扩展后的函数，支持路径计算结果注入

    Args:
        path_results: 路径计算结果，格式示例：
            [
                {
                    "id": "P001",
                    "rname": "带宽优先",
                    "rname": "BandwidthPriority",
                    "e": ["0", "15", "12"],
                    "cn": ["0", "15")]
                },
                # 其他路径...
            ]
    """
    strategy_color_map = {
        1: "#FF0000",  # 红
        2: "#FFFF00",  # 黄
        3: "#0000FF",  # 蓝
        4: "#00FF00"   # 绿
    }

    with open(input_json_file, 'r') as f:
        data = json.load(f)
    with open(config, 'r') as f:
        node_configs = json.load(f)

    # 基础拓扑构建（原逻辑）
    degree_counter = defaultdict(int)
    for item in data:
        degree_counter[item["src"]] += 1
        degree_counter[item["dest"]] += 1

    nodes, links = [], []
    # 添加基础节点和边（略，与用户原代码相同）
    node_names = set()  # 用于记录已经添加过的节点名称

    for item in data:
        label = item.get("label")
        src = item.get("src")
        dest = item.get("dest")
        bw = item.get('bw')
        weight = item.get('weight')
        delay = item.get('delay')
        LinkUtilization = item.get('LinkUtilization')

        # 添加源节点（带度数判断）
        if src not in node_names:
            node_degree = degree_counter[src]
            nodes.append({
                "name": src,
                "value": [80, 80],
                "symbol": get_image_url(src, node_degree),
                "symbolSize": [80, 80],
                "degree": node_degree  # 可选：存储度数用于调试
            })
            node_names.add(src)

        # 添加目标节点（带度数判断）
        if dest not in node_names:
            node_degree = degree_counter[dest]
            nodes.append({
                "name": dest,
                "value": [80, 80],
                "symbol": get_image_url(src, node_degree),
                "symbolSize": [80, 80],
                "degree": node_degree  # 可选：存储度数用于调试
            })
            node_names.add(dest)

        # 添加链接
        links.append(
            {"source": src,
             "target": dest,
             "bw": str(bw) + 'Mbps',
             "weight": weight,
             "delay": str(delay) + 'ms',
             'utl': str(LinkUtilization) + '%',
             "path_id": None,
             "path_name": None,
             "strategy": None,  # 传递策略类型
             "lineStyle": {"color": "#000000","curveness":0},  # 根据策略设置颜色
             "label": {"show": False, "formatter": None}
             })

    # 构造拓扑数据
    with open(config, 'r') as f:
        data2 = json.load(f)
    for i in nodes:
        for j in data2:
            if str(i['name']) == str(j['id']):
                i['storage'] = {}
                i['computing'] = {}
                i['cost'] = j['cost']
                i['storage']['capacity'] = j['storage']['capacity']
                i['storage']['performance'] = j['storage']['performance']
                i['computing']['cpu_power'] = j['computing']['cpu_power']
                i['computing']['gpu_power'] = j['computing']['gpu_power']
                i['computing']['gpu_Utilization'] = str(j['computing']['gpu_Utilization']) + '%'
                i['path_info']=[]

    # ---- 新增：处理路径计算结果 ----
    # 统计节点被多少路径经过
    node_path_counts = defaultdict(int)
    path_edges = []
    node_dict = {node['name']: node for node in nodes}  # 新增
    #node_dict=nodes
    for path in path_results:
        strategy = path.get("rname")
        # 根据策略类型获取颜色，默认灰色（#CCCCCC）
        color = strategy_color_map.get(int(strategy))
        path_id = path["id"]
        path_name = path.get("mname", path.get("rname", "Unnamed Path"))  # 优先使用mname

        # 记录节点被路径经过的次数
        # for node in path["cn"]:
        #     node_path_counts[node] += 1

        for node_id in path["cn"]:
            # 统计经过次数
            node_path_counts[node_id] += 1

            # 获取节点对象
            node = node_dict.get(str(node_id))  # 确保类型一致
            if node:
                # 初始化路径信息列表
                if 'path_info' not in node:
                    node['path_info'] = []

                # 避免重复添加相同路径
                if not any(p['path_id'] == path_id for p in node['path_info']):
                    node['path_info'].append({
                        "path_id": path_id,
                        "path_name": path_name,
                        "color": color
                    })


        # 生成路径边（独立边）
        for src, target in path["e"]:
            # 查找原始边属性
            original_edge = next(
                (e for e in data if e["src"] == src and e["dest"] == target),
                {"bw": "N/A", "delay": "N/A"}
            )
            path_edges.append({
                "source": str(src),
                "target": str(target),
                "bw": original_edge.get("bw", "N/A"),
                "delay": original_edge.get("delay", "N/A"),
                "utl": original_edge.get("LinkUtilization", "N/A"),
                "path_id": path["id"],
                "path_name": path["mname"],
                "strategy": int(strategy),  # 传递策略类型
                "lineStyle": {"color": color,'curveness':-100},  # 根据策略设置颜色
                "label": {"show": False, "formatter": path["mname"]}
            })

    for node in nodes:
        count = node_path_counts.get(int(node["name"]), 0)
        base_size = 80
        node["symbolSize"] = [base_size * (1 + 0.5 * count)] * 2

        if count > 0:
            # 使用高对比度颜色配置
            border_color = "#00FF00" if count <= 2 else "#FFD700"  # 亮绿/金色
            pulse_effect = {
                "period": 2,  # 呼吸周期2秒
                "loop": True,
                "effect": {
                    "borderColor": border_color,
                    "borderWidth": 3 + count * 2
                }
            }

            node.update({
                "itemStyle": {
                    "borderColor": border_color,
                    "borderWidth": 5 + count,  # 加粗边框
                    "borderType": "solid",
                    "color": "rgba(255,255,255,0)",  # 完全透明填充
                    "shadowBlur": 20 + count * 5,  # 增强阴影
                    "shadowColor": "#AAAFFF"  # 金色阴影
                },
                "emphasis": {
                    "itemStyle": {
                        "borderWidth": 8 + count,
                        "shadowBlur": 30
                    }
                },
                "animationEasing": "elasticOut",
                "animationDuration": 1000,
                "animationDelay": count * 100,
                "pulseEffect": pulse_effect  # 添加呼吸动画
            })

            # 添加外发光特效（覆盖在图片上层）
            node.setdefault("emphasis", {}).update({
                "focus": "self",
                "blurScope": "coordinateSystem",
                "label": {
                    "show": True,
                    "formatter": f"★{count}",
                    "color": "#ABCDEF",
                    "fontSize": 14 + count * 2,
                    "position": "top"
                }
            })


    # 1. 标准化边分组（不考虑方向）
    edge_groups = defaultdict(list)
    for edge in path_edges:
         # 标准化节点对（排序后小值在前）
        sorted_nodes = sorted([edge["source"], edge["target"]], key=int)
        group_key = f"{sorted_nodes[0]}-{sorted_nodes[1]}"
        edge_groups[group_key].append(edge)

        # 2. 为每组边分配递增曲率
    for group_key, edges in edge_groups.items():
            # 对每个边按顺序分配0.02、0.04、0.06...的曲率
        for idx, edge in enumerate(edges):
                # 基础曲率步长设为0.02
            base_curveness = 0.04
                # 计算当前边的曲率（使用索引+1保证最小0.02）
            curveness = base_curveness * (idx + 1)
                # 应用曲率设置
            edge["lineStyle"]["curveness"] = curveness

    for i in nodes:
        i=node_dict.get(i['name'])

    topology_data = {
        "nodes": nodes,
        "links": links+path_edges,
        "path_config": {
            # 生成策略类型到颜色的映射表（去重）
            "strategies": {
                s: strategy_color_map.get(int(s), "#CCCCCC")
                for s in {p["rname"] for p in path_results}
            }
        }
    }
    return topology_data



if __name__ == "__main__":
    # 获取脚本所在的目录
    #script_dir = os.path.dirname(os.path.abspath(__file__))

    # 构造文件的相对路径
    #input_file = os.path.join(script_dir, 'api', 'bandcon_topo.txt')
    #output_file = os.path.join(script_dir, 'api', 'topo.json')
    input_file = '../api/bandcon_topo.txt'  # 指定输入文件名
    output_file = '../api/topo.json'  # 指定输出文件名
    route = load_json_file('../api/route.json')
    #route = load_json_file('./api/testroute.json')
    #topology_data = drawroute('../api/topo.json', '../api/node_config.json', route)
    convert_txt_to_json(input_file, output_file)