import json
from collections import defaultdict
import random
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
                    "bw": random.randint(5000, 10000),
                    "delay": round(random.random()*3,2),
                    "lost": random.randint(0,100),
                    "LinkUtilization":random.randint(0,100)
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


# 使用示例：
# topology_data = convert_json_to_echarts_topology('output.json')
# print(json.dumps(topology_data, indent=4, ensure_ascii=False))




if __name__ == "__main__":
    # 获取脚本所在的目录
    #script_dir = os.path.dirname(os.path.abspath(__file__))

    # 构造文件的相对路径
    #input_file = os.path.join(script_dir, 'api', 'bandcon_topo.txt')
    #output_file = os.path.join(script_dir, 'api', 'topo.json')
    input_file = '../api/bandcon_topo.txt'  # 指定输入文件名
    output_file = '../api/topo.json'  # 指定输出文件名
    convert_txt_to_json(input_file, output_file)