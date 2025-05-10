from flask import Flask, render_template, jsonify, request
import json

from network.utils.intent_handel import load_json_file
from utils.intentroute import NetworkPlanner
from utils.topo_handle import convert_json_to_echarts_topology, drawroute

app = Flask(__name__, template_folder='templates')

# 模拟生成拓扑数据的函数
def generate_topology_data():
    nodes = [
        {"name": "核心交换机", "value": [80, 80], "symbol": "image://https://cdn-icons-png.flaticon.com/512/126/126083.png","symbolSize": [80, 60]},
        {"name": "防火墙", "value": [80, 80], "symbol": "image://https://cdn-icons-png.flaticon.com/512/126/126102.png","symbolSize": [80, 60]},
        {"name": "Web服务器", "value": [80, 80], "symbol": "image://https://cdn-icons-png.flaticon.com/512/126/126091.png","symbolSize": [80, 60]},
        {"name": "数据库", "value": [80, 80], "symbol": "image://https://cdn-icons-png.flaticon.com/512/126/126094.png","symbolSize": [80, 60]},
        {"name": "办公网络", "value": [80, 80], "symbol": "image://https://cdn-icons-png.flaticon.com/512/126/126082.png","symbolSize": [80, 60]}
    ]

    links = [
        {"source": "核心交换机", "target": "防火墙", "bw": "50000", "weight": "8840", "delay": "1"},
        {"source": "防火墙", "target": "Web服务器", "bw": "50000", "weight": "8840", "delay": "1"},
        {"source": "防火墙", "target": "数据库", "bw": "50000", "weight": "8840", "delay": "1"},
        {"source": "核心交换机", "target": "办公网络", "bw": "50000", "weight": "8840", "delay": "1"}
    ]

    return {"nodes": nodes, "links": links}

# 主路由，渲染 HTML 模板
@app.route('/t', methods=['GET'])
def index():
    #topology_data = generate_topology_data()
    #return render_template('test.html')
    topology_data = convert_json_to_echarts_topology('./api/topo.json', './api/node_config.json')

    return render_template('test.html', topology_data=topology_data)




@app.route('/', methods=['GET'])
def final():
    route=load_json_file('./api/route.json')
    #route = load_json_file('./api/testroute.json')
    topology_data = drawroute('./api/topo.json', './api/node_config.json',route)

    return render_template('testshowroute.html', topology_data=topology_data)

# API 路由，返回拓扑数据（如果需要动态获取）
@app.route('/api/topology', methods=['GET'])
def get_topology():
    return jsonify(generate_topology_data())


@app.route('/api/test', methods=['POST'])
def handle_post():
    # 获取接收到的 JSON 数据
    data = request.get_json()

    planner = NetworkPlanner.from_config_file('../api/topo.json', '../api/node_config.json')
    #data解析
    path, compute_nodes = planner.find_time_optimal_route(
        src=0,
        dst=18,
        required_compute_nodes4=2,
        min_computing_power=10,  # 要求计算节点算力≥800
        min_bandwidth=10 # 要求链路带宽≥10
    )
    # 在控制台输出接收到的参数
    print("Received data:")
    print(data)

    # 返回响应
    return jsonify({"message": "Data received successfully", "your_data": data})

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0",port='12138')
