import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import json
import time
from collections import defaultdict
import heapq
class NetworkPlanner:
    def __init__(self, num_nodes, num_links):
        """
        初始化网络规划器
        :param num_nodes: 节点数量
        :param num_links: 链路数量
        """
        self.num_nodes = num_nodes
        self.num_links = num_links
        self.G = nx.Graph()  # 创建一个空的无向图

        # 生成节点（每个节点有计算能力）
        for node_id in range(num_nodes):
            # 随机分配计算能力 100-1000 单位
            computing_power = random.randint(100, 1000)
            # 向图中添加节点，并设置计算能力和节点 ID 属性
            self.G.add_node(node_id, computing_power=computing_power, id=node_id)

        # 生成链路（确保连通性）
        if num_links < num_nodes - 1:
            # 若链路数量少于节点数减一，无法保证网络连通，抛出异常
            raise ValueError("链路数量必须至少为节点数量减一以保持连通性")

        # 首先生成一个生成树确保连通性
        tree = nx.minimum_spanning_tree(nx.complete_graph(num_nodes))
        for u, v in tree.edges():
            # 随机分配传播时延 1-10 单位
            propagation_delay = random.randint(1, 10)
            # 向图中添加边，并设置传播时延属性
            self.G.add_edge(u, v, propagation_delay=propagation_delay)

        # 添加剩余的链路
        remaining_links = num_links - (num_nodes - 1)
        # 获取图中所有不存在的边
        possible_edges = list(nx.non_edges(self.G))
        # 随机打乱可能的边的顺序
        random.shuffle(possible_edges)
        for u, v in possible_edges[:remaining_links]:
            propagation_delay = random.randint(1, 10)
            self.G.add_edge(u, v, propagation_delay=propagation_delay)


    def apply_constraints(self,num_computing_nodes,min_computing_power,min_bandwidth):
        """
        应用节点算力和边带宽的约束条件
        """

        self.required_compute_nodes = num_computing_nodes
        self.min_computing_power = min_computing_power
        self.min_bandwidth = min_bandwidth
        # 删除不满足算力约束的节点及其相连的边
        nodes_to_remove = [node for node, data in self.G.nodes(data=True) if
                           data['computing_power'] < self.min_computing_power]
        for node in nodes_to_remove:
            self.G.remove_node(node)

        # 删除不满足带宽约束的边
        edges_to_remove = [(u, v) for u, v, data in self.G.edges(data=True) if data['bandwidth'] < self.min_bandwidth]
        for u, v in edges_to_remove:
            self.G.remove_edge(u, v)

    @classmethod
    def from_config_file(cls, topofile,nodefile):
        """
        从配置文件中读取网络信息并初始化网络规划器
        :param config_file_path: 配置文件路径
        :return: NetworkPlanner 实例
        """
        with open(topofile, 'r') as f:
            # 读取配置文件内容
            configlink = json.load(f)
        with open(nodefile, 'r') as f:
            # 读取配置文件内容
            confignode = json.load(f)
        num_links = len(configlink)
        num_nodes = len(confignode)
        planner = cls(num_nodes, num_links)
        planner.G = nx.Graph()

        # 添加节点
        for node in confignode:
            node_id = int(node["id"])
            computing_power = node["computing"]["cpu_power"]
            gpu_power=node["computing"]["gpu_power"]
            capacity=node["storage"]["capacity"]
            performance=node["storage"]["performance"]
            cost = node["cost"]
            uti= node["computing"]["gpu_Utilization"]
            # 从配置文件读取节点信息并添加到图中
            planner.G.add_node(node_id, computing_power=gpu_power, id=node_id, cost=cost,gpu_power=gpu_power,capacity=capacity,performance=performance,uti=uti)

        # 添加边
        for edge in configlink:
            u = int(edge["src"])
            v = int(edge["dest"])
            propagation_delay = edge["delay"]
            cost = int(edge["weight"])
            bandwith=edge['bw']
            loss=edge['lost']
            luti=edge['LinkUtilization']
            # 从配置文件读取边信息并添加到图中
            planner.G.add_edge(u, v, propagation_delay=propagation_delay, cost=cost,bandwidth=bandwith,loss=loss,luti=luti)

        return planner

    def visualize(self, path=None, selected_nodes=None):
        """
        可视化网络拓扑图
        :param path: 可选参数，要高亮显示的路径
        :param selected_nodes: 可选参数，选择的计算节点列表
        """
        # 使用弹簧布局算法确定节点位置
        pos = nx.spring_layout(self.G, k=0.5, iterations=50)

        # 绘制节点
        node_colors = []
        for node in self.G.nodes():
            if path and node == path[0]:
                # 若节点是路径的起点，标记为黄色
                node_colors.append('yellow')
            elif path and node == path[-1]:
                # 若节点是路径的终点，标记为黄色
                node_colors.append('yellow')
            elif selected_nodes and node in selected_nodes:
                # 若节点在选择的计算节点列表中，标记为绿色
                node_colors.append('green')
            elif path and node in path:
                # 若节点在指定路径中（但不是起点或终点），标记为红色
                node_colors.append('red')
            else:
                # 否则标记为蓝色
                node_colors.append('blue')

        # 创建一个新的图形对象
        plt.figure()
        plt.title('Network Topology')
        plt.axis('off')  # 隐藏坐标轴

        # 绘制图的节点
        nx.draw(self.G, pos, with_labels=True, node_color=node_colors, node_size=700)

        # 绘制边
        if path:
            # 获取路径中的边
            path_edges = list(zip(path, path[1:]))
            for edge in self.G.edges():
                if (edge in path_edges) or (edge[::-1] in path_edges):
                    # 若边在指定路径中，标记为红色
                    nx.draw_networkx_edges(self.G, pos, edgelist=[edge], edge_color='red', width=2)
        else:
            # 若未指定路径，绘制黑色边
            nx.draw_networkx_edges(self.G, pos, edge_color='black', width=2)

        # 添加节点标签（ID和其他属性）
        labels = {node: f"ID: {node}\nGPU: {data['gpu_power']}\nCost: {data['cost']}" for node, data in self.G.nodes(data=True)}
        nx.draw_networkx_labels(self.G, pos, labels=labels, font_size=8)

        # 显示图形
        plt.show()

    def get_all_routes(self, source, destination):
        """
        获取从源节点到目的节点的所有路径
        :param source: 源节点ID
        :param destination: 目的节点ID
        :return: 所有路径的列表
        """
        if source not in self.G.nodes():
            # 若源节点不存在，抛出异常
            raise ValueError(f"源节点 {source} 不存在")
        if destination not in self.G.nodes():
            # 若目的节点不存在，抛出异常
            raise ValueError(f"目的节点 {destination} 不存在")

        # 使用深度优先搜索获取所有路径
        all_routes = []
        # 记录已访问的节点
        visited = set()
        # 记录当前路径
        path = []

        def dfs(node):
            # 将当前节点添加到路径中
            path.append(node)
            # 标记当前节点为已访问
            visited.add(node)
            if node == destination:
                # 若到达目的节点，将当前路径的副本添加到结果列表中
                all_routes.append(path.copy())
            else:
                # 遍历当前节点的所有未访问邻居节点
                for neighbor in self.G.neighbors(node):
                    if neighbor not in visited:
                        dfs(neighbor)
            # 回溯，移除当前节点
            path.pop()
            # 标记当前节点为未访问
            visited.remove(node)

        dfs(source)
        return all_routes

    def get_node_computing_cost(self, node_id):
        """
        获取指定节点的计算成本
        :param node_id: 节点ID
        :return: 节点的计算成本
        """
        if node_id not in self.G.nodes():
            # 若节点不存在，抛出异常
            raise ValueError(f"节点 {node_id} 不存在")
        # 直接访问节点的计算能力属性
        return self.G.nodes[node_id]['cost']

    def get_link_propagation_cost(self, u, v):
        """
        获取指定两边之间的传播成本
        :param u: 第一个节点ID
        :param v: 第二个节点ID
        :return: 两边之间的传播成本
        """
        if u not in self.G.nodes() or v not in self.G.nodes():
            # 若任一节点不存在，抛出异常
            raise ValueError(f"节点 {u} 或 {v} 不存在")
        if self.G.has_edge(u, v):
            # 若边存在，直接访问边的传播时延属性
            return self.G[u][v]['cost']
        else:
            # 若边不存在，抛出异常
            raise ValueError(f"节点 {u} 和 {v} 之间不存在链路")

    def find_time_optimal_route(self, source, destination, num_computing_nodes,min_computing_power,min_bandwidth,packet_size=100, num_ants=20, max_iter=50,
                                alpha=1, beta=2, rho=0.5):
        """
        改进的贪心算法实现：优先选择GPU最高的计算节点
        :param src: 源节点ID
        :param dst: 目的节点ID
        :param required_compute_nodes: 需要选择的计算节点数量
        :param min_computing_power: 计算节点最低GPU要求
        :param min_bandwidth: 链路最低带宽要求
        :return: (路径节点列表, 使用的计算节点列表, 总时间)
        """
        # # 参数校验
        # self._validate_nodes_exist([src, dst])
        # if required_compute_nodes < 0:
        #     raise ValueError("需要计算节点数不能为负")
        #
        # # 生成所有简单路径并筛选
        # valid_paths = []
        # for path in nx.all_simple_paths(self.G, src, dst):
        #     # 提取中间节点（排除源和目的）
        #     intermediates = path[1:-1]
        #
        #     # 筛选满足GPU的节点
        #     valid_nodes = [n for n in intermediates
        #                    if self.G.nodes[n]['gpu_power'] >= min_computing_power]
        #     # 检查计算节点数量
        #     if len(valid_nodes) < required_compute_nodes:
        #         continue
        #
        #     # 检查带宽约束
        #     if not self._check_bandwidth(path, min_bandwidth):
        #         continue
        #
        #     # 记录候选路径及其特征
        #     valid_paths.append((
        #         path,
        #         self._select_best_nodes(valid_nodes, required_compute_nodes)
        #     ))
        #
        # if not valid_paths:
        #     return [], [], float('inf')
        #
        # # 选择GPU总和最大的路径
        # best_path, best_nodes = max(valid_paths,key=lambda x: sum(self.G.nodes[n]['gpu_power'] for n in x[1]))
        #
        # return best_path, best_nodes
        self.apply_constraints(num_computing_nodes, min_computing_power, min_bandwidth)
        num_nodes = len(self.G.nodes())
        # 初始化信息素矩阵
        pheromone = np.ones((num_nodes, num_nodes))
        best_path = None
        best_delay = float('inf')

        for _ in range(max_iter):
            all_ant_paths = []
            all_ant_delays = []

            for _ in range(num_ants):
                # 初始化蚂蚁路径，从源节点开始
                path = [source]
                current = source
                while current != destination:
                    # 获取当前节点的所有邻居节点
                    neighbors = list(self.G.neighbors(current))
                    # 获取未访问过的邻居节点
                    unvisited_neighbors = [n for n in neighbors if n not in path]
                    if not unvisited_neighbors:
                        break
                    probabilities = []
                    for neighbor in unvisited_neighbors:
                        # 计算启发式信息：传播时延和计算能力的倒数
                        edge_delay = self.G[current][neighbor]['propagation_delay']
                        node_computing_power = self.G.nodes[neighbor]['computing_power']
                        compute_delay = packet_size / node_computing_power
                        total_delay = edge_delay + compute_delay
                        heuristic = 1 / total_delay
                        pheromone_value = pheromone[current][neighbor]
                        probability = (pheromone_value ** alpha) * (heuristic ** beta)
                        probabilities.append(probability)
                    # 计算概率分布
                    probabilities = np.array(probabilities) / np.sum(probabilities)
                    # 根据概率选择下一个节点
                    next_node = np.random.choice(unvisited_neighbors, p=probabilities)
                    path.append(next_node)
                    current = next_node

                if len(path) - 2 >= num_computing_nodes and path[-1] == destination:
                    all_ant_paths.append(path)
                    total_delay = self.calculate_total_delay(path, packet_size, num_computing_nodes)
                    all_ant_delays.append(total_delay)
                    if total_delay < best_delay:
                        best_delay = total_delay
                        best_path = path

            # 更新信息素
            pheromone *= (1 - rho)  # 信息素挥发
            if all_ant_paths:
                for path, delay in zip(all_ant_paths, all_ant_delays):
                    delta_pheromone = 1 / delay
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i + 1]
                        pheromone[u][v] += delta_pheromone
                        pheromone[v][u] += delta_pheromone
        best_path= [int(node) if isinstance(node, np.int64) else node for node in best_path]

        intermediate_nodes = best_path[1:-1]
        sorted_nodes = sorted(intermediate_nodes, key=lambda node: planner.get_node_computing_power(node), reverse=True)
        # 选取计算能力最强的前三个节点
        top_three_computing_nodes = sorted_nodes[:3]
        return best_path, top_three_computing_nodes
    def calculate_total_delay(self, path, packet_size, num_computing_nodes):
        """
        计算路径的总时延
        :param path: 路径
        :param packet_size: 数据包大小
        :param num_computing_nodes: 计算节点数
        :return: 总时延
        """
        total_delay = 0
        # 计算传播时延
        for i in range(len(path) - 1):
            total_delay += self.get_link_propagation_delay(path[i], path[i + 1])

        # 提取中间节点
        intermediate_nodes = path[1:-1]
        # 根据计算能力对中间节点进行排序
        sorted_nodes = sorted(intermediate_nodes, key=lambda node: self.get_node_computing_power(node), reverse=True)
        # 选取计算能力最强的前三个节点
        top_three_computing_nodes = sorted_nodes[:3]

        # 计算计算能力最强的前三个节点的计算时延
        for node in top_three_computing_nodes:
            computing_power = self.get_node_computing_power(node)
            compute_delay = packet_size / computing_power
            total_delay += compute_delay

        return total_delay
    def get_node_computing_power(self, node_id):
        """
        获取指定节点的计算能力
        :param node_id: 节点ID
        :return: 节点的计算能力
        """
        if node_id not in self.G.nodes():
            # 若节点不存在，抛出异常
            raise ValueError("节点 %s 不存在" % node_id)
        # 直接访问节点的计算能力属性
        return self.G.nodes[node_id]['computing_power']

    def get_link_propagation_delay(self, u, v):
        """
        获取指定两边之间的传播时延
        :param u: 第一个节点ID
        :param v: 第二个节点ID
        :return: 两边之间的传播时延
        """
        if u not in self.G.nodes() or v not in self.G.nodes():
            # 若任一节点不存在，抛出异常
            raise ValueError("节点 %s 或 %s 不存在" % (u, v))
        if self.G.has_edge(u, v):
            # 若边存在，直接访问边的传播时延属性
            return self.G[u][v]['propagation_delay']
        else:
            # 若边不存在，抛出异常
            raise ValueError("节点 %s 和 %s 之间不存在链路" % (u, v))

    def _select_best_nodes(self, candidates, required):
        """选择GPU最高的前required个节点"""
        sorted_nodes = sorted(candidates,
                              key=lambda n: self.G.nodes[n]['gpu_power'],
                              reverse=True)
        return sorted_nodes[:required]

    def _check_bandwidth(self, path, min_bw):
        """检查路径带宽是否达标"""
        for u, v in zip(path, path[1:]):
            if int(self.G.edges[u, v]['bandwidth']) < min_bw:
                return False
        return True


    def _validate_nodes_exist(self, nodes):
        """验证节点存在性"""
        for node in nodes:
            if node not in self.G.nodes:
                raise ValueError(f"节点 {node} 不存在")


    def find_cost_optimal_route(self, source, destination, num_computing_nodes,min_computing_power,min_bandwidth,packet_size=100, num_ants=20, max_iter=50,
                                alpha=1, beta=2, rho=0.5):
        """
        使用蚁群算法搜索成本最小路径
        :param source: 源节点 ID
        :param destination: 目的节点 ID
        :param packet_size: 数据包大小
        :param num_computing_nodes: 计算节点数
        :param num_ants: 蚂蚁数量
        :param max_iter: 最大迭代次数
        :param alpha: 信息素重要程度因子
        :param beta: 启发式因子
        :param rho: 信息素挥发因子
        :return: 成本最小路径，如果不存在满足条件的路径则返回 None
        """
        self.apply_constraints(num_computing_nodes,min_computing_power,min_bandwidth)
        num_nodes = len(self.G.nodes())
        # 初始化信息素矩阵
        pheromone = np.ones((num_nodes, num_nodes))
        best_path = None
        best_cost = float('inf')

        # 提前计算所有边和节点的最大成本，用于归一化
        max_edge_cost = max(self.G[u][v]['cost'] for u, v in self.G.edges())
        max_node_cost = max(self.G.nodes[node]['cost'] for node in self.G.nodes())

        for _ in range(max_iter):
            all_ant_paths = []
            all_ant_cost = []

            for _ in range(num_ants):
                # 初始化蚂蚁路径，从源节点开始
                path = [source]
                current = source
                while current != destination:
                    # 获取当前节点的所有邻居节点
                    neighbors = list(self.G.neighbors(current))
                    # 获取未访问过的邻居节点
                    unvisited_neighbors = [n for n in neighbors if n not in path]
                    if not unvisited_neighbors:
                        break
                    probabilities = []
                    for neighbor in unvisited_neighbors:
                        # 计算启发式信息：成本的倒数，并进行归一化处理
                        edge_cost = self.G[current][neighbor]['cost'] / max_edge_cost
                        node_computing_cost = self.G.nodes[neighbor]['cost'] / max_node_cost
                        total_cost = node_computing_cost + edge_cost
                        heuristic = 1 / (total_cost + 1e-6)  # 避免除零错误
                        pheromone_value = pheromone[current][neighbor]
                        probability = (pheromone_value ** alpha) * (heuristic ** beta)
                        probabilities.append(probability)
                    # 计算概率分布
                    probabilities = np.array(probabilities) / np.sum(probabilities)
                    # 根据概率选择下一个节点
                    next_node = np.random.choice(unvisited_neighbors, p=probabilities)
                    path.append(next_node)
                    current = next_node

                if len(path) - 2 >= num_computing_nodes and path[-1] == destination:
                    all_ant_paths.append(path)
                    total_cost = self.calculate_total_cost(path, num_computing_nodes)
                    all_ant_cost.append(total_cost)
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_path = path

            # 更新信息素
            pheromone *= (1 - rho)  # 信息素挥发
            if all_ant_paths:
                # 归一化所有路径的成本
                min_cost = min(all_ant_cost)
                max_cost = max(all_ant_cost)
                normalized_costs = [(cost - min_cost) / (max_cost - min_cost + 1e-6) for cost in all_ant_cost]

                for path, norm_cost in zip(all_ant_paths, normalized_costs):
                    delta_pheromone = 1 / (norm_cost + 1e-6)  # 避免除零错误
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i + 1]
                        pheromone[u][v] += delta_pheromone
                        pheromone[v][u] += delta_pheromone
        if best_path is None:
            return [], []
        best_path = [int(node) if isinstance(node, np.int64) else node for node in best_path]
        intermediate_nodes = best_path[1:-1]
        sorted_nodes = sorted(intermediate_nodes, key=lambda node: self.get_node_cost(node))
        # 选取成本最小的的前 num_computing_nodes 个节点
        top_three_computing_nodes = sorted_nodes[:num_computing_nodes]
        return best_path,top_three_computing_nodes

    def calculate_total_cost(self, path,  num_computing_nodes):
        """
        计算路径的总成本
        :param path: 路径
        :param packet_size: 数据包大小
        :param num_computing_nodes: 计算节点数
        :return: 总成本
        """
        total_cost = 0

        for i in range(len(path) - 1):
            total_cost += self.get_link_propagation_cost(path[i], path[i + 1])

        # 提取中间节点
        intermediate_nodes = path[1:-1]
        # 根据成本对中间节点进行排序
        sorted_nodes = sorted(intermediate_nodes, key=lambda node: self.get_node_cost(node))
        # 选取成本最小的前三个节点
        top_three_computing_nodes = sorted_nodes[:num_computing_nodes]

        # 计算计算能力最强的前三个节点的计算时延
        for node in top_three_computing_nodes:
            computing_power_cost = self.get_node_cost(node)

            total_cost += computing_power_cost

        return total_cost





    def find_utl_optimal_route(self, source, destination, num_computing_nodes,min_computing_power,min_bandwidth,packet_size=100, num_ants=20, max_iter=50,
                                alpha=1, beta=2, rho=0.5):
        """
        使用蚁群算法搜索利用率最小路径
        :param source: 源节点 ID
        :param destination: 目的节点 ID
        :param packet_size: 数据包大小
        :param num_computing_nodes: 计算节点数
        :param num_ants: 蚂蚁数量
        :param max_iter: 最大迭代次数
        :param alpha: 信息素重要程度因子
        :param beta: 启发式因子
        :param rho: 信息素挥发因子
        :return: 利用率最小路径，如果不存在满足条件的路径则返回 None
        """
        self.apply_constraints(num_computing_nodes,min_computing_power,min_bandwidth)
        num_nodes = len(self.G.nodes())
        # 初始化信息素矩阵
        pheromone = np.ones((num_nodes, num_nodes))
        best_path = None
        best_utilization = float('inf')

        # 提前计算所有节点和边的最大利用率，用于归一化
        max_node_uti = max(self.G.nodes[node]['uti'] for node in self.G.nodes())
        max_link_luti = max(self.G[u][v]['luti'] for u, v in self.G.edges())

        for _ in range(max_iter):
            all_ant_paths = []
            all_ant_utilization = []

            for _ in range(num_ants):
                # 初始化蚂蚁路径，从源节点开始
                path = [source]
                current = source
                while current != destination:
                    # 获取当前节点的所有邻居节点
                    neighbors = list(self.G.neighbors(current))
                    # 获取未访问过的邻居节点
                    unvisited_neighbors = [n for n in neighbors if n not in path]
                    if not unvisited_neighbors:
                        break
                    probabilities = []
                    for neighbor in unvisited_neighbors:
                        # 计算启发式信息：利用率的倒数，并进行归一化处理
                        node_uti = self.G.nodes[neighbor]['uti'] / max_node_uti
                        link_luti = self.G[current][neighbor]['luti'] / max_link_luti
                        total_utilization = node_uti + link_luti
                        heuristic = 1 / (total_utilization + 1e-6)  # 避免除零错误
                        pheromone_value = pheromone[current][neighbor]
                        probability = (pheromone_value ** alpha) * (heuristic ** beta)
                        probabilities.append(probability)
                    # 计算概率分布
                    probabilities = np.array(probabilities) / np.sum(probabilities)
                    # 根据概率选择下一个节点
                    next_node = np.random.choice(unvisited_neighbors, p=probabilities)
                    path.append(next_node)
                    current = next_node

                if len(path) - 2 >= num_computing_nodes and path[-1] == destination:
                    all_ant_paths.append(path)
                    total_utilization = self.calculate_total_utilization(path, num_computing_nodes)
                    all_ant_utilization.append(total_utilization)
                    if total_utilization < best_utilization:
                        best_utilization = total_utilization
                        best_path = path

            # 更新信息素
            pheromone *= (1 - rho)  # 信息素挥发
            if all_ant_paths:
                # 归一化所有路径的利用率
                min_utilization = min(all_ant_utilization)
                max_utilization = max(all_ant_utilization)
                normalized_utilization = [(util - min_utilization) / (max_utilization - min_utilization + 1e-6) for util in all_ant_utilization]

                for path, norm_util in zip(all_ant_paths, normalized_utilization):
                    delta_pheromone = 1 / (norm_util + 1e-6)  # 避免除零错误
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i + 1]
                        pheromone[u][v] += delta_pheromone
                        pheromone[v][u] += delta_pheromone
        if best_path is None:
            return [], []
        best_path = [int(node) if isinstance(node, np.int64) else node for node in best_path]

        intermediate_nodes = best_path[1:-1]
        sorted_nodes = sorted(intermediate_nodes, key=lambda node: self.G.nodes[node]['uti'])
        # 选取 uti 最小的的前三个节点
        top_three_computing_nodes = sorted_nodes[:num_computing_nodes]
        return best_path,top_three_computing_nodes




    def find_utl_optimal_route(self, source, destination, num_computing_nodes, min_computing_power, min_bandwidth):
        """
        找出利用率最小的路径
        :param source: 源节点 ID
        :param destination: 目的节点 ID
        :param num_computing_nodes: 计算节点数
        :param min_computing_power: 计算节点最低算力要求
        :param min_bandwidth: 链路最低带宽要求
        :return: 利用率最小化路径和选中的计算节点，如果不存在满足条件的路径则返回空列表
        """
        self.apply_constraints(num_computing_nodes, min_computing_power, min_bandwidth)

        # 检查源节点和目标节点是否存在
        if source not in self.G.nodes() or destination not in self.G.nodes():
            return [], []

        # 获取所有可能的路径
        all_paths = list(nx.all_simple_paths(self.G, source, destination))

        valid_paths = []
        for path in all_paths:
            # 检查中间节点数量是否满足计算节点要求
            intermediate_nodes = path[1:-1]
            if len(intermediate_nodes) >= num_computing_nodes:
                valid_paths.append(path)

        if not valid_paths:
            return [], []

        # 计算每条有效路径的利用率
        utilization_per_path = []
        for path in valid_paths:
            total_utilization = self.calculate_total_utilization(path, num_computing_nodes)
            utilization_per_path.append(total_utilization)

        # 找到利用率最小的路径索引
        min_utilization_index = utilization_per_path.index(min(utilization_per_path))
        best_path = valid_paths[min_utilization_index]

        # 选取中间节点作为计算节点
        intermediate_nodes = best_path[1:-1]
        sorted_nodes = sorted(intermediate_nodes, key=lambda node: self.G.nodes[node]['uti'])
        top_computing_nodes = sorted_nodes[:num_computing_nodes]

        return best_path, top_computing_nodes

    def calculate_total_utilization(self, path, num_computing_nodes):
        """
        计算路径的总利用率
        :param path: 路径
        :param num_computing_nodes: 计算节点数
        :return: 总利用率
        """
        total_utilization = 0

        # 累加边的利用率
        for i in range(len(path) - 1):
            total_utilization += self.G[path[i]][path[i + 1]]['luti']

        # 提取中间节点
        intermediate_nodes = path[1:-1]
        # 根据 uti 对中间节点进行排序
        sorted_nodes = sorted(intermediate_nodes, key=lambda node: self.G.nodes[node]['uti'])
        # 选取 uti 最小的前 num_computing_nodes 个节点
        top_computing_nodes = sorted_nodes[:num_computing_nodes]

        # 计算 uti 最小的前 num_computing_nodes 个节点的 uti 总和
        for node in top_computing_nodes:
            total_utilization += self.G.nodes[node]['uti']

        return total_utilization

    def find_shortest_path_dijkstra(self, source, destination, num_computing_nodes,min_computing_power,min_bandwidth):
        """
        使用迪杰斯特拉算法搜索最短路径（基于跳数）
        :param source: 源节点 ID
        :param destination: 目的节点 ID
        :return: 最短路径，如果不存在路径则返回空列表
        """
        # 验证节点是否存在
        self._validate_nodes_exist([source, destination])

        # 初始化距离字典，所有节点的初始距离设为无穷大
        distances = {node: float('inf') for node in self.G.nodes()}
        distances[source] = 0

        # 初始化前驱节点字典，用于记录路径
        predecessors = {node: None for node in self.G.nodes()}

        # 优先队列，存储 (距离, 节点) 元组
        priority_queue = [(0, source)]

        while priority_queue:
            # 从优先队列中取出距离最小的节点
            current_distance, current_node = heapq.heappop(priority_queue)

            # 如果已经到达目标节点，提前退出循环
            if current_node == destination:
                break

            # 如果当前距离大于已记录的距离，跳过
            if current_distance > distances[current_node]:
                continue

            # 遍历当前节点的邻居节点
            for neighbor in self.G.neighbors(current_node):
                # 跳数加 1
                new_distance = distances[current_node] + 1

                # 如果新的距离小于已记录的距离，更新距离和前驱节点
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    predecessors[neighbor] = current_node
                    heapq.heappush(priority_queue, (new_distance, neighbor))

        # 构建最短路径
        path = []
        at = destination
        while at is not None:
            path.append(at)
            at = predecessors[at]

        # 反转路径，使其从源节点到目标节点
        path.reverse()

        # 如果路径的起点不是源节点，说明不存在路径
        if path and path[0] == source:
            return path
        else:
            return []


    def find_loss_optimal_route(self, source, destination, num_computing_nodes, min_computing_power, min_bandwidth):
        """
        找出丢包率最小的路径
        :param source: 源节点 ID
        :param destination: 目的节点 ID
        :param num_computing_nodes: 计算节点数
        :param min_computing_power: 计算节点最低算力要求
        :param min_bandwidth: 链路最低带宽要求
        :return: 丢包率最小化路径和选中的计算节点，如果不存在满足条件的路径则返回空列表
        """
        self.apply_constraints(num_computing_nodes, min_computing_power, min_bandwidth)

        # 检查源节点和目标节点是否存在
        if source not in self.G.nodes() or destination not in self.G.nodes():
            return [], []

        # 获取所有可能的路径
        all_paths = list(nx.all_simple_paths(self.G, source, destination))

        valid_paths = []
        for path in all_paths:
            # 检查中间节点数量是否满足计算节点要求
            intermediate_nodes = path[1:-1]
            if len(intermediate_nodes) >= num_computing_nodes:
                valid_paths.append(path)

        if not valid_paths:
            return [], []

        # 计算每条有效路径的丢包率
        loss_per_path = []
        for path in valid_paths:
            total_loss = self.calculate_total_loss(path)
            loss_per_path.append(total_loss)

        # 找到丢包率最小的路径索引
        min_loss_index = loss_per_path.index(min(loss_per_path))
        best_path = valid_paths[min_loss_index]

        # 选取中间节点作为计算节点
        intermediate_nodes = best_path[1:-1]
        top_computing_nodes = intermediate_nodes[:num_computing_nodes]

        return best_path, top_computing_nodes

    def calculate_total_loss(self, path):
        """
        计算路径的总丢包率
        :param path: 路径
        :return: 总丢包率
        """
        total_loss = 0
        for i in range(len(path) - 1):
            # 累加边的丢包率
            total_loss += self.G[path[i]][path[i + 1]]['loss']
        return total_loss

    def get_path_varible(self, path,computing_nodes):
        varible=[]
        time=round(self.calculate_total_delay(path,100,computing_nodes),2)
        cost=self.calculate_total_cost(path,computing_nodes)
        loss=self.calculate_total_loss(path)
        utl=self.calculate_total_utilization(path,computing_nodes)
        varible.append(time)
        varible.append(cost)
        varible.append(loss)
        varible.append(utl)
        return varible



# 示例用法
if __name__ == "__main__":
    # 从配置文件中读取网络信息

    planner = NetworkPlanner.from_config_file('../api/testtopo.json','../api/output.json')
    print(planner.find_time_optimal_route)
    path1,compute_nodes= planner.find_time_optimal_route(
        16,
        18,
        2,
        10,  # 要求计算节点算力≥800
        10  # 要求链路带宽≥10
    )
    path2, compute_nodes2 = planner.find_cost_optimal_route(
        16,
        18,
        2,
        10,  # 要求计算节点算力≥800
        10  # 要求链路带宽≥10
    )
    path3, compute_nodes3 = planner.find_loss_optimal_route(
        16,
        18,
        2,
        10,  # 要求计算节点算力≥800
        10  # 要求链路带宽≥10
    )
    path4, compute_nodes4 = planner.find_utl_optimal_route(
        16,
        18,
        2,
        10,  # 要求计算节点算力≥800
        10  # 要求链路带宽≥10
    )
    # list的前四个参数分别代表 时延，成本，丢包率，利用率
    list=planner.get_path_varible(path1,2)
    list2=planner.get_path_varible(path2,2)
    list3=planner.get_path_varible(path3,2)
    list4=planner.get_path_varible(path4,2)
    print("成本最小化路由策略成本为："+str(list2[1]))
    print("策略2成本为："+str(list[1]))
    print("策略3成本为："+str(list3[1]))
    print("策略4成本为："+str(list4[1]))

    # planner.visualize(path,compute_nodes)


