import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import json
import time


class NetworkPlanner:
    def __init__(self, num_nodes, num_links, required_compute_nodes=2, min_computing_power=0, min_bandwidth=0):
        """
        初始化网络规划器
        :param num_nodes: 节点数量
        :param num_links: 链路数量
        :param min_computing_power: 节点最小计算能力约束
        :param min_bandwidth: 链路最小带宽约束
        """
        self.num_nodes = num_nodes
        self.num_links = num_links
        self.required_compute_nodes = required_compute_nodes
        self.min_computing_power = min_computing_power
        self.min_bandwidth = min_bandwidth
        self.G = nx.Graph()  # 创建一个空的无向图




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
    def from_config_file(cls, topofile, nodefile):
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
            computing_power = node["computing"]["gpu_power"]
            capacity = node["storage"]["capacity"]
            performance = node["storage"]["performance"]
            cost = int(node["cost"])
            uti = int(node["computing"]["gpu_Utilization"])
            # 从配置文件读取节点信息并添加到图中
            planner.G.add_node(node_id, computing_power=computing_power, id=node_id, cost=cost,
                               capacity=capacity, performance=performance, uti=uti)

        # 添加边
        for edge in configlink:
            u = int(edge["src"])
            v = int(edge["dest"])
            propagation_delay = float(edge["delay"])
            cost = int(edge["weight"])
            bandwidth = int(edge['bw'])

            loss = int(edge['lost'])
            luti = int(edge['LinkUtilization'])
            # 从配置文件读取边信息并添加到图中
            planner.G.add_edge(u, v, propagation_delay=propagation_delay, cost=cost, bandwidth=bandwidth, loss=loss,
                               luti=luti)


        return planner

    def visualize(self, path=None):
        """
        可视化网络拓扑图
        :param path: 可选参数，要高亮显示的路径
        """
        # 使用弹簧布局算法确定节点位置
        pos = nx.spring_layout(self.G)

        # 绘制节点
        node_colors = []
        for node in self.G.nodes():
            if path and node in path:
                # 若节点在指定路径中，标记为红色
                node_colors.append('red')
            else:
                # 否则标记为蓝色
                node_colors.append('blue')

        # 绘制图的节点
        nx.draw(self.G, pos, with_labels=True, node_color=node_colors, node_size=700)

        # 绘制边
        edge_colors = []
        if path:
            # 获取路径中的边
            path_edges = list(zip(path, path[1:]))
            for edge in self.G.edges():
                if (edge in path_edges) or (edge[::-1] in path_edges):
                    # 若边在指定路径中，标记为红色
                    edge_colors.append('red')
                else:
                    # 否则标记为黑色
                    edge_colors.append('black')
            # 绘制带颜色的边
            nx.draw_networkx_edges(self.G, pos, edge_color=edge_colors, width=2)
        else:
            # 若未指定路径，绘制黑色边
            nx.draw_networkx_edges(self.G, pos, edge_color='black', width=2)

        # 添加节点标签（计算能力）
        labels = {node: f"ID: {node}\nComp: {data['computing_power']}" for node, data in self.G.nodes(data=True)}
        # 绘制节点标签
        nx.draw_networkx_labels(self.G, pos, labels=labels, font_size=8)

        # 添加边标签（传播时延和带宽）
        edge_labels = {(u, v): f"Delay: {d['propagation_delay']}\nBand: {d['bandwidth']}" for u, v, d in
                       self.G.edges(data=True)}
        # 绘制边标签
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels, font_size=7)

        plt.title('Network Topology')
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
            raise ValueError("源节点 %s 不存在" % source)
        if destination not in self.G.nodes():
            # 若目的节点不存在，抛出异常
            raise ValueError("目的节点 %s 不存在" % destination)

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

    def get_node_uti(self, node_id):
        """
        获取指定节点的利用率
        :param node_id: 节点ID
        :return: 节点的利用率
        """
        if node_id not in self.G.nodes():
            # 若节点不存在，抛出异常
            raise ValueError("节点 %s 不存在" % node_id)
        # 直接访问节点的计算能力属性
        return self.G.nodes[node_id]['uti']

    def get_link_propagation_luti(self, u, v):
        """
        获取指定两边之间的传播链路成本
        :param u: 第一个节点ID
        :param v: 第二个节点ID
        :return: 两边之间的传播链路成本
        """
        if u not in self.G.nodes() or v not in self.G.nodes():
            # 若任一节点不存在，抛出异常
            raise ValueError("节点 %s 或 %s 不存在" % (u, v))
        if self.G.has_edge(u, v):
            # 若边存在，直接访问边的传播时延属性
            return self.G[u][v]['luti']
        else:
            # 若边不存在，抛出异常
            raise ValueError("节点 %s 和 %s 之间不存在链路" % (u, v))

    def ant_colony_optimization(self, source, destination, num_computing_nodes,min_computing_power,min_bandwidth,packet_size=100, num_ants=20, max_iter=50,
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
                        # 计算启发式信息：利用率的倒数
                        node_uti = self.G.nodes[neighbor]['uti']
                        link_luti = self.G[current][neighbor]['luti']
                        total_utilization = node_uti + link_luti
                        heuristic = 1 / total_utilization
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
                for path, utilization in zip(all_ant_paths, all_ant_utilization):
                    delta_pheromone = 1 / utilization
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i + 1]
                        pheromone[u][v] += delta_pheromone
                        pheromone[v][u] += delta_pheromone
        intermediate_nodes = best_path[1:-1]
        sorted_nodes = sorted(intermediate_nodes, key=lambda node: planner.G.nodes[node]['uti'])
        # 选取 uti 最小的的前三个节点
        top_three_computing_nodes = sorted_nodes[:3]
        return best_path,top_three_computing_nodes

    def calculate_total_utilization(self, path, num_computing_nodes):
        """
        计算路径的总利用率
        :param path: 路径
        :param num_computing_nodes: 计算节点数
        :return: 总利用率
        """
        total_utilization = 0

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


# 示例用法
if __name__ == "__main__":
    # 从配置文件中读取网络信息
    planner = NetworkPlanner.from_config_file('../api/topo.json', '../api/node_config.json')
    start_time = time.time();
    # 路径规划需求
    source = 0
    destination = 18
    packet_size = 100  # 要处理的数据包大小
    num_computing_nodes = 3  # 源、目的以及中间3个计算节点（共5个）

    # 使用蚁群算法寻找路径
    path,select_node = planner.ant_colony_optimization(source, destination,  num_computing_nodes,10,10)

    if path:
        min_utilization = planner.calculate_total_utilization(path, num_computing_nodes)
        print("找到的路径：", path)

        print("中间节点：", select_node)
        print("最小利用率为", min_utilization)
        end_time = time.time();
        print('时间差为' + str(end_time - start_time))

        # 可视化拓扑图，高亮显示路径
        planner.visualize(path=path)
    else:
        print("不存在路径")
