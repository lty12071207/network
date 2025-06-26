import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import json
import time


class cNetworkPlanner:
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
        self.fig, self.ax = None, None  # 初始化 figure 和 axes 对象
        self.node_pos = None  # 初始化节点位置

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

        for node in confignode:
            node_id = int(node["id"])
            computing_power = node["computing"]["cpu_power"]
            gpu_power = node["computing"]["gpu_power"]
            capacity = node["storage"]["capacity"]
            performance = node["storage"]["performance"]
            cost = node["cost"]
            uti = node["computing"]["gpu_Utilization"]
            # 从配置文件读取节点信息并添加到图中
            planner.G.add_node(node_id, computing_power=gpu_power, id=node_id, cost=cost, gpu_power=gpu_power,
                               capacity=capacity, performance=performance, uti=uti)

            # 添加边
        for edge in configlink:
            u = int(edge["src"])
            v = int(edge["dest"])
            propagation_delay = edge["delay"]
            cost = int(edge["weight"])
            bandwith = edge['bw']
            loss = edge['lost']
            luti = edge['LinkUtilization']
            # 从配置文件读取边信息并添加到图中
            planner.G.add_edge(u, v, propagation_delay=propagation_delay, cost=cost, bandwidth=bandwith, loss=loss,
                               luti=luti)
        # 计算并保存节点位置
        planner.node_pos = nx.spring_layout(planner.G, k=0.5, iterations=50)

        return planner


    def visualize(self, path=None, iteration=None,delay=0.5):
        """
        可视化网络拓扑图
        :param path: 可选参数，要高亮显示的路径
        :param iteration: 可选参数，当前迭代轮数
        """
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(12, 8))  # 创建 figure 和 axes 对象
        else:
            self.ax.clear()  # 清除之前的绘图内容

        pos = self.node_pos  # 使用预先计算好的节点位置

        # 绘制节点
        node_colors = []
        for node in self.G.nodes():
            if path and node == path[0]:
                node_colors.append('yellow')  # 起点为黄色
            elif path and node == path[-1]:
                node_colors.append('yellow')  # 终点为黄色
            elif path and node in path:
                node_colors.append('red')  # 路径节点为红色
            else:
                node_colors.append('lightblue')  # 其他节点为浅蓝色

        # 绘制节点
        nx.draw_networkx_nodes(self.G, pos, node_color=node_colors, node_size=800, edgecolors='black', ax=self.ax)
        nx.draw_networkx_labels(self.G, pos, font_size=10, font_family='sans-serif', ax=self.ax)

        # 绘制边
        if path:
            path_edges = list(zip(path, path[1:]))
            edge_colors = ['red' if (edge in path_edges) or (edge[::-1] in path_edges) else 'gray' for edge in self.G.edges()]
            edge_widths = [3 if (edge in path_edges) or (edge[::-1] in path_edges) else 1 for edge in self.G.edges()]
        else:
            edge_colors = 'gray'
            edge_widths = 1

        nx.draw_networkx_edges(self.G, pos, edge_color=edge_colors, width=edge_widths, ax=self.ax)

        # 添加节点标签（ID、GPU、Cost）
        labels = {node: "ID: {}\nGPU: {}\nCost: {}".format(node, data['gpu_power'], data['cost']) for node, data in self.G.nodes(data=True)}
        nx.draw_networkx_labels(self.G, pos, labels=labels, font_size=8, ax=self.ax)

        # 添加边标签（传播时延和带宽）
        edge_labels = {(u, v): "Delay: {}\nBand: {}".format(d['propagation_delay'], d['bandwidth']) for u, v, d in self.G.edges(data=True)}
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels, font_size=7, ax=self.ax)

        if iteration is not None:
            self.ax.set_title('Network Topology - Iteration {}'.format(iteration))
        else:
            self.ax.set_title('Network Topology')

        self.ax.axis('off')  # 隐藏坐标轴
        plt.tight_layout()  # 自动调整布局
        plt.pause(delay)  # 暂停一段时间，方便观察变化

    def find_cost_optimal_route(self, source, destination, num_computing_nodes, min_computing_power, min_bandwidth,
                                packet_size=100, num_ants=20, max_iter=2,
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
        self.apply_constraints(num_computing_nodes, min_computing_power, min_bandwidth)
        num_nodes = len(self.G.nodes())
        # 初始化信息素矩阵
        pheromone = np.ones((num_nodes, num_nodes))
        best_path = None
        best_cost = float('inf')

        # 提前计算所有边和节点的最大成本，用于归一化
        max_edge_cost = max(self.G[u][v]['cost'] for u, v in self.G.edges())
        max_node_cost = max(self.G.nodes[node]['cost'] for node in self.G.nodes())

        for iter_num in range(max_iter):
            all_ant_paths = []
            all_ant_cost = []
            print(f"iter_num迭代轮次{iter_num}")

            for _ in range(num_ants):
                print(f"iter_num迭代轮次{iter_num} ant<UNK>{_}")
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
                    print(f"Ant path: {path},Path cost: {total_cost}")  # 输出路径的成本
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_path = path
                        self.visualize(best_path)
                        print(f"Best path updated: {best_path}, Best cost: {best_cost}")  # 输出更新后的最优路径和成本

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
        sorted_nodes = sorted(intermediate_nodes, key=lambda node: self.get_node_computing_cost(node))
        # 选取成本最小的的前 num_computing_nodes 个节点
        top_three_computing_nodes = sorted_nodes[:num_computing_nodes]
        return best_path, top_three_computing_nodes
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
        sorted_nodes = sorted(intermediate_nodes, key=lambda node: self.get_node_computing_cost(node))
        # 选取成本最小的前三个节点
        top_three_computing_nodes = sorted_nodes[:num_computing_nodes]

        # 计算计算能力最强的前三个节点的计算时延
        for node in top_three_computing_nodes:
            computing_power_cost = self.get_node_computing_cost(node)

            total_cost += computing_power_cost

        return total_cost


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


    def _check_bandwidth(self, path, min_bandwidth):
        """
        检查路径上的所有边是否满足最小带宽约束
        :param path: 路径
        :param min_bandwidth: 最小带宽
        :return: 如果路径上所有边都满足带宽约束则返回 True，否则返回 False
        """
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if self.G.has_edge(u, v) and self.G[u][v]['bandwidth'] < min_bandwidth:
                return False
        return True

# 示例用法
if __name__ == "__main__":
    # 从配置文件中读取网络信息
    planner = cNetworkPlanner.from_config_file('../api/output.json', '../api/node_config.json')
    start_time = time.time();
    # 路径规划需求
    source = 0
    destination = 19
    packet_size = 100  # 要处理的数据包大小
    num_computing_nodes = 3  # 源、目的以及中间3个计算节点（共5个）

    # 使用蚁群算法寻找路径
    path,select_node= planner.find_cost_optimal_route(source, destination,  num_computing_nodes,10,2500)

    if path:
        min_cost = planner.calculate_total_cost(path,num_computing_nodes)
        print("找到的路径：", path)

        print("中间节点：", select_node)
        print("最小成本为", min_cost)
        end_time = time.time();
        print('时间差为' + str(end_time - start_time))

        # 可视化拓扑图，高亮显示路径
        planner.visualize(path=path,delay=10)
    else:
        print("不存在路径")





# 示例用法
