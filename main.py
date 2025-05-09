import networkx as nx
import matplotlib.pyplot as plt
import random


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

        # 生成节点（每个节点有可用内存资源）
        for node_id in range(num_nodes):
            memory = random.randint(100, 1000)  # 随机分配内存资源 100-1000 MB
            self.G.add_node(node_id, memory=memory, id=node_id)

        # 生成链路（确保连通性）
        if num_links < num_nodes - 1:
            raise ValueError("链路数量必须至少为节点数量减一以保持连通性")

        # 首先生成一个生成树确保连通性
        tree = nx.minimum_spanning_tree(nx.complete_graph(num_nodes))
        for u, v in tree.edges():
            bandwidth = random.randint(10, 100)  # 随机分配带宽 10-100 Mbps
            self.G.add_edge(u, v, bandwidth=bandwidth)

        # 添加剩余的链路
        remaining_links = num_links - (num_nodes - 1)
        possible_edges = list(nx.non_edges(self.G))
        random.shuffle(possible_edges)
        for u, v in possible_edges[:remaining_links]:
            bandwidth = random.randint(10, 100)
            self.G.add_edge(u, v, bandwidth=bandwidth)


    def visualize(self, path=None):
        """
        可化视网络拓扑图
        :param path: 可选参数，要高亮显示的路径
        """
        pos = nx.spring_layout(self.G)

        # 绘制节点
        node_colors = []
        for node in self.G.nodes():
            if path and node in path:
                node_colors.append('red')  # 路径中的节点用红色显示
            else:
                node_colors.append('blue')

        nx.draw(self.G, pos, with_labels=True, node_color=node_colors, node_size=700)

        # 绘制边
        edge_colors = []
        if path:
            path_edges = list(zip(path, path[1:]))
            for edge in self.G.edges():
                if (edge in path_edges) or (edge[::-1] in path_edges):
                    edge_colors.append('red')  # 路径中的边用红色显示
                else:
                    edge_colors.append('black')
            nx.draw_networkx_edges(self.G, pos, edge_color=edge_colors, width=2)
        else:
            nx.draw_networkx_edges(self.G, pos, edge_color='black', width=2)

        # 添加节点标签（内存资源）
        labels = {node: f"ID: {node}\nMem: {data['memory']} MB" for node, data in self.G.nodes(data=True)}
        nx.draw_networkx_labels(self.G, pos, labels=labels, font_size=8)

        # 添加边标签（带宽）
        edge_labels = {(u, v): f"{d['bandwidth']} Mbps" for u, v, d in self.G.edges(data=True)}
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
            raise ValueError(f"源节点 {source} 不存在")
        if destination not in self.G.nodes():
            raise ValueError(f"目的节点 {destination} 不存在")

        # 使用深度优先搜索获取所有路径
        all_routes = []
        visited = set()
        path = []

        def dfs(node):
            path.append(node)
            visited.add(node)
            if node == destination:
                all_routes.append(path.copy())
            else:
                for neighbor in self.G.neighbors(node):
                    if neighbor not in visited:
                        dfs(neighbor)
            path.pop()
            visited.remove(node)

        dfs(source)
        return all_routes

    def get_node_memory(self, node_id):
        """
        获取指定节点的内存大小
        :param node_id: 节点ID
        :return: 节点的内存大小（MB）
        """
        if node_id not in self.G.nodes():
            raise ValueError(f"节点 {node_id} 不存在")
        return self.G.nodes[node_id]['memory']  # 直接访问节点的内存属性

    def get_link_bandwidth(self, u, v):
        """
        获取指定两边之间的带宽大小
        :param u: 第一个节点ID
        :param v: 第二个节点ID
        :return: 两边之间的带宽大小（Mbps）
        """
        if u not in self.G.nodes() or v not in self.G.nodes():
            raise ValueError(f"节点 {u} 或 {v} 不存在")
        if self.G.has_edge(u, v):
            return self.G[u][v]['bandwidth']  # 直接访问边的带宽属性
        else:
            raise ValueError(f"节点 {u} 和 {v} 之间不存在链路")

# 示例用法
if __name__ == "__main__":
    # 生成网络拓扑（10个节点，15条链路）
    planner = NetworkPlanner(num_nodes=10, num_links=20)

    # 路径规划需求
    source = 0
    destination = 3
    min_memory = 700# 每个计算节点需要至少 200 MB 内存
    min_bandwidth = 10  # 相邻计算节点之间链路需要至少 30 Mbps 带宽
    num_computing_nodes = 3  # 源、目的以及中间3个计算节点（共5个）
    rlist=planner.get_all_routes(0,3)
    #print(rlist)
    select=[]
    for i in rlist:
        if len(i)-2>=num_computing_nodes: #路径除了起点终点必须大于等于计算节点数
            select.append(i)
    print(select)
    if len(select)==0:
        print("不存在路径")
    else:
        chaselect=[]
        for i in select:
            chalist=[]
            for j in range(1,len(i)-1):
                m=planner.get_node_memory(i[j])
                cha=m-min_memory
                chalist.append(cha)
            sorted_arr = sorted(chalist, reverse = True)
            #print(sorted_arr)
            chaselect.append(sorted_arr)
        resultofmemory=[]
        item=0
        for i in chaselect:
            biaozhi=0
            for j in range(0,num_computing_nodes):

                if i[j]<0:
                    biaozhi=1

            if biaozhi==0:
                resultofmemory.append(select[item])
            item+=1

        if len(resultofmemory)==0:
            maxlist=[]
            for i in chaselect:
                sum=0
                for j in range(0,num_computing_nodes):
                    sum+=i[j]
                maxlist.append(sum)
            max_value = max(maxlist)
            max_index = maxlist.index(max_value)
            resultofmemory.append(select[max_index])
        print(resultofmemory)
        #用带宽做约束条件
        bandlist=[]
        standard=[]
        for i in resultofmemory:
            sum=0
            for j in range(0,len(i)-1):
                b=planner.get_link_bandwidth(i[j],i[j+1])
                sum+=b
            bandlist.append(sum)
            standard.append(min_bandwidth*(len(i)-1))
        chalist=[]
        for i in range(0,len(bandlist)):
            item=bandlist[i]-standard[i]
            chalist.append(item)
        # 找到比0大的最小值
        positive_numbers = [x for x in chalist if x > 0]
        if positive_numbers:
            min_positive = min(positive_numbers)
            print(f"比0大的最小值是：{min_positive}")
            idx=chalist.index(min_positive)
        else:
            negative_numbers = [x for x in chalist if x < 0]
            if negative_numbers:
                max_negative = max(negative_numbers)
                print(f"比0小的最大值是：{max_negative}")
                idx = chalist.index(max_negative)
        path=resultofmemory[idx]
        # cnodeid=[]
        # for item in path:
        #     m=planner.get_node_memory(item)
        #     cha=m-min_memory
        #     zidian={"id":item,"cha":cha}
        #     cnodeid.append(zidian)
        # sorted_dict_list = sorted(cnodeid, key=lambda x: x['cha'])
        # tubian=0
        # for i in range(0,len(sorted_dict_list)-1):
        #     if sorted_dict_list[i]['cha']*sorted_dict_list[i+1]['cha']<=0:
        #         tubian=i
        # if i>0 and i+-num_computing_nodes










    # 寻找路径
    #path = planner.find_path_with_computing_nodes_count(source, destination, num_computing_nodes,min_memory, min_bandwidth)


    # 打印结果
    #print("找到的路径：", path)

    # 可视化拓扑图，高亮显示路径
    planner.visualize(path=path)