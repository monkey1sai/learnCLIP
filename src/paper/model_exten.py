import networkx as nx

# 創建指稱圖
G = nx.DiGraph()

# 添加節點和邊
G.add_node("desc1", images={"img1", "img2", "img3"})
G.add_node("desc2", images={"img2", "img3", "img4"})
G.add_edge("desc1", "desc2")  # desc1 是 desc2 的上位語義

# 可視化指稱圖
print("Nodes:", G.nodes(data=True))
print("Edges:", G.edges())
