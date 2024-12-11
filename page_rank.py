import pickle
import networkx


def get_pagerank():

    with open("data/index_set/id_jumpUrls_dic.pkl", "rb") as tf:
        id_jumpUrls_dic = pickle.load(tf)
    print(id_jumpUrls_dic)
    with open("data/index_set/url_id_dic.pkl", "rb") as tf:
        url_id_dic = pickle.load(tf)
    print(url_id_dic)

    # 根据 url 之间的跳转关系构建了一个有向图
    graph = networkx.DiGraph()
    for i in range(0, len(id_jumpUrls_dic)):
        for j in id_jumpUrls_dic[i]:
            if j in url_id_dic.keys():
                # 构造从 i 到 j 的边
                graph.add_edge(i, url_id_dic[j])
    print('add edges end')
    
    # 计算 PageRank
    # 指向该网页的超链接越多，随机跳转到该网页的概率也就越高，该网页的PageRank值就越高
    # 字典，记录网页对应的PageRank
    id_pagerank_dic = networkx.pagerank(graph)
    print('pagerank result is ', id_pagerank_dic)
    # 保存所有网页的PageRank值
    with open("data/index_set/id_pagerank_dic.pkl", "wb") as tf:
        pickle.dump(id_pagerank_dic, tf)
    print("write pagerank dic finish")

if __name__ == '__main__':
    get_pagerank()