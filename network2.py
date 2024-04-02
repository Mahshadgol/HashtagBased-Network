import pandas as pd
import json
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import lognorm
import igraph as ig
import cairocffi as cairo
import random
import math


df = pd.read_csv('/home/taha/Downloads/NetworkScience/dataset/20230527_UkraineCombinedTweetsDeduped.csv.gzip', compression='gzip')
# df = pd.read_csv('/home/taha/Downloads/NetworkScience/dataset/UkraineWar/UkraineWar/UkraineCombinedTweetsDeduped_MAR31.csv.gzip', compression='gzip')

df.columns
# len(df)
# df[['userid','hashtags']]
hashtags = df['hashtags']
# hashtags[0].replace("'", "\"")
# temp = json.loads(hashtags[0].replace("'", "\""))
# l = []
# for item in temp:
#     l.append(item['text'])
# print(l)

# hashtags.apply(lambda x: json.loads(x.replace("'", "\"")))
extracted = hashtags.apply(lambda row: [item['text'] for item in json.loads(row.replace("'", "\""))])
extracted_filtered = extracted[extracted.apply(lambda x: len(x) != 0)]

len(extracted_filtered)

network = {}
for row in extracted_filtered:
    for pair in itertools.product(row, row):
        if pair[0]!=pair[1] and not(pair[::-1] in network) and not pair[0].isdigit() and not pair[1].isdigit():
            network.setdefault(pair,0)
            network[pair] += 1

network_df = pd.DataFrame.from_dict(network, orient="index")
network_df.index.name = 'pair'
network_df.columns = ["weight"]
network_df = network_df.reset_index()
network_df.sort_values(by="weight",inplace=True, ascending=False)


up_weighted = []
for edge in network:
    if(network[edge]) > 2:
        up_weighted.append((edge[0],edge[1],network[edge]))

G = nx.Graph()
G.add_weighted_edges_from(up_weighted)


print(len(G.nodes()))
print(len(G.edges()))

path = '/home/taha/Downloads/NetworkScience/twitterGraphEdgeList.csv'
nx.write_weighted_edgelist(G, path, delimiter=",")


network_df[['tag1','tag2']] = network_df['pair'].apply(pd.Series)
df1 = network_df[['tag1','weight']].rename(columns={'tag1': 'tag', 'weight': 'weight'})
df2 = network_df[['tag2','weight']].rename(columns={'tag2': 'tag', 'weight': 'weight'})
df3 = pd.concat([df1, df2], ignore_index=True)
df4 = df3.groupby('tag')['weight'].sum()#.sort_values(by="weight",inplace=True, ascending=False)

bins = np.logspace(np.log10(df4.min()), np.log10(df4.max()), 20)
fig, ax = plt.subplots()
ax.hist(df4, bins=bins, histtype='step', density=True)

x = np.logspace(np.log10(df4.min()), np.log10(df4.max()), 100)
shape, loc, scale = lognorm.fit(df4)
# mean = np.log(scale)
# sigma = shape
# compute the PDF of the log-normal distribution
pdf = lognorm.pdf(x, shape, scale=scale)

# plot the PDF
ax.plot(x, pdf)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Degree', fontsize=14)
ax.set_ylabel('Density', fontsize=14)
plt.show()

# mean = 2.62
# sigma = 1.61


path = '/home/taha/Downloads/NetworkScience/twitterGraphEdgeList.csv'
friendship_nw = nx.read_weighted_edgelist(path,delimiter=",")
g = ig.Graph.from_networkx(friendship_nw)

layout = g.layout_fruchterman_reingold()
g.vs["label"] = g.vs["_nx_name"]

visual_style = {}
visual_style["vertex_size"] = 5
# visual_style["vertex_color"] =  random.sample(colors,len(g.vs()))
# visual_style["vertex_color"] =  random.sample(colors,len(g.vs()))
visual_style["vertex_label"] = g.vs["label"]
visual_style["vertex_label_color"] = "black"
visual_style["vertex_label_size"] = [0.1*degree for degree in g.degree()]
visual_style["edge_width"] = [0.01 * int(weight) for weight in g.es["weight"]]
visual_style["bbox"] = (500, 500)
visual_style["margin"] = 20
visual_style["layout"] = g.layout_fruchterman_reingold()


ig.plot(g, 'graph.png', **visual_style)

# ig.plot(g, 'graph.png', 
#         layout=layout, 
#         vertex_size = 10, 
#         vertex_label = g.vs["label"],
#         vertex_label_size = [0.1*degree for degree in g.degree()],
#         edge_width = [0.01 * int(weight) for weight in g.es["weight"]])

edges = pd.read_csv(path, names=['Source','Target','Weight'])
friendship_nw_prop = nx.from_pandas_edgelist(edges, 'Source', 'Target', ['Weight'])

# nodes = pd.read_csv('nodes.csv', header=0,delim_whitespace=True)
# nodes = nodes.set_index('Id').to_dict('index').items()

# friendship_nw_prop.add_nodes_from(nodes)
# print(friendship_nw_prop.nodes(data=True))
# print(friendship_nw_prop.edges(data=True))

g_prop = ig.Graph.from_networkx(friendship_nw_prop)
g_prop = g_prop.clusters().giant()

nodes = g_prop.vs()
edges = g_prop.es()

visual_style = {}
visual_style["vertex_size"] = 5
visual_style["vertex_color"] =  ['red' for node in g_prop.vs()]
visual_style["vertex_label"] = g_prop.vs["label"]
visual_style["vertex_label_color"] = "black"
visual_style["vertex_label_size"] = [0.1*degree for degree in g_prop.degree()]
visual_style["edge_width"] = [0.05 * int(weight) for weight in g_prop.es["Weight"]]
visual_style["edge_color"] = ["#bdbdbd" for edge in edges]
visual_style["bbox"] = (500, 500)
visual_style["margin"] = 20
visual_style["layout"] = g_prop.layout_fruchterman_reingold()


print("Graph order:", len(nodes))

# GRAPH SIZE = NO OF EDGES

print("Graph size:", len(edges))

# DENSITY - HOW CONNECTED ARE THE NODES? NO OF EDGES/NO OF POSSIBLE EDGES

print("Number of possible edges (N*(N-1)/2):", int(len(nodes)*(len(nodes)-1)/2))
print("Graph density:", g_prop.density())

# get the largest connected component
len(g_prop.components())
for component in g_prop.components():
  print(len(component))


# visual_style["layout"] = g_prop.layout_fruchterman_reingold()
# visual_style["layout"] = g_prop.layout_kamada_kawai()
# visual_style["layout"] = g_prop.layout_reingold_tilford()
ig.plot(g_prop, 'graph2.png', **visual_style)




# DIAMETER - HOW FAR ARE THE TWO MOST DISTANT NODES

print("Network diameter:", g_prop.diameter(directed=False))
d = g_prop.get_diameter()
diameter_path = []
for i in range(0, g_prop.diameter()):
  diameter_path.append((d[i], d[i+1]))
diameter_edges = g_prop.get_eids(pairs=diameter_path, directed=False)


#COLOR THE DIAMETER PATH
visual_style["vertex_color"] = ["red" if node.index in d else "white" for node in nodes]
# visual_style["edge_width"] = [0.1 * int(weight) for weight in g.es["weight"]]
visual_style["edge_color"] = ["red" if edge.index in diameter_edges else "#ebebeb" for edge in edges]

# AVERAGE PATH LENGTH - HOW CLOSE ARE THE NODES TO EACH OTHER ON AVERAGE
print("Average path length:", g_prop.average_path_length(directed=False))

ig.plot(g_prop, 'diameter.png', **visual_style)




print("Maximal cliques in graph")
# maximal_cliques = g_prop.maximal_cliques()
largest_clique = g_prop.largest_cliques()[0]
clique_path = []
for i in range(len(largest_clique)):
  clique_path.append((largest_clique[i], largest_clique[i+1]))
clique_edges = g_prop.get_eids(pairs=clique_path, directed=False)

visual_style["vertex_color"] = ["red" if node.index in largest_clique else "grey" for node in nodes]
visual_style["edge_width"] = [0.05 * int(weight) for weight in g_prop.es["weight"]]
visual_style["edge_color"] = ["red" if edge.index in clique_edges else "#ebebeb" for edge in edges]
ig.plot(g_prop, 'largest_clique.png', **visual_style)

clique_tags = [g_prop.vs[index]['_nx_name'] for index in clique_path]
flat_list = list(set([item for sublist in clique_tags for item in sublist]))



# colors = ig.drawing.colors.known_colors
# colors = list(colors.keys())
# communities = g_prop.community_optimal_modularity()
# community_colors = random.sample(colors,len(communities))
# node_colors = {}
# counter = 0

# print("Communities in the network:")

# for community in communities:
#     print("  ",[nodes[member]["Label"] for member in community])
#     for member in community:
#         node_colors[member] = community_colors[counter]
   
#     counter += 1

# visual_style["vertex_color"] = [node_colors[node.index] for node in nodes]
# visual_style["vertex_label"] = g.vs["label"]

# ig.plot(g_prop, 'community.png', **visual_style)


# GLOBAL OR LOCAL CLUSTERING COEFFICIENT - GENERAL INDICATION OF THE GRAPH'S TENDENCY TO BE ORGANISED INTO CLUSTERS

# GLOBAL CC - NUMBER OF CLOSED TRIPLETS/NUMBER OF POSSIBLE TRIPLETS

print("Global clustering coefficient", g_prop.transitivity_undirected())

# LOCAL CC - ARE THE NEIGHBOURS OF THE NODES ALSO CONNECTED?

print("Local clustering components:")
local_ccs = g_prop.transitivity_local_undirected()
# sum_cc = 0
# for local_cc in local_ccs:
#     if not math.isnan(local_cc):
#         sum_cc += local_cc

# for node in nodes:
#         print("   Local clustering coefficient of node", node,":",local_ccs[node.index])

# AVERAGE CC

# print("Average clustering component", sum_cc/len(g_prop.vs()))
# count, hist, _ = ax.hist(local_ccs, bins = 30)

fig, ax = plt.subplots()
ax.hist(local_ccs, histtype='step', bins = 30)
ax.set_xlabel('Transitivity', fontsize=14)
ax.set_ylabel('Frequency', fontsize=14)
# plt.plot(hist[1:],count)
plt.show()


g_prop.assortativity_degree()
correlation_matrix = g_prop.assortativity(attribute='degree', directed=False, types1=None, types2=None)
g_prop.assortativity_degree(directed=False)
fig, ax = plt.subplots()
ax.imshow(correlation_matrix, cmap='RdBu')
plt.show()
# # nodes.degree()
# degrees = g_prop.degree()
# # compute the degree correlation matrix
# corr_matrix = np.corrcoef(degrees)
# print(corr_matrix)

madularity_list = [152,61,23,38,97,76,27,25,18,158]
madularity_list.sort()
plt.figure()
plt.scatter(range(len(madularity_list)), madularity_list)
plt.xlabel('Community Class', fontsize=14)
plt.ylabel('Number of Nodes', fontsize=14)
plt.show()




# BETWEENESS - BEING A BRIDGE BETWEEN NODES; BETWEENNES CENTRALITY: NUMBER OF SHORTEST PATHS THROUGH A NODE


print("Betweenness centrality:"),
betweenness = g_prop.betweenness(directed=False) 
# for bc in betweenness:
#     print("   Betweeness centrality of", nodes[betweenness.index(bc)]["Label"],":",bc)
plt.hist(betweenness,bins=25, histtype='step')
plt.show()
    
# CLOSENESS - BEING IN THE MIDDLE OF A NETWORK

print("Closeness centrality:"),
closeness = g_prop.closeness() 
# for node in nodes:
#         print("   Closeness centrality of", node["Label"],":",closeness[node.index])
plt.hist(closeness,bins=25)
plt.xlabel('Closeness', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.show()

# PRESTIGE(EIGENVECTOR CENTRALITY) - BEING CLOSE TO WELL CONNECTED NODES

eigenvector_centralities = g_prop.eigenvector_centrality() 
print("Eigenvector centrality:"),
eigenvector_centralities = g_prop.eigenvector_centrality() 
for node in nodes:
    print("   Eigenvector centrality of", node["Label"],":",eigenvector_centralities[node.index])

visual_style["vertex_label"] = g_prop.vs["Label"]
visual_style["vertex_size"] = [50*ec for ec in eigenvector_centralities]
visual_style["vertex_color"] = ["maroon" if bc>0.0 else "white" for bc in betweenness]
visual_style["edge_color"] = "grey"

plt.hist(eigenvector_centralities,bins=25)
plt.xlabel('Eigenvalue Centrality', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.show()


ig.plot(g_prop, **visual_style)
