#!/usr/bin/env python
# coding: utf-8

# ## Code to generate input for the Model

# In[12]:


import pandas as pd


# In[13]:


prefix = "SNF_iminf"
# loading the similarity matrices
drug_df = pd.read_csv('../../data/processed_drug_simmat_SNF19.csv', index_col=0)
disease_df = pd.read_csv('../../data/processed_disease_simmat_SNF14.csv', index_col=0)
labels = pd.read_csv('../../data/labels_SNFd14_SNFd19_iminf.csv', index_col=0)


# In[14]:


# loading the PPI
df = pd.read_csv('../data/HumanNet-PI.tsv', sep='\t')
network = []
for i in range(len(df)):
    network.append(df.index[i])
# print(network)
net = pd.DataFrame(network)


# In[15]:


import networkx as nx
# creating the graph
G = nx.from_pandas_edgelist(net, source=0, target=1, create_using=nx.Graph())
valid_nodes = list(G.nodes)

#network proximity calcuation based on shortest path(Dijkstras)
def calculate_network_proximity1(G, node1, node2, k ):
    try:
        shortest_path = nx.shortest_path_length(G, source=node1, target=node2)
        network_proximity = 1 / shortest_path
        return network_proximity
    except:
        return 0
        print("exception")

# jaccard similarity
def jaccard_similarity(G, node1, node2):
    neighbors1 = set(G.neighbors(node1))
    neighbors2 = set(G.neighbors(node2))
    if len(neighbors1) == 0 or len(neighbors2) == 0:
        return 0
    return len(neighbors1.intersection(neighbors2)) / len(neighbors1.union(neighbors2))

# Cosine similarity
def cosine_similarity(G, node1, node2):
    neighbors1 = set(G.neighbors(node1))
    neighbors2 = set(G.neighbors(node2))
    if len(neighbors1) == 0 or len(neighbors2) == 0:
        return 0
    intersection = len(neighbors1.intersection(neighbors2))
    norm1 = len(neighbors1)
    norm2 = len(neighbors2)
    return intersection / ((norm1 * norm2) ** 0.5)

# Adamic-Adar index
def adamic_adar_index(G, node1, node2):
    neighbors1 = set(G.neighbors(node1))
    neighbors2 = set(G.neighbors(node2))
    common_neighbors = neighbors1.intersection(neighbors2)
    score = 0
    for neighbor in common_neighbors:
        degree = G.degree(neighbor)
        if degree > 1:
            score += 1 / (degree ** 0.5)
    return score

# Preferential attachment
def preferential_attachment(G, node1, node2):
    degree1 = G.degree(node1)
    degree2 = G.degree(node2)
    return degree1 * degree2


# In[ ]:





# In[16]:


import json
# loading the known drug-gene interactions
with open('../data/gene_dict_drug.json', 'r') as f:
    gene_dict_drug = json.load(f)
with open('../data/gene_dict_disease.json', 'r') as f:
    gene_dict_disease = json.load(f)


# In[17]:


drug_names = drug_df.index
disease_names = disease_df.index


# In[18]:


new_gene_dict_drug = {}
for key, value in gene_dict_drug.items():
    if(key in drug_names):
        value_int_list = [int(x) for x in value]
        intersection = list(set(value_int_list).intersection(valid_nodes))
        new_gene_dict_drug[key] = intersection


# In[19]:


new_gene_dict_disease = {}
for key, value in gene_dict_disease.items():
    if(key in disease_names):
        value_int_list = [int(x) for x in value]
        intersection = list(set(value_int_list).intersection(valid_nodes))
        new_gene_dict_disease[key] = intersection


# In[20]:


from tqdm import tqdm


# In[ ]:





# In[21]:


labels


# In[22]:


from multiprocessing import Pool

def calculate_proximities(args):
    index, row, new_gene_dict_drug = args
    drug_genes = new_gene_dict_drug[row[1]]
    disease_genes = new_gene_dict_disease[row[0]]

    network_proximities = []
    for drug_gene in drug_genes:
        for disease_gene in disease_genes:
            network_proximities.append(calculate_network_proximity1(G, drug_gene, disease_gene,10000))

    mean = sum(network_proximities) / len(network_proximities)
    median = sorted(network_proximities)[len(network_proximities) // 2]
    minimum = min(network_proximities)
    maximum = max(network_proximities)

    return (mean, median, minimum, maximum)

proximities = []
with Pool() as p:
    for result in tqdm(p.imap(calculate_proximities, [(index, row, new_gene_dict_drug) for index, row in labels.iterrows()]), total=len(labels)):
        proximities.append(result)


# In[23]:


proximities


# In[28]:


import numpy as np
dd_list = []
label_list = []
dds_list = []
proximity_dict = {}
for index, row in labels.iterrows():
    drug_name = row[1]
    disease_name = row[0]
    label = row[2]
    drug_vector = list(drug_df.loc[drug_name,:].values)
    disease_vector = list(disease_df.loc[disease_name,:].values)
    proximity_vector = list(proximities[index])
    proximity_dict[(drug_name,disease_name)]  = proximity_vector
    label_list.append(label)
    dd_vector = drug_vector + disease_vector + [label]
    dds_vector = drug_vector + disease_vector + proximity_vector + [label]
    # print(len(dd_vector), len(dds_vector))
    # break
    dd_vector = np.array(dd_vector)
    dds_vector = np.array(dds_vector)
    dd_list.append(dd_vector)
    dds_list.append(dds_vector)

proximity_dict_str_keys = {str(k): v for k, v in proximity_dict.items()}

with open('../../data/prox_dict_'+ prefix +'.json', 'w') as f:
    # Write the dictionary with string keys to the file as JSON
    json.dump(json.dumps(proximity_dict_str_keys, default=lambda x: list(x)), f)


# In[ ]:





# In[29]:


import numpy as np

# Save the list of arrays
np.save('../../data/dds_'+prefix + '.npy', dds_list)
# np.save('dd10000_o.npy', dd_list)


# In[ ]:
