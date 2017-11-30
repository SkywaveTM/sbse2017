"""
Writer: Donguk Jung, Yongjun Shin
Date: 2017-11-27
"""

import random
import numpy as np
import numpy.linalg as lin

def csv_to_list(file_name):
    f = open(file_name, 'r')
    graph_list = []
    for line in f.readlines():
        list_line = line.strip().split(',')
        cur_line = []
        for i in range(len(list_line)):
            cur_line.append(int(list_line[i]))
        graph_list.append(cur_line)
    return graph_list


def print_cluster_result_list(cluster):
    for i in range(len(cluster)):
        print(cluster[i], end=' ')
    print()


def print_graph(graph):
    length = len(graph)
    for i in range(length):
        for j in range(length):
            print(graph[i][j], end=' ')
        print()


def clustering_random(graph):
    num_of_module = len(graph[0])
    cluster = []
    for i in range(num_of_module):
        cluster.append(random.randrange(num_of_module))

    cluster = normalize(cluster)
    return cluster


def swap(target_list, i, j):
    temp = target_list[i]
    target_list[i] = target_list[j]
    target_list[j] = temp


def clustering_Spectral_Bisection(graph):
    n = len(graph)
    cluster = []
    for i in range(n): cluster.append(0)
    Lp = []
    for i in range(n): Lp.append([x for x in range(n)])
    for i in range(n):
        for j in range(n):
            if(i==j):
                sum = 0
                for k in range(n): sum += graph[i][k]
                Lp[i][j] = sum
            else:
                if(graph[i][j]): Lp[i][j] = -1
                else: Lp[i][j] = 0

    exEin = np.array(Lp)
    eigval = lin.eig(exEin)[0]
    eigvec = lin.eig(exEin)[1]
    clmvec = []
    for i in range(n):
        temp = []
        for j in range(n): temp.append(eigvec[j][i])
        clmvec.append(temp)
    valvec = []
    for i in range(n): valvec.append((eigval[i],clmvec[i]))
    def getfirst(tuple):
        return tuple[0]
    svv = sorted(valvec, key = getfirst)
    vec2 = svv[1][1]
    for i in range(n):
        if(vec2[i]<0): cluster[i] = 1

    return cluster


def one_step_more(graph, cluster0):
    n = len(graph)
    cluster = normalize(cluster0)
#    print(cluster)

    cno = max(cluster) + 1 # number of clusters

    cluster_valueview = [0 for i in range(cno)]
    for i in range(n): cluster_valueview[cluster[i]] += 1
    #print(cluster_valueview)
    cvmax = max(cluster_valueview)
    candidate = []
    for i in range(cno):
        if(cluster_valueview[i] == cvmax): candidate.append(i)
    random.shuffle(candidate)
    S = candidate[0]            # select 1 random cluster among the max size one
    #print('cno is')
    #print(cno)
    #print(S)
    indexlist = []
    for i in range(n):
        if(cluster[i] == S): indexlist.append(i)
    #print(indexlist)

    subgraph = [[l[j] for j in indexlist] for l in [graph[i] for i in indexlist]]    # make subgraph of that cluster


    # run function for that subgraph
    # get categorization
    new_sub_cluster = normalize(clustering_Spectral_Bisection(subgraph))  # Expect categorization size cno and only consist of 0 and 1
    #new_sub_cluster = normalize(clustering_Kernighan_Lin(subgraph))                        ############################# SWITCH 1 #########################
    new_cluster = cluster.copy()

    for i in range(len(indexlist)):
        if(new_sub_cluster[i] == 0): new_cluster[indexlist[i]] = cno
 #   print(new_cluster)
    return normalize(new_cluster)

    # marge categorization onto prev result

    return


def clustering_Spectral_Bisection_nver(graph):
    cluster = normalize(clustering_Spectral_Bisection(graph))
#    cluster = normalize(clustering_Kernighan_Lin(graph))                                   ############################# SWITCH 2 ############################
    mq = MQ(cluster, graph)
    flag = 0
    while True:
 #       print('loop ongoing...')
 #       print(cluster)
 #       print('mq ongoing')
 #       print(mq)
        new_cluster = one_step_more(graph, cluster)
        nmq = MQ(new_cluster, graph)
        if(mq > nmq): break
        if(mq == nmq):
            if(flag > 3): break
            flag += 1
        cluster = new_cluster.copy()
        mq = nmq
#    print('final result')
#    print(cluster)
#    print('final mq')
#    print(mq)
    return cluster


def one_step_more2(graph, cluster0):
    n = len(graph)
    cluster = normalize(cluster0)
#    print(cluster)

    cno = max(cluster) + 1 # number of clusters

    cluster_valueview = [0 for i in range(cno)]
    for i in range(n): cluster_valueview[cluster[i]] += 1
    #print(cluster_valueview)
    cvmax = max(cluster_valueview)
    candidate = []
    for i in range(cno):
        if(cluster_valueview[i] == cvmax): candidate.append(i)
    random.shuffle(candidate)
    S = candidate[0]            # select 1 random cluster among the max size one
    #print('cno is')
    #print(cno)
    #print(S)
    indexlist = []
    for i in range(n):
        if(cluster[i] == S): indexlist.append(i)
    #print(indexlist)

    subgraph = [[l[j] for j in indexlist] for l in [graph[i] for i in indexlist]]    # make subgraph of that cluster


    # run function for that subgraph
    # get categorization
    #new_sub_cluster = normalize(clustering_Spectral_Bisection(subgraph))  # Expect categorization size cno and only consist of 0 and 1
    new_sub_cluster = normalize(clustering_Kernighan_Lin(subgraph))                        ############################# SWITCH 1 #########################
    new_cluster = cluster.copy()

    for i in range(len(indexlist)):
        if(new_sub_cluster[i] == 0): new_cluster[indexlist[i]] = cno
 #   print(new_cluster)
    return normalize(new_cluster)

    # marge categorization onto prev result

    return


def clustering_Kernighan_Lin_nver(graph):
#    cluster = normalize(clustering_Spectral_Bisection(graph))
    cluster = normalize(clustering_Kernighan_Lin(graph))                                   ############################# SWITCH 2 ############################
    mq = MQ(cluster, graph)
    flag = 0
    while True:
 #       print('loop ongoing...')
 #       print(cluster)
 #       print('mq ongoing')
 #       print(mq)
        new_cluster = one_step_more(graph, cluster)
        nmq = MQ(new_cluster, graph)
        if(mq > nmq): break
        if(mq == nmq):
            if(flag > 3): break
            flag += 1
        cluster = new_cluster.copy()
        mq = nmq
#    print('final result')
#    print(cluster)
#    print('final mq')
#    print(mq)
    return cluster


def clustering_Kernighan_Lin(graph):
    num_of_module = len(graph[0])
    cluster = []

    '''TODO'''
    for i in range(num_of_module):
        if i < num_of_module//2:
            cluster.append(0)
        else:
            cluster.append(1)
    random.shuffle(cluster)

    maximum_MQ = MQ(cluster, graph)
    cur_cluster = cluster.copy()
    while True:
        partition1 = []
        partition2 = []
        cur_maximum_MQ = 0
        for i in range(num_of_module):
            if cluster[i] == 0:
                partition1.append(i)
            else:
                partition2.append(i)

        for i in range(len(partition1)):
            for j in range(len(partition2)):
                new_cluster = cluster.copy()
                swap(new_cluster, partition1[i], partition2[j])
                cur_MQ = MQ(new_cluster, graph)
                if cur_MQ > cur_maximum_MQ:
                    cur_maximum_MQ = cur_MQ
                    cur_cluster = new_cluster.copy()

        if maximum_MQ >= cur_maximum_MQ:
            break

        cluster = cur_cluster.copy()
        maximum_MQ = cur_maximum_MQ

    return cluster


def normalize(cluster):
    n = len(cluster)
    norm = []
    minusval = []
    for i in range(n): norm.append(0)
    for i in range(n): norm[cluster[i]] = norm[cluster[i]] + 1
    temp = 0
    for i in range(n):
        minusval.append(temp)
        if(norm[i] == 0): temp = temp + 1
    newcluster = []
    for i in range(n):
        newcluster.append(cluster[i] - minusval[cluster[i]])
    return newcluster


def MQ(cluster0, graph):
    cluster = normalize(cluster0)
    n = len(cluster)
    cnum = 0
    for i in range(n):
        if cluster[i] > cnum:
            cnum = cluster[i]
    cnum = cnum + 1
    cresult = []
    for i in range(cnum):
        eachcluster = []
        for j in range(n):
            if cluster[j] == i:
                eachcluster.append(j)
        cresult.append(eachcluster)

    mq = 0
    for i in range(cnum):
        mf = 0
        coh = 0
        cup = 0
        elm = cresult[i]
        m = len(elm)
        for j in range(m):
            for k in range(m):
                if j != k:
                    coh = coh + graph[elm[j]][elm[k]]
            for k in range(n):
                cup = cup + graph[elm[j]][k]
        cup = cup - coh
        coh = coh / 2
        if coh == 0:
            mf = 0
        else:
            mf = coh / (coh + (cup/2))
        mq = mq + mf
    return mq


def clustering_bottom_up(graph):
    num_of_module = len(graph[0])
    cluster = [i for i in range(num_of_module)]

    mq = MQ(cluster, graph)
    r=1
    while True:
        print(r, " round", mq, cluster)
        r+=1
        next_MQ = 0
        unique_cluster = list(set(cluster))
        next_cluster = cluster.copy()

        #choose two cluster and merge
        for i in range(len(unique_cluster)):
            for j in range(i+1, len(unique_cluster)):
                temp = cluster.copy()
                for c in range(len(temp)):
                    if temp[c] == unique_cluster[j]:
                        temp[c] = unique_cluster[i]
                cur_MQ = MQ(temp, graph)
                if next_MQ < cur_MQ:
                    next_MQ = cur_MQ
                    next_cluster = temp.copy()
        if next_MQ > mq:
            cluster = normalize(next_cluster.copy())
            mq = next_MQ
        else:
            break

    return cluster


def main():

    file_name = "data/undirect3.csv"
    graph = csv_to_list(file_name)
    #print_graph(graph)
    #print()

    print("random...")
    cluster_result1 = clustering_random(graph)
    print("Kernighan Lin nver...")
    cluster_result2 = clustering_Kernighan_Lin_nver(graph)  #needs long time...
    print("Spectral Bisection nver...")
    cluster_result3 = clustering_Spectral_Bisection_nver(graph)
    print("Kernighan Lin...")
    cluster_result4 = clustering_Kernighan_Lin(graph)
    print("Spectral Bisection...")
    cluster_result5 = clustering_Spectral_Bisection(graph)
    print("Bottom Up...")
    cluster_result6 = clustering_bottom_up(graph)

    print("random")
    print_cluster_result_list(cluster_result1)
    print(MQ(cluster_result1, graph))
    print()

    print("Kerninghan Lin")
    print_cluster_result_list(cluster_result4)
    print(MQ(cluster_result4, graph))
    print()

    print("Kerninghan Lin nver")
    print_cluster_result_list(cluster_result2)
    print(MQ(cluster_result2, graph))
    print()

    print("Spectral Bisection")
    print_cluster_result_list(cluster_result5)
    print(MQ(cluster_result5, graph))
    print()


    print("Spectral Bisection nver")
    print_cluster_result_list(cluster_result3)
    print(MQ(cluster_result3, graph))
    print()

    print("Bottom Up")
    print_cluster_result_list(cluster_result6)
    print(MQ(cluster_result6, graph))
    print()

    return


if __name__ == "__main__":
    main()