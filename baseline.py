"""
writer: Yongjun Shin
date: 2017.11.30
USAGE: change directory path of inputs
"""

import os
import time
from graph_partition import *

def main():
    data_dir = "./data/"  #terminate path with one slash
    out_dir = "./heuristic_out/"

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    f_r = open(out_dir + "result_random.csv", 'w')
    f_K = open(out_dir + "result_Kerninghan.csv", 'w')
    f_S = open(out_dir + "result_Spectral.csv", 'w')
    f_B = open(out_dir + "result_buttom_up.csv", 'w')
    f_r.write(",MQ, time(s), cluster\n")
    f_K.write(",MQ, time(s), cluster\n")
    f_S.write(",MQ, time(s), cluster\n")
    f_B.write(",MQ, time(s), cluster\n")

    for fname in os.listdir(data_dir):
        if not os.path.isfile(data_dir+fname):
            continue
        graph = csv_to_list(data_dir+fname)
        f_r.write(fname + ",")
        f_K.write(fname + ",")
        f_S.write(fname + ",")
        f_B.write(fname + ",")
        #print_graph(graph)
        #print()

        print("=========="+fname+"==========")
        start = time.time()
        print("random...")
        cluster_result1 = clustering_random(graph)
        end = time.time()
        t_r = end - start

        print("Kernighan Lin nver...")
        start = time.time()
        cluster_result2 = clustering_Kernighan_Lin_nver(graph)  #needs long time...
        end = time.time()
        t_K = end - start

        print("Spectral Bisection nver...")
        start = time.time()
        cluster_result3 = clustering_Spectral_Bisection_nver(graph)
        end = time.time()
        t_S = end - start

        '''print("Kernighan Lin...")
        cluster_result4 = clustering_Kernighan_Lin(graph)
        print("Spectral Bisection...")
        cluster_result5 = clustering_Spectral_Bisection(graph)'''

        print("Bottom Up...")
        start = time.time()
        cluster_result6 = clustering_bottom_up(graph)
        end = time.time()
        t_B = end - start

        print()
        print("random")
        print_cluster_result_list(cluster_result1)
        MQ_r = MQ(cluster_result1, graph)
        print(MQ_r)
        print()
        f_r.write(str(MQ_r)+","+str(t_r)+",")
        for c in cluster_result1:
            f_r.write(str(c)+",")
        f_r.write("\n")

        '''
        print("Kerninghan Lin")
        print_cluster_result_list(cluster_result4)
        print(MQ(cluster_result4, graph))
        print()
        '''
        print("Kerninghan Lin nver")
        print_cluster_result_list(cluster_result2)
        MQ_K = MQ(cluster_result2, graph)
        print(MQ_K)
        print()
        f_K.write(str(MQ_K) + "," + str(t_K) + ",")
        for c in cluster_result2:
            f_K.write(str(c) + ",")
        f_K.write("\n")
        '''
        print("Spectral Bisection")
        print_cluster_result_list(cluster_result5)
        print(MQ(cluster_result5, graph))
        print()
        '''

        print("Spectral Bisection nver")
        print_cluster_result_list(cluster_result3)
        MQ_S = MQ(cluster_result3, graph)
        print(MQ_S)
        print()
        f_S.write(str(MQ_S) + "," + str(t_S) + ",")
        for c in cluster_result3:
            f_S.write(str(c) + ",")
        f_S.write("\n")

        print("Bottom Up")
        print_cluster_result_list(cluster_result6)
        MQ_B = MQ(cluster_result6, graph)
        print(MQ_B)
        print()
        print()
        f_B.write(str(MQ_B) + "," + str(t_B) + ",")
        for c in cluster_result6:
            f_B.write(str(c) + ",")
        f_B.write("\n")

    f_r.close()
    f_K.close()
    f_S.close()
    f_B.close()

    return


if __name__ == "__main__":
    main()


