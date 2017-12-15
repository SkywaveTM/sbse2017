"""
Input: csv file
Output: csv file
"""

import csv
import os


for filename in os.listdir(os.getcwd()):
    # Read files into list
    if ".csv" not in filename:
        continue
    with open(filename, "r") as f_read:
        reader = csv.reader(f_read, delimiter=",")
        data = list(reader)

    # Check length of rows and columns
    len_rows = len(data)
    len_columns = len(data[0])
    assert len_rows < len_columns

    n = len_columns-1

    # Add empty rows until length of rows becomes same to length of columns
    for i in range(len_rows, len_columns):
        data.append([data[0][i]]+['0' for j in range(n)])

    # Check order of names in column and rows
    assert len(data) == len(data[0])
    assert len(data[0]) == len(data[n])
    for i in range(n+1):
        assert data[0][i] == data[i][0]

    # Check self-edge
    for i in range(n+1):
        assert data[i][i] != '1'

    # Make adjacency matrix symmetric and count edges
    edges = 0
    for i in range(n+1):
        for j in range(i):
            if data[i][j] == '1' or data[j][i] == '1':
                data[i][j] = '1'
                data[j][i] = '1'
                edges += 1

    # Assert symmetry
    for i in range(1, n+1):
        for j in range(1, i):
            assert data[i][j] == data[j][i]

    # Delete the first column
    for row in data:
        row.pop(0)

    # Write csv
    with open("undirected_MDG/%s_n%d_e%d.csv" % (filename.replace(".csv", ""), n, edges), "w", newline='') as f_write:
        writer = csv.writer(f_write)
        writer.writerows(data)
