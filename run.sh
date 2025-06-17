#!/bin/bash

./build/CDSM -q ../datasets/amazon-dyna/6/query_graph/sparse_6/Q_0.in -d ../datasets/amazon-dyna/6/data_graph/data.graph.dyna --gpu 1 > output.txt

./CDSM -q ../test_data/example_q_01.in -d ../test_data/example_g_01.in --gpu 1