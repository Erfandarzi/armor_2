import os
import copy
import os
import random
import re
import shutil
import torch
import numpy as np
from armor_py.options import args_parser
from prettytable import PrettyTable
from armor_py.utils import del_tensor_element, fix_random
import matplotlib.pyplot as plt
import wandb
import itertools
import io
from dataclasses import dataclass
import pandas
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 14})



import json
dir="./by_client/json_results"
json_list=[]
for item in os.listdir(dir):
    try:
        # Opening JSON file
        full_path=os.path.join(dir,item)
        with open(full_path) as json_file:
            json_list.append(json.load(json_file))
    except json.decoder.JSONDecodeError:
            pass
#Plotting DP

df1= pandas.DataFrame.from_dict(json_list)
# print(df1)
import pandasgui
pandasgui.show(df1)
#Experiment EPS
'''
Here we show pathology data with eps <0.3
'''
def compute_graph(modality,training_method):
    eps_range=[0.01,0.05,0.08,0.01]
    step_range =[ eps/6 for eps in eps_range]
    asr_list=[]
    asr_self=[]
    eps_list=[]
    asr_list_dp=[]
    asr_self_dp=[]
    for datapoint in json_list:
        if (datapoint["modality"]== modality and  datapoint["training_method"]==training_method ):
            print(datapoint["step"])
            print("EPS",datapoint["eps"])
            if (datapoint["eps"] in eps_range and datapoint["step"] =='0.05'):

            

                if( datapoint["dp"]==["dp"]):
                        eps_list.append(datapoint["eps"])

                        asr_self_dp.append(datapoint["asr_self"])
                        asr_list_dp.append(datapoint["asr_avg"])
                if( datapoint["dp"]!= ["dp"]):
                    
                        asr_self.append(datapoint["asr_self"])
                        asr_list.append(datapoint["asr_avg"])
    eps_list.sort()
    asr_list_dp.sort()
    asr_list_dp.sort()
    asr_self.sort()
    asr_self_dp.sort()

# print(eps_list)
# print(asr_list)

# eps_list=sorted(eps_list)
# asr_self=[x for y, x in sorted(zip(eps_list, asr_self))]
# asr_list=[x for y, x in sorted(zip(eps_list, asr_list))]
# asr_list_dp=[x for y, x in sorted(zip(eps_list, asr_list_dp))]
# asr_self_dp=[x for y, x in sorted(zip(eps_list, asr_self_dp))]



    plt.plot(eps_list,asr_list,label="asr", marker='o')
    plt.plot(eps_list,asr_self,label='self',marker="v")
    plt.plot(eps_list,asr_list_dp,label="asr dp",marker="s")
    plt.plot(eps_list,asr_self_dp,label="self dp",marker="X")
    plt.legend()
    plt.show()

        # if datapoint["dp"]

# def draw_tables_AATR():
#     list=megalist(path_aatr)
#     x=PrettyTable()
#     columns_to_show= ["attack", "AATR","eps"]
#     x.field_names =columns_to_show
#     for item in list:
#             if (not ("BIM" in item['attack']) and item['step']=='0.002' and item['eps'] in['0.03'] ):
#                     items_to_show = [item[columns] for columns in columns_to_show]
#                     x.add_row(items_to_show)
#     print((x.get_string(sortby="attack")))

# compute_graph(["MRI"] ,["Client_no_noise_adv"])

# compute_graph(["pathology"] ,["Client_noise_adv"])
# compute_graph(["Pathology"] ,["Client_noise_only"])