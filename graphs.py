import os


import hashlib
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
from armor_py.options import args_parser
from armor_py.utils import alter_re, del_blank_line, alter, test_mkdir, test_cpdir,del_line_num
# from armor_py.utils import alter_re, alter, del_blank_line, test_cpdir, test_mkdir
import re
from typing import Optional
import ast
import copy
import os
import random
import re
import shutil
from dataclasses import dataclass,field
import torch
import numpy as np
import base64

files_acc = []
# for file_acc in os.listdir(path_acc):


# def log_list(file)
#         if file_acc.endswith(".out"):
#             files_acc.append(path_acc + file_acc)

#     files_asr = []
#     for file_asr in os.listdir(path_asr):
#         if file_asr.endswith(".out"):
#             files_asr.append(path_asr + file_asr)
  

# for file in files_acc:
#     rem_text_acc_file(file)
        
# for file in files_asr:
#     rem_text_asr_file(file)


# for file_acc in os.listdir(path_acc):

#     full_path_acc=path_acc + file_acc
#     full_path_ASR = path_asr  + file_acc

@dataclass
class run():
   
    pgd: float=0
    eps: float=0
    step: float=0
    projected: bool=0
    iterrround: int=0
    useGC: bool=0
    noiseinj: bool=0
    beta: float=0
    gamma: float=0
    advpercent: float=0
    client_num: int=0
    modality: str=0
    dp: bool=0
    training_method: bool=0
    
    asr_self:float=0
    asr_avg:float=0
    asr_std:float=0
    asr_self:float=0
    asr_avg :float=0
    asr_std : float=0
    acc_avg: float=0
    acc_std : float=0

    
    modality_list:list=field(default_factory=lambda:["MRI","Glioma","Pathology"])
    dp_list:list=field(default_factory=lambda:["dp","no_dp"])
    training_method_list: list=field(default_factory=lambda:["Client_no_noise_adv","Client_noise_adv","Client_noise_only"])

    
    def compute_attributes(self,path):
        self.compute_attack_attributes(path)
        self.find_modality(path)
        self.compute_results(path)
        return self.__dict__
    

    def find_modality(self,path):
       
        self.modality=[modality for modality in self.modality_list if modality in path][0]
        self.dp=["no_dp" if "no_dp"  in path else "dp" ][0]
        self.training_method=[training_method for training_method  in self.training_method_list if training_method in path][0]
        
        del  self.training_method_list
        del  self.dp_list
        del  self.modality_list

    '''
        path_to_file="by_client/dp/MRI/client_num_3/
        Attack_fixed/Client_noise_adv/attack_log/pgd_0.010_eps_0.020_
        step_0.003_projected_1.000_iterrround_6.000_useGC_1.000_noiseinj_
        0.000_beta_0.500_gamma_1.050_advpercent_1.000.out"
    '''
   

    def compute_attack_attributes(self,path):
       for item in vars(self):
            regex = re.compile("{}\_(\d+(?:\.\d+)?)".format(item))
            value=regex.findall(path)
            if (len(value)):
                self.__dict__[item]=tryeval(value[0])
    # def files_out_list(path):
    #     files=[]
    #     for file in os.listdir(path):
    #      if file.endswith(".out"):
    #             files.append(path + file)
    #     return files
    # files_list: list=field(init=False)

    # def __post_init__(self):
    #           object.__setattr__(self, 'files_list',self.files_list_out(self.path))
    def rem_text_acc_file(self,file)-> None:
            alter_re(file, "noise_scale=.*", "")
            alter_re(file, "Model Path.*", "")
            alter(file, "nohup: ignoring input", "")
            alter(file, "################################ Attack begin ################################", "")
            alter(file, "##############################################################################", "")
            alter(file, "Adversary Examples Generated on Client ", "")
            alter_re(file, "Test on Client .* Acc: ", "")
            alter_re(file, "Test on Client .* Acc: ", "")
            alter_re(file, "\(%\) / ASR: .*", "")
            del_blank_line(file)
    ### ASR ###
    def rem_text_asr_file(self,file)->None:
            alter_re(file, "noise_scale=.*", "")
            alter_re(file, "Model Path.*", "")
            alter(file, "nohup: ignoring input", "")
            alter(file, "################################ Attack begin ################################", "")
            alter(file, "##############################################################################", "")
            alter(file, "Adversary Examples Generated on Client ", "")
            alter_re(file, "Test on Client .* ASR: ", "")
            alter_re(file, "\(%\).*", "")
            del_blank_line(file)
            # del_line_num(self.file,12)

    def compute_results(self,file)->float:
        self.rem_text_asr_file(file)
        self.rem_text_acc_file(file)
        file_Acc = open(file)
        file_ASR = open(file)
        acc, asr, asr_self = [0] * self.client_num, [0] * self.client_num, [0] * self.client_num
        acc_avg, acc_var, acc_std = [0] * self.client_num, [0] * self.client_num, [0] * self.client_num
        asr_avg, asr_var, asr_std = [0] * self.client_num, [0] * self.client_num, [0] * self.client_num
        
        idx = 0
        for line_num,i in enumerate (file_Acc):
            if line_num < (self.client_num)*(self.client_num+1):
                if idx % (self.client_num+1) == 0:
                    generated_idx = int(i)
                    acc[generated_idx] = [0] * self.client_num
                else:
                    acc[generated_idx][idx % (self.client_num+1) - 1] = float(i) / 100
                idx = idx + 1

        idx = 0
        for line_num,i in enumerate (file_ASR):
          if line_num < (self.client_num)*(self.client_num+1):
            if idx % (self.client_num+1) == 0:
                generated_idx = int(i)
                asr[generated_idx] = [0] * self.client_num
            else:
                asr[generated_idx][idx % (self.client_num+1) - 1] = float(i) / 100
            idx = idx + 1

        for generated_idx in range(self.client_num):
            asr[generated_idx]
            asr_self[generated_idx] = asr[generated_idx][generated_idx]
            del asr[generated_idx][generated_idx]
            asr_avg[generated_idx] = np.average(asr[generated_idx])
            asr_var[generated_idx] = np.var(asr[generated_idx])
            asr_std[generated_idx] = np.std(asr[generated_idx])

            acc_avg[generated_idx] = np.average(acc[generated_idx])
            acc_var[generated_idx] = np.var(acc[generated_idx])
            acc_std[generated_idx] = np.std(acc[generated_idx])
        
        self.asr_self = np.average(asr_self)
        self.asr_avg = np.average(asr_avg)
        self.asr_std = np.average(asr_std)

        self.acc_avg = np.average(acc_avg)
        self.acc_std = np.average(acc_std)


import json


def tryeval(val):
    try:
        val = ast.literal_eval(val)
    except ValueError:
        pass
    return val

    # def compute(path):
    #     compute_values_dict(self,path)
    #     compute_values_dict_other(self,path)

    # def   print (self.dp)

    # def 
    # dp/MRI/client_num_3/Attack_fixed/Client_noise_adv/
        
    # files_asr = []
    # for file_asr in os.listdir(path_asr):
    #     if file_asr.endswith(".out"):
    #         files_asr.append(path_asr + file_asr)
        
    #     for file_acc in os.listdir(path_acc):

        # full_path_acc=path_acc + file_acc
        # full_path_ASR = path_asr  + file_acc



    # asr_self_avg = np.average(asr_self)
    # asr_avg_avg = np.average(asr_avg)
    # asr_std_avg = np.average(asr_std)

    # acc_avg_avg = np.average(acc_avg)
    # acc_std_avg = np.average(acc_std)

    # acc_list.append(acc_avg_avg)
    # asr_list.append(asr_avg_avg)
    # self_list.append(asr_self_avg)

    # line = "noise={:.3f} asr_self_avg={:.2f}% Acc_avg={:.2f}% ASR_avg={:.2f}% Acc_std={:.3f} ASR_std={:.3f}\n". \
    #     format(args.global_noise_scale, asr_self_avg * 100,
    #         acc_avg_avg * 100, asr_avg_avg * 100,
    #         acc_std_avg, asr_std_avg)
    # file_data += line

    # with open(result_file, "w", encoding="utf-8") as f:
    #     f.write(file_data)

    # alter_re(result_file, "noise=", "")
    # alter_re(result_file, " asr_self_avg=", "\t")
    # alter_re(result_file, " Acc_avg=", "\t")
    # alter_re(result_file, " ASR_avg=", "\t")
    # alter_re(result_file, " Acc_std=", "\t\t")
    # alter_re(result_file, " ASR_std=", "\t\t")
    # del_blank_line(result_file)
    # print(result_file)
    # for i in file_Acc:
    #     if idx % (client_num_in_total+1) == 0:
    #         generated_idx = int(i)
    #         acc[generated_idx] = [0] * client_num_in_total
    #     else:
    #         acc[generated_idx][idx % (client_num_in_total+1) - 1] = float(i) / 100
    #     idx = idx + 1
    # for file_acc in os.listdir(path_acc):

    #             # file_acc=file_list
    #             file_data = "Noise\tAttack\tAcc_avg\tASR_avg\t\tAcc_std\tASR_std\n"
    #             # result_file = out_path + dataset +  "_client_num_{}".format(client_num_in_total) + file_list
    #             result_file = out_path  + file_acc

    #             acc_list = []
    #             asr_list = []
    #             aatr_list = []
    #             self_list = []


if __name__ == '__main__':
    path_to_file="by_client/dp/MRI/client_num_3/Attack_fixed/Client_noise_adv/attack_log/pgd_0.010_eps_0.020_step_0.003_projected_1.000_iterrround_6.000_useGC_1.000_noiseinj_0.000_beta_0.500_gamma_1.050_advpercent_1.000.out"
    # jsons=create_json()
    import uuid
    random_name = str(uuid.uuid4())
    
    for DP in ["dp","no_dp"]:
        for MOD in ["MRI","Pathology","Glioma"]:
             for METHOD in ["Client_no_noise_adv","Client_noise_adv" ,"Client_noise_only"]:
                    dirs="./by_client/"+DP+"/"+MOD+"/client_num_3/Attack_fixed/"+METHOD+"/attack_log"
                    
                    for item in os.listdir(dirs):
                            try:
                                if(len(item)>20):
                                    run1=run()
                                    dict=run1.compute_attributes(os.path.join(dirs,item))
                                    print(dict)
                                    random_name = hashlib.sha1(str.encode(str(dict))).hexdigest()
                                    # random_name = str(uuid.uuid4())
                                    # print(str(dict).encode('base64','strict'))
                                    with open("./by_client/json_results/"+random_name+'.json', 'w') as fp:
                                        json.dump(dict, fp)
                            except  (TypeError, ValueError) as e:
                                    pass
                     
                    

    # jsons.compute_values_dict(path_to_file)
    # jsons.compute_values_dict_other(path_to_file)
    # jsons.to_json()

    # myfile=average(path_to_file)
    # print(myfile.average_ASR_ACC_from_edited_file())
    # "by_client/dp/MRI/client_num_3/Attack_fixed/Client_noise_adv/attack_log/pgd_0.010_eps_0.020_step_0.003_projected_1.000_iterrround_6.000_useGC_1.000_noiseinj_0.000_beta_0.500_gamma_1.050_advpercent_1.000.out"