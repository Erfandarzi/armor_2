import copy
import os
import random
import re
import shutil

import torch
import numpy as np
import stat


def removeReadOnly(func, path, excinfo):
    # Using os.chmod with stat.S_IWRITE to allow write permissions
    os.chmod(path, stat.S_IWRITE)
    func(path)


def test_mkdir(path):
    if not os.path.exists((path)):
        os.mkdir(path)

def test_cpdir(source, target):
    if os.path.exists(target):
        shutil.rmtree(target,onerror=removeReadOnly)
    shutil.copytree(source, target)

def alter(file, old_str, new_str):
    file_data = ""
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            if old_str in line:
                line = line.replace(old_str, new_str)
            file_data += line
    with open(file, "w", encoding="utf-8") as f:
        f.write(file_data)


def alter_re(file, pattern, repl):
    file_data = ""
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            line = re.sub(pattern, repl, line)
            file_data += line
    with open(file, "w", encoding="utf-8") as f:
        f.write(file_data)


def del_line_num(file,num):
    file_data = ""
    with open(file, 'r', encoding="utf-8") as f:
        for i,line in enumerate(f):
            if i != int(num):
                file_data += line
    with open(file, "w", encoding="utf-8") as f:
        f.write(file_data)


def del_blank_line(file):
    file_data = ""
    with open(file, 'r', encoding="utf-8") as f:
        for line in f:
            if line.split():
                file_data += line
    with open(file, "w", encoding="utf-8") as f:
        f.write(file_data)


def fix_random(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True


def aggregate(w):
    w_avg = copy.deepcopy(w[0])
    # w[0].load_state_dict(w[0])
    if isinstance(w[0], np.ndarray):
        for i in range(1, len(w)):
            w_avg += w[i]
        w_avg = w_avg / len(w)
    else:
        for k in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[k] += w[i][k]
            w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg



def glob_with_grad(w,net_glob,grad_bank):
    w_model= copy.deepcopy(net_glob)
    for client in range (len(w)):
         w_model.load_state_dict(w[client])
         grad_client=grad_bank[client]
         for parameter,grad in zip( w_model.parameters(),grad_client):
          
               pass

         print('cient',client)


def del_tensor_element(tensor, index):
    top = tensor[0:index]
    tail = tensor[index + 1:]
    new_tensor = torch.cat((top, tail), dim=0)
    return new_tensor


def noise_add_global(global_noise_scale, net, device):
    model_par = net.state_dict()
    new_par = copy.deepcopy(model_par)
    noise_list = []
    for name in new_par:
        noise_normal = np.random.normal(0, global_noise_scale, new_par[name].size())
        noise_normal = torch.from_numpy(noise_normal).float().to(device)
        noise_list.append(noise_normal)
        new_par[name] = model_par[name] + noise_normal
    return new_par
