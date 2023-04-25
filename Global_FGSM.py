import copy
import os
import matplotlib.image
import numpy as np
import torch
from easydict import EasyDict
from attack.Global_PGD   import Global_projected_gradient_descent
from armor_py.utils import del_tensor_element, fix_random
from armor_py.models import CNN_CIFAR, CNN_MNIST, CNN_MRI,CNN_Tumors,CNN_Pathology
from armor_py.options import args_parser
from armor_py.sampling import ld_cifar10, ld_mnist,ld_MRI,ld_GLioma,ld_Tumors,ld_Pathology
np.set_printoptions(threshold=np.inf)
matplotlib.use('Agg')


def per_attack_noise(eps,eps_step,rand_init,iter_round,beta=None,use_gc=True,noise_inj=True,gamma=None,loaded_models=0):
    fix_random(1)
    if (args.uplink_noise==True):
       path = "by_client/uplink_noise/" + args.dataset + "/client_num_{}/".format(args.client_num_in_total)
    if (args.uplink_noise==False):
       path = "by_client/no_uplink_noise/" + args.dataset + "/client_num_{}/".format(args.client_num_in_total)

    pgd_path = path + "Attack_fixed/Client_noise_only/attack_log/"
    if not os.path.exists(pgd_path):
        os.makedirs(pgd_path)

    attack_list_path = path + "Attack_fixed/Client_noise_only/attack_list/"
    if not os.path.exists(attack_list_path):
        os.makedirs(attack_list_path)
    if loaded_models>0:
        pgd_file = pgd_path + "Global_pgd_{:.3f}_eps_{:.3f}_step_{:.3f}_projected_{:.3f}_iterrround_{:.3f}_useGC_{:.3f}_noiseinj_{:.3f}_beta_{:.3f}_gamma_{:.3f}".format(args.global_noise_scale,eps,eps_step,rand_init,iter_round,use_gc,noise_inj,beta,gamma) + ".out"
        attack_list_file= attack_list_path + "Global_pgd_{:.3f}_eps_{:.3f}_step_{:.3f}_projected_{:.3f}_iterrround_{:.3f}_noiseinj_{:.3f}_beta_{:.3f}_gamma_{:.3f}".format(args.global_noise_scale,eps,eps_step,rand_init,iter_round,use_gc,noise_inj,beta,gamma) + ".out"
    else:
        pgd_file = pgd_path + "pgd_{:.3f}_eps_{:.3f}_step_{:.3f}_projected_{:.3f}_iterrround_{:.3f}_useGC_{:.3f}_noiseinj_{:.3f}_beta_{:.3f}_gamma_{:.3f}".format(args.global_noise_scale,eps,eps_step,rand_init,iter_round,use_gc,noise_inj,beta,gamma) + ".out"
        attack_list_file= attack_list_path + "pgd_{:.3f}_eps_{:.3f}_step_{:.3f}_projected_{:.3f}_iterrround_{:.3f}_noiseinj_{:.3f}_beta_{:.3f}_gamma_{:.3f}".format(args.global_noise_scale,eps,eps_step,rand_init,iter_round,use_gc,noise_inj,beta,gamma) + ".out"


    global_path = path + "Global/"

  
    # attack_list_file = attack_list_path + "attack_list_{:.3f}".format(args.global_noise_scale) + ".out"

    pgd_data = "Model Path: " + path + prefix + "\n"
    list_data = "Model Path: " + path + prefix + "\n"

    ######################### Attack begin #########################
    attack_overview= "eps:{:.3f} step:{:.3f} projected:{:.3f} iterrround:{:.3f} GradientCentralization: {:.3f} Noise injection:{:.3f} Beta for noise injection:{:.3f}".format(eps,eps_step,rand_init,iter_round,use_gc,noise_inj,beta)
    print(attack_overview)
    pgd_data += "noise_scale={:.3f}, eps={:.3f}, eps_step={:.3f}, iter_round={}\n".\
        format(args.global_noise_scale, eps, eps_step, iter_round)
    list_data += "noise_scale={:.3f}, eps={:.3f}, eps_step={:.3f}, iter_round={}\n".\
        format(args.global_noise_scale, eps, eps_step, iter_round)

    pgd_data += "################################ Attack begin ################################\n"
    list_data += "################################ Attack begin ################################\n"
    net = []
    for idx in range(args.client_num_in_total):
        net.append(copy.deepcopy(net_glob))
        file_path = path + prefix + "/{:.3f}/".format(args.global_noise_scale) + prefix + "_{}.pth".format(idx)
        net[idx].load_state_dict(torch.load(file_path))
        net[idx].eval()
    ASR_bank=[]
    global_models=[]
    counter=0
    for global_saved_iter in range (args.total_models-1-args.loaded_models,args.total_models-1):
        global_models.append(copy.deepcopy(net_glob))
        global_models_path=global_path + "Global_noise_{:.3f}_round_{:.3f}.pth".format(args.global_noise_scale,global_saved_iter)
        global_models[counter].load_state_dict(torch.load(global_models_path))
        global_models[counter].eval()
        counter+=1
    for corrupted_idx in range(args.client_num_in_total):
        fix_random(2)
        pgd_data += "##############################################################################\n"
        pgd_data += "Adversary Examples Generated on Client {}\n".format(corrupted_idx)

        list_data += "##############################################################################\n"
        list_data += "Adversary Examples Generated on Client {}\n".format(corrupted_idx)

        test_round = 0
        for x, y in data.test:
            if test_round >= 1:
                break
            x = x.to(device)
            y = y.to(device)
            file_path = path + prefix + "/{:.3f}/".format(args.global_noise_scale) + 'Global'+ "_{}.pth".format(idx)
            
            x_update=copy.deepcopy(x)
            #loading global model in past iterations
            counter =0
            if loaded_models>0:
                for global_saved_iter in range (args.total_models-1-loaded_models,args.total_models-1):
                        x_update=Global_projected_gradient_descent( global_models[counter],x, x_update, eps, eps_step, args.epoch, np.inf,rand_init,use_gc=use_gc,noise_inj=noise_inj,beta=beta,gamma=gamma)
                        counter +=1
            # x_pgd = projected_gradient_descent(net[corrupted_idx], x_update, eps, eps_step, iter_round, np.inf,rand_init,use_gc=use_gc,noise_inj=noise_inj,beta=beta,gamma=gamma)
            x_pgd = Global_projected_gradient_descent(net[corrupted_idx],x, x_update, eps, eps_step, 1, np.inf,rand_init,use_gc=use_gc,noise_inj=noise_inj,beta=beta,gamma=gamma)
            for idx in range(args.client_num_in_total):
                report = EasyDict(nb_test=0, nb_correct=0, correct_pgd_predict=0, correct_pgd_in_corrected=0)
                _, y_pred = net[idx](x).max(1)
                _, y_pred_pgd = net[idx](x_pgd).max(1)
                report.nb_test += y.size(0)
                report.nb_correct += y_pred.eq(y).sum().item()
                
                # 0 predict incorrectly
                # 1 predict correctly & attack failed
                # 2 predict correctly & attack succeed

                list_mask = y_pred.eq(y)                                        # predict correctly 1
                list_value = ~y_pred_pgd.eq(y_pred)                             # attack successfully
                list_result = (list_mask & list_value).long().cpu().numpy()     # predict correctly & attack successfully 2
                list_mask = list_mask.long().cpu().numpy()                      # predict correctly 1
                list_result = str(list_result + list_mask).replace("\n", "")
                list_result = list_result.replace("[", "")
                list_result = list_result.replace("]", "")
                list_data += list_result + "\n"

                y_pred_correct = y_pred             #predicted values before corruptuoio data
                y_pred_correct_pgd = y_pred_pgd     #prediction values after corruption of data
                for i in range(x.shape[0]):
                    if y_pred[x.shape[0] - 1 - i] != y[x.shape[0] - 1 - i]:
                        y_pred_correct = del_tensor_element(y_pred_correct, x.shape[0]-1-i)
                        y_pred_correct_pgd = del_tensor_element(y_pred_correct_pgd, x.shape[0]-1-i)
                report.correct_pgd_in_corrected += y_pred_correct_pgd.eq(y_pred_correct).sum().item() #number of changed predictions

                if idx == corrupted_idx:
                    pgd_data += "Test on Client {}: Clean Acc: {:.2f}(%) / ASR: {:.2f}(%) ************* Generated\n".format(
                        idx,
                        (report.nb_correct / report.nb_test * 100.0),
                        ((1 - report.correct_pgd_in_corrected / report.nb_correct) * 100.0))
                    ASR_bank.append((1 - report.correct_pgd_in_corrected / report.nb_correct) * 100.0)
                else:
                    pgd_data += "Test on Client {}: Clean Acc: {:.2f}(%) / ASR: {:.2f}(%)\n".format(
                        idx,
                        (report.nb_correct / report.nb_test * 100.0),
                        ((1 - report.correct_pgd_in_corrected / report.nb_correct) * 100.0))
                    ASR_bank.append((1 - report.correct_pgd_in_corrected / report.nb_correct) * 100.0)


                with open(pgd_file, "w", encoding="utf-8") as f:
                    f.write(pgd_data)

                with open(attack_list_file, "w", encoding="utf-8") as f:
                    f.write(list_data)
            pgd_data += "##############################################################################\n"
            list_data += "##############################################################################\n"

            test_round += 1

    pgd_data += "ASR:{}\n".format(sum(ASR_bank)/len(ASR_bank))

    with open(pgd_file, "w", encoding="utf-8") as f:
                    f.write(pgd_data)
if __name__ == '__main__':
    args = args_parser()
    device = torch.device("cuda:{}".format(args.cuda))
    args.device = device
    Client, Client_after = {}, {}
    prefix = "Client_final"
    fix_random(0)

    if args.dataset == "cifar":
        eps_step = 0.008
        iter_round = 20
        eps = 0.025
        data = ld_cifar10()
        net_glob = CNN_CIFAR()
        net_glob.to(device)
        net_glob.eval()
        args.random_seed = 1000

    elif args.dataset == "mnist":
        eps_step = 0.01
        iter_round = 40
        eps = 0.2
        data = ld_mnist()
        net_glob = CNN_MNIST()
        net_glob.to(device)
        net_glob.eval()
        args.random_seed = 50
    elif args.dataset == "MRI":
        args.clip_threshold = 20
        eps_step = 0.01
        iter_round = 40
        eps = 0.2
        use_gc=True
        data = ld_MRI()
        net_glob = CNN_MRI()
        net_glob.to(device)
        net_glob.eval()
        args.random_seed = 50
    elif args.dataset == "Glioma":
        args.clip_threshold = 20
        eps_step = 0.01
        iter_round = 40
        eps = 0.2
        use_gc=True
        data = ld_GLioma()
        net_glob = CNN_MRI()
        net_glob.to(device)
        net_glob.eval()
        args.random_seed = 50
    elif args.dataset == "Tumors":
        args.clip_threshold = 20
        eps_step = 0.01
        iter_round = 40
        eps = 0.2
        use_gc=True
        data = ld_Tumors()
        net_glob = CNN_Tumors()
        net_glob.to(device)
        net_glob.eval()
        args.random_seed = 50
    elif args.dataset == "Pathology":
        args.clip_threshold = 20
        eps_step = 0.01
        iter_round = 40
        eps = 0.2
        use_gc=True
        data = ld_Pathology()
        net_glob = CNN_Pathology()
        net_glob.to(device)
        net_glob.eval()
        args.random_seed = 50
    print("dataset = " + args.dataset + ", num of client = {} , noise = {:.3f} begins...".format(args.client_num_in_total, args.global_noise_scale))
    print("dataset = " + args.dataset + ", num of client = {} , noise = {:.3f} begins...".format(args.client_num_in_total, args.global_noise_scale))
    print("################################################## models", list(range (args.total_models-1-args.loaded_models,args.total_models-1)),"are being loaded.. ##########################################################") 
    


    #For both versions (PGD and GlobalPGD)
    for loaded_models in [0,10]:

    #Evaluating different EPS
        if args.all_numbers:
            eps_step=0.01
            for rand_init in [True,False]:
                for eps in [0.01,0.02,0.03,0.04,0.05,0.06,0.07]:
                    per_attack_noise(eps=eps,eps_step=eps_step,rand_init=True,iter_round=1,beta=0.5,loaded_models)
                    per_attack_noise(eps=eps,eps_step=eps_step,rand_init=True,iter_round=20,noise_inj=False,loaded_models)
            
        #evaluating different EPS steps
            eps=0.02
            for rand_init in [True,False]:
                        for eps_step in [0.003,0.007,0.010,0.015,0.020]:
                            per_attack_noise(eps=eps,eps_step=eps_step,rand_init=True,iter_round=20,noise_inj=False,loaded_models)
    else:
        for eps in [0.4]:
                for noise_inj in [ False]:
                         per_attack_noise(eps=eps,eps_step=eps/6,rand_init=True,iter_round=6,beta=0.5,use_gc=True,noise_inj=noise_inj,gamma=1.05)
    
    
    
    
    
    
    
    # if args.all_numbers:
    #     for use_gc in [1,0]:
    #         for rand_init in [True,False]:
    #             for eps in [0.03]:
    #             for eps_step in [0.01]:
    #                     per_attack_noise(eps=eps,eps_step=eps/6,rand_init=True,iter_round=6,beta=0.5,use_gc=True,noise_inj=noise_inj,gamma=1.05)
        
    #     #evaluating different EPS
    #     for use_gc in [1,0]: 
    #         for eps in [0.1,0.2,0.3,0.5]:
    #                     per_attack_noise(eps,0.05,False,1)
    #                     for rand_init in [True,False]:
    #                         per_attack_noise(eps,0.05,rand_init,40,use_gc)
    # else:
    #     for eps in [0.4]:
    #             for noise_inj in [False]:

    #                     per_attack_noise(eps=eps,eps_step=eps/6,rand_init=True,iter_round=6,beta=0.5,use_gc=True,noise_inj=noise_inj,gamma=1.05)
            # per_attack_noise(0.5,0.3,False,5,use_gc=True,noise_inj=False,beta)
    print("dataset = " + args.dataset + ", num of client = {} , noise = {:.3f} completed!".format(args.client_num_in_total, args.global_noise_scale))
