import copy
import os
import matplotlib.image
import numpy as np
import torch
from easydict import EasyDict
from armor_py.utils import del_tensor_element, fix_random
from armor_py.models import CNN_CIFAR, CNN_MNIST, CNN_MRI,CNN_Tumors,CNN_Pathology
from armor_py.options import args_parser
from armor_py.sampling import ld_cifar10, ld_mnist,ld_MRI,ld_GLioma,ld_Tumors,ld_Pathology
from attack.Global_PGD   import Global_projected_gradient_descent
import matplotlib.pyplot as plt
import time
import wandb
import json


np.set_printoptions(threshold=np.inf)

plt.rcParams.update({
    "lines.color": "white",
    "patch.edgecolor": "white",
    "text.color": "black",
    "axes.facecolor": "white",
    "axes.edgecolor": "lightgray",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "grid.color": "lightgray",
    "figure.facecolor": "black",
    "figure.edgecolor": "black",
    "savefig.facecolor": "black",
    "savefig.edgecolor": "black"})

def wandb_init(args):
        run = wandb.init(reinit=True, project="Outputs",
                         name="dataset" + str(args.dataset),
                         config=args)
        return run

# matplotlib.use('Agg')

def show(img):
    npimg= ( img-torch.min(img)/ ( torch.max(img)-torch.min(img)))
    print(torch.max(npimg))
    print(torch.min(npimg))
    npimg = npimg.detach().cpu().numpy()
    img=np.transpose(npimg, (1,2,0))[:,:,0]
    # plt.imshow(np.transpose(npimg, (1,2,0))[:,:,0],cmap='gray')
    # plt.xticks([])
    # plt.yticks([])
    # plt.show img()

    return img
def show_noise(img):
    # img= ( img-torch.min(img))/(torch.max(img)-torch.min(img))
    img= ( img-torch.min(img)) 
    # npimg= npimg/torch.max(img)
    # img=torch.abs(img)
    img=img.detach().cpu().numpy()
    if (args.dataset=='Pathology'):
           img=(np.transpose(img, (1,2,0)))
    else:
           img=(np.transpose(img, (1,2,0)))[:,:,0]
    return img
def show_rgb(img):
    npimg = img.detach().cpu().numpy()
    img=np.transpose(npimg, (1,2,0))
    # plt.imshow(np.transpose(npimg, (1,2,0)))
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()
    return img

def centralized_gradient(x,use_gc=True,gc_conv_only=False):
    if use_gc:
      if gc_conv_only:
        if len(list(x.size()))>3:
            x.add_(-x.mean(dim = tuple(range(2,len(list(x.size())))), keepdim = True))
      else:
        if len(list(x.size()))>1:
            # print(torch.mean(x))
            # print(x.mean(dim = tuple(range(1,len(list(x.size()))))))
            # print(-x.mean(dim = tuple(range(1,len(list(x.size())))), keepdim = True))
            x.add_(-x.mean(dim = tuple(range(2,len(list(x.size())))), keepdim = True))
    return x  


class Angle_gradients():

    def __init__(self,model1,model2,x,y,centralized=0):
        self.model1=copy.deepcopy(model1)
        self.model2=copy.deepcopy(model2)
        self.x=x.detach().clone()
        self.y=y
        self.centralized=centralized

    def gradient_calculator(self,model):

        x = self.x.clone().detach().to(torch.float).requires_grad_(True)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(model(x), self.y)
        loss.backward()
        grad_matrix=copy.deepcopy(x.grad)

        if self.centralized:
                grad_matrix=centralized_gradient(grad_matrix)
    
        return   grad_matrix

    
    def angle_calculator(self):

        grad_corrupt= self.gradient_calculator(self.model1)
        self.centralized=0
        grad= self.gradient_calculator(self.model2)

        tensor1=torch.flatten(grad_corrupt).unsqueeze(1).t()
        tensor2=torch.flatten(grad).unsqueeze(1)
        L2_norms=((tensor1**2).sum().sqrt())*((tensor2**2).sum().sqrt())
    
        dot=torch.mm(tensor1,tensor2)/L2_norms

        # matmul_gradients=torch.bmm(grad,grad_corrupt)
        # dot_gradients=torch.diagonal(matmul_gradients,dim1=-2,dim2=-1)
        return dot.detach().cpu().numpy()[0][0]

def per_attack_noise(eps,eps_step,rand_init,iter_round,beta=0,use_gc=True,noise_inj=False,gamma=0,loaded_models=0):
    fix_random(1)
    if (args.uplink_noise==True):
       path = "by_client/uplink_noise/" + args.dataset + "/client_num_{}/".format(args.client_num_in_total)
    if (args.uplink_noise==False):
       path = "by_client/no_uplink_noise/" + args.dataset + "/client_num_{}/".format(args.client_num_in_total)

    pgd_path = path + "Attack_fixed/Client_noise_only/attack_log/"
    psp_path = path + "Attack_fixed/Client_noise_only/attack_log_with_average_ASR/"
    time_path = path + "Attack_fixed/Client_noise_only/attack_time/"
    alignment_path = path + "alignment.json"

    if not os.path.exists(pgd_path):
        os.makedirs(pgd_path)
    if not os.path.exists(psp_path):
        os.makedirs(psp_path)
    if not os.path.exists(time_path):
        os.makedirs(time_path)
    attack_list_path = path + "Attack_fixed/Client_noise_only/attack_list/"
    if not os.path.exists(attack_list_path):
        os.makedirs(attack_list_path)
    # pgd_file = pgd_path + "Global_pgd_{:.3f}_eps_{:.3f}_step_{:.3f}_projected_{:.3f}_iterrround_{:.3f}_useGC_{:.3f}_noiseinj_{:.3f}_beta_{:.3f}_gamma_{:.3f}".format(args.global_noise_scale,eps,eps_step,rand_init,iter_round,use_gc,noise_inj,beta,gamma) + ".out"
    # attack_list_file= attack_list_path + "Global_pgd_{:.3f}_eps_{:.3f}_step_{:.3f}_projected_{:.3f}_iterrround_{:.3f}_noiseinj_{:.3f}_beta_{:.3f}_gamma_{:.3f}".format(args.global_noise_scale,eps,eps_step,rand_init,iter_round,use_gc,noise_inj,beta,gamma) + ".out"
  
    if loaded_models>0:
        file_name= "Global_pgd_{:.3f}_eps_{:.3f}_step_{:.3f}_projected_{:.3f}_iterrround_{:.3f}_useGC_{:.3f}_noiseinj_{:.3f}_beta_{:.3f}_loadedmodels_{:.3f}".format(args.global_noise_scale,eps,eps_step,rand_init,iter_round,use_gc,noise_inj,beta,loaded_models) + ".out"
        time_file= time_path + file_name
        psp_file=psp_path+file_name
        pgd_file = pgd_path + file_name
        attack_list_file= attack_list_path + file_name
    else:
        file_name= "pgd_{:.3f}_eps_{:.3f}_step_{:.3f}_projected_{:.3f}_iterrround_{:.3f}_useGC_{:.3f}_noiseinj_{:.3f}_beta_{:.3f}_gamma_{:.3f}".format(args.global_noise_scale,eps,eps_step,rand_init,iter_round,use_gc,noise_inj,beta,gamma) + ".out"
        time_file= time_path + file_name
        psp_file=psp_path+file_name
        pgd_file = pgd_path + file_name
        attack_list_file= attack_list_path + file_name

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
    for global_saved_iter in range (args.total_models-1-loaded_models,args.total_models-1):
        global_models.append(copy.deepcopy(net_glob))
        global_models_path=global_path + "Global_noise_{:.3f}_round_{:.3f}.pth".format(args.global_noise_scale,global_saved_iter)
        global_models[counter].load_state_dict(torch.load(global_models_path))
        global_models[counter].eval()
        counter+=1
    alignment_list={}
    alignment_list['adversarial']={}
    alignment_list['test']={}
    for corrupted_idx in range(args.client_num_in_total):
        alignment_list['adversarial'][corrupted_idx]={}
        alignment_list['test'][corrupted_idx]={}

        fix_random(2)
        pgd_data += "##############################################################################\n"
        pgd_data += "Adversary Examples Generated on Client {}\n".format(corrupted_idx)

        list_data += "##############################################################################\n"
        list_data += "Adversary Examples Generated on Client {}\n".format(corrupted_idx)

        test_round = 0
        for x, y in data.test:
           
            x = x.to(device)
            y = y.to(device)
            file_path = path + prefix + "/{:.3f}/".format(args.global_noise_scale) + 'Global'+ "_{}.pth".format(idx)
            x_update=copy.deepcopy(x)

            #Rescaling eps to min-max
            scale=((torch.max(x)-torch.min(x)).cpu().numpy())/2
            eps=eps*scale
            eps_step=eps_step*scale
            if loaded_models>0:
            #loading global model in past iterations to perform globalPGD
                counter =0
                for global_saved_iter in range (args.total_models-1-loaded_models,args.total_models-1):
                        x_update=Global_projected_gradient_descent( global_models[counter],x, x_update, eps, eps_step, args.epoch, np.inf,rand_init,use_gc=use_gc,noise_inj=noise_inj,beta=beta,gamma=gamma)
                        counter +=1
            if (iter_round==1):
                eps_step=eps
            if (args.timer==1):
                start = time.time()
            x_pgd = Global_projected_gradient_descent(net[corrupted_idx],x, x_update, eps, eps_step, iter_round, np.inf,rand_init,use_gc=use_gc,noise_inj=noise_inj,beta=beta,gamma=gamma)
            if (args.timer==1):
                end = time.time()
            if( args.all_numbers == 0):
                if (args.dataset=='Pathology'):
                    x1=show_rgb(x[index])
                    x2=show_rgb(x_pgd[index])
                else:
                    x1=show(x[index])
                    x2=show(x_pgd[index])
                x_noise=show_noise((x[index]-x_pgd[index]))
                images_x1 = wandb.Image(x1, caption="Maing")
                images_x2 = wandb.Image(x2, caption="Corrupted")
              
                images_x_noise = wandb.Image(x_noise, caption="Noise")
                # images_x1_zoom = wandb.Image(x1[40:55,40:55], caption="Maing")
                # images_x2_zoom = wandb.Image(x2[40:55,40:55], caption="Corrupted")
                images_x_noise_zoom = wandb.Image(x_noise, caption="Noise")



            for idx in range(args.client_num_in_total):
                report = EasyDict(nb_test=0, nb_correct=0, correct_pgd_predict=0, correct_pgd_in_corrected=0)
                y_pred_prob, y_pred = net[idx](x).max(1)
                y_pred_pgd_prob, y_pred_pgd = net[idx](x_pgd).max(1)
                if (args.all_numbers==0):
                    sm=torch.nn.functional.softmax(net[idx](x)[index].float()).detach().cpu().numpy()
                    sm_pgd=torch.nn.functional.softmax(net[idx](x_pgd)[index].float()).detach().cpu().numpy()
                


                angle_gradients=Angle_gradients(net[corrupted_idx],net[idx],x_pgd,y).angle_calculator()
                alignment_list['adversarial'][corrupted_idx][idx]=str(angle_gradients)
            
            
                angle_gradients=Angle_gradients(net[corrupted_idx],net[idx],x,y).angle_calculator()
                alignment_list['test'][corrupted_idx][idx]=str(angle_gradients)
                    # gradient_alignments['']
                    
                                

                report.nb_test += y.size(0)
                report.nb_correct += y_pred.eq(y).sum().item()
                
                # 0 predict incorrectly
                # 1 predict correctly & attack failed
                # 2 predict correctly & attack succeed
                if( args.all_numbers == 0):

                        wandb.log({ 'main':images_x1,
                                'corrupt': images_x2,
                                'noise': images_x_noise,
                                # 'main_zoom': images_x1_zoom,
                                # 'corrupt_zoom':images_x2_zoom,
                                'index' : index,
                                'eps':eps/scale,
                                'iterations':iter_round,
                                'client_idx':idx,
                                'confidence':sm,
                                'PGD confidenx':sm_pgd,
                                'true label':y_pred[index],
                                'PGD label':y_pred_pgd[index]
                                })

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
            # if(args.all_numbers==0):
            #     return
            pgd_data += "##############################################################################\n"
            list_data += "##############################################################################\n"
            test_round += 1
            break
    if args.timer==1:
        with open(time_file, "w", encoding="utf-8") as f:
                        elaped="time:{:.3f}\n".format(end-start)
                        asr = "ASR:{}\n".format(sum(ASR_bank)/len(ASR_bank))

                        f.write(elaped+asr)

    with open(alignment_path, 'w') as fp:
            json.dump(alignment_list, fp)           
                        
                        
    psp_data = pgd_data + "ASR:{}\n".format(sum(ASR_bank)/len(ASR_bank))
    with open(psp_file, "w", encoding="utf-8") as f:
                    f.write(psp_data)

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
        rand_init=True
        net_glob.eval()
        args.random_seed = 50
    elif args.dataset == "Glioma":
        args.clip_threshold = 20
        eps_step = 0.01
        iter_round = 40
        eps = 0.2
        use_gc=True
        rand_init=True
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
        rand_init=True
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
        rand_init=True
        data = ld_Pathology()
        net_glob = CNN_Pathology()
        net_glob.to(device)
        net_glob.eval()
        args.random_seed = 50
    print("dataset = " + args.dataset + ", num of client = {} , noise = {:.3f} begins...".format(args.client_num_in_total, args.global_noise_scale))
    print("dataset = " + args.dataset + ", num of client = {} , noise = {:.3f} begins...".format(args.client_num_in_total, args.global_noise_scale))
    print("################################################## models", list(range (args.total_models-1-args.loaded_models,args.total_models-1)),"are being loaded.. ##########################################################") 
    wandb_on=0
    eps_step=0.002

    # Evaluating different EPS
    if args.all_numbers==1:
        #For both versions (PGD and GlobalPGD)
        for loaded_models in [10,5,0]:
            for rand_init in [True]:
                eps_step=0.005
                for eps in [0.03,0.01,0.02,0.07,0.04,0.05,0.06]:
                    per_attack_noise(eps=eps,eps_step=eps_step,use_gc=use_gc,loaded_models=loaded_models,rand_init=rand_init,iter_round=1,beta=0.5,noise_inj=False)
                    per_attack_noise(eps=eps,eps_step=eps_step,use_gc=use_gc,loaded_models=loaded_models,rand_init=rand_init,iter_round=20,noise_inj=False)
                
        # evaluating different EPS steps
            eps=0.03
            for rand_init in [True]:
                        for eps_step in [0.001,0.002,0.003,0.005,0.007,0.010,0.015]:
                            per_attack_noise(eps=eps,eps_step=eps_step,loaded_models=loaded_models,rand_init=rand_init,iter_round=30,noise_inj=False)
    if args.all_numbers==0:
        run = wandb_init(args)

        for eps in [0.01]:
                for index in range (100):
                        per_attack_noise(eps=eps,eps_step=eps_step,use_gc=use_gc,loaded_models=args.loaded_models,rand_init=False,iter_round=1,beta=0.5,noise_inj=False)
        run.finish()
    if args.timer==1:
       eps_step=0.001
       rand_init=True
       eps =0.05
       for loaded_models in [10,0]:
                per_attack_noise(eps=eps,eps_step=eps_step,use_gc=use_gc,loaded_models=loaded_models,rand_init=rand_init,iter_round=1,beta=0.5,noise_inj=False)
                per_attack_noise(eps=eps,eps_step=eps_step,use_gc=use_gc,loaded_models=loaded_models,rand_init=rand_init,iter_round=20,noise_inj=False)
                per_attack_noise(eps=eps,eps_step=eps_step,use_gc=use_gc,loaded_models=loaded_models,rand_init=rand_init,iter_round=40,beta=0.5,noise_inj=False)
                # per_attack_noise(eps=eps,eps_step=eps_step,use_gc=use_gc,loaded_models=loaded_models,rand_init=rand_init,iter_round=20,noise_inj=False)
                
    
    
    
    
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
  