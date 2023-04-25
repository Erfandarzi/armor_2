import os
import matplotlib.pyplot as plt
import numpy as np

from armor_py.options import args_parser
from armor_py.utils import alter_re, alter, del_blank_line, test_cpdir, test_mkdir


def atr_per_process():
    client_num_in_total = args.client_num_in_total
    model_name = "Client_" + args.model_name

    path = dataset_path + "client_num_{}/Attack_fixed/".format(client_num_in_total) + model_name + "/"
    path_list = path + prefix + "/"
    path_list_processed = path + prefix + "_processed/"
    AATR_list = []

    # if loaded_models>0:
    #     file_name= "Global_pgd_{:.3f}_eps_{:.3f}_step_{:.3f}_projected_{:.3f}_iterrround_{:.3f}_useGC_{:.3f}_noiseinj_{:.3f}_beta_{:.3f}_loadedmodels_{:.3f}".format(args.global_noise_scale,eps,eps_step,rand_init,iter_round,use_gc,noise_inj,beta,loaded_models) + ".out"
    #     time_file= time_path + file_name
    #     psp_file=psp_path+file_name
    #     pgd_file = pgd_path + file_name
    #     attack_list_file= attack_list_path + file_name
    # else:
    #     file_name= "pgd_{:.3f}_eps_{:.3f}_step_{:.3f}_projected_{:.3f}_iterrround_{:.3f}_useGC_{:.3f}_noiseinj_{:.3f}_beta_{:.3f}_gamma_{:.3f}".format(args.global_noise_scale,eps,eps_step,rand_init,iter_round,use_gc,noise_inj,beta,gamma) + ".out"
    #     time_file= time_path + file_name
    #     psp_file=psp_path+file_name
    #     pgd_file = pgd_path + file_name
    #     attack_list_file= attack_list_path + file_name

    test_cpdir(path_list, path_list_processed)

    files_list = []
    for file_list in os.listdir(path_list_processed):
        if file_list.endswith(".out"):
            files_list.append(path_list_processed+file_list)
    for file in files_list:
        alter_re(file, "noise_scale=.*", "")
        alter_re(file, "Model Path: .*", "")
        alter(file, "################################ Attack begin ################################", "")
        alter(file, "##############################################################################", "")
        alter(file, "Adversary Examples Generated on Client ", "")
        del_blank_line(file)
    for file_list in files_list:

        # full_path =  path_list_processed + prefix + "_{:.3f}.out".format(args.global_noise_scale)

        file_attack_list = open(file_list)
        # 0 predict incorrectly
        # 1 predict correctly & attack failed
        # 2 predict correctly & attack succeed
        print('###################')
        raw_arr = np.zeros((client_num_in_total, client_num_in_total, num_items))
        i_idx = 0
        generated_idx = 0
        for i in file_attack_list:
            if i_idx % (client_num_in_total+1) == 0:
                generated_idx = int(i)
            else:
                raw_arr[generated_idx][i_idx % (client_num_in_total+1) - 1] = i.split()
            i_idx = i_idx + 1

        attack_samples_used = {}
        num_attack_incorrect, num_attack_fail, num_attack_succeed, num_predict_correct = [0]*client_num_in_total, [0]*client_num_in_total, [0]*client_num_in_total, [0]*client_num_in_total
        TR, ATR = [0]*client_num_in_total, [0]*client_num_in_total
        for corrupted_idx in range(client_num_in_total):
            attack_samples_used[corrupted_idx] = []
            for image_idx in range(num_items):
                if raw_arr[corrupted_idx,corrupted_idx,image_idx] == 2:
                    attack_samples_used[corrupted_idx].append(raw_arr[corrupted_idx, :,image_idx]) # sample hayi ke to har client toonestan movafaq labele test ro misclassify konan, adade client vaindexeshuno zakhire mikone
        for corrupted_idx in range(client_num_in_total):
            num_image_used = len(attack_samples_used[corrupted_idx])
            if num_image_used == 0:
                ATR[corrupted_idx] = 0
            else:
                num_attack_incorrect[corrupted_idx] = [0]*num_image_used
                num_attack_fail[corrupted_idx] = [0] * num_image_used
                num_attack_succeed[corrupted_idx] = [0] * num_image_used
                num_predict_correct[corrupted_idx] = [0] * num_image_used
                TR[corrupted_idx] = [0] * num_image_used
                for image_idx_used in range(num_image_used):
                    num_attack_incorrect[corrupted_idx][image_idx_used]=(np.equal(attack_samples_used[corrupted_idx][image_idx_used], np.zeros(client_num_in_total)).sum())
                    num_attack_fail[corrupted_idx][image_idx_used]=(np.equal(attack_samples_used[corrupted_idx][image_idx_used], np.ones(client_num_in_total)).sum())
                    num_attack_succeed[corrupted_idx][image_idx_used]=(np.equal(attack_samples_used[corrupted_idx][image_idx_used], 2 * np.ones(client_num_in_total)).sum())
                    num_predict_correct[corrupted_idx][image_idx_used]=(num_attack_fail[corrupted_idx][image_idx_used] + num_attack_succeed[corrupted_idx][image_idx_used])
                    TR[corrupted_idx][image_idx_used]=np.minimum(1, num_attack_succeed[corrupted_idx][image_idx_used]/num_predict_correct[corrupted_idx][image_idx_used])
                ATR[corrupted_idx]=(np.average(TR[corrupted_idx]))
        AATR = np.average(ATR)
        AATR_list.append(AATR)

    TR_array = []
    for TR_idx in range(len(TR)):
        TR_array.append(np.array(TR[TR_idx]))
    TR_flatten = np.hstack(TR_array)

    fontsize_ticks = 22
    fontsize_label = 26
    fontsize_legend = 18
    linewidth = 1.5

    plt.figure()
    plt.xlabel("ATR on benign",fontsize=fontsize_label)
    plt.ylabel("Cumulative probability",fontsize=fontsize_label)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.ylim(0, 1.1)
    plt.grid(True, linestyle='-.')
    plt.tight_layout()
    plt.xlim(0, 1.05)
    bins = 10
    plt.hist(TR_flatten, bins, density=True, histtype='step', cumulative=True, linewidth=linewidth, label="ATR on benign")
    plt.legend(loc='upper right', fontsize=fontsize_legend)
    fig_path = dataset_path + "ATR_pic/cdf/" + "client_num_{}/".format(client_num_in_total)
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    plt.savefig(fig_path + model_name + "_ATR_noise={:.3f}.pdf".format(args.global_noise_scale))
    plt.close()

    weights = np.zeros_like(TR_flatten) + 1. / TR_flatten.size
    plt.figure()
    plt.xlabel("ATR on benign",fontsize=fontsize_label)
    plt.ylabel("Frequency of samples",fontsize=fontsize_label)
    plt.ylim(0, 0.65)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.grid(True, linestyle='-.')
    plt.tight_layout()
    plt.xlim(0, 1.05)
    bins = 10
    plt.hist(TR_flatten, bins, density=False, weights=weights, alpha=0.6, label="ATR on benign")
    plt.legend(loc='upper right',fontsize=fontsize_legend)
    fig_path = dataset_path + "ATR_pic/pdf/" + "client_num_{}/".format(client_num_in_total)
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    plt.savefig(fig_path + model_name + "_ATR_noise={:.3f}.pdf".format(args.global_noise_scale))
    plt.close()


    print(len(AATR_list))
    print(len(files_list))
    if save_log:
        pgd_data=""
        for files,ATR in zip(files_list,AATR_list):
            pgd_data += "Files {}: AATR{}(%)\n".format(
                        files, ATR)
            pgd_data += '################################'
        return pgd_data
    else:
        return files_list, AATR

if __name__ == '__main__':
    args = args_parser()
    prefix = "attack_list"
    # args.dataset = "mnist"
    args.global_noise_scale=0.01
    args.client_num_in_total=3
    save_log=1
    args.model_name='noise_only'
    if args.dataset == "cifar":
        noise_scale = np.linspace(0, 0.12, 25)
    elif args.dataset == "mnist":
        noise_scale = np.linspace(0, 0.32, 33)
    elif args.dataset =='MRI':
         noise_scale=[0.01]
         num_items=100
    elif args.dataset=='Pathology':
         noise_scale=[0.01]
    elif args.dataset=='Glioma':
         noise_scale=[0.01]
         num_items=60

    model_list = ['no_noise_adv', 'noise_adv', 'noise_only']
    # model_list = ['noise_only']
    
    if (args.uplink_noise==True):
        dataset_path = "by_client/uplink_noise/" + args.dataset + "/"
    if (args.uplink_noise==False):
       dataset_path = "by_client/no_uplink_noise/" + args.dataset + "/"
       
    result_path = dataset_path + "ATR_pic/out/"
    test_mkdir(result_path)
    eps=0.020
    eps_step=0.001
    noise_scale=[0.01]
    for args.client_num_in_total in [3]:
        for args.model_name in model_list:
            for loaded_models in [10,0]:
                result_file = result_path + args.dataset + "_client_num_{}".format(args.client_num_in_total) + "_" + args.model_name + ".out"
                file_data = "Noise\tAATR\n"
                for args.global_noise_scale in noise_scale:
                    AATR = atr_per_process()
                    file_data += "{:.3f}\t{:.2f}%\n".format(args.global_noise_scale, AATR * 100)
                    print(file_data)
                    print("dataset = " + args.dataset + ", " + args.model_name + ", num of client = {} , noise = {:.3f} completed!".format(args.client_num_in_total, args.global_noise_scale))
                with open(result_file, "w", encoding="utf-8") as f:
                    f.write(file_data)

    # result_file = result_path + args.dataset + "_client_num_{}".format(args.client_num_in_total) + "_" + args.model_name + ".out"
    # file_data = "Noise\tAATR\n"
    # pgd_data = atr_per_process()
    # # file_data += "{}\t{}%\n".format(str(files_list), AATR * 100)
    # print("dataset = " + args.dataset + ", " + args.model_name + ", num of client = {} , noise = {:.3f} completed!".format(args.client_num_in_total, args.global_noise_scale))
    # with open(result_file, "w", encoding="utf-8") as f:
    #     f.write(pgd_data)