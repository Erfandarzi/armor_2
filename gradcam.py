from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
from torchvision.models import resnet18
from torchcam.methods import CAM


from torchcam.methods import SmoothGradCAMpp
from torchsummary import summary
import matplotlib.pyplot as plt

import copy
import os
import matplotlib.image
import numpy as np
import torch
from easydict import EasyDict
from attack.projected_gradient_descent import projected_gradient_descent
from armor_py.utils import del_tensor_element, fix_random
from armor_py.models import CNN_CIFAR, CNN_MNIST, CNN_MRI,CNN_Tumors,CNN_Pathology,CNN_Pathologym
from armor_py.options import args_parser
from armor_py.sampling import ld_cifar10, ld_mnist,ld_MRI,ld_GLioma,ld_Tumors,ld_Pathology
np.set_printoptions(threshold=np.inf)
matplotlib.use('Agg')

target_layers=[-1]
def gradcam():
    fix_random(1)
    if (args.dp==True):
       path = "by_client/dp/" + args.dataset + "/client_num_{}/".format(args.client_num_in_total)
    if (args.dp==False):
       path = "by_client/no_dp/" + args.dataset + "/client_num_{}/".format(args.client_num_in_total)

    # pgd_path = path + "Attack_fixed/Client_noise_only/attack_log/"
    # if not os.path.exists(pgd_path):
    #     os.makedirs(pgd_path)

    # attack_list_path = path + "Attack_fixed/Client_noise_only/attack_list/"
    # if not os.path.exists(attack_list_path):
    #     os.makedirs(attack_list_path)

    # pgd_file = pgd_path + "pgd_{:.3f}_eps_{:.3f}_step_{:.3f}_projected_{:.3f}_iterrround_{:.3f}".format(args.global_noise_scale,eps,eps_step,rand_init,iter_round) + ".out"
    # attack_list_file= attack_list_path + "pgd_{:.3f}_eps_{:.3f}_step_{:.3f}_projected_{:.3f}_iterrround_{:.3f}".format(args.global_noise_scale,eps,eps_step,rand_init,iter_round) + ".out"

    # attack_list_file = attack_list_path + "attack_list_{:.3f}".format(args.global_noise_scale) + ".out"

    # pgd_data = "Model Path: " + path + prefix + "\n"
    # list_data = "Model Path: " + path + prefix + "\n"

    # ######################### Attack begin #########################
    # pgd_data += "noise_scale={:.3f}, eps={:.3f}, eps_step={:.3f}, iter_round={}\n".\
    #     format(args.global_noise_scale, eps, eps_step, iter_round)
    # list_data += "noise_scale={:.3f}, eps={:.3f}, eps_step={:.3f}, iter_round={}\n".\
    #     format(args.global_noise_scale, eps, eps_step, iter_round)

    # pgd_data += "################################ Attack begin ################################\n"
    # list_data += "################################ Attack begin ################################\n"
    net = []
    for idx in range(args.client_num_in_total):
        net.append(copy.deepcopy(net_glob))

        file_path = path + prefix + "/{:.3f}/".format(args.global_noise_scale) + prefix + "_N{}.pth".format(idx)
        net[idx].load_state_dict(torch.load(file_path))
        # net[idx].eval()
        model = net[idx].eval()
        # target_layers = [model.network[4]]
    for corrupted_idx in range(args.client_num_in_total):
        fix_random(2)
        # pgd_data += "##############################################################################\n"
        # pgd_data += "Adversary Examples Generated on Client {}\n".format(corrupted_idx)

        # list_data += "##############################################################################\n"
        # list_data += "Adversary Examples Generated on Client {}\n".format(corrupted_idx)

        test_round = 0
        for x, y in data.test:
            if test_round >= 1:
                break
            x = x.to(device)
            y = y.to(device)
            # x_pgd = projected_gradient_descent(net[corrupted_idx], x, eps, eps_step, iter_round, np.inf,rand_init)

            for idx in range(args.client_num_in_total):
                report = EasyDict(nb_test=0, nb_correct=0, correct_pgd_predict=0, correct_pgd_in_corrected=0)
                # _, y_pred = net[idx](x).max(1)
                # _, y_pred_pgd = net[idx](x_pgd).max(1)
                report.nb_test += y.size(0)
                # report.nb_correct += y_pred.eq(y).sum().item()
                print(summary(model, (3, 96, 96)))

                cam_extractor = SmoothGradCAMpp(net[idx])
                print(10*'*'+'passed!'+10*'*')
                                
                # # Get your input
                # img = read_image("path/to/your/image.png")
                # # Preprocess it for your chosen model
                # input_tensor = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

                # Preprocess your data and feed it to the model
                out = model(x[0])
                # Retrieve the CAM by passing the class index and the model output
                activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
                # If you want to visualize your heatmap, you only need to cast the CAM to a numpy ndarray:

                # Visualize the raw CAM
                plt.imshow(activation_map[0].squeeze(0).numpy()); plt.axis('off'); plt.tight_layout(); plt.show()
                # 0 predict incorrectly
                # 1 predict correctly & attack failed
                # # 2 predict correctly & attack succeed
                # input_tensor=x
                # cam = CAM(model, 'network.4')
                # with torch.no_grad(): out = model(input_tensor)
                # cam(class_idx=0)
             
                input_tensor = x[[0,1],:,:,:]# Create an input tensor image for your model..
# # Note: input_tensor can be a batch tensor with several images!
                print(input_tensor.size())
# Construct the CAM object once, and then re-use it on many images:
                cam = GradCAM(model=net[idx], target_layers=target_layers, use_cuda=True)

# You can also use it within a with statement, to make sure it is freed,
# In case you need to re-create it inside an outer loop:
# with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
#   ...

# We have to specify the target we want to generate
# the Class Activation Maps for.
# If targets is None, the highest scoring category
# will be used for every image in the batch.
# Here we use ClassifierOutputTarget, but you can define your own custom targets
# That are, for example, combinations of categories, or specific outputs in a non standard model.
               
                targets = [y]
                target_category=[y]
# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
                # grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
                grayscale_cam = cam(input_tensor=input_tensor)

# In this example grayscale_cam has only one image in the batch:
                grayscale_cam = grayscale_cam[0, :]
                rgb_img=x[1].cpu().numpy() 
                visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

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
        data = ld_Pathology()
        net_glob = CNN_Pathologym()
        net_glob.to(device)
        net_glob.eval()
        args.random_seed = 50
    print("dataset = " + args.dataset + ", num of client = {} , noise = {:.3f} begins...".format(args.client_num_in_total, args.global_noise_scale))
   
    gradcam()
    print("dataset = " + args.dataset + ", num of client = {} , noise = {:.3f} completed!".format(args.client_num_in_total, args.global_noise_scale))
