from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
from torchvision.models import resnet18
from torchcam.methods import CAM


from torchcam.methods import SmoothGradCAMpp
from torchsummary import summary
import matplotlib.pyplot as plt
import torchattacks
from skimage.metrics import structural_similarity as ssim

import torch
from torchvision import transforms
from PIL import Image
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

import os
import torch
import torchattacks
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet50
import matplotlib.pyplot as plt
import numpy as np

target_layers=[-1]
np.set_printoptions(threshold=np.inf)
 
matplotlib.use('TkAgg')

def load_images(image_folder):
    image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder)]
    return image_paths

def transform_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(100),
        transforms.CenterCrop(100),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor
def apply_attack(model, image_tensor, labels, eps, steps):
    atk = torchattacks.PGD(model, eps=eps, alpha=0.01, steps=steps)
    image_tensor = image_tensor.clamp(0, 1)  # Clip the tensor values to the range [0, 1]
    adv_images = atk(image_tensor, labels)
    return adv_images
def tensor_to_image(tensor):
    img_np = tensor.squeeze(0).cpu().detach().numpy().transpose((1, 2, 0))  # Add .cpu() before .detach()
    img_np = (img_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
    img_np = np.clip(img_np, 0, 1)
    return img_np
def calculate_mse(original_image, adversarial_image):
    mse = np.mean((original_image - adversarial_image) ** 2)
    return mse


def main():
    fix_random(1)
    if (args.dp==True):
       path = "by_client/dp/" + args.dataset + "/client_num_{}/".format(args.client_num_in_total)
    if (args.dp==False):
       path = "by_client/no_dp/" + args.dataset + "/client_num_{}/".format(args.client_num_in_total)
    net = []
    idx = 0
    net.append(copy.deepcopy(net_glob))
    image_folders = ["dataset\Tumors\Testing-4class\meningioma_tumor",
                     "dataset\Testing-Glioma\glioma_tumor",
                     "dataset\histopathologic-cancer-detection\Testing\Cancer"]
    ssim_lists = []
    mse_lists = []
    for image_folder in image_folders:
        image_paths = load_images(image_folder)
        ssim_list = []
        mse_list = []
        file_path = path + prefix + "/{:.3f}/".format(args.global_noise_scale) + prefix + "_N{}.pth".format(idx)
        net[idx].load_state_dict(torch.load(file_path))
        model = net[idx].eval()
        labels = torch.tensor([0])
        epsilons = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2]

        for image_path in image_paths[:5]:  # Process only the first 5 images for demonstration purposes
            original_image_tensor = transform_image(image_path)
            original_image = tensor_to_image(original_image_tensor)

            ssim_per_eps = []
            mse_per_eps = []

            for eps in epsilons:
                adv_images = apply_attack(model, original_image_tensor, labels, eps,10)
                adv_image_np = tensor_to_image(adv_images)
                s = ssim(original_image, adv_image_np, multichannel=True)
                mse = calculate_mse(original_image, adv_image_np)
                ssim_per_eps.append(s)
                mse_per_eps.append(mse)
                print("SSIM for eps={}: {}".format(eps, s))
                print("MSE for eps={}: {}".format(eps, mse))

            ssim_list.append(ssim_per_eps)
            mse_list.append(mse_per_eps)

        # Calculate the average of each epsilon's SSIM and MSE values across all images in the dataset
        ssim_list = np.mean(ssim_list, axis=0)
        mse_list = np.mean(mse_list, axis=0)
        ssim_lists.append(ssim_list)
        mse_lists.append(mse_list)

    # Plot the SSIM and MSE vs Epsilon for all three datasets
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    dataset_names = ['Meningioma', 'Glioma', 'Pathology']

    for i, (ssim_list, mse_list) in enumerate(zip(ssim_lists, mse_lists)):
        ax1.plot(epsilons, ssim_list, marker='o', label=dataset_names[i])
        ax2.plot(epsilons, mse_list, marker='o', label=dataset_names[i])

    ax1.set_xlabel('Epsilon')
    ax1.set_ylabel('Structural Similarity Index (SSIM)')
    ax1.set_title('SSIM vs Epsilon')
    ax1.legend()
    ax1.grid()

    ax2.set_xlabel('Epsilon')
    ax2.set_ylabel('Mean Squared Error (MSE)')
    ax2.set_title('MSE vs Epsilon')
    ax2.legend()
    ax2.grid()

    fig.savefig('plots/ssim_mse_vs_epsilon.pdf')
    fig.savefig('plots/ssim_mse_vs_epsilon.png')

    ssim_lists = []
    mse_lists = []

    for image_folder in image_folders:
        image_paths = load_images(image_folder)
        ssim_list = []
        mse_list = []
        file_path = path + prefix + "/{:.3f}/".format(args.global_noise_scale) + prefix + "_N{}.pth".format(idx)
        net[idx].load_state_dict(torch.load(file_path))
        model = net[idx].eval()
        labels = torch.tensor([0])
        eps = 0.1
        steps_list = [1, 2, 3,4, 5,6, 10, 20]

        for image_path in image_paths[:5]:  # Process only the first 5 images for demonstration purposes
            original_image_tensor = transform_image(image_path)
            original_image = tensor_to_image(original_image_tensor)

            ssim_per_steps = []
            mse_per_steps = []

            for steps in steps_list:
                adv_images = apply_attack(model, original_image_tensor, labels, eps, steps)
                adv_image_np = tensor_to_image(adv_images)
                s = ssim(original_image, adv_image_np, multichannel=True)
                mse = calculate_mse(original_image, adv_image_np)
                ssim_per_steps.append(s)
                mse_per_steps.append(mse)
                print("SSIM for steps={}: {}".format(steps, s))
                print("MSE for steps={}: {}".format(steps, mse))

            ssim_list.append(ssim_per_steps)
            mse_list.append(mse_per_steps)

        # Calculate the average of each step's SSIM and MSE values across all images in the dataset
        ssim_list = np.mean(ssim_list, axis=0)
        mse_list = np.mean(mse_list, axis=0)
        ssim_lists.append(ssim_list)
        mse_lists.append(mse_list)

    # Plot the SSIM and MSE vs Steps for all three datasets
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    dataset_names = ['Meningioma', 'Glioma', 'Pathology']
    markers = ['o', 's', 'd']
    colors = ['C0', 'C1', 'C2']

    for i, (ssim_list, mse_list) in enumerate(zip(ssim_lists, mse_lists)):
        ax1.plot(steps_list, ssim_list, marker=markers[i], markersize=8, color=colors[i], linestyle='-', linewidth=2, label=f'{dataset_names[i]}')
        ax2.plot(steps_list, mse_list, marker=markers[i], markersize=8, color=colors[i], linestyle='-', linewidth=2, label=f'{dataset_names[i]}')

    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Structural Similarity Index (SSIM)')
    ax1.set_title('SSIM vs Steps')
    ax1.legend()
    ax1.grid()

    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Mean Squared Error (MSE)')
    ax2.set_title('MSE vs Steps')
    ax2.legend()
    ax2.grid()

    fig.savefig('plots/ssim_mse_vs_steps.pdf')
    fig.savefig('plots/ssim_mse_vs_steps.png')

 

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
   
    main()
    print("dataset = " + args.dataset + ", num of client = {} , noise = {:.3f} completed!".format(args.client_num_in_total, args.global_noise_scale))
