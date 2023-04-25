import copy
import os
import matplotlib.image
import numpy as np
import torch
import wandb
from torchvision.datasets import ImageFolder
from torch import device
from torchvision import datasets, transforms
from armor_py.utils import fix_random, aggregate, noise_add_global,glob_with_grad
from armor_py.models import CNN_MNIST, CNN_CIFAR, CNN_MRI, CNN_Chest,CNN_Tumors,CNN_Pathology, CNN_Pathologym
from armor_py.options import args_parser
from armor_py.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid, MRI_iid, MRI_noniid, chest_iid,chest_noniid,Pathology_iid,Pathology_noniid, Tumors_iid,Tumors_noniid,Glioma_iid,Glioma_noniid
from armor_py.update import LocalUpdate
import gc

gc.collect()

torch.cuda.empty_cache()
matplotlib.use('Agg')
print(torch.cuda.is_available(),torch.cuda.current_device(),torch.cuda.device(0),torch.cuda.device_count(),torch.cuda.get_device_name(0),torch.cuda.get_device_name(1

))

def load_data(args):
    if args.dataset =="MRI":
        transform=transforms.Compose([
        transforms.RandomRotation(10),      # rotate +/- 10 degrees
        transforms.RandomHorizontalFlip(),  # reverse 50% of images
        transforms.Resize(100),             # resize shortest side
        transforms.CenterCrop(100),   
        # transforms.Grayscale(num_output_channels=1),      # crop longest side
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])
        # transforms.Normalize ([ 0.456],
        #                      [ 0.22]) ]  )
        dataset_train = ImageFolder('./dataset/TRAINING-LIGHTG/', transform=transform)

        dataset_test = ImageFolder('./dataset/TEST-LIGHT/', transform=transform) 

    elif args.dataset =="Tumors":
        transform=transforms.Compose([
        transforms.RandomRotation(10),      # rotate +/- 10 degrees
        transforms.RandomHorizontalFlip(),  # reverse 50% of images
        transforms.Resize(60),             # resize shortest side
        transforms.CenterCrop(60),   
        # transforms.Grayscale(num_output_channels=1),      # crop longest side
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])
        # transforms.Normalize ([ 0.456],
        #                      [ 0.22]) ]  )
        dataset_train = ImageFolder('./dataset/Tumors/Training-4class/', transform=transform)

        dataset_test = ImageFolder('./dataset/Tumors/Testing-4class/', transform=transform) 


    elif args.dataset =="Glioma":
        transform=transforms.Compose([
        transforms.RandomRotation(10),      # rotate +/- 10 degrees
        transforms.RandomHorizontalFlip(),  # reverse 50% of images
        transforms.Resize(100),             # resize shortest side
        transforms.CenterCrop(100),   
        # transforms.Grayscale(num_output_channels=1),      # crop longest side
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])
        # transforms.Normalize ([ 0.456],
        #                      [ 0.22]) ]  )
        dataset_train = ImageFolder('./dataset/Training-Glioma/', transform=transform)
        dataset_test = ImageFolder('./dataset/Testing-Glioma/', transform=transform) 


    elif args.dataset =="Pathology":
        transform = transforms.Compose([
                                  transforms.RandomHorizontalFlip(), 
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])
      
        dataset_train = ImageFolder('./dataset/histopathologic-cancer-detection/Training', transform=transform)
        dataset_test = ImageFolder('./dataset/histopathologic-cancer-detection/Testing', transform=transform) 

    elif args.dataset=="chest":
        transform = transforms.Compose([transforms.Resize(80),
                                            transforms.CenterCrop(64),
                                            transforms.Grayscale(num_output_channels=1),
                                            transforms.ToTensor()])
        dataset_train = ImageFolder('./dataset/chest_xray/train/', transform=transform)
        print( torch.unique(dataset_train.targets, return_counts=True))
        # dataset_train = torch.utils.data.Subset(dataset_train1, np.random.choice(len(dataset_train1), 100, replace=False))

        dataset_test = ImageFolder('./dataset/chest_xray/train/', transform=transform)
        print(torch.unique(dataset_test.targets, return_counts=True))

        # dataset_test = torch.utils.data.Subset(dataset_test1, np.random.choice(len(dataset_test1), 20, replace=False))


          
    # load dataset and split users
    elif args.dataset == "mnist":
        transform = transforms.Compose([transforms.ToTensor()])
        dataset_train = datasets.MNIST('./dataset/mnist/', train=True, download=True, transform=transform)
        dataset_test = datasets.MNIST('./dataset/mnist/', train=False, download=True, transform=transform)
    elif args.dataset == "cifar":
        transform = transforms.Compose([transforms.ToTensor()])
        dataset_train = datasets.CIFAR10('./dataset/cifar10/', train=True, download=True, transform=transform)
        dataset_test = datasets.CIFAR10('./dataset/cifar10/', train=False, download=True, transform=transform)
    else:
        exit('Error: unrecognized dataset')
    return dataset_train, dataset_test


def sample_user(args, dataset_train, dataset_test):
    if args.dataset == "mnist":
        dict_train = mnist_noniid(dataset_train, args.client_num_in_total)
        dict_test = mnist_iid(dataset_test, args.client_num_in_total, args.num_items_server)
    elif args.dataset == "cifar" and args.iid == 0:
        dict_train = cifar_noniid(dataset_train, args.client_num_in_total)
        dict_test = cifar_iid(dataset_test, args.client_num_in_total, args.num_items_server)
    elif args.dataset == "chest":
        dict_train = chest_noniid(dataset_train, args.client_num_in_total)
        dict_test = chest_iid(dataset_test, args.client_num_in_total, args.num_items_server)
    elif args.dataset == "MRI":
        dict_train = MRI_noniid(dataset_train, args.client_num_in_total)
        dict_test = MRI_iid(dataset_test, args.client_num_in_total, args.num_items_server)
    elif args.dataset == "Glioma":
        dict_train = Glioma_noniid(dataset_train, args.client_num_in_total)
        dict_test = Glioma_iid(dataset_test, args.client_num_in_total, args.num_items_server)
    elif args.dataset == "Tumors":
        dict_train = Tumors_noniid(dataset_train, args.client_num_in_total)
        dict_test = Tumors_iid(dataset_test, args.client_num_in_total, args.num_items_server)
    elif args.dataset == "Pathology":
        dict_train = Pathology_noniid(dataset_train, args.client_num_in_total)
        dict_test = Pathology_iid(dataset_test, args.client_num_in_total, args.num_items_server)
    else:
        exit('Error: unrecognized dataset')
    return dict_train, dict_test


def create_model(args):
    if args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNN_MNIST()
    elif args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNN_CIFAR()
    elif args.model == 'cnn' and args.dataset == 'MRI':
        net_glob = CNN_MRI()
    elif args.model == 'cnn' and args.dataset == 'chest':
        net_glob = CNN_Chest()
    elif args.model == 'cnn' and args.dataset == 'Glioma':
        net_glob = CNN_MRI()
    elif args.model == 'cnn' and args.dataset == 'Tumors':
        net_glob = CNN_Tumors()
    elif args.model == 'cnn' and args.dataset == 'Pathology':
        net_glob = CNN_Pathology()
    

    else:
        exit('Error: unrecognized model')
    net_glob.to(device)
    return net_glob


def test_noise_local(args, net_noise, dataset_test, dict_server):
    net_local = LocalUpdate(args=args, dataset=dataset_test, idxs=dict_server, device=device)
    acc, loss = net_local.test(net=net_noise, device=device)
    return acc, loss


def local_test_on_all_clients(args, net_glob, dataset_test, dict_server):
    list_acc, list_loss = [], []
    for c in range(args.client_num_in_total):
        net_local = LocalUpdate(args=args, dataset=dataset_test, idxs=dict_server[c], device=device)
        acc, loss = net_local.test(net=net_glob, device=device)
        list_acc.append(acc)
        list_loss.append(loss)
    return list_acc, list_loss


def wandb_init(args):
    if args.global_noise_scale != 0:
        run = wandb.init(reinit=True, project="num of client-" + args.dataset,
                         name="num of client =" + str(args.client_num_in_total) + ",noise={:.3f}".format(
                             args.global_noise_scale) +
                              ",model=" + str(args.model) + ",lr=" + str(args.lr) + ",round=" + str(args.comm_round),
                         config=args)
    else:
        run = wandb.init(reinit=True, project="num of client-" + args.dataset,
                         name="num of client=" + str(args.client_num_in_total) + ",no noise" +
                              ",model=" + str(args.model) + ",lr=" + str(args.lr) + ",round=" + str(args.comm_round),
                         config=args)
    return run


def train(net_glob, dataset_train, dataset_test, dict_users, dict_server):
    # make directory
    if (args.dp==True):
        model_path = "by_client/dp/{}/client_num_{}/".format(args.dataset, args.client_num_in_total)
    if (args.dp==False):
        model_path = "by_client/no_dp/{}/client_num_{}/".format(args.dataset, args.client_num_in_total)

    global_path = model_path + "Global/"
    if not os.path.exists(global_path):
        os.makedirs(global_path)
    final_path = model_path + "Client_final/{:.3f}/".format(args.global_noise_scale)
    if not os.path.exists(final_path):
        os.makedirs(final_path)

    loss_test, loss_train = [], []
    acc_test, acc_train = [], []
    Client = {}
    # copy weights
    w_glob = net_glob.state_dict()
    for idx in range(args.client_num_in_total):
        Client[idx] = w_glob
    # training
    loss_avg_list, acc_avg_list, list_loss, loss_avg,grad_bank = [], [], [], [],[]
    for iter_round in range(args.comm_round):
        print('\n', '*' * 20, 'Communication Round: {}'.format(iter_round), '*' * 20)
        w_locals, loss_locals, acc_locals ,grad_bank= [], [], [],[]
        list_acc_noise, list_loss_noise = [], []
        for idx in range(args.client_num_in_total):
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], device=device)
            if args.global_noise_scale != 0:
                net_noise_train = copy.deepcopy(net_glob)
                #added by me
                net_noise_train.load_state_dict(Client[idx])
                grad, w, loss, acc = local.update_weights(net=copy.deepcopy(net_noise_train), device=device)
            else:
                grad,w, loss, acc = local.update_weights(net=copy.deepcopy(net_glob), device=device)
    
            grad_bank.append(copy.deepcopy(grad))
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
            acc_locals.append(copy.deepcopy(acc))
         

        if (args.dp):
            # Adding uplink noise
            for idx in range(args.client_num_in_total):
                w_locals_i= copy.deepcopy(net_glob)
                w_locals_i.load_state_dict(w_locals[idx])
                local_noise_scale=args.global_noise_scale/args.client_num_in_total
                w_locals_noisy = noise_add_global(args.global_noise_scale, copy.deepcopy(w_locals_i), device=device)
                # w_locals_dpd[idx] = w_locals_noisy
                w_locals[idx] = w_locals_noisy

        # update global weights
        w_glob = aggregate(w_locals)
        #Show grads
        glob_with_grad(w_locals,net_glob,grad_bank)
        # global test
        net_glob.load_state_dict(w_glob)
        list_acc, list_loss = local_test_on_all_clients(args, net_glob, dataset_test, dict_server)

        for idx in range(args.client_num_in_total):
            net_noise = copy.deepcopy(net_glob)
            Client[idx] = noise_add_global(args.global_noise_scale, copy.deepcopy(net_glob), device=device)
            net_noise.load_state_dict(Client[idx] )
            acc_noise, loss_noise = test_noise_local(args, copy.deepcopy(net_noise), dataset_test, dict_server[idx])
            list_acc_noise.append(acc_noise)
            list_loss_noise.append(loss_noise)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        acc_avg = sum(acc_locals) / len(acc_locals)
        loss_avg_list.append(loss_avg)
        acc_avg_list.append(acc_avg)

        print("\nTrain loss: {}, Train acc: {}".format(loss_avg_list[-1], acc_avg_list[-1]))
        print("\nTest loss: {}, Test acc: {}".format(sum(list_loss) / len(list_loss), sum(list_acc) / len(list_acc)))

        if args.global_noise_scale != 0:
            wandb.log({"Server Noise Scale": args.global_noise_scale,
                       "Test/Acc": sum(list_acc) / len(list_acc),
                       "Test/Loss": sum(list_loss) / len(list_loss),
                       "Test_noise/Acc": sum(list_acc_noise) / len(list_acc_noise),
                       "Test_noise/Loss": sum(list_loss_noise) / len(list_loss_noise),
                       "Train/Acc": acc_avg_list[-1],
                       "Train/Loss": loss_avg_list[-1],
                       })
        else:
            wandb.log({"Server Noise Scale": args.global_noise_scale,
                       "Test/Acc": sum(list_acc) / len(list_acc),
                       "Test/Loss": sum(list_loss) / len(list_loss),
                       "Train/Acc": acc_avg_list[-1],
                       "Train/Loss": loss_avg_list[-1],
                       })

        print('\nServer Noise Scale:', args.global_noise_scale)
        loss_train.append(loss_avg)
        acc_train.append(acc_avg)
        loss_test.append(sum(list_loss) / len(list_loss))
        acc_test.append(sum(list_acc) / len(list_acc))

        for idx in range(args.client_num_in_total):
             torch.save(Client[idx], final_path + "Client_final_N{}.pth".format(idx))
        torch.save(w_glob, global_path + "Global_noise_{:.3f}_round_{:.3f}.pth".format(args.global_noise_scale,iter_round))
    for idx in range(args.client_num_in_total):
        torch.save(Client[idx], final_path + "Client_final_{}.pth".format(idx))
    torch.save(w_glob, global_path + "Global_{:.3f}.pth".format(args.global_noise_scale))
    # record results
    final_train_loss = copy.deepcopy(sum(loss_train) / len(loss_train))
    final_train_accuracy = copy.deepcopy(sum(acc_train) / len(acc_train))
    final_test_loss = copy.deepcopy(sum(loss_test) / len(loss_test))
    final_test_accuracy = copy.deepcopy(sum(acc_test) / len(acc_test))

    print('\nFinal train loss:', final_train_loss)
    print('\nFinal train acc:', final_train_accuracy)
    print('\nFinal test loss:', final_test_loss)
    print('\nFinal test acc:', final_test_accuracy)


def main_fed_train(dataset_train, dataset_test, dict_train, dict_test):
    fix_random(args.random_seed)
    print("##############################################################################")
    print("##############################################################################")
    print('Training Model of global noise: {:.3f}'.format(args.global_noise_scale))

    # offline
    # os.environ["WANDB_MODE"] = "offline"
    run = wandb_init(args)

    print("dataset =", args.dataset,
          ", global_noise =", args.global_noise_scale,
          ", num_users =", args.client_num_in_total,
          ", comm_round =", args.comm_round)

    net_glob = create_model(args)
    net_glob.train()
    train(net_glob, dataset_train, dataset_test, dict_train, dict_test)
    run.finish()


if __name__ == '__main__':
    args = args_parser()
    device = torch.device("cuda:{}".format(args.cuda))

    if args.dataset == "cifar":
        args.clip_threshold = 30
        noise_scale = np.arange(0, 0.12, 0.005)
        args.comm_round = 100
        args.random_seed = 1000
    elif args.dataset == "mnist":
        args.clip_threshold = 20
        noise_scale = np.arange(0.0, 0.32, 0.01)
        args.comm_round = 50
        args.random_seed = 50
    elif args.dataset == "MRI":
        args.clip_threshold = 20
        noise_scale = np.arange(0.0, 0.032, 0.01)
        args.comm_round = 100
        args.random_seed = 50
    elif args.dataset == "Pathology":
        args.clip_threshold = 20
        noise_scale = np.arange(0.0, 0.032, 0.01)
        args.comm_round = 50
        args.random_seed = 50
    elif args.dataset == "chest":
        args.clip_threshold = 20
        noise_scale = np.arange(0.0, 0.32, 0.01)
        args.comm_round = 100
        args.random_seed = 50
    elif args.dataset == "Glioma":
        args.clip_threshold = 20
        noise_scale = np.arange(0.0, 0.32, 0.01)
        args.comm_round = 100
        args.random_seed = 50
    elif args.dataset == "Tumors":
        args.clip_threshold = 20
        noise_scale = np.arange(0.0, 0.32, 0.01)
        args.comm_round = 100
        args.random_seed = 50

    fix_random(args.random_seed)
    dataset_train, dataset_test = load_data(args)
    
    dict_train, dict_test = sample_user(args, dataset_train, dataset_test)
    print(dict_train)
    for keys,values in dict_train.items():
            print('keys',keys)
            print('values',values.shape)

            print('Number of Train Items: ', len(values))
    for keys,values in dict_test.items():
            print('Number of test Items', len(values))

    
    main_fed_train(dataset_train, dataset_test, dict_train, dict_test)
    print("dataset = " + args.dataset + ", num of client = {} , noise = {:.3f} completed!".format(args.client_num_in_total, args.global_noise_scale))
