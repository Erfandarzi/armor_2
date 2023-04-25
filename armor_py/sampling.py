import numpy as np
import torch
from easydict import EasyDict
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

from armor_py.options import args_parser
args=args_parser()


def ld_cifar10():
    test_transforms = transforms.ToTensor()
    test_dataset = datasets.CIFAR10('./dataset/cifar10/', train=False, download=True, transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    return EasyDict(test=test_loader)


def ld_mnist():
    test_transforms = transforms.ToTensor()
    test_dataset = datasets.MNIST('./dataset/mnist/', train=False, download=True, transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    return EasyDict(test=test_loader)

def ld_MRI():
    test_transforms = transforms.Compose([
        # transforms.RandomRotation(10),      # rotate +/- 10 degrees
        transforms.RandomHorizontalFlip(),  # reverse 50% of images
        transforms.Resize(100),             # resize shortest side
        transforms.CenterCrop(100), 
        # transforms.Grayscale(num_output_channels=1),        # crop longest side
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])
        # transforms.Normalize ([ 0.456],
        #                      [ 0.22]) ]  )
    test_dataset = ImageFolder('./dataset/TEST-LIGHT/', transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=True)
    return EasyDict(test=test_loader)


def ld_Tumors():
    test_transforms = transforms.Compose([
        transforms.RandomRotation(10),      # rotate +/- 10 degrees
        transforms.RandomHorizontalFlip(),  # reverse 50% of images
        transforms.Resize(60),             # resize shortest side
        transforms.CenterCrop(60), 
        # transforms.Grayscale(num_output_channels=1),        # crop longest side
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])
        # transforms.Normalize ([ 0.456],
        #                      [ 0.22]) ]  )
    
    test_dataset = ImageFolder('./dataset/Tumors/Testing-4class-pit/', transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=600, shuffle=True)
    return EasyDict(test=test_loader)

def ld_GLioma():
    test_transforms = transforms.Compose([
        # transforms.RandomRotation(10),      # rotate +/- 10 degrees
        transforms.RandomHorizontalFlip(),  # reverse 50% of images
        transforms.Resize(100),             # resize shortest side
        transforms.CenterCrop(100), 
        # transforms.Grayscale(num_output_channels=1),        # crop longest side
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])
        # transforms.Normalize ([ 0.456],
        #                      [ 0.22]) ]  )
    test_dataset = ImageFolder('./dataset/Testing-Glioma/', transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=60, shuffle=True)
    return EasyDict(test=test_loader)



def ld_Chest():
    test_transforms = transforms.ToTensor()
    test_dataset = ImageFolder('./dataset/chest_xray/test', train=False, download=True, transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=True)
    return EasyDict(test=test_loader)


def mnist_iid(dataset, num_users, num_items):
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
    return dict_users


def ld_Pathology():
    transform = transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])
    test_dataset = ImageFolder('./dataset/histopathologic-cancer-detection/Testing', transform=transform) 
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=True)
    return EasyDict(test=test_loader)


def MRI_iid(dataset, num_users, num_items):
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=True))
    return dict_users

def Glioma_iid(dataset, num_users, num_items):
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=True))
    return dict_users

def Tumors_iid(dataset, num_users, num_items):
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=True))
    return dict_users


def chest_iid(dataset, num_users, num_items):
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=True))
    print(dict_users[2])
    return dict_users

def Pathology_iid(dataset, num_users, num_items):
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=True)) #uniqie nists
    return dict_users

def chest_noniid(dataset, num_users):
   
    if num_users >= 25:
        num_shards = int(10)
    else: 
        num_shards = int(10)
    
    num_imgs = int(len(dataset)/ num_shards)

    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.targets[:len(idxs)]
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]
    print(idxs)
    if num_users >= 25:
        chosen_shards = int(3)
    else:
        chosen_shards = int(5)
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, chosen_shards, replace=False)) 
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0) #Hatman ghesmataye mokhtalefi az data to entelhab mikone bara har user va unique hast
    return dict_users

def MRI_noniid(dataset, num_users):
    if num_users >= 25:
        num_shards = int(20)
    else:
        num_shards = int(15)

    num_imgs = int(len(dataset)/ num_shards)
    print(len(dataset))
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.targets[:len(idxs)]
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]
    if num_users >= 25:
        chosen_shards = int(3)
    else:
        chosen_shards = int(5)
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, chosen_shards, replace=False))
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

def Tumors_noniid(dataset, num_users):
    if num_users >= 25:
        num_shards = int(20)
    else:
        num_shards = int(15)

    num_imgs = int(len(dataset)/ num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.targets[:len(idxs)]
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]
    if num_users >= 25:
        chosen_shards = int(3)
    else:
        chosen_shards = int(5)
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, chosen_shards, replace=False))
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

def Glioma_noniid(dataset, num_users):
    if num_users >= 25:
        num_shards = int(20)
    else:
        num_shards = int(15)

    num_imgs = int(len(dataset)/ num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.targets[:len(idxs)]
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]
    if num_users >= 25:
        chosen_shards = int(3)
    else:
        chosen_shards = int(5)
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, chosen_shards, replace=False))
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

def Pathology_noniid(dataset, num_users):
    if num_users >= 25:
        num_shards = int(20)
    else:
        num_shards = int(220)

    num_imgs = int(len(dataset)/ num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.targets[:len(idxs)]
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]
    if num_users >= 25:
        chosen_shards = int(3)
    else:
        chosen_shards = int(5)
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, chosen_shards, replace=False))
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def mnist_noniid(dataset, num_users):
    if num_users >= 25:
        num_shards = int(300)
    else:
        num_shards = int(200)
    num_imgs = int(dataset.data.shape[0] / num_shards)

    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.targets.numpy()
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]
    print(idxs)
    if num_users >= 25:
        chosen_shards = int(3)
    else:
        chosen_shards = int(5)
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, chosen_shards, replace=False))
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def cifar_iid(dataset, num_users, num_items):
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
    return dict_users


def cifar_noniid(dataset, num_users):
    print(dataset)
    if num_users >= 25:
        num_shards = int(200)
    else:
        num_shards = int(100)
    num_imgs = int(dataset.data.shape[0] / num_shards)

    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = np.array(dataset.targets)
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    if num_users >= 25:
        chosen_shards = int(4)
    else:
        chosen_shards = int(6)
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, chosen_shards, replace=False))
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users