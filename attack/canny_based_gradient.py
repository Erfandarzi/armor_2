"""The  GC Fast Gradient Method attack."""
import numpy as np
import torch
from attack.utils import optimize_linear

#from torch.optim.optimizer import Optimizer, required
import torch
import torch.nn as nn
import numpy as np
from scipy.signal import gaussian


class Net(nn.Module):
    def __init__(self, threshold=10.0, use_cuda=False):
        super(Net, self).__init__()

        self.threshold = threshold
        self.use_cuda = use_cuda

        filter_size = 5
        generated_filters = gaussian(filter_size,std=1.0).reshape([1,filter_size])

        self.gaussian_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,filter_size), padding=(0,filter_size//2))
        self.gaussian_filter_horizontal.weight.data.copy_(torch.from_numpy(generated_filters))
        self.gaussian_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.gaussian_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(filter_size,1), padding=(filter_size//2,0))
        self.gaussian_filter_vertical.weight.data.copy_(torch.from_numpy(generated_filters.T))
        self.gaussian_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])

        self.sobel_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
        self.sobel_filter_horizontal.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.sobel_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.sobel_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
        self.sobel_filter_vertical.weight.data.copy_(torch.from_numpy(sobel_filter.T))
        self.sobel_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        # filters were flipped manually
        filter_0 = np.array([   [ 0, 0, 0],
                                [ 0, 1, -1],
                                [ 0, 0, 0]])

        filter_45 = np.array([  [0, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, -1]])

        filter_90 = np.array([  [ 0, 0, 0],
                                [ 0, 1, 0],
                                [ 0,-1, 0]])

        filter_135 = np.array([ [ 0, 0, 0],
                                [ 0, 1, 0],
                                [-1, 0, 0]])

        filter_180 = np.array([ [ 0, 0, 0],
                                [-1, 1, 0],
                                [ 0, 0, 0]])

        filter_225 = np.array([ [-1, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        filter_270 = np.array([ [ 0,-1, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        filter_315 = np.array([ [ 0, 0, -1],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        all_filters = np.stack([filter_0, filter_45, filter_90, filter_135, filter_180, filter_225, filter_270, filter_315])

        self.directional_filter = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=filter_0.shape, padding=filter_0.shape[-1] // 2)
        self.directional_filter.weight.data.copy_(torch.from_numpy(all_filters[:, None, ...]))
        self.directional_filter.bias.data.copy_(torch.from_numpy(np.zeros(shape=(all_filters.shape[0],))))

    def forward(self, img):
        img_r = img[:,0:1]
        img_g = img[:,1:2]
        img_b = img[:,2:3]

        blur_horizontal = self.gaussian_filter_horizontal(img_r)
        blurred_img_r = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_g)
        blurred_img_g = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_b)
        blurred_img_b = self.gaussian_filter_vertical(blur_horizontal)

        blurred_img = torch.stack([blurred_img_r,blurred_img_g,blurred_img_b],dim=1)
        blurred_img = torch.stack([torch.squeeze(blurred_img)])

        grad_x_r = self.sobel_filter_horizontal(blurred_img_r)
        grad_y_r = self.sobel_filter_vertical(blurred_img_r)
        grad_x_g = self.sobel_filter_horizontal(blurred_img_g)
        grad_y_g = self.sobel_filter_vertical(blurred_img_g)
        grad_x_b = self.sobel_filter_horizontal(blurred_img_b)
        grad_y_b = self.sobel_filter_vertical(blurred_img_b)

        # COMPUTE THICK EDGES

        grad_mag = torch.sqrt(grad_x_r**2 + grad_y_r**2)
        grad_mag += torch.sqrt(grad_x_g**2 + grad_y_g**2)
        grad_mag += torch.sqrt(grad_x_b**2 + grad_y_b**2)
        grad_orientation = (torch.atan2(grad_y_r+grad_y_g+grad_y_b, grad_x_r+grad_x_g+grad_x_b) * (180.0/3.14159))
        grad_orientation += 180.0
        grad_orientation =  torch.round( grad_orientation / 45.0 ) * 45.0

        # THIN EDGES (NON-MAX SUPPRESSION)

        all_filtered = self.directional_filter(grad_mag)

        inidices_positive = (grad_orientation / 45) % 8
        inidices_negative = ((grad_orientation / 45) + 4) % 8

        height = inidices_positive.size()[2]
        width = inidices_positive.size()[3]
        pixel_count = height * width
        pixel_range = torch.FloatTensor([range(pixel_count)])
        if self.use_cuda:
            pixel_range = torch.cuda.FloatTensor([range(pixel_count)])

        indices = (inidices_positive.view(-1).data * pixel_count + pixel_range).squeeze()
        channel_select_filtered_positive = all_filtered.view(-1)[indices.long()].view(1,height,width)

        indices = (inidices_negative.view(-1).data * pixel_count + pixel_range).squeeze()
        channel_select_filtered_negative = all_filtered.view(-1)[indices.long()].view(1,height,width)

        channel_select_filtered = torch.stack([channel_select_filtered_positive,channel_select_filtered_negative])

        is_max = channel_select_filtered.min(dim=0)[0] > 0.0
        is_max = torch.unsqueeze(is_max, dim=0)

        thin_edges = grad_mag.clone()
        thin_edges[is_max==0] = 0.0

        # THRESHOLD

        thresholded = thin_edges.clone()
        thresholded[thin_edges<self.threshold] = 0.0

        early_threshold = grad_mag.clone()
        early_threshold[grad_mag<self.threshold] = 0.0

        assert grad_mag.size() == grad_orientation.size() == thin_edges.size() == thresholded.size() == early_threshold.size()

        return blurred_img, grad_mag, grad_orientation, thin_edges, thresholded, early_threshold


def canny(raw_img, use_cuda=true):
    img = raw_img
    print(image.size())
    batch = torch.stack([img]).float()
    print(image.size())
    net = Net(threshold=3.0, use_cuda=use_cuda)
    if use_cuda:
        net.cuda()
    net.eval()

    data = Variable(batch)
    if use_cuda:
        data = Variable(batch).cuda()

    blurred_img, grad_mag, grad_orientation, thin_edges, thresholded, early_threshold = net(data)

    print(threshold)



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

def Canny_fast_gradient_method(
    model_fn,
    x,
    eps,
    norm,
    clip_min=None,
    clip_max=None,
    y=None,
    targeted=False,
    sanity_checks=False,
    use_gc=True
):
    """
    PyTorch implementation of the Fast Gradient Method.
    :param model_fn: a callable that takes an input tensor and returns the model logits.
    :param x: input tensor.
    :param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572.
    :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
    :param clip_min: (optional) float. Minimum float value for adversarial example components.
    :param clip_max: (optional) float. Maximum float value for adversarial example components.
    :param y: (optional) Tensor with true labels. If targeted is true, then provide the
              target label. Otherwise, only provide this parameter if you'd like to use true
              labels when crafting adversarial samples. Otherwise, model predictions are used
              as labels to avoid the "label leaking" effect (explained in this paper:
              https://arxiv.org/abs/1611.01236). Default is None.
    :param targeted: (optional) bool. Is the attack targeted or untargeted?
              Untargeted, the default, will try to make the label incorrect.
              Targeted will instead try to move in the direction of being more like y.
    :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
              memory or for unit tests that intentionally pass strange input)
    :return: a tensor for the adversarial example
    """
    if norm not in [np.inf, 1, 2]:
        raise ValueError(
            "Norm order must be either np.inf, 1, or 2, got {} instead.".format(norm)
        )
    if eps < 0:
        raise ValueError(
            "eps must be greater than or equal to 0, got {} instead".format(eps)
        )
    if eps == 0:
        return x
    if clip_min is not None and clip_max is not None:
        if clip_min > clip_max:
            raise ValueError(
                "clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}".format(
                    clip_min, clip_max
                )
            )

    asserts = []

    # If a data range was specified, check that the input was in that range
    if clip_min is not None:
        assert_ge = torch.all(
            torch.ge(x, torch.tensor(clip_min, device=x.device, dtype=x.dtype))
        )
        asserts.append(assert_ge)

    if clip_max is not None:
        assert_le = torch.all(
            torch.le(x, torch.tensor(clip_max, device=x.device, dtype=x.dtype))
        )
        asserts.append(assert_le)

    # x needs to be a leaf variable, of floating point type and have requires_grad being True for
    # its grad to be computed and stored properly in a backward call
    x = x.clone().detach().to(torch.float).requires_grad_(True)
    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        _, y = torch.max(model_fn(x), 1)

    # Compute loss
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(model_fn(x), y)
    # If attack is targeted, minimize loss of target label rather than maximize loss of correct label
    if targeted:
        loss = -loss

    # Define gradient of loss wrt input
    loss.backward()
    
    #GC operation  
    # print(x.shape)   
    uniform_noise = torch.rand(x.shape,device=x.device)
    uniform_noise=2*eps*uniform_noise-eps

    # 
    # x_neighbor = x + uniform_noise
    x_neighbor=x
    # print(x[10].size())
    canyy=canny(x)
    d_p =centralized_gradient(x_neighbor.grad)
 
    optimal_perturbation = optimize_linear(d_p, eps, norm)

    # Add perturbation to original example to obtain adversarial example
    adv_x = x + optimal_perturbation

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) or (clip_max is not None):
        if clip_min is None or clip_max is None:
            raise ValueError(
                "One of clip_min and clip_max is None but we don't currently support one-sided clipping"
            )
        adv_x = torch.clamp(adv_x, clip_min, clip_max)

    if sanity_checks:
        assert np.all(asserts)
    return adv_x
