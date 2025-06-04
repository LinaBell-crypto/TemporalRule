import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import numpy as np

THRESHOLD = 0.5
INIT_RANGE = 0.5
EPSILON1 = 1e-10
EPSILON2 = 1e-3  # or 1e-2


# discrete weight
class Binarize(torch.autograd.Function):
    """Deterministic binarization."""
    
    @staticmethod
    def forward(ctx, X):
        y = torch.where(X >= 0, torch.ones_like(X), torch.zeros_like(X))
        return y

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class Enforced(torch.autograd.Function):
    """Deterministic binarization."""
    
    @staticmethod
    def forward(ctx, X):
        y = torch.where(X > EPSILON2, X, torch.full_like(X, EPSILON2))
        return y

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class BinarizeLayer(nn.Module):
    def __init__(self, input_dim, bin_t, time_series_length, i_min, i_max, interval, use_not=False):
        super(BinarizeLayer, self).__init__()
        self.interval_number = len(interval[0])
        self.input_dim = input_dim
        self.use_not = use_not
        self.output_dim = self.interval_number * self.input_dim
        self.i_min = i_min
        self.i_max = i_max
        self.layer_type = 'binarization'
        self.dim2id = {i: i for i in range(int(self.output_dim))}
        self.interval = nn.Parameter(torch.Tensor(interval), requires_grad=True)
        self.bin_t = bin_t
        self.time_series_length = time_series_length

        self.t1 = 1000
        self.t2 = 1000

    def forward(self, x):
        interval_pos = Enforced.apply(self.interval)
        kmeans_loss = None

        for fea in range(x.shape[1]):
            one_feature_x = x[:, fea]
            if fea < self.time_series_length * 3:
                one_feature_interval = interval_pos[fea // self.bin_t]
            else:
                one_feature_interval = interval_pos[
                    fea + self.time_series_length * 3 // self.bin_t - self.time_series_length * 3
                ]
            
            start = self.i_min[fea]
            for val in range(self.interval_number):
                if val == 0:
                    start += one_feature_interval[val]
                    Distance = (one_feature_x - start) ** 2
                    Distance = Distance.unsqueeze(-1)
                else:
                    start += one_feature_interval[val]
                    dis = (one_feature_x - start) ** 2
                    dis = dis.unsqueeze(-1)
                    Distance = torch.cat((Distance, dis), dim=1)
            
            Dis_min = torch.min(Distance, dim=1)[0].unsqueeze(-1)
            Dis_exp = torch.exp(-self.t1 * (Distance - Dis_min))
            Dis_exp_sum = torch.sum(Dis_exp, dim=1).unsqueeze(-1)
            Dis_softmax = Dis_exp / Dis_exp_sum
            loss = Distance * Dis_softmax
            
            if fea == 0:
                kmeans_loss = torch.mean(torch.sum(loss, dim=1), dim=0)
            else:
                kmeans_loss += torch.mean(torch.sum(loss, dim=1), dim=0)

            X_exp = torch.exp(-self.t2 * (Distance - Dis_min))
            X_exp_sum = torch.sum(X_exp, dim=1).unsqueeze(-1)
            X_softmax = X_exp / X_exp_sum
            X_argmax = torch.argmax(X_softmax, dim=1)
            X_b = F.one_hot(X_argmax, num_classes=self.interval_number)
            out = X_b.detach() + X_softmax - X_softmax.detach()
            
            if fea == 0:
                total_out = out
            else:
                total_out = torch.cat((total_out, out), dim=1)
                
        return total_out, kmeans_loss

    def binarized_forward(self, x):
        with torch.no_grad():
            return self.forward(x)

    def clip(self):
        pass

    def get_bound_name(self, feature_name, mean=None, std=None):
        bound_name = []
        
        if self.input_dim > 0:
            interval = torch.where(
                self.interval > EPSILON2, 
                self.interval,
                torch.full_like(self.interval, EPSILON2)
            )
            
            interval = interval.detach().cpu().numpy()
            for i, fi_name in enumerate(feature_name):
                if i < self.time_series_length * 3:
                    ii = interval[i // self.bin_t]
                    fi_name_index = i // self.time_series_length
                else: 
                    ii = interval[
                        i + self.time_series_length * 3 // self.bin_t - self.time_series_length * 3
                    ]
                    fi_name_index = None
                    
                mini = self.i_min[i]
                maxi = self.i_max[i]
                interval_list = []
                
                for j in range(len(ii)):
                    if j == 0:
                        cl = mini
                        cr = cl + ii[j] + 1 / 2 * ii[j + 1]
                        interval_list.append((cl, cr))
                    elif j == len(ii) - 1:
                        cl = cr
                        cr = maxi
                        interval_list.append((cl, cr))
                    else:
                        cl = cr
                        cr += 1/2 * (ii[j] + ii[j + 1])
                        interval_list.append((cl, cr))
                        
                for cl, cr in interval_list:
                    if mean is not None and std is not None and fi_name_index is not None:
                        cl = cl * std[fi_name_index] + mean[fi_name_index]
                        cr = cr * std[fi_name_index] + mean[fi_name_index]
                    bound_name.append('{:.3f} < {} < {:.3f}'.format(cl, fi_name, cr))
                    
        return bound_name


class Product(torch.autograd.Function):
    """Tensor product function."""
    
    @staticmethod
    def forward(ctx, X):
        y = (-1. / (-1. + torch.sum(torch.log(X), dim=1)))
        ctx.save_for_backward(X, y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        X, y, = ctx.saved_tensors
        grad_input = grad_output.unsqueeze(1) * (y.unsqueeze(1) ** 2 / (X + EPSILON1))
        return grad_input


class EstimatedProduct(torch.autograd.Function):
    """Tensor product function with a estimated derivative."""
    
    @staticmethod
    def forward(ctx, X):
        y = (-1. / (-1. + torch.sum(torch.log(X), dim=1)))
        ctx.save_for_backward(X, y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        X, y, = ctx.saved_tensors
        grad_input = grad_output.unsqueeze(1) * (
            (-1. / (-1. + torch.log(y.unsqueeze(1) ** 2))) / (X + EPSILON1)
        )
        return grad_input


class LRLayer(nn.Module):
    """The LR layer is used to learn the linear part of the data."""

    def __init__(self, n, input_dim):
        super(LRLayer, self).__init__()
        self.n = n
        self.input_dim = input_dim
        self.output_dim = self.n
        self.layer_type = 'linear'
        self.fc1 = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x):
        return self.fc1(x)

    def binarized_forward(self, x):
        return self.forward(x)

    def clip(self):
        for param in self.fc1.parameters():
            param.data.clamp_(-1.0, 1.0)


class ConjunctionLayer(nn.Module):
    """The conjunction layer is used to learn the conjunction of nodes."""

    def __init__(self, n, input_dim, use_not=False, estimated_grad=False):
        super(ConjunctionLayer, self).__init__()
        self.n = n
        self.use_not = use_not
        self.input_dim = input_dim if not use_not else input_dim * 2
        self.output_dim = self.n
        self.layer_type = 'conjunction'

        self.W = nn.Parameter(INIT_RANGE * torch.rand(self.n, self.input_dim))
        self.Product = EstimatedProduct if estimated_grad else Product
        self.node_activation_cnt = None

    def forward(self, x):
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        return self.Product.apply(1 - (1 - x).unsqueeze(-1) * self.W.t())

    def binarized_forward(self, x):
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        Wb = Binarize.apply(self.W - THRESHOLD)
        return torch.prod(1 - (1 - x).unsqueeze(-1) * Wb.t(), dim=1)

    def clip(self):
        self.W.data.clamp_(0.0, 1.0)


class DisjunctionLayer(nn.Module):
    """The disjunction layer is used to learn the disjunction of nodes."""

    def __init__(self, n, input_dim, use_not=False, estimated_grad=False):
        super(DisjunctionLayer, self).__init__()
        self.n = n
        self.use_not = use_not
        self.input_dim = input_dim if not use_not else input_dim * 2
        self.output_dim = self.n
        self.layer_type = 'disjunction'

        self.W = nn.Parameter(INIT_RANGE * torch.rand(self.n, self.input_dim))
        self.Product = EstimatedProduct if estimated_grad else Product
        self.node_activation_cnt = None

    def forward(self, x):
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        return 1 - self.Product.apply(1 - x.unsqueeze(-1) * self.W.t())

    def binarized_forward(self, x):
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        Wb = Binarize.apply(self.W - THRESHOLD)
        return 1 - torch.prod(1 - x.unsqueeze(-1) * Wb.t(), dim=1)

    def clip(self):
        self.W.data.clamp_(0.0, 1.0)


class AlwaysLayer(nn.Module):
    """The always layer is used to learn the always of nodes."""

    def __init__(self, n, input_dim, bin1, bin_t, use_not=False, estimated_grad=False):
        super(AlwaysLayer, self).__init__()
        self.n = n
        self.use_not = use_not
        self.input_dim = input_dim if not use_not else input_dim * 2
        self.output_dim = int(n * self.input_dim / bin_t)
        self.layer_type = 'Always'
        self.bin1 = bin1
        self.bin_t = bin_t

        start_parameters = (bin_t - 1) * torch.rand(self.output_dim)
        end_parameters = (bin_t - 1) * torch.rand(self.output_dim)
        self.start_parameters_list = nn.ParameterList([nn.Parameter(p) for p in start_parameters])
        self.end_parameters_list = nn.ParameterList([nn.Parameter(p) for p in end_parameters])
        self.Product = EstimatedProduct if estimated_grad else Product
        self.node_activation_cnt = None

    def forward(self, x):
        x_output = []
        l = torch.arange(self.bin_t).to(x.device)
        
        for k, i in enumerate(range(0, x.shape[1], self.bin1 * self.bin_t)): 
            x_part = x[:, i:i + self.bin1 * self.bin_t]
            for j in range(self.bin1):
                x_input = x_part[:, j::self.bin1]
                for num in range(self.n):
                    W_start = self.start_parameters_list[k * self.bin1 * self.n + j * self.n + num]
                    W_end = self.end_parameters_list[k * self.bin1 * self.n + j * self.n + num]
                    Weight = torch.sigmoid((l - W_start) * (W_end - l)).unsqueeze(0)
                    x_output.append(
                        self.Product.apply(1 - (1 - x_input).unsqueeze(-1) * Weight.t())
                    )
        
        return torch.cat(x_output, dim=1)

    def binarized_forward(self, x):
        x_output = []
        l = torch.arange(self.bin_t).to(x.device)
        
        for k, i in enumerate(range(0, x.shape[1], self.bin1 * self.bin_t)): 
            x_part = x[:, i:i + self.bin1 * self.bin_t]
            for j in range(self.bin1):
                x_input = x_part[:, j::self.bin1]
                for num in range(self.n):
                    W_start = self.start_parameters_list[k * self.bin1 * self.n + j * self.n + num]
                    W_end = self.end_parameters_list[k * self.bin1 * self.n + j * self.n + num]
                    Weight = torch.sigmoid((l - W_start) * (W_end - l)).unsqueeze(0)
                    Wb = Binarize.apply(Weight - THRESHOLD)
                    x_output.append(
                        torch.prod(1 - (1 - x_input).unsqueeze(-1) * Wb.t(), dim=1)
                    )
        
        return torch.cat(x_output, dim=1)

    def clip(self):
        for W_start in self.start_parameters_list:
            W_start.data.clamp_(-1.0, self.bin_t)
        for W_end in self.end_parameters_list:
            W_end.data.clamp_(-1.0, self.bin_t)


class EventuallyLayer(nn.Module):
    """The eventually layer is used to learn the eventually of nodes."""

    def __init__(self, n, input_dim, bin1, bin_t, use_not=False, estimated_grad=False):
        super(EventuallyLayer, self).__init__()
        self.n = n
        self.use_not = use_not
        self.input_dim = input_dim if not use_not else input_dim * 2
        self.output_dim = int(n * self.input_dim / bin_t) 
        self.layer_type = 'Eventually'
        self.bin1 = bin1
        self.bin_t = bin_t

        start_parameters = (bin_t - 1) * torch.rand(self.output_dim)
        end_parameters = (bin_t - 1) * torch.rand(self.output_dim)
        self.start_parameters_list = nn.ParameterList([nn.Parameter(p) for p in start_parameters])
        self.end_parameters_list = nn.ParameterList([nn.Parameter(p) for p in end_parameters])
        self.Product = EstimatedProduct if estimated_grad else Product
        self.node_activation_cnt = None

    def forward(self, x):
        x = 1 - x   # Not x
        x_output = []
        l = torch.arange(self.bin_t).to(x.device)
        
        for k, i in enumerate(range(0, x.shape[1], self.bin1 * self.bin_t)): 
            x_part = x[:, i:i + self.bin1 * self.bin_t]
            for j in range(self.bin1):
                x_input = x_part[:, j::self.bin1]
                for num in range(self.n):
                    W_start = self.start_parameters_list[k * self.bin1 * self.n + j * self.n + num]
                    W_end = self.end_parameters_list[k * self.bin1 * self.n + j * self.n + num]
                    Weight = torch.sigmoid((l - W_start) * (W_end - l)).unsqueeze(0)
                    x_output.append(
                        1 - self.Product.apply(1 - (1 - x_input).unsqueeze(-1) * Weight.t())
                    )
        
        return torch.cat(x_output, dim=1)

    def binarized_forward(self, x):
        x = 1 - x
        x_output = []
        l = torch.arange(self.bin_t).to(x.device)
        
        for k, i in enumerate(range(0, x.shape[1], self.bin1 * self.bin_t)): 
            x_part = x[:, i:i + self.bin1 * self.bin_t]
            for j in range(self.bin1):
                x_input = x_part[:, j::self.bin1]
                for num in range(self.n):
                    W_start = self.start_parameters_list[k * self.bin1 * self.n + j * self.n + num]
                    W_end = self.end_parameters_list[k * self.bin1 * self.n + j * self.n + num]
                    Weight = torch.sigmoid((l - W_start) * (W_end - l)).unsqueeze(0)
                    Wb = Binarize.apply(Weight - THRESHOLD)
                    x_output.append(
                        1 - torch.prod(1 - (1 - x_input).unsqueeze(-1) * Wb.t(), dim=1)
                    )
        
        return torch.cat(x_output, dim=1)

    def clip(self):
        for Weight in self.start_parameters_list:
            Weight.data.clamp_(-1.0, self.bin_t)
        for Weight in self.end_parameters_list:
            Weight.data.clamp_(-1.0, self.bin_t)


class UntilLayer(nn.Module):
    """The until layer is used to learn the until of nodes."""

    def __init__(self, input_dim, bin1, bin_t, use_not=False, estimated_grad=False):
        super(UntilLayer, self).__init__()
        self.use_not = use_not
        self.input_dim = input_dim if not use_not else input_dim * 2
        self.output_dim = int(self.input_dim * (bin1 - 1) / bin_t)
        self.layer_type = 'Until'
        self.bin1 = bin1
        self.bin_t = bin_t

        bound1_parameters = (bin_t - 1) * torch.rand(self.output_dim)
        bound2_parameters = (bin_t - 1) * torch.rand(self.output_dim)
        self.bound1_parameters_list = nn.ParameterList([nn.Parameter(p) for p in bound1_parameters])
        self.bound2_parameters_list = nn.ParameterList([nn.Parameter(p) for p in bound2_parameters])
        self.Product = EstimatedProduct if estimated_grad else Product
        self.node_activation_cnt = None

        self.almost_zero = np.finfo(float).eps

    def forward(self, x):
        x_output = []
        l = torch.arange(self.bin_t).to(x.device)
        
        for i in range(0, x.shape[1], self.bin1 * self.bin_t):
            x_patch = x[:, i:i + self.bin1 * self.bin_t]    
            for j in range(self.bin1):
                x_input = x_patch[:, j::self.bin1]
                num = 0 
                
                for k in range(self.bin1):
                    if j == k:
                        continue
                        
                    weight_index = i * (self.bin1 - 1) // self.bin_t + j * (self.bin1 - 1) + num
                    num += 1
                    x_input_other = x_patch[:, k::self.bin1]
                    
                    # bound to start/end
                    bound1 = self.bound1_parameters_list[weight_index]
                    bound2 = self.bound2_parameters_list[weight_index]
                    W_end = 1/2 * (torch.abs(bound1 - bound2) + torch.abs(bound1 + bound2))
                    W_start = W_end - torch.abs(bound1 - bound2)
                    
                    # first layer
                    x_input_layer2 = []
                    for m in range(self.bin_t):
                        Weight_layer1 = torch.sigmoid((l - W_start) * (m - EPSILON2 - l)).unsqueeze(0)
                        x_input_layer2.append(
                            self.Product.apply(1 - (1 - x_input_other).unsqueeze(-1) * Weight_layer1.t())
                        )
                    
                    x_input_layer2 = torch.cat(x_input_layer2, dim=1)
                    x_until_input = torch.mul(x_input, x_input_layer2)
                    Weight = torch.sigmoid((l - W_start) * (W_end - l)).unsqueeze(0)
                    x_output.append(
                        1 - self.Product.apply(1 - (1 - (1 - x_until_input)).unsqueeze(-1) * Weight.t())
                    )
        
        return torch.cat(x_output, dim=1)

    def binarized_forward(self, x):
        x_output = []
        l = torch.arange(self.bin_t).to(x.device)
        
        for i in range(0, x.shape[1], self.bin1 * self.bin_t):
            x_patch = x[:, i:i + self.bin1 * self.bin_t]    
            for j in range(self.bin1):
                x_input = x_patch[:, j::self.bin1]
                num = 0 
                
                for k in range(self.bin1):
                    if j == k:
                        continue
                        
                    weight_index = i * (self.bin1 - 1) // self.bin_t + j * (self.bin1 - 1) + num
                    num += 1
                    x_input_other = x_patch[:, k::self.bin1]
                    
                    # bound to start/end
                    bound1 = self.bound1_parameters_list[weight_index]
                    bound2 = self.bound2_parameters_list[weight_index]
                    W_end = 1/2 * (torch.abs(bound1 - bound2) + torch.abs(bound1 + bound2))
                    W_start = W_end - torch.abs(bound1 - bound2)
                    
                    # first layer
                    x_input_layer2 = []
                    for m in range(self.bin_t):
                        Weight_layer1 = torch.sigmoid((l - W_start) * (m - EPSILON2 - l)).unsqueeze(0)
                        Wb_layer1 = Binarize.apply(Weight_layer1 - THRESHOLD)
                        x_input_layer2.append(
                            torch.prod(1 - (1 - x_input_other).unsqueeze(-1) * Wb_layer1.t(), dim=1)
                        )
                    
                    x_input_layer2 = torch.cat(x_input_layer2, dim=1)
                    x_until_input = torch.mul(x_input, x_input_layer2)
                    Weight = torch.sigmoid((l - W_start) * (W_end - l)).unsqueeze(0)
                    Wb = Binarize.apply(Weight - THRESHOLD)
                    x_output.append(
                        1 - torch.prod(1 - (1 - (1 - x_until_input)).unsqueeze(-1) * Wb.t(), dim=1)
                    )
        
        return torch.cat(x_output, dim=1)

    def clip(self):
        for Weight in self.bound1_parameters_list:
            Weight.data.clamp_(-1.0, self.bin_t)
        for Weight in self.bound2_parameters_list:
            Weight.data.clamp_(-1.0, self.bin_t)


class Box_diamond(nn.Module):
    def __init__(self, n, input_dim, bin1, bin_t, use_not=False, estimated_grad=False):
        super(Box_diamond, self).__init__()
        self.n = n
        self.input_dim = input_dim
        self.output_dim = int(n * self.input_dim / bin_t)
        self.layer_type = 'Box_diamond'
        self.bin1 = bin1
        self.bin_t = bin_t
        self.W = torch.zeros(self.output_dim, input_dim)

        t0_parameters = ((bin_t - 1) // 4) * torch.rand(self.output_dim)  # t0
        t1_parameters = ((bin_t - 1) // 4) * torch.rand(self.output_dim)  # t1
        t2_parameters = ((bin_t - 1) // 4) * torch.rand(self.output_dim)  # length
        self.t0_parameters_list = nn.ParameterList([nn.Parameter(p) for p in t0_parameters])
        self.t1_parameters_list = nn.ParameterList([nn.Parameter(p) for p in t1_parameters])
        self.t2_parameters_list = nn.ParameterList([nn.Parameter(p) for p in t2_parameters])

        self.Product = EstimatedProduct if estimated_grad else Product
        self.node_activation_cnt = None

    def forward(self, x):
        x_output = []
        l = torch.arange(self.bin_t).to(x.device)

        for k, i in enumerate(range(0, x.shape[1], self.bin1 * self.bin_t)): 
            x_part = x[:, i:i + self.bin1 * self.bin_t]
            for j in range(self.bin1):
                x_input = x_part[:, j::self.bin1]
                t_0 = self.t0_parameters_list[k * self.bin1 + j]
                t_1 = self.t1_parameters_list[k * self.bin1 + j]
                t_2 = self.t2_parameters_list[k * self.bin1 + j]
                x_input_layer2 = []

                for m in range(self.bin_t):
                    Weight_layer1 = torch.sigmoid((l - m) * (m + t_2 - l)).unsqueeze(0)
                    x_input_layer2.append(
                        1 - self.Product.apply(1 - x_input.unsqueeze(-1) * Weight_layer1.t())
                    )
                
                x_input_layer2 = torch.cat(x_input_layer2, dim=1)
                Weight = (torch.sigmoid((l - t_0) * (t_1 - l)) * 
                         torch.sigmoid((self.bin_t - 1 - t_2 - l) * (l - 0))).unsqueeze(0)
                x_output.append(
                    self.Product.apply(1 - (1 - x_input_layer2).unsqueeze(-1) * Weight.t())
                )

        return torch.cat(x_output, dim=1)

    def binarized_forward(self, x):
        x_output = []
        l = torch.arange(self.bin_t).to(x.device)

        for k, i in enumerate(range(0, x.shape[1], self.bin1 * self.bin_t)): 
            x_part = x[:, i:i + self.bin1 * self.bin_t]
            for j in range(self.bin1):
                x_input = x_part[:, j::self.bin1]
                t_0 = self.t0_parameters_list[k * self.bin1 + j]
                t_1 = self.t1_parameters_list[k * self.bin1 + j]
                t_2 = self.t2_parameters_list[k * self.bin1 + j]
                x_input_layer2 = []

                for m in range(self.bin_t):
                    Weight_layer1 = torch.sigmoid((l - m) * (m + t_2 - l)).unsqueeze(0)
                    Wb_layer1 = Binarize.apply(Weight_layer1 - THRESHOLD)
                    x_input_layer2.append(
                        1 - self.Product.apply(1 - x_input.unsqueeze(-1) * Wb_layer1.t())
                    )
                
                x_input_layer2 = torch.cat(x_input_layer2, dim=1)
                Weight_layer2_1 = torch.sigmoid((l - t_0) * (t_1 - l)).unsqueeze(0)
                Weight_layer2_2 = torch.sigmoid((self.bin_t - 1 - t_2 - l) * (l - 0)).unsqueeze(0)
                Wb_layer2_1 = Binarize.apply(Weight_layer2_1 - THRESHOLD)
                Wb_layer2_2 = Binarize.apply(Weight_layer2_2 - THRESHOLD)
                Wb_layer2 = Wb_layer2_1 * Wb_layer2_2
                x_output.append(
                    self.Product.apply(1 - (1 - x_input_layer2).unsqueeze(-1) * Wb_layer2.t())
                )
        
        return torch.cat(x_output, dim=1)

    def clip(self):
        for t_0 in self.t0_parameters_list:
            t_0.data.clamp_(0.0, self.bin_t - 1)
        for t_1 in self.t1_parameters_list:
            t_1.data.clamp_(0.0, self.bin_t - 1)
        for t_2 in self.t2_parameters_list:
            t_2.data.clamp_(0.0, self.bin_t - 1)


class Diamond_box(nn.Module):
    def __init__(self, n, input_dim, bin1, bin_t, use_not=False, estimated_grad=False):
        super(Diamond_box, self).__init__()
        self.n = n
        self.input_dim = input_dim
        self.output_dim = int(n * self.input_dim / bin_t)
        self.layer_type = 'Diamond_box'
        self.bin1 = bin1
        self.bin_t = bin_t
        self.W = torch.zeros(self.output_dim, input_dim)

        t0_parameters = ((bin_t - 1) // 4) * torch.rand(self.output_dim)  # t0
        t1_parameters = ((bin_t - 1) // 4) * torch.rand(self.output_dim)  # t1
        t2_parameters = ((bin_t - 1) // 4) * torch.rand(self.output_dim)  # length
        self.t0_parameters_list = nn.ParameterList([nn.Parameter(p) for p in t0_parameters])
        self.t1_parameters_list = nn.ParameterList([nn.Parameter(p) for p in t1_parameters])
        self.t2_parameters_list = nn.ParameterList([nn.Parameter(p) for p in t2_parameters])

        self.Product = EstimatedProduct if estimated_grad else Product
        self.node_activation_cnt = None

    def forward(self, x):
        x_output = []
        l = torch.arange(self.bin_t).to(x.device)

        for k, i in enumerate(range(0, x.shape[1], self.bin1 * self.bin_t)): 
            x_part = x[:, i:i + self.bin1 * self.bin_t]
            for j in range(self.bin1):
                x_input = x_part[:, j::self.bin1]
                t_0 = self.t0_parameters_list[k * self.bin1 + j]
                t_1 = self.t1_parameters_list[k * self.bin1 + j]
                t_2 = self.t2_parameters_list[k * self.bin1 + j]
                x_input_layer2 = []

                for m in range(self.bin_t):
                    Weight_layer1 = torch.sigmoid((l - m) * (m + t_2 - l)).unsqueeze(0)
                    x_input_layer2.append(
                        self.Product.apply(1 - (1 - x_input).unsqueeze(-1) * Weight_layer1.t())
                    )
                
                x_input_layer2 = torch.cat(x_input_layer2, dim=1)
                Weight = (torch.sigmoid((l - t_0) * (t_1 - l)) * 
                         torch.sigmoid((self.bin_t - 1 - t_2 - l) * (l - 0))).unsqueeze(0)
                x_output.append(
                    1 - self.Product.apply(1 - (x_input_layer2).unsqueeze(-1) * Weight.t())
                )

        return torch.cat(x_output, dim=1)

    def binarized_forward(self, x):
        x_output = []
        l = torch.arange(self.bin_t).to(x.device)

        for k, i in enumerate(range(0, x.shape[1], self.bin1 * self.bin_t)): 
            x_part = x[:, i:i + self.bin1 * self.bin_t]
            for j in range(self.bin1):
                x_input = x_part[:, j::self.bin1]
                t_0 = self.t0_parameters_list[k * self.bin1 + j]
                t_1 = self.t1_parameters_list[k * self.bin1 + j]
                t_2 = self.t2_parameters_list[k * self.bin1 + j]
                x_input_layer2 = []

                for m in range(self.bin_t):
                    Weight_layer1 = torch.sigmoid((l - m) * (m + t_2 - l)).unsqueeze(0)
                    Wb_layer1 = Binarize.apply(Weight_layer1 - THRESHOLD)
                    x_input_layer2.append(
                        self.Product.apply(1 - (1 - x_input).unsqueeze(-1) * Wb_layer1.t())
                    )
                
                x_input_layer2 = torch.cat(x_input_layer2, dim=1)
                Weight_layer2_1 = torch.sigmoid((l - t_0) * (t_1 - l)).unsqueeze(0)
                Weight_layer2_2 = torch.sigmoid((self.bin_t - 1 - t_2 - l) * (l - 0)).unsqueeze(0)
                Wb_layer2_1 = Binarize.apply(Weight_layer2_1 - THRESHOLD)
                Wb_layer2_2 = Binarize.apply(Weight_layer2_2 - THRESHOLD)
                Wb_layer2 = Wb_layer2_1 * Wb_layer2_2
                x_output.append(
                    1 - self.Product.apply(1 - (1 - (1 - x_input_layer2)).unsqueeze(-1) * Wb_layer2.t())
                )
        
        return torch.cat(x_output, dim=1)

    def clip(self):
        for t_0 in self.t0_parameters_list:
            t_0.data.clamp_(0.0, self.bin_t - 1)
        for t_1 in self.t1_parameters_list:
            t_1.data.clamp_(0.0, self.bin_t - 1)
        for t_2 in self.t2_parameters_list:
            t_2.data.clamp_(0.0, self.bin_t - 1)


class UnionLayer(nn.Module):
    """The union layer is used to learn the rule-based representation."""

    def __init__(self, n, input_dim, use_not=False, estimated_grad=False):
        super(UnionLayer, self).__init__()
        self.n = n
        self.use_not = use_not
        self.input_dim = input_dim
        self.output_dim = self.n * 2
        self.layer_type = 'union'
        self.forward_tot = None
        self.node_activation_cnt = None
        self.dim2id = None
        self.rule_list = None
        self.rule_name = None

        self.con_layer = ConjunctionLayer(
            self.n, self.input_dim, use_not=use_not, estimated_grad=estimated_grad
        )
        self.dis_layer = DisjunctionLayer(
            self.n, self.input_dim, use_not=use_not, estimated_grad=estimated_grad
        )

    def forward(self, x):
        return torch.cat([self.con_layer(x), self.dis_layer(x)], dim=1)

    def binarized_forward(self, x):
        return torch.cat([
            self.con_layer.binarized_forward(x),
            self.dis_layer.binarized_forward(x)
        ], dim=1)

    def clip(self):
        self.con_layer.clip()
        self.dis_layer.clip()


class TemporalLayer(nn.Module):
    """The temporal logic layer is used to learn the rule-based representation."""

    def __init__(self, bin1, time_series_length, bin_t, use_not=False, estimated_grad=False):
        super(TemporalLayer, self).__init__()
        self.use_not = use_not
        self.input_dim = time_series_length * 3 * bin1  
        self.n = 1
        self.output_dim = (
            int(4 * self.n * self.input_dim / bin_t) + 
            int((bin1 - 1) * self.input_dim / bin_t) + 
            5 * time_series_length * 3 * bin1 // bin_t
        )
        self.layer_type = 'temporal'
        self.forward_tot = None
        self.node_activation_cnt = None
        self.dim2id = {i: i for i in range(int(self.output_dim))}
        self.rule_list = None
        self.rule_name = None

        self.time_series_length = time_series_length
        self.bin1 = bin1
        self.bin_t = bin_t

        # temporal logic layers
        self.always_layer = AlwaysLayer(
            self.n, self.input_dim, bin1, bin_t, use_not=use_not, estimated_grad=estimated_grad
        )
        self.eventually_layer = EventuallyLayer(
            self.n, self.input_dim, bin1, bin_t, use_not=use_not, estimated_grad=estimated_grad
        )
        self.until_layer = UntilLayer(
            self.input_dim, bin1, bin_t, use_not=use_not, estimated_grad=estimated_grad
        )
        self.box_diamond_layer = Box_diamond(
            1, self.input_dim, bin1, bin_t, use_not=use_not, estimated_grad=estimated_grad
        )
        self.diamond_box_layer = Diamond_box(
            1, self.input_dim, bin1, bin_t, use_not=use_not, estimated_grad=estimated_grad
        )

    def forward(self, x):
        return torch.cat([
            self.always_layer(x[:, :self.input_dim]), 
            self.eventually_layer(x[:, :self.input_dim]),
            self.until_layer(x[:, :self.input_dim]),
            self.box_diamond_layer(x[:, :self.input_dim]),
            self.diamond_box_layer(x[:, :self.input_dim]),
            x[:, self.input_dim:]
        ], dim=1)

    def binarized_forward(self, x):
        return torch.cat([
            self.always_layer.binarized_forward(x[:, :self.input_dim]),
            self.eventually_layer.binarized_forward(x[:, :self.input_dim]),
            self.until_layer.binarized_forward(x[:, :self.input_dim]),
            self.diamond_box_layer.binarized_forward(x[:, :self.input_dim]),
            self.box_diamond_layer.binarized_forward(x[:, :self.input_dim]),
            x[:, self.input_dim:]
        ], dim=1)

    def clip(self):
        self.always_layer.clip()
        self.eventually_layer.clip()
        self.until_layer.clip()
        self.box_diamond_layer.clip()
        self.diamond_box_layer.clip()