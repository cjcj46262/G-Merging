import torch
import torch.nn as nn
import sys

import functools
from collections import OrderedDict


class L2Regularization(nn.Module):

    def __init__(self, model: nn.Module):
        super(L2Regularization, self).__init__()
        self.model = model

    def forward(self):
        output = 0.0
        for param in self.model.parameters():
            output += 0.5 * torch.norm(param) ** 2
        return output


class SPRegularization(nn.Module):

    def __init__(self, source_model: nn.Module, target_model: nn.Module):
        super(SPRegularization, self).__init__()
        self.target_model = target_model
        self.source_weight = {}
        for name, param in source_model.named_parameters():
            self.source_weight[name] = param.detach()

    def forward(self):
        output = 0.0

        for name, param in self.target_model.named_parameters():
            output += 0.5 * torch.norm(param - self.source_weight[name]) ** 2
        return output


class FrobeniusRegularization(nn.Module):

    def __init__(self, source_model: nn.Module, target_model: nn.Module):
        super(FrobeniusRegularization, self).__init__()
        self.target_model = target_model
        self.source_weight = {}
        self.D = 1
        self.scale_factor = 2
        for name, param in source_model.named_parameters():
            self.source_weight[name] = param.detach()

    def forward(self):
        output = 0.0
        for name, param in self.target_model.named_parameters():
            if name.startswith('gnns') or name.startswith('batch_norms'):
                gamma = int(name.split('.')[1])
            else:
                gamma = 1
            #     norm = torch.norm(param - self.source_weight[name]) Frobi
            norm = torch.abs(param - self.source_weight[name]).sum(-1).max()  # MARS  max absolute row sum
            output = output + torch.clamp_min(
                norm - self.D * torch.pow(torch.tensor([self.scale_factor], device=norm.device),
                                          gamma), 0)

        # for name, param in self.target_model.named_parameters():
        #     output += 0.5 * torch.norm(param - self.source_weight[name]) ** 2
        # for name, param in self.target_model.named_parameters():
        #     if name.startswith('gnns'):
        #         # print(int(name.split('.')[1]))
        #         print(torch.norm(param - self.source_weight[name]))
        #         print(torch.pow(torch.tensor([1.2]),int(name.split('.')[1])))
        return output


class BehavioralRegularization(nn.Module):

    def __init__(self):
        super(BehavioralRegularization, self).__init__()

    def forward(self, layer_outputs_source, layer_outputs_target):
        output = 0.0
        for fm_src, fm_tgt in zip(layer_outputs_source.values(), layer_outputs_target.values()):
            output += 0.5 * (torch.norm(fm_tgt - fm_src.detach()) ** 2)
        return output


class AttentionBehavioralRegularization(nn.Module):

    def __init__(self, channel_attention):
        super(AttentionBehavioralRegularization, self).__init__()
        self.channel_attention = channel_attention
        # self.channel_attention = torch.random

    def forward(self, layer_outputs_source, layer_outputs_target):
        output = 0.0
        for i, (fm_src, fm_tgt) in enumerate(zip(layer_outputs_source.values(), layer_outputs_target.values())):
            # b, c, h, w = fm_src.shape
            b, c = fm_src.shape
            # fm_src = fm_src.reshape(b, c, h * w)
            # fm_tgt = fm_tgt.reshape(b, c, h * w)
            # todo 欧式距离-> warstrass distance
            distance = torch.norm(fm_tgt - fm_src.detach(), p=2, dim=0)  #

            # self.channel_attention[i] = torch.ones_like(self.channel_attention[i])/self.channel_attention[i].shape[0]
            distance = torch.mul(self.channel_attention[i], distance ** 2)
            # distance = c * torch.mul(self.channel_attention[i], distance ** 2) / (h * w)
            output += 0.5 * torch.sum(distance)

        return output


def get_attribute(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


class IntermediateLayerGetter(object):

    def __init__(self, model, return_layers, keep_output=True):
        self._model = model
        self.return_layers = return_layers
        self.keep_output = keep_output

    def __call__(self, *args, **kwargs):
        ret = OrderedDict()
        handles = []
        for name in self.return_layers:
            layer = get_attribute(self._model, name)

            def hook(module, input, output, name=name):
                ret[name] = output

            try:
                h = layer.register_forward_hook(hook)
            except AttributeError as e:
                raise AttributeError(f'Module {name} not found')
            handles.append(h)
        # todo
        if self.keep_output:
            output,feature = self._model(*args, **kwargs)
        else:
            self._model(*args, **kwargs)
            output = None

        for h in handles:
            h.remove()

        return ret, feature
