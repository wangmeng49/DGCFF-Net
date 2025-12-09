import torch
from torch import nn
import sys
sys.path.append('rscd')
from utils.build import build_from_cfg
from rscd.models.decoderheads.FC_EF import Unet
from rscd.models.decoderheads.SEIFnet import SEIFNet

class myModel(nn.Module):
    def __init__(self, cfg):
        super(myModel, self).__init__()
        # 如果 cfg 里有 backbone，就构建；没有则设为 None
        if hasattr(cfg, 'backbone') and cfg.backbone is not None:
            self.backbone = build_from_cfg(cfg.backbone)
        else:
            self.backbone = None
        #self.backbone = build_from_cfg(cfg.backbone)

        #self.decoderhead = build_from_cfg(cfg.decoderhead, default_args=dict(args=cfg))  #SEIFNet的
        self.decoderhead = build_from_cfg(cfg.decoderhead)#原来的
    
    def forward(self, x1, x2, gtmask=None):
        if self.backbone is not None:
            backbone_outputs = self.backbone(x1, x2)
        else:
            # 如果无 backbone，则直接传入两个输入
            backbone_outputs = (x1, x2)
        #backbone_outputs = self.backbone(x1, x2)
        if gtmask == None:
            x_list = self.decoderhead(backbone_outputs)#*号只针对FC_EF网络，其他网络要去掉，*号表示可以接收多个输入，STNet网络要去掉
        else:
            x_list = self.decoderhead(backbone_outputs, gtmask)
        return x_list

"""
对于不满足该范式的模型可在backbone部分进行定义, 并在此处导入
"""

# model_config
def build_model(cfg):
    c = myModel(cfg)
    return c


if __name__ == "__main__":
    x1 = torch.randn(4, 3, 512, 512)
    x2 = torch.randn(4, 3, 512, 512)
    target = torch.randint(low=0,high=2,size=[4, 512, 512])
    file_path = r""

    from utils.config import Config
    from rscd.losses import build_loss

    cfg = Config.fromfile(file_path)
    net = build_model(cfg.model_config)
    res = net(x1, x2)
    print(res.shape)
    loss = build_loss(cfg.loss_config)

    compute = loss(res,target)
    print(compute)