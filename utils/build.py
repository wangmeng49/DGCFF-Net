from utils.config import Config, ConfigDict
from rscd.models.backbones import *
from rscd.models.decoderheads import *

from rscd.losses import *
# utils/build.py 顶部
from rscd.models.decoderheads.FC_EF import Unet
from rscd.models.decoderheads.STADENet import BASE_Transformer#不用
from rscd.models.decoderheads.FC_siam_conc import SiamUnet_conc
from rscd.models.decoderheads.SEIFnet import SEIFNet
from rscd.models.decoderheads.DMInet import DMINet
from rscd.models.decoderheads.FTn import Encoder
from rscd.models.decoderheads.A2net import A2Net
def build_from_cfg(cfg,):  #default_args=None是因为SEIFNet新加的,SEIFNet需要
    if not isinstance(cfg, (dict, ConfigDict, Config)):
        raise TypeError(
            f'cfg should be a dict, ConfigDict or Config, but got {type(cfg)}')
    if 'type' not in cfg:
        raise KeyError(
                '`cfg` must contain the key "type", '
                f'but got {cfg}')

    '''
    # 因为SEIFNet新增的。新增：复制cfg，防止原地pop修改
    cfg = dict(cfg)
    obj_type = cfg.pop('type')

    # 合并 default_args
    if default_args:
        for  k, v in default_args.items():
            cfg.setdefault(k, v)

    obj_cls = eval(obj_type)
    obj = obj_cls(**cfg)
    '''


    #原来的
    obj_type = cfg.pop('type')
    obj_cls = eval(obj_type)
    obj = obj_cls(**cfg)


    return obj