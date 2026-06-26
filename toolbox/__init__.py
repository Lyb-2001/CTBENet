from .metrics import averageMeter, runningScore
from .log import get_logger
from .optim.AdamW import AdamW
from .optim.Lookahead import Lookahead
from .optim.RAdam import RAdam
from .optim.Ranger import Ranger


from .utils import ClassWeight, save_ckpt, load_ckpt, class_to_RGB, \
    compute_speed, setup_seed, group_weight_decay


def get_dataset(cfg):
    assert cfg['dataset'] in ['nyuv2', 'sunrgbd', 'cityscapes', 'camvid', 'irseg', 'PST900', "glassrgbt", "mirrorrgbd", 'glassrgbt_merged', 'SUS', 'CART',
                              'CART_Terrain', 'KP', 'FMB', 'MSRS']

    if cfg['dataset'] == 'irseg':
        from .datasets.MFNet import IRSeg
        return {'train': IRSeg(cfg, mode='trainval'),
                'test': IRSeg(cfg, mode='test')}
        # return IRSeg(cfg, mode='train'), IRSeg(cfg, mode='val'), IRSeg(cfg, mode='test')

    if cfg['dataset'] == 'MSRS':
        from .datasets.MSRS import MSRS
        return {'train':MSRS(cfg, mode='train'),
                'test':MSRS(cfg, mode='test'),
                'test_night':MSRS(cfg, mode='test_night'),
                'test_day':MSRS(cfg, mode='test_day')}
        # return IRSeg(cfg, mode='train'), IRSeg(cfg, mode='val'), IRSeg(cfg, mode='test')

    if cfg['dataset'] == 'SUS':
        from .datasets.SUS import SUS
        return {'train': SUS(cfg, mode='train'),
                'val': SUS(cfg, mode='val'),
                'test': SUS(cfg, mode='test')}
        # return SUS(cfg, mode='train_night'), SUS(cfg, mode='val'), SUS(cfg, mode='test_night')

    if cfg['dataset'] == 'CART':
        from .datasets.CART import CART
        # return SUS(cfg, mode='trainval'), SUS(cfg, mode='test')
        return {'train': CART(cfg, mode='train'),
                'val': CART(cfg, mode='val'),
                'test': CART(cfg, mode='test')}

    if cfg['dataset'] == 'CART_Terrain':
        from .datasets.CART_Terrain import Terrain
        return {'train': Terrain(cfg, mode='train'),
                'val': Terrain(cfg, mode='val'),
                'test': Terrain(cfg, mode='test')}

    if cfg['dataset'] == 'PST900':
        from .datasets.pst900 import PST900

        return {'train': PST900(cfg, mode='train'),
                'test': PST900(cfg, mode='test')}

    if cfg['dataset'] == 'KP':
        from .datasets.KP import KP

        return {'train': KP(cfg, mode='train'),
                'val': KP(cfg, mode='val'),
                'tets': KP(cfg, mode='test')}

    if cfg['dataset'] == 'FMB':
        from .datasets.FMB import FMB

        return {'train': FMB(cfg, mode='train'),
                'test': FMB(cfg, mode='test')}
def get_metrics(cfg):
    if cfg['dataset'] in  ['MFNet', 'MSRS', 'PST900', 'FMB']:  # with background
        from toolbox.metrics_MFNet import averageMeter, runningScore
    elif cfg['dataset'] == 'SUS': # without background
        from toolbox.metrics_SUS import averageMeter, runningScore
    elif cfg['dataset'] == 'CART': # from 2:
        from toolbox.metrics_CART import averageMeter, runningScore
    else:
        raise ValueError(f"Unsupported dataset: {cfg['dataset']}")
    return averageMeter(), runningScore(cfg['n_classes'])

def get_weights_semantic(cfg):
    if cfg['dataset'] in ['FMB']:
        class_weight_semantic = torch.from_numpy(np.array(
            [10.2249, 9.6609, 32.8497, 6.0635, 48.1396, 44.9108, 4.4491, 3.1748,
             43.9271, 15.9236, 43.1266, 44.8469, 48.6038, 50.4826, 27.1057])).float()
    elif cfg['dataset'] in ['SUS']:
        class_weight_semantic = torch.from_numpy(np.array([2.0268,  4.2508, 23.6082, 23.0149, 11.6264, 25.8710])).float()
    elif cfg['dataset'] in ['CART']:
        class_weight_semantic = torch.from_numpy(np.array([50.2527, 50.4935, 4.8389, 6.3680, 24.0135, 26.3811, 9.7799, 14.6093, 16.8741, 2.7478, 49.2211, 50.2928])).float()
    elif cfg['dataset'] in ['PST900']:
        class_weight_semantic = torch.from_numpy(np.array([1.4590, 41.9667, 32.1435, 46.7086, 26.7601])).float()
    elif cfg['dataset'] in ['MSRS']:
        class_weight_semantic = torch.from_numpy(np.array([ 1.5070, 17.0317, 30.0930, 34.9505, 40.5934, 40.5476, 48.0715, 46.1221, 45.6460])).float()
    else:
        raise ValueError(f"Unsupported dataset: {cfg['dataset']}")
    return class_weight_semantic
def get_model(cfg):



    if cfg['model_name'] == 'upfusion':
        from models.fusion.upfusion import UPFusion
        return UPFusion()
    if cfg['model_name'] == 'test':
        from models.fusion.test import LiteUPFusion
        return LiteUPFusion(inp_channels=1, out_channels=1, dim=12)
    if cfg['model_name'] == 'fu_seg':
        from models.dinov3.test import LiteUPFusion
        return LiteUPFusion(inp_channels=1, out_channels=1, dim=12)
    if cfg['model_name'] == 'segnet':
        from models.seg.net1 import stuNet
        return stuNet()
    if cfg['model_name'] == 'CDT':
        from CDTFusion.fmb_train_to import stuNet
        return stuNet()

    # if cfg['model_name'] == 'text':
    #     from models.dinov3.dinov3unet_clip import DINOv3_UNet
    #     return DINOv3_UNet()
    if cfg['model_name'] == 'CTBENet':
        from net.CTBENet import DINOv3_UNet
        return DINOv3_UNet()








