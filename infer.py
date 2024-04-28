import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from srec.scheduler import SrecSchduler
from data.data_module import DataModule
from srec.dataclass_configs import DataConfigs
from srec.utils.utils import collate_fn
from functools import partial


@hydra.main(config_path='configs', config_name='configs_crnn_infer', version_base='v1.2')
def infer(configs: OmegaConf):

    data_conf: DataConfigs = instantiate(configs.data)
    collate = partial(collate_fn, img_size=data_conf.img_shape)
    dm = DataModule(data_conf)
    infer_dl = dm.data_loader(
        data_conf.infer_ds,
        collate_fn=collate
    )

    scheduler = SrecSchduler(configs)
    scheduler.resume(configs.task.pretrain_weights, freeze=True)
    scheduler.infer(infer_dl)


if __name__ == '__main__':
    infer()
