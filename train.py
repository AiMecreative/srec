import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from srec.scheduler import SrecSchduler
from data.data_module import DataModule
from srec.builder import ModelBuilder
from srec.dataclass_configs import DataConfigs
from srec.utils.utils import collate_fn
from functools import partial


@hydra.main(config_path='configs', config_name='configs_crnn2', version_base='v1.2')
def main(configs: OmegaConf):

    data_conf: DataConfigs = instantiate(configs.data)
    collate = partial(collate_fn, img_size=data_conf.img_shape)
    dm = DataModule(data_conf)
    train_dl = dm.data_loader(
        data_conf.train_ds,
        collate_fn=collate
    )
    eval_dl = dm.data_loader(
        data_conf.eval_ds,
        collate_fn=collate
    )

    scheduler = SrecSchduler(configs)
    scheduler.train(train_dl, eval_dl)


if __name__ == '__main__':
    main()
