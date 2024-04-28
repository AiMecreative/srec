from hydra.utils import instantiate
from srec.arch import EDDArch, EDDModule
from srec.utils.utils import InitDevice
from srec.models.srec_model import Srec
from srec.models.encoder import HREncoder
from srec.models.augment import Augment
from srec.models.recognizer import Recognizer
from srec.models.sr import SR


class ModelBuilder(object):

    def __init__(self, configs) -> None:

        self.encoder_configs = instantiate(configs.encoder)
        self.augment_configs = instantiate(configs.augment)
        self.recognizer_configs = instantiate(configs.recognizer)
        self.sr_configs = instantiate(configs.sr)
        device = configs.task.device

        EDDModule.INIT_DEVICE = InitDevice(device)

    def get_model(
        self,
        load_pretrain: bool = False,
        load_recognizer: bool = True,
        pretrain_weights: str = '',
        pretrain_recognizer: str = ''
    ):
        encoder = HREncoder(self.encoder_configs)
        augment = Augment(self.augment_configs)
        recognizer = Recognizer(self.recognizer_configs)
        sr = SR(self.sr_configs)
        return Srec(
            encoder,
            augment,
            recognizer,
            sr,
            load_pretrain,
            load_recognizer,
            pretrain_weights,
            pretrain_recognizer,
        )
