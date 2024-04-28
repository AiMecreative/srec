from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM


class Metrics(object):

    def __init__(
        self,
        device: str,
    ) -> None:
        
        self.psnr_metric = PSNR().to(device)
        self.ssim_metric = SSIM().to(device)

    def psnr(self, img1, img2):
        return self.psnr_metric(img1, img2)
    
    def ssim(self, img1, img2):
        return self.ssim_metric(img1, img2)
    
    def acc(self, preds, gts):
        count = 0
        sum_gt = 0
        for idx, pred in enumerate(preds):
            if pred == gts[idx]:
                count += 1
            sum_gt += 1
        return count / sum_gt
