import torch
import lpips
from basicsr.utils.registry import METRIC_REGISTRY
from basicsr.metrics.metric_util import reorder_image, to_tensor

@METRIC_REGISTRY.register()
def calculate_lpips(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """
    Calculate LPIPS between img and img2.
    Args:
        img, img2 (ndarray): Images with range [0, 255], shape HWC, dtype uint8 or float32.
        crop_border (int): Pixels to crop at border before calculation.
        input_order (str): 'HWC' or 'CHW'
        test_y_channel (bool): Not used here.
    Returns:
        float: LPIPS score (lower is better).
    """
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, :]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, :]

    # Normalize and convert to tensor
    img_tensor = to_tensor(img, bgr2rgb=True, float32=True).unsqueeze(0).cuda()
    img2_tensor = to_tensor(img2, bgr2rgb=True, float32=True).unsqueeze(0).cuda()

    # Normalize to [-1, 1]
    img_tensor = (img_tensor - 0.5) / 0.5
    img2_tensor = (img2_tensor - 0.5) / 0.5

    loss_fn = lpips.LPIPS(net='vgg').cuda()
    lpips_score = loss_fn(img_tensor, img2_tensor).item()
    return lpips_score
