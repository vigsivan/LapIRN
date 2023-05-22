from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from Functions import (
    generate_grid_unit,
    imgnorm,
    load_4D,
    save_flow,
    save_img,
    transform_unit_flow_to_flow,
)
from miccai2020_model_stage import (
    Miccai2020_LDR_laplacian_unit_add_lvl1,
    Miccai2020_LDR_laplacian_unit_add_lvl2,
    Miccai2020_LDR_laplacian_unit_add_lvl3,
    SpatialTransform_unit,
)
from pydantic import BaseModel
from tqdm import tqdm

def compute_dice_coefficient(mask_gt: torch.Tensor, mask_pred: torch.Tensor) -> torch.Tensor:
    """Computes soerensen-dice coefficient.

  compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
  and the predicted mask `mask_pred`.

  Args:
    mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
    mask_pred: 3-dim Numpy array of type bool. The predicted mask.

  Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN.
  """
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return torch.Tensor(0, device=mask_gt.device)
    volume_intersect = (mask_gt * mask_pred).sum()
    return 2 * volume_intersect / volume_sum


class TestConfig(BaseModel):
    modelpath: Path
    savepath: Path
    fixed: List[Path]
    moving: List[Path]
    fixed_segmentations: Optional[List[Path]]
    moving_segmentations: Optional[List[Path]]
    imgshape_4: List[int]
    imgshape_2: List[int]
    imgshape: List[int]
    range_flow: float = 0.4
    start_channel: int = 7


def test(config: TestConfig):
    model_lvl1 = Miccai2020_LDR_laplacian_unit_add_lvl1(
        2,
        3,
        config.start_channel,
        is_train=True,
        imgshape=config.imgshape_4,
        range_flow=config.range_flow,
    ).cuda()
    model_lvl2 = Miccai2020_LDR_laplacian_unit_add_lvl2(
        2,
        3,
        config.start_channel,
        is_train=True,
        imgshape=config.imgshape_2,
        range_flow=config.range_flow,
        model_lvl1=model_lvl1,
    ).cuda()

    model = Miccai2020_LDR_laplacian_unit_add_lvl3(
        2,
        3,
        config.start_channel,
        is_train=False,
        imgshape=config.imgshape,
        range_flow=config.range_flow,
        model_lvl2=model_lvl2,
    ).cuda()

    transform = SpatialTransform_unit().cuda()

    model.load_state_dict(torch.load(config.modelpath))
    model.eval()
    transform.eval()

    grid = generate_grid_unit(config.imgshape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).cuda().float()

    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    do_segmentations = config.fixed_segmentations is not None and config.moving_segmentations is not None

    for i, (fixed_path, moving_path) in tqdm(enumerate(zip(config.fixed, config.moving))):
        fixed_img = load_4D(str(fixed_path))
        moving_img = load_4D(str(moving_path))

        # normalize image to [0, 1]
        norm = True
        if norm:
            fixed_img = imgnorm(fixed_img)
            moving_img = imgnorm(moving_img)

        fixed_img = torch.from_numpy(fixed_img).float().to(device).unsqueeze(dim=0)
        moving_img = torch.from_numpy(moving_img).float().to(device).unsqueeze(dim=0)

        with torch.no_grad():
            F_X_Y = model(moving_img, fixed_img)

            X_Y = (
                transform(moving_img, F_X_Y.permute(0, 2, 3, 4, 1), grid)
                .data.cpu()
                .numpy()[0, 0, :, :, :]
            )

            F_X_Y_cpu = F_X_Y.data.cpu().numpy()[0, :, :, :, :].transpose(1, 2, 3, 0)
            F_X_Y_cpu = transform_unit_flow_to_flow(F_X_Y_cpu)

            fname = fixed_path.name.split('.')[0]
            mname = moving_path.name.split('.')[0]
            o_prefix = f"{mname}2{fname}"

            save_flow(F_X_Y_cpu, str(config.savepath / f"{o_prefix}_flow.nii.gz"))
            save_img(X_Y, str(config.savepath / f"{o_prefix}.nii.gz"))

            if do_segmentations:
                fixed_seg = load_4D(str(config.fixed_segmentations[i]))  # type: ignore
                moving_seg = load_4D(str(config.moving_segmentations[i]))  # type: ignore

                moved_seg = (
                    transform(moving_seg, F_X_Y.permute(0, 2, 3, 4, 1), grid)
                    .data.cpu()
                    .numpy()[0, 0, :, :, :]
                )
                save_img(moved_seg, str(config.savepath / f"{mname}_warped_seg.nii.gz"))
                compute_dice_coefficient(torch.from_numpy(fixed_seg), torch.from_numpy(moved_seg))

if __name__ == "__main__":
    import sys
    import json
    argv = sys.argv
    if len(argv) != 2:
        print(f"Usage {argv[0]} <CONFIG_JSON>")
    with open(argv[1], 'r') as f:
        test_config = TestConfig(**json.load(f))
    test(test_config)
