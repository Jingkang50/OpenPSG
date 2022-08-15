import os
import tempfile
import shutil
from typing import List
from cog import BasePredictor, Path, Input, BaseModel

from openpsg.utils.utils import show_result
from mmdet.apis import init_detector, inference_detector
from mmcv import Config
import mmcv


class ModelOutput(BaseModel):
    image: Path


class Predictor(BasePredictor):
    def setup(self):
        model_ckt = "epoch_60.pth"
        cfg = Config.fromfile("configs/psgtr/psgtr_r50_psg_inference.py")
        self.model = init_detector(cfg, model_ckt, device="cpu")

    def predict(
        self,
        image: Path = Input(
            description="Input image.",
        ),
        num_rel: int = Input(
            description="Number of Relations. Each relation will generate a scene graph",
            default=5,
            ge=1,
            le=20,
        ),
    ) -> List[ModelOutput]:
        input_image = mmcv.imread(str(image))
        result = inference_detector(self.model, input_image)
        out_path = Path(tempfile.mkdtemp()) / "output.png"
        out_dir = "temp"
        show_result(
            str(image),
            result,
            is_one_stage=True,
            num_rel=num_rel,
            out_dir=out_dir,
            out_file=str(out_path),
        )
        output = []
        output.append(ModelOutput(image=out_path))
        for i, img_path in enumerate(os.listdir(out_dir)):
            img = mmcv.imread(os.path.join(out_dir, img_path))
            out_path = Path(tempfile.mkdtemp()) / f"output_{i}.png"
            mmcv.imwrite(img, str(out_path))
            output.append(ModelOutput(image=out_path))
        shutil.rmtree(out_dir)

        return output
