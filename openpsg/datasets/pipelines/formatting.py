from mmcv.parallel import DataContainer as DC
from mmdet.datasets import PIPELINES
from mmdet.datasets.pipelines import DefaultFormatBundle, to_tensor


@PIPELINES.register_module()
class SceneGraphFormatBundle(DefaultFormatBundle):
    def __call__(self, results):
        results = super().__call__(results)

        if 'rel_fields' in results and len(results['rel_fields']) > 0:
            for key in results['rel_fields']:
                results[key] = DC(to_tensor(results[key]))
        if 'gt_scenes' in results:
            results['gt_scenes'] = DC(to_tensor(results['gt_scenes']))

        return results


@PIPELINES.register_module()
class PanopticSceneGraphFormatBundle(SceneGraphFormatBundle):
    def __call__(self, results):
        results = super().__call__(results)

        for key in ['all_gt_bboxes', 'all_gt_labels']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]))

        return results
