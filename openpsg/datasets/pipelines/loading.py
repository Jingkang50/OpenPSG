import os.path as osp

import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC
from mmdet.core import BitmapMasks
from mmdet.datasets import PIPELINES
from mmdet.datasets.pipelines import (DefaultFormatBundle, LoadAnnotations,
                                      to_tensor)
from mmdet.datasets.pipelines.loading import LoadPanopticAnnotations

try:
    from panopticapi.utils import rgb2id
except ImportError:
    rgb2id = None


@PIPELINES.register_module()
class RelsFormatBundle(DefaultFormatBundle):
    """Transfer gt_rels to tensor too."""
    def __call__(self, results):
        results = super().__call__(results)
        if 'gt_rels' in results:
            results['gt_rels'] = DC(to_tensor(results['gt_rels']))

        return results


@PIPELINES.register_module()
class LoadSceneGraphAnnotations(LoadAnnotations):
    def __init__(
            self,
            with_bbox=True,
            with_label=True,
            with_mask=False,
            with_seg=False,
            poly2mask=True,
            file_client_args=dict(backend='disk'),
            # New args
            with_rel=False,
    ):
        super().__init__(
            with_bbox=with_bbox,
            with_label=with_label,
            with_mask=with_mask,
            with_seg=with_seg,
            poly2mask=poly2mask,
            file_client_args=dict(backend='disk'),
        )
        self.with_rel = with_rel

    def _load_rels(self, results):
        ann_info = results['ann_info']
        results['gt_rels'] = ann_info['rels']
        results['gt_relmaps'] = ann_info['rel_maps']

        assert 'rel_fields' in results

        results['rel_fields'] += ['gt_rels', 'gt_relmaps']
        return results

    def __call__(self, results):
        results = super().__call__(results)

        if self.with_rel:
            results = self._load_rels(results)

        return results

    def __repr__(self):
        repr_str = super().__repr__()

        repr_str += f', with_rel={self.with_rel})'

        return repr_str


@PIPELINES.register_module()
class LoadPanopticSceneGraphAnnotations(LoadPanopticAnnotations):
    def __init__(
            self,
            with_bbox=True,
            with_label=True,
            with_mask=True,
            with_seg=True,
            file_client_args=dict(backend='disk'),
            # New args
            with_rel=False,
    ):
        super().__init__(
            with_bbox=with_bbox,
            with_label=with_label,
            with_mask=with_mask,
            with_seg=with_seg,
            file_client_args=dict(backend='disk'),
        )
        self.with_rel = with_rel

    def _load_rels(self, results):
        ann_info = results['ann_info']
        results['gt_rels'] = ann_info['rels']
        results['gt_relmaps'] = ann_info['rel_maps']

        assert 'rel_fields' in results

        results['rel_fields'] += ['gt_rels', 'gt_relmaps']
        return results

    def _load_masks_and_semantic_segs(self, results):
        """Private function to load mask and semantic segmentation annotations.

        In gt_semantic_seg, the foreground label is from `0` to
        `num_things - 1`, the background label is from `num_things` to
        `num_things + num_stuff - 1`, 255 means the ignored label (`VOID`).

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded mask and semantic segmentation
                annotations. `BitmapMasks` is used for mask annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        filename = osp.join(results['seg_prefix'],
                            results['ann_info']['seg_map'])
        img_bytes = self.file_client.get(filename)
        pan_png = mmcv.imfrombytes(img_bytes,
                                   flag='color',
                                   channel_order='rgb').squeeze()
        pan_png = rgb2id(pan_png)

        gt_masks = []
        gt_seg = np.zeros_like(pan_png) + 255  # 255 as ignore

        for mask_info in results['ann_info']['masks']:
            mask = (pan_png == mask_info['id'])
            gt_seg = np.where(mask, mask_info['category'], gt_seg)

            # # The legal thing masks
            # if mask_info.get('is_thing'):
            #     gt_masks.append(mask.astype(np.uint8))
            gt_masks.append(mask.astype(np.uint8))  # get all masks

        if self.with_mask:
            h, w = results['img_info']['height'], results['img_info']['width']
            gt_masks = BitmapMasks(gt_masks, h, w)
            # print('origin_size')
            # print(gt_masks)
            results['gt_masks'] = gt_masks
            results['mask_fields'].append('gt_masks')

        if self.with_seg:
            results['gt_semantic_seg'] = gt_seg
            results['seg_fields'].append('gt_semantic_seg')
        return results

    def __call__(self, results):
        results = super().__call__(results)

        if self.with_rel:
            results = self._load_rels(results)

        return results

    def __repr__(self):
        repr_str = super().__repr__()

        repr_str += f', with_rel={self.with_rel})'

        return repr_str
