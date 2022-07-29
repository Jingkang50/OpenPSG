from PIL import Image
import numpy as np
import random
import json
import PIL
from openpsg.models.relation_heads.approaches import Result
from detectron2.data.detection_utils import read_image
from panopticapi.utils import rgb2id

def save_results(results):
    all_img_dicts = []
    for idx, result in enumerate(results):
        if not isinstance(result, Result):
            continue

        labels = result.labels
        rels = result.rels
        masks = result.masks

        segments_info = []
        img = np.full(masks.shape[1:3], 0)
        for i, (label, mask) in enumerate(zip(labels, masks)):
            r,g,b = random.choices(range(0,255), k=3)
            coloring_mask = 1 * np.vstack([[masks[i]]]*3)
            for j, color in enumerate([r,g,b]):
                coloring_mask[j,:,:] = coloring_mask[j,:,:] * color
            img = img + coloring_mask

            segment = dict(
                category_id=int(label),
                id=rgb2id((r,g,b))
            )
            segments_info.append(segment)

        image_path = 'submission/images/%d.png'%idx
        # image_array = np.uint8(img).transpose((2,1,0))
        image_array = np.uint8(img).transpose((1,2,0))
        PIL.Image.fromarray(image_array).save(image_path)

        single_result_dict = dict(
            relations=rels.astype(np.int32).tolist(),
            segments_info=segments_info,
            pan_seg_file_name=image_path,
        )

        all_img_dicts.append(single_result_dict)

    with open('submission/submission_result.json', 'w') as outfile:
        json.dump(all_img_dicts, outfile, default=str)


def load_results(filename):
    with open(filename) as infile:
        all_img_dicts = json.load(infile)

    results=[]
    for single_result_dict in all_img_dicts:
        pan_seg_filename = single_result_dict['pan_seg_file_name']
        # pan_seg_img = np.array(Image.open(pan_seg_filename)).transpose((1, 0, 2))
        pan_seg_img = np.array(Image.open(pan_seg_filename))
        pan_seg_img = pan_seg_img.copy()  # (H, W, 3)
        seg_map = rgb2id(pan_seg_img)

        segments_info = single_result_dict['segments_info']
        num_obj = len(segments_info)

        # get separate masks
        labels = []
        masks = []
        for _, s in enumerate(segments_info):
            label = s['category_id']
            labels.append(label) #TODO:1-index for gt?
            masks.append(seg_map == s['id'])

        rel_array = np.asarray(single_result_dict['relations'])
        if len(rel_array) > 20:
            rel_array = rel_array[:20]
        rel_dists = np.zeros((len(rel_array), 57))
        for idx_rel, rel in enumerate(rel_array):
            rel_dists[idx_rel, rel[2]] += 1 # TODO:1-index for gt?

        result = Result(
            rels=rel_array,
            rel_pair_idxes=rel_array[:, :2],
            masks=masks,
            labels=np.asarray(labels),
            rel_dists=rel_dists,
            refine_bboxes=np.ones((num_obj, 5)),
        )
        results.append(result)
    
    return results

