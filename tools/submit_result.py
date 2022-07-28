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
        rels = result.rels #done
        masks = result.masks
        height, width, channel = result.img_shape #done

        img = np.full(masks.shape[1:3], 0)
        for i in range(masks.shape[0]):
            r,g,b = random.choices(range(0,255), k=3)
            coloring_mask = 1 * np.vstack([[masks[i]]]*3)
            for j, color in enumerate([r,g,b]):
                coloring_mask[j,:,:] = coloring_mask[j,:,:] * color
            img = img + coloring_mask
        
        image_path = 'submission/images/%d.png'%idx
        PIL.Image.fromarray(np.array(img).astype(np.uint8).transpose(1,2,0)).save(image_path)

        segments_info = []
        for i, (label, mask) in enumerate(zip(labels, masks)):
            r,g,b = random.choices(range(0,255), k=3)
            coloring_mask = 1 * np.vstack([[masks[i]]]*3)
            for j, color in enumerate([r,g,b]):
                coloring_mask[j,:,:] = coloring_mask[j,:,:] * color
            img = img + coloring_mask

            segment = dict(
                category_id=label,
                id=rgb2id((r,g,b))
            )
            segments_info.append(segment)

        single_result_dict = dict(
            relations=rels.astype(np.int32).tolist(),
            height=height,
            width=width,
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
        pan_seg_img = read_image(pan_seg_filename,format='RGB')
        pan_seg_img = pan_seg_img.copy()  # (H, W, 3)
        seg_map = rgb2id(pan_seg_img)

        segments_info = single_result_dict['segments_info']
        num_obj = len(segments_info)

        # get separate masks
        labels = []
        masks = []
        for _, s in enumerate(segments_info):
            label = s['category_id']
            labels.append(label)
            masks.append(seg_map == s['id'])

        rel_array = np.asarray(single_result_dict['relations'])
        rel_dists = np.zeros((num_obj*(num_obj-1), 57))

        crt = 0
        for sub in range(num_obj):
            for obj in range(num_obj):
                for rel in rel_array:
                    rel_sub, rel_obj, predicate = rel
                    if rel_sub == sub and rel_obj == obj:
                        rel_dists[crt, predicate] += 1
                if sub != obj:
                    crt+=1

        result = Result(
            rels=rel_array,
            rel_pair_idxes=rel_array[:, :2],
            img_shape=(single_result_dict['height'], single_result_dict['width'], 3),
            masks=masks,
            labels=labels,
            rel_dists=rel_dists,
        )
        results.append(result)
    
    return results

