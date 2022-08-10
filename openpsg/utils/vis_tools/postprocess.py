from collections import Counter, defaultdict
from pathlib import Path

import cv2
import h5py
import numpy as np
from detectron2.data import DatasetCatalog
from detectron2.data.detection_utils import read_image
from detectron2.data.transforms import ScaleTransform
from detectron2.structures import BitMasks, Boxes, pairwise_ioa, pairwise_iou
from panopticapi.utils import rgb2id
from tqdm import tqdm
from .datasets import (init_coco_panoptic_dataset, init_gqa_dataset,
                                init_vg_dataset, init_vrr_vg_dataset)
from .preprocess import (load_json, resize_bbox, save_json,
                                  segment_to_bbox, x1y1wh_to_xyxy,
                                  xyxy_to_xcycwh)


def process_gqa_and_coco(
        gqa_img_dir: Path,
        gqa_img_rs_dir: Path,
        output_dir: Path,
        vg_id_to_coco_id_path: Path = Path('data/vg/vg_id_to_coco_id.json'),
):
    init_coco_panoptic_dataset()
    init_gqa_dataset()

    # Get and combine datasets
    coco_train_dataset = DatasetCatalog.get('coco_train')
    coco_val_dataset = DatasetCatalog.get('coco_val')
    coco_dataset = coco_train_dataset + coco_val_dataset

    gqa_train_dataset = DatasetCatalog.get('gqa_train')
    gqa_val_dataset = DatasetCatalog.get('gqa_val')
    gqa_dataset = gqa_train_dataset + gqa_val_dataset

    # Check GQA overlap with COCO
    vg_id_to_coco_id = load_json(vg_id_to_coco_id_path)
    vg_coco_ids = set(vg_id_to_coco_id.keys())
    gqa_ids = set(d['image_id'] for d in gqa_dataset)

    vg_overlap_ids = vg_coco_ids & gqa_ids

    # Merge GQA and COCO
    id_to_coco_data = {d['image_id']: d for d in coco_dataset}

    merged_dataset = []

    for gqa_d in tqdm(gqa_dataset):
        vg_id = gqa_d['image_id']

        if vg_id in vg_id_to_coco_id:
            coco_id = vg_id_to_coco_id[vg_id]

            merged_dataset.append((gqa_d, id_to_coco_data[coco_id]))

    # Resize GQA images to COCO dimensions
    for gqa_d, coco_d in tqdm(merged_dataset):
        # Resize image
        img = cv2.imread(str(gqa_img_dir / gqa_d['file_name']))
        img_resized = cv2.resize(
            img,
            (coco_d['width'], coco_d['height']),
        )
        cv2.imwrite(str(gqa_img_rs_dir / gqa_d['file_name']), img_resized)

        # Resize bboxes
        for anno in gqa_d['annotations']:
            transform = ScaleTransform(
                gqa_d['height'],
                gqa_d['width'],
                coco_d['height'],
                coco_d['width'],
            )
            bbox = x1y1wh_to_xyxy(anno['bbox'])
            bbox_resized = transform.apply_box(np.array(bbox))[0].tolist()

            anno['bbox'] = bbox_resized
            anno['bbox_mode'] = 0

        gqa_d['height'] = coco_d['height']
        gqa_d['width'] = coco_d['width']

        # Add bbox info for panoptic coco
        seg_map = read_image(coco_d['pan_seg_file_name'], format='RGB')
        # Convert to segment ids
        seg_map = rgb2id(seg_map)

        for s in coco_d['segments_info']:
            curr_seg = seg_map == s['id']
            # [x1, y1, x2, y2]
            s['bbox'] = segment_to_bbox(curr_seg)

    print(f'Resized images saved to {gqa_img_rs_dir}')

    # Saved merged and processed dataset
    save_path = output_dir / 'data.json'
    print(f'Merged dataset saved to {save_path}')
    save_json(merged_dataset, save_path)


def process_vrr_and_coco(
        vrr_img_dir: Path,
        vrr_img_rs_dir: Path,
        output_path: Path,
        vg_id_to_coco_id_path: Path = Path('data/vg/vg_id_to_coco_id.json'),
):
    init_coco_panoptic_dataset()
    init_vrr_vg_dataset()

    # Get and combine datasets
    coco_train_dataset = DatasetCatalog.get('coco_train')
    coco_val_dataset = DatasetCatalog.get('coco_val')
    coco_dataset = coco_train_dataset + coco_val_dataset

    vrr_dataset = DatasetCatalog.get('vrr_vg')

    # Check GQA overlap with COCO
    vg_id_to_coco_id = load_json(vg_id_to_coco_id_path)
    vg_coco_ids = set(vg_id_to_coco_id.keys())
    vrr_ids = set(d['image_id'] for d in vrr_dataset)

    vg_overlap_ids = vg_coco_ids & vrr_ids

    # Merge GQA and COCO
    id_to_coco_data = {d['image_id']: d for d in coco_dataset}

    merged_dataset = []

    for vrr_d in tqdm(vrr_dataset):
        vg_id = vrr_d['image_id']

        if vg_id in vg_id_to_coco_id:
            coco_id = vg_id_to_coco_id[vg_id]

            merged_dataset.append((vrr_d, id_to_coco_data[coco_id]))

    # Resize GQA images to COCO dimensions
    for vrr_d, coco_d in tqdm(merged_dataset):
        # Resize image
        img = cv2.imread(str(vrr_img_dir / vrr_d['file_name']))
        img_resized = cv2.resize(
            img,
            (coco_d['width'], coco_d['height']),
        )
        cv2.imwrite(str(vrr_img_rs_dir / vrr_d['file_name']), img_resized)

        # Resize bboxes
        for anno in vrr_d['annotations']:
            transform = ScaleTransform(
                vrr_d['height'],
                vrr_d['width'],
                coco_d['height'],
                coco_d['width'],
            )
            # bbox = x1y1wh_to_xyxy(anno["bbox"])
            bbox = anno['bbox']
            bbox_resized = transform.apply_box(np.array(bbox))[0].tolist()

            anno['bbox'] = bbox_resized
            anno['bbox_mode'] = 0

        vrr_d['height'] = coco_d['height']
        vrr_d['width'] = coco_d['width']

        # Add bbox info for panoptic coco
        seg_map = read_image(coco_d['pan_seg_file_name'], format='RGB')
        # Convert to segment ids
        seg_map = rgb2id(seg_map)

        for s in coco_d['segments_info']:
            curr_seg = seg_map == s['id']
            # [x1, y1, x2, y2]
            s['bbox'] = segment_to_bbox(curr_seg)

    print(f'Resized images saved to {vrr_img_rs_dir}')

    # Saved merged and processed dataset
    save_path = output_path
    print(f'Merged dataset saved to {save_path}')
    save_json(merged_dataset, save_path)


def process_vg_and_coco(
        vg_img_dir: Path,
        vg_img_rs_dir: Path,
        output_path: Path,
        vg_id_to_coco_id_path: Path = Path('data/vg/vg_id_to_coco_id.json'),
):
    init_coco_panoptic_dataset()
    init_vg_dataset()

    # Get and combine datasets
    coco_train_dataset = DatasetCatalog.get('coco_train')
    coco_val_dataset = DatasetCatalog.get('coco_val')
    coco_dataset = coco_train_dataset + coco_val_dataset

    vg_train_dataset = DatasetCatalog.get('vg_train')
    vg_val_dataset = DatasetCatalog.get('vg_val')
    vg_dataset = vg_train_dataset + vg_val_dataset

    # Check GQA overlap with COCO
    vg_id_to_coco_id = load_json(vg_id_to_coco_id_path)
    vg_coco_ids = set(vg_id_to_coco_id.keys())
    vg_ids = set(d['image_id'] for d in vg_dataset)

    vg_overlap_ids = vg_coco_ids & vg_ids

    # Merge GQA and COCO
    id_to_coco_data = {d['image_id']: d for d in coco_dataset}

    merged_dataset = []

    for vg_d in tqdm(vg_dataset):
        vg_id = vg_d['image_id']

        if vg_id in vg_id_to_coco_id:
            coco_id = vg_id_to_coco_id[vg_id]

            merged_dataset.append((vg_d, id_to_coco_data[coco_id]))

    #  Resize GQA images to COCO dimensions
    for vg_d, coco_d in tqdm(merged_dataset):
        # NOTE Resize image
        # img = cv2.imread(str(vg_img_dir / vg_d["file_name"]))
        # img_resized = cv2.resize(
        #     img,
        #     (coco_d["width"], coco_d["height"]),
        # )
        # cv2.imwrite(str(vg_img_rs_dir / vg_d["file_name"]), img_resized)

        # Resize bboxes
        for anno in vg_d['annotations']:
            transform = ScaleTransform(
                vg_d['height'],
                vg_d['width'],
                coco_d['height'],
                coco_d['width'],
            )
            # bbox = x1y1wh_to_xyxy(anno["bbox"])
            bbox = anno['bbox']
            bbox_resized = transform.apply_box(np.array(bbox))[0].tolist()

            anno['bbox'] = bbox_resized
            anno['bbox_mode'] = 0

        vg_d['height'] = coco_d['height']
        vg_d['width'] = coco_d['width']

        # Add bbox info for panoptic coco
        seg_map = read_image(coco_d['pan_seg_file_name'], format='RGB')
        # Convert to segment ids
        seg_map = rgb2id(seg_map)

        for s in coco_d['segments_info']:
            curr_seg = seg_map == s['id']
            # [x1, y1, x2, y2]
            s['bbox'] = segment_to_bbox(curr_seg)

    # print(f"Resized images saved to {vrr_img_rs_dir}")

    # Saved merged and processed dataset
    save_path = output_path
    print(f'Merged dataset saved to {save_path}')
    save_json(merged_dataset, save_path)


def compute_gqa_coco_overlap(output_path: Path):
    dataset = load_json(Path('data/psg/data.json'))

    gqa_obj_cats = load_json(Path('data/gqa/detectron/object_categories.json'))
    coco_thing_cats = load_json(
        Path('data/coco/detectron/thing_categories.json'))
    coco_stuff_cats = load_json(
        Path('data/coco/detectron/stuff_categories.json'))

    # For each GQA class, what is the average proportion of COCO classes
    # within its bounding box?
    # gqa_obj_id -> coco_obj_id -> prop
    out = defaultdict(lambda: defaultdict(float))

    # Counts the number of instance for each GQA class
    # gqa_id -> num_instances
    out_counts = defaultdict(int)

    for gqa_d, coco_d in tqdm(dataset):

        # Get seg_id to obj_id map
        seg_id_to_obj_id = {}

        for s in coco_d['segments_info']:
            obj_id = s['category_id']
            # Differentiate between thing and stuff classes
            if not s['isthing']:
                obj_id += 100

            seg_id_to_obj_id[s['id']] = obj_id

        # Load segment
        seg_map = read_image(coco_d['pan_seg_file_name'], format='RGB')
        # Convert to segment ids
        seg_map = rgb2id(seg_map)

        # Convert invalid pixels to -1
        seg_map[seg_map == 0] = -1

        # Convert to object ids
        for seg_id, obj_id in seg_id_to_obj_id.items():
            seg_map[seg_map == seg_id] = obj_id

        # Compute overlap for each bounding box
        for anno in gqa_d['annotations']:
            bbox = [int(c) for c in anno['bbox']]
            x1, y1, x2, y2 = bbox

            # Get region in segmentation map
            seg_bbox = seg_map[y1:y2, x1:x2]

            # Get id proportion
            unique, counts = np.unique(seg_bbox, return_counts=True)
            prop = counts / counts.sum()

            gqa_obj_id = anno['category_id']

            for coco_obj_id, p in zip(unique.tolist(), prop.tolist()):
                out[gqa_obj_id][coco_obj_id] += p

            # Update instance count of each GQA class
            out_counts[gqa_obj_id] += 1

    new_out = {}

    # Process ids to class names
    for gqa_id, props in tqdm(out.items()):

        gqa_name = gqa_obj_cats[gqa_id]

        new_props = {}

        for coco_id, p in props.items():
            if coco_id == -1:
                coco_name = 'NA'
            elif coco_id >= 100:
                coco_name = coco_stuff_cats[coco_id - 100]
            else:
                coco_name = coco_thing_cats[coco_id]

            # Normalize proportions
            new_props[coco_name] = p / out_counts[gqa_id]

        new_out[gqa_name] = new_props

    # Save final output
    save_json(new_out, output_path)


def compute_gqa_coco_overlap_norm(output_path: Path):
    """Given a GQA class and a COCO class, what is the average IoA of the COCO
    segment with the GQA bbox?"""
    dataset = load_json(Path('data/psg/data.json'))

    gqa_obj_cats = load_json(Path('data/gqa/detectron/object_categories.json'))
    coco_thing_cats = load_json(
        Path('data/coco/detectron/thing_categories.json'))
    coco_stuff_cats = load_json(
        Path('data/coco/detectron/stuff_categories.json'))

    # For each GQA class, what is the average proportion of COCO classes
    # within its bounding box?
    # gqa_obj_id -> coco_obj_id -> prop
    out = defaultdict(lambda: defaultdict(float))

    # NOTE Different normalizing scheme
    # Counts the number of instances for each GQA class for each COCO class
    # gqa_obj_id -> coco_obj_id -> num_instances
    out_counts = defaultdict(lambda: defaultdict(int))

    for gqa_d, coco_d in tqdm(dataset):

        # Get seg_id to obj_id map
        seg_id_to_obj_id = {}

        for s in coco_d['segments_info']:
            obj_id = s['category_id']
            # Differentiate between thing and stuff classes
            if not s['isthing']:
                obj_id += 100

            seg_id_to_obj_id[s['id']] = obj_id

        # Load segment
        seg_map = read_image(coco_d['pan_seg_file_name'], format='RGB')
        # Convert to segment ids
        seg_map = rgb2id(seg_map)

        # Convert invalid pixels to -1
        seg_map[seg_map == 0] = -1

        # Convert to object ids
        for seg_id, obj_id in seg_id_to_obj_id.items():
            seg_map[seg_map == seg_id] = obj_id

        # Compute overlap for each bounding box
        for anno in gqa_d['annotations']:
            bbox = [int(c) for c in anno['bbox']]
            x1, y1, x2, y2 = bbox

            # Get region in segmentation map
            seg_bbox = seg_map[y1:y2, x1:x2]

            # Get id proportion
            unique, counts = np.unique(seg_bbox, return_counts=True)
            prop = counts / counts.sum()

            gqa_obj_id = anno['category_id']

            for coco_obj_id, p in zip(unique.tolist(), prop.tolist()):
                out[gqa_obj_id][coco_obj_id] += p

                # Update instance count of each GQA class
                out_counts[gqa_obj_id][coco_obj_id] += 1

        # FIXME Sort by score

    new_out = {}

    # Process ids to class names
    for gqa_id, props in tqdm(out.items()):

        gqa_name = gqa_obj_cats[gqa_id]

        new_props = {}

        for coco_id, p in props.items():
            if coco_id == -1:
                coco_name = 'NA'
            elif coco_id >= 100:
                coco_name = coco_stuff_cats[coco_id - 100]
            else:
                coco_name = coco_thing_cats[coco_id]

            # Normalize proportions
            new_props[coco_name] = p / out_counts[gqa_id][coco_id]

        new_out[gqa_name] = new_props

    # Save final output
    save_json(new_out, output_path)


def compute_coco_gqa_overlap(
        output_path: Path,
        method: str = 'iou',  # One of {iou, ioa}
):
    """For each COCO class, compute the average IoU of its bbox with the bbox
    of all GQA bboxes in the image.

    For each COCO class, what is the average IoU of its bbox with the bbox of
    each GQA class?
    """
    dataset = load_json(Path('data/psg/data.json'))

    gqa_obj_cats = load_json(Path('data/gqa/detectron/object_categories.json'))
    coco_thing_cats = load_json(
        Path('data/coco/detectron/thing_categories.json'))
    coco_stuff_cats = load_json(
        Path('data/coco/detectron/stuff_categories.json'))

    # coco_obj_id -> gqa_obj_id -> avg_iou
    out = defaultdict(lambda: defaultdict(float))
    # coco_obj_id -> n_instances
    out_counts = defaultdict(int)

    for gqa_d, coco_d in tqdm(dataset):
        # Get gqa cats and bboxes
        gqa_annos = [(anno['category_id'], anno['bbox'])
                     for anno in gqa_d['annotations']]
        # NOTE What if no annotations
        if gqa_annos == []:
            continue

        gqa_cats, gqa_bboxes = zip(*gqa_annos)
        gqa_bboxes = Boxes(gqa_bboxes)

        # Get coco cats and bboxes
        coco_cats = []

        # Load segment
        seg_map = read_image(coco_d['pan_seg_file_name'], format='RGB')
        # Convert to segment ids
        seg_map = rgb2id(seg_map)
        # Convert to bitmasks
        bit_masks = np.zeros(
            (len(coco_d['segments_info']), coco_d['height'], coco_d['width']))
        for i, s in enumerate(coco_d['segments_info']):
            if s['isthing']:
                coco_cats.append(s['category_id'])
            else:
                coco_cats.append(s['category_id'] + 80)

            bit_masks[i] = seg_map == s['id']

        bit_masks = BitMasks(bit_masks)
        coco_bboxes = bit_masks.get_bounding_boxes()

        # Compute pairwise IoU / IoA
        # NOTE Change compute function here
        if method == 'iou':
            iou_matrix = pairwise_iou(gqa_bboxes, coco_bboxes)
        elif method == 'ioa':
            iou_matrix = pairwise_ioa(gqa_bboxes, coco_bboxes)

        n_gqa, n_coco = iou_matrix.shape

        # For each coco instance
        for i, coco_id in enumerate(coco_cats):
            # IoU of each gqa box with current coco instance
            ious = iou_matrix[:, i].tolist()

            for gqa_id, iou in zip(gqa_cats, ious):
                out[coco_id][gqa_id] += iou

            out_counts[coco_id] += 1

    # Process ids to class names
    new_out = {}

    for coco_id, ious in tqdm(out.items()):
        if coco_id >= 80:
            coco_name = coco_stuff_cats[coco_id - 80]
        else:
            coco_name = coco_thing_cats[coco_id]

        new_ious = {}

        for gqa_id, iou in ious.items():
            gqa_name = gqa_obj_cats[gqa_id]

            # Normalize proportions
            new_ious[gqa_name] = iou / out_counts[coco_id]

        new_out[coco_name] = new_ious

    # Save final output
    save_json(new_out, output_path)


def psg_to_kaihua(
    dataset_path: Path,
    thing_cats_path: Path,
    stuff_cats_path: Path,
    pred_cats_path: Path,
    output_dir: Path,
):
    dataset = load_json(dataset_path)
    pred_cats = load_json(pred_cats_path)

    thing_cats = load_json(thing_cats_path)
    stuff_cats = load_json(stuff_cats_path)
    obj_cats = thing_cats + stuff_cats

    # Generate metadata dicts
    idx_to_label = {str(i + 1): c for i, c in enumerate(obj_cats)}
    label_to_idx = {v: int(k) for k, v in idx_to_label.items()}
    idx_to_predicate = {str(i + 1): c for i, c in enumerate(pred_cats)}
    predicate_to_idx = {v: int(k) for k, v in idx_to_predicate.items()}

    all_predicates = []

    for d in dataset:
        rel_names = [pred_cats[r[2]] for r in d['relations']]
        all_predicates.extend(rel_names)

    predicate_count = dict(Counter(all_predicates))

    save_json(
        {
            'label_to_idx': label_to_idx,
            'idx_to_label': idx_to_label,
            'predicate_to_idx': predicate_to_idx,
            'idx_to_predicate': idx_to_predicate,
            'predicate_count': predicate_count,
            'attribute_count': {},
            'idx_to_attribute': {},
            'attribute_to_idx': {},
        },
        output_dir / 'PSG-dicts.json',
    )

    # Generate image metadata
    image_data = []

    for d in dataset:
        image_data.append({
            'file_name': d['file_name'],
            'image_id': d['vg_image_id'],
            'height': d['height'],
            'width': d['width'],
        })

    save_json(image_data, output_dir / 'image_data.json')

    # Generate hdf5 dataset
    n_objs = sum([len(d['segments_info']) for d in dataset])
    attributes = np.zeros((n_objs, 10))

    boxes_1024 = []
    boxes_512 = []
    img_to_first_box = []
    img_to_last_box = []
    labels = []

    predicates = []
    relationships = []
    img_to_first_rel = []
    img_to_last_rel = []

    box_idx = 0
    rel_idx = 0

    for d in tqdm(dataset):
        old_height = d['height']
        old_width = d['width']

        for r in d['relations']:
            s_i, o_i, pred_id = r
            predicates.append(pred_id + 1)
            relationships.append([box_idx + s_i, box_idx + o_i])

        img_to_first_rel.append(rel_idx)
        rel_idx += len(d['relations'])
        img_to_last_rel.append(rel_idx - 1)

        for s in d['segments_info']:
            # [x1, y1, x2, y2]
            # Compute new height, width
            boxes_1024.append(
                xyxy_to_xcycwh(
                    resize_bbox(old_height, old_width, s['bbox'], 1024)))
            boxes_512.append(
                xyxy_to_xcycwh(
                    resize_bbox(old_height, old_width, s['bbox'], 512)))

            labels.append(s['category_id'] +
                          1 if s['isthing'] else s['category_id'] + 81)

        img_to_first_box.append(box_idx)
        box_idx += len(d['segments_info'])
        img_to_last_box.append(box_idx - 1)

    boxes_1024 = np.array(boxes_1024)
    boxes_512 = np.array(boxes_512)
    img_to_first_box = np.array(img_to_first_box)
    img_to_last_box = np.array(img_to_last_box)
    labels = np.array(labels)

    predicates = np.array(predicates)
    relationships = np.array(relationships)
    img_to_first_rel = np.array(img_to_first_rel)
    img_to_last_rel = np.array(img_to_last_rel)

    labels = labels[..., None]
    predicates = predicates[..., None]

    # Get val indices
    coco_val_img_dir = Path('data/coco/val2017')
    coco_val_ids = set(
        [p.stem.lstrip('0') for p in coco_val_img_dir.glob('*.jpg')])

    split = [2 if d['image_id'] in coco_val_ids else 0 for d in dataset]
    split = np.array(split)

    # Save hdf5
    f = h5py.File(output_dir / 'PSG.h5', 'w')

    f.create_dataset('attributes', data=attributes, dtype='i8')
    f.create_dataset('boxes_1024', data=boxes_1024, dtype='i4')
    f.create_dataset('boxes_512', data=boxes_512, dtype='i4')
    f.create_dataset('img_to_first_box', data=img_to_first_box, dtype='i4')
    f.create_dataset('img_to_last_box', data=img_to_last_box, dtype='i4')
    f.create_dataset('img_to_first_rel', data=img_to_first_rel, dtype='i4')
    f.create_dataset('img_to_last_rel', data=img_to_last_rel, dtype='i4')
    f.create_dataset('labels', data=labels, dtype='i4')
    f.create_dataset('predicates', data=predicates, dtype='i4')
    f.create_dataset('relationships', data=relationships, dtype='i4')
    f.create_dataset('split', data=split, dtype='i4')

    f.close()
