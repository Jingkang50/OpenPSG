import json
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import xmltodict
from detectron2.data.transforms import ScaleTransform
from tqdm import tqdm


def x1y1wh_to_xyxy(xywh):
    """Convert [x1 y1 w h] box format to [x1 y1 x2 y2] format."""
    if isinstance(xywh, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xywh) == 4
        x1, y1 = xywh[0], xywh[1]
        x2 = x1 + np.maximum(0.0, xywh[2] - 1.0)
        y2 = y1 + np.maximum(0.0, xywh[3] - 1.0)

        xyxy = [x1, y1, x2, y2]

        return [int(c) for c in xyxy]

    elif isinstance(xywh, np.ndarray):
        # Multiple boxes given as a 2D ndarray
        return np.hstack(
            (xywh[:, 0:2], xywh[:, 0:2] + np.maximum(0, xywh[:, 2:4] - 1)))
    else:
        raise TypeError('Argument xywh must be a list, tuple, or numpy array.')


def xyxy_to_x1y1wh(
        xyxy: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """Convert [x1 y1 x2 y2] box format to [x1 y1 w h] format."""
    x1, y1, x2, y2 = xyxy

    w = x2 - x1
    h = y2 - y1

    return [x1, y1, w, h]


def xcycwh_to_xyxy(
        xywh: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """Convert [xc yc w h] box format to [x1 y1 x2 y2] format."""
    xc, yc, w, h = xywh

    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2

    xyxy = [x1, y1, x2, y2]

    return [int(c) for c in xyxy]


def xyxy_to_xcycwh(
        xyxy: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """Convert [x1 y1 x2 y2] box format to [xc yc w h] format."""
    x1, y1, x2, y2 = xyxy

    w = x2 - x1
    h = y2 - y1

    xc = x1 + w / 2
    yc = y1 + h / 2

    xywh = [xc, yc, w, h]

    return [int(c) for c in xywh]


def segment_to_bbox(seg_mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Parameters
    ----------
    seg_mask : np.ndarray
        Boolean mask containing segmented object, (H, W)

    Returns
    -------
    Tuple[int, int, int, int]
        [x1, y1, x2, y2]
    """
    ind = np.nonzero(seg_mask.any(axis=0))[0]
    x1, x2 = ind[0], ind[-1]
    # Indices of non-empty rows
    ind = np.nonzero(seg_mask.any(axis=1))[0]
    y1, y2 = ind[0], ind[-1]

    bbox = [x1, y1, x2, y2]

    return bbox


def process_vg_bbox(og_height: int,
                    og_width: int,
                    bbox: Tuple[int, int, int, int],
                    resize: int = 1024) -> Tuple[int, int, int, int]:
    """For VG_150 dataset. Rescales the bbox coords back to the original.

    Parameters
    ----------
    og_height : int
        Original image height
    og_width : int
        Original image width
    bbox : Tuple[int, int, int, int]
        In XYXY format
    resize : int, optional
        The dim that the image was resized to in VG_150, by default 1024

    Returns
    -------
    Tuple[int, int, int, int]
        Original bbox in XYXY format
    """
    if og_height > og_width:
        height = resize
        width = int(resize / og_height * og_width)
    else:
        width = resize
        height = int(resize / og_width * og_height)

    transform = ScaleTransform(height, width, og_height, og_width)
    og_bbox = transform.apply_box(np.array(bbox))[0].tolist()

    return og_bbox


def resize_bbox(old_height: int, old_width: int,
                bbox: Tuple[int, int, int, int], resize: int):
    if old_height > old_width:
        new_height = resize
        new_width = int(resize / old_height * old_width)
    else:
        new_width = resize
        new_height = int(resize / old_width * old_height)

    transform = ScaleTransform(old_height, old_width, new_height, new_width)
    new_bbox = transform.apply_box(np.array(bbox))[0].tolist()

    return [int(b) for b in new_bbox]


def load_json(path: Path):
    with path.open() as f:
        data = json.load(f)

    return data


def save_json(obj, path: Path):
    with path.open('w') as f:
        json.dump(obj, f)


def load_xml(path: Path):
    with path.open() as f:
        data = xmltodict.parse(f.read())

    return data


def process_vg_150_to_detectron(
    img_json_path: Path,
    metadata_json_path: Path,
    data_path: Path,
    output_dir: Path,
    val_split_idx: int = 75651,
):
    img_data = load_json(img_json_path)
    vg_metadata = load_json(metadata_json_path)

    print(f'{len(img_data)} Annotations Found.')

    # Extract VG Categories ###################################################
    obj_cats = sorted(list(vg_metadata['idx_to_label'].values()))
    attr_cats = sorted(list(vg_metadata['idx_to_attribute'].values()))
    rel_cats = sorted(list(vg_metadata['idx_to_predicate'].values()))

    print(f'{len(obj_cats)} Object Categories')
    print(f'{len(attr_cats)} Attribute Categories')
    print(f'{len(rel_cats)} Relation Categories')

    # Save categories to JSON file
    obj_cats_save_path = output_dir / 'object_categories.json'
    save_json(obj_cats, obj_cats_save_path)
    print(f'Object categories saved to {obj_cats_save_path}')

    attr_cats_save_path = output_dir / 'attribute_categories.json'
    save_json(attr_cats, attr_cats_save_path)
    print(f'Attribute categories saved to {attr_cats_save_path}')

    rel_cats_save_path = output_dir / 'relation_categories.json'
    save_json(rel_cats, rel_cats_save_path)
    print(f'Relation categories saved to {rel_cats_save_path}')

    obj_to_id = {obj: i for i, obj in enumerate(obj_cats)}
    attr_to_id = {attr: i for i, attr in enumerate(attr_cats)}
    rel_to_id = {rel: i for i, rel in enumerate(rel_cats)}

    # Process to Detectron2 Format ############################################
    # Extract data from h5py
    with h5py.File(data_path, 'r') as f:
        img_to_first_box = f['img_to_first_box'][:]  # (N,)
        img_to_last_box = f['img_to_last_box'][:]  # (N,)
        img_to_first_rel = f['img_to_first_rel'][:]  # (N,)
        img_to_last_rel = f['img_to_last_rel'][:]  # (N,)

        attributes = f['attributes'][:]  # (N_b, 10)
        boxes_1024 = f['boxes_1024'][:]  # (N_b, 4)
        labels = f['labels'][:]  # (N_b, 1)

        relationships = f['relationships'][:]  # (N_r, 2)
        predicates = f['predicates'][:]  # (N_r, 1)

        split = f['split'][:]  # (N,)

    for name, (start_idx, end_idx) in zip(
        ['train_data', 'val_data'],
        [(0, val_split_idx), (val_split_idx, len(img_to_first_box))],
    ):

        output_dicts = []

        for img_idx in tqdm(range(start_idx, end_idx)):
            out = {}

            img = img_data[img_idx]

            # Store img info
            image_id = img['image_id']
            # FIXME Temp change
            # out["file_name"] = f"{image_id}.jpg"
            out['file_name'] = img['file_name']
            out['height'] = img['height']
            out['width'] = img['width']
            out['image_id'] = str(image_id)

            # Store bbox
            out['annotations'] = []

            # Keep an obj_id to idx mapping
            obj_id_to_idx = {}

            first_box_idx = img_to_first_box[img_idx]
            last_box_idx = img_to_last_box[img_idx]

            # Store per box annotations
            for i, box_idx in enumerate(range(first_box_idx,
                                              last_box_idx + 1)):
                anno = {}

                # Store bbox coords
                bbox = boxes_1024[box_idx].tolist()
                # FIXME Take note of box format
                bbox = xcycwh_to_xyxy(bbox)
                bbox = [int(b) for b in bbox]

                # Transform to original coords
                bbox = process_vg_bbox(
                    img['height'],
                    img['width'],
                    bbox,
                )

                anno['bbox'] = bbox
                anno['bbox_mode'] = 0

                # Store obj id
                old_obj_id = labels[box_idx][0]
                obj_name = vg_metadata['idx_to_label'][str(old_obj_id)]
                obj_id = obj_to_id[obj_name]
                anno['category_id'] = obj_id

                # Store attributes
                anno['attribute_ids'] = []

                attrs = attributes[box_idx].tolist()
                for old_attr_id in attrs:
                    if old_attr_id != 0:
                        attr_name = vg_metadata['idx_to_attribute'][str(
                            old_attr_id)]
                        attr_id = attr_to_id[attr_name]

                        anno['attribute_ids'].append(attr_id)

                obj_id_to_idx[box_idx] = i
                out['annotations'].append(anno)

            # Store relations
            out['relations'] = []

            first_rel_idx = img_to_first_rel[img_idx]
            last_rel_idx = img_to_last_rel[img_idx]

            # If there exist relationships
            if first_rel_idx != -1 and last_rel_idx != -1:
                for rel_idx in range(first_rel_idx, last_rel_idx + 1):
                    old_rel_id = predicates[rel_idx][0]
                    rel_name = vg_metadata['idx_to_predicate'][str(old_rel_id)]
                    rel_id = rel_to_id[rel_name]

                    s_idx = obj_id_to_idx[relationships[rel_idx][0]]
                    o_idx = obj_id_to_idx[relationships[rel_idx][1]]

                    out['relations'].append([s_idx, o_idx, rel_id])

            output_dicts.append(out)

        # Save data to a JSON file
        data_save_path = output_dir / f'{name}.json'
        print(f'{name} in Detectron2 format saved to {data_save_path}')
        save_json(output_dicts, data_save_path)


def process_vrr_vg_to_detectron(
    data_dir: Path,
    output_dir: Path,
):
    vrr_xmls = list(data_dir.glob('*.xml'))
    print(f'{len(vrr_xmls)} Annotations Found.')

    # Extract VRR Categories ###################################################
    obj_cats = set()
    attr_cats = set()
    rel_cats = set()

    for xml in tqdm(vrr_xmls):
        data = load_xml(xml)['annotation']

        for obj in data['object']:
            obj_cats.add(obj['name'])

            if 'attribute' in obj:
                attr = obj['attribute']

                if isinstance(attr, str):
                    attr_cats.add(attr)
                elif isinstance(attr, list):
                    attr_cats.update(attr)
                else:
                    raise Exception('Unknown attribute type!')

        relations = data['relation']
        if isinstance(relations, dict):
            rel_cats.add(relations['predicate'])
        elif isinstance(relations, list):
            rel_cats.update(r['predicate'] for r in relations)
        else:
            raise Exception('Unknown relation type!')

    obj_cats = sorted(list(obj_cats))
    attr_cats = sorted(list(attr_cats))
    rel_cats = sorted(list(rel_cats))

    print(f'{len(obj_cats)} Object Categories')
    print(f'{len(attr_cats)} Attribute Categories')
    print(f'{len(rel_cats)} Relation Categories')

    # Save categories to JSON file
    obj_cats_save_path = output_dir / 'object_categories.json'
    save_json(obj_cats, obj_cats_save_path)
    print(f'Object categories saved to {obj_cats_save_path}')

    attr_cats_save_path = output_dir / 'attribute_categories.json'
    save_json(attr_cats, attr_cats_save_path)
    print(f'Attribute categories saved to {attr_cats_save_path}')

    rel_cats_save_path = output_dir / 'relation_categories.json'
    save_json(rel_cats, rel_cats_save_path)
    print(f'Relation categories saved to {rel_cats_save_path}')

    obj_to_id = {obj: i for i, obj in enumerate(obj_cats)}
    attr_to_id = {attr: i for i, attr in enumerate(attr_cats)}
    rel_to_id = {rel: i for i, rel in enumerate(rel_cats)}

    # Process to Detectron2 Format #############################################
    # FIXME Just convert all xmls to json first?
    output_dicts = []

    for xml in tqdm(vrr_xmls):
        out = {}

        data = load_xml(xml)['annotation']

        out['file_name'] = data['filename']
        out['height'] = int(data['size']['height'])
        out['width'] = int(data['size']['width'])
        out['image_id'] = str(data['source']['image_id'])

        out['annotations'] = []
        out['relations'] = []
        # Keep an obj_id to idx mapping
        obj_id_to_idx = {}

        for i, obj in enumerate(data['object']):
            anno = {}

            # Store bbox
            bbox = obj['bndbox']
            anno['bbox'] = [
                float(bbox['xmin']),
                float(bbox['ymin']),
                float(bbox['xmax']),
                float(bbox['ymax']),
            ]
            anno['bbox_mode'] = 0
            anno['category_id'] = obj_to_id[obj['name']]

            # Store attributes
            anno['attribute_ids'] = []

            if 'attribute' in obj:
                attr = obj['attribute']
                if isinstance(attr, str):
                    attr = [attr]

                anno['attribute_ids'].extend([attr_to_id[a] for a in attr])

            obj_id_to_idx[obj['object_id']] = i
            out['annotations'].append(anno)

        # Store relations
        relations = data['relation']
        if isinstance(relations, dict):
            relations = [relations]

        for rel in relations:
            s_idx = obj_id_to_idx[rel['subject_id']]
            o_idx = obj_id_to_idx[rel['object_id']]
            rel_id = rel_to_id[rel['predicate']]

            out['relations'].append([s_idx, o_idx, rel_id])

        output_dicts.append(out)

    # Save data to a JSON file
    data_save_path = output_dir / f'data.json'
    print(f'Detectron2 format saved to {data_save_path}')
    save_json(output_dicts, data_save_path)


def process_coco_panoptic_to_detectron(
    train_json_path: Path,
    val_json_path: Path,
    panoptic_img_train_dir: Path,
    panoptic_img_val_dir: Path,
    output_dir: Path,
):
    train_data = load_json(train_json_path)
    val_data = load_json(val_json_path)

    # Extract COCO Thing / Stuff Categories ####################################
    cats_dict = train_data['categories']

    old_id_to_cat_data = {cat['id']: cat for cat in cats_dict}

    thing_cats = sorted([cat['name'] for cat in cats_dict if cat['isthing']])
    stuff_cats = sorted(
        [cat['name'] for cat in cats_dict if not cat['isthing']])

    thing_cats_save_path = output_dir / 'thing_categories.json'
    save_json(thing_cats, thing_cats_save_path)
    print(f'Thing categories saved to {thing_cats_save_path}')

    stuff_cats_save_path = output_dir / 'stuff_categories.json'
    save_json(stuff_cats, stuff_cats_save_path)
    print(f'Attribute categories saved to {stuff_cats_save_path}')

    # Mapping from old_id -> new_id
    new_thing_cat_id_map = {
        cat['id']: thing_cats.index(cat['name'])
        for cat in cats_dict if cat['isthing']
    }
    new_stuff_cat_id_map = {
        cat['id']: stuff_cats.index(cat['name'])
        for cat in cats_dict if not cat['isthing']
    }

    # Process to Detectron2 Format #############################################
    for name, data, img_dir, panoptic_dir in zip(
        ['train_data', 'val_data'],
        [train_data, val_data],
        ['train2017', 'val2017'],
        [panoptic_img_train_dir, panoptic_img_val_dir],
    ):
        print(f'Processing {name}...')

        # Mapping from image_id -> image_data
        img_id_to_img_data = {}

        for img in data['images']:
            img_id_to_img_data[img['id']] = img

        output_dicts = []

        for anno in tqdm(data['annotations']):
            out = {}

            img = img_id_to_img_data[anno['image_id']]

            out['file_name'] = img_dir + '/' + img['file_name']
            out['height'] = img['height']
            out['width'] = img['width']
            out['image_id'] = str(anno['image_id'])

            out['pan_seg_file_name'] = str(panoptic_dir / anno['file_name'])

            out['segments_info'] = []

            for segment in anno['segments_info']:
                isthing = old_id_to_cat_data[segment['category_id']]['isthing']
                id_map = new_thing_cat_id_map if isthing else new_stuff_cat_id_map
                category_id = id_map[segment['category_id']]

                out['segments_info'].append({
                    'id': segment['id'],
                    'category_id': category_id,
                    'iscrowd': segment['iscrowd'],
                    'isthing': isthing,
                })

            output_dicts.append(out)

        # Save data to a JSON file
        data_save_path = output_dir / f'{name}.json'
        print(f'{name} in Detectron2 format saved to {data_save_path}')
        save_json(output_dicts, data_save_path)


def process_gqa_to_detectron(
    train_scene_graphs_path: Path,
    val_scene_graphs_path: Path,
    output_dir: Path,
):
    train_data = load_json(train_scene_graphs_path)
    val_data = load_json(val_scene_graphs_path)

    print(f'{len(train_data)} train images')
    print(f'{len(val_data)} val images')

    # Extract GQA Categories ###################################################
    obj_cats = set()
    attr_cats = set()
    rel_cats = set()

    # Iterate through train data
    print('Extracting categories from train data...')
    for img_id, img in tqdm(train_data.items()):

        for obj_id, obj in img['objects'].items():
            obj_cats.add(obj['name'])
            attr_cats.update(obj['attributes'])

            for rel in obj['relations']:
                rel_cats.add(rel['name'])

    # Iterate through val data
    print('Extracting categories from val data...')
    for img_id, img in tqdm(val_data.items()):

        for obj_id, obj in img['objects'].items():
            obj_cats.add(obj['name'])
            attr_cats.update(obj['attributes'])

            for rel in obj['relations']:
                rel_cats.add(rel['name'])

    obj_cats = sorted(list(obj_cats))
    attr_cats = sorted(list(attr_cats))
    rel_cats = sorted(list(rel_cats))

    print(f'{len(obj_cats)} Object Categories')
    print(f'{len(attr_cats)} Attribute Categories')
    print(f'{len(rel_cats)} Relation Categories')

    # Save categories to JSON file
    obj_cats_save_path = output_dir / 'object_categories.json'
    save_json(obj_cats, obj_cats_save_path)
    print(f'Object categories saved to {obj_cats_save_path}')

    attr_cats_save_path = output_dir / 'attribute_categories.json'
    save_json(attr_cats, attr_cats_save_path)
    print(f'Attribute categories saved to {attr_cats_save_path}')

    rel_cats_save_path = output_dir / 'relation_categories.json'
    save_json(rel_cats, rel_cats_save_path)
    print(f'Relation categories saved to {rel_cats_save_path}')

    obj_to_id = {obj: i for i, obj in enumerate(obj_cats)}
    attr_to_id = {attr: i for i, attr in enumerate(attr_cats)}
    rel_to_id = {rel: i for i, rel in enumerate(rel_cats)}

    # Process to Detectron2 Format #############################################
    # Process both train and val data
    for name, data in zip(['train_data', 'val_data'], [train_data, val_data]):
        print(f'Processing {name}...')

        output_dicts = []

        for img_id, img in tqdm(data.items()):
            # Processed dict for each image
            out = {}

            out['file_name'] = f'{img_id}.jpg'
            out['height'] = img['height']
            out['width'] = img['width']
            out['image_id'] = str(img_id)

            # Auxiliary information
            out['location'] = img['location'] if 'location' in img else ''
            out['weather'] = img['weather'] if 'weather' in img else ''

            out['annotations'] = []
            out['relations'] = []
            # Keep an obj_id to idx mapping
            obj_id_to_idx = {}

            for i, (obj_id, obj) in enumerate(img['objects'].items()):
                anno = {}

                # Store bbox
                anno['bbox'] = [obj['x'], obj['y'], obj['w'], obj['h']]
                anno['bbox_mode'] = 1
                anno['category_id'] = obj_to_id[obj['name']]

                # Store attributes
                anno['attribute_ids'] = [
                    attr_to_id[attr_name] for attr_name in obj['attributes']
                ]

                obj_id_to_idx[obj_id] = i
                out['annotations'].append(anno)

                # Store relations
                for rel in obj['relations']:
                    out['relations'].append(
                        [obj_id, rel['object'], rel_to_id[rel['name']]])

            # Convert obj_ids to idx
            for rel in out['relations']:
                rel[0] = obj_id_to_idx[rel[0]]
                rel[1] = obj_id_to_idx[rel[1]]

            output_dicts.append(out)

        # Save data to a JSON file
        data_save_path = output_dir / f'{name}.json'
        print(f'{name} in Detectron2 format saved to {data_save_path}')
        save_json(output_dicts, data_save_path)
