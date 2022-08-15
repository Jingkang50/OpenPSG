from pathlib import Path

from detectron2.data import DatasetCatalog, MetadataCatalog
from .preprocess import load_json
from .viz import get_colormap

data_dir = Path('data')

# COCO
coco_dir = data_dir / 'coco'
# coco_img_train_dir = coco_dir / 'train2017'
# coco_img_val_dir = coco_dir / 'val2017'
coco_detectron_dir = coco_dir / 'detectron'

# VG
vg_dir = data_dir / 'vg'
vg_img_dir = vg_dir / 'VG_100K'
vg_detectron_dir = vg_dir / 'detectron'

# VRR
vrr_dir = data_dir / 'vrr_vg'
vrr_img_dir = vg_img_dir
vrr_detectron_dir = vrr_dir / 'detectron'

# GQA
gqa_dir = data_dir / 'gqa'
gqa_img_dir = gqa_dir / 'images'
gqa_detectron_dir = gqa_dir / 'detectron'


def init_coco_panoptic_dataset():
    def load_coco_train():
        return load_json(coco_detectron_dir / 'train_data.json')

    def load_coco_val():
        return load_json(coco_detectron_dir / 'val_data.json')

    DatasetCatalog.register('coco_train', load_coco_train)
    DatasetCatalog.register('coco_val', load_coco_val)

    thing_cats = load_json(coco_detectron_dir / 'thing_categories.json')
    stuff_cats = load_json(coco_detectron_dir / 'stuff_categories.json')

    for name in ['coco_train', 'coco_val']:
        metadata = MetadataCatalog.get(name)

        metadata.thing_classes = thing_cats
        metadata.stuff_classes = stuff_cats
        metadata.thing_colors = get_colormap(len(thing_cats))
        metadata.stuff_colors = get_colormap(len(stuff_cats))


def init_vg_dataset():
    def load_vg_train():
        return load_json(vg_detectron_dir / 'train_data.json')

    def load_vg_val():
        return load_json(vg_detectron_dir / 'val_data.json')

    DatasetCatalog.register('vg_train', load_vg_train)
    DatasetCatalog.register('vg_val', load_vg_val)

    obj_cats = load_json(vg_detectron_dir / 'object_categories.json')
    rel_cats = load_json(vg_detectron_dir / 'relation_categories.json')
    obj_colormap = get_colormap(len(obj_cats))

    for name in ['vg_train', 'vg_val']:
        metadata = MetadataCatalog.get(name)

        metadata.thing_classes = obj_cats
        metadata.relation_classes = rel_cats
        metadata.thing_colors = obj_colormap
        metadata.thing_dataset_id_to_contiguous_id = {
            i: i
            for i in range(len(obj_cats))
        }


def init_vrr_vg_dataset():
    # FIXME Make train / val split?
    def load_vrr_vg():
        return load_json(vrr_detectron_dir / 'data.json')

    DatasetCatalog.register('vrr_vg', load_vrr_vg)

    obj_cats = load_json(vrr_detectron_dir / 'object_categories.json')
    rel_cats = load_json(vrr_detectron_dir / 'relation_categories.json')
    obj_colormap = get_colormap(len(obj_cats))

    metadata = MetadataCatalog.get('vrr_vg')

    metadata.thing_classes = obj_cats
    metadata.relation_classes = rel_cats
    # Set a fixed alternating colormap
    metadata.thing_colors = obj_colormap
    metadata.thing_dataset_id_to_contiguous_id = {
        i: i
        for i in range(len(obj_cats))
    }


def init_gqa_dataset():
    def load_gqa_train():
        return load_json(gqa_detectron_dir / 'train_data.json')

    def load_gqa_val():
        return load_json(gqa_detectron_dir / 'val_data.json')

    DatasetCatalog.register('gqa_train', load_gqa_train)
    DatasetCatalog.register('gqa_val', load_gqa_val)

    obj_cats = load_json(gqa_detectron_dir / 'object_categories.json')
    rel_cats = load_json(gqa_detectron_dir / 'relation_categories.json')
    obj_colormap = get_colormap(len(obj_cats))

    for name in ['gqa_train', 'gqa_val']:
        metadata = MetadataCatalog.get(name)

        metadata.thing_classes = obj_cats
        metadata.relation_classes = rel_cats
        metadata.thing_colors = obj_colormap
        metadata.thing_dataset_id_to_contiguous_id = {
            i: i
            for i in range(len(obj_cats))
        }
