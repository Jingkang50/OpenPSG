from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import graphviz
import numpy as np
from detectron2.data import Metadata
from detectron2.data.detection_utils import read_image
from detectron2.data.transforms import ScaleTransform
from detectron2.utils.colormap import colormap
from detectron2.utils.visualizer import VisImage
from panopticapi.utils import rgb2id
from .. import Visualizer
from .preprocess import x1y1wh_to_xyxy


def get_colormap(num_colors: int):
    return (np.resize(colormap(), (num_colors, 3))).tolist()


def adjust_text_color(color: Tuple[float, float, float],
                      viz: Visualizer) -> Tuple[float, float, float]:
    color = viz._change_color_brightness(color, brightness_factor=0.7)
    color = np.maximum(color, 0.2)
    color[np.argmax(color)] = max(0.8, np.max(color))

    return color


def draw_text(
    viz_img: VisImage = None,
    text: str = None,
    x: float = None,
    y: float = None,
    color: Tuple[float, float, float] = [0, 0, 0],
    size: float = 10,
    padding: float = 5,
    box_color: str = 'black',
    font: str = None,
) -> float:
    text_obj = viz_img.ax.text(
        x,
        y,
        text,
        size=size,
        # family="sans-serif",
        bbox={
            'facecolor': box_color,
            'alpha': 0.8,
            'pad': padding,
            'edgecolor': 'none',
        },
        verticalalignment='top',
        horizontalalignment='left',
        color=color,
        zorder=10,
        rotation=0,
    )
    viz_img.get_image()
    text_dims = text_obj.get_bbox_patch().get_extents()

    return text_dims.width


def viz_annotations(
    data: Dict[str, Any],
    data_dir: Path,
    metadata: Metadata,
    return_graph: bool = True,
    graph_size: str = '10,10',
    filter_out_left_right: bool = False,
):
    """
    Parameters
    ----------
    data : Dict[str, Any]
        In standard Detectron2 format.
        Scene graph annotations should be stored in the `relations` key,
        which contains a list of relations, each being a
        Tuple[subject_idx, object_idx, relation_id]
    data_dir : Path
    metadata : Metadata
        Should contain object / relation class labels, as well as color maps
    return_graph : bool, optional
        , by default True
    graph_size : str, optional
        , by default "10,10"

    Returns
    -------
    Tuple[np.ndarray, graphviz.Digraph]
        RGB image, (H, W, C), [0, 255]
    """
    # Viz instance annotations
    img = read_image(data_dir / data['file_name'], format='RGB')
    viz = Visualizer(
        img,
        metadata=metadata,
        # instance_mode=instance_mode,
    )

    viz_out = viz.draw_dataset_dict(data)
    viz_img = viz_out.get_image()

    # Viz scene graph
    if return_graph:
        g = graphviz.Digraph()
        g.attr('node', style='filled', shape='record')
        g.attr(size=graph_size)

        annos = data['annotations']

        # Draw nodes (objects)
        for idx, obj in enumerate(annos):
            g.node(f'obj_{idx}',
                   metadata.thing_classes[obj['category_id']],
                   color='orange')

        # Show scene graph in text format
        g_text = ''

        # Draw edges (relations)
        for idx, rel in enumerate(sorted(data['relations'])):
            s_idx, o_idx, rel_id = rel

            # FIXME What about stuff classes?
            s_name = metadata.thing_classes[annos[s_idx]['category_id']]
            o_name = metadata.thing_classes[annos[o_idx]['category_id']]
            rel_name = metadata.relation_classes[rel_id]

            if filter_out_left_right and rel_name in [
                    'to the left of',
                    'to the right of',
            ]:
                continue

            g_text += f'({s_name} -> {rel_name} -> {o_name})\n'

            # NOTE Draw w/o explicit node for each edge
            # g.edge(str(s_idx), str(o_idx), metadata.relation_classes[rel_id])

            # Draw with explicit node for each edge
            g.node(f'rel_{idx}', rel_name, color='lightblue2')
            g.edge(f'obj_{s_idx}', f'rel_{idx}')
            g.edge(f'rel_{idx}', f'obj_{o_idx}')

        return viz_img, g, g_text

    else:
        return viz_img


def viz_annotations_alt(
    data: Dict[str, Any],
    data_dir: Path,
    obj_cats: List[str],
    rel_cats: List[str],
    type: str = 'boxes',  # One of {'boxes', 'masks'}
    show_annos: bool = True,
    rel_ids_to_keep: List[int] = None,
    rel_ids_to_filter: List[int] = None,
    n_rels: int = None,
    resize: Tuple[int, int] = None,
    font: str = None,
):
    ch_font_name = 'Source Han Serif SC'

    # Viz instance annotations #################################################
    img = read_image(data_dir / data['file_name'], format='RGB')
    if resize:
        img = cv2.resize(img, resize)

    # Visualize COCO Annotations ###########################################
    if type == 'masks':
        # Load panoptic segmentation
        seg_map = read_image(data['pan_seg_file_name'], format='RGB')
        # Convert to segment ids
        seg_map = rgb2id(seg_map)

        masks = []
        labels_coco = []

        for i, s in enumerate(data['segments_info']):
            label = (
                obj_cats[s['category_id']] if s['isthing']
                # coco things and stuff are concatenated with each other
                else obj_cats[s['category_id'] + 80])
            labels_coco.append(label)

            masks.append(seg_map == s['id'])

        # Prepend instance id
        labels_coco = [f'{i}-{l}' for i, l in enumerate(labels_coco)]

        # Choose colors for each instance in coco
        colormap_coco = get_colormap(len(data['segments_info']))
        colormap_coco = (np.array(colormap_coco) / 255).tolist()

        # Draw coco annotations
        viz = Visualizer(img)
        if show_annos:
            viz.overlay_instances(
                labels=labels_coco,
                masks=masks,
                assigned_colors=colormap_coco,
            )
            viz_img = viz.get_output().get_image()
        else:
            viz_img = img

    elif type == 'boxes':
        boxes = []
        for a in data['annotations']:
            # Depending on bbox mode
            if a['bbox_mode'] == 1:
                box = x1y1wh_to_xyxy(a['bbox'])
            else:
                box = a['bbox']

            # If resizing image
            if resize:
                transform = ScaleTransform(
                    data['height'],
                    data['width'],
                    resize[1],
                    resize[0],
                )
                box = transform.apply_box(np.array(box))[0].tolist()

            boxes.append(box)

        boxes = np.array(boxes)

        # Choose colors for each instance in coco
        colormap_coco = get_colormap(len(data['annotations']))
        colormap_coco = (np.array(colormap_coco) / 255).tolist()

        labels_coco = [obj_cats[a['category_id']] for a in data['annotations']]

        # Draw coco annotations
        viz = Visualizer(img)
        viz.overlay_instances(
            labels=labels_coco,
            boxes=boxes,
            assigned_colors=colormap_coco,
        )
        viz_img = viz.get_output().get_image()

    # Draw relationship triplets ###############################################
    # rel_ids_to_filter = [267, 268]

    # If using custom number of relations
    if n_rels is not None:
        pass
    elif not rel_ids_to_keep and not rel_ids_to_filter:
        n_rels = len(data['relations'])
    elif rel_ids_to_keep:
        n_rels = len([r for r in data['relations'] if r[2] in rel_ids_to_keep])
    elif rel_ids_to_filter:
        n_rels = len(
            [r for r in data['relations'] if r[2] not in rel_ids_to_filter])

    top_padding = 20
    bottom_padding = 20
    left_padding = 20

    text_size = 10
    text_padding = 5
    text_height = text_size + 2 * text_padding

    row_padding = 10

    height = (top_padding + bottom_padding + n_rels *
              (text_height + row_padding) - row_padding)
    width = resize[0] if resize else data['width']

    curr_x = left_padding
    curr_y = top_padding

    # Adjust colormaps
    colormap_coco = [adjust_text_color(c, viz) for c in colormap_coco]

    viz_graph = VisImage(np.full((height, width, 3), 255))

    for i, r in enumerate(data['relations']):
        s_idx, o_idx, rel_id = r

        # Filter for specific relations
        if rel_ids_to_keep:
            if rel_id not in rel_ids_to_keep:
                continue
        elif rel_ids_to_filter:
            if rel_id in rel_ids_to_filter:
                continue

        s_label = labels_coco[s_idx]
        o_label = labels_coco[o_idx]
        rel_label = rel_cats[rel_id]

        # Draw index
        text_width = draw_text(
            viz_img=viz_graph,
            text=f'{i + 1}.',
            x=curr_x,
            y=curr_y,
            size=text_size,
            padding=text_padding,
            box_color='white',
            # font=font,
        )
        curr_x += text_width

        # Special case for chinese predicates
        if '…' in rel_label:
            rel_a, rel_b = rel_label.split('…')

            # Draw subject text
            text_width = draw_text(
                viz_img=viz_graph,
                text=s_label,
                x=curr_x,
                y=curr_y,
                color=colormap_coco[s_idx],
                size=text_size,
                padding=text_padding,
                # font=font,
            )
            curr_x += text_width

            # Draw relation text
            text_width = draw_text(
                viz_img=viz_graph,
                text=rel_a,
                x=curr_x,
                y=curr_y,
                size=text_size,
                padding=text_padding,
                box_color='gainsboro',
                # font=font,
            )
            curr_x += text_width

            # Draw object text
            text_width = draw_text(
                viz_img=viz_graph,
                text=o_label,
                x=curr_x,
                y=curr_y,
                color=colormap_coco[o_idx],
                size=text_size,
                padding=text_padding,
                # font=font,
            )
            curr_x += text_width

            # Draw relation text
            text_width = draw_text(
                viz_img=viz_graph,
                text=rel_b,
                x=curr_x,
                y=curr_y,
                size=text_size,
                padding=text_padding,
                box_color='gainsboro',
                # font=font,
            )
            curr_x += text_width

        else:
            # Draw subject text
            text_width = draw_text(
                viz_img=viz_graph,
                text=s_label,
                x=curr_x,
                y=curr_y,
                color=colormap_coco[s_idx],
                size=text_size,
                padding=text_padding,
                # font=font,
            )
            curr_x += text_width

            # Draw relation text
            text_width = draw_text(
                viz_img=viz_graph,
                text=rel_label,
                x=curr_x,
                y=curr_y,
                size=text_size,
                padding=text_padding,
                box_color='gainsboro',
                # font=font,
            )
            curr_x += text_width

            # Draw object text
            text_width = draw_text(
                viz_img=viz_graph,
                text=o_label,
                x=curr_x,
                y=curr_y,
                color=colormap_coco[o_idx],
                size=text_size,
                padding=text_padding,
                # font=font,
            )
            curr_x += text_width

        curr_x = left_padding
        curr_y += text_height + row_padding

    viz_graph = viz_graph.get_image()

    return np.vstack([viz_img, viz_graph])


def viz_annotations_v1(
    data: Dict[str, Any],
    data_dir: Path,
    obj_cats: List[str],
    rel_cats: List[str],
    type: str = 'boxes',  # One of {'boxes', 'masks'}
    show_annos: bool = True,
    rel_ids_to_keep: List[int] = None,
    rel_ids_to_filter: List[int] = None,
    n_rels: int = None,
    resize: Tuple[int, int] = None,
    font: str = None,
):
    ch_font_name = 'Source Han Serif SC'

    # Viz instance annotations #############################################
    img = read_image(data_dir / data['file_name'], format='RGB')
    if resize:
        img = cv2.resize(img, resize)

    # Visualize COCO Annotations ###########################################
    if type == 'masks':
        # Load panoptic segmentation
        seg_map = read_image(data['pan_seg_file_name'], format='RGB')
        # Convert to segment ids
        seg_map = rgb2id(seg_map)

        masks = []
        labels_coco = []

        for i, s in enumerate(data['segments_info']):
            label = (
                obj_cats[s['category_id']] if s['isthing']
                # coco things and stuff are concatenated with each other
                else obj_cats[s['category_id'] + 80])
            labels_coco.append(label)

            masks.append(seg_map == s['id'])

        # Choose colors for each instance in coco
        colormap_coco = get_colormap(len(data['segments_info']))
        colormap_coco = (np.array(colormap_coco) / 255).tolist()

        # Draw coco annotations
        viz = Visualizer(img)
        if show_annos:
            viz.overlay_instances(
                labels=labels_coco,
                masks=masks,
                assigned_colors=colormap_coco,
            )
            viz_img = viz.get_output().get_image()
        else:
            viz_img = img

    elif type == 'boxes':
        boxes = []
        for a in data['annotations']:
            # Depending on bbox mode
            if a['bbox_mode'] == 1:
                box = x1y1wh_to_xyxy(a['bbox'])
            else:
                box = a['bbox']

            # If resizing image
            if resize:
                transform = ScaleTransform(
                    data['height'],
                    data['width'],
                    resize[1],
                    resize[0],
                )
                box = transform.apply_box(np.array(box))[0].tolist()

            boxes.append(box)

        boxes = np.array(boxes)

        # Choose colors for each instance in coco
        colormap_coco = get_colormap(len(data['annotations']))
        colormap_coco = (np.array(colormap_coco) / 255).tolist()

        labels_coco = [obj_cats[a['category_id']] for a in data['annotations']]

        # Draw coco annotations
        viz = Visualizer(img)
        viz.overlay_instances(
            labels=labels_coco,
            boxes=boxes,
            assigned_colors=colormap_coco,
        )
        viz_img = viz.get_output().get_image()

    # Draw relationship triplets ###############################################
    # rel_ids_to_filter = [267, 268]

    # If using custom number of relations
    if n_rels is not None:
        pass
    elif not rel_ids_to_keep and not rel_ids_to_filter:
        n_rels = len(data['relations'])
    elif rel_ids_to_keep:
        n_rels = len([r for r in data['relations'] if r[2] in rel_ids_to_keep])
    elif rel_ids_to_filter:
        n_rels = len(
            [r for r in data['relations'] if r[2] not in rel_ids_to_filter])

    top_padding = 20
    bottom_padding = 20
    left_padding = 20

    text_size = 10
    text_padding = 5
    text_height = text_size + 2 * text_padding

    row_padding = 10

    height = (top_padding + bottom_padding + n_rels *
              (text_height + row_padding) - row_padding)
    width = resize[0] if resize else data['width']

    curr_x = left_padding
    curr_y = top_padding

    # Adjust colormaps
    colormap_coco = [adjust_text_color(c, viz) for c in colormap_coco]

    viz_graph = VisImage(np.full((height, width, 3), 255))

    for i, r in enumerate(data['relations']):
        s_idx, o_idx, rel_id = r

        # Filter for specific relations
        if rel_ids_to_keep:
            if rel_id not in rel_ids_to_keep:
                continue
        elif rel_ids_to_filter:
            if rel_id in rel_ids_to_filter:
                continue

        s_label = labels_coco[s_idx]
        o_label = labels_coco[o_idx]
        rel_label = rel_cats[rel_id]

        # Special case for chinese predicates
        if '…' in rel_label:
            rel_a, rel_b = rel_label.split('…')

            # Draw subject text
            text_width = draw_text(
                viz_img=viz_graph,
                text=s_label,
                x=curr_x,
                y=curr_y,
                color=colormap_coco[s_idx],
                size=text_size,
                padding=text_padding,
                # font=font,
            )
            curr_x += text_width

            # Draw relation text
            text_width = draw_text(
                viz_img=viz_graph,
                text=rel_a,
                x=curr_x,
                y=curr_y,
                size=text_size,
                padding=text_padding,
                box_color='gainsboro',
                # font=font,
            )
            curr_x += text_width

            # Draw object text
            text_width = draw_text(
                viz_img=viz_graph,
                text=o_label,
                x=curr_x,
                y=curr_y,
                color=colormap_coco[o_idx],
                size=text_size,
                padding=text_padding,
                # font=font,
            )
            curr_x += text_width

            # Draw relation text
            text_width = draw_text(
                viz_img=viz_graph,
                text=rel_b,
                x=curr_x,
                y=curr_y,
                size=text_size,
                padding=text_padding,
                box_color='gainsboro',
                # font=font,
            )
            curr_x += text_width

        else:
            # Draw subject text
            text_width = draw_text(
                viz_img=viz_graph,
                text=s_label,
                x=curr_x,
                y=curr_y,
                color=colormap_coco[s_idx],
                size=text_size,
                padding=text_padding,
                # font=font,
            )
            curr_x += text_width

            # Draw relation text
            text_width = draw_text(
                viz_img=viz_graph,
                text=rel_label,
                x=curr_x,
                y=curr_y,
                size=text_size,
                padding=text_padding,
                box_color='gainsboro',
                # font=font,
            )
            curr_x += text_width

            # Draw object text
            text_width = draw_text(
                viz_img=viz_graph,
                text=o_label,
                x=curr_x,
                y=curr_y,
                color=colormap_coco[o_idx],
                size=text_size,
                padding=text_padding,
                # font=font,
            )
            curr_x += text_width

        curr_x = left_padding
        curr_y += text_height + row_padding

    viz_graph = viz_graph.get_image()

    return np.vstack([viz_img, viz_graph])


def viz_annotations_psg(
    data: Dict[str, Any],
    data_dir: Path,
    obj_cats: List[str],
    rel_cats: List[str],
    type: str = 'boxes',  # One of {'boxes', 'masks'}
    show_annos: bool = True,
    rel_ids_to_keep: List[int] = None,
    rel_ids_to_filter: List[int] = None,
    n_rels: int = None,
    resize: Tuple[int, int] = None,
    font: str = None,
):
    ch_font_name = 'Source Han Serif SC'

    # Viz instance annotations #################################################
    img = read_image(data_dir / data['file_name'], format='RGB')
    if resize:
        img = cv2.resize(img, resize)

    # Visualize COCO Annotations ###########################################
    if type == 'masks':
        # Load panoptic segmentation
        seg_map = read_image(data_dir / data['pan_seg_file_name'],
                             format='RGB')
        # Convert to segment ids
        seg_map = rgb2id(seg_map)

        masks = []
        labels_coco = []

        for i, s in enumerate(data['segments_info']):
            label = (
                obj_cats[s['category_id']]
                # if s["isthing"]
                # # coco things and stuff are concatenated with each other
                # else obj_cats[s["category_id"] + 80]
            )
            labels_coco.append(label)

            masks.append(seg_map == s['id'])

        # Choose colors for each instance in coco
        colormap_coco = get_colormap(len(data['segments_info']))
        colormap_coco = (np.array(colormap_coco) / 255).tolist()

        # Draw coco annotations
        viz = Visualizer(img)
        if show_annos:
            viz.overlay_instances(
                labels=labels_coco,
                masks=masks,
                assigned_colors=colormap_coco,
            )
            viz_img = viz.get_output().get_image()
        else:
            viz_img = img

    elif type == 'boxes':
        boxes = []
        for a in data['annotations']:
            # Depending on bbox mode
            if a['bbox_mode'] == 1:
                box = x1y1wh_to_xyxy(a['bbox'])
            else:
                box = a['bbox']

            # If resizing image
            if resize:
                transform = ScaleTransform(
                    data['height'],
                    data['width'],
                    resize[1],
                    resize[0],
                )
                box = transform.apply_box(np.array(box))[0].tolist()

            boxes.append(box)

        boxes = np.array(boxes)

        # Choose colors for each instance in coco
        colormap_coco = get_colormap(len(data['annotations']))
        colormap_coco = (np.array(colormap_coco) / 255).tolist()

        labels_coco = [obj_cats[a['category_id']] for a in data['annotations']]

        # Draw coco annotations
        viz = Visualizer(img)
        viz.overlay_instances(
            labels=labels_coco,
            boxes=boxes,
            assigned_colors=colormap_coco,
        )
        viz_img = viz.get_output().get_image()

    # Draw relationship triplets ###############################################
    # rel_ids_to_filter = [267, 268]

    # If using custom number of relations
    if n_rels is not None:
        pass
    elif not rel_ids_to_keep and not rel_ids_to_filter:
        n_rels = len(data['relations'])
    elif rel_ids_to_keep:
        n_rels = len([r for r in data['relations'] if r[2] in rel_ids_to_keep])
    elif rel_ids_to_filter:
        n_rels = len(
            [r for r in data['relations'] if r[2] not in rel_ids_to_filter])

    top_padding = 20
    bottom_padding = 20
    left_padding = 20

    text_size = 10
    text_padding = 5
    text_height = text_size + 2 * text_padding

    row_padding = 10

    height = (top_padding + bottom_padding + n_rels *
              (text_height + row_padding) - row_padding)
    width = resize[0] if resize else data['width']

    curr_x = left_padding
    curr_y = top_padding

    # Adjust colormaps
    colormap_coco = [adjust_text_color(c, viz) for c in colormap_coco]

    viz_graph = VisImage(np.full((height, width, 3), 255))

    for i, r in enumerate(data['relations']):
        s_idx, o_idx, rel_id = r

        # Filter for specific relations
        if rel_ids_to_keep:
            if rel_id not in rel_ids_to_keep:
                continue
        elif rel_ids_to_filter:
            if rel_id in rel_ids_to_filter:
                continue

        s_label = labels_coco[s_idx]
        o_label = labels_coco[o_idx]
        rel_label = rel_cats[rel_id]

        # Special case for chinese predicates
        if '…' in rel_label:
            rel_a, rel_b = rel_label.split('…')

            # Draw subject text
            text_width = draw_text(
                viz_img=viz_graph,
                text=s_label,
                x=curr_x,
                y=curr_y,
                color=colormap_coco[s_idx],
                size=text_size,
                padding=text_padding,
                # font=font,
            )
            curr_x += text_width

            # Draw relation text
            text_width = draw_text(
                viz_img=viz_graph,
                text=rel_a,
                x=curr_x,
                y=curr_y,
                size=text_size,
                padding=text_padding,
                box_color='gainsboro',
                # font=font,
            )
            curr_x += text_width

            # Draw object text
            text_width = draw_text(
                viz_img=viz_graph,
                text=o_label,
                x=curr_x,
                y=curr_y,
                color=colormap_coco[o_idx],
                size=text_size,
                padding=text_padding,
                # font=font,
            )
            curr_x += text_width

            # Draw relation text
            text_width = draw_text(
                viz_img=viz_graph,
                text=rel_b,
                x=curr_x,
                y=curr_y,
                size=text_size,
                padding=text_padding,
                box_color='gainsboro',
                # font=font,
            )
            curr_x += text_width

        else:
            # Draw subject text
            text_width = draw_text(
                viz_img=viz_graph,
                text=s_label,
                x=curr_x,
                y=curr_y,
                color=colormap_coco[s_idx],
                size=text_size,
                padding=text_padding,
                # font=font,
            )
            curr_x += text_width

            # Draw relation text
            text_width = draw_text(
                viz_img=viz_graph,
                text=rel_label,
                x=curr_x,
                y=curr_y,
                size=text_size,
                padding=text_padding,
                box_color='gainsboro',
                # font=font,
            )
            curr_x += text_width

            # Draw object text
            text_width = draw_text(
                viz_img=viz_graph,
                text=o_label,
                x=curr_x,
                y=curr_y,
                color=colormap_coco[o_idx],
                size=text_size,
                padding=text_padding,
                # font=font,
            )
            curr_x += text_width

        curr_x = left_padding
        curr_y += text_height + row_padding

    viz_graph = viz_graph.get_image()

    return np.vstack([viz_img, viz_graph])
