#%%
from pathlib import Path
import argparse
import time
import shutil
import numpy as np
import predict
from openpsg.utils.utils import CLASSES, PREDICATES
import pydot
from PIL import Image
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--image', '-i', type=str, required=True)
parser.add_argument('--num_rel', '-n', type=int, default=20)
args = parser.parse_args()
image = args.image
num_rel = args.num_rel
predictor = predict.Predictor()
predictor.setup()
G = pydot.Dot(graph_type = "digraph")

start_time = time.perf_counter()
output, result = predictor.predict_verbose(image, num_rel)  # Result class defined at openpsg/models/relation_heads/approaches/relation_util.py
end_time = time.perf_counter()

out_path = f'./out_{Path(image).stem}.png'
shutil.copyfile(output[0].image, out_path)

# Taken from openpsg/utils/utils.py
rel_obj_labels = result.labels
rel_obj_labels = [CLASSES[l - 1] for l in rel_obj_labels]
unq = {}
for i, label in enumerate(rel_obj_labels):
    if label in unq:
        unq[label] += 1

        label = '_'.join([label,str(unq[label])])
        rel_obj_labels[i] = label
    else:
        unq[label] = 1
        rel_obj_labels[i] = label + '_1'

for obj in rel_obj_labels:
    node = node = pydot.Node(obj)
    G.add_node(node)

# Filter out relations
n_rel_topk = num_rel
# Exclude background class
rel_dists = result.rel_dists[:, 1:]
rel_scores = rel_dists.max(1)
# Extract relations with top scores
rel_topk_idx = np.argpartition(rel_scores, -n_rel_topk)[-n_rel_topk:]
rel_labels_topk = rel_dists[rel_topk_idx].argmax(1)
rel_pair_idxes_topk = result.rel_pair_idxes[rel_topk_idx]
relations = np.concatenate(
    [rel_pair_idxes_topk, rel_labels_topk[..., None]], axis=1)
n_rels = len(relations)

for i, r in enumerate(relations):
    s_idx, o_idx, rel_id = r
    s_label = rel_obj_labels[s_idx]
    o_label = rel_obj_labels[o_idx]
    rel_label = PREDICATES[rel_id]
    edge = pydot.Edge(s_label, o_label)
    edge.set_label(rel_label)
    G.add_edge(edge)
    print(s_label, rel_label, o_label)
graph_path = f'./graph_{Path(image).stem}.jpg'
G.write_jpeg(graph_path)
print(f"Inference took {end_time-start_time:0.4f} seconds. See temp/ for relation visualizations.")

out_img = Image.open(out_path)
out_graph = Image.open(graph_path)

fig, ax = plt.subplots(1,2, figsize=(20,10))
ax[0].imshow(out_img)
ax[1].imshow(out_graph)
for a in ax:
    a.axis('off')
plt.tight_layout
plt.savefig(f'./final_{Path(image).stem}.jpg')
#%%