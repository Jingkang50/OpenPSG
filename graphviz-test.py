#%%
import pydot
import os
from IPython.display import Image, display
# %%
rootDir = os.getcwd()
# %%
G = pydot.Dot(graph_type = "digraph")
# %%
currentDir = rootDir.split("/")[-1]
# %%
node = pydot.Node(currentDir, style = "filled", fillcolor = "green")
G.add_node(node)
# %%
im = Image(G.create_jpeg())
# %%
display(im)
# %%
