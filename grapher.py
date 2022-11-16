import json
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, ConnectionStyle, ConnectionPatch
from matplotlib.collections import PatchCollection
from numpy import linspace, pi

f = open('json_out/graph.json')

figure, ax = plt.subplots()
#figure.set_figheight(8)
#figure.set_figwidth(8)
# ax.set_ylim(-10, 30)
# ax.set_xlim(-10, 30)
ax.set_autoscale_on(True)
ax.axis("equal")
ax.grid()
data = json.load(f)
circles = data["circles"]
edges = data["edges"]

patches = []

for c in circles:
    location = circles[c]["location"]
    radius = circles[c]["radius"]
    circle_to_draw = Circle((location["x"], location["y"]), radius, fill=None)
    patches.append(circle_to_draw)

for e in edges:
    node_in_graph = json.loads(e)
    loc = node_in_graph["location"]
    plt.plot(loc['x'], loc['y'], 'o',)
    for n in edges[e]:
        n_loc = n["node"]["location"]
        n_theta = n["theta"]
        if n_theta is None :
            plt.plot([loc["x"], n_loc['x']], [loc['y'], n_loc['y']])
        # else:
            # arc_con = ConnectionStyle.Arc3(n_theta).connect([loc['x'], loc['y']], [n_loc['x'], n_loc['y']])
            # ax.add_patch(arc_con)

p = PatchCollection(patches, match_original = True)
ax.add_collection(p)
plt.savefig("vis_graph.jpg")
