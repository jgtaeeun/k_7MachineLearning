from imageio import imread
import matplotlib.pyplot as plt


def plot_animal_tree(ax=None):
    import graphviz
    plt.figure(dpi=100)
    if ax is None:
        ax = plt.gca()
    mygraph = graphviz.Digraph(node_attr={'shape': 'box'},
                               edge_attr={'labeldistance': "10.5"},
                               format="png")
    mygraph.node("0", "날개가 있나요?" , fontname='Malgun Gothic')
    mygraph.node("1", "날 수 있나요?", fontname='Malgun Gothic')
    mygraph.node("2", "지느러미가 있나요?", fontname='Malgun Gothic')
    mygraph.node("3", "매", fontname='Malgun Gothic')
    mygraph.node("4", "펭귄", fontname='Malgun Gothic')
    mygraph.node("5", "돌고래", fontname='Malgun Gothic')
    mygraph.node("6", "곰", fontname='Malgun Gothic')
    mygraph.edge("0", "1", label="True", fontname='Malgun Gothic')
    mygraph.edge("0", "2", label="False", fontname='Malgun Gothic')
    mygraph.edge("1", "3", label="True", fontname='Malgun Gothic')
    mygraph.edge("1", "4", label="False", fontname='Malgun Gothic')
    mygraph.edge("2", "5", label="True", fontname='Malgun Gothic')
    mygraph.edge("2", "6", label="False", fontname='Malgun Gothic')
    mygraph.render("tmp")
    ax.imshow(imread("tmp.png"))
    ax.set_axis_off()
