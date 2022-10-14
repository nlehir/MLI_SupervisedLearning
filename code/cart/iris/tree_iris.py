from sklearn import tree
from sklearn.datasets import load_iris
import graphviz
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
classifier = tree.DecisionTreeClassifier(min_impurity_decrease=0.1)
classifier = classifier.fit(iris.data, iris.target)

# plot the graph
dot_data = tree.export_graphviz(classifier,
                                out_file=None,
                                feature_names=iris.feature_names,
                                class_names=iris.target_names,
                                filled=True,
                                rounded=True,
                                special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("tree_iris")
