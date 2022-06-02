import numpy as np

# Node class
class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        
        # decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain

        # leaf node
        self.value = value




# Tree class
class DecisionTreeClasifier():
    def __init__(self, min_sample_split=2, max_depth=2):

        # initalize root
        self.root = None

        # stopping conds
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth

    def build_tree(self, dataset, curr_depth=0):
        
        X,y =dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = X.shape
        if num_samples >= self.min_sample_split and curr_depth <= self.max_depth:

            best_split = self.calculate_best_split(dataset, num_samples, num_features)
            if best_split['info_gain'] > 0:

                left_tree = self.build_tree(best_split['dataset_left'], curr_depth+1)
                right_tree = self.build_tree(best_split['dataset_right'], curr_depth+1)

                return Node(best_split['feature_index'],best_split['threshold'],left_tree, right_tree, best_split['info_gain'])
        leaf_value = self.calculate_leaf_value(y)
        return Node(value=leaf_value)
    
    def calculate_best_split(self, dataset, num_samples, num_features):
        best_split = {}
        max_info_gain = -float('inf')

        for feature_index in range(num_features):
            feature_values = dataset[:,feature_index]
            thresholds = np.unique(feature_values)
            for threshold in thresholds:
                dataset_left , dataset_right = self.split(dataset, feature_index, threshold)
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, y_left, y_right = dataset[:,-1], dataset_left[:, -1], dataset_right[:, -1]
                    current_info_gain = self.calculate_info_gain(y, y_left, y_right, 'gini')
                    if current_info_gain>max_info_gain:
                        best_split['feature_index']= feature_index
                        best_split['dataset_left'] = dataset_left
                        best_split['dataset_right'] = dataset_right
                        best_split['threshold'] = threshold
                        best_split['info_gain'] = current_info_gain
                        max_info_gain = current_info_gain
        
        return best_split
    
    def split(self, dataset, feature_index, threshold):
        # Note that this will not work well if values are categorical
        dataset_left = np.array([row for row in dataset if row[feature_index] <= threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index] > threshold])
        return dataset_left, dataset_right

    def calculate_info_gain(self, parent, l_child, r_child, mode="entropy"):

        w_l = len(l_child) / len(parent)
        w_r = len(r_child) / len(parent)

        if mode=='gini':
            gain = self.gini(parent) - ((w_l * self.gini(l_child)) + (w_r * self.gini(r_child)))
        else:
            gain = self.entropy(parent) - ((w_l * self.entropy(l_child)) + (w_r * self.entropy(r_child)))
        
        return gain

    def gini(self, y):
        class_labels = np.unique(y)
        gini = 0

        for i in class_labels:
            class_prob = len(y[y == i]) / len(y)
            gini += class_prob**2
        return 1 - gini


    def entropy(self, y):
        class_labels = np.unique(y)
        entropy = 0
        for i in class_labels:
            class_prob = len(y[y==i]) / len(y)
            entropy += - class_prob * np.log2(class_prob)
        return entropy

    def calculate_leaf_value(self, y):
        y = list(y)
        return max(y, key=y.count)
    
    def print_Tree(self, tree=None, indent=" "):
        if not tree:
            tree = self.root
        
        if tree.value is not None:
            print(tree.value)
        
        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_Tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_Tree(tree.right, indent + indent)

    def fit(self, X, y):
        dataset = np.concatenate((X, y), axis=1)
        self.root = self.build_tree(dataset)

    def predict(self , X):
        predictions = [self.make_predictions(x, self.root) for x in X]
        return predictions

    def make_predictions(self, x, tree):
        if tree.value != None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_predictions(x, tree.left)
        else:
            return self.make_predictions(x, tree.right)

