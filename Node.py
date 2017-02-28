class Node:
    def __init__(self,  dataset, depth, column_name=None, column_value=None, left_child=None, right_child=None, class_label=None):
        self.dataset = dataset
        self.depth = depth
        self.column_name = column_name
        self.column_value = column_value
        self.left_child = left_child
        self.right_child = right_child
        self.class_label = class_label

