import collections as cl
import math as mt

import numpy as np

from Node import Node

depth_accuracy_dict = cl.defaultdict(list)


# function which returns log2(x). Returns 0 if x=0
def log2(x):
    # Condition to handle log2(0)
    if x != 0:
        return np.log2(x)
    else:
        return 0


# function which divides dataset into left dataset and right dataset based on feature(column_name) and its value
def divide_dataset(dataset, column_name, value):

    left_dataset = dataset[dataset[column_name] <= value]
    right_dataset = dataset[dataset[column_name] > value]

    return left_dataset, right_dataset


# Function which return feature with max information gain
def max_information_gain(dataset):

    column_names = list(dataset.columns)
    class_column = column_names[0]
    information_gain_columns = {}
    column_values = {}
    # No. of rows in dataset
    dataset_count = dataset.shape[0]
    # No. of rows in dataset with class = 0
    rows_y0 = dataset[dataset[column_names[0]] == 0].shape[0]
    # No. of rows in dataset with class = 1
    rows_y1 = dataset[dataset[column_names[0]] == 1].shape[0]

    # Probabilitis of class = 0 and class = 1
    p_y0 = rows_y0/dataset_count

    p_y1 = rows_y1/dataset_count

    # entropy before split i.e. H(y)
    entropy_y = - (p_y0 * log2(p_y0) + p_y1 * log2(p_y1))

    # Loop to calculate H(y|x) for each feature x
    for i in range(1, (len(column_names))):

        column_name = column_names[i]
        # Get distinct values of feature in the dataset
        distinct_values = list(dataset[column_name].unique())
        if len(distinct_values) > 4:
            # x = int(len(distinct_values)/2)
            div = round(len(distinct_values)/8)
            values = []
            count = 0
            for k in range(div, len(distinct_values), div):
                count += 1
                if count % 2 != 0:
                    values.append(distinct_values[k])

            distinct_values = values

        # Initializing max_gain with large negative value
        max_gain = -10

        # Loop for each distinct value of feature x and find the best value to divide the dataset
        for j in range(0, len(distinct_values)):
            column_value = distinct_values[j]

            # No. of rows in dataset with respective value of x and class label
            rows_x_left = dataset[dataset[column_name] <= column_value].shape[0]
            rows_y0_x_left = dataset[(dataset[column_name] <= column_value) & (dataset[class_column] == 0)].shape[0]
            rows_y1_x_left = dataset[(dataset[column_name] <= column_value) & (dataset[class_column] == 1)].shape[0]

            rows_x_right = dataset[dataset[column_name] > column_value].shape[0]
            rows_y0_x_right = dataset[(dataset[column_name] > column_value) & (dataset[class_column] == 0)].shape[0]
            rows_y1_x_right = dataset[(dataset[column_name] > column_value) & (dataset[class_column] == 1)].shape[0]

            # Condition to handle division by zero in probability
            if rows_x_left != 0:
                p_x_left = rows_x_left/dataset_count
                # print('p_x_left', p_x_left)
                p_y0_x_left = rows_y0_x_left/rows_x_left
                # print('p_y0_x_left', p_y0_x_left)
                p_y1_x_left = rows_y1_x_left/rows_x_left
                # print('p_y1_x_left', p_y1_x_left)
            else:
                p_x_left = 0
                p_y0_x_left = 0
                p_y1_x_left = 0

            # Condition to handle division by zero in probability
            if rows_x_right != 0:
                p_x_right = rows_x_right/dataset_count
                # print('p_x_right', p_x_right)
                p_y0_x_right = rows_y0_x_right/rows_x_right
                # print('p_y0_x_right', p_y0_x_right)
                p_y1_x_right = rows_y1_x_right/rows_x_right
                # print('p_y1_x_right', p_y1_x_right)
            else:
                p_x_right = 0
                p_y0_x_right = 0
                p_y1_x_right = 0

            # entropy after split i.e. H(y|x)
            entropy_y_x = - (p_x_left * ((p_y0_x_left * log2(p_y0_x_left)) + (p_y1_x_left * log2(p_y1_x_left))) + \
                            p_x_right * ((p_y0_x_right * log2(p_y0_x_right)) + (p_y1_x_right * log2(p_y1_x_right))))

            # information gain calculation H(y) - H(y|x)for a given value of x
            information_gain = entropy_y - entropy_y_x

            # To get the max information gain of a feature x amongst the distinct values of x
            if information_gain > max_gain:
                max_gain = information_gain
                column_values[column_name] = column_value

        # Storing the gain values in a dictionary for all features
        information_gain_columns[column_name] = max_gain

    # Getting the max infornmation gain value and its corresponding feature and its value
    max_gain_column_name = max(information_gain_columns, key=information_gain_columns.get)
    max_gain_column_value = column_values[max_gain_column_name]

    return max_gain_column_name, max_gain_column_value


# Function that builds tree
def build_tree(parent_node, depth_threshold):

    dataset = parent_node.dataset
    depth = parent_node.depth
    column_name = parent_node.column_name
    column_value = parent_node.column_value
    stop = 0
    columns = dataset.columns
    class_column = columns[0]

    dataset_class_0_count = int(dataset[dataset[class_column] == 0].shape[0])
    dataset_class_1_count = int(dataset[dataset[class_column] == 1].shape[0])
    dataset_count = int(dataset.shape[0])

    # Condition to decide the label if the node turns out a leaf
    if dataset_class_0_count >= dataset_class_1_count:
        class_label = 0
    else:
        class_label = 1

    # Conditions to stop building the tree
    if (depth != depth_threshold) and (dataset_class_0_count > 0 and dataset_class_1_count > 0 and dataset_count > 0):
        depth += 1
        left_dataset, right_dataset = divide_dataset(dataset, column_name, column_value)
        left_dataset_count = int(left_dataset.shape[0])
        right_dataset_count = int(right_dataset.shape[0])

        # Creating left child and calling recursion with that node
        if left_dataset_count != 0:
            left_best_column_name, left_best_column_value = max_information_gain(left_dataset)
            parent_node.left_child = Node(left_dataset, depth, left_best_column_name, left_best_column_value)
            build_tree(parent_node.left_child, depth_threshold)

        # Creating right child and calling recursion with that node
        if right_dataset_count != 0:
            right_best_column_name, right_best_column_value = max_information_gain(right_dataset)
            parent_node.right_child = Node(right_dataset, depth, right_best_column_name, right_best_column_value)
            build_tree(parent_node.right_child, depth_threshold)

    # Setting the flag to stop building the tree
    else:
        stop = 1

    # Setting the label for leaf node
    if stop == 1:
        parent_node.class_label = class_label


def print_tree(node):
    print(node.column_name, ',', node.column_value, ',', node.class_label, ',', node.depth)
    if node.left_child is not None:
        print('left')
        print_tree(node.left_child)
    if node.right_child is not None:
        print('right')
        print_tree(node.right_child)


def print_fulltree(root):
    if root.class_label is None:
        print("Leaf node")
        print("column:", root.column_name, " value:", root.column_value, " depth:", root.depth, " label:", root.class_label)
    else:
        print("Node")
        ld, rd = divide_dataset(root.dataset, root.column_name, root.column_value)
        print("column:", root.column_name, " value:", root.column_value, " depth:", root.depth, " label:", root.class_label, " ld:", ld.shape[0], " rd:", rd.shape[0])

        if root.left_child is not None:
            print("left child")
            print_fulltree(root.left_child)
        if root.right_child is not None:
            print("right child")
            print_fulltree(root.right_child)


# Function to test the model using test data
def test_row(node, row):

    # Traverse the row using node's column name and value
    while True:
        column_name = node.column_name
        column_value = node.column_value

        # if the leaf is node is reached then return node's class label
        if node.class_label is not None:
            return node.class_label
        else:
            # Traverse until leaf node is reached
            if row[column_name] <= column_value:
                node = node.left_child
            else:
                node = node.right_child


















