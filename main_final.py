from functions import max_information_gain, build_tree, test_row
from Node import Node
import pandas as pd
import matplotlib.pyplot as plt

# Making list of Training and Test files
training_datasets = ['Transfusion-1.train.txt', 'Transfusion-2.train.txt', 'Transfusion-3.train.txt', \
                     'monks-1.train.txt', 'monks-2.train.txt', 'monks-3.train.txt']
test_datasets = ['Transfusion-1.test.txt', 'Transfusion-2.test.txt', 'Transfusion-3.test.txt', \
                 'monks-1.test.txt', 'monks-2.test.txt', 'monks-3.test.txt']

# Dictionary to store confusion matrix values for depth 1 and 2
confusion_matrix = {}

# Dictionary and lists to store file respective values
monk_accuracy_wrt_depth = {}
monk_depth_list = []
monk_accuracy_list = []

Transfusion_accuracy_wrt_depth = {}
Transfusion_depth_list = []
Transfusion_accuracy_list = []

# loop until depth 16
for i in range(0, 17):
    print('')
    print('Depth:', i)
    depth_threshold = i
    accuracy_wrt_file = {}

    # Condition to calculate depth 0 accuracy in else clause
    if i > 0:
        # Initialize confusion matrix dictionary for depth 1 and 2
        if i <= 2:
            confusion_matrix[i] = {}

        # Initialize avg accuracy variable for each type of file to 0
        monk_avg_accuracy_wrt_file = 0
        Transfusion_avg_accuracy_wrt_file = 0

        # --------------------------TRAIN MODEL------------------------------
        # loop for each training file
        for j in range(0, len(training_datasets)):
            # Create a dataframe for each file
            training_dataset_file = training_datasets[j]
            df = pd.read_csv(training_dataset_file, sep=",")
            train_dataset = pd.DataFrame(df)

            # initialize root node depth as 0
            depth = 0

            # retrieve best column and value to divide on for root node
            best_column, best_column_value = max_information_gain(train_dataset)
            # create root node instance
            root_node = Node(train_dataset, depth, best_column, best_column_value)
            # build tree recursively
            build_tree(root_node, depth_threshold)
            # print_tree(root_node)

            # --------------------TESTING the MODEL--------------------------
            # Create a dataframe for each file
            test_dataset_file = test_datasets[j]
            df = pd.read_csv(test_dataset_file, sep=",")
            test_dataset = pd.DataFrame(df)

            # Count of rows for the test dataset
            test_dataset_count = test_dataset.shape[0]
            column_names = list(test_dataset.columns)

            # getting class feature name
            class_column = column_names[0]

            # calculating class count in the dataset for each file
            class_0_count = test_dataset[test_dataset[class_column] == 0].shape[0]
            class_1_count = test_dataset[test_dataset[class_column] == 1].shape[0]

            # Initializing variables for confusion matrix
            if i <= 2:
                # creating empty dictionary for each file in confusion matrix dictionary for depth i
                confusion_matrix[i][test_dataset_file] = {}
                # initialize TP, FN, TN and FP to 0
                TP = 0
                FN = 0

                TN = 0
                FP = 0

                # Adding actual positive and actual negative values to confusion matrix ith and file's dictionary
                confusion_matrix[i][test_dataset_file]['AP'] = class_1_count
                confusion_matrix[i][test_dataset_file]['AN'] = class_0_count

            # Initializing count for correct predictions
            predict_right_count = 0
            # for each row of test, traverse the tree
            for index, row in test_dataset.iterrows():
                #  return the class label it reaches in the end
                predicted_class = test_row(root_node, row)
                # Condition to get no. of accurate predictions
                if predicted_class == row[class_column]:
                    predict_right_count += 1

                # Confusion matrix for depth 1 and 2
                if i <= 2:
                    # Calculation of TP, FP, TN and FN
                    if predicted_class == 1:
                        if predicted_class == row[class_column]:
                            TP += 1
                        else:
                            FP += 1
                    else:
                        if predicted_class == row[class_column]:
                            TN += 1
                        else:
                            FN += 1

            if i <= 2:
                # Adding TP,FP,TN and FN values to confusion matrix ith and file's dictionary
                confusion_matrix[i][test_dataset_file]['TP'] = TP
                confusion_matrix[i][test_dataset_file]['FP'] = FP
                confusion_matrix[i][test_dataset_file]['TN'] = TN
                confusion_matrix[i][test_dataset_file]['FN'] = FN

            # calculating avg accuracy for respective monk and transfusion datasets
            if 'monk' in test_dataset_file:
                monk_avg_accuracy_wrt_file += predict_right_count/test_dataset_count * 100
            else:
                Transfusion_avg_accuracy_wrt_file += predict_right_count/test_dataset_count * 100

        monk_accuracy_wrt_depth[i] = format(monk_avg_accuracy_wrt_file/3, '.2f')
        print('Monk_depth_', i, ':', monk_accuracy_wrt_depth[i])
        Transfusion_accuracy_wrt_depth[i] = format(Transfusion_avg_accuracy_wrt_file/3, '.2f')
        print('Transfusion_depth_', i, ':', Transfusion_accuracy_wrt_depth[i])
        if i <= 2:
            print('Confusion Matrix for depth: ', i)
            print(confusion_matrix[i])

    else:
        # Calculate avg accuracy for depth 0
        monk_avg_depth_0_accuracy = 0
        Transfusion_avg_depth_0_accuracy = 0
        for k in range(0, len(test_datasets)):
            test_dataset_file = test_datasets[k]
            df = pd.read_csv(test_dataset_file, sep=",")
            test_dataset = pd.DataFrame(df)
            test_dataset_count = test_dataset.shape[0]
            column_names = list(test_dataset.columns)
            class_column = column_names[0]

            # calculating depth 0 accuracy for each file
            class_0_count = test_dataset[test_dataset[class_column] == 0].shape[0]
            class_1_count = test_dataset[test_dataset[class_column] == 1].shape[0]

            # calculating depth 0 accuracy for each file
            if class_0_count >= class_1_count:
                depth_0_accuracy = class_0_count/test_dataset_count * 100
            else:
                depth_0_accuracy = class_1_count/test_dataset_count * 100

            if 'monk' in test_dataset_file:
                monk_avg_depth_0_accuracy += depth_0_accuracy
            else:
                Transfusion_avg_depth_0_accuracy += depth_0_accuracy

        monk_accuracy_wrt_depth[i] = format(monk_avg_depth_0_accuracy/3, '.2f')
        print('Monk_depth_', i, ':', monk_accuracy_wrt_depth[i])
        Transfusion_accuracy_wrt_depth[i] = format(Transfusion_avg_depth_0_accuracy/3, '.2f')
        print('Transfusion_depth_', i, ':', Transfusion_accuracy_wrt_depth[i])

print('Monk Accuracy wrt all Depths:')
print(monk_accuracy_wrt_depth)
print('')
print('Transfusion Accuracy wrt all Depths:')
print(Transfusion_accuracy_wrt_depth)


# ---------------------PLOT DEPTH VS ACCURACY--------------------
# convert dictionary to depth and accuracy individual lists for monk
for depth, accuracy in monk_accuracy_wrt_depth.items():
    monk_depth_list.append(depth)
    monk_accuracy_list.append(accuracy)

# -----PYPLOT MONK-------------
fig = plt.figure()
axes = plt.gca()
axes.set_ylim((50, 100))
plt.plot(monk_depth_list, monk_accuracy_list)
fig.suptitle('depth vs accuracy')
plt.xlabel('depth')
plt.ylabel('accuracy')
fig.savefig('monk_depth_vs_accuracy.jpg')


# convert dictionary to depth and accuracy individual lists for Transfusion
for depth, accuracy in Transfusion_accuracy_wrt_depth.items():
    Transfusion_depth_list.append(depth)
    Transfusion_accuracy_list.append(accuracy)

# -----PYPLOT TRANSFUSION-------------
fig = plt.figure()
axes = plt.gca()
axes.set_ylim((65, 80))
plt.plot(Transfusion_depth_list, Transfusion_accuracy_list)
fig.suptitle('depth vs accuracy')
plt.xlabel('depth')
plt.ylabel('accuracy')
fig.savefig('Transfusion_depth_vs_accuracy.jpg')


