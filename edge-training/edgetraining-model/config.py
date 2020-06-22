# path to custom data and label files. Labels must be in COCO annotation format
dataset_detection_type = "coco"
dataset_segmentation_type = "coco_kp"
root_data_dir = "/home/mahesh/workspace/source/iotedge/data/D2S"
train_data_dir = "/home/mahesh/workspace/source/iotedge/data/D2S/train"
train_coco_annotations = "/home/mahesh/workspace/source/iotedge/data/D2S/annotations/D2S_training.json"
test_data_dir = "/home/mahesh/workspace/source/iotedge/data/D2S/val"
test_coco_annotations = "/home/mahesh/workspace/source/iotedge/data/D2S/annotations/D2S_validation.json"

# Batch sizes
train_batch_size = 10
test_batch_size = 1

# Params for data loader
train_shuffle = True
test_shuffle = False
num_workers = 4

# Params for training
num_classes = 60
num_epochs = 70
lr = 0.01
lr_step_size = 8
lr_steps = [16, 22]
lr_gamma = 0.1
momentum = 0.9
weight_decay = 1e-4
aspect_ratio_group_factor = 3
print_freq = 20
