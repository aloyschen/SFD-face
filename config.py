learning_rate = 0.001
weight_decay = 0.0005
neg_pos_ratio = 3
pos_anchor_threshold = 0.3
train_batch = 12
Epoch_num = 200
print_iter = 3
conf_threshold = 0.9
nms_threshold = 0.5
top_k = 200
anchor_stride = [4, 8, 16, 32, 64, 128]
anchor_size = [16, 32, 64, 128, 256, 512]
image_size = [640, 640]
gpu_index = "0"
checkpoint_dir = './train_model/'
vgg_path = './vgg16-397923af.pth'
wider_face_train_dir = '/data0/gaochen3/FaceDetect/data/WIDER_train/images/'
wider_face_val_dir = '/data0/gaochen3/FaceDetect/data/WIDER_val/images/'
wider_face_train_annotations = '/data0/gaochen3/FaceDetect/data/wider_face_split/wider_face_train_bbx_gt.txt'
wider_face_val_annotations = '/data0/gaochen3/FaceDetect/data/wider_face_split/wider_face_val_bbx_gt.txt'
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
