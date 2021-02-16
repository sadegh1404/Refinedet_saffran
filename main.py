import argparse
import numpy as np
import os
from os import path
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

from models import RefineDetVGG16
from utils import read_jpeg_image, resize_image_and_boxes, absolute2relative
from saffran.saffran_data_loader import load_saffran_dataset
from saffran.augmentations import Augmentation

from saffran.config import IMAGE_SIZE, BATCH_SIZE, SHUFFLE_BUFFER, NUM_CLASS, LR_SCHEDULE, MOMENTUM, NUM_EPOCHS, STEPS_PER_EPOCH



parser = argparse.ArgumentParser()
parser.add_argument('--saffran_root', type=str, default='./data/Saffron_Dataset/Labeled/',
                    help='Path to the VOCdevkit directory.')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='Path to the weights file, in the case of resuming training.')
parser.add_argument('--initial_epoch', type=int, default=0,
                    help='Starting epoch. Give a value bigger than zero to resume training.')
parser.add_argument('--batch_size', type=int, default=None,
                    help='Useful for quick tests. If not provided, the value in the config file is used instead.')
args = parser.parse_args()

BATCH_SIZE = args.batch_size or BATCH_SIZE


def build_dataset(img_paths, bboxes, repeat=False, shuffle=False,
                  drop_remainder=False, augmentation_fn=None):
    row_lengths = [len(img_bboxes) for img_bboxes in bboxes]
    bboxes_concat = np.concatenate(bboxes, axis=0)
    
    bboxes = tf.RaggedTensor.from_row_lengths(values=bboxes_concat,
                                              row_lengths=row_lengths)
    
    
    dataset = tf.data.Dataset.from_tensor_slices((img_paths, bboxes))

    if repeat:
        dataset = dataset.repeat()
    if shuffle:
        dataset = dataset.shuffle(len(img_paths),
                                  reshuffle_each_iteration=True)

    dataset = dataset.map(lambda img_path, boxes:
                          (read_jpeg_image(img_path), boxes))

    if augmentation_fn:
        dataset = dataset.map(augmentation_fn)

    dataset = dataset.map(lambda image, boxes:
                          resize_image_and_boxes(image, boxes, IMAGE_SIZE))
    dataset = dataset.map(lambda image, boxes:
                          (image, absolute2relative(boxes, tf.shape(image))))

    # This hack is to allow batching into ragged tensors
    dataset = dataset.map(lambda image, boxes:
                          (image, tf.expand_dims(boxes, 0)))        
    dataset = dataset.map(lambda image, boxes:
                          (image, tf.RaggedTensor.from_tensor(boxes)))
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=drop_remainder)
    dataset = dataset.map(lambda image, boxes:
                          (image, boxes.merge_dims(1, 2)))
    
    return dataset



train_img_paths, train_bboxes = load_saffran_dataset(dataroot=args.saffran_root)
print('INFO: Loaded %d training samples' % len(train_img_paths))

# Classes starts at 0
for i in train_bboxes:
  i[:,-1] = i[:,-1] -1


train_data = build_dataset(train_img_paths, train_bboxes,
                           repeat=True, shuffle=True, drop_remainder=True,
                           augmentation_fn=Augmentation())

print(train_data)

print('INFO: Instantiating model...')

model = RefineDetVGG16(num_classes=NUM_CLASS,aspect_ratios=[1.0])
model.build(input_shape=(BATCH_SIZE, IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

if args.checkpoint:
    model.load_weights(args.checkpoint)
else:
    model.base.load_weights(
        path.join('weights', 'VGG_ILSVRC_16_layers_fc_reduced.h5'), by_name=True)


lr_scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(*LR_SCHEDULE)
optimizer = tf.keras.optimizers.SGD(lr_scheduler, momentum=MOMENTUM)
optimizer.iterations = tf.Variable(STEPS_PER_EPOCH * args.initial_epoch)

print('Trainint at learning rate =', optimizer._decayed_lr(tf.float32))

model.compile(optimizer=optimizer)

os.makedirs('weights', exist_ok=True)
callbacks = [
    ModelCheckpoint(path.join('weights', 'refinedet_vgg16_{epoch:0d}.h5'),
                    monitor='total_loss')    
]

history = model.fit(train_data, epochs=NUM_EPOCHS, steps_per_epoch=STEPS_PER_EPOCH,
          initial_epoch=args.initial_epoch, callbacks=callbacks)

import cv2 
import matplotlib.pyplot as plt

def sind(x):
    return np.sin(x / 180*np.pi)
    
def cosd(x):
    return np.cos(x / 180*np.pi)
    
def draw_line_segment(image, center, angle, color, length=40, thickness=3):
    x1 = center[0] - cosd(angle) * length / 2
    x2 = center[0] + cosd(angle) * length / 2
    y1 = center[1] - sind(angle) * length / 2
    y2 = center[1] + sind(angle) * length / 2

    cv2.line(image, (int(x1 + .5), int(y1 + .5)), (int(x2 + .5), int(y2 + .5)), color, thickness)

def draw_ouput_lines(centers_box,test,print_conf=False,resize=False):
    out = []
    if resize:
        test = cv2.resize(test,resize)
        SIZE2,SIZE1 = resize
    else:
        SIZE1,SIZE2 = 640,640
    if print_conf:
        print(centers_box[:,-1])
    for i in centers_box:
        cx = i[0] * SIZE2
        cy = i[1] * SIZE1 
        label = i[-2]
        confidence = i[-1]
        angle = np.arccos(label/NUM_CLASS)*(180/np.pi)
        draw_line_segment(test,(cx,cy),angle,(255,255,0))
        out.append('{} {} {} {}'.format(str(cx),str(cy),str(angle),str(confidence)))
    plt.figure(figsize=(10,10))
    plt.imshow(test)
    plt.show()
    return out

SIZE=640
test_dir = 'data/Saffron_Dataset/Test/' # CHANGE HERE TO CHANGE TEST DIRECTORY 
test_images = os.listdir(test_dir)
for img_name in test_images:
    if img_name.endswith('.txt'):
        continue
    img = cv2.imread(test_dir+img_name)
    img = img.astype(np.float64)
    org_shape = img.shape
    img = cv2.resize(img,(SIZE,SIZE))
    img = np.expand_dims(img,0)
    out_boxes = model(img,decode=True)
    nms_box = NMS(out_boxes[0],top_k=500,nms_threshold=0.1)
    centers_box = minmax2xywh(nms_box)

    out = draw_ouput_lines(centers_box,img[0].astype(np.uint8),False,resize=org_shape[:2][::-1])
    out = '\n'.join(out)
    with open(test_dir + img_name.split('.')[0]+'.txt','w') as f:
        f.write(out)