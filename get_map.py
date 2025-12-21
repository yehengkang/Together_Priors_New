import os
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm

from yolo import YOLO
from utils.utils import get_classes
from utils.utils_map import get_coco_map, get_map
import wandb


def extract_datetime_from_filenames(filenames):
    pattern = r"loss_\d+_(\d+)_(\d+)_(\d+)_"

    m = re.search(pattern, filenames)
    if m:
        month, day, hour = m.groups()
        return month + day + '-' + hour


if __name__ == "__main__":
    # wandb.init(project="feature-visualization", name="run1")
    map_mode        = 0
    map_vis         = True
    MINOVERLAP      = [0.5]
    # MINOVERLAP      = [0.5,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
    DEBUG           = True
    with_clear      = False
    if DEBUG: map_mode = 1  # 仅预测

    classes_path    = 'model_data/rtts_classes.txt'
    model_path      = 'logs/loss_2025_12_20_21_22_55_priors_city/ep100-loss1.335-val_loss2.666.pth'
    VOCdevkit_path  = 'datasets/cityscapes_foggy_val_0.01.txt'
    if Path(VOCdevkit_path).suffix == '.txt':
        from_txt = True

    data_time       = extract_datetime_from_filenames(model_path.split('/')[1])
    map_out_path = 'map_out/test_' + data_time + '_priors_city' + '/'
    if DEBUG:
        map_out_path = map_out_path.replace('test_', 'debug_')

    #image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Main/test.txt")).read().strip().split()
    image_ids = open("datasets/test.txt").read().strip().split()

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    class_names, _ = get_classes(classes_path)
    if from_txt:
        class_names[3] = "motorcycle"

    if map_mode == 0 or map_mode == 1:
        see_feat_in_dir = False
        if DEBUG:
            see_feat_in_dir = True
        print("Load model.")
        yolo = YOLO(model_path = model_path, confidence = 0.001,nms_iou = 0.65)
        print("Load model done.")
        # all_name = os.listdir("./RTTStest/images")
        print("Get predict result.")
        if from_txt:
            txt_path = Path(VOCdevkit_path)
            with txt_path.open('r', encoding='utf-8') as f:
                image_path_all = f.read()
            for idx, txt_line in tqdm(enumerate(image_path_all.split('\n'))):
                if DEBUG and idx > 100: break
                image_path = txt_line.split(' ')[0]
                image_name = Path(os.path.basename(image_path)).stem
                image      = Image.open(image_path)
                if map_vis:
                    image.save(os.path.join(map_out_path, "images-optional/" + image_name + ".jpg"))
                clear_img = None
                if with_clear:
                    clear_img = Image.open(image_path.replace("VOCtest-FOG", "JPEGImages"))
                
                yolo.get_map_txt(image_name, image, class_names, map_out_path, clear_img=clear_img, see_feat_in_dir = see_feat_in_dir)
        else:
            for idx, image_id in tqdm(enumerate(image_ids)):
                if DEBUG and idx > 100: break
                image_path = os.path.join(VOCdevkit_path, "VOCtest-FOG/" + image_id + ".jpg")

                format_a = image_id + ".jpg"
                format_b = image_id + '.jpeg'
                format_c = image_id + '.png'

                image = Image.open(image_path)
                if map_vis:
                    image.save(os.path.join(map_out_path, "images-optional/" + image_id + ".jpg"))
                clear_img = None
                if with_clear:
                    clear_img = Image.open(image_path.replace("VOCtest-FOG", "JPEGImages"))
                
                yolo.get_map_txt(image_id, image, class_names, map_out_path, clear_img=clear_img, see_feat_in_dir = see_feat_in_dir)

        print("Get predict result done.")
        
    if map_mode == 0 or map_mode == 2:
        print("Get ground truth result.")
        if from_txt:
            class_to_id = {
                "person": 0,
                "bicycle": 1,
                "car": 2,
                "motorcycle": 3,
                "bus": 4
            }
            inverted_class_to_id = {v: k for k, v in class_to_id.items()}
                
            txt_path = Path(VOCdevkit_path)
            with txt_path.open('r', encoding='utf-8') as f:
                image_path_all = f.read()
            for idx, txt_line in tqdm(enumerate(image_path_all.split('\n'))):
                image_path = txt_line.split(' ')[0]
                image_name = Path(os.path.basename(image_path)).stem
                with open(os.path.join(map_out_path, "ground-truth/"+image_name+".txt"), "w") as new_f:
                    objects = txt_line.split(' ')[1:]
                    for obj in objects:
                        obj_info = obj.split(',')
                        obj_name = inverted_class_to_id[int(obj_info[4])]
                        if obj_name not in class_names:
                            continue
                        left    = obj_info[0]
                        top     = obj_info[1]
                        right   = obj_info[2]
                        bottom  = obj_info[3]

                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        else:
            for idx, image_id in tqdm(enumerate(image_ids)):
                with open(os.path.join(map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
                    root = ET.parse(os.path.join(VOCdevkit_path, "Annotations/"+image_id+".xml")).getroot()
                    for obj in root.findall('object'):
                        difficult_flag = False
                        if obj.find('difficult')!=None:
                            difficult = obj.find('difficult').text
                            if int(difficult)==1:
                                difficult_flag = True
                        obj_name = obj.find('name').text
                        if obj_name not in class_names:
                            continue
                        bndbox  = obj.find('bndbox')
                        left    = bndbox.find('xmin').text
                        top     = bndbox.find('ymin').text
                        right   = bndbox.find('xmax').text
                        bottom  = bndbox.find('ymax').text

                        if difficult_flag:
                            new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                        else:
                            new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print("Get ground truth result done.")

    if map_mode == 0 or map_mode == 3:
        print("Get map.")
        for i in MINOVERLAP:
            get_map(i, True, path = map_out_path)
        print("Get map done.")

    if map_mode == 4:
        print("Get map.")
        get_coco_map(class_names = class_names, path = map_out_path)
        print("Get map done.")
