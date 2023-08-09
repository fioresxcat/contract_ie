import os
import json
from pathlib import Path
import numpy as np
import yaml
import torch
import unidecode
from PIL import Image, ImageDraw, ImageFont
import pdb
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon
import cv2
import albumentations as A

normal_transform = A.Compose([
    A.GaussNoise(p=0.3),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussianBlur(blur_limit=(3,5), p=0.3),
])

geometric_transform = A.Compose([
    A.OneOf([
        A.ShiftScaleRotate(p=0.5, shift_limit=0.1, scale_limit=0.1, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255)),
        # A.Affine(p=0.3, 
        #     scale=1, 
        #     translate_percent={
        #         'x': (0, 0.1),
        #         'y': (0, 0.1)
        #     }, 
        #     rotate=0, 
        #     shear={
        #         'x': (-7, 7),
        #         'y': (-7, 7)
        #     }, 
        #     mode=cv2.BORDER_CONSTANT, 
        #     cval=(255, 255, 255), 
        #     fit_output=False
        # ),
        A.SafeRotate(p=0.5, limit=7, border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255))
    ])
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


LABEL_LIST = [
    'text',
    'marker_represented_name',
    'marker_account_number',
    'marker_seller',
    'marker_swift_code',
    'bank_name',
    'bank_address',
    'company_address',
    'marker_represented_position',
    'marker_fax',
    'tax',
    'account_number',
    'marker_tax',
    'marker_bank_address',
    'marker_bank_name',
    'phone',
    'swift_code',
    'fax',
    'represented_position',
    'company_name',
    'represented_name',
    'marker_phone',
    'marker_buyer',
    'marker_company_name',
    'marker_company_address',
]

def find_all_labels(data_dir, ls_exclude_dir=[]):
    labels = []
    for jp in Path(data_dir).rglob('*.json'):
        is_excluded = False
        for exclude_dir in ls_exclude_dir:
            if Path(exclude_dir).name in str(jp):
                is_excluded = True
                break
        if is_excluded:
            continue

        data = json.load(open(jp))
        # exclude marker seller buyer
        labels.extend([shape['label'] if shape['label'] not in ['marker_seller', 'marker_buyer'] else 'marker_company_name' for shape in data['shapes']])
    return set(labels)


def get_file_paths(data_dir, ls_exclude_dir=[]):
    ls_img_fp, ls_xml_fp, ls_json_fp = [], [], []
    for img_fp in Path(data_dir).rglob('*.jpg'):
        is_excluded = False
        for exclude_dir in ls_exclude_dir:
            if Path(exclude_dir).name in str(img_fp):
                is_excluded = True
                break
        if is_excluded:
            continue
        
        json_fp = img_fp.with_suffix('.json')
        xml_fp = img_fp.with_suffix('.xml')

        ls_xml_fp.append(str(xml_fp))
        ls_img_fp.append(str(img_fp))
        ls_json_fp.append(str(json_fp))
    
    return ls_img_fp, ls_xml_fp, ls_json_fp


def normalize_bbox(bbox, width, height):
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]

    
def parse_xml(xml_path):
    root = ET.parse(xml_path).getroot()
    objs = root.findall('object')
    boxes, obj_names = [], []
    for obj in objs:
        obj_name = obj.find('name').text
        box = obj.find('bndbox')
        xmin = int(box.find('xmin').text)
        ymin = int(box.find('ymin').text)
        xmax = int(box.find('xmax').text)
        ymax = int(box.find('ymax').text)
        boxes.append([xmin, ymin, xmax, ymax])
        obj_names.append(obj_name)
    return boxes, obj_names


def widen_box(box, percent_x, percent_y):
        xmin, ymin, xmax, ymax = box
        w = xmax - xmin
        h = ymax - ymin
        xmin -= w * percent_x
        ymin -= h * percent_y
        xmax += w * percent_x
        ymax += h * percent_y
        return (int(xmin), int(ymin), int(xmax), int(ymax))

    
def draw_json_on_img(img, json_data):
    labels = list(set(shape['label'] for shape in json_data['shapes']))
    color = {}
    for i in range(len(labels)):
        color[labels[i]] = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        
    img = img.copy()
    draw = ImageDraw.Draw(img)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.5# Draw the text on the image
    # font = ImageFont.truetype(font.font.family, font_size)
    for i, shape in enumerate(json_data['shapes']):
        polys = shape['points']
        polys = [(int(pt[0]), int(pt[1])) for pt in polys]
        label = shape['label']
        draw.polygon(polys, outline=color[label], width=2)
        # Draw the text on the image
        img = np.array(img)
        cv2.putText(img, shape['label'], (polys[0][0], polys[0][1]-5), font, font_size, color[label], thickness=1)
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
    return img
    
    
def mask_image(img, boxes, json_data, widen_range_x, widen_range_y):
    # widen block
    if isinstance(widen_range_x, list) and isinstance(widen_range_y, list):
        boxes = [widen_box(box, np.random.uniform(widen_range_x[0], widen_range_x[1]), np.random.uniform(widen_range_y[0], widen_range_y[1])) for box in boxes]
    else:
        boxes = [widen_box(box, widen_range_x, widen_range_y) for box in boxes]
    
    ls_polys2keep = []
    ls_area2keep = []
    iou_threshold = 0.
    for box_idx, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box
        box_pts = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        p_box = Polygon(box_pts)
        for shape_idx, shape in enumerate(json_data['shapes']):
            if shape_idx in ls_polys2keep:
                continue
            pts = shape['points']
            p_shape = Polygon(pts)
            intersect_area = p_box.intersection(p_shape).area
            if intersect_area / p_shape.area > iou_threshold:
                ls_polys2keep.append(shape_idx)
                pts = [coord for pt in pts for coord in pt]
                poly_xmin = min(pts[::2])
                poly_ymin = min(pts[1::2])
                poly_xmax = max(pts[::2])
                poly_ymax = max(pts[1::2])
                ls_area2keep.append((poly_xmin, poly_ymin, poly_xmax, poly_ymax))

    # mask white all area of image that is not in block
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(img.shape[1], xmax)
        ymax = min(img.shape[0], ymax)
        mask[ymin:ymax, xmin:xmax] = 255

    for area2keep in ls_area2keep:
        xmin, ymin, xmax, ymax = area2keep
        xmin = int(max(0, xmin))
        ymin = int(max(0, ymin))
        xmax = int(min(img.shape[1], xmax))
        ymax = int(min(img.shape[0], ymax))
        mask[ymin:ymax, xmin:xmax] = 255

    # mask white
    img[mask == 0] = 255

    # delete all poly that is not in block
    ls_idx2del = [idx for idx, shape in enumerate(json_data['shapes']) if idx not in ls_polys2keep]
    for idx in sorted(ls_idx2del, reverse=True):
        del json_data['shapes'][idx]

    return img, json_data
        
def get_random_area_not_in_block(img_w, img_h, block_boxes):
    ls_block_w = [box[2]-box[0] for box in block_boxes]
    ls_block_h = [box[3]-box[1] for box in block_boxes]
    min_w, max_w = min(ls_block_w), max(ls_block_w)
    min_h, max_h = min(ls_block_h), max(ls_block_h)
    w = np.random.randint(min_w, max_w)
    h = np.random.randint(min_h, max_h)
    
    mask = np.zeros((img_h, img_w))
    for xmin, ymin, xmax, ymax in block_boxes:
        mask[ymin:ymax, xmin:xmax] = 1
    for _ in range(10):
        xmin = np.random.randint(0, img_w-w)
        ymin = np.random.randint(0, img_h-h)
        if np.any(mask[ymin:ymin+h, xmin:xmin+w]==1):
            continue
        else:
            return (xmin, ymin, xmin+w, ymin+h)
    
    return None


def mask_image_poly(img: Image, poly):
    img_w, img_h = img.size
    pts = [coord for pt in poly for coord in pt]
    xmin = int(min(pts[::2]))
    ymin = int(min(pts[1::2]))
    xmax = int(max(pts[::2]))
    ymax = int(max(pts[1::2]))

    if xmin < 0 or xmax > img_w or ymin < 0 or ymax > img_h:  # neu box bi loi ra
        xmin = max(xmin, 0)
        xmax = max(0, min(xmax, img_w))
        ymin = max(ymin, 0)
        ymax = max(0, min(ymax, img_h))

    img = np.array(img)
    img[ymin:ymax, xmin:xmax] = np.random.randint(240, 255)

    return Image.fromarray(img)
    

def gen_annotation_for_img(img_fp, 
                           xml_fp, 
                           json_fp, 
                           mask_type='unmasked', 
                           widen_range_x=[0.1, 0.2], 
                           widen_range_y=[0.1, 0.25], 
                           ls_disable_marker=[], 
                           remove_accent=True, 
                           augment=False):
    
    img = Image.open(img_fp).convert("RGB")
    json_data = json.load(open(json_fp))

    is_masked = False
    if 'cThao_generated_data_Tung' in str(img_fp):
        mask_type = 'unmasked'
    if mask_type == 'masked' or (mask_type=='unified' and np.random.rand() < 0.5):
        try:
            block_boxes, obj_names = parse_xml(xml_fp)  # get detected blocks

            if np.random.rand() < 0.3:   # them ngau nhien 1 block bat nham
                try:
                    new_block = get_random_area_not_in_block(img.size[0], img.size[1], block_boxes)
                    if new_block is not None:
                        if np.random.rand() < 0.8:
                            block_boxes.append(new_block)
                        else:
                            del block_boxes[np.random.randint(0, len(block_boxes))]
                            block_boxes.append(new_block)
                except:
                    pass

            img, json_data = mask_image(np.array(img), boxes=block_boxes, json_data=json_data, widen_range_x=widen_range_x, widen_range_y=widen_range_y)
            img = Image.fromarray(img)
            is_masked = True

        except:
            pass

    if augment and np.random.rand() < 0.3:
        if np.random.rand() < 0.8:   # random drop some boxes
            size = int(0.08*len(json_data['shapes'])) if not is_masked else int(0.05*len(json_data['shapes']))
            idx2drop = np.random.choice(list(range(len(json_data['shapes']))), size=size)
            # drop on image
            for idx in idx2drop:
                shape = json_data['shapes'][idx]
                poly = shape['points']
                img = mask_image_poly(img, poly)
            # drop on json_data
            json_data['shapes'] = [shape for i, shape in enumerate(json_data['shapes']) if i not in idx2drop]

        if np.random.rand() < 0.5:  # noise contrast brightness augment
            img = normal_transform(image=np.array(img))['image']
            img = Image.fromarray(img)

        if np.random.rand() < 0.5:   # shift scale rotate auigment
            ls_orig_poly = [shape['points'] for shape in json_data['shapes']]
            keypoints = np.array([pt for poly in ls_orig_poly for pt in poly])
            transformed = geometric_transform(image=np.array(img), keypoints=keypoints)
            transformed_image = transformed['image']
            transformed_keypoints = transformed['keypoints']
            transformed_keypoints = [(int(pt[0]), int(pt[1])) for pt in transformed_keypoints]
            ls_transformed_poly = [transformed_keypoints[i:i+4] for i in range(0, len(transformed_keypoints), 4)]

            # del poly outside of image
            img_w, img_h = transformed_image.shape[1], transformed_image.shape[0]
            ls_idx2del = []
            for i, poly in enumerate(ls_transformed_poly):
                pts = [coord for pt in poly for coord in pt]
                xmin = int(min(pts[::2]))
                ymin = int(min(pts[1::2]))
                xmax = int(max(pts[::2]))
                ymax = int(max(pts[1::2]))

                if xmin < 0 or xmax > img_w or ymin < 0 or ymax > img_h:  # neu box bi loi ra
                    xmin = min(img_w, max(xmin, 0))
                    xmax = max(0, min(xmax, img_w))
                    ymin = min(img_h, max(ymin, 0))
                    ymax = max(0, min(ymax, img_h))
                    transformed_image[ymin:ymax, xmin:xmax] = np.random.randint(240, 255)  # mask white img
                    ls_idx2del.append(i)  # del this box

            new_shapes = []
            for i, shape in enumerate(json_data['shapes']):
                if i not in ls_idx2del:
                    shape['points'] = ls_transformed_poly[i]
                    new_shapes.append(shape)
            json_data['shapes'] = new_shapes

            img = Image.fromarray(transformed_image)
        
    # pdb.set_trace()
    words, orig_polys, boxes, labels = [], [], [], []
    img_w, img_h = img.size
    for i, shape in enumerate(json_data['shapes']):
        if len(shape['points']) != 4:
            continue
        if shape['label'] in ls_disable_marker:
            current_label = 'text'
        else:
            current_label = shape['label']
        
        if current_label in ['marker_seller', 'marker_buyer']:
            current_label = 'marker_company_name'
        
        if remove_accent:
            words.append(unidecode.unidecode(shape['text'].lower()))
        else:
            words.append(shape['text'].lower())
            
        labels.append(current_label)
        pts = [coord for pt in shape['points'] for coord in pt]
        xmin = min(pts[0::2])
        xmax = max(pts[0::2])
        ymin = min(pts[1::2])
        ymax = max(pts[1::2])

        xmin = max(xmin, 0)
        ymin = max(ymin, 0)
        xmax = min(img_w, xmax)
        ymax = min(img_h, ymax)

        boxes.append((xmin, ymin, xmax, ymax))
        orig_polys.append(tuple([tuple(pt) for pt in shape['points']]))

    normalized_boxes = [normalize_bbox(box, img_w, img_h) for box in boxes]
        
    return img, words, orig_polys, normalized_boxes, labels


def check_json_files(data_dir):
    ls_json_fp = list(Path(data_dir).rglob('*.json'))
    for fp in ls_json_fp:
        json_data = json.load(open(fp))
        ls_idx2del = []
        for i, shape in enumerate(json_data['shapes']):
            if len(shape['points']) != 4:
                print(f'{fp} contains shape with {len(shape["points"])} points!')
                ls_idx2del.append(i)
            if shape['label'] not in LABEL_LIST:
                print(f'{fp} contains shape with label {shape["label"]}!')
                # json_data['shapes'][i]['label'] = 'company_name'
            if 'text' not in shape:
                print(f'{fp} contains shape without text!')
    
        if len(ls_idx2del) > 0:
            json_data['shapes'] = [shape for i, shape in enumerate(json_data['shapes']) if i not in ls_idx2del]

            json.dump(json_data, open(fp, 'w'))

if __name__ == '__main__':
    check_json_files('/data/tungtx2/huggingface/latest_data_245_final')