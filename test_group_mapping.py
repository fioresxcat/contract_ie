import os
os.environ['TRANSFORMERS_CACHE'] = '/data/tungtx2/tmp/transformers_hub'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from os import listdir
from torch.utils.data import Dataset
import torch
from PIL import Image
import unidecode
from PIL import Image, ImageDraw, ImageFont
import pdb
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon
import cv2

print(torch.__version__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        xmin = int(float(box.find('xmin').text))
        ymin = int(float(box.find('ymin').text))
        xmax = int(float(box.find('xmax').text))
        ymax = int(float(box.find('ymax').text))
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


def gen_annotation_for_img(img_fp, xml_fp, json_fp, mask_type='unified', widen_range_x=[0.1, 0.2], widen_range_y=[0.1, 0.25], disable_marker=False, remove_accent=True, augment=False):
    img = Image.open(img_fp).convert("RGB")
    json_data = json.load(open(json_fp))
    
    is_masked = False
    if mask_type == 'masked' or (mask_type=='unified' and np.random.rand() < 0.5):
        block_boxes, obj_names = parse_xml(xml_fp)
        img, json_data = mask_image(np.array(img), boxes=block_boxes, json_data=json_data, widen_range_x=widen_range_x, widen_range_y=widen_range_y)
        img = Image.fromarray(img)
        is_masked = True
    
    if augment and np.random.rand() < 0.3:  # random drop some boxes
        size = int(0.08*len(json_data['shapes'])) if not is_masked else int(0.05*len(json_data['shapes']))
        idx2drop = np.random.choice(list(range(len(json_data['shapes']))), size=size)
        json_data['shapes'] = [shape for i, shape in enumerate(json_data['shapes']) if i not in idx2drop]
            
    # pdb.set_trace()
        
    words, orig_polys, normalized_boxes, labels = [], [], [], []
    img_h, img_w = json_data['imageHeight'], json_data['imageWidth']
    for i, shape in enumerate(json_data['shapes']):
        if disable_marker and 'marker' in shape['label']:
            current_label = 'text'
        else:
            current_label = shape['label']
        
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

        normalized_boxes.append(normalize_bbox((xmin, ymin, xmax, ymax), img_w, img_h))
        orig_polys.append(tuple([tuple(pt) for pt in shape['points']]))
    
    return img, words, orig_polys, normalized_boxes, labels

from transformers import LayoutLMv3Processor, LayoutLMv3Model

processor = LayoutLMv3Processor.from_pretrained('microsoft/layoutlmv3-base', apply_ocr=False)
processor.tokenizer.only_label_first_subword = False

label2id = {'swift_code': 0, 'marker_swift_code': 1, 'bank_name': 2, 'company_name': 3, 'tax': 4, 'bank_address': 5, 'marker_bank_address': 6, 'marker_represented_name': 7, 'marker_account_number': 8, 'marker_represented_position': 9, 'marker_fax': 10, 'marker_phone': 11, 'marker_company_address': 12, 'text': 13, 'marker_bank_name': 14, 'account_number': 15, 'company_address': 16, 'fax': 17, 'marker_tax': 18, 'marker_company_name': 19, 'represented_position': 20, 'phone': 21, 'represented_name': 22}

img_fp = Path('real_data/val_labeled_ocred/CTR292 (1)-001_0.jpg')
img, words, orig_polys, normalized_boxes, labels = gen_annotation_for_img(img_fp=img_fp, 
                                                                          xml_fp=img_fp.with_suffix('.xml'),
                                                                         json_fp=img_fp.with_suffix('.json'),
                                                                         mask_type='unmasked')
idx_labels = [label2id[label] for label in labels]
# encode input for model
encoded_inputs = processor(img, words, boxes=normalized_boxes, word_labels=idx_labels, truncation=True, stride=128,
                           padding="max_length", max_length=512, return_overflowing_tokens=True, return_offsets_mapping=True, return_tensors="pt")
encoded_inputs.pop('overflow_to_sample_mapping')
encoded_inputs.pop('offset_mapping')


input_ids = encoded_inputs['input_ids'].to(device)
bbox = encoded_inputs['bbox'].to(device)
attention_mask = encoded_inputs['attention_mask'].to(device)
pixel_values = torch.stack(encoded_inputs['pixel_values'], dim=0).to(device)
labels = encoded_inputs['labels'].to(device)

# from transformers import LayoutLMv3ForTokenClassification

# lmv3_model = LayoutLMv3ForTokenClassification.from_pretrained('microsoft/layoutlmv3-base', num_labels=23).to(device)
import torch.nn as nn
from transformers import LayoutLMv3PreTrainedModel

class LayoutLMv3ClassificationHead(nn.Module):
    """
    Head for sentence-level classification tasks. Reference: RobertaClassificationHead
    """

    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)
        self.dropout = nn.Dropout(p=0.1, inplace=False)
        self.out_proj =  nn.Linear(in_features=config.hidden_size, out_features=num_labels, bias=True)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    
    
class MyTransformerForTokenClassification(LayoutLMv3PreTrainedModel):
    def __init__(self, backbone, dropout_prob, num_groups, num_labels):
        super().__init__(backbone.config)
        self.backbone = backbone
        self.config = self.backbone.config
        self.dropout = nn.Dropout(dropout_prob)
        self.num_labels = num_labels
        self.num_groups = num_groups
        self.group_classifier = nn.Linear(self.config.hidden_size, num_groups)
        self.ner_classifier = LayoutLMv3ClassificationHead(self.config, num_labels)

            
    def forward(self, pixel_values, input_ids, bbox, attention_mask):
        backbone_out = self.backbone(pixel_values=pixel_values, input_ids=input_ids, bbox=bbox, attention_mask=attention_mask)
        if input_ids is not None:
            input_shape = input_ids.size()  # (batch, sequence length)
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        # only take the text part of the output representations
        sequence_output = backbone_out[0][:, :seq_length]
        sequence_output = self.dropout(sequence_output)
        
        ner_logits = self.ner_classifier(sequence_output)
        group_logits = self.group_classifier(sequence_output)

#         loss = None
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

#         if not return_dict:
#             output = (logits,) + outputs[1:]
#             return ((loss,) + output) if loss is not None else output

#         return TokenClassifierOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )
        # pdb.set_trace()
        return ner_logits, group_logits
    
backbone = LayoutLMv3Model.from_pretrained('microsoft/layoutlmv3-base')
dropout_prob = 0.1
num_groups = 4
num_labels = 23
lmv3_model = MyTransformerForTokenClassification(backbone, dropout_prob, num_groups, num_labels).to(device)


import shutil
import torch
import torch.nn as nn

optimizer = torch.optim.AdamW(lmv3_model.parameters(), lr=5e-5)
criterion_ner = nn.CrossEntropyLoss()
criterion_group = nn.CrossEntropyLoss()
num_train_epochs = 300

group_label = torch.randint(low=0, high=4, size=(2, 512), dtype=torch.int64).to(device)
lmv3_model.train()
lmv3_model.backbone.train()
for epoch in range(num_train_epochs):
    # zero the parameter gradients
    optimizer.zero_grad()
    # forward + backward + optimize
    ner_out, group_out = lmv3_model(input_ids=input_ids,
                            bbox=bbox,
                            pixel_values=pixel_values,
                            attention_mask=attention_mask)
    loss_ner = criterion_ner(ner_out.view(-1, lmv3_model.num_labels), labels.view(-1))
    loss_group = criterion_group(group_out.view(-1, lmv3_model.num_groups), group_label.view(-1))
    loss = loss_ner +loss_group
    loss.backward()
    optimizer.step()
    print('loss: ', loss)