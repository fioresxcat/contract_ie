from torch.utils.data import Dataset
from myutils import *
import numpy as np

# class CORDDataset(Dataset):

#     def __init__(self, 
#                  file_paths, 
#                  processor=None,
#                  label2id=None,
#                  model_type=None,
#                  max_length=512, 
#                  mask_type='unified', 
#                  widen_range_x=[0., 0.01], 
#                  widen_range_y=[0.15, 0.3], 
#                  ls_disable_marker=[], 
#                  augment=False,
#                  remove_accent = False):
        
#         self.ls_img_fp, self.ls_xml_fp, self.ls_json_fp = file_paths
#         assert len(self.ls_img_fp) == len(self.ls_json_fp) == len(self.ls_xml_fp)
#         self.processor = processor
#         self.label2id = label2id
#         self.model_type = model_type
#         self.mask_type = mask_type
#         self.widen_range_x = widen_range_x
#         self.widen_range_y = widen_range_y
#         self.ls_disable_marker = ls_disable_marker
#         self.augment = augment
#         self.remove_accent = remove_accent

#     def __len__(self):
#         return len(self.ls_img_fp)

#     def __getitem__(self, index):
#         # first, take an image
#         img_fp = self.ls_img_fp[index]
#         xml_fp = self.ls_xml_fp[index]
#         json_fp = self.ls_json_fp[index]
        
#         img, words, _, boxes, text_labels = gen_annotation_for_img(img_fp, xml_fp, json_fp, 
#                                                                    mask_type=self.mask_type, 
#                                                                    widen_range_x=self.widen_range_x, widen_range_y=self.widen_range_y, 
#                                                                    ls_disable_marker=self.ls_disable_marker, 
#                                                                    augment=self.augment,
#                                                                    remove_accent=self.remove_accent)
#         idx_labels = [self.label2id[label] for label in text_labels]

#         if self.model_type == 'lilt':
#             encoded_inputs = self.processor(img, words, boxes=boxes, word_labels=idx_labels, truncation=True, stride =128, 
#                                 padding="max_length", max_length=512, return_overflowing_tokens=True, return_offsets_mapping=True, return_tensors="pt")  
#             encoded_inputs.pop('overflow_to_sample_mapping')
#             encoded_inputs.pop('offset_mapping')
#             encoded_inputs.pop('image')

#         elif self.model_type == 'layoutlmv3':
#             encoded_inputs = self.processor(img, words, boxes=boxes, word_labels=idx_labels, truncation=True, stride =128, 
#                             padding="max_length", max_length=512, return_overflowing_tokens=True, return_offsets_mapping=True, return_tensors="pt")  
#             encoded_inputs.pop('overflow_to_sample_mapping')
#             encoded_inputs.pop('offset_mapping')

#         # remove batch dimension
#         idx = np.random.randint(0, len(encoded_inputs['bbox']))
#         for k, v in encoded_inputs.items():
#             encoded_inputs[k] = v[idx]
      
#         return encoded_inputs
    

class TestCORDDataset(Dataset):

    def __init__(self,
                 mode,
                 file_paths, 
                 processor=None,
                 label2id=None,
                 model_type=None,
                 max_length=512, 
                 mask_type='unmasked', 
                 widen_range_x=[0., 0.01], 
                 widen_range_y=[0.15, 0.3], 
                 ls_disable_marker=[], 
                 augment=False,
                 remove_accent = False,
                 stride=128,
                 carefully_choose_idx=False):
        
        self.mode = mode
        self.ls_img_fp, self.ls_xml_fp, self.ls_json_fp = file_paths
        assert len(self.ls_img_fp) == len(self.ls_json_fp) == len(self.ls_xml_fp)
        self.processor = processor
        self.label2id = label2id
        self.model_type = model_type
        self.mask_type = mask_type
        self.widen_range_x = widen_range_x
        self.widen_range_y = widen_range_y
        self.ls_disable_marker = ls_disable_marker
        self.augment = augment
        self.remove_accent = remove_accent
        self.stride = stride
        self.carefully_choose_idx = carefully_choose_idx
        
        if self.carefully_choose_idx:
            self.multi_split = {}

    def __len__(self):
        return len(self.ls_img_fp)

    def __getitem__(self, index):
        # first, take an image
        img_fp = self.ls_img_fp[index]
        xml_fp = self.ls_xml_fp[index]
        json_fp = self.ls_json_fp[index]
        
        img, words, _, boxes, text_labels = gen_annotation_for_img(img_fp, xml_fp, json_fp, 
                                                                   mask_type=self.mask_type, 
                                                                   widen_range_x=self.widen_range_x, widen_range_y=self.widen_range_y, 
                                                                   ls_disable_marker=self.ls_disable_marker, 
                                                                   augment=self.augment,
                                                                   remove_accent=self.remove_accent)
        idx_labels = [self.label2id[label] for label in text_labels]

        encoded_inputs = self.processor(img, words, boxes=boxes, word_labels=idx_labels, truncation=True, stride = self.stride, 
                    padding="max_length", max_length=512, return_overflowing_tokens=True, return_offsets_mapping=True, return_tensors="pt")  
        if self.model_type == 'lilt':
            encoded_inputs.pop('overflow_to_sample_mapping')
            encoded_inputs.pop('offset_mapping')
            encoded_inputs.pop('image')
        elif self.model_type == 'layoutlmv3':
            encoded_inputs.pop('overflow_to_sample_mapping')
            encoded_inputs.pop('offset_mapping')

        if self.mode == 'train':
            # remove batch dimension    
            if self.carefully_choose_idx:
                if str(img_fp) not in self.multi_split:
                    idx = np.random.randint(0, len(encoded_inputs['bbox']))
                    self.multi_split[str(img_fp)] = [idx]
                else:
                    if len(self.multi_split[str(img_fp)]) >= len(encoded_inputs['bbox']):
                        idx = np.random.randint(0, len(encoded_inputs['bbox']))
                    else:
                        ls_available_idx = list(set(range(len(encoded_inputs['bbox']))) - set(self.multi_split[str(img_fp)]))
                        idx = np.random.choice(ls_available_idx)
                        self.multi_split[str(img_fp)].append(idx)
            else:
                idx = np.random.randint(0, len(encoded_inputs['bbox']))

            for k, v in encoded_inputs.items():
                encoded_inputs[k] = v[idx]

        elif self.mode == 'val':
            if self.model_type == 'layoutlmv3':
                encoded_inputs['pixel_values'] = torch.stack(encoded_inputs['pixel_values'], dim=0)

        return encoded_inputs


def check_data_loader(data_loader):
    for item in data_loader:
        for k, v in item.items():
            print('\t', k, v.shape)
        break
    print()


    
