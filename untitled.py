import os
import shutil
from pathlib import Path
import numpy as np
# train_dir = '/data/tungtx2/huggingface/unified_data_fixed_marker_2/train/train_labeled_ocred_fixed_marker_2'
# val_dir = '/data/tungtx2/huggingface/unified_data_fixed_marker_2/val_labeled_ocred_fixed_marker_2'

# # find all jpg in val_dir that also in train_dir and move those files in train_dir to '/data/tungtx2/huggingface/unified_data_fixed_marker_2/train_dupicate'
# out_dir = '/data/tungtx2/huggingface/unified_data_fixed_marker_2/train_dupicate'
# os.makedirs(out_dir, exist_ok=True)
# ls_train_fp = list(Path(train_dir).glob('*.json'))
# ls_train_name = [fp.name.split()[0] for fp in ls_train_fp]

# for jpg_fp in Path(val_dir).glob('*.json'):
#     name = jpg_fp.name
#     name = name.split()[0]
#     if name in ls_train_name:
#         idx = ls_train_name.index(name)
#         fp2move = ls_train_fp[idx]
#         xml_fp2move = fp2move.with_suffix('.xml')
#         shutil.move(str(fp2move), out_dir)
#         shutil.move(str(xml_fp2move), out_dir)
#         print(f'move {ls_train_fp[idx].name} to {out_dir}')

root_dir = '/data/tungtx2/huggingface/latest_data_245_final/train/result_01042023_regen_fixed_marker_scanned_text_detected_ocred_mapped'
# delete 20% of jpg file in root_dir
ls_fp = list(Path(root_dir).glob('*.jpg'))
np.random.shuffle(ls_fp)
ls_fp_to_del = ls_fp[:116]
for fp in ls_fp_to_del:
    xml_fp = fp.with_suffix('.xml')
    json_fp = fp.with_suffix('.json')
    os.remove(str(fp))
    os.remove(str(xml_fp))
    os.remove(str(json_fp))
    print(f'delete {fp.name}')
