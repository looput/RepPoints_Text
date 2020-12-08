from operator import le
import os
from os import path
import numpy as np
import json
from dataset import DatasetSplit, DatasetRegistry
from dataset.eval_tools.eval import eval_result

__all__ = ["register_text"]


class TextDection(DatasetSplit):
    def __init__(self, base_dir, split):
        # assert split in ["train", "test"]
        base_dir = os.path.expanduser(base_dir)
        self.base_dir = base_dir
        # self.imgdir = os.path.join(base_dir, split)
        # assert os.path.isdir(self.imgdir), self.imgdir
    
    def load(self,split,with_gt=True):
        sample_file = os.path.join(self.base_dir, f"{split}_list.txt")
        if os.path.isfile(sample_file):
            with open(sample_file) as f:
                sample_list = f.readlines()
            img_gt_lst = [sampe_line.strip().split(',') for sampe_line in sample_list] 
        else:
            allfiles = [os.path.join(root,f) for root,dirs,files in os.walk(self.base_dir) for f in files]
            # 'bmp','BMP'
            imgpath_list = [fname \
                for fname in allfiles if fname.endswith(('.jpg','.JPG','.JPEG','.PNG','.png','.jpeg'))]
            img_gt_lst = [(img_pth,os.path.splitext(img_pth)[0]+'.txt') for img_pth in imgpath_list]
        ret = []
        for image_id,sampe_line in enumerate(img_gt_lst):
            img_fname, gt_fname =sampe_line
            if os.path.isfile(img_fname):
                fname,gt_fname = img_fname, gt_fname
            else:
                fname = os.path.join(self.base_dir,split, img_fname)
                gt_fname = os.path.join(self.base_dir,split, gt_fname)

            if not os.path.isfile(gt_fname):
                # print('no gt found')
                continue
            roidb = {"file_name": fname}
            roidb['image_id'] = image_id

            boxes = []
            segs = []
            crowds = []
            reader = open(gt_fname).readlines()
            for line in reader:
                line=line.replace("\ufeff", "")

                parts = line.strip().split(';')
                num_parts = parts[1:-1]
                num_parts = [nn for nn in num_parts if nn.isdigit()]
                num_parts = num_parts[:(len(num_parts)//2)*2]
                text = ''.join(parts[1+len(num_parts):])
                poly = np.asarray(num_parts, dtype=np.float32)
                poly = poly.reshape(-1,2)+0.5

                maxxy = poly.max(axis=0)
                minxy = poly.min(axis=0)

                boxes.append([minxy[0], minxy[1], maxxy[0], maxxy[1]])
                segs.append([poly])

                if text=='###' or 'arc_seal' in text or 'rect_seal' in text or 'square_stamp' in text or 'dots' in text:
                    crowds.append(1)
                else:
                    crowds.append(0)

            N = len(boxes)
            roidb["boxes"] = np.asarray(boxes, dtype=np.float32)
            roidb["segmentation"] = segs
            roidb["class"] = np.ones((N, ), dtype=np.int32)
            roidb["is_crowd"] = np.array(crowds, dtype=np.int8)
            ret.append(roidb)
        return ret
        

    def training_roidbs(self):
        return self.load('train',with_gt=True)

    def inference_roidbs(self):
        return self.load('test',with_gt=True)
    
    def eval_inference_results(self, results, output=None):
        # import pudb; pudb.set_trace()
        if not os.path.isdir(output):
            # os.removedirs(output)
            os.makedirs(output)
        for idx,res_per_img in enumerate(results):
            lines = []
            for b_idx, instance in enumerate(res_per_img):
                poly= instance['polygons'].reshape(-1).tolist()
                values = [int(v) for v in poly]
                # values.append(angle[b_idx])
                # line = os.path.basename(instance['image_id'])+';'+"%d"%values[0]
                line = "%d"%values[0]
                for v_id in range(1, len(values)):
                    line += "; %d"%values[v_id]
                line +='; 0'
                line += '\n'
                lines.append(line)

            if len(res_per_img)>0:
                image_name=os.path.basename(res_per_img[0]['image_id'])
                filename=os.path.join(output,image_name.split('.')[0]+'.txt')

                with open(filename,'w') as f:
                    for line in lines:
                        f.write(line)

        # import pudb; pudb.set_trace()
        res = eval_result(output,self.base_dir)

        return res
        # if len(results):
        #     # sometimes may crash if the results are empty?
        #     return self.print_coco_metrics(results)
        # else:
        #     return {}

def register_text(basedir):
    for split in ["train", "test"]:
        name = "text_" + split
        DatasetRegistry.register(name, lambda x=split: TextDection(basedir, x))
        DatasetRegistry.register_metadata(name, "class_names", ["BG", "Text"])

def register_text_full(basedir):
    dataset_list = ['zhongyuan','insurance_form','jinyu_medical','pufa_v2','tiny_invoice','lading_bill','zhongyuan_v2','credit_real_pufa','train_tickets','train_tickets_1023','zhongchuang','taobao_text','gaopaiyi','financial_statement','zhongchuang_v2','Docs_elec','table_text','tiny_docs','pufa_gen_1022','gen_doc_1021']
    for idx,split in enumerate(dataset_list):
        name = f'general_text_{idx}'
        path = os.path.join(basedir,split)
        # DatasetRegistry.register(name, lambda x=split: TextDection(path, x))
        DatasetRegistry.register(name,lambda x=path: TextDection(x, ''))
        DatasetRegistry.register_metadata(name, "class_names", ["BG", "Text"])

if __name__ == '__main__':
    basedir = '/home/lupu/27_screenshot/'
    roidbs = TextDection(basedir, "zhongyuan/train").training_roidbs()
    print("#images:", len(roidbs))

    from viz import draw_annotation
    from tensorpack.utils.viz import interactive_imshow as imshow
    import cv2
    for r in roidbs:
        im = cv2.imread(r["file_name"])
        vis = draw_annotation(im, r["boxes"], r["class"], r["segmentation"])
        imshow(vis)
