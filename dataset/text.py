import os
import numpy as np
import json
from dataset import DatasetSplit, DatasetRegistry

__all__ = ["register_text"]


class TextDection(DatasetSplit):
    def __init__(self, base_dir, split):
        assert split in ["train", "test"]
        base_dir = os.path.expanduser(base_dir)
        self.base_dir = base_dir
        # self.imgdir = os.path.join(base_dir, split)
        # assert os.path.isdir(self.imgdir), self.imgdir
    
    def load(self,split,with_gt=True):
        sample_file = os.path.join(self.base_dir, f"{split}_list.txt")
        with open(sample_file) as f:
            sample_list = f.readlines()
        
        ret = []
        for image_id,sampe_line in enumerate(sample_list):
            img_fname, gt_fname =sampe_line.strip().split(',')
            fname = os.path.join(self.base_dir,split, img_fname)
            gt_fname = os.path.join(self.base_dir,split, gt_fname)

            if not (fname.endswith('.jpg') or fname.endswith('.JPG') \
                or fname.endswith('.JPEG') or fname.endswith('.PNG') \
                or fname.endswith('.png')):
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

                if text=='###':
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

        if output is not None:
            with open(output, 'w') as f:
                json.dump(results, f)
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


if __name__ == '__main__':
    basedir = '/home/lupu/27_screenshot/LSVT'
    roidbs = TextDection(basedir, "train").training_roidbs()
    print("#images:", len(roidbs))

    from viz import draw_annotation
    from tensorpack.utils.viz import interactive_imshow as imshow
    import cv2
    for r in roidbs:
        im = cv2.imread(r["file_name"])
        vis = draw_annotation(im, r["boxes"], r["class"], r["segmentation"])
        imshow(vis)
