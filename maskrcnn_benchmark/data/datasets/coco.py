# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints


min_keypoints_per_image = 10


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False


class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.transforms = transforms

    def __getitem__(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)
        # img: an PIL image('RBG')
        # anno: [{'segmentation': [[164.5, 479.38, 120.26, 448.4, 93.7, 442.87, 91.5, 440.65, 
        # 36.17, 383.12, 28.43, 384.23, 35.06, 382.02, 4.1, 307.89, 9.62, 297.94, 6.3, 224.92, 
        # 0.0, 224.92, 9.62, 219.4, 38.39, 146.38, 52.77, 143.06, 111.4, 88.85, 106.97, 78.89, 
        # 119.16, 83.32, 204.34, 61.2, 203.23, 50.12, 208.76, 57.87, 302.8, 70.04, 302.8, 63.4, 
        # 306.12, 71.15, 383.55, 120.92, 387.98, 117.62, 383.55, 124.25, 433.34, 193.94, 439.97, 
        # 192.83, 433.34, 199.48, 452.14, 274.71, 457.68, 274.71, 451.05, 280.23, 434.45, 355.47, 
        # 436.67, 364.33, 428.92, 358.79, 395.72, 404.15, 380.23, 327.81, 362.54, 318.96, 341.52, 
        # 310.11, 344.84, 255.9, 344.84, 221.6, 332.67, 200.59, 326.03, 197.26, 318.29, 171.82, 
        # 288.42, 160.76, 265.18, 157.44, 245.27, 162.98, 229.78, 171.82, 207.65, 195.05, 203.23,
        #  250.36, 220.94, 295.72, 232.0, 307.89, 232.0, 376.49, 223.14, 375.39, 225.35, 318.96, 
        # 194.37, 323.38, 182.21, 338.87, 171.15, 393.09, 191.07, 480.47, 162.3, 480.47], 
        # [226.46, 416.32, 220.94, 463.89, 230.89, 468.31, 229.78, 411.89]], 
        # 'area': 97486.80810000001, 
        # 'iscrowd': 0, 
        # 'image_id': 36, 
        # 'bbox': [0.0, 50.12, 457.68, 430.35], 'category_id': 28, 'id': 284996}, 
        # {....}...... ]

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")
        # target: BoxList(object): []

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)
        # target: BoxList(object).extra_fields = {"labels":classes}

        masks = [obj["segmentation"] for obj in anno]
        masks = SegmentationMask(masks, img.size)
        target.add_field("masks", masks)
        # target: BoxList(object).extra_fields = {"labels":classes, "masks": masks}

        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = PersonKeypoints(keypoints, img.size)
            target.add_field("keypoints", keypoints)

        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
