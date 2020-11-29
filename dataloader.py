import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
from torchvision import ops
from PIL import Image, ImageDraw

IMAGES_FOLDER = "Image"
MARKUP_FOLDER = "Markup"

IMAGES_EXT = ['.jpg', '.jpeg', '.png']
MARKUP_EXT = '.csv'

ANCHOR_SIZE = 16
ANCHOR_SCALES = [1, 2, 4]
ANCHOR_RATIOS = [0.5, 1., 2.]
IOU_THRESHOLD_POSITIVE = 0.7
IOU_THRESHOLD_NEGATIVE = 0.3

def is_image(image_path):
    return os.path.splitext(image_path)[-1].lower() in IMAGES_EXT


class TargetMapCreator:
    def __init__(self):
        return

    @staticmethod
    def generate_anchors(anchor_size=ANCHOR_SIZE):
        base_anchor = torch.tensor([0, 0, anchor_size, anchor_size])
        base_anchor = ops.box_convert(base_anchor, 'xywh', 'cxcywh')
        size = torch.prod(base_anchor[2:])
        widths = torch.round(torch.sqrt(size / torch.tensor(ANCHOR_RATIOS)))
        heights = widths * torch.tensor(ANCHOR_RATIOS)
        anchors = torch.zeros(len(widths), 4)
        anchors[:, 0:2] = base_anchor[0:2]
        anchors[:, 2] = widths
        anchors[:, 3] = heights
        scaled_anchors = torch.vstack([anchors for _ in ANCHOR_SCALES])

        scaled_anchors[:, 2:] *= torch.repeat_interleave(torch.tensor(ANCHOR_SCALES), len(ANCHOR_SCALES)).view(-1, 1)
        return ops.box_convert(scaled_anchors, 'cxcywh', 'xywh')

    @staticmethod
    def generate_target_anchors(w_out, h_out, scale):
        base_anchors = TargetMapCreator.generate_anchors()
        anchors = base_anchors.repeat(w_out * h_out, 1)
        out_xy_coords = torch.vstack((torch.arange(w_out).repeat(h_out), torch.repeat_interleave(torch.arange(h_out), w_out)))
        out_xy_coords = torch.repeat_interleave(out_xy_coords, len(base_anchors), 1)
        shifts = (out_xy_coords * scale).T

        anchors[:, :2] += shifts
        return anchors, out_xy_coords[0], out_xy_coords[1]


    @staticmethod
    def generate_target_for_classification(w_out, h_out, scale, gt_bboxes):
        anchors, out_x_coords, out_y_coords = TargetMapCreator.generate_target_anchors(w_out, h_out, scale)

        gt_bboxes = ops.box_convert(gt_bboxes, 'xywh', 'xyxy')
        anchors = ops.box_convert(anchors, 'xywh', 'xyxy')
        ious = ops.box_iou(anchors, gt_bboxes)
        max_iou_overall_gt, _ = torch.max(ious, dim=1)
        positive_anchors_inds = torch.nonzero(max_iou_overall_gt > IOU_THRESHOLD_POSITIVE).view(-1)
        max_iou_overall_anhors, max_anchors_inds = torch.max(ious, dim=0)
        additional_pos_anchors = torch.nonzero(max_iou_overall_anhors < IOU_THRESHOLD_POSITIVE).view(-1)
        additional_pos_anchors = max_anchors_inds[additional_pos_anchors]

        positive_anchors_inds = torch.cat((positive_anchors_inds, additional_pos_anchors), dim=0)
        positive_anchors = anchors[positive_anchors_inds]
        positive_anchors = ops.box_convert(positive_anchors, 'xyxy', 'xywh')

        positive_anchors_out_x = out_x_coords[positive_anchors_inds]
        positive_anchors_out_y = out_y_coords[positive_anchors_inds]
        base_anchor_count = len(ANCHOR_SCALES) * len(ANCHOR_RATIOS)
        positive_anchors_out_c = positive_anchors_inds % base_anchor_count

        target = torch.zeros((base_anchor_count, h_out, w_out))
        target[positive_anchors_out_c, positive_anchors_out_y, positive_anchors_out_x] = 1.
        mask = target.detach().clone().bool()

        negative_anchors_inds = torch.nonzero(max_iou_overall_gt < IOU_THRESHOLD_NEGATIVE).view(-1)

        negative_anchors = anchors[negative_anchors_inds]
        negative_anchors = ops.box_convert(negative_anchors, 'xyxy', 'xywh')

        negative_anchors_out_x = out_x_coords[negative_anchors_inds]
        negative_anchors_out_y = out_y_coords[negative_anchors_inds]
        negative_anchors_out_c = negative_anchors_inds % base_anchor_count

        mask[negative_anchors_out_c, negative_anchors_out_y, negative_anchors_out_x] = 1

        return target, mask



class ObjectDetectionDataset(Dataset):
    def __init__(self, root):
        images_names = os.listdir(os.path.join(root, IMAGES_FOLDER))
        images_paths = [os.path.join(root, IMAGES_FOLDER, image_name)
                               for image_name in filter(is_image, images_names)]
        images_names_2_markup_names = dict()

        for image_path in images_paths:
            image_name, image_ext = os.path.splitext(os.path.basename(image_path))
            markup_nameext = image_name + MARKUP_EXT
            markup_path = os.path.join(root, MARKUP_FOLDER, markup_nameext)
            if os.path.exists(markup_path):
                images_names_2_markup_names[image_path] = markup_path

        self.__image_paths = sorted([key for key, item in images_names_2_markup_names.items()])
        self.__markup_paths = [images_names_2_markup_names[image_path] for image_path in self.__image_paths]
        assert len(self.__image_paths) == len(self.__markup_paths)

    def __len__(self):
        return len(self.__image_paths)

    def __getitem__(self, index):
        image = Image.open(self.__image_paths[index])
        df = pd.read_csv(self.__markup_paths[index])
        bboxes = ops.box_convert(df.values, 'xyxy', 'xywh').view(-1, 4)
        return image, bboxes


if __name__ == '__main__':
    target, mask = TargetMapCreator.generate_target_for_classification(4, 4, 2, torch.tensor([[3, 3, 3, 3]]))

    img_size = 1000
    image = Image.new("L", (img_size, img_size), color=255)
    imageDraw = ImageDraw.Draw(image)
    anchors, _, _ = TargetMapCreator.generate_target_anchors(img_size // 10, img_size // 10, 10)

    anchors = ops.box_convert(anchors, 'xywh', 'xyxy')

    step = (img_size // 10 * (img_size // 10 + 1)) // 2 * 9
    for i in range(step, step + 9):
        anchor = anchors[i].numpy()
        print(anchor)
        imageDraw.rectangle(anchor, outline=0)
    image.save('image.png')