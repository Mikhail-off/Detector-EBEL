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

ANCHOR_SIZE = 32
ANCHOR_SCALES = [1, 2, 4]
ANCHOR_RATIOS = [0.5, 1., 2.]

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
    def generate_target_anchors(w, h, scale):
        base_anchors = TargetMapCreator.generate_anchors()
        anhors = base_anchors.repeat(w * h, 1)
        shifts = torch.vstack((torch.arange(w).repeat(h), torch.repeat_interleave(torch.arange(h), w))) * scale
        shifts = torch.repeat_interleave(shifts, len(base_anchors), 1).T

        anhors[:, :2] += shifts
        return anhors

    @staticmethod
    def generate_target_for_classification(w, h, scale):
        pass

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
        bboxes = ops.box_convert(df.values, 'xyxy', 'xywh')
        return image, bboxes


if __name__ == '__main__':
    dataset = ObjectDetectionDataset("C:\\Codes\\Detector-EBEL\\raw_data")
    rects = TargetMapCreator.generate_target_anchors(1000//10, 1000//10, 10)
    rects = ops.box_convert(rects, 'xywh', 'xyxy')
    #TargetMapCreator.generate_target_anchors(2, 2, 4)
    image = Image.new("L", size=(1000, 1000), color=255)
    imageDraw = ImageDraw.Draw(image)
    n = 100**2 // 2 + 50
    for rect in rects[9 * n:9 * n + 9]:
        imageDraw.rectangle(rect.numpy(), outline=0, width=2)
    image.save("image.png")