import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import transforms
import torch
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.distributed import get_world_size
import torch.distributed as dist
import torchvision.transforms as standard_transforms
import PIL.Image as Image

matplotlib.use('agg')


MAPS = ['map1','map2','map3','map4']
MAPS_dim = {'map1':64,'map2':256,'map3':512,'map4':1024}

Scales = [0.9, 1.1]
MIN_HW = 384
MAX_HW = 1584
IM_NORM_MEAN = [0.485, 0.456, 0.406]
IM_NORM_STD = [0.229, 0.224, 0.225]


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict

def select_exemplar_rois(image):
    all_rois = []

    print("Press 'q' or Esc to quit. Press 'n' and then use mouse drag to draw a new examplar, 'space' to save.")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord('n') or key == '\r':
            rect = cv2.selectROI("image", image, False, False)
            x1 = rect[0]
            y1 = rect[1]
            x2 = x1 + rect[2] - 1
            y2 = y1 + rect[3] - 1

            all_rois.append([y1, x1, y2, x2])
            for rect in all_rois:
                y1, x1, y2, x2 = rect
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            print("Press q or Esc to quit. Press 'n' and then use mouse drag to draw a new examplar")

    return all_rois

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def PerturbationLoss(output,boxes,sigma=8, use_gpu=True):
    Loss = 0.
    if boxes.shape[1] > 1:
        boxes = boxes.squeeze()
        for tempBoxes in boxes.squeeze():
            y1 = int(tempBoxes[1])
            y2 = int(tempBoxes[3])
            x1 = int(tempBoxes[2])
            x2 = int(tempBoxes[4])
            out = output[:,:,y1:y2,x1:x2]
            GaussKernel = matlab_style_gauss2D(shape=(out.shape[2],out.shape[3]),sigma=sigma)
            GaussKernel = torch.from_numpy(GaussKernel).float()
            if use_gpu: GaussKernel = GaussKernel.cuda()
            Loss += F.mse_loss(out.squeeze(),GaussKernel)
    else:
        boxes = boxes.squeeze()
        y1 = int(boxes[1])
        y2 = int(boxes[3])
        x1 = int(boxes[2])
        x2 = int(boxes[4])
        out = output[:,:,y1:y2,x1:x2]
        Gauss = matlab_style_gauss2D(shape=(out.shape[2],out.shape[3]),sigma=sigma)
        GaussKernel = torch.from_numpy(Gauss).float()
        if use_gpu: GaussKernel = GaussKernel.cuda()
        Loss += F.mse_loss(out.squeeze(),GaussKernel) 
    return Loss


def MincountLoss(output,boxes, use_gpu=True):
    ones = torch.ones(1)
    if use_gpu: ones = ones.cuda()
    Loss = 0.
    if boxes.shape[1] > 1:
        boxes = boxes.squeeze()
        for tempBoxes in boxes.squeeze():
            y1 = int(tempBoxes[1])
            y2 = int(tempBoxes[3])
            x1 = int(tempBoxes[2])
            x2 = int(tempBoxes[4])
            X = output[:,:,y1:y2,x1:x2].sum()
            if X.item() <= 1:
                Loss += F.mse_loss(X,ones)
    else:
        boxes = boxes.squeeze()
        y1 = int(boxes[1])
        y2 = int(boxes[3])
        x1 = int(boxes[2])
        x2 = int(boxes[4])
        X = output[:,:,y1:y2,x1:x2].sum()
        if X.item() <= 1:
            Loss += F.mse_loss(X,ones)  
    return Loss


def pad_to_size(feat, desire_h, desire_w):
    """ zero-padding a four dim feature matrix: N*C*H*W so that the new Height and Width are the desired ones
        desire_h and desire_w should be largers than the current height and weight
    """

    cur_h = feat.shape[-2]
    cur_w = feat.shape[-1]

    left_pad = (desire_w - cur_w + 1) // 2
    right_pad = (desire_w - cur_w) - left_pad
    top_pad = (desire_h - cur_h + 1) // 2
    bottom_pad =(desire_h - cur_h) - top_pad

    return F.pad(feat, (left_pad, right_pad, top_pad, bottom_pad))

@torch.no_grad()
def extract_features(feature_model, images, rects, feat_map_keys=['map3','map4'], exemplar_scales=[0.9, 1.1], Image_features=None, return_all_feat=False):
    """
    Getting features for the image N * C * H * W
    """
    N, C, H, W = images.shape
    if Image_features is None:
        Image_features = feature_model(images)
    examples_features = []
    h_scales = []
    w_scales = []
    """
    Getting features for the examples (N*M) * C * h * w
    """
    for ix in range(0,N):
        boxes = rects[ix]
        # cnter = 0
        # Cnter1 = 0
        for key in feat_map_keys:
            image_features = Image_features[key][ix].unsqueeze(0)
            if key == 'map1' or key == 'map2':
                Scaling = 4.0
            elif key == 'map3':
                Scaling = 8.0
            elif key == 'map4':
                Scaling =  16.0
            else:
                Scaling = 32.0
            boxes_scaled = boxes / Scaling
            boxes_scaled[:, 0:2] = torch.floor(boxes_scaled[:, 0:2])
            boxes_scaled[:, 2:4] = torch.ceil(boxes_scaled[:, 2:4])
            boxes_scaled[:, 2:4] = boxes_scaled[:, 2:4] + 1 # make the end indices exclusive 
            feat_h, feat_w = image_features.shape[-2], image_features.shape[-1]
            # make sure exemplars don't go out of bound
            boxes_scaled[:, 0:2] = torch.clamp_min(boxes_scaled[:, 0:2], 0)
            boxes_scaled[:, 2] = torch.clamp_max(boxes_scaled[:, 2], feat_h)
            boxes_scaled[:, 3] = torch.clamp_max(boxes_scaled[:, 3], feat_w)            

            tmp_feature = []    
            for j in range(0,len(boxes)):
                y1, x1 = int(boxes_scaled[j,0]), int(boxes_scaled[j,1])  
                y2, x2 = int(boxes_scaled[j,2]), int(boxes_scaled[j,3]) 
                examples_feature = image_features[:,:,y1:y2, x1:x2]
                # mean pooling
                examples_feature = F.adaptive_avg_pool2d(examples_feature, (1, 1))
                tmp_feature.append(examples_feature)

            h_scale = (boxes[:, 2] - boxes[:, 0])
            w_scale = (boxes[:, 3] - boxes[:, 1])
            h_scales.append(h_scale)
            w_scales.append(w_scale)
            
            
            tmp_feature = torch.cat(tmp_feature, dim=0)
            examples_features.append(tmp_feature)         
        h_scales = torch.cat(h_scales, dim=0)
        w_scales = torch.cat(w_scales, dim=0)
    return Image_features['map4'], examples_features, h_scales, w_scales


class resizeImage(object):
    """
    If either the width or height of an image exceed a specified value, resize the image so that:
        1. The maximum of the new height and new width does not exceed a specified value
        2. The new height and new width are divisible by 8
        3. The aspect ratio is preserved
    No resizing is done if both height and width are smaller than the specified value
    By: Minh Hoai Nguyen (minhhoai@gmail.com)
    """
    
    def __init__(self, MAX_HW=1504):
        self.max_hw = MAX_HW

    def __call__(self, sample):
        image,lines_boxes = sample['image'], sample['lines_boxes']
        
        W, H = image.size
        if W > self.max_hw or H > self.max_hw:
            scale_factor = float(self.max_hw)/ max(H, W)
            new_H = 8*int(H*scale_factor/8)
            new_W = 8*int(W*scale_factor/8)
            resized_image = transforms.Resize((new_H, new_W))(image)
        else:
            scale_factor = 1
            resized_image = image

        boxes = list()
        for box in lines_boxes:
            box2 = [int(k*scale_factor) for k in box]
            y1, x1, y2, x2 = box2[0], box2[1], box2[2], box2[3]
            boxes.append([0, y1,x1,y2,x2])

        boxes = torch.Tensor(boxes).unsqueeze(0)
        resized_image = Normalize(resized_image)
        sample = {'image':resized_image,'boxes':boxes}
        return sample


class resizeImageWithGT(object):
    """
    If either the width or height of an image exceed a specified value, resize the image so that:
        1. The maximum of the new height and new width does not exceed a specified value
        2. The new height and new width are divisible by 8
        3. The aspect ratio is preserved
    No resizing is done if both height and width are smaller than the specified value
    By: Minh Hoai Nguyen (minhhoai@gmail.com)
    Modified by: Viresh
    """
    
    def __init__(self, MAX_HW=1504):
        self.max_hw = MAX_HW

    def __call__(self, sample):
        image,lines_boxes,density = sample['image'], sample['lines_boxes'],sample['gt_density']
        
        W, H = image.size
        if W > self.max_hw or H > self.max_hw:
            scale_factor = float(self.max_hw)/ max(H, W)
            new_H = 8*int(H*scale_factor/8)
            new_W = 8*int(W*scale_factor/8)
            resized_image = transforms.Resize((new_H, new_W))(image)
            resized_density = cv2.resize(density, (new_W, new_H))
            orig_count = np.sum(density)
            new_count = np.sum(resized_density)

            if new_count > 0: resized_density = resized_density * (orig_count / new_count)
            
        else:
            scale_factor = 1
            resized_image = image
            resized_density = density
        boxes = list()
        for box in lines_boxes:
            box2 = [int(k*scale_factor) for k in box]
            y1, x1, y2, x2 = box2[0], box2[1], box2[2], box2[3]
            boxes.append([0, y1,x1,y2,x2])

        boxes = torch.Tensor(boxes).unsqueeze(0)
        resized_image = Normalize(resized_image)
        resized_density = torch.from_numpy(resized_density).unsqueeze(0).unsqueeze(0)
        sample = {'image':resized_image,'boxes':boxes,'gt_density':resized_density}
        return sample


Normalize = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize(mean=IM_NORM_MEAN, std=IM_NORM_STD)])
Transform = transforms.Compose([resizeImage( MAX_HW)])
TransformTrain = transforms.Compose([resizeImageWithGT(MAX_HW)])


def denormalize(tensor, means=IM_NORM_MEAN, stds=IM_NORM_STD):
    """Reverses the normalisation on a tensor.
    Performs a reverse operation on a tensor, so the pixel value range is
    between 0 and 1. Useful for when plotting a tensor into an image.
    Normalisation: (image - mean) / std
    Denormalisation: image * std + mean
    Args:
        tensor (torch.Tensor, dtype=torch.float32): Normalized image tensor
    Shape:
        Input: :math:`(N, C, H, W)`
        Output: :math:`(N, C, H, W)` (same shape as input)
    Return:
        torch.Tensor (torch.float32): Demornalised image tensor with pixel
            values between [0, 1]
    Note:
        Symbols used to describe dimensions:
            - N: number of images in a batch
            - C: number of channels
            - H: height of the image
            - W: width of the image
    """

    denormalized = tensor.clone()

    for channel, mean, std in zip(denormalized, means, stds):
        channel.mul_(std).add_(mean)

    return denormalized


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def scale_and_clip(val, scale_factor, min_val, max_val):
    "Helper function to scale a value and clip it within range"

    new_val = int(round(val*scale_factor))
    new_val = max(new_val, min_val)
    new_val = min(new_val, max_val)
    return new_val

def getImg(img_raw, save_path, points, color=(0, 0, 255), text=None):
    size = 2
    img_to_draw = img_raw.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()
    for p in points:
        img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, color, -1)
    if text is not None:
        img_to_draw = cv2.putText(img_to_draw, text+f":{len(points)}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
    # save the visualized image
    cv2.imwrite(save_path, img_to_draw)

def vis(input_, pred_dots, boxes, save_path, pred_sizes=None, pred_score=None, figsize=(20, 12), ds=2, dots=None, cam_img=None):
    # get the total count
    pred_cnt = pred_dots.shape[0]
    if len(boxes.shape) == 3:
        boxes = boxes.squeeze(0)

    boxes2 = []
    for i in range(0, boxes.shape[0]):
        y1, x1, y2, x2 = int(boxes[i, 0].item()), int(boxes[i, 1].item()), int(boxes[i, 2].item()), int(
            boxes[i, 3].item())
        # roi_cnt = output[0,0,y1:y2, x1:x2].sum().item()
        boxes2.append([y1, x1, y2, x2])
    
    restore_transform = standard_transforms.Compose([
        DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        standard_transforms.ToPILImage()
    ])
    img1 = restore_transform(input_[0])

    fig = plt.figure(figsize=figsize)
    
    rc = 1
    if pred_sizes is not None:
        rc = 2
    
    # display the input image
    ax = fig.add_subplot(rc, 3, 1)
    ax.set_axis_off()
    ax.imshow(img1)

    for bbox in boxes2:
        y1, x1, y2, x2 = bbox[0], bbox[1], bbox[2], bbox[3]
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=3, edgecolor='y', facecolor='none')
        rect2 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='k', linestyle='--', facecolor='none')
        ax.add_patch(rect)
        ax.add_patch(rect2)

    if dots is not None:
        ax.scatter(dots[:, 0], dots[:, 1], c='red', edgecolors='blue')
        # ax.scatter(dots[:,0], dots[:,1], c='black', marker='+')
        ax.set_title("Input image, gt count: {}".format(dots.shape[0]))
    else:
        ax.set_title("Input image")

    ax = fig.add_subplot(rc, 3, 2)
    ax.set_axis_off()
    ax.set_title("Overlaid result, predicted count: {:.2f}, error count: {:.2f}".format(pred_cnt, abs(pred_cnt - dots.shape[0])))

    ax.imshow(img1)
    if len(pred_dots)>0:
        ax.scatter(pred_dots[:, 0], pred_dots[:, 1], c='red', edgecolors='blue')
    
    if pred_sizes is not None:
        ax = fig.add_subplot(rc, 3, 3)
        ax.set_axis_off()
        ax.set_title("Predicted sizes")
        ax.imshow(img1)
        assert pred_sizes.shape[0] == pred_dots.shape[0]
        for center, size, score in zip(pred_dots, pred_sizes, pred_score):
            x1, y1 = int(center[0] - size[1] / 2), int(center[1] - size[0] / 2)
            rect = patches.Rectangle((x1, y1), size[1], size[0], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            
        # just draw rect with top 10 score ones
        if pred_score is not None:
            ax = fig.add_subplot(rc, 3, 4)
            ax.set_axis_off()
            ax.set_title("Predicted sizes with top 10 score ones")
            ax.imshow(img1)
            assert pred_score.shape[0] == pred_dots.shape[0]
            top_index = np.argsort(pred_score)[::-1][:10]
            dots = pred_dots[top_index]
            sizes = pred_sizes[top_index]
            scores = pred_score[top_index]
            for center, size, score in zip(dots, sizes, scores):
                x1, y1 = int(center[0] - size[1] / 2), int(center[1] - size[0] / 2)
                rect = patches.Rectangle((x1, y1), size[1], size[0], linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                # 在矩形框左上角打印文本
                ax.text(x1, y1+10, '{:.2f}'.format(score), fontsize=10, color='r')

    if cam_img is not None:
        ax = fig.add_subplot(rc, 3, 5)
        ax.set_axis_off()
        ax.set_title("Cam image")
        cam_img = cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB)
        ax.imshow(cam_img)

    fig.savefig(save_path, bbox_inches="tight")
    plt.close()
