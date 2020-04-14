from rtv.network import RasterToVector
from rtv.ip import *
import torch
from skimage import io, transform
from os import listdir
from os.path import isfile, join
from datetime import datetime
import cv2 as cv2
device = torch.device("cuda:0" if use_cuda else "cpu")
RTV = RasterToVector()
RTV.load_state_dict(
    torch.load('rtv/checkpoints/rtv.pth', map_location=device)

folder_inputs='rtv_inputs/'
folder_outputs='rtv_outputs/{}/'.format(
    datetime.now().strftime('%m-%d_%H-%M-%S'))
os.makedirs(folder_outputs)
paths=[f for f in listdir(folder_inputs) if isfile(join(folder_inputs, f))]


def load_img(path_sample):
    img=io.imread(path_sample)
    if img.shape[2] == 4:
        img[np.where(img[:, :, 3] == 0)]=255
    img=transform.resize(img, (256, 256))
    img=img[:, :, :3].astype('float32')

    # image_bis = cv2.Canny(img, 200, 300)
    # image_bis = np.expand_dims(image_bis, axis=2)
    # img = np.concatenate((image_bis,image_bis,image_bis), axis=2)

    # minval = np.percentile(img, 2)
    # maxval = np.percentile(img, 98)
    # img = np.clip(img, minval, maxval)
    # img = ((img - minval) / (maxval - minval))
    # th = 0.3
    # img[np.where(img > th)] = 1.
    # img[np.where(img < th)] = 0.

    image=torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img, image


def apply_rtv(img, image, output_prefix, gap=-1,
              distanceThreshold=-1,
              lengthThreshold=-1,
              heatmapValueThresholdWall=None,
              heatmapValueThresholdDoor=None,
              heatmapValueThresholdIcon=None):
    output_prefix += '_'
    corner_pred, icon_pred, room_pred=RTV(image)
    corner_pred, icon_pred, room_pred=corner_pred.squeeze(
        0), icon_pred.squeeze(0), room_pred.squeeze(0)
    corner_heatmaps=corner_pred.detach().cpu().numpy()
    icon_heatmaps=torch.nn.functional.softmax(icon_pred,
                                                dim=-1).detach().cpu().numpy()
    room_heatmaps=torch.nn.functional.softmax(room_pred,
                                                dim=-1).detach().cpu().numpy()

    reconstructFloorplan(
        corner_heatmaps[:, :, :NUM_WALL_CORNERS],
        corner_heatmaps[:, :, NUM_WALL_CORNERS:NUM_WALL_CORNERS + 4],
        corner_heatmaps[:, :, -4:],
        icon_heatmaps,
        room_heatmaps,
        output_prefix=output_prefix,
        densityImage=None,
        gt_dict=None,
        gt=False,
        gap=gap,
        distanceThreshold=distanceThreshold,
        lengthThreshold=lengthThreshold,
        debug_prefix='test',
        heatmapValueThresholdWall=heatmapValueThresholdWall,
        heatmapValueThresholdDoor=heatmapValueThresholdDoor,  # same threshold
        heatmapValueThresholdIcon=heatmapValueThresholdIcon,  # same threshold
        enableAugmentation=True)
    dicts={
        'corner': corner_pred.max(-1)[1].detach().cpu().numpy(),
        'icon': icon_pred.max(-1)[1].detach().cpu().numpy(),
        'room': room_pred.max(-1)[1].detach().cpu().numpy()
    }
    cv2.imwrite(output_prefix + 'image.png', img * 255)
    for info in ['corner', 'icon', 'room']:
        cv2.imwrite(
            output_prefix + info + '.png',
            drawSegmentationImage(dicts[info], blackIndex=0,
                                  blackThreshold=0.5))


gaps=[5]  # range(1, 8, 1)
distances=[5]  # range(3, 9)
lengths=[6]  # range(3, 9)
heatmaps_wall=[0.3, 0.4, 0.5, 0.6, 0.7]  # [x * 0.1 for x in range(2, 9, 1)]
heatmaps_door=[0.3, 0.5, 0.7]
heatmaps_icon=[0.3, 0.5, 0.7]
# generalize to all good parameters and several images
for path_sample in paths:

    img, image=load_img(folder_inputs + path_sample)

    for gap in gaps:
        for distanceThreshold in distances:
            for lengthThreshold in lengths:
                for heatmapValueThresholdWall in heatmaps_wall:
                    for heatmapValueThresholdDoor in heatmaps_door:
                        for heatmapValueThresholdIcon in heatmaps_icon:
                            output_prefix=folder_outputs + path_sample[:-4] + \
                                '_gap_{}_dist_{}_length_{}_wall_{}_door_{}_icon_{}'.format(
                                    gap, distanceThreshold, lengthThreshold,
                                    heatmapValueThresholdWall, heatmapValueThresholdDoor, heatmapValueThresholdIcon)
                            print(output_prefix)
                            apply_rtv(img, image, output_prefix, gap=gap,
                                      distanceThreshold=distanceThreshold,
                                      lengthThreshold=lengthThreshold,
                                      heatmapValueThresholdWall=heatmapValueThresholdWall,
                                      heatmapValueThresholdDoor=heatmapValueThresholdDoor,  # same threshold
                                      heatmapValueThresholdIcon=heatmapValueThresholdIcon)

    files=os.listdir(folder_outputs)
    images=np.zeros((256, 256, 3))
    for file in files:
        if file.endswith("result_line.png") and path_sample[:-4] in file:
            images += cv2.imread(os.path.join(folder_outputs, file), 1)

    cv2.imwrite(
        folder_outputs + path_sample[:-4] + '_sum' + '.png', images)
