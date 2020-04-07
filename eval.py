from rtv.network import RasterToVector
from rtv.ip import *
import torch
from skimage import io, transform
RTV = RasterToVector()
RTV.load_state_dict(
    torch.load('rtv/checkpoints/rtv.pth', map_location=torch.device('cpu')))
image = torch.randn((1, 3, 64, 64))

img = io.imread('lala.png')
if img.shape[2] == 4:
    img[np.where(img[:, :, 3] == 0)] = 255
img = transform.resize(img, (256, 256))
img = img[:, :, :3].astype('float32')
image = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
#image = torch.stack(image)
print('immmaaage', image.shape)
corner_pred, icon_pred, room_pred = RTV(image)
corner_pred, icon_pred, room_pred = corner_pred.squeeze(0), icon_pred.squeeze(
    0), room_pred.squeeze(0)
corner_heatmaps = corner_pred.detach().cpu().numpy()
icon_heatmaps = torch.nn.functional.softmax(icon_pred,
                                            dim=-1).detach().cpu().numpy()
room_heatmaps = torch.nn.functional.softmax(room_pred,
                                            dim=-1).detach().cpu().numpy()
print(corner_heatmaps.shape, icon_heatmaps.shape, room_heatmaps.shape)
reconstructFloorplan(corner_heatmaps[:, :, :NUM_WALL_CORNERS],
                     corner_heatmaps[:, :, NUM_WALL_CORNERS:NUM_WALL_CORNERS +
                                     4],
                     corner_heatmaps[:, :, -4:],
                     icon_heatmaps,
                     room_heatmaps,
                     output_prefix='temp/blaa_',
                     densityImage=None,
                     gt_dict=None,
                     gt=False,
                     gap=-1,
                     distanceThreshold=-1,
                     lengthThreshold=-1,
                     debug_prefix='test',
                     heatmapValueThresholdWall=None,
                     heatmapValueThresholdDoor=None,
                     heatmapValueThresholdIcon=None,
                     enableAugmentation=True)
dicts = {
    'corner': corner_pred.max(-1)[1].detach().cpu().numpy(),
    'icon': icon_pred.max(-1)[1].detach().cpu().numpy(),
    'room': room_pred.max(-1)[1].detach().cpu().numpy()
}

for info in ['corner', 'icon', 'room']:
    cv2.imwrite(
        'temp/bla_' + info + '.png',
        drawSegmentationImage(dicts[info], blackIndex=0, blackThreshold=0.5))
