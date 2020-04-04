from rtv.network import RasterToVector
import torch
RTV = RasterToVector()
image = torch.randn((1, 3, 64, 64))
corner_pred, icon_pred, room_pred = RTV(image)
for output in outputs:
    print(output.shape)

# visualizeBatch(
#     options,
#     images.detach().cpu().numpy(), [('pred', {
#         'corner': corner_pred.max(-1)[1].detach().cpu().numpy(),
#         'icon': icon_pred.max(-1)[1].detach().cpu().numpy(),
#         'room': room_pred.max(-1)[1].detach().cpu().numpy()
#     })])
for batchIndex in range(len(images)):
    corner_heatmaps = corner_pred[batchIndex].detach().cpu().numpy()
    icon_heatmaps = torch.nn.functional.softmax(icon_pred[batchIndex],
                                                dim=-1).detach().cpu().numpy()
    room_heatmaps = torch.nn.functional.softmax(room_pred[batchIndex],
                                                dim=-1).detach().cpu().numpy()
    reconstructFloorplan(
        corner_heatmaps[:, :, :NUM_WALL_CORNERS],
        corner_heatmaps[:, :, NUM_WALL_CORNERS:NUM_WALL_CORNERS + 4],
        corner_heatmaps[:, :, -4:],
        icon_heatmaps,
        room_heatmaps,
        output_prefix=test_dir + '/' + str(batchIndex) + '_',
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


def visualizeImage(path, images, dicts, indexOffset=0, prefix=''):
    #cornerColorMap = {'gt': np.array([255, 0, 0]), 'pred': np.array([0, 0, 255]), 'inp': np.array([0, 255, 0])}
    #pointColorMap = ColorPalette(20).getColorMap()
    image = ((image.transpose((2, 3, 1)) + 0.5) * 255).astype(np.uint8)
    image = images.copy()
    filename = path + '_image.png'
    cv2.imwrite(filename, image)
    for name, result_dict in dicts:
        for info in ['corner', 'icon', 'room']:
            cv2.imwrite(
                filename.replace('image', info + '_' + name),
                drawSegmentationImage(result_dict[info][batchIndex],
                                      blackIndex=0,
                                      blackThreshold=0.5))
