#from gurobipy import *
from pulp import *
import cv2
import numpy as np
import sys
import csv
import copy
from IP_utils import *
from skimage import measure



## Reconstruct a floorplan via IP optimization
def reconstructFloorplan(wallCornerHeatmaps,
                         doorCornerHeatmaps,
                         roomHeatmaps,
                         iconCornerHeatmaps=None,
                         iconHeatmaps=None,
                         output_prefix='test/',
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
                         enableAugmentation=False,
                         print_info=False,
                         save_image=False):

  baseH,baseW = doorCornerHeatmaps.shape[:2]
  if roomHeatmaps is None:
      roomHeatmaps = np.zeros([baseH,baseW,10])
  if iconCornerHeatmaps is None:
      iconCornerHeatmaps = np.zeros([baseH,baseW,4])
  if iconHeatmaps is None:
      iconHeatmaps = np.zeros([baseH,baseW,9])


  wallPoints = []
  iconPoints = []
  doorPoints = []

  numWallPoints = 100
  numDoorPoints = 100
  numIconPoints = 100
  if heatmapValueThresholdWall is None:
    heatmapValueThresholdWall = 0.5
    pass
  heatmapValueThresholdDoor = 0.5
  heatmapValueThresholdIcon = 0.5

  if gap > 0:
    for k in GAPS:
      GAPS[k] = gap
      continue
    pass
  if distanceThreshold > 0:
    for k in DISTANCES:
      DISTANCES[k] = distanceThreshold
      continue
    pass
  if lengthThreshold > 0:
    for k in LENGTH_THRESHOLDS:
      LENGTH_THRESHOLDS[k] = lengthThreshold
      continue
    pass

  wallPoints, wallLines, wallPointOrientationLinesMap, wallPointNeighbors, augmentedPointMask = extractCorners(wallCornerHeatmaps, heatmapValueThresholdWall, gap=GAPS['wall_extraction'], augment=enableAugmentation, gt=gt)
  doorPoints, doorLines, doorPointOrientationLinesMap, doorPointNeighbors, _ = extractCorners(doorCornerHeatmaps, heatmapValueThresholdDoor, gap=GAPS['door_extraction'], cornerType='door', gt=gt)
  iconPoints, iconLines, iconPointOrientationLinesMap, iconPointNeighbors, _ = extractCorners(iconCornerHeatmaps, heatmapValueThresholdIcon, gap=GAPS['icon_extraction'], cornerType='icon', gt=gt)

  if not gt:
    for pointIndex, point in enumerate(wallPoints):
      #print((pointIndex, np.array(point[:2]).astype(np.int32).tolist(), point[2], point[3]))
      continue

    wallPoints, wallLines, wallPointOrientationLinesMap, wallPointNeighbors = filterWalls(wallPoints, wallLines)
    pass


  sortLines(doorPoints, doorLines)
  sortLines(wallPoints, wallLines)
  if print_info:
      print('the number of points', len(wallPoints), len(doorPoints), len(iconPoints))
      print('the number of lines', len(wallLines), len(doorLines), len(iconLines))


  drawPoints(os.path.join(debug_prefix, "points.png"), width, height, wallPoints, densityImage, pointSize=3)
  drawPointsSeparately(os.path.join(debug_prefix, 'points'), width, height, wallPoints, densityImage, pointSize=3)
  drawLines(os.path.join(debug_prefix, 'lines.png'), width, height, wallPoints, wallLines, [], None, 1, lineColor=255)

  wallMask = drawLineMask(width, height, wallPoints, wallLines)

  labelVotesMap = np.zeros((NUM_ROOMS, height, width))
  #labelMap = np.zeros((NUM_LABELS, height, width))
  #semanticHeatmaps = np.concatenate([iconHeatmaps, roomHeatmaps], axis=2)
  for segmentIndex in range(NUM_ROOMS):
    segmentation_img = roomHeatmaps[:, :, segmentIndex]
    #segmentation_img = (segmentation_img > 0.5).astype(np.float)
    labelVotesMap[segmentIndex] = segmentation_img
    #labelMap[segmentIndex] = segmentation_img
    continue

  labelVotesMap = np.cumsum(np.cumsum(labelVotesMap, axis=1), axis=2)

  icons = findIconsFromLines(iconPoints, iconLines)

  if not gt:
    newIcons, newIconPoints, newDoorLines, newDoorPoints = findCandidatesFromHeatmaps(iconHeatmaps, len(iconPoints), len(doorPoints))

    icons += newIcons
    iconPoints += newIconPoints
    doorLines += newDoorLines
    doorPoints += newDoorPoints
    pass

  if True:
    drawLines(os.path.join(debug_prefix, 'lines.png'), width, height, wallPoints, wallLines, [], None, 2, lineColor=255)
    drawLines(os.path.join(debug_prefix, 'doors.png'), width, height, doorPoints, doorLines, [], None, 2, lineColor=255)
    drawRectangles(os.path.join(debug_prefix, 'icons.png'), width, height, iconPoints, icons, {}, 2)
    if print_info:
        print('number of walls: ' + str(len(wallLines)))
        print('number of doors: ' + str(len(doorLines)))
        print('number of icons: ' + str(len(icons)))
    pass


  doorWallLineMap = findLineMap(doorPoints, doorLines, wallPoints, wallLines, gap=GAPS['wall_door_neighbor'])

  newDoorLines = []
  newDoorWallLineMap = []
  for lineIndex, walls in enumerate(doorWallLineMap):
    if len(walls) > 0:
      newDoorLines.append(doorLines[lineIndex])
      newDoorWallLineMap.append(walls)
      pass
    continue
  doorLines = newDoorLines
  doorWallLineMap = newDoorWallLineMap


  conflictWallLinePairs = findConflictLinePairs(wallPoints, wallLines, gap=GAPS['wall_conflict'], distanceThreshold=DISTANCES['wall'], considerEndPoints=True)

  conflictDoorLinePairs = findConflictLinePairs(doorPoints, doorLines, gap=GAPS['door_conflict'], distanceThreshold=DISTANCES['door'])
  conflictIconPairs = findConflictRectanglePairs(iconPoints, icons, gap=GAPS['icon_conflict'])



  wallLineNeighbors = findLineNeighbors(wallPoints, wallLines, labelVotesMap, gap=GAPS['wall_neighbor'])

  iconWallLineNeighbors = findRectangleLineNeighbors(iconPoints, icons, wallPoints, wallLines, wallLineNeighbors, gap=GAPS['wall_icon_neighbor'], distanceThreshold=DISTANCES['wall_icon'])
  conflictIconWallPairs = findConflictRectangleLinePairs(iconPoints, icons, wallPoints, wallLines, gap=GAPS['wall_icon_conflict'])



  exteriorLines = {}
  for lineIndex, neighbors in enumerate(wallLineNeighbors):
    if len(neighbors[0]) == 0 and len(neighbors[1]) > 0:
      exteriorLines[lineIndex] = 0
    elif len(neighbors[0]) > 0 and len(neighbors[1]) == 0:
      exteriorLines[lineIndex] = 1
      pass
    continue

  if False:
    filteredWallLines = []
    for lineIndex, neighbors in enumerate(wallLineNeighbors):
      if len(neighbors[0]) == 0 and len(neighbors[1]) > 0:
        print(lineIndex)
        filteredWallLines.append(wallLines[lineIndex])
        pass
      continue
    drawLines(os.path.join(debug_prefix, 'exterior_1.png'), width, height, wallPoints, filteredWallLines, lineColor=255)

    filteredWallLines = []
    for lineIndex, neighbors in enumerate(wallLineNeighbors):
      if len(neighbors[0]) > 0 and len(neighbors[1]) == 0:
        print(lineIndex)
        filteredWallLines.append(wallLines[lineIndex])
        pass
      continue
    drawLines(os.path.join(debug_prefix, 'exterior_2.png'), width, height, wallPoints, filteredWallLines, lineColor=255)
    exit(1)
    pass

  if True:
    #model = Model("JunctionFilter")
    model = LpProblem("JunctionFilter", LpMinimize)

    #add variables
    w_p = [LpVariable(cat=LpBinary, name="point_" + str(pointIndex)) for pointIndex in range(len(wallPoints))]
    w_l = [LpVariable(cat=LpBinary, name="line_" + str(lineIndex)) for lineIndex in range(len(wallLines))]

    d_l = [LpVariable(cat=LpBinary, name="door_line_" + str(lineIndex)) for lineIndex in range(len(doorLines))]

    i_r = [LpVariable(cat=LpBinary, name="icon_rectangle_" + str(lineIndex)) for lineIndex in range(len(icons))]

    i_types = []
    for iconIndex in range(len(icons)):
      i_types.append([LpVariable(cat=LpBinary, name="icon_type_" + str(iconIndex) + "_" + str(typeIndex)) for typeIndex in range(NUM_ICONS)])
      continue

    l_dir_labels = []
    for lineIndex in range(len(wallLines)):
      dir_labels = []
      for direction in range(2):
        labels = []
        for label in range(NUM_ROOMS):
          labels.append(LpVariable(cat=LpBinary, name="line_" + str(lineIndex) + "_" + str(direction) + "_" + str(label)))
        dir_labels.append(labels)
      l_dir_labels.append(dir_labels)



    #model.update()
    #obj = QuadExpr()
    obj = LpAffineExpression()

    if gt:
      for pointIndex in range(len(wallPoints)):
        model += (w_p[pointIndex] == 1, 'gt_point_active_' + str(pointIndex))
        continue

      pointIconMap = {}
      for iconIndex, icon in enumerate(icons):
        for pointIndex in icon:
          if pointIndex not in pointIconMap:
            pointIconMap[pointIndex] = []
            pass
          pointIconMap[pointIndex].append(iconIndex)
          continue
        continue
      for pointIndex, iconIndices in pointIconMap.items():
        break
        iconSum = LpAffineExpression()
        for iconIndex in iconIndices:
          iconSum += i_r[iconIndex]
          continue
        model += (iconSum == 1)
        continue
      pass

    ## Semantic label one hot constraints
    for lineIndex in range(len(wallLines)):
      for direction in range(2):
        labelSum = LpAffineExpression()
        for label in range(NUM_ROOMS):
          labelSum += l_dir_labels[lineIndex][direction][label]
          continue
        model += (labelSum == w_l[lineIndex], 'label_sum_' + str(lineIndex) + '_' + str(direction))
        continue
      continue

    ## Opposite room constraints
    if False:
      oppositeRoomPairs = [(1, 1), (2, 2), (4, 4), (5, 5), (7, 7), (9, 9)]
      for lineIndex in range(len(wallLines)):
        for oppositeRoomPair in oppositeRoomPairs:
          model += (l_dir_labels[lineIndex][0][oppositeRoomPair[0]] + l_dir_labels[lineIndex][0][oppositeRoomPair[1]] <= 1)
          if oppositeRoomPair[0] != oppositeRoomPair[1]:
            model += (l_dir_labels[lineIndex][0][oppositeRoomPair[1]] + l_dir_labels[lineIndex][0][oppositeRoomPair[0]] <= 1)
            pass
          continue
        continue
      pass

    ## Loop constraints
    closeRooms = {}
    for label in range(NUM_ROOMS):
      closeRooms[label] = True
      continue
    closeRooms[1] = False
    closeRooms[2] = False
    #closeRooms[3] = False
    closeRooms[8] = False
    closeRooms[9] = False

    for label in range(NUM_ROOMS):
      if not closeRooms[label]:
        continue
      for pointIndex, orientationLinesMap in enumerate(wallPointOrientationLinesMap):
        for orientation, lines in orientationLinesMap.items():
          direction = int(orientation in [1, 2])
          lineSum = LpAffineExpression()
          for lineIndex in lines:
            lineSum += l_dir_labels[lineIndex][direction][label]
            continue
          for nextOrientation in range(orientation + 1, 8):
            if not (nextOrientation % 4) in orientationLinesMap:
              continue
            nextLines = orientationLinesMap[nextOrientation % 4]
            nextDirection = int((nextOrientation % 4) in [0, 3])
            nextLineSum = LpAffineExpression()
            for nextLineIndex in nextLines:
              nextLineSum += l_dir_labels[nextLineIndex][nextDirection][label]
              continue
            model += (lineSum == nextLineSum)
            break
          continue
        continue
      continue


    ## Exterior constraints
    exteriorLineSum = LpAffineExpression()
    for lineIndex in range(len(wallLines)):
      if lineIndex not in exteriorLines:
        continue
      #direction = exteriorLines[lineIndex]
      label = 0
      model += (l_dir_labels[lineIndex][0][label] + l_dir_labels[lineIndex][1][label] == w_l[lineIndex], 'exterior_wall_' + str(lineIndex))
      exteriorLineSum += w_l[lineIndex]
      continue
    model += (exteriorLineSum >= 1, 'exterior_wall_sum')


    ## Wall line room semantic objectives
    for lineIndex, directionNeighbors in enumerate(wallLineNeighbors):
      for direction, neighbors in enumerate(directionNeighbors):
        labelVotesSum = np.zeros(NUM_ROOMS)
        for neighbor, labelVotes in neighbors.items():
          labelVotesSum += labelVotes
          continue

        votesSum = labelVotesSum.sum()
        if votesSum == 0:
          continue
        labelVotesSum /= votesSum

        for label in range(NUM_ROOMS):
          obj += (l_dir_labels[lineIndex][direction][label] * (0.0 - labelVotesSum[label]) * labelWeight)
          continue
        continue
      continue

    ## Icon corner constraints (one icon corner belongs to at most one icon)
    pointIconsMap = {}
    for iconIndex, icon in enumerate(icons):
      for cornerIndex in range(4):
        pointIndex = icon[cornerIndex]
        if pointIndex not in pointIconsMap:
          pointIconsMap[pointIndex] = []
          pass
        pointIconsMap[pointIndex].append(iconIndex)
        continue
      continue

    for pointIndex, iconIndices in pointIconsMap.items():
      iconSum = LpAffineExpression()
      for iconIndex in iconIndices:
        iconSum += i_r[iconIndex]
        continue
      model += (iconSum <= 1)
      continue

    ## Wall confidence objective
    # wallLineConfidenceMap = roomHeatmaps[:, :, WALL_LABEL_OFFSET]
    # #cv2.imwrite(output_prefix + 'confidence.png', (wallLineConfidenceMap * 255).astype(np.uint8))
    # wallConfidences = []
    # for lineIndex, line in enumerate(wallLines):
    #   point_1 = np.array(wallPoints[line[0]][:2])
    #   point_2 = np.array(wallPoints[line[1]][:2])
    #   lineDim = calcLineDim(wallPoints, line)
    #
    #   fixedValue = int(round((point_1[1 - lineDim] + point_2[1 - lineDim]) // 2))
    #   point_1[lineDim], point_2[lineDim] = min(point_1[lineDim], point_2[lineDim]), max(point_1[lineDim], point_2[lineDim])
    #
    #   point_1[1 - lineDim] = fixedValue - wallLineWidth
    #   point_2[1 - lineDim] = fixedValue + wallLineWidth
    #
    #   point_1 = np.maximum(point_1, 0).astype(np.int32)
    #   point_2 = np.minimum(point_2, sizes - 1).astype(np.int32)
    #
    #   wallLineConfidence = np.sum(wallLineConfidenceMap[point_1[1]:point_2[1] + 1, point_1[0]:point_2[0] + 1]) / ((point_2[1] + 1 - point_1[1]) * (point_2[0] + 1 - point_1[0])) - 0.5
    #
    #   obj += (-wallLineConfidence * w_l[lineIndex] * wallWeight)
    #
    #   wallConfidences.append(wallLineConfidence)
    #   continue

    if not gt:
      for wallIndex, wallLine in enumerate(wallLines):
        #print('wall confidence', wallIndex, [np.array(wallPoints[pointIndex][:2]).astype(np.int32).tolist() for pointIndex in wallLine], wallConfidences[wallIndex])
        continue
      pass


    ## Door confidence objective
    doorLineConfidenceMap = iconHeatmaps[:, :, DOOR_LABEL_OFFSET]
    #cv2.imwrite(output_prefix + 'confidence.png', (doorLineConfidenceMap * 255).astype(np.uint8))
    #cv2.imwrite(output_prefix + 'segmentation.png', drawSegmentationImage(doorCornerHeatmaps))

    for lineIndex, line in enumerate(doorLines):
      point_1 = np.array(doorPoints[line[0]][:2])
      point_2 = np.array(doorPoints[line[1]][:2])
      lineDim = calcLineDim(doorPoints, line)

      fixedValue = int(round((point_1[1 - lineDim] + point_2[1 - lineDim]) // 2))

      #assert(point_1[lineDim] < point_2[lineDim], 'door line reversed')
      point_1[lineDim], point_2[lineDim] = min(point_1[lineDim], point_2[lineDim]), max(point_1[lineDim], point_2[lineDim])

      point_1[1 - lineDim] = fixedValue - doorLineWidth
      point_2[1 - lineDim] = fixedValue + doorLineWidth

      point_1 = np.maximum(point_1, 0).astype(np.int32)
      point_2 = np.minimum(point_2, sizes - 1).astype(np.int32)

      if not gt:
        doorLineConfidence = np.sum(doorLineConfidenceMap[point_1[1]:point_2[1] + 1, point_1[0]:point_2[0] + 1]) / ((point_2[1] + 1 - point_1[1]) * (point_2[0] + 1 - point_1[0]))

        if lineDim == 0:
          doorPointConfidence = (doorCornerHeatmaps[point_1[1], point_1[0], 3] + doorCornerHeatmaps[point_2[1], point_2[0], 1]) / 2
        else:
          doorPointConfidence = (doorCornerHeatmaps[point_1[1], point_1[0], 0] + doorCornerHeatmaps[point_2[1], point_2[0], 2]) / 2
          pass
        doorConfidence = (doorLineConfidence + doorPointConfidence) * 0.5 - 0.5
        #print('door confidence', doorConfidence)
        obj += (-doorConfidence * d_l[lineIndex] * doorWeight)
      else:
        obj += (-0.5 * d_l[lineIndex] * doorWeight)
        pass
      continue

    ## Icon confidence objective
    for iconIndex, icon in enumerate(icons):
      point_1 = iconPoints[icon[0]]
      point_2 = iconPoints[icon[1]]
      point_3 = iconPoints[icon[2]]
      point_4 = iconPoints[icon[3]]

      x_1 = int((point_1[0] + point_3[0]) // 2)
      x_2 = int((point_2[0] + point_4[0]) // 2)
      y_1 = int((point_1[1] + point_2[1]) // 2)
      y_2 = int((point_3[1] + point_4[1]) // 2)

      iconArea = (x_2 - x_1 + 1) * (y_2 - y_1 + 1)

      if iconArea <= 1e-4:
        print(icon)
        print([iconPoints[pointIndex] for pointIndex in icon])
        print('zero size icon')
        exit(1)
        pass

      iconTypeConfidence = iconHeatmaps[y_1:y_2 + 1, x_1:x_2 + 1, :NUM_ICONS + 1].sum(axis=(0, 1)) / iconArea
      iconTypeConfidence = iconTypeConfidence[1:] - iconTypeConfidence[0]

      if not gt:
        iconPointConfidence = (iconCornerHeatmaps[int(round(point_1[1])), int(round(point_1[0])), 2] + iconCornerHeatmaps[int(round(point_2[1])), int(round(point_2[0])), 3] + iconCornerHeatmaps[int(round(point_3[1])), int(round(point_3[0])), 1] + iconCornerHeatmaps[int(round(point_4[1])), int(round(point_4[0])), 0]) // 4 - 0.5
        iconConfidence = (iconTypeConfidence + iconPointConfidence) * 0.5
      else:
        iconConfidence = iconTypeConfidence
        pass

      #print('icon confidence', iconConfidence)
      for typeIndex in range(NUM_ICONS):
        obj += (-i_types[iconIndex][typeIndex] * (iconConfidence[typeIndex]) * iconTypeWeight)
        continue
      continue

    ## Icon type one hot constraints
    for iconIndex in range(len(icons)):
      typeSum = LpAffineExpression()
      for typeIndex in range(NUM_ICONS - 1):
        typeSum += i_types[iconIndex][typeIndex]
        continue
      model += (typeSum == i_r[iconIndex])
      continue


    ## Line sum constraints (each orientation has at most one wall line)
    for pointIndex, orientationLinesMap in enumerate(wallPointOrientationLinesMap):
      for orientation, lines in orientationLinesMap.items():
        #if len(lines) > 1:
        #print(lines)
        lineSum = LpAffineExpression()
        for lineIndex in lines:
          lineSum += w_l[lineIndex]
          continue

        model += (lineSum == w_p[pointIndex], "line_sum_" + str(pointIndex) + "_" + str(orientation))
        continue
      continue

    ## Conflict constraints
    for index, conflictLinePair in enumerate(conflictWallLinePairs):
      model += (w_l[conflictLinePair[0]] + w_l[conflictLinePair[1]] <= 1, 'conflict_wall_line_pair_' + str(index))
      continue

    for index, conflictLinePair in enumerate(conflictDoorLinePairs):
      model += (d_l[conflictLinePair[0]] + d_l[conflictLinePair[1]] <= 1, 'conflict_door_line_pair_' + str(index))
      continue

    for index, conflictIconPair in enumerate(conflictIconPairs):
      model += (i_r[conflictIconPair[0]] + i_r[conflictIconPair[1]] <= 1, 'conflict_icon_pair_' + str(index))
      continue

    for index, conflictLinePair in enumerate(conflictIconWallPairs):
      model += (i_r[conflictLinePair[0]] + w_l[conflictLinePair[1]] <= 1, 'conflict_icon_wall_pair_' + str(index))
      continue


    ## Door wall constraints (a door must sit on one and only one wall)
    for doorIndex, lines in enumerate(doorWallLineMap):
      if len(lines) == 0:
        model += (d_l[doorIndex] == 0, 'door_not_on_walls_' + str(doorIndex))
        continue
      lineSum = LpAffineExpression()
      for lineIndex in lines:
        lineSum += w_l[lineIndex]
        continue
      model += (d_l[doorIndex] <= lineSum, 'd_wall_line_sum_' + str(doorIndex))
      continue

    doorWallPointMap = findLinePointMap(doorPoints, doorLines, wallPoints, gap=GAPS['door_point_conflict'])
    for doorIndex, points in enumerate(doorWallPointMap):
      if len(points) == 0:
        continue
      pointSum = LpAffineExpression()
      for pointIndex in points:
        model += (d_l[doorIndex] + w_p[pointIndex] <= 1, 'door_on_two_walls_' + str(doorIndex) + '_' + str(pointIndex))
        continue
      continue

    if False:
      #model += (w_l[6] == 1)
      pass

    model += obj
    model.solve()

    #model.writeLP(debug_prefix + '/model.lp')
    #print('Optimization information', LpStatus[model.status], value(model.objective))

    if LpStatus[model.status] == 'Optimal':
      filteredWallLines = []
      filteredWallLabels = []
      filteredWallTypes = []
      wallPointLabels = [[-1, -1, -1, -1] for pointIndex in range(len(wallPoints))]

      for lineIndex, lineVar in enumerate(w_l):
        if lineVar.varValue < 0.5:
          continue
        filteredWallLines.append(wallLines[lineIndex])

        filteredWallTypes.append(0)

        labels = [11, 11]
        for direction in range(2):
          for label in range(NUM_ROOMS):
            if l_dir_labels[lineIndex][direction][label].varValue > 0.5:
              labels[direction] = label
              break
            continue
          continue

        filteredWallLabels.append(labels)
        #print('wall', lineIndex, labels, [np.array(wallPoints[pointIndex][:2]).astype(np.int32).tolist() for pointIndex in wallLines[lineIndex]], wallLineNeighbors[lineIndex][0].keys(), wallLineNeighbors[lineIndex][1].keys())
        line = wallLines[lineIndex]
        lineDim = calcLineDim(wallPoints, line)
        if lineDim == 0:
          wallPointLabels[line[0]][0] = labels[0]
          wallPointLabels[line[0]][1] = labels[1]
          wallPointLabels[line[1]][3] = labels[0]
          wallPointLabels[line[1]][2] = labels[1]
        else:
          wallPointLabels[line[0]][1] = labels[0]
          wallPointLabels[line[0]][2] = labels[1]
          wallPointLabels[line[1]][0] = labels[0]
          wallPointLabels[line[1]][3] = labels[1]
          pass
        continue

      if not gt:
        adjustPoints(wallPoints, filteredWallLines)
        mergePoints(wallPoints, filteredWallLines)
        adjustPoints(wallPoints, filteredWallLines)
        filteredWallLabels = [filteredWallLabels[lineIndex] for lineIndex in range(len(filteredWallLines)) if filteredWallLines[lineIndex][0] != filteredWallLines[lineIndex][1]]
        filteredWallLines = [line for line in filteredWallLines if line[0] != line[1]]
        pass

      # if output_prefix is not None:
      #     drawLines(output_prefix + 'result_line.png', width, height, wallPoints, filteredWallLines, filteredWallLabels, lineColor=255)
      resultImage = drawLines('', width, height, wallPoints, filteredWallLines, filteredWallLabels, None, lineWidth=5, lineColor=255)

      filteredDoorLines = []
      filteredDoorTypes = []
      for lineIndex, lineVar in enumerate(d_l):
        if lineVar.varValue < 0.5:
          continue
        #print(('door', lineIndex, [doorPoints[pointIndex][:2] for pointIndex in doorLines[lineIndex]]))
        filteredDoorLines.append(doorLines[lineIndex])

        filteredDoorTypes.append(0)
        continue

      filteredDoorWallMap = findLineMapSingle(doorPoints, filteredDoorLines, wallPoints, filteredWallLines, gap=GAPS['wall_door_neighbor'])
      adjustDoorPoints(doorPoints, filteredDoorLines, wallPoints, filteredWallLines, filteredDoorWallMap)
      # if output_prefix is not None:
      #     drawLines(output_prefix + 'result_door.png', width, height, doorPoints, filteredDoorLines, lineColor=128)

      filteredIcons = []
      filteredIconTypes = []
      for iconIndex, iconVar in enumerate(i_r):
        if iconVar.varValue < 0.5:
          continue

        filteredIcons.append(icons[iconIndex])
        iconType = -1
        for typeIndex in range(NUM_ICONS):
          if i_types[iconIndex][typeIndex].varValue > 0.5:
            iconType = typeIndex
            break
          continue

        #print(('icon', iconIndex, iconType, [iconPoints[pointIndex][:2] for pointIndex in icons[iconIndex]]))

        filteredIconTypes.append(iconType)
        continue

      #adjustPoints(iconPoints, filteredIconLines)
      #drawLines(output_prefix + 'lines_results_icon.png', width, height, iconPoints, filteredIconLines)
      # if output_prefix is not None:
      #     drawRectangles(output_prefix + 'result_icon.png', width, height, iconPoints, filteredIcons, filteredIconTypes)

      resultImage = drawLines('', width, height, doorPoints, filteredDoorLines, [], resultImage, lineWidth=3, lineColor=0)
      resultImage = drawRectangles('', width, height, iconPoints, filteredIcons, filteredIconTypes, 2, resultImage)
      if save_image:
          cv2.imwrite(output_prefix + '_image.png', resultImage)

      filteredWallPoints = []
      filteredWallPointLabels = []
      orientationMap = {}
      for pointType, orientationOrientations in enumerate(POINT_ORIENTATIONS):
        for orientation, orientations in enumerate(orientationOrientations):
          orientationMap[orientations] = orientation

      for pointIndex, point in enumerate(wallPoints):
        orientations = []
        orientationLines = {}
        for orientation, lines in wallPointOrientationLinesMap[pointIndex].items():
          orientationLine = -1
          for lineIndex in lines:
            if w_l[lineIndex].varValue > 0.5:
              orientations.append(orientation)
              orientationLines[orientation] = lineIndex
              break
            continue
          continue

        if len(orientations) == 0:
          continue

        #print((pointIndex, orientationLines))

        if len(orientations) < len(wallPointOrientationLinesMap[pointIndex]):
          print('invalid point', pointIndex, orientations, wallPointOrientationLinesMap[pointIndex])
          print(wallPoints[pointIndex])
          wallPoints[pointIndex][2] = len(orientations) - 1
          orientations = tuple(orientations)
          if orientations not in orientationMap:
            continue
          wallPoints[pointIndex][3] = orientationMap[orientations]
          print(wallPoints[pointIndex])
          exit(1)
          pass

        filteredWallPoints.append(wallPoints[pointIndex])
        filteredWallPointLabels.append(wallPointLabels[pointIndex])
        continue

      if output_prefix is not None:
          with open(output_prefix + 'floorplan.txt', 'w') as result_file:
            result_file.write(str(width) + '\t' + str(height) + '\n')
            result_file.write(str(len(filteredWallLines)) + '\n')
            for wallIndex, wall in enumerate(filteredWallLines):
              point_1 = wallPoints[wall[0]]
              point_2 = wallPoints[wall[1]]

              result_file.write(str(point_1[0]) + '\t' + str(point_1[1]) + '\t')
              result_file.write(str(point_2[0]) + '\t' + str(point_2[1]) + '\t')
              result_file.write(str(filteredWallLabels[wallIndex][0]) + '\t' + str(filteredWallLabels[wallIndex][1]) + '\n')

            for doorIndex, door in enumerate(filteredDoorLines):
              point_1 = doorPoints[door[0]]
              point_2 = doorPoints[door[1]]

              result_file.write(str(point_1[0]) + '\t' + str(point_1[1]) + '\t')
              result_file.write(str(point_2[0]) + '\t' + str(point_2[1]) + '\t')
              result_file.write('door\t')
              result_file.write(str(filteredDoorTypes[doorIndex] + 1) + '\t1\n')

            for iconIndex, icon in enumerate(filteredIcons):
              point_1 = iconPoints[icon[0]]
              point_2 = iconPoints[icon[1]]
              point_3 = iconPoints[icon[2]]
              point_4 = iconPoints[icon[3]]

              x_1 = int((point_1[0] + point_3[0]) // 2)
              x_2 = int((point_2[0] + point_4[0]) // 2)
              y_1 = int((point_1[1] + point_2[1]) // 2)
              y_2 = int((point_3[1] + point_4[1]) // 2)

              result_file.write(str(x_1) + '\t' + str(y_1) + '\t')
              result_file.write(str(x_2) + '\t' + str(y_2) + '\t')
              result_file.write(iconNumberNameMap[filteredIconTypes[iconIndex]] + '\t')
              #result_file.write(str(iconNumberStyleMap[filteredIconTypes[iconIndex]]) + '\t')
              result_file.write('1\t')
              result_file.write('1\n')

            result_file.close()


      # writePoints(filteredWallPoints, filteredWallPointLabels, output_prefix=output_prefix)

      # if len(filteredDoorLines) > 0:
      #   writeDoors(doorPoints, filteredDoorLines, filteredDoorTypes, output_prefix=output_prefix)
      #   pass
      # else:
      #   try:
      #     os.remove(output_prefix + 'doors_out.txt')
      #   except OSError:
      #     pass

      # if len(filteredIcons) > 0:
      #   writeIcons(iconPoints, filteredIcons, filteredIconTypes, output_prefix=output_prefix)
      #   pass
      # else:
      #   try:
      #     os.remove(output_prefix + 'icons_out.txt')
      #   except OSError:
      #     pass
      #   pass

    else:
      print('infeasible')
      #model.ComputeIIS()
      #model.write("test/model.ilp")
      return {}
      pass
  result_dict = {'wall': [wallPoints, filteredWallLines, filteredWallLabels], 'door': [doorPoints, filteredDoorLines, []]}
  return result_dict#['wall'][0]
