## **前言**



之前对训练train.py、val.py进行解读了，接下来就是对检测代码detect.py 进行解读。整体逻辑和train.py一致，并且大部分函数和val.py中的一样，所以读起来应该问题不大。



## **代码解读**



开头依旧是进行解析配置参数：parse_opt，以及主要的run函数。



```python
def parse_opt():
    """
    opt参数解析
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--weights', nargs='+', type=str, default='weights/best.pt', help='model.pt path(s)')       #   weights: 模型的权重地址 默认 weights/best.pt

    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')        # source: 测试数据文件(图片或视频)的保存路径 默认data/images

    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')    # imgsz: 网络输入图片的大小 默认640

    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')      #   conf-thres: object置信度阈值 默认0.25

    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')      #  iou-thres: 做nms的iou阈值 默认0.45

    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')       #  max-det: 每张图片最大的目标个数 默认1000

    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')   #   device: 设置代码执行的设备 cuda device, i.e. 0 or 0,1,2,3 or cpu

    parser.add_argument('--view-img', action='store_true', help='show results')     #  view-img: 是否展示预测之后的图片或视频 默认False

    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')        #  save-txt: 是否将预测的框坐标以txt文件格式保存 默认True 会在runs/detect/expn/labels下生成每张图片预测的txt文件

    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')   #  save-conf: 是否保存预测每个目标的置信度到预测tx文件中 默认True

    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')       #  save-crop: 是否需要将预测到的目标从原图中扣出来 剪切好 并保存 会在runs/detect/expn下生成crops文件，将剪切的图片保存在里面  默认False

    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')      #  nosave: 是否不要保存预测后的图片  默认False 就是默认要保存预测后的图片

    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')  # classes: 在nms中是否是只保留某些特定的类 默认是None 就是所有类只要满足条件都可以保留

    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')   #  agnostic-nms: 进行nms是否也除去不同类别之间的框 默认False

    parser.add_argument('--augment', action='store_true', help='augmented inference')       #  augment: 预测是否也要采用数据增强 TTA

    parser.add_argument('--update', action='store_true', help='update all models')      #  update: 是否将optimizer从ckpt中删除  更新模型  默认False

    parser.add_argument('--project', default='runs/detect', help='save results to project/name')    #   project: 当前测试结果放在哪个主文件夹下 默认runs/detect

    parser.add_argument('--name', default='exp', help='save results to project/name')   #  name: 当前测试结果放在run/detect下的文件名  默认是exp

    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')   #  exist-ok: 是否存在当前文件 默认False 一般是 no exist-ok 连用  所以一般都要重新创建文件夹

    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')    # line-thickness: 画框的框框的线宽  默认是 3

    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')    # hide-labels: 画出的框框是否需要隐藏label信息 默认False

    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences') # hide-conf: 画出的框框是否需要隐藏conf信息 默认False

    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')    # half: 是否使用半精度 Float16 推理 可以缩短推理时间 但是默认是False

    parser.add_argument('--prune-model', default=False, action='store_true', help='model prune')    # 是否对模型进行裁剪

    parser.add_argument('--fuse', default=False, action='store_true', help='fuse conv and bn')  #   融合conv 和bn 层

    opt = parser.parse_args()

    return opt
```



在run上加上`@torch.no_grad()` ，就表示模型只进行前向推理而不用进行梯度的反向更新。



```python
@torch.no_grad()
def run(weights='weights/yolov5s.pt', # 权重文件地址 默认 weights/best.pt
    source='data/images',     # 测试数据文件(图片或视频)的保存路径 默认data/images
    imgsz=640,          # 输入图片的大小 默认640(pixels)
    conf_thres=0.25,       # object置信度阈值 默认0.25 用于nms中
    iou_thres=0.45,        # 做nms的iou阈值 默认0.45 用于nms中
    max_det=1000,         # 一张图片中最多存在多少目标 用于nms中
    device='',          # 使用哪块GPU或者使用CPU
    view_img=False,        # 是否展示预测之后的图片或视频 默认False
    save_txt=False, # 是否将预测的框坐标以txt文件格式保存 默认True 会在runs/detect/expn/labels下生成每张图片预测的txt文件
    save_conf=False, # 是否保存预测每个目标的置信度到预测tx文件中 默认True
    save_crop=False, # 是否需要将预测到的目标从原图中扣出来 剪切好 并保存 会在runs/detect/expn下生成crops文件，将剪切的图片保存在里面 默认False
    nosave=False,  # 是否不要保存预测后的图片 默认False 就是默认要保存预测后的图片
    classes=None,  # 在nms中是否是只保留某些特定的类 默认是None 就是所有类只要满足条件都可以保留
    agnostic_nms=False,  # 进行nms是否也除去不同类别之间的框 默认False
    augment=False,     # 预测是否也要采用数据增强 TTA 默认False
    update=False,     # 是否将optimizer从ckpt中删除 更新模型 默认False
    project='runs/detect', # 当前测试结果放在哪个主文件夹下 默认runs/detect
    name='exp',      # 当前测试结果放在run/detect下的文件名 默认是exp => run/detect/exp，每运行一次，次数都会增加一
    exist_ok=False,    # 判断是否存在当前文件 默认False 一般是 no exist-ok 连用 所以一般都要重新创建文件夹
    line_thickness=3,   # 用于画框，opencv画框时的线宽 默认是 3
    hide_labels=False,   # 用于画框，看画出的框框是否需要隐藏label信息 默认False
    hide_conf=False,    # 用于画框，看画出的框框是否需要隐藏conf信息 默认False
    half=False,      # 是否使用半精度fp16推理，进行推理加速，默认是False
    prune_model=False,   # 是否使用模型剪枝 进行推理加速
    fuse=False,      # 是否使用conv + bn融合技术，这样可以对推理进行加速
    ):
```



下面的是初始化配置、载入模型参数、加载推理数据，和val.py中的开始流程差不多。



```python
  save_img = not nosave and not source.endswith('.txt') # save inference images True
  webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
    ('rtsp://', 'rtmp://', 'http://', 'https://'))
  save_dir = increment_path(Path(project) / name, exist_ok=exist_ok) # increment run
  (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True) # 创建文件夹

  set_logging()
  device = select_device(device)

  # 如果设备是GPU，就可以使用half进行半精度的推理
  half &= device.type != 'cpu' # half precision only supported on CUDA

  model = attempt_load(weights, map_location=device)

  # 这里判断是否需要使用剪枝，同样是为了加速推理
  if prune_model:
    model_info(model) # 打印模型信息
    prune(model, 0.3) # 对模型进行剪枝 加速推理
    model_info(model) # 再打印模型信息 观察剪枝后模型变化

  # 是否使用模型的conv+bn融合技术，同样是为了加速推理
  if fuse:
    model = model.fuse() # 将模型的conv+bn融合 可以加速推理

  # 2.2、载入模型的参数
  # stride: 模型最大的下采样率，为[8, 16, 32]，所有stride一般为32，最大为32
  stride = int(model.stride.max()) # model stride

  # 确保输入图片的尺寸imgsz能被stride整除，如果不能则调整为能被整除并返回
  imgsz = check_img_size(imgsz, s=stride) # check image size 保证img size必须是32的倍数

  # 获取数据集中所有类别
  names = model.module.names if hasattr(model, 'module') else model.names # get class names

  # 判断是否需要用到fp16精度进行推理，如果要进行fp16精度的推理，就执行
  if half:
    model.half() # to float16

  # 判断是否需要进行二次分类，如果是的话就加载二次分类模型 默认为False
  classify = False
  if classify:
    modelc = load_classifier(name='resnet50', n=2) # initialize
    modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()

  # ===================================== 3、加载推理数据 =====================================
  # Set Dataloader
  # 通过不同的输入源来设置不同的数据加载方式
  vid_path, vid_writer = None, None
  if webcam:
    # 从网页上的webcam模式获取数据
    view_img = check_imshow()
    cudnn.benchmark = True # set True to speed up constant image size inference
    dataset = LoadStreams(source, img_size=imgsz, stride=stride)
  else:
    # 从source文件夹下读取视频或者图片
    dataset = LoadImages(source, img_size=imgsz, stride=stride)
```



然后再推理之前使用一个空的tensor进行测试，我觉得相当是进行热身。



```python
# 这里先设置一个全零的Dummy Tensor来进行一次前向推理，测试程序是否能正常运行
  if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters()))) # run once
```



正式推理的过程如下：



```python
  for path, img, im0s, vid_cap in dataset:
      # path: 图片/视频的路径
      # img: 进行resize + pad之后的图片
      # img0s: 原尺寸的图片
      # vid_cap: 如果是图片则为None，视频则为视频源
      img = torch.from_numpy(img).to(device) # numpy array to tensor and device
      img = img.half() if half else img.float()
      img /= 255.0 # 将图片进行归一化 0 - 255 to 0.0 - 1.0
      # 如果图片是3维(RGB) ，就要再扩充一个维度，即添加一个维度1当中batch_size=1
      # 因为输入网络的图片需要是四维的 [batch_size, channel, w, h]
      if img.ndimension() == 3:
        img = img.unsqueeze(0)
  
      t1 = time_synchronized()
      pred = model(img, augment=augment)[0]  # 这里的shape为 [1, num_boxes, xywh+obj_conf+classes] = [1, 18900, 25]
  
      pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
      # agnostic_nms代表进行nms时是否也去除不同类别之间的框，默认False，需要的时候可以打开
      # pred: [num_obj, 6] = [5, 6] 这里的预测信息pred还是相对于 img_size(640) 的
      t2 = time_synchronized()
  
      # Apply Classifier 如果需要二次分类 就进行二次分类
      if classify:
        pred = apply_classifier(pred, modelc, img, im0s)
```



在每一张图片中，还会对每张图片的各个对象进行处理。处理完之后，会打印模型的前向推理加上nms后处理所花的时间。



```python
  for i, det in enumerate(pred): # detections per image
        if webcam:
          p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
        else:
          p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
          # p: 当前图片/视频的绝对路径
          # s: 输出信息 初始为 ''
          # im0: 原始图片 letterbox + pad 之前的图片
          # frame: 初始为0 代表当前图片属于视频中的第几帧
  
        p = Path(p) # to Path
        save_path = str(save_dir / p.name) # img.jpg
        txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}') # img.txt
  
        # s为输出信息
        s += '%gx%g ' % img.shape[2:]
  
        # normalization gain gn = [w, h, w, h] 用于后面的归一化
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
  
        # imc: for save_crop 在save_crop中使用
        imc = im0.copy() if save_crop else im0
  
        if len(det):
          # Rescale boxes from img_size to im0 size
          # 将预测结果映射回原图，也就是从相对img_size 640映射到img0 size
          det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
  
          # Print results
          # 计算每个类别的个数，将结果中每个类别的目标个数添加到输出信息s中
          for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum() # detections per class
            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, " # add to string
  
          # Write results
          # 保存预测信息: txt、img0上画框、crop_img
          for *xyxy, conf, cls in reversed(det):
            # 将每个图片的预测信息分别存入save_dir/labels下的xxx.txt中 每行: class_id+score+xywh
            if save_txt: # Write to file(txt)
            # 将xyxy，即左上角 + 右下角两点的格式转换为xywh，即中心的+宽高的格式 并除以gn(whwh)作归一化，转为coco格式的坐标
              xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist() # normalized xywh
              line = (cls, *xywh, conf) if save_conf else (cls, *xywh) # label format
              with open(txt_path + '.txt', 'a') as f:
                f.write(('%g ' * len(line)).rstrip() % line + '\n')
  
            # 在原图上进行画框
            if save_img or save_crop or view_img:
              c = int(cls) # integer class
              label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
              plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)
              if save_crop:
                # 如果需要裁剪，则将预测到的目标小图裁剪出来，并保存在save_dir/crops下
                save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
print(f'{s}Done. ({t2 - t1:.3f}s)')
```



我觉得里面的nms可以单独拿出来讲讲，具体的注释如下。



```python
def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None,
            agnostic=False, multi_label=True, labels=(), max_det=300, merge=False):
  """
  Runs Non-Maximum Suppression (NMS) on inference results
  Params:
     prediction: [batch, num_anchors(3个yolo预测层), (x+y+w+h+1+num_classes)] = [1, 18900, 25] 3个anchor的预测结果总和
     conf_thres: 先进行一轮筛选，将分数过低的预测框（<conf_thres）删除（分数置0）
     iou_thres: iou阈值, 如果其余预测框与target的iou>iou_thres, 就将那个预测框置0
     classes: 是否nms后只保留特定的类别 默认为None
     agnostic: 进行nms是否也去除不同类别之间的框 默认False
     multi_label: 是否是多标签 nc>1 一般是True
     labels: {list: bs} 第一张图片的target[17, 5] 第二张[1, 5] 第三张[7, 5] 第四张[6, 5]
     max_det: 每张图片的最大目标个数 默认1000
     merge: use merge-NMS 多个bounding box给它们一个权重进行融合 默认False
  Returns:
     [num_obj, x1y1x2y2+object_conf+cls] = [5, 6]
  """
  # Checks 检查传入的conf_thres和iou_thres两个阈值是否符合范围
  assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
  assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

  # Settings 设置一些变量
  nc = prediction.shape[2] - 5 # number of classes
  min_wh, max_wh = 2, 4096 # (pixels) 预测物体宽度和高度的大小范围 [min_wh, max_wh]
  max_nms = 30000 # 每个图像最多检测物体的个数 maximum number of boxes into torchvision.ops.nms()
  time_limit = 10.0 # nms执行时间阈值 超过这个时间就退出了 seconds to quit after
  redundant = True # 是否需要冗余的detections require redundant detections
  multi_label &= nc > 1 # multiple labels per box (adds 0.5ms/img)
  # batch_size个output 存放最终筛选后的预测框结果
  output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
  # 定义第二层过滤条件
  xc = prediction[..., 4] > conf_thres # candidates

  t = time.time() # 记录当前时刻时间
  for xi, x in enumerate(prediction): # image index, image inference
    # Apply constraints
    # 第一层过滤 虑除超小anchor标和超大anchor x=[18900, 25]
    x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0 # width-height

    # 第二层过滤 根据conf_thres虑除背景目标(obj_conf<conf_thres 0.1的目标 置信度极低的目标) x=[59, 25]
    x = x[xc[xi]] # confidence

    # {list: bs} 第一张图片的target[17, 5] 第二张[1, 5] 第三张[7, 5] 第四张[6, 5]
    # Cat apriori labels if autolabelling 自动标注label时调用 一般不用
    # 自动标记在非常高的置信阈值（即 0.90 置信度）下效果最佳,而 mAP 计算依赖于非常低的置信阈值（即 0.001）来正确评估 PR 曲线下的区域。
    if labels and len(labels[xi]): #
      l = labels[xi]
      v = torch.zeros((len(l), nc + 5), device=x.device) # [17:85] [1,85] [7,85] [6,85]
      v[:, :4] = l[:, 1:5] # v[:, :4]=box
      v[:, 4] = 1.0    # v[:, 4]=conf
      v[range(len(l)), l[:, 0].long() + 5] = 1.0 # v[:, target相应位置cls,其他位置为0]=1
      x = torch.cat((x, v), 0) # x: [1204, 85] v: [17, 85] => x: [1221, 85]

    # 经过前两层过滤后如果该feature map没有目标框了，就结束这轮直接进行下一张图
    if not x.shape[0]:
      continue

    # 计算conf_score
    x[:, 5:] *= x[:, 4:5] # conf = obj_conf * cls_conf

    # Box (center x, center y, width, height) to (x1, y1, x2, y2) 左上角 右下角 [59, 4]
    box = xywh2xyxy(x[:, :4])

    # Detections matrix nx6 (xyxy, conf, cls)
    if multi_label:
      # 第三轮过滤:针对每个类别score(obj_conf * cls_conf) > conf_thres  [59, 6] -> [51, 6]
      # 这里一个框是有可能有多个物体的，所以要筛选
      # nonzero: 获得矩阵中的非0(True)数据的下标 a.t(): 将a矩阵拆开
      # i: 下标 [43] j: 类别index [43] 过滤了两个score太低的
      i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
      # pred = [43, xyxy+score+class] [43, 6]
      # unsqueeze(1): [43] => [43, 1] add batch dimension
      # box[i]: [43,4] xyxy
      # pred[i, j + 5].unsqueeze(1): [43,1] score 对每个i,取第（j+5）个位置的值（第j个class的值cla_conf）
      # j.float().unsqueeze(1): [43,1] class
      x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
    else: # best class only
      conf, j = x[:, 5:].max(1, keepdim=True)  # 一个类别直接取分数最大类的即可
      x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

    # Filter by class 是否只保留特定的类别 默认None 不执行这里
    if classes is not None:
      x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

    # 检测数据是否为有限数 Apply finite constraint 这轮可有可无，一般没什么用 所以这里给他注释了
    # if not torch.isfinite(x).all():
    #  x = x[torch.isfinite(x).all(1)]

    # Check shape
    n = x.shape[0] # number of boxes
    if not n: # 如果经过第三轮过滤该feature map没有目标框了，就结束这轮直接进行下一张图
      continue
    elif n > max_nms: # 如果经过第三轮过滤该feature map还要很多框(>max_nms) 就需要排序
      x = x[x[:, 4].argsort(descending=True)[:max_nms]] # sort by confidence

    # 第4轮过滤 Batched NMS [51, 6] -> [5, 6]
    c = x[:, 5:6] * (0 if agnostic else max_wh) # classes
    # 做个切片 得到boxes和scores 不同类别的box位置信息加上一个很大的数但又不同的数c
    # 这样作非极大抑制的时候不同类别的框就不会掺和到一块了 这是一个作nms挺巧妙的技巧
    boxes, scores = x[:, :4] + c, x[:, 4] # boxes (offset by class), scores
    # 返回nms过滤后的bounding box(boxes)的索引（降序排列）
    # i=tensor([18, 19, 32, 25, 27]) nms后只剩下5个预测框了
    i = torchvision.ops.nms(boxes, scores, iou_thres) # NMS

    if i.shape[0] > max_det: # limit detections
      i = i[:max_det]

    if merge and (1 < n < 3E3): # Merge NMS (boxes merged using weighted mean)
      # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
      iou = box_iou(boxes[i], boxes) > iou_thres # iou matrix
      weights = iou * scores[None] # box weights 正比于 iou * scores
      # bounding box合并 其实就是把权重和框相乘再除以权重之和
      x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True) # merged boxes
      if redundant:
        i = i[iou.sum(1) > 1] # require redundancy

    output[xi] = x[i] # 最终输出 [5, 6]

    # 看下时间超没超时 超时没做完的就不做了
    if (time.time() - t) > time_limit:
      print(f'WARNING: NMS time limit {time_limit}s exceeded')
      break # time limit exceeded

  return output
```



主要的检测结果就是上面的，后面的就是显示推理结果、保存推理结果，我觉得主要用于调试以及进行持久化保存。



```python
      # Save results (image with detections)
      # 判断是否需要保存推理后的图片结果或视频视频结果
      if save_img:
        if dataset.mode == 'image':
          cv2.imwrite(save_path, im0)
        else: # 'video' or 'stream'
          if vid_path != save_path: # new video
            vid_path = save_path
            if isinstance(vid_writer, cv2.VideoWriter):
              vid_writer.release() # release previous video writer
            if vid_cap: # video
              fps = vid_cap.get(cv2.CAP_PROP_FPS)
              w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
              h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            else: # stream
              fps, w, h = 30, im0.shape[1], im0.shape[0]
              save_path += '.mp4'
            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
          vid_writer.write(im0)
```



至此，推理就结束了。程序的最后，还会根据你的设置来进行保存、打印信息。



```python
  # 打印将预测的label信息 保存到哪里
  if save_txt or save_img:
    s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    print(f"Results saved to {save_dir}{s}")

  if update:
    # strip_optimizer函数将optimizer从ckpt中删除 更新模型
    strip_optimizer(weights) # update model (to fix SourceChangeWarning)

  # 打印预测的总耗时
  print(f'Done. ({time.time() - t0:.3f}s)')
```



以上有个函数为strip_optimizer，在train.py和detect.py文件中都有调用。它的作用是将optimizer、training_results如epoch等信息从保存的模型文件f中删除，目的是优化最后所保存的模型大小。



```python
def strip_optimizer(f='best.pt', s=''):
  """用在train.py模型训练完后
  将optimizer、training_results、updates...从保存的模型文件f中删除
  Strip optimizer from 'f' to finalize training, optionally save as 's'
  :params f: 传入的原始保存的模型文件
  :params s: 删除optimizer等变量后的模型保存的地址 dir
  """
  # x: 为加载训练的模型
  x = torch.load(f, map_location=torch.device('cpu'))
  # 如果模型是ema replace model with ema
  if x.get('ema'):
    x['model'] = x['ema']
  # 以下模型训练涉及到的若干个指定变量置空
  for k in 'optimizer', 'training_results', 'wandb_id', 'ema', 'updates': # keys
    x[k] = None
  x['epoch'] = -1 # 模型epoch恢复初始值-1
  x['model'].half() # to FP16
  for p in x['model'].parameters():
    p.requires_grad = False
  # 保存模型 x -> s/f
  torch.save(x, s or f)
  mb = os.path.getsize(s or f) / 1E6 # filesize
  print(f"Optimizer stripped from {f},{(' saved as %s,' % s) if s else ''} {mb:.1f}MB")
```



## 文末



YOLO系列的八篇文章就解读完了，从网络结构的原理到代码的解读，都讲了一遍，但是总感觉只是浅尝辄止。

所以笔者还是认为，这里只是起到抛砖引玉的作用，真正要做到深刻理解，还是要自己去调试、运行，应用到自己的项目当中。多去问自己为什么要这么做，相信这样的话，你会收获更多！

