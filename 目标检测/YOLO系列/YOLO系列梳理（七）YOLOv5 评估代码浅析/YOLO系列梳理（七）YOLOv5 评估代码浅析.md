## 前言



之前笔者对YOLOv5中的训练代码train.py进行了解读，今天就继续对评估代码，也就是对val.py进行解读。val.py的代码量比train.py少很多，之前有了解读train.py的基础，那么接下来这里解读起来也比较得心应手。



## 代码解读



主要逻辑仍然是解析配置参数，也就是parse_opt函数，然后进行评估，也就是run函数。



```python
def parse_opt():
    """
    opt参数详解

    """
    parser = argparse.ArgumentParser(prog='val.py')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='dataset.yaml path')      #   data: 数据集配置文件地址 包含数据集的路径、类别个数、类名、下载地址等信息
    parser.add_argument('--weights', nargs='+', type=str, default='weights/yolov5s.pt', help='model.pt path(s)')        #  weights: 模型的权重文件地址 weights/yolov5s.pt
    # parser.add_argument('--weights', nargs='+', type=str,
    #                     default=['weights/yolov5s.pt', 'weights/yolov5m.pt'], help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=4, help='batch size')     #  batch_size: 前向传播的批次大小 默认32
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')        #   imgsz: 输入网络的图片分辨率 默认640
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')     #   conf-thres: object置信度阈值 默认0.25
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')       #  iou-thres: 进行NMS时IOU的阈值 默认0.6
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')   #  task: 设置测试的类型 有train, val, test, speed or study几种 默认为val
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')   #   device: 测试的设备
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')      #   single-cls: 数据集是否只用一个类别 默认False
    parser.add_argument('--augment', action='store_true', help='augmented inference')       #  augment: 测试是否使用TTA Test Time Augment 默认False
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')       #  verbose: 是否打印出每个类别的mAP 默认False
    parser.add_argument('--save-txt', default=False, action='store_true', help='save results to *.txt')     #   save-txt: 是否以txt文件的形式保存模型预测框的坐标 默认True
    parser.add_argument('--save-hybrid', default=False, action='store_true', help='save label+prediction hybrid results to *.txt')      #  save-hybrid: 是否save label+prediction hybrid results to *.txt  默认False 是否将gt_label+pre_label一起输入nms
    parser.add_argument('--save-conf', default=False, action='store_true', help='save confidences in --save-txt labels')    #  save-conf: 是否保存预测每个目标的置信度到预测tx文件中 默认True
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')     #  save-json: 是否按照coco的json格式保存预测框，并且使用cocoapi做评估（需要同样coco的json格式的标签） 默认False
    parser.add_argument('--project', default='runs/test', help='save to project/name')      #  project: 测试保存的源文件 默认runs/test
    parser.add_argument('--name', default='exp', help='save to project/name')       #  name: 测试保存的文件地址 默认exp  保存在runs/test/exp下
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')       #  exist-ok: 是否存在当前文件 默认False 一般是 no exist-ok 连用  所以一般都要重新创建文件夹
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')        #  half: 是否使用半精度推理 默认False
    opt = parser.parse_args()   # 解析上述参数
    opt.save_json |= opt.data.endswith('coco.yaml')  # |或 左右两个变量有一个为True 左边变量就为True
    opt.save_txt |= opt.save_hybrid
    opt.data = check_file(opt.data)  # check file
    return opt
```



在run函数上面有一行这个：`@torch.no_grad() `。代表的是模型只进行前向推理而不进行反向传播。



```python
@torch.no_grad()   
def run(data, weights=None, batch_size=32, imgsz=640, conf_thres=0.001, iou_thres=0.6,
        task='val', device='', single_cls=False, augment=False, verbose=False, save_txt=False,
        save_hybrid=False, save_conf=False, save_json=False, project='runs/test', name='exp',
        exist_ok=False, half=True, model=None, dataloader=None, save_dir=Path(''), plots=True,
        wandb_logger=None, compute_loss=None,
        ):
```



首先是进行模型初始化的配置。



```python
 	# 判断是否在训练时调用run函数(执行train.py脚本), 如果是就使用训练时的设备 一般都是train
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # 选择训练时对应的计算设备
    # 如果不是train.py调用run函数(执行val.py脚本)就调用select_device选择可用的设备
    # 并生成save_dir + make dir + 加载model + check imgsz + 加载data配置信息
    else:
        device = select_device(device, batch_size=batch_size)
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        # make dir run\test\expn\labels
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

        # 加载模型(可以加载普通模型或者集成模型) load FP32 model  只在运行test.py才需要自己加载model
        model = attempt_load(weights, map_location=device)

        # gs: 模型最大的下采样stride 
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)，使用模型最大的下采样gs，一般为[8, 16, 32] 所以gs一般是32
        # 检测输入图片的分辨率imgsz是否能被gs整除 只在运行test.py才需要自己生成check imgsz
        # imgsz_test
        imgsz = check_img_size(imgsz, s=gs)  # check image size，检查图片的分辨率是否能被gs整除

        # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
        # if device.type != 'cpu' and torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)

        # 因为训练时是有testloader，所以只有在单独运行这个脚本的时候才需要加载数据配置的信息，
        # 根据data生成新的dataloader
        with open(data, encoding='utf-8') as f:
            data = yaml.safe_load(f)
        check_dataset(data)  # check
```



在进行一系列的初始化配置后，这里还会判断是否需要使用half来调整模型。



```python
	# 如果使用half, 那么模型和图片都需要设为half
    half &= device.type != 'cpu'  # half precision only supported on CUDA。half model 只能在单GPU设备上才能使用。cpu设备和多gpu设备都不行
    if half:
        model.half()

    # from utils.torch_utils import prune
    # prune(model, 0.3)  # 模型剪枝

    # model = model.fuse()  # 模型融合  融合conv+bn

    model.eval()  # 启动模型验证模式
```



然后进行后面所需要的评估参数初始化、日志初始化和数据集加载。



```python
    # 测试数据是否是coco数据集 + class类别个数
    is_coco = type(data['val']) is str and data['val'].endswith('coco/val2017.txt')  # COCO dataset bool
    nc = 1 if single_cls else int(data['nc'])  # number of classes

    # 计算mAP相关参数
    # 设置iou阈值 从0.5-0.95取10个(0.05间隔)   iou vector for mAP@0.5:0.95
    iouv = torch.linspace(0.5, 0.95, 10).to(device) # 设置10个阈值，阈值从0.5-0.95中，每间隔0.05取值
    # iouv: [0.50000, 0.55000, 0.60000, 0.65000, 0.70000, 0.75000, 0.80000, 0.85000, 0.90000, 0.95000]
    # mAP@0.5:0.95 iou个数=10个
    niou = iouv.numel()

    # 初始化日志 Logging
    log_imgs = 0
    if wandb_logger and wandb_logger.wandb:
        log_imgs = min(wandb_logger.log_imgs, 100)

    # 因为训练时是有testloader，所以只有在单独运行这个脚本的时候才需要加载数据配置的信息，
    # 然后根据data调用create_dataloader生成新的dataloader
    if not training:
        if device.type != 'cpu':
            # 测试前向传播是否能正常运行，这里是创建一个全零的数组进行测试
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        # 创建dataloader 这里的rect默认为True 矩形推理用于测试集 在不影响mAP的情况下可以大大提升推理速度
        dataloader = create_dataloader(data[task], imgsz, batch_size, gs, single_cls, pad=0.5, rect=True,
                                       prefix=colorstr(f'{task}: '))[0]

    # 初始化一些测试需要的参数
    seen = 0  # 初始化测试的图片的数量
    # 初始化混淆矩阵
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    # 获取coco数据集的类别索引
    # coco数据集是80个类 索引范围本应该是0~79,但是这里返回的确是0~90  coco官方就是这样规定的
    # coco80_to_coco91_class就是为了与上述索引对应起来，返回一个范围在0~80的索引数组
    coco91class = coco80_to_coco91_class()
    # 使用tqdm设置进度条所要显示的信息
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1, t2 = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
```



进行完一系列的准备过程后，就会开始对数据集中的每一张图片进行评估验证。



```python
 for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        
        t_ = time_synchronized() 
        img = img.to(device, non_blocking=True)  # img to device
        # 如果half为True 就把图片也变为half  uint8 to fp16/32
        img = img.half() if half else img.float()
        img /= 255.0  # 归一化  0 - 255 to 0.0 - 1.0
        targets = targets.to(device)  # targets to device
        # batch size, channels, height, width
        nb, _, height, width = img.shape
        t = time_synchronized()
        t0 += t - t_ 

        # 6.2、Run model  前向推理
        # out:       推理结果 1个 [bs, anchor_num*grid_w*grid_h, xywh+c+20classes] = [1, 19200+4800+1200, 25]
        # train_out: 训练结果 3个 [bs, anchor_num, grid_w, grid_h, xywh+c+20classes]
        #                    如: [1, 3, 80, 80, 25] [1, 3, 40, 40, 25] [1, 3, 20, 20, 25]
        out, train_out = model(img, augment=augment)  # inference and training outputs
        t1 += time_synchronized() - t 

        # 6.3、计算验证损失
        # compute_loss不为空 说明正在执行train.py  根据传入的compute_loss计算损失值
        if compute_loss:
            loss += compute_loss([x.float() for x in train_out], targets)[1][:3]  # lbox, lobj, lcls

        # 6.4、Run NMS
        # 将gt的target中的xywh映射到img的test尺寸
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)
        # save_hybrid: adding the dataset labels to the model predictions before NMS
        #              
        # 在NMS之前将数据集标签targets添加到模型预测中，允许在数据集中自动标记(for autolabelling)其他对象(在pred中混入gt) 并且mAP反映了新的混合标签
        # targets: [num_target, img_index+class_index+xywh] = [31, 6]
        # lb: {list: bs} 第一张图片的target[17, 5] 第二张[1, 5] 第三张[7, 5] 第四张[6, 5]
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []
        t = time_synchronized()
        out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
        t2 += time_synchronized() - t   
```



在每一张图中，会对里面的所有的真实框与预测框进行统计。



```python
 # 统计每张图片的真实框、预测框信息，并将结果写入到txt文件，生成json文件字典，统计tp等
        # out: list{bs}  [300, 6] [42, 6] [300, 6] [300, 6]  [pred_obj_num, x1y1x2y2+object_conf+cls]
        for si, pred in enumerate(out):
           
            # 获取第si张图片的gt标签信息 包括class, x, y, w, h    target[:, 0]为标签属于哪张图片的编号
            labels = targets[targets[:, 0] == si, 1:]   # [:, class+xywh]
            nl = len(labels)  # 这张图片的gt个数
           
            tcls = labels[:, 0].tolist() if nl else []      # 获取gt的类别
            path = Path(paths[si])  # 这张图片的路径
            # 统计测试图片数量 +1
            seen += 1

            # 如果结果为空，就把空的信息加到stats里
            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # 把预测的坐标映射回原图中

            # 6.6、保存预测信息到txt文件，路径为 runs\test\exp7\labels\image_name.txt
            if save_txt:
                # gn为图片的宽高  用于后面归一化  [w, h, w, h] 
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                for *xyxy, conf, cls in predn.tolist():
                    
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    # 保存预测的类别和坐标到对应图片id.txt文件中
                    with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    # with open(save_dir / 'labels' / ('test' + '.txt'), 'a') as f:
                    #     f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # 使用wandb 保存预测信息，和tensorboard差不多
            if len(wandb_images) < log_imgs and wandb_logger.current_epoch > 0:  # Check for test operation
                if wandb_logger.current_epoch % wandb_logger.bbox_interval == 0:
                    box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                                 "class_id": int(cls),
                                 "box_caption": "%s %.3f" % (names[cls], conf),
                                 "scores": {"class_score": conf},
                                 "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
                    boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
                    wandb_images.append(wandb_logger.wandb.Image(img[si], boxes=boxes, caption=path.name))
            wandb_logger.log_training_progress(predn, path, names) if wandb_logger and wandb_logger.wandb_run else None
```



如果要保存为coco格式的json文件，也可以通过设置save_json这个参数来进行保存。



```python
 if save_json:
        # 获取图片id
        image_id = int(path.stem) if path.stem.isnumeric() else path.stem
        # 获取预测框 并将xyxy转为xywh格式
        box = xyxy2xywh(predn[:, :4])  # xywh
        # 之前的的xyxy格式是左上角右下角坐标  xywh是中心的坐标和宽高
        # 而coco的json格式的框坐标是xywh(左上角坐标 + 宽高)
        # 所以这行代码是将中心点坐标 -> 左上角坐标  xy center to top-left corner
        box[:, :2] -= box[:, 2:] / 2
        # image_id: 图片id 即属于哪张图片
        # category_id: 类别 coco91class()从索引0~79映射到索引0~90
        # bbox: 预测框坐标
        # score: 预测得分
        for p, b in zip(pred.tolist(), box.tolist()):
            jdict.append({'image_id': image_id,
                          'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                          'bbox': [round(x, 3) for x in b],
                          'score': round(p[4], 5)})
```



然后还会计算混淆矩阵。



```python
# 初始化预测评定 niou为iou阈值的个数  Assign all predictions as incorrect 
# correct = [pred_obj_num, 10] = [300, 10]  全是False
correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
if nl:
    detected = []  # target indices  放置已经检测到的目标
    tcls_tensor = labels[:, 0]  # 获取当前图片的所有gt的类别

    tbox = xywh2xyxy(labels[:, 1:5])  # gt boxes  获取xyxy格式的框
    # 将预测框映射到原图img
    scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
    if plots:
        # 计算混淆矩阵 confusion_matrix
        confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

        # Per target class
        # 单独处理每个图片中的每个类别
        for cls in torch.unique(tcls_tensor):
            # gt中该类别的索引 
            ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  #  nonzero: 获取列表中为True的index
            # 预测框中该类别的索引  prediction indices
            pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)

            # Search for detections
            if pi.shape[0]:
                # Prediction to target ious
                # predn[pi, :4]: 属于该类的预测框[144, 4]  tbox[ti]: 属于该类的gt框[13, 4]
                # box_iou: [144, 4] + [13, 4] => [144, 13]  计算属于该类的预测框与属于该类的gt框的iou

                ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # 选出每个预测框与所有gt box中最大的iou值, i为最大iou值时对应的gt索引

                # Append detections
                detected_set = set()  
                for j in (ious > iouv[0]).nonzero(as_tuple=False):  # j: ious中>0.5的索引 只有iou>=0.5才是TP
                    # 获得检测到的目标
                    d = ti[i[j]]  # detected target
                    if d.item() not in detected_set:
                        detected_set.add(d.item())  
                        detected.append(d) # 将当前检测到的gt框d添加到detected()
                        # iouv是以0.05为步长  从0.5到0.95的序列
                        # 从所有TP中获取不同iou阈值下的TP true positive  并在correct中记录下哪个预测框是哪个iou阈值下的TP
                        # correct: [pred_num, 10] = [300, 10]  记录着哪个预测框在哪个iou阈值下是TP
                        correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                        if len(detected) == nl:  # 当检测框的个数等于gt框的个数，就退出
                            break

                            # 将每张图片的预测结果统计到stats中 Append statistics
                            # stats: correct, conf, pcls, tcls   bs个 correct, conf, pcls, tcls
                            # correct: [pred_num, 10] bool 判断当前图片的每一个预测框在每一个iou条件下是否是TP
                            # pred[:, 4]: [pred_num, 1] 当前图片每一个预测框的conf
                            # pred[:, 5]: [pred_num, 1] 当前图片每一个预测框的类别
                            # tcls: [gt_num, 1] 当前图片所有gt框的class
                            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
```



在val.py中，其实会定义一个ConfusionMatrix类，用于绘制混淆矩阵。具体定义如下：



```python
class ConfusionMatrix:
   
    def __init__(self, nc, conf=0.25, iou_thres=0.45): 

        # 初始化混淆矩阵 pred x gt  其中横坐标/纵坐标第81类为背景类
        # 如果某个gt[j]没用任何pred正样本匹配到 那么[nc, gt[j]_class] += 1
        # 如果某个pred[i]负样本且没有哪个gt与之对应 那么[pred[i]_class nc] += 1
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        """
        :params detections: [N, 6] = [pred_obj_num, x1y1x2y2+object_conf+cls] = [300, 6]
                            一个batch中一张图的预测信息  其中x1y1x2y2是映射到原图img的
        :params labels: [M, 5] = [gt_num, class+x1y1x2y2] = [17, 5] 其中x1y1x2y2是映射到原图img的
        :return: None, updates confusion matrix accordingly
        """
        # 筛除置信度过低的预测框(和nms差不多)  [10, 6]
        detections = detections[detections[:, 4] > self.conf]

        gt_classes = labels[:, 0].int()  # 所有gt框类别(int) [17]  类别可能会重复
        detection_classes = detections[:, 5].int()  # 所有pred框类别(int) [10] 类别可能会重复  Positive + Negative

        # 求出所有gt框和所有pred框的iou [17, x1y1x2y2] + [10, x1y1x2y2] => [17, 10] [i, j] 第i个gt框和第j个pred的iou
        iou = box_iou(labels[:, 1:], detections[:, :4])

        # iou > self.iou_thres: [17, 10] bool 符合条件True 不符合False
        # x[0]: [10] gt_index  x[1]: [10] pred_index   x合起来看就是第x[0]个gt框和第x[1]个pred的iou符合条件
        # 17 x 10个iou 经过iou阈值筛选后只有10个满足iou阈值条件
        x = torch.where(iou > self.iou_thres)

        # 后面会专门对这里一连串的matches变化给个实例再解释
        if x[0].shape[0]:   # 存在大于阈值的iou时
            # torch.stack(x, 1): [10, gt_index+pred_index]
            # iou[x[0], x[1]][:, None]): [10, 1]   x[0]和x[1]的iou
            # 1、matches: [10, gt_index+pred_index+iou] = [10, 3]
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                # 2、matches按第三列iou从大到小重排序
                matches = matches[matches[:, 2].argsort()[::-1]]
                # 3、取第二列中各个框首次出现(不同预测的框)的行(即每一种预测的框中iou最大的那个)
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # 4、matches再按第三列iou从大到小重排序
                matches = matches[matches[:, 2].argsort()[::-1]]
                # 5、取第一列中各个框首次出现(不同gt的框)的行(即每一种gt框中iou最大的那个)
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]  # [9, gt_index+pred_index+iou]
                # 经过这样的处理 最终得到每一种预测框与所有gt框中iou最大的那个(在大于阈值的前提下)
                # 预测框唯一  gt框也唯一  这样得到的matches对应的Pred都是正样本Positive  9个
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0  # 满足条件的iou是否大于0个 bool
        # a.transpose(): 转换维度 对二维数组就是转置 这里的matches: [9, gt_index+pred_index+iou] -> [gt_index+pred_index+iou, 17]
        # m0: [1, 9] 满足条件(正样本)的gt框index(不重复)  m1: [1, 9] 满足条件(正样本)的pred框index(不重复)
        m0, m1, _ = matches.transpose().astype(np.int16)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                # 如果sum(j)=1 说明gt[i]这个真实框被某个预测框检测到了 但是detection_classes[m1[j]]并不一定等于gc 所以此时可能是TP或者是FP
                # m1[j]: gt框index=i时, 满足条件的pred框index  detection_classes[m1[j]]: pred_class_index
                # gc: gt_class_index    matrix[pred_class_index,gt_class_index] += 1
                self.matrix[detection_classes[m1[j]], gc] += 1  # TP + FP  某个gt检测到了(Positive IOU ≥ threshold) 但是有可能分类分错了 也有可能分类分对了
            else:
                # 如果sum(j)=0 说明gt[i]这个真实框没用被任何预测框检测到 也就是说这个真实框被检测成了背景框
                # 所以对应的混淆矩阵 [背景类, gc] += 1   其中横坐标第81类是背景background  IOU < threshold
                self.matrix[self.nc, gc] += 1  # background FP +1    某个gt没检测到 被检测为background了

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    # detection_classes - matrix[1] = negative  且没用对应的gt和negative相对应 所以background FN+1
                    self.matrix[dc, self.nc] += 1  # background FN

    def matrix(self):
        # 返回这个混淆矩阵
        return self.matrix

    def plot(self, normalize=True, save_dir='', names=()):
 
        try:
            import seaborn as sn  # seaborn 为matplotlib可视化更好看的一个模块

            array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1E-6) if normalize else 1)  # 混淆矩阵归一化 0~1
            array[array < 0.005] = np.nan  # 混淆矩阵中小于0.005的值被认为NaN

            fig = plt.figure(figsize=(12, 9), tight_layout=True)  # 初始化画布
            sn.set(font_scale=1.0 if self.nc < 50 else 0.8)  # 设置label的字体大小
            labels = (0 < len(names) < 99) and len(names) == self.nc  # 绘制混淆矩阵时 是否使用names作为labels

            # 绘制热力图 即混淆矩阵可视化
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
                # sean.heatmap: 热力图  data: 数据矩阵  annot: 为True时为每个单元格写入数据值 False用颜色深浅表示
                # annot_kws: 格子外框宽度  fmt: 添加注释时要使用的字符串格式代码 cmap: 指色彩颜色的选择
                # square: 是否是正方形  xticklabels、yticklabels: xy标签
                sn.heatmap(array, annot=self.nc < 30, annot_kws={"size": 8}, cmap='Blues', fmt='.2f', square=True,
                           xticklabels=names + ['background FP'] if labels else "auto",
                           yticklabels=names + ['background FN'] if labels else "auto").set_facecolor((1, 1, 1))
            # 设置figure的横坐标 纵坐标及保存该图片
            fig.axes[0].set_xlabel('True')
            fig.axes[0].set_ylabel('Predicted')
            fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
        except Exception as e:
            print(f'WARNING: ConfusionMatrix plot failure: {e}')

    def print(self):
        # print按行输出打印混淆矩阵matrix
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i])))
```



下面就会保存评估的结果图片，以及打印评估的结果。



```python
        # 画出前三个batch的图片的ground truth和预测框predictions，两张图一起保存下来
        if plots and batch_i < 3:
            # ground truth
            f = save_dir / f'test_batch{batch_i}_labels.jpg'
            Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()   # 多线程下画图
            f = save_dir / f'test_batch{batch_i}_pred.jpg'
            Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()

    # 统计stats中的结果 将stats列表的信息拼接到一起来即算mAP   list{4} correct, conf, pcls, tcls  统计出的整个数据集的GT

    # correct [img_sum, 10] 整个数据集所有图片中所有预测框在每一个iou条件下是否是TP  [1905, 10]
    # conf [img_sum] 整个数据集所有图片中所有预测框的conf  [1905]
    # pcls [img_sum] 整个数据集所有图片中所有预测框的类别   [1905]
    # tcls [gt_sum] 整个数据集所有图片所有gt框的class     [929]
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy

    # stats[0].any(): stats[0]是否全部为False, 是则返回 False, 如果有一个为 True, 则返回 True
    if len(stats) and stats[0].any():
        # 根据上面的统计预测结果计算p, r, ap, f1, ap_class（ap_per_class函数是计算每个类的mAP等指标的）等指标
        # p: [nc] 最大平均f1时每个类别的precision
        # r: [nc] 最大平均f1时每个类别的recall
        # ap: [71, 10] 数据集每个类别在10个iou阈值下的mAP
        # f1 [nc] 最大平均f1时每个类别的f1
        # ap_class: [nc] 返回数据集中所有的类别index
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        # ap50: [nc] 所有类别的mAP@0.5   ap: [nc] 所有类别的mAP@0.5:0.95
        ap50, ap = ap[:, 0], ap.mean(1)
        # mp: [1] 所有类别的平均precision(最大f1时)
        # mr: [1] 所有类别的平均recall(最大f1时)
        # map50: [1] 所有类别的平均mAP@0.5
        # map: [1] 所有类别的平均mAP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        # nt: [nc] 统计出整个数据集的gt框中数据集各个类别的个数
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    # 打印每个类别下的各个指标  类别 + 数据集图片数量 + 这个类别的gt框数量 + 这个类别的precision +
    #                        这个类别的recall + 这个类别的mAP@0.5 + 这个类别的mAP@0.5:0.95
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # 打印耗时，包括前向传播耗费的总时间、nms耗费总时间、总时间
    t = tuple(x / seen * 1E3 for x in (t0, t1, t2))  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)
```



上述有个ap_per_class函数，是用于计算每个类的mAP。最后还会保存pr曲线、f1曲线、P_conf曲线、R_conf这个四条曲线的图片下来。



```python
def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=()):
    """
    计算每一个类的AP指标(average precision)还可以 绘制P-R曲线
    mAP基本概念: https://www.bilibili.com/video/BV1ez4y1X7g2
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics
    :params tp(correct): [pred_sum, 10]=[1905, 10] bool 整个数据集所有图片中所有预测框在每一个iou条件下(0.5~0.95)10个是否是TP
    :params conf: [img_sum]=[1905] 整个数据集所有图片的所有预测框的conf
    :params pred_cls: [img_sum]=[1905] 整个数据集所有图片的所有预测框的类别
            这里的tp、conf、pred_cls是一一对应的
    :params target_cls: [gt_sum]=[929] 整个数据集所有图片的所有gt框的class
    :params plot: bool
    :params save_dir: runs\train\exp30
    :params names: dict{key(class_index):value(class_name)} 获取数据集所有类别的index和对应类名
    :return p[:, i]: [nc] 最大平均f1时每个类别的precision
    :return r[:, i]: [nc] 最大平均f1时每个类别的recall
    :return ap: [71, 10] 数据集每个类别在10个iou阈值下的mAP
    :return f1[:, i]: [nc] 最大平均f1时每个类别的f1
    :return unique_classes.astype('int32'): [nc] 返回数据集中所有的类别index
    """
    # 计算mAP 需要将tp按照conf降序排列
    # Sort by objectness  按conf从大到小排序 返回数据对应的索引
    i = np.argsort(-conf)
    # 得到重新排序后对应的 tp, conf, pre_cls
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes  对类别去重, 因为计算ap是对每类进行
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # 数据集类别数 number of classes

    # Create Precision-Recall curve and compute AP for each class
    # px: [0, 1] 中间间隔1000个点 x坐标(用于绘制P-Conf、R-Conf、F1-Conf)
    # py: y坐标[] 用于绘制IOU=0.5时的PR曲线
    px, py = np.linspace(0, 1, 1000), []  # for plotting

    # 初始化 对每一个类别在每一个IOU阈值下 计算AP P R   ap=[nc, 10]  p=[nc, 1000] r=[nc, 1000]
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):  # ci: index 0   c: class 0  unique_classes: 所有gt中不重复的class
        # i: 记录着所有预测框是否是c类别框   是c类对应位置为True, 否则为False
        i = pred_cls == c
        # n_l: gt框中的c类别框数量  = tp+fn   254
        n_l = (target_cls == c).sum()  # number of labels
        # n_p: 预测框中c类别的框数量   695
        n_p = i.sum()  # number of predictions

        # 如果没有预测到 或者 ground truth没有标注 则略过类别c
        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs(False Positive) and TPs(Ture Positive)   FP + TP = all_detections
            # tp[i] 可以根据i中的的True/False觉定是否删除这个数  所有tp中属于类c的预测框
            #       如: tp=[0,1,0,1] i=[True,False,False,True] b=tp[i]  => b=[0,1]
            # a.cumsum(0)  会按照对象进行累加操作
            # 一维按行累加如: a=[0,1,0,1]  b = a.cumsum(0) => b=[0,1,1,2]   而二维则按列累加
            # fpc: 类别为c 顺序按置信度排列 截至到每一个预测框的各个iou阈值下FP个数 最后一行表示c类在该iou阈值下所有FP数
            # tpc: 类别为c 顺序按置信度排列 截至到每一个预测框的各个iou阈值下TP个数 最后一行表示c类在该iou阈值下所有TP数
            fpc = (1 - tp[i]).cumsum(0)  # fp[i] = 1 - tp[i]
            tpc = tp[i].cumsum(0)

            # Recall=TP/(TP+FN)  加一个1e-16的目的是防止分母为0
            # n_l=TP+FN=num_gt: c类的gt个数=预测是c类而且预测正确+预测不是c类但是预测错误
            # recall: 类别为c 顺序按置信度排列 截至每一个预测框的各个iou阈值下的召回率
            recall = tpc / (n_l + 1e-16)  # recall curve  用于计算mAP
            # 返回所有类别, 横坐标为conf(值为px=[0, 1, 1000] 0~1 1000个点)对应的recall值  r=[nc, 1000]  每一行从小到大
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # 用于绘制R-Confidence(R_curve.png)

            # Precision=TP/(TP+FP)
            # precision: 类别为c 顺序按置信度排列 截至每一个预测框的各个iou阈值下的精确率
            precision = tpc / (tpc + fpc)  # precision curve  用于计算mAP
            # 返回所有类别, 横坐标为conf(值为px=[0, 1, 1000] 0~1 1000个点)对应的precision值  p=[nc, 1000]
            # 总体上是从小到大 但是细节上有点起伏 如: 0.91503 0.91558 0.90968 0.91026 0.90446 0.90506
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # 用于绘制P-Confidence(P_curve.png)

            # AP from recall-precision curve
            # 对c类别, 分别计算每一个iou阈值(0.5~0.95 10个)下的mAP
            for j in range(tp.shape[1]):  # tp [pred_sum, 10]
                # 这里执行10次计算ci这个类别在所有mAP阈值下的平均mAP  ap[nc, 10]
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # py: 用于绘制每一个类别IOU=0.5时的PR曲线

    # 计算F1分数 P和R的调和平均值  综合评价指标
    # 我们希望的是P和R两个越大越好, 但是P和R常常是两个冲突的变量, 经常是P越大R越小, 或者R越大P越小 所以我们引入F1综合指标
    # 不同任务的重点不一样, 有些任务希望P越大越好, 有些任务希望R越大越好, 有些任务希望两者都大, 这时候就看F1这个综合指标了
    # 返回所有类别, 横坐标为conf(值为px=[0, 1, 1000] 0~1 1000个点)对应的f1值  f1=[nc, 1000]
    f1 = 2 * p * r / (p + r + 1e-16)   # 用于绘制P-Confidence(F1_curve.png)

    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)                # 画pr曲线
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')       # 画F1_conf曲线
        plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')  # 画P_conf曲线
        plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')     # 画R_conf曲线

    # f1=[nc, 1000]   f1.mean(0)=[1000]求出所有类别在x轴每个conf点上的平均f1
    # .argmax(): 求出每个点平均f1中最大的f1对应conf点的index
    i = f1.mean(0).argmax()  # max F1 index

    # p=[nc, 1000] 每个类别在x轴每个conf值对应的precision
    # p[:, i]: [nc] 最大平均f1时每个类别的precision
    # r[:, i]: [nc] 最大平均f1时每个类别的recall
    # f1[:, i]: [nc] 最大平均f1时每个类别的f1
    # ap: [71, 10] 数据集每个类别在10个iou阈值下的mAP
    # unique_classes.astype('int32'): [nc] 返回数据集中所有的类别index
    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')
```



可是，这里是如何给每一个类来计算ap的呢？上述函数中会使用到`compute_ap`来计算：



```python
def compute_ap(recall, precision):
    """
    Compute the average precision, given the recall and precision curves
    :params recall: (list)  在某个iou阈值下某个类别所有的预测框的recall  从小到大
                    (每个预测框的recall都是截至到这个预测框为止的总recall)
    :params precision: (list) [1635] 在某个iou阈值下某个类别所有的预测框的precision
                       总体上是从大到小 但是细节上有点起伏 如: 0.91503 0.91558 0.90968 0.91026 0.90446 0.90506
                       (每个预测框的precision都是截至到这个预测框为止的总precision)
    :ap: Average precision 返回某类别在某个iou下的mAP(均值) [1]
    在开头和末尾添加保护值 防止全零的情况出现 所以最后元素会增加2个
    :mpre: precision curve [1637] 返回 开头 + 输入precision(排序后) + 末尾
    :mrec: recall curve [1637] 返回 开头 + 输入recall + 末尾
    """
    # value Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [recall[-1] + 0.01]))  # [1637]
    mpre = np.concatenate(([1.], precision, [0.]))  # [1637]

    # Compute the precision envelope  np.flip翻转顺序
    # np.flip(mpre): 把一维数组每个元素的顺序进行翻转 第一个翻转成为最后一个
    # np.maximum.accumulate(np.flip(mpre)): 计算数组(或数组的特定轴)的累积最大值 令mpre是单调的 从小到大
    # np.flip(np.maximum.accumulate(np.flip(mpre))): 从大到小
    # 到这大概看明白了这步的目的: 要保证mpre是从大到小单调的(左右可以相同)
    # 我觉得这样可能是为了更好计算mAP 因为如果一直起起伏伏太难算了(x间隔很小就是一个矩形) 而且这样做误差也不会很大 两个之间的数都是间隔很小的
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':  # 用一些典型的间断点来计算AP
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)  [0, 0.01, ..., 1]
        #  np.trapz(list,list) 计算两个list对应点与点之间四边形的面积 以定积分形式估算AP 第一个参数是y 第二个参数是x
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'  # 采用连续的方法计算AP
        # 通过错位的方式 判断哪个点当前位置到下一个位置值发生改变 并通过！=判断 返回一个布尔数组
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        # 值改变了就求出当前矩阵的面积  值没变就说明当前矩阵和下一个矩阵的高相等所有可以合并计算
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec
```



此外，还会根据你的设置判断你是否要保存到wandb上，以及保存为coco格式的json文件。



```python
if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        if wandb_logger and wandb_logger.wandb:
            val_batches = [wandb_logger.wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]
            wandb_logger.log({"Validation": val_batches})
    if wandb_images:
        wandb_logger.log({"Bounding Box Debugger/Images": wandb_images})

    # 保存coco格式的json文件
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')  # annotations json
        # 获取预测框的json文件路径并打开
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            # 评估coco数据集
            check_requirements(['pycocotools'])
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            # 获取并初始化测试集标签的json文件
            anno = COCO(anno_json)  # init annotations api
            # 初始化预测框的文件
            pred = anno.loadRes(pred_json)  # init predictions api
            # 创建评估器
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            # 评估
            eval.evaluate()
            eval.accumulate()
            # 展示结果
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print(f'pycocotools unable to run: {e}')
```



至此，评估的代码也差不多结束了。下面的内容就是为了训练的过程而设，最后返回一些值给训练过程。



```python
    model.float()  # for training

    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    maps = np.zeros(nc) + map  # [80] 80个类别的平均mAP@0.5:0.95
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]  # maps [80] 所有类别的mAP@0.5:0.95
    # (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()): {tuple:7}
    #      0: mp [1] 所有类别的平均precision(最大f1时)
    #      1: mr [1] 所有类别的平均recall(最大f1时)
    #      2: map50 [1] 所有类别的平均mAP@0.5
    #      3: map [1] 所有类别的平均mAP@0.5:0.95
    #      4: val_box_loss [1] 验证集回归损失
    #      5: val_obj_loss [1] 验证集置信度损失
    #      6: val_cls_loss [1] 验证集分类损失
    # maps: [80] 所有类别的mAP@0.5:0.95
    # t: {tuple: 3} 0: 打印前向传播耗费的总时间   1: nms耗费总时间   2: 总时间
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t
```



## 文末



以上就是对val.py的解读，代码量比train.py少了很多，并且有一部分的逻辑也和train.py相似。整体来看，并没有太多复杂的函数，因此这里更多的是了解整体运行的逻辑。

