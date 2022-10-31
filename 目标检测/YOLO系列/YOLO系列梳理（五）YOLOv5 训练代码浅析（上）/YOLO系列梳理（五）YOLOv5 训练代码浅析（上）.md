## **前言**



因为 YOLOv5 主要是面向工程进行设计，所以它的代码写的十分不错。今天笔者就来对 YOLOv5 的部分训练代码进行解读。大家可以学习这种代码风格，来提高代码的可读性。



## **代码解读**



这里有两种方式进行训练，流程都一致，同样是先解析配置，然后进入到 main 函数中，根据配置进行训练。



```python
if __name__ == "__main__":
    opt = parse_opt()	
    main(opt)
```



run函数的逻辑为：



```python
def run(**kwargs):
    opt = parse_opt(True)	# 解析配置，True代表参数known为True，作用就是当仅获取到基本设置时，如果运行命令传入了之后才会获取到其他的配置，不会报错，而是将多余的部分保存起来，留到后面使用。默认为False
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)	# 执行main函数
```



解析配置参数：



```python
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    # --------------------------------------------------- 常用参数 ---------------------------------------------
    parser.add_argument('--weights', type=str, default='weights/yolov5s.pt', help='initial weights path')   # weights: 权重文件
    parser.add_argument('--cfg', type=str, default='models/yolov5s.yaml', help='model.yaml path')   # cfg: 模型配置文件 包括nc、depth_multiple、width_multiple、anchors、backbone、head等
    parser.add_argument('--data', type=str, default='data/VOC.yaml', help='dataset.yaml path')  # data: 数据集配置文件 包括path、train、val、test、nc、names、download等
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch.yaml', help='hyperparameters path')   #  hyp: 初始超参文件
    parser.add_argument('--epochs', type=int, default=20)   # epochs: 训练轮次
    parser.add_argument('--batch-size', type=int, default=4, help='total batch size for all GPUs') # batch-size: 训练批次大小
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')    # img-size: 输入网络的图片分辨率大小
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')   #  resume: 断点续训, 从上次打断的训练结果处接着训练  默认False
    parser.add_argument('--nosave', action='store_true', help='True only save final checkpoint')    # nosave: 不保存模型  默认False(保存)      True: only test final epoch
    parser.add_argument('--notest', action='store_true', help='True only test final epoch')     #  notest: 是否只测试最后一轮 默认False  True: 只测试最后一轮   False: 每轮训练完都测试mAP
    parser.add_argument('--workers', type=int, default=0, help='maximum number of dataloader workers')      #  workers: dataloader中的最大work数（线程个数）
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')       #  device: 训练的设备
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')     #  single-cls: 数据集是否只有一个类别 默认False
    # --------------------------------------------------- 数据增强参数 ---------------------------------------------
    parser.add_argument('--rect', action='store_true', help='rectangular training')     # rect: 训练集是否采用矩形训练  默认False
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')     #   noautoanchor: 不自动调整anchor 默认False(自动调整anchor)
    parser.add_argument('--evolve', default=False, action='store_true', help='evolve hyperparameters')  # evolve: 是否进行超参进化 默认False
    parser.add_argument('--multi-scale', default=True, action='store_true', help='vary img-size +/- 50%%')      #  multi-scale: 是否使用多尺度训练 默认False
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')       #  label-smoothing: 标签平滑增强 默认0.0不增强  要增强一般就设为0.1
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')     #   adam: 是否使用adam优化器 默认False(使用SGD)
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')     #  sync-bn: 是否使用跨卡同步bn操作,再DDP中使用  默认False
    parser.add_argument('--linear-lr', default=False, action='store_true', help='linear LR')        # linear-lr: 是否使用linear lr  线性学习率  默认False 使用cosine lr
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')     #  cache-image: 是否提前缓存图片到内存cache,以加速训练  默认False
    parser.add_argument('--image-weights', default=True, action='store_true', help='use weighted image selection for training')     #  image-weights: 是否使用图片采用策略(selection img to training by class weights) 默认False 不使用
    # --------------------------------------------------- 其他参数 ---------------------------------------------
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')     #  bucket: 谷歌云盘bucket 一般用不到
    parser.add_argument('--project', default='runs/train', help='save to project/name')     #  project: 训练结果保存的根目录 默认是runs/train
    parser.add_argument('--name', default='exp', help='save to project/name')       #  name: 训练结果保存的目录 默认是exp  最终: runs/train/exp
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')       #  exist-ok: 如果文件存在就ok不存在就新建或increment name  默认False(默认文件都是不存在的)
    parser.add_argument('--quad', action='store_true', help='quad dataloader')      #  quad: dataloader取数据时, 是否使用collate_fn4代替collate_fn  默认False
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')        #  save_period: Log model after every "save_period" epoch    默认-1 不需要log model 信息
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')      #  artifact_alias: which version of dataset artifact to be stripped  默认lastest  貌似没用到这个参数？
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, wins do not modify')     #  local_rank: rank为进程编号  -1且gpu=1时不进行分布式  -1且多块gpu使用DataParallel模式
    # --------------------------------------------------- wandb 参数 ---------------------------------------------
    parser.add_argument('--entity', default=None, help='W&B entity')        # entity: wandb entity 默认None
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')       #  upload_dataset: 是否上传dataset到wandb tabel(将数据集作为交互式 dsviz表 在浏览器中查看、查询、筛选和分析数据集) 默认False
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')        # bbox_interval: 设置界框图像记录间隔 Set bounding-box image logging interval for W&B 默认-1   opt.epochs // 10
    # parser.parse_known_args()
    # 作用就是当仅获取到基本设置时，如果运行命令中传入了之后才会获取到的其他配置，不会报错；而是将多出来的部分保存起来，留到后面使用
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt
```



进入到 main 函数，这里使用到遗传进化，主要是为了自动调参，使用最佳的参数进行训练。

```python
def main(opt):

    set_logging(RANK)
    if RANK in [-1, 0]:
        print(colorstr('train: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
        # 检查代码版本
        check_git_status()
        check_requirements(exclude=['thop'])

    wandb_run = check_wandb_resume(opt)

    # 2、判断是否断点重新训练
    if opt.resume and not wandb_run:
        # 使用断点续训
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # 如果resume是str，则载入模型路径，否则根据runs找last.pt
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist' # check

        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.safe_load(f))  # replace
        opt.cfg, opt.weights, opt.resume = '', ckpt, True  # reinstate
        logger.info('Resuming training from %s' % ckpt)    # print
    else:
        # 不使用断点续训 读取文件中的参数 给到后面的训练
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'

        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))   # 将opt.img_size扩展为[train_img_size, test_img_size]
        # opt.evolve=False,opt.name='exp'    opt.evolve=True,opt.name='evolve'
        opt.name = 'evolve' if opt.evolve else opt.name
        # 生成保存的目录
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve))

    # 3、DDP mode设置
    # 选择设备  cpu/cuda:0
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        # 当LOCAL_RANK != -1 表示进行多GPU训练
        from datetime import timedelta
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        # 初始化进程组  distributed backend
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo", timeout=timedelta(seconds=60))
        assert opt.batch_size % WORLD_SIZE == 0, '--batch-size must be multiple of CUDA device count'
        assert not opt.image_weights, '--image-weights argument is not compatible with DDP training'

    # 不使用超参数进化算法 正常训练
    if not opt.evolve:
        train(opt.hyp, opt, device)
        # 如果是使用多卡训练, 那么销毁进程组
        if WORLD_SIZE > 1 and RANK == 0:
            _ = [print('Destroying process group... ', end=''), dist.destroy_process_group(), print('Done.')]

    else:
        # 使用遗传进化，边训练边进化来优化超参数，求出最佳超参，再进行训练
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        # 超参进化列表 (突变规模, 最小值, 最大值)
        meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
                'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
                'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
                'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                'box': (1, 0.02, 0.2),  # box loss gain
                'cls': (1, 0.2, 4.0),  # cls loss gain
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
                'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
                'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
                'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
                'mixup': (1, 0.0, 1.0)}  # image mixup (probability)

        with open(opt.hyp) as f:
            hyp = yaml.safe_load(f)  # 载入初始超参
        assert LOCAL_RANK == -1, 'DDP mode not implemented for --evolve'
        opt.notest, opt.nosave = True, True  # only test/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        yaml_file = Path(opt.save_dir) / 'hyp_evolved.yaml'  # 超参进化后文件保存文件
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists

        """
        使用遗传算法进行参数进化 默认是进化300代
        这里的进化算法是：根据之前训练时的hyp来确定一个base hyp再进行突变；
        如何根据？通过之前每次进化得到的results来确定之前每个hyp的权重
        有了每个hyp和每个hyp的权重之后有两种进化方式；
        1.根据每个hyp的权重随机选择一个之前的hyp作为base hyp，random.choices(range(n), weights=w)
        2.根据每个hyp的权重对之前所有的hyp进行融合获得一个base hyp，(x * w.reshape(n, 1)).sum(0) / w.sum()
        evolve.txt会记录每次进化之后的results+hyp
        每次进化时，hyp会根据之前的results进行从大到小的排序；
        再根据fitness函数计算之前每次进化得到的hyp的权重
        再确定哪一种进化方式，从而进行进化
        """
        for _ in range(300):  # generations to evolve
            if Path('evolve.txt').exists():  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                # 选择超参进化方式 只用single和weighted两种方式
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                # 加载evolve.txt
                x = np.loadtxt('evolve.txt', ndmin=2)
                # 选取至多前五次进化的结果
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                # 根据 resluts计算超参的权重
                w = fitness(x) - fitness(x).min() + 1E-6  # weights (sum > 0)
                # 根据不同进化方式获得base hyp
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # 根据突变概率进行超参进化
                mp, s = 0.8, 0.2  # mutation probability 突变概率, sigma
                npr = np.random
                npr.seed(int(time.time()))
                # 获取突变初始值
                g = np.array([x[0] for x in meta.values()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                # 设置突变
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                # 将突变添加到base hyp上
                # [i+7]是因为x中前7个数字为results的指标(P,R,mAP,F1,test_loss=(box,obj,cls)),之后才是超参数hyp
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits  限制超参再规定范围
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # 使用突变后的超参进行突变
            results = train(hyp.copy(), opt, device)

            # 写入结果，并保存hyp
            print_mutation(hyp.copy(), results, yaml_file, opt.bucket)

        # Plot results
        plot_evolution(yaml_file, Path(opt.save_dir))
        print(f'Hyperparameter evolution complete. Best results saved as: {yaml_file}\n'
              f'Command to train a new model with these hyperparameters: $ python train.py --hyp {yaml_file}')

```



train 函数的逻辑如下，代码较长，但是笔者在其中做了注释：



```python
def train(hyp, opt, device):
    init_seeds(1 + RANK)

    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, notest, nosave, workers, = \
        opt.save_dir, opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.notest, opt.nosave, opt.workers

    save_dir = Path(save_dir)   # 设置保存的路径以及权重名称
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'

    if isinstance(hyp, str):
        with open(hyp, encoding='utf-8') as f:
            hyp = yaml.safe_load(f)   # 加载Hyperparameters超参

    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))  # 日志输出超参信息

    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)

    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)

    plots = not evolve
    cuda = device.type != 'cpu'

    # 加载配置信息，如data里的VOC.yaml
    with open(data) as f:
        data_dict = yaml.safe_load(f)

    loggers = {'wandb': None, 'tb': None}
    if RANK in [-1, 0]:
        if not evolve:
            prefix = colorstr('tensorboard: ')  # 彩色打印信息
            logger.info(f"{prefix}Start with 'tensorboard --logdir {opt.project}', view at http://localhost:6006/")
            loggers['tb'] = SummaryWriter(str(save_dir))

        opt.hyp = hyp
        run_id = torch.load(weights).get('wandb_id') if weights.endswith('.pt') and os.path.isfile(weights) else None
        run_id = run_id if opt.resume else None  # 重新训练或者使用迁移学习加载权重
        wandb_logger = WandbLogger(opt, save_dir.stem, run_id, data_dict)
        loggers['wandb'] = wandb_logger.wandb
        if loggers['wandb']:
            data_dict = wandb_logger.data_dict
            weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp  # may update weights, epochs if resuming

    nc = 1 if single_cls else int(data_dict['nc'])

    names = ['item'] if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # 数据集所有类别的名字
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, data)  # check

    is_coco = data.endswith('coco.yaml') and nc == 80  # COCO dataset

    pretrained = weights.endswith('.pt')
    if pretrained:
        # torch_distributed_zero_first(RANK): 用于同步不同进程对数据读取的上下文管理器
        with torch_distributed_zero_first(RANK):
            weights = attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
        state_dict = ckpt['model'].float().state_dict()  # to FP32

        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)   # 筛选字典中的键值对  把exclude删除
        model.load_state_dict(state_dict, strict=False)     # strict 为True的话就要权重名称和模型名称完全匹配
        logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))
    else:
        #
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # creat

    with torch_distributed_zero_first(RANK):
        check_dataset(data_dict)

    train_path = data_dict['train']
    test_path = data_dict['val']

    freeze = []  # parameter names to freeze (full or partial)
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False     # 冻结权重层，也就是训练部分层，虽然训练会更快但是有可能效果变差

    # nbs 标称的batch_size,模拟的batch_size 比如默认的话上面设置的opt.batch_size=16 -> nbs=64
    # 也就是
    nbs = 64  # bns是模拟的batch_size意思，然后根据实际的batch_size，如16，来进行模型梯度累计，
              # 比如64/16=4次之后来对模型进行更新，变相扩大了batch_size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing

    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    # 将模型参数分为三组(weights、biases、bn)来进行分组优化
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay

    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))  # 日志
    del pg0, pg1, pg2

    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    plot_lr_scheduler(optimizer, scheduler, epochs, save_dir=save_dir)  # 根据学习率的变化进行绘图

    # EMA
    # 单卡训练: 一种给予近期数据更高权重的平均方法, 以求提高测试指标并增加模型鲁棒。
    ema = ModelEMA(model) if RANK in [-1, 0] else None  # 使用EMA，即指数移动平均对模型的参数做平均, 相当于赋予权重一个动量，
                                                        # 对于近期的数据赋予更高的权重，最后进行平均，来提高鲁棒性

    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # EMA
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']

        # Results
        if ckpt.get('training_results') is not None:
            results_file.write_text(ckpt['training_results'])  # write results.txt

        # Epochs
        start_epoch = ckpt['epoch'] + 1
        if resume:
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)
        if epochs < start_epoch:
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt, state_dict

    gs = max(int(model.stride.max()), 32)  # 获取模型最大stride=32   [32 16 8]
    nl = model.model[-1].nl  # nl = number of layer 有多少个检测层，默认为3
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # 检查图片的分辨率并获取分辨率

    # 使用多卡训练模式，
    # 如果rank=-1且gpu数量>1 就会使用DataParallel单机多卡模式  但是分布不平均
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        logging.warning('DP not recommended, instead use torch.distributed.run for best DDP Multi-GPU results.\n'
                        'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
        model = torch.nn.DataParallel(model)

    # 使用多卡训练模式，
    # 如果rank !=-1, 则使用DistributedDataParallel模式 ，可以实现真正的分步平均，真正意义上的单机单卡
    if cuda and RANK != -1:
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

    # S如果机器支持，可以使用跨卡BN
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')

    # Trainloader
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size // WORLD_SIZE, gs, single_cls,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect,
                                            rank=RANK, workers=workers, image_weights=opt.image_weights,
                                            quad=opt.quad, prefix=colorstr('train: '))

    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # 检查标签的最大值，再与类别数作比较，看是否有问题
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, data, nc - 1)
    nb = len(dataloader)  # number of batches

    # TestLoader
    if RANK in [-1, 0]:
        testloader = create_dataloader(test_path, imgsz_test, batch_size // WORLD_SIZE * 2, gs, single_cls,
                                       hyp=hyp, cache=opt.cache_images and not notest, rect=True, rank=-1,
                                       workers=workers, pad=0.5, prefix=colorstr('val: '))[0]

        # 不断点 继续训练
        if not resume:
            labels = np.concatenate(dataset.labels, 0)  # 统计数据集中的label
            # 将labels从nparray转为tensor格式
            c = torch.tensor(labels[:, 0])
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # model._initialize_biases(cf.to(device))
            if plots:

                plot_labels(labels, names, save_dir, loggers)
                if loggers['tb']:
                    loggers['tb'].add_histogram('classes', c, 0)

            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)   # 检查anchor
                # 标签的高h宽w与anchor的高h_a宽h_b的比值 即h/h_a, w/w_a都要在(1/hyp['anchor_t'], hyp['anchor_t'])是可以接受的
                # 如果bpr小于98%，则根据k-mean算法聚类新的锚框
            model.half().float()  # pre-reduce anchor precision

    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # 分类损失系数
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou) 用于loss计算
    # 根据类别占比计算类别权重
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names  # 获取类别名称

    t0 = time.time()
    # 获取热身迭代的次数  # number of warmup iterations, max(3 epochs, 1k iterations)
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1
    maps = np.zeros(nc)  # 给每一个类别初始化map
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)

    scheduler.last_epoch = start_epoch - 1  # do not move
    # 设置amp混合精度训练，即使用GradScaler + autocast
    scaler = amp.GradScaler(enabled=cuda)
    compute_loss = ComputeLoss(model)  # init loss class
    logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                f'Using {dataloader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting training for {epochs} epochs...')

    for epoch in range(start_epoch, epochs):   # epoch
        model.train()
        # 如果为True 进行图片采样策略(按数据集各类别权重采样)
        if opt.image_weights:
            # Generate indices
            if RANK in [-1, 0]:
                # 从数据集中获得每个类的权重，其中标签频率高的类权重低
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc
                # 得到每一张图片对应的采样权重[128]
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)
                # random.choices: 从range(dataset.n)序列中按照weights(参考每张图片采样权重)进行采样, 一次取一个数字  采样次数为k
                # 最终得到所有图片的采样顺序(参考每张图片采样权重) list [128]
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
            # Broadcast if DDP  采用广播采样策略
            if RANK != -1:
                indices = (torch.tensor(dataset.indices) if RANK == 0 else torch.zeros(dataset.n)).int()
                dist.broadcast(indices, 0)
                if RANK != 0:
                    dataset.indices = indices.cpu().numpy()

        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        # 初始化训练时打印的平均损失信息
        mloss = torch.zeros(4, device=device)  # mean losses

        if RANK != -1:
            # DDP模式打乱数据，并且dpp.sampler的随机采样数据是基于epoch+seed作为随机种子，每次epoch不同，随机种子不同
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'labels', 'img_size'))
        if RANK in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # 进度条

        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:   # batch
            ni = i + nb * epoch  # 当前的迭代次数
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            # 选取较小的学习率进行Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    # bias的学习率从0.1下降到基准学习率lr*lf(epoch) 其他的参数学习率增加到lr*lf(epoch)
                    # lf为上面设置的余弦退火的衰减函数
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            if opt.multi_scale: # Multi-scale 多尺度训练   从[imgsz*0.5, imgsz*1.5+gs]间随机选取一个尺寸(32的倍数)作为当前batch的尺寸进行训练
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # gs为模型最大stride=32   [32 16 8]
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    # 进行图像下采样
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # 混合精度训练
            with amp.autocast(enabled=cuda):
                # pred: [8, 3, 68, 68, 25] [8, 3, 34, 34, 25] [8, 3, 17, 17, 25]
                # [bs, anchor_num, grid_w, grid_h, xywh+c+20classes]
                pred = model(imgs)  # forward
                # 计算损失，包括分类损失，置信度损失和框的回归损失
                # loss是总损失  loss_items是一个元组，包括分类损失、置信度损失、框的回归损失以及总损失
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                if RANK != -1:
                    # 采用DDP训练的话会平均不同gpu之间的梯度
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                if opt.quad:
                    # 如果采用collate_fn4取出mosaic4数据  loss也要乘上4
                    loss *= 4.

            # 反向传播，会将梯度放大，amp混合精度训练会使用到
            scaler.scale(loss).backward()
            # 模型反向传播accumulate次后 才根据累计的梯度更新一次参数
            if ni - last_opt_step >= accumulate:
                # 首先调用scaler.step。如果梯度的值不是 inf 或者 NaN, 那么调用optimizer.step()来更新权重,
                # 否则，忽略step调用，从而保证权重不更新（不被破坏）
                scaler.step(optimizer)  # optimizer.step参数更新，并且把梯度的值unscale回来

                scaler.update()  # 先准备，看是否要增大scaler
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            # 打印epcoh、loss、显存等信息
            if RANK in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 6) % (
                    f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)

                # Plot  将前三次迭代的batch的标签框在图片中画出来并保存  train_batch0/1/2.jpg
                if plots and ni < 3:
                    f = save_dir / f'train_batch{ni}.jpg'  # filename
                    Thread(target=plot_images, args=(imgs, targets, paths, f), daemon=True).start()
                    if loggers['tb'] and ni == 0:  # TensorBoard
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')  # suppress jit trace warning
                            loggers['tb'].add_graph(torch.jit.trace(de_parallel(model), imgs[0:1], strict=False), [])

                elif plots and ni == 10 and loggers['wandb']:
                    wandb_logger.log({'Mosaics': [loggers['wandb'].Image(str(x), caption=x.name) for x in
                                                  save_dir.glob('train*.jpg') if x.exists()]})

        lr = [x['lr'] for x in optimizer.param_groups]  # group中三个学习率（pg0、pg1、pg2）每个都要调整
        scheduler.step()

        # validation
        # DDP process 0 or single-GPU
        if RANK in [-1, 0]:
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights']) #ema更新模型属性
            final_epoch = epoch + 1 == epochs
            if not notest or final_epoch:
                wandb_logger.current_epoch = epoch + 1
                results, maps, _ = val.run(data_dict,
                                           batch_size=batch_size // WORLD_SIZE * 2,
                                           imgsz=imgsz_test,  # 测试图片的尺寸
                                           model=ema.ema,  # ema model
                                           single_cls=single_cls,  # 是否是单类数据集
                                           dataloader=testloader,
                                           save_dir=save_dir,
                                           save_json=is_coco and final_epoch,  # 是否按照coco的json格式保存预测框
                                           verbose=nc < 50 and final_epoch,  # 是否打印出每个类别的mAP
                                           plots=plots and final_epoch,  # 是否可视化
                                           wandb_logger=wandb_logger,  # 类似tensorboard的网页可视化
                                           compute_loss=compute_loss)  #

            with open(results_file, 'a') as f:
                f.write(s + '%10.4g' * 7 % results + '\n')  # append metrics, val_loss

            tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                    'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                    'x/lr0', 'x/lr1', 'x/lr2']  # params
            for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
                if loggers['tb']:
                    loggers['tb'].add_scalar(tag, x, epoch)  # TensorBoard
                if loggers['wandb']:
                    wandb_logger.log({tag: x})  # W&B

            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi
            wandb_logger.end_epoch(best_result=best_fitness == fi)

            # 除了保存模型权重, 还保存了epoch, results, optimizer等训练信息
            # optimizer将不会在最后一轮完成后保存
            # model保存的是EMA的模型
            if (not nosave) or (final_epoch and not evolve):  # if save
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'training_results': results_file.read_text(),
                        'model': deepcopy(de_parallel(model)).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict(),
                        'wandb_id': wandb_logger.wandb_run.id if loggers['wandb'] else None}

                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if loggers['wandb']:
                    if ((epoch + 1) % opt.save_period == 0 and not final_epoch) and opt.save_period != -1:
                        wandb_logger.log_model(last.parent, opt, epoch, fi, best_model=best_fitness == fi)
                del ckpt

    # 打印信息
    if RANK in [-1, 0]:
        logger.info(f'{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.\n')
        if plots:
            plot_results(save_dir=save_dir)
            plot_results_overlay()
            if loggers['wandb']:
                files = ['results1.png', 'confusion_matrix.png', *[f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R')]]
                wandb_logger.log({"Results": [loggers['wandb'].Image(str(save_dir / f), caption=f) for f in files
                                              if (save_dir / f).exists()]})

        # 根据cooc数据集进行评价
        if not evolve:
            if is_coco:  # COCO dataset
                for m in [last, best] if best.exists() else [last]:  # speed, mAP tests
                    results, _, _ = val.run(data_dict,
                                            batch_size=batch_size // WORLD_SIZE * 2,
                                            imgsz=imgsz_test,
                                            conf_thres=0.001,
                                            iou_thres=0.7,
                                            model=attempt_load(m, device).half(),
                                            single_cls=single_cls,
                                            dataloader=testloader,
                                            save_dir=save_dir,
                                            save_json=True,
                                            plots=False)

            # Strip optimizers
            # 模型训练完后, strip_optimizer函数将optimizer从ckpt中删除
            # 并对模型进行model.half() 将Float32->Float16 这样可以减少模型大小, 提高inference速度
            for f in last, best:
                if f.exists():
                    strip_optimizer(f)  # strip optimizers，最后一轮会删除

            if loggers['wandb']:
                loggers['wandb'].log_artifact(str(best if best.exists() else last), type='model',
                                              name='run_' + wandb_logger.wandb_run.id + '_model',
                                              aliases=['latest', 'best', 'stripped'])
        wandb_logger.finish_run()
    torch.cuda.empty_cache()
    return results
```



## 文末



以上就是笔者对 YOLOv5 训练代码的初步解读，后续还会对训练代码中所引用的函数进行解读，以及还会对检测、评估等 py 脚本进行解读。大家敬请期待~