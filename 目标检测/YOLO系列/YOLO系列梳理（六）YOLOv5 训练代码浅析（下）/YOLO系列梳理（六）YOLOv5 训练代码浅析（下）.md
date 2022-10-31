## 前言



之前梳理了一遍 yolov5 中的训练代码 train.py，但是只是对它的整个流程进行分析，并没有深入到里面的函数。那么今天就来简单介绍一下，我觉得比较重要的几个函数。



## **代码解读**



在 train 函数的开头就有 init_seeds 这个函数。其中 rank 为进程编号，当 rank 为 -1 且 gpu=1 时不进行分布式 -1且多块 gpu 则代表使用 DataParallel 的模式。



```python
def train(hyp, opt, device):
    init_seeds(1 + RANK)
```



具体的init_seeds函数为：



```python
def init_seeds(seed=0):
  """设置一系列的随机种子，保证结果一致，可重现
  """
  # 设置随机数 针对使用random.random()生成随机数的时候相同
  random.seed(seed)
  # 设置随机数 针对使用np.random.rand()生成随机数的时候相同
  np.random.seed(seed)
  # 为CPU设置种子用于生成随机数的时候相同 并确定训练模式
  inittorchseeds(seed)
```



设置完这个之后就是一系列的配置文件，之后还会根据你的训练模式是否为分布式来进行加载模型、检查数据集等操作。下面这个主要是为了进行上下文的管理。



```python
with torch_distributed_zero_first(RANK):
    weights = attempt_download(weights)  # download if not found locally
```



然后让我们一起看看这个函数里面的逻辑。



```python
@contextmanager
def torch_distributed_zero_first(local_rank: int):
  """用在train.py
  用于同步分布式训练，
  是基于torch.distributed.barrier()函数的上下文管理器，为了完成数据的正常同步操作
  Decorator to make all processes in distributed training wait for each local_master to do something.
  :params local_rank: 代表当前进程号 0代表主进程 1、2、3代表子进程
  """
  if local_rank not in [-1, 0]:
    # 如果执行create_dataloader()函数的进程不是主进程，即rank不等于0或者-1，
    # 上下文管理器会执行相应的torch.distributed.barrier()，设置一个阻塞栅栏，
    # 让此进程处于等待状态，等待所有进程到达栅栏处（包括主进程数据处理完毕）；
    dist.barrier()
  yield # yield语句 中断后执行上下文代码，然后返回到此处继续往下执行
  if local_rank == 0:
    # 如果执行create_dataloader()函数的进程是主进程，其会直接去读取数据并处理，
    # 然后其处理结束之后会接着遇到torch.distributed.barrier()，
    # 此时，所有进程都到达了当前的栅栏处，这样所有进程就达到了同步，并同时得到释放。
    dist.barrier()
```



在下面有个加载模型的函数，可以看看怎么生成模型的。思路很简单，基本就是遇到指定的层，就生成指定的层如 ReLu、 BN，然后后面还会带着参数。统一放在一个yaml文件中，就可以更好地进行管理了。



```python
model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # creat
```



但是通过yaml文件来加载的方式还是很巧妙的，通过修改一个yaml文件就可以轻松地配置不同的模型。我们看看这个模型是如何定义的。



```python
class Model(nn.Module):
  def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):
    """
    :params cfg:模型配置文件
    :params ch: input img channels 一般是3 RGB文件
    :params nc: number of classes 数据集的类别个数
    :anchors: 一般是None
    """
    super(Model, self).__init__()
    if isinstance(cfg, dict):
      self.yaml = cfg # model dict
    else:
      # is *.yaml 一般执行这里
      import yaml # for torch hub
      self.yaml_file = Path(cfg).name # cfg file name = yolov5s.yaml
      # 如果配置文件中有中文，打开时要加encoding参数
      with open(cfg, encoding='utf-8') as f:
        # model dict 取到配置文件中每条的信息（没有注释内容）
        self.yaml = yaml.safe_load(f)

    # input channels ch=3
    ch = self.yaml['ch'] = self.yaml.get('ch', ch)
    # 设置类别数 一般不执行, 因为nc=self.yaml['nc']恒成立
    if nc and nc != self.yaml['nc']:
      logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
      self.yaml['nc'] = nc # override yaml value
    # 重写anchor，一般不执行, 因为传进来的anchors一般都是None
    if anchors:
      logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
      self.yaml['anchors'] = round(anchors) # override yaml value

    # 创建网络模型
    # self.model: 初始化的整个网络模型(包括Detect层结构)
    # self.save: 所有层结构中from不等于-1的序号，并排好序 [4, 6, 10, 14, 17, 20, 23]
    self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])

    # default class names ['0', '1', '2',..., '19']
    self.names = [str(i) for i in range(self.yaml['nc'])]

    # self.inplace=True 默认True 不使用加速推理
    # AWS Inferentia Inplace compatiability
    # https://github.com/ultralytics/yolov5/pull/2953
    self.inplace = self.yaml.get('inplace', True)
    # logger.info([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

    # 获取Detect模块的stride(相对输入图像的下采样率)和anchors在当前Detect输出的feature map的尺度
    m = self.model[-1] # Detect()
    if isinstance(m, Detect):
      s = 256 # 2x min stride
      m.inplace = self.inplace
      # 计算三个feature map下采样的倍率 [8, 16, 32]
      m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))]) # forward
      # 求出相对当前feature map的anchor大小 如[10, 13]/8 -> [1.25, 1.625]
      m.anchors /= m.stride.view(-1, 1, 1)
      # 检查anchor顺序与stride顺序是否一致
      check_anchor_order(m)
      self.stride = m.stride
      self._initialize_biases() # only run once 初始化偏置
      # logger.info('Strides: %s' % m.stride.tolist())

    # Init weights, biases
    initialize_weights(self) # 调用torch_utils.py下initialize_weights初始化模型权重
    self.info() # 打印模型信息
    logger.info('')

  def forward(self, x, augment=False, profile=False):
    # augmented inference, None 上下flip/左右flip
    # 是否在测试时也使用数据增强 Test Time Augmentation(TTA)
    if augment:
      return self.forward_augment(x)
    else:
      # 默认执行 正常前向推理
      # single-scale inference, train
      return self.forward_once(x, profile)

  def forward_augment(self, x):
    """
    TTA Test Time Augmentation
    """
    img_size = x.shape[-2:] # height, width
    s = [1, 0.83, 0.67] # scales ratio
    f = [None, 3, None] # flips (2-ud上下flip, 3-lr左右flip)
    y = [] # outputs
    for si, fi in zip(s, f):
      # scale_img缩放图片尺寸
      xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
      yi = self.forward_once(xi)[0] # forward
      # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1]) # save
      # _descale_pred将推理结果恢复到相对原图图片尺寸
      yi = self._descale_pred(yi, fi, si, img_size)
      y.append(yi)
    return torch.cat(y, 1), None # augmented inference, train

  def forward_once(self, x, profile=False, feature_vis=False):
    """
    :params x: 输入图像
    :params profile: True 可以做一些性能评估
    :params feature_vis: True 可以做一些特征可视化
    :return train: 一个tensor list 存放三个元素 [bs, anchor_num, grid_w, grid_h, xywh+c+20classes]
            分别是 [1, 3, 80, 80, 25] [1, 3, 40, 40, 25] [1, 3, 20, 20, 25]
        inference: 0 [1, 19200+4800+1200, 25] = [bs, anchor_num*grid_w*grid_h, xywh+c+20classes]
              1 一个tensor list 存放三个元素 [bs, anchor_num, grid_w, grid_h, xywh+c+20classes]
               [1, 3, 80, 80, 25] [1, 3, 40, 40, 25] [1, 3, 20, 20, 25]
    """
    # y: 存放着self.save=True的每一层的输出，因为后面的层结构concat等操作要用到
    # dt: 在profile中做性能评估时使用
    y, dt = [], []
    for m in self.model:
      # 前向推理每一层结构 m.i=index m.f=from m.type=类名 m.np=number of params
      # if not from previous layer m.f=当前层的输入来自哪一层的输出 s的m.f都是-1
      if m.f != -1:
        # 这里需要做4个concat操作和1个Detect操作
        # concat操作如m.f=[-1, 6] x就有两个元素,一个是上一层的输出,另一个是index=6的层的输出 再送到x=m(x)做concat操作
        # Detect操作m.f=[17, 20, 23] x有三个元素,分别存放第17层第20层第23层的输出 再送到x=m(x)做Detect的forward
        x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f] # from earlier layers

      # 打印日志信息 FLOPs time等
      if profile:
        o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0 # FLOPs
        t = time_synchronized()
        for _ in range(10):
          _ = m(x)
        dt.append((time_synchronized() - t) * 100)
        if m == self.model[0]:
          logger.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s} {'module'}")
        logger.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f} {m.type}')

      x = m(x) # run正向推理 执行每一层的forward函数(除Concat和Detect操作)

      # 存放着self.save的每一层的输出，因为后面需要用来作concat等操作要用到 不在self.save层的输出就为None
      y.append(x if m.i in self.save else None)

      # 特征可视化 可以自己改动想要哪层的特征进行可视化
      if feature_vis and m.type == 'models.common.SPP':
        feature_visualization(x, m.type, m.i)

    # 打印日志信息 前向推理时间
    if profile:
      logger.info('%.1fms total' % sum(dt))
    return x
```



加载完了模型，就使用 create_dataloader 来进行数据集的加载，create_dataloader 中有个 LoadImagesAndLabels 函数，主要是通过这个来获取数据增强后的数据集。在这里会对训练集的数据进行数据增强，并且可以使用开源的数据增强库 albumentations。



```python
class LoadImagesAndLabels(Dataset):
  # for training/testing
  """
  init函数主要是在定义参数，实际作用是在getitem()中
  """
  def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False,
         image_weights=False, cache_images=False, single_cls=False, stride=32, pad=0.0, prefix=''):
    """
    self.img_files: {list: N} 存放着整个数据集图片的相对路径
    self.label_files: {list: N} 存放着整个数据集图片的相对路径
    cache label -> verify_image_label
    self.labels: 如果数据集所有图片中没有一个多边形label labels存储的label就都是原始label(都是正常的矩形label)
           否则将所有图片正常gt的label存入labels 不正常gt(存在一个多边形)经过segments2boxes转换为正常的矩形label
    self.shapes: 所有图片的shape
    self.segments: 如果数据集所有图片中没有一个多边形label self.segments=None
            否则存储数据集中所有存在多边形gt的图片的所有原始label(肯定有多边形label 也可能有矩形正常label 未知数)
    self.batch: 记载着每张图片属于哪个batch
    self.n: 数据集中所有图片的数量
    self.indices: 记载着所有图片的index
    self.rect=True时self.batch_shapes记载每个batch的shape(同一个batch的图片shape相同)
    """
    # 1、赋值一些基础的self变量 用于后面在__getitem__中调用
    self.img_size = img_size # 经过数据增强后的数据图片的大小
    self.augment = augment  # 是否启动数据增强 一般训练时打开 验证时关闭
    self.hyp = hyp      # 超参列表
    # 图片按权重采样 True就可以根据类别频率(频率高的权重小,反正大)来进行采样 默认False: 不作类别区分
    self.image_weights = image_weights
    self.rect = False if image_weights else rect # 是否启动矩形训练 一般训练时关闭 验证时打开 可以加速
    self.mosaic = self.augment and not self.rect # load 4 images at a time into a mosaic (only during training)
    # mosaic增强的边界值 [-320, -320]
    self.mosaic_border = [-img_size // 2, -img_size // 2]
    self.stride = stride   # 最大下采样率 32
    self.path = path     # 图片路径

    # 2、得到path路径下的所有图片的路径self.img_files 这里需要自己debug一下 不会太难
    try:
      f = [] # image files
      for p in path if isinstance(path, list) else [path]:
        # 获取数据集路径path，包含图片路径的txt文件或者包含图片的文件夹路径
        # 使用pathlib.Path生成与操作系统无关的路径，因为不同操作系统路径的‘/’会有所不同
        p = Path(p) # os-agnostic
        # 如果路径path为包含图片的文件夹路径
        if p.is_dir(): # dir
          # glob.glab: 返回所有匹配的文件路径列表 递归获取p路径下所有文件
          f += glob.glob(str(p / '**' / '*.*'), recursive=True)
          # f = list(p.rglob('**/*.*')) # pathlib
        # 如果路径path为包含图片路径的txt文件
        elif p.is_file(): # file
          with open(p, 'r') as t:
            t = t.read().strip().splitlines() # 获取图片路径，更换相对路径
            # 获取数据集路径的上级父目录 os.sep为路径里的分隔符（不同路径的分隔符不同，os.sep可以根据系统自适应）
            parent = str(p.parent) + os.sep
            f += [x.replace('./', parent) if x.startswith('./') else x for x in t] # local to global path
            # f += [p.parent / x.lstrip(os.sep) for x in t] # local to global path (pathlib)
        else:
          raise Exception(f'{prefix}{p} does not exist')
      # 破折号替换为os.sep，os.path.splitext(x)将文件名与扩展名分开并返回一个列表
      # 筛选f中所有的图片文件
      self.img_files = sorted([x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in img_formats])
      # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in img_formats]) # pathlib
      assert self.img_files, f'{prefix}No images found'
    except Exception as e:
      raise Exception(f'{prefix}Error loading data from {path}: {e}\nSee {help_url}')

    # 3、根据imgs路径找到labels的路径self.label_files
    self.label_files = img2label_paths(self.img_files) # labels

    # 4、cache label 下次运行这个脚本的时候直接从cache中取label而不是去文件中取label 速度更快
    cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache') # cached labels path
    # Check cache
    if cache_path.is_file():
      # 如果有cache文件，就直接加载 exists=True: 判断是否已从cache文件中读出了nf, nm, ne, nc, n等信息
      cache, exists = torch.load(cache_path), True # load
      # 如果图片版本信息或者文件列表的hash值对不上号 说明本地数据集图片和label可能发生了变化 就重新cache label文件
      if cache.get('version') != 0.3 or cache.get('hash') != get_hash(self.label_files + self.img_files):
        cache, exists = self.cache_labels(cache_path, prefix), False # re-cache
    else:
      # 否则调用cache_labels缓存标签及标签相关信息
      cache, exists = self.cache_labels(cache_path, prefix), False # cache

    # 打印cache的结果 nf nm ne nc n = 找到的标签数量，漏掉的标签数量，空的标签数量，损坏的标签数量，总的标签数量
    nf, nm, ne, nc, n = cache.pop('results') # found, missing, empty, corrupted, total
    # 如果已经从cache文件读出了nf nm ne nc n等信息，直接显示标签信息 msgs信息等
    if exists:
      d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
      tqdm(None, desc=prefix + d, total=n, initial=n) # display all cache results
      if cache['msgs']:
        logging.info('\n'.join(cache['msgs'])) # display all warnings msg
    # 数据集没有标签信息 就发出警告并显示标签label下载地址help_url
    assert nf > 0 or not augment, f'{prefix}No labels in {cache_path}. Can not train without labels. See {help_url}'

    # 5、Read cache 从cache中读出最新变量赋给self 方便给forward中使用
    # cache中的键值对最初有: cache[img_file]=[l, shape, segments] cache[hash] cache[results] cache[msg] cache[version]
    # 先从cache中去除cache文件中其他无关键值如:'hash', 'version', 'msgs'等都删除
    [cache.pop(k) for k in ('hash', 'version', 'msgs')] # remove items
    # pop掉results、hash、version、msgs后只剩下cache[img_file]=[l, shape, segments]
    # cache.values(): 取cache中所有值 对应所有l, shape, segments
    # labels: 如果数据集所有图片中没有一个多边形label labels存储的label就都是原始label(都是正常的矩形label)
    #    否则将所有图片正常gt的label存入labels 不正常gt(存在一个多边形)经过segments2boxes转换为正常的矩形label
    # shapes: 所有图片的shape
    # self.segments: 如果数据集所有图片中没有一个多边形label self.segments=None
    #        否则存储数据集中所有存在多边形gt的图片的所有原始label(肯定有多边形label 也可能有矩形正常label 未知数)
    # zip 是因为cache中所有labels、shapes、segments信息都是按每张img分开存储的, zip是将所有图片对应的信息叠在一起
    labels, shapes, self.segments = zip(*cache.values()) # segments: 都是[]
    self.labels = list(labels) # labels to list
    self.shapes = np.array(shapes, dtype=np.float64) # image shapes to float64
    self.img_files = list(cache.keys()) # 更新所有图片的img_files信息 update img_files from cache result
    self.label_files = img2label_paths(cache.keys()) # 更新所有图片的label_files信息(因为img_files信息可能发生了变化)
    if single_cls:
      for x in self.labels:
        x[:, 0] = 0
    n = len(shapes) # number of images
    bi = np.floor(np.arange(n) / batch_size).astype(np.int) # batch index
    nb = bi[-1] + 1 # number of batches
    self.batch = bi # batch index of image
    self.n = n # number of images
    self.indices = range(n) # 所有图片的index

    # 6、为Rectangular Training作准备
    # 这里主要是注意shapes的生成 这一步很重要 因为如果采样矩形训练那么整个batch的形状要一样 就要计算这个符合整个batch的shape
    # 而且还要对数据集按照高宽比进行排序 这样才能保证同一个batch的图片的形状差不多相同 再选则一个共同的shape代价也比较小
    if self.rect:
      # Sort by aspect ratio
      s = self.shapes # wh
      ar = s[:, 1] / s[:, 0] # aspect ratio
      irect = ar.argsort() # 根据高宽比排序
      self.img_files = [self.img_files[i] for i in irect]   # 获取排序后的img_files
      self.label_files = [self.label_files[i] for i in irect] # 获取排序后的label_files
      self.labels = [self.labels[i] for i in irect]      # 获取排序后的labels
      self.shapes = s[irect]                 # 获取排序后的wh
      ar = ar[irect]                     # 获取排序后的aspect ratio

      # 计算每个batch采用的统一尺度 Set training image shapes
      shapes = [[1, 1]] * nb  # nb: number of batches
      for i in range(nb):
        ari = ar[bi == i]  # bi: batch index
        mini, maxi = ari.min(), ari.max() # 获取第i个batch中，最小和最大高宽比
        # 如果高/宽小于1(w > h)，将w设为img_size（保证原图像尺度不变进行缩放）
        if maxi < 1:
          shapes[i] = [maxi, 1] # maxi: h相对指定尺度的比例 1: w相对指定尺度的比例
        # 如果高/宽大于1(w < h)，将h设置为img_size（保证原图像尺度不变进行缩放）
        elif mini > 1:
          shapes[i] = [1, 1 / mini]

      # 计算每个batch输入网络的shape值(向上设置为32的整数倍)
      # 要求每个batch_shapes的高宽都是32的整数倍，所以要先除以32，取整再乘以32（不过img_size如果是32倍数这里就没必要了）
      self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride

    # 7、是否需要cache image 一般是False 因为RAM会不足 cache label还可以 但是cache image就太大了 所以一般不用
    # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
    self.imgs = [None] * n
    if cache_images:
      gb = 0 # Gigabytes of cached images
      self.img_hw0, self.img_hw = [None] * n, [None] * n
      results = ThreadPool(num_threads).imap(lambda x: load_image(*x), zip(repeat(self), range(n)))
      pbar = tqdm(enumerate(results), total=n)
      for i, x in pbar:
        self.imgs[i], self.img_hw0[i], self.img_hw[i] = x # img, hw_original, hw_resized = load_image(self, i)
        gb += self.imgs[i].nbytes
        pbar.desc = f'{prefix}Caching images ({gb / 1E9:.1f}GB)'
      pbar.close()

  def cache_labels(self, path=Path('./labels.cache'), prefix=''):
    """用在__init__函数中 cache数据集label
    加载label信息生成cache文件，用于检查标注和读取shape 
    :params path: cache文件保存地址
    :params prefix: 日志头部信息(彩打高亮部分)
    :return x: cache中保存的字典
        包括的信息有: x[im_file] = [l, shape, segments]
             一张图片一个label相对应的保存到x, 最终x会保存所有图片的相对路径、gt框的信息、形状shape、所有的多边形gt信息
               im_file: 当前这张图片的path相对路径
               l: 当前这张图片的所有gt框的label信息(不包含segment多边形标签) [gt_num, cls+xywh(normalized)]
               shape: 当前这张图片的形状 shape
               segments: 当前这张图片所有gt的label信息(包含segment多边形标签) [gt_num, xy1...]
              hash: 当前图片和label文件的hash值 1
              results: 找到的label个数nf, 丢失label个数nm, 空label个数ne, 破损label个数nc, 总img/label个数len(self.img_files)
              msgs: 所有数据集的msgs信息
              version: 当前cache version
    """
    x = {} # 初始化最终cache中保存的字典dict
    # 初始化number missing, found, empty, corrupt, messages
    # 初始化整个数据集: 漏掉的标签(label)总数量, 找到的标签(label)总数量, 空的标签(label)总数量, 错误标签(label)总数量, 所有错误信息
    nm, nf, ne, nc, msgs = 0, 0, 0, 0, []
    desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels..." # 日志
    # 多进程调用verify_image_label函数
    with Pool(num_threads) as pool:
      # 定义pbar进度条
      # pool.imap_unordered: 对大量数据遍历多进程计算 返回一个迭代器
      # 把self.img_files, self.label_files, repeat(prefix) list中的值作为参数依次送入(一次送一个)verify_image_label函数
      pbar = tqdm(pool.imap_unordered(verify_image_label, zip(self.img_files, self.label_files, repeat(prefix))),
            desc=desc, total=len(self.img_files))
      # im_file: 当前这张图片的path相对路径
      # l: [gt_num, cls+xywh(normalized)]
      #  如果这张图片没有一个segment多边形标签 l就存储原label(全部是正常矩形标签)
      #  如果这张图片有一个segment多边形标签 l就存储经过segments2boxes处理好的标签(正常矩形标签不处理 多边形标签转化为矩形标签)
      # shape: 当前这张图片的形状 shape
      # segments: 如果这张图片没有一个segment多边形标签 存储None
      #     如果这张图片有一个segment多边形标签 就把这张图片的所有label存储到segments中(若干个正常gt 若干个多边形标签) [gt_num, xy1...]
      # nm_f(nm): number missing 当前这张图片的label是否丢失    丢失=1  存在=0
      # nf_f(nf): number found 当前这张图片的label是否存在     存在=1  丢失=0
      # ne_f(ne): number empty 当前这张图片的label是否是空的    空的=1  没空=0
      # nc_f(nc): number corrupt 当前这张图片的label文件是否是破损的 破损的=1 没破损=0
      # msg: 返回的msg信息 label文件完好=‘’ label文件破损=warning信息
      for im_file, l, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
        nm += nm_f # 累加总number missing label
        nf += nf_f # 累加总number found label
        ne += ne_f # 累加总number empty label
        nc += nc_f # 累加总number corrupt label
        if im_file:
          x[im_file] = [l, shape, segments] # 信息存入字典 key=im_file value=[l, shape, segments]
        if msg:
          msgs.append(msg) # 将msg加入总msg
        pbar.desc = f"{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupted" # 日志
    pbar.close() # 关闭进度条
    # 日志打印所有msg信息
    if msgs:
      logging.info('\n'.join(msgs))
    # 一张label都没找到 日志打印help_url下载地址
    if nf == 0:
      logging.info(f'{prefix}WARNING: No labels found in {path}. See {help_url}')
    x['hash'] = get_hash(self.label_files + self.img_files) # 将当前图片和label文件的hash值存入最终字典dist
    x['results'] = nf, nm, ne, nc, len(self.img_files) # 将nf, nm, ne, nc, len(self.img_files)存入最终字典dist
    x['msgs'] = msgs # 将所有数据集的msgs信息存入最终字典dist
    x['version'] = 0.3 # 将当前cache version存入最终字典dist
    try:
      torch.save(x, path) # save cache to path
      logging.info(f'{prefix}New cache created: {path}')
    except Exception as e:
      logging.info(f'{prefix}WARNING: Cache directory {path.parent} is not writeable: {e}') # path not writeable
    return x

  def __len__(self):
    return len(self.img_files)

  # def __iter__(self):
  #  self.count = -1
  #  print('ran dataset iter')
  #  #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
  #  return self

  def __getitem__(self, index):
    """
    训练 数据增强: mosaic(random_perspective) + hsv + 上下左右翻转
    测试 数据增强: letterbox
    :return torch.from_numpy(img): 这个index的图片数据(增强后) [3, 640, 640]
    :return labels_out: 这个index图片的gt label [6, 6] = [gt_num, 0+class+xywh(normalized)]
    :return self.img_files[index]: 这个index图片的路径地址
    :return shapes: 这个batch的图片的shapes 测试时(矩形训练)才有 验证时为None for COCO mAP rescaling
    """
    # 这里可以通过三种形式获取要进行数据增强的图片index linear, shuffled, or image_weights
    index = self.indices[index]

    hyp = self.hyp # 超参 包含众多数据增强超参
    mosaic = self.mosaic and random.random() < hyp['mosaic']
    # mosaic增强：对图像进行4张图拼接训练 
    # mosaic + MixUp
    if mosaic:
      # Load mosaic
      img, labels = load_mosaic(self, index)
      # img, labels = load_mosaic9(self, index)
      shapes = None

      # MixUp augmentation
      # mixup数据增强
      if random.random() < hyp['mixup']: # hyp['mixup']=0 默认为0则关闭 默认为1则100%打开
        # *load_mosaic(self, random.randint(0, self.n - 1)) 随机从数据集中任选一张图片和本张图片进行mixup数据增强
        # img: 两张图片融合之后的图片 numpy (640, 640, 3)
        # labels: 两张图片融合之后的标签label [M+N, cls+x1y1x2y2]
        img, labels = mixup(img, labels, *load_mosaic(self, random.randint(0, self.n - 1)))

        # 测试代码 测试MixUp效果
        # cv2.imshow("MixUp", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print(img.shape) # (640, 640, 3)

    # 否则: 载入图片 + Letterbox (val)
    else:
      # Load image
      # 载入图片 载入图片后还会进行一次resize 将当前图片的最长边缩放到指定的大小(512), 较小边同比例缩放
      # load image img=(343, 512, 3)=(h, w, c) (h0, w0)=(335, 500) numpy index=4
      # img: resize后的图片 (h0, w0): 原始图片的hw (h, w): resize后的图片的hw
      # 这一步是将(335, 500, 3) resize-> (343, 512, 3)
      img, (h0, w0), (h, w) = load_image(self, index)

      # 测试代码 测试load_image效果
      # cv2.imshow("load_image", img)
      # cv2.waitKey(0)
      # cv2.destroyAllWindows()
      # print(img.shape) # (640, 640, 3)

      # Letterbox
      # letterbox之前确定这张当前图片letterbox之后的shape 如果不用self.rect矩形训练shape就是self.img_size
      # 如果使用self.rect矩形训练shape就是当前batch的shape 因为矩形训练时，整个batch的shape会统一
      shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size # final letterboxed shape
      # letterbox 这一步将第一步缩放得到的图片再缩放到当前batch所需要的尺度 (343, 512, 3) pad-> (384, 512, 3)
      # (矩形推理需要一个batch的所有图片的shape必须相同，而这个shape在init函数中保持在self.batch_shapes中)
      # 这里没有缩放操作，所以这里的ratio永远都是(1.0, 1.0) pad=(0.0, 20.5)
      img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
      shapes = (h0, w0), ((h / h0, w / w0), pad) # for COCO mAP rescaling

      # 图片letterbox之后label的坐标也要相应变化 根据pad调整label坐标 并将归一化的xywh -> 未归一化的xyxy
      labels = self.labels[index].copy()
      if labels.size: # normalized xywh to pixel xyxy format
        labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

      # 测试代码 测试letterbox效果
      # cv2.imshow("letterbox", img)
      # cv2.waitKey(0)
      # cv2.destroyAllWindows()
      # print(img.shape) # (640, 640, 3)

    if self.augment:
      # Augment imagespace
      if not mosaic:
        # 不做mosaic的话就要做random_perspective增强 因为mosaic函数内部执行了random_perspective增强
        # random_perspective增强: 随机对图片进行旋转，平移，缩放，裁剪，透视变换
        img, labels = random_perspective(img, labels,
                         degrees=hyp['degrees'],
                         translate=hyp['translate'],
                         scale=hyp['scale'],
                         shear=hyp['shear'],
                         perspective=hyp['perspective'])

      # 色域空间增强Augment colorspace
      augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

      # 测试代码 测试augment_hsv效果
      # cv2.imshow("augment_hsv", img)
      # cv2.waitKey(0)
      # cv2.destroyAllWindows()
      # print(img.shape) # (640, 640, 3)

      # Apply cutouts 随机进行cutout增强 0.5的几率使用 这里可以自行测试
      if random.random() < hyp['cutout']: # hyp['cutout']=0 默认为0则关闭 默认为1则100%打开
        labels = cutout(img, labels)

        # 测试代码 测试cutout效果
        # cv2.imshow("cutout", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print(img.shape) # (640, 640, 3)

    nL = len(labels) # number of labels
    if nL:
      # xyxy to xywh normalized
      labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0])

    # 平移增强 随机左右翻转 + 随机上下翻转
    if self.augment:
      # 随机上下翻转 flip up-down
      if random.random() < hyp['flipud']:
        img = np.flipud(img) # np.flipud 将数组在上下方向翻转。
        if nL:
          labels[:, 2] = 1 - labels[:, 2] # 1 - y_center label也要映射

      # 随机左右翻转 flip left-right
      if random.random() < hyp['fliplr']:
        img = np.fliplr(img) # np.fliplr 将数组在左右方向翻转
        if nL:
          labels[:, 1] = 1 - labels[:, 1] # 1 - x_center label也要映射

    # 6个值的tensor 初始化标签框对应的图片序号, 配合下面的collate_fn使用
    labels_out = torch.zeros((nL, 6))
    if nL:
      labels_out[:, 1:] = torch.from_numpy(labels) # numpy to tensor

    # Convert BGR->RGB HWC->CHW
    img = img[:, :, ::-1].transpose(2, 0, 1) # BGR to RGB, to 3 x img_height x img_width
    img = np.ascontiguousarray(img) # img变成内存连续的数据 加快运算

    return torch.from_numpy(img), labels_out, self.img_files[index], shapes

  @staticmethod
  def collate_fn(batch):
    """这个函数会在create_dataloader中生成dataloader时调用：
    整理函数 将image和label整合到一起
    :return torch.stack(img, 0): 如[16, 3, 640, 640] 整个batch的图片
    :return torch.cat(label, 0): 如[15, 6] [num_target, img_index+class_index+xywh(normalized)] 整个batch的label
    :return path: 整个batch所有图片的路径
    :return shapes: (h0, w0), ((h / h0, w / w0), pad)  for COCO mAP rescaling
    pytorch的DataLoader打包一个batch的数据集时要经过此函数进行打包 通过重写此函数实现标签与图片对应的划分，一个batch中哪些标签属于哪一张图片,形如
      [[0, 6, 0.5, 0.5, 0.26, 0.35],
       [0, 6, 0.5, 0.5, 0.26, 0.35],
       [1, 6, 0.5, 0.5, 0.26, 0.35],
       [2, 6, 0.5, 0.5, 0.26, 0.35],]
      前两行标签属于第一张图片, 第三行属于第二张。。。
    """
    # img: 一个tuple 由batch_size个tensor组成 整个batch中每个tensor表示一张图片
    # label: 一个tuple 由batch_size个tensor组成 每个tensor存放一张图片的所有的target信息
    #    label[6, object_num] 6中的第一个数代表一个batch中的第几张图
    # path: 一个tuple 由4个str组成, 每个str对应一张图片的地址信息
    img, label, path, shapes = zip(*batch) # transposed
    for i, l in enumerate(label):
      l[:, 0] = i # add target image index for build_targets()
    # 返回的img=[batch_size, 3, 736, 736]
    #   torch.stack(img, 0): 将batch_size个[3, 736, 736]的矩阵拼成一个[batch_size, 3, 736, 736]
    # label=[target_sums, 6] 6：表示当前target属于哪一张图+class+x+y+w+h
    #   torch.cat(label, 0): 将[n1,6]、[n2,6]、[n3,6]...拼接成[n1+n2+n3+..., 6]
    # 这里之所以拼接的方式不同是因为img拼接的时候它的每个部分的形状是相同的，都是[3, 736, 736]
    # 而我label的每个部分的形状是不一定相同的，每张图的目标个数是不一定相同的（label肯定也希望用stack,更方便,但是不能那样拼）
    # 如果每张图的目标个数是相同的，那我们就可能不需要重写collate_fn函数了
    return torch.stack(img, 0), torch.cat(label, 0), path, shapes

  @staticmethod
  def collate_fn4(batch):
    """同样在create_dataloader中生成dataloader时调用：
    这里是yolo-v5作者实验性的一个代码 quad-collate function 当train.py的opt参数quad=True 则调用collate_fn4代替collate_fn
    作用: 如之前用collate_fn可以返回图片[16, 3, 640, 640] 经过collate_fn4则返回图片[4, 3, 1280, 1280]
       将4张mosaic图片[1, 3, 640, 640]合成一张大的mosaic图片[1, 3, 1280, 1280]
       将一个batch的图片每四张处理, 0.5的概率将四张图片拼接到一张大图上训练, 0.5概率直接将某张图片上采样两倍训练
    """
    # img: 整个batch的图片 [16, 3, 640, 640]
    # label: 整个batch的label标签 [num_target, img_index+class_index+xywh(normalized)]
    # path: 整个batch所有图片的路径
    # shapes: (h0, w0), ((h / h0, w / w0), pad)  for COCO mAP rescaling
    img, label, path, shapes = zip(*batch) # transposed
    n = len(shapes) // 4 # collate_fn4处理后这个batch中图片的个数
    img4, label4, path4, shapes4 = [], [], path[:n], shapes[:n] # 初始化

    ho = torch.tensor([[0., 0, 0, 1, 0, 0]])
    wo = torch.tensor([[0., 0, 1, 0, 0, 0]])
    s = torch.tensor([[1, 1, .5, .5, .5, .5]]) # scale
    for i in range(n): # zidane torch.zeros(16,3,720,1280) # BCHW
      i *= 4 # 采样 [0, 4, 8, 16]
      if random.random() < 0.5:
        # 随机数小于0.5就直接将某张图片上采样两倍训练
        im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2., mode='bilinear', align_corners=False)[
          0].type(img[i].type())
        l = label[i]
      else:
        # 随机数大于0.5就将四张图片(mosaic后的)拼接到一张大图上训练
        im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
        l = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
      img4.append(im)
      label4.append(l)

    # 后面返回的部分和collate_fn就差不多了 原因和解释都写在上一个函数了 自己debug看一下吧
    for i, l in enumerate(label4):
      l[:, 0] = i # add target image index for build_targets()

    return torch.stack(img4, 0), torch.cat(label4, 0), path4, shapes4
```



其实这里还会根据类别来赋予不同的权重，逻辑就是根据类别的占比进行计算，然后再根据比例来进行采样，但这里默认不同类别的权重是一样的。



```python
model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
```



那么是如何为不同的类别分配不同的权重的呢？



```python
def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
  """利用上面得到的每个类别的权重得到每一张图片的权重 再对图片进行按照权重进行不同比例的采样，默认权重都是一样的
  通过每张图片真实gt框的真实标签labels和上一步labels_to_class_weights得到的每个类别的权重进行采样
  Produces image weights based on class_weights and image contents
  :params labels: 每张图片真实gt框的真实标签
  :params nc: 数据集的类别数 默认80
  :params class_weights: [80] 上一步labels_to_class_weights得到的每个类别的权重
  """
  # class_counts: 每个类别出现的次数 [num_labels, nc] 每一行是当前这张图片每个类别出现的次数 num_labels=图片数量=label数量
  class_counts = np.array([np.bincount(x[:, 0].astype(np.int), minlength=nc) for x in labels])
  # [80] -> [1, 80]
  # 整个数据集的每个类别权重[1, 80] * 每张图片的每个类别出现的次数[num_labels, 80] = 得到每一张图片每个类对应的权重[128, 80]
  # 另外注意: 这里不是矩阵相乘, 是元素相乘 [1, 80] 和每一行图片的每个类别出现的次数 [1, 80] 分别按元素相乘
  # 再sum(1): 按行相加 得到最终image_weights: 得到每一张图片对应的采样权重[128]
  image_weights = (class_weights.reshape(1, nc) * class_counts).sum(1)
  # index = random.choices(range(n), weights=image_weights, k=1) # weight image sample
  return image_weights
```



接下来就是计算损失了，计算的接口是 ComputeLoss。根据超参中的损失权重参数 对各个损失进行平衡 防止总损失被某个损失所主导。



```python
class ComputeLoss:
    # Compute losses 这个函数会用在train.py中进行损失计算
    def __init__(self, model, autobalance=False):
        super(ComputeLoss, self).__init__()
        self.sort_obj_iou = False  # 筛选置信度损失正样本前是否先对iou排序

        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria 定义分类损失和置信度损失
        # h['cls_pw']=1  BCEWithLogitsLoss默认的正样本权重也是1
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # 标签平滑  eps=0代表不做标签平滑-> cp=1 cn=0  eps!=0代表做标签平滑 cp代表positive的标签值 cn代表negative的标签值
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss  g=0 代表不用focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            # g>0 将分类损失和置信度损失(BCE)都换成focalloss损失函数
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
            # BCEcls, BCEobj = QFocalLoss(BCEcls, g), QFocalLoss(BCEobj, g)  # 调用QFocalLoss来代替FocalLoss

        # det: 返回的是模型的检测头 Detector 3个 分别对应产生三个输出feature map
        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module

        # balance用来设置三个feature map对应输出的置信度损失系数(平衡三个feature map的置信度损失)
        # 从左到右分别对应大feature map(检测小目标)到小feature map(检测大目标)
        # 思路:  It seems that larger output layers may overfit earlier, so those numbers may need a bit of adjustment
        #       一般来说，检测小物体的难度大一点，所以会增加大特征图的损失系数，让模型更加侧重小物体的检测
        # 如果det.nl=3就返回[4.0, 1.0, 0.4]否则返回[4.0, 1.0, 0.25, 0.06, .02]
        # self.balance = {3: [4.0, 1.0, 0.4], 4: [4.0, 1.0, 0.25, 0.06], 5: [4.0, 1.0, 0.25, 0.06, .02]}[det.nl]
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7

        # 三个预测头的下采样率det.stride: [8, 16, 32]  .index(16): 求出下采样率stride=16的索引
        # 这个参数会用来自动计算更新3个feature map的置信度损失系数self.balance
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index

        # self.BCEcls: 类别损失函数   self.BCEobj: 置信度损失函数   self.hyp: 超参数
        # self.gr: 计算真实框的置信度标准的iou ratio    self.autobalance: 是否自动更新各feature map的置信度损失平衡系数  默认False
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance

        # na: number of anchors  每个grid_cell的anchor数量 = 3
        # nc: number of classes  数据集的总类别 = 80
        # nl: number of detection layers   Detect的个数 = 3
        # anchors: [3, 3, 2]  3个feature map 每个feature map上有3个anchor(w,h) 这里的anchor尺寸是相对feature map的
        for k in 'na', 'nc', 'nl', 'anchors':
            # setattr: 给对象self的属性k赋值为getattr(det, k)
            # getattr: 返回det对象的k属性
            # 所以这句话的意思: 讲det的k属性赋值给self.k属性 其中k in 'na', 'nc', 'nl', 'anchors'
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        """
        :params p:  预测框 由模型构建中的三个检测头Detector返回的三个yolo层的输出
                    tensor格式 list列表 存放三个tensor 对应的是三个yolo层的输出
                    如: [4, 3, 112, 112, 85]、[4, 3, 56, 56, 85]、[4, 3, 28, 28, 85]
                    [bs, anchor_num, grid_h, grid_w, xywh+class+classes]
                    可以看出来这里的预测值p是三个yolo层每个grid_cell(每个grid_cell有三个预测值)的预测值,后面肯定要进行正样本筛选
        :params targets: 数据增强后的真实框 [63, 6] [num_object,  batch_index+class+xywh]
        :params loss * bs: 整个batch的总损失  进行反向传播
        :params torch.cat((lbox, lobj, lcls, loss)).detach(): 回归损失、置信度损失、分类损失和总损失 这个参数只用来可视化参数或保存信息
        """
        device = targets.device  # 确定运行的设备

        # 初始化lcls, lbox, lobj三种损失值  tensor([0.])
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)

        # 每一个都是append的 有feature map个 每个都是当前这个feature map中3个anchor筛选出的所有的target(3个grid_cell进行预测)
        # tcls: 表示这个target所属的class index
        # tbox: xywh 其中xy为这个target对当前grid_cell左上角的偏移量
        # indices: b: 表示这个target属于的image index
        #          a: 表示这个target使用的anchor index
        #          gj: 经过筛选后确定某个target在某个网格中进行预测(计算损失)  gj表示这个网格的左上角y坐标
        #          gi: 表示这个网格的左上角x坐标
        # anch: 表示这个target所使用anchor的尺度（相对于这个feature map）  注意可能一个target会使用大小不同anchor进行计算
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # 依次遍历三个feature map的预测输出pi
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image_index, anchor_index, gridy, gridx

            tobj = torch.zeros_like(pi[..., 0], device=device)  # 初始化target置信度(先全是负样本 后面再筛选正样本赋值)

            n = b.shape[0]  # number of targets
            if n:
                # 精确得到第b张图片的第a个feature map的grid_cell(gi, gj)对应的预测值
                # 用这个预测值与我们筛选的这个grid_cell的真实框进行预测(计算损失)
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression loss  只计算所有正样本的回归损失
                # 新的公式:  pxy = [-0.5 + cx, 1.5 + cx]    pwh = [0, 4pw]   这个区域内都是正样本
                # Get more positive samples, accelerate convergence and be more stable
                pxy = ps[:, :2].sigmoid() * 2. - 0.5  # 一个归一化操作 和论文里不同
                # https://github.com/ultralytics/yolov3/issues/168
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]  # 和论文里不同 这里是作者自己提出的公式
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                # 这里的tbox[i]中的xy是这个target对当前grid_cell左上角的偏移量[0,1]  而pbox.T是一个归一化的值
                # 就是要用这种方式训练 传回loss 修改梯度 让pbox越来越接近tbox(偏移量)
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness loss stpe1
                # iou.detach()  不会更新iou梯度  iou并不是反向传播的参数 所以不需要反向传播梯度信息
                score_iou = iou.detach().clamp(0).type(tobj.dtype)  # .clamp(0)必须大于等于0
                if self.sort_obj_iou: # 如果需要对iou进行排序
                    # https://github.com/ultralytics/yolov5/issues/3605
                    # There maybe several GTs match the same anchor when calculate ComputeLoss in the scene with dense targets
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                # 预测信息有置信度 但是真实框信息是没有置信度的 所以需要我们认为的给一个标准置信度
                # self.gr是iou ratio [0, 1]  self.gr越大置信度越接近iou  self.gr越小越接近1(人为加大训练难度)
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio
                # tobj[b, a, gj, gi] = 1  # 如果发现预测的score不高 数据集目标太小太拥挤 困难样本过多 可以试试这个

                # Classification loss  只计算所有正样本的分类损失
                if self.nc > 1:  # cls loss (only if multiple classes)
                    # targets 原本负样本是0  这里使用smooth label 就是cn
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)
                    t[range(n), tcls[i]] = self.cp  # 筛选到的正样本对应位置值是cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            # Objectness loss stpe2 置信度损失是用所有样本(正样本 + 负样本)一起计算损失的
            obji = self.BCEobj(pi[..., 4], tobj)
            # 每个feature map的置信度损失权重不同  要乘以相应的权重系数self.balance[i]
            # 一般来说，检测小物体的难度大一点，所以会增加大特征图的损失系数，让模型更加侧重小物体的检测
            lobj += obji * self.balance[i]  # obj loss

            if self.autobalance:
                # 如果为True 那么自动更新各个feature map的置信度损失系数
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]

        # 根据超参中的损失权重参数 对各个损失进行平衡  防止总损失被某个损失所左右
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']

        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls  # 平均每张图片的总损失

        # loss * bs: 整个batch的总损失
        # .detach()  利用损失值进行反向传播 利用梯度信息更新的是损失函数的参数 而对于损失这个值是不需要梯度反向传播的
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

    def build_targets(self, p, targets):
        """
        Build targets for compute_loss()
        :params p: 预测框 由模型构建中的三个检测头Detector返回的三个yolo层的输出
                   tensor格式 list列表 存放三个tensor 对应的是三个yolo层的输出
                   如: [4, 3, 112, 112, 85]、[4, 3, 56, 56, 85]、[4, 3, 28, 28, 85]
                   [bs, anchor_num, grid_h, grid_w, xywh+class+classes]
                   可以看出来这里的预测值p是三个yolo层每个grid_cell(每个grid_cell有三个预测值)的预测值,后面肯定要进行正样本筛选
        :params targets: 数据增强后的真实框 [63, 6] [num_target,  image_index+class+xywh] xywh为归一化后的框
        :return tcls: 表示这个target所属的class index
                tbox: xywh 其中xy为这个target对当前grid_cell左上角的偏移量
                indices: b: 表示这个target属于的image index
                         a: 表示这个target使用的anchor index
                        gj: 经过筛选后确定某个target在某个网格中进行预测(计算损失)  gj表示这个网格的左上角y坐标
                        gi: 表示这个网格的左上角x坐标
                anch: 表示这个target所使用anchor的尺度（相对于这个feature map）  注意可能一个target会使用大小不同anchor进行计算
        """
        na, nt = self.na, targets.shape[0]  # number of anchors 3, targets 63
        tcls, tbox, indices, anch = [], [], [], []   # 初始化tcls tbox indices anch

        # gain是为了后面将targets=[na,nt,7]中的归一化了的xywh映射到相对feature map尺度上
        # 7: image_index+class+xywh+anchor_index
        gain = torch.ones(7, device=targets.device)

        # 需要在3个anchor上都进行训练 所以将标签赋值na=3个  ai代表3个anchor上在所有的target对应的anchor索引 就是用来标记下当前这个target属于哪个anchor
        # [1, 3] -> [3, 1] -> [3, 63]=[na, nt]   三行  第一行63个0  第二行63个1  第三行63个2
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)

        # [63, 6] [3, 63] -> [3, 63, 6] [3, 63, 1] -> [3, 63, 7]  7: [image_index+class+xywh+anchor_index]
        # 对每一个feature map: 这一步是将target复制三份 对应一个feature map的三个anchor
        # 先假设所有的target对三个anchor都是正样本(复制三份) 再进行筛选  并将ai加进去标记当前是哪个anchor的target
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        # 这两个变量是用来扩展正样本的 因为预测框预测到target有可能不止当前的格子预测到了
        # 可能周围的格子也预测到了高质量的样本 我们也要把这部分的预测信息加入正样本中
        g = 0.5  # bias  中心偏移  用来衡量target中心点离哪个格子更近
        # 以自身 + 周围左上右下4个网格 = 5个网格  用来计算offsets
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        # 遍历三个feature 筛选每个feature map(包含batch张图片)的每个anchor的正样本
        for i in range(self.nl):  # self.nl: number of detection layers   Detect的个数 = 3
            # anchors: 当前feature map对应的三个anchor尺寸(相对feature map)  [3, 2]
            anchors = self.anchors[i]

            # gain: 保存每个输出feature map的宽高 -> gain[2:6]=gain[whwh]
            # [1, 1, 1, 1, 1, 1, 1] -> [1, 1, 112, 112, 112,112, 1]=image_index+class+xywh+anchor_index
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # t = [3, 63, 7]  将target中的xywh的归一化尺度放缩到相对当前feature map的坐标尺度
            #     [3, 63, image_index+class+xywh+anchor_index]
            t = targets * gain

            if nt:  # 开始匹配  Matches
                # t=[na, nt, 7]   t[:, :, 4:6]=[na, nt, 2]=[3, 63, 2]
                # anchors[:, None]=[na, 1, 2]
                # r=[na, nt, 2]=[3, 63, 2]
                # 当前feature map的3个anchor的所有正样本(没删除前是所有的targets)与三个anchor的宽高比(w/w  h/h)
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio (w/w  h/h)

                # 筛选条件  GT与anchor的宽比或高比超过一定的阈值 就当作负样本
                # torch.max(r, 1. / r)=[3, 63, 2] 筛选出宽比w1/w2 w2/w1 高比h1/h2 h2/h1中最大的那个
                # .max(2)返回宽比 高比两者中较大的一个值和它的索引  [0]返回较大的一个值
                # j: [3, 63]  False: 当前gt是当前anchor的负样本  True: 当前gt是当前anchor的正样本
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # yolov3 v4的筛选方法: wh_iou  GT与anchor的wh_iou超过一定的阈值就是正样本
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))

                # 根据筛选条件j, 过滤负样本, 得到当前feature map上三个anchor的所有正样本t(batch_size张图片)
                # t: [3, 63, 7] -> [126, 7]  [num_Positive_sample, image_index+class+xywh+anchor_index]
                t = t[j]  # filter

                # Offsets 筛选当前格子周围格子 找到2个离target中心最近的两个格子  可能周围的格子也预测到了高质量的样本 我们也要把这部分的预测信息加入正样本中
                # 除了target所在的当前格子外, 还有2个格子对目标进行检测(计算损失) 也就是说一个目标需要3个格子去预测(计算损失)
                # 首先当前格子是其中1个 再从当前格子的上下左右四个格子中选择2个 用这三个格子去预测这个目标(计算损失)
                # feature map上的原点在左上角 向右为x轴正坐标 向下为y轴正坐标
                gxy = t[:, 2:4]  # grid xy 取target中心的坐标xy(相对feature map左上角的坐标)
                gxi = gain[[2, 3]] - gxy  # inverse  得到target中心点相对于右下角的坐标  gain[[2, 3]]为当前feature map的wh
                # 筛选中心坐标 距离当前grid_cell的左、上方偏移小于g=0.5 且 中心坐标必须大于1(坐标不能在边上 此时就没有4个格子了)
                # j: [126] bool 如果是True表示当前target中心点所在的格子的左边格子也对该target进行回归(后续进行计算损失)
                # k: [126] bool 如果是True表示当前target中心点所在的格子的上边格子也对该target进行回归(后续进行计算损失)
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                # 筛选中心坐标 距离当前grid_cell的右、下方偏移小于g=0.5 且 中心坐标必须大于1(坐标不能在边上 此时就没有4个格子了)
                # l: [126] bool 如果是True表示当前target中心点所在的格子的右边格子也对该target进行回归(后续进行计算损失)
                # m: [126] bool 如果是True表示当前target中心点所在的格子的下边格子也对该target进行回归(后续进行计算损失)
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                # j: [5, 126]  torch.ones_like(j): 当前格子, 不需要筛选全是True  j, k, l, m: 左上右下格子的筛选结果
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                # 得到筛选后所有格子的正样本 格子数<=3*126 都不在边上等号成立
                # t: [126, 7] -> 复制5份target[5, 126, 7]  分别对应当前格子和左上右下格子5个格子
                # j: [5, 126] + t: [5, 126, 7] => t: [378, 7] 理论上是小于等于3倍的126 当且仅当没有边界的格子等号成立
                t = t.repeat((5, 1, 1))[j]
                # torch.zeros_like(gxy)[None]: [1, 126, 2]   off[:, None]: [5, 1, 2]  => [5, 126, 2]
                # j筛选后: [378, 2]  得到所有筛选后的网格的中心相对于这个要预测的真实框所在网格边界（左右上下边框）的偏移量
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image_index, class
            gxy = t[:, 2:4]  # target的xy
            gwh = t[:, 4:6]  # target的wh
            gij = (gxy - offsets).long()   # 预测真实框的网格所在的左上角坐标(有左上右下的网格)
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor index
            # b: image index  a: anchor index  gj: 网格的左上角y坐标  gi: 网格的左上角x坐标
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))
            # tbix: xywh 其中xy为这个target对当前grid_cell左上角的偏移量
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # 对应的所有anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
```



## 文末



以上，就是对于 YOLOv5 的训练代码中比较重要的代码段的解析，因为篇幅有限，就没有对每一个函数都进行解读，但整体的篇幅还是较长，大家可以耐心地仔细看看~