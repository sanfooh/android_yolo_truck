# 如何在android上跑yolo
当训练好yolo的模型以后，想要将其移到移动端进行验证

![image](https://github.com/sanfooh/android_yolo_truck/edit/master/snap.png)

 ## 利用tensorflow
 * 训练darknet yolo模型
 * 将利用darkflow将darknet yolo模型文件转成tensorflow的模型
 * 修改tensorflow的官方例子
 

## 训练darknet yolo模型
先要制作数据集，利用darknet对作者提供的预训练模型进行微调，具体可以看[这里](https://github.com/sanfooh/yolo_truck)，如果需标注工具，可以看[这里](https://github.com/sanfooh/quick_yolo2_label_tool)

## 利用darkflow将darknet yolo模型文件转成tensorflow的模型
darkflow是一个使用tensorflow框架来实现darknet的项目，它的网址在[这里](https://github.com/thtrieu/darkflow)
此项目需要git clone下来，再用pip进行安装。

```
git clone https://github.com/thtrieu/darkflow.git
cd darkflow
pip install .
darkflow -help
```

这样安装完以后，就可以系统全局范围内使用flow工具：
我倾向使用原生的darknet来训练模型，而只是利用darkflow的转换模型功能，转换模型需要三个文件，一个是darknet的网络定义文件，也就是后缀名为.cfg的文件，另一个是.weights为后缀的权重量文件，还有一个对象标签文件。
```
flow --model cfg/yolo.cfg --load bin/yolo.weights --labels xxx.names --savepb
```
这个命令要注意的是，
* 1、权值的文件名可以随意修改，但后缀名一定要是weights,不然会出错：
```
  File "/root/anaconda3/lib/python3.6/site-packages/darkflow/dark/darknet.py", l    ine 50, in get_weight_src
    cfg_path = os.path.join(FLAGS.config, name + '.cfg')
TypeError: unsupported operand type(s) for +: 'NoneType' and 'str'

```
* 2、要指定--labels参数，如果是自己训练的模型，则要指向自己的obj.names,如果是直接下载官方的模型，一般是指向coco.names,在darnet源代码里的data目录下。
如果不指定，则可能会出现如下错误：
```
    with open(file, 'r') as f:
FileNotFoundError: [Errno 2] No such file or directory: 'labels.txt'

```



* 3、如果没有什么问题则形如：
```
[root@fsi-centos truckdata]# flow --model yolo-truck.cfg --load yolo-truck.weights --labels labels.txt  --savepb

/root/anaconda3/lib/python3.6/site-packages/darkflow/dark/darknet.py:54: UserWarning: ./cfg/yolo-truck.cfg not found, use yolo-truck.cfg instead
  cfg_path, FLAGS.model))
Parsing yolo-truck.cfg
Loading yolo-truck.weights ...
Successfully identified 202314764 bytes
Finished in 0.04830145835876465s

Building net ...
Source | Train? | Layer description                | Output size
-------+--------+----------------------------------+---------------
       |        | input                            | (?, 416, 416, 3)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 416, 416, 32)
 Load  |  Yep!  | maxp 2x2p0_2                     | (?, 208, 208, 32)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 208, 208, 64)
 Load  |  Yep!  | maxp 2x2p0_2                     | (?, 104, 104, 64)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 104, 104, 128)
 Load  |  Yep!  | conv 1x1p0_1  +bnorm  leaky      | (?, 104, 104, 64)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 104, 104, 128)
 Load  |  Yep!  | maxp 2x2p0_2                     | (?, 52, 52, 128)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 52, 52, 256)
 Load  |  Yep!  | conv 1x1p0_1  +bnorm  leaky      | (?, 52, 52, 128)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 52, 52, 256)
 Load  |  Yep!  | maxp 2x2p0_2                     | (?, 26, 26, 256)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 26, 26, 512)
 Load  |  Yep!  | conv 1x1p0_1  +bnorm  leaky      | (?, 26, 26, 256)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 26, 26, 512)
 Load  |  Yep!  | conv 1x1p0_1  +bnorm  leaky      | (?, 26, 26, 256)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 26, 26, 512)
 Load  |  Yep!  | maxp 2x2p0_2                     | (?, 13, 13, 512)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 13, 13, 1024)
 Load  |  Yep!  | conv 1x1p0_1  +bnorm  leaky      | (?, 13, 13, 512)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 13, 13, 1024)
 Load  |  Yep!  | conv 1x1p0_1  +bnorm  leaky      | (?, 13, 13, 512)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 13, 13, 1024)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 13, 13, 1024)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 13, 13, 1024)
 Load  |  Yep!  | concat [16]                      | (?, 26, 26, 512)
 Load  |  Yep!  | conv 1x1p0_1  +bnorm  leaky      | (?, 26, 26, 64)
 Load  |  Yep!  | local flatten 2x2                | (?, 13, 13, 256)
 Load  |  Yep!  | concat [27, 24]                  | (?, 13, 13, 1280)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 13, 13, 1024)
 Load  |  Yep!  | conv 1x1p0_1    linear           | (?, 13, 13, 30)
-------+--------+----------------------------------+---------------
Running entirely on CPU
2018-03-26 16:15:55.255095: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2018-03-26 16:15:55.255132: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2018-03-26 16:15:55.255171: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2018-03-26 16:15:55.255190: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2018-03-26 16:15:55.255209: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
Finished in 8.368546724319458s

Rebuild a constant version ...

```


* 4、执行完这个命令以后，就可以生成两个文件，分别是后缀.meta和.pb的文件，其中.pb文件就是我们想要的。



需要注意的是目前yolov3的版本是无法转换成功的会提示,具体可能需要darkflow有更新：
```
/root/anaconda3/lib/python3.6/site-packages/darkflow/dark/darknet.py:54: UserWarning: ./cfg/yolov3.cfg not found, use yolov3.cfg instead
  cfg_path, FLAGS.model))
Parsing yolov3.cfg
Layer [shortcut] not implemented

```


## 修改tensorflow的官方例子

* 下载
```
git clone https://github.com/tensorflow/tensorflow.git

```

* 使用android studio打开项目
* 将上面产生的pb文件拷贝到asset目录下
* build.gradle 我们不编译tensorflow的代码，而是把tensorflow作为一个ARR包从JCenter直接导入，这样比较简单。相当于我们把tensorflow作为一个库来使用，修改如下：
 ```
def nativeBuildSystem = 'none'
 ```

 * 修改DetectorActivity.java代码,将其中的文件名改成自己的pb文件名
 ```
   private static final String YOLO_MODEL_FILE = "file:///android_asset/graph-tiny-yolo-voc.pb";
 ```
 将探测器改成yolo 
  ```
  private static final DetectorMode MODE = DetectorMode.YOLO;
 ```

* 修改TensorFlowYoloDetector.java的对象标签,比如我只有一个对象叫truck改成：
```
private static final String[] LABELS = {
         "truck"
 };
 
```

## 结束
过程并不复杂，但挺烦琐，需要多操作几次。darkflow的参数写错往往会报各种错误，跟踪一下代码，一般都可以知道为什么。

参考：

https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android
https://pjreddie.com/darknet/yolo/
https://github.com/thtrieu/darkflow
