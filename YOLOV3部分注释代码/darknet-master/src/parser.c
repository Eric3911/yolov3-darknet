#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include "activation_layer.h"
#include "logistic_layer.h"
#include "l2norm_layer.h"
#include "activations.h"
#include "avgpool_layer.h"
#include "batchnorm_layer.h"
#include "blas.h"
#include "connected_layer.h"
#include "deconvolutional_layer.h"
#include "convolutional_layer.h"
#include "cost_layer.h"
#include "crnn_layer.h"
#include "crop_layer.h"
#include "detection_layer.h"
#include "dropout_layer.h"
#include "gru_layer.h"
#include "list.h"
#include "local_layer.h"
#include "maxpool_layer.h"
#include "normalization_layer.h"
#include "option_list.h"
#include "parser.h"
#include "region_layer.h"
#include "yolo_layer.h"
#include "iseg_layer.h"
#include "reorg_layer.h"
#include "rnn_layer.h"
#include "route_layer.h"
#include "upsample_layer.h"
#include "shortcut_layer.h"
#include "softmax_layer.h"
#include "lstm_layer.h"
#include "utils.h"

typedef struct{
    char *type;
    list *options;
}section;

list *read_cfg(char *filename);



/****************************************************************************
*  func: 返回type所对应的LAYER_TYPE的枚举名称                               *
*  args type：LAYER_TYPE的cfg文件对应名称                                   *
****************************************************************************/
LAYER_TYPE string_to_layer_type(char * type)
{

    if (strcmp(type, "[shortcut]")==0) return SHORTCUT;
    if (strcmp(type, "[crop]")==0) return CROP;
    if (strcmp(type, "[cost]")==0) return COST;
    if (strcmp(type, "[detection]")==0) return DETECTION;
    if (strcmp(type, "[region]")==0) return REGION;
    if (strcmp(type, "[yolo]")==0) return YOLO;
    if (strcmp(type, "[iseg]")==0) return ISEG;
    if (strcmp(type, "[local]")==0) return LOCAL;
    if (strcmp(type, "[conv]")==0
            || strcmp(type, "[convolutional]")==0) return CONVOLUTIONAL;
    if (strcmp(type, "[deconv]")==0
            || strcmp(type, "[deconvolutional]")==0) return DECONVOLUTIONAL;
    if (strcmp(type, "[activation]")==0) return ACTIVE;
    if (strcmp(type, "[logistic]")==0) return LOGXENT;
    if (strcmp(type, "[l2norm]")==0) return L2NORM;
    if (strcmp(type, "[net]")==0
            || strcmp(type, "[network]")==0) return NETWORK;
    if (strcmp(type, "[crnn]")==0) return CRNN;
    if (strcmp(type, "[gru]")==0) return GRU;
    if (strcmp(type, "[lstm]") == 0) return LSTM;
    if (strcmp(type, "[rnn]")==0) return RNN;
    if (strcmp(type, "[conn]")==0
            || strcmp(type, "[connected]")==0) return CONNECTED;
    if (strcmp(type, "[max]")==0
            || strcmp(type, "[maxpool]")==0) return MAXPOOL;
    if (strcmp(type, "[reorg]")==0) return REORG;
    if (strcmp(type, "[avg]")==0
            || strcmp(type, "[avgpool]")==0) return AVGPOOL;
    if (strcmp(type, "[dropout]")==0) return DROPOUT;
    if (strcmp(type, "[lrn]")==0
            || strcmp(type, "[normalization]")==0) return NORMALIZATION;
    if (strcmp(type, "[batchnorm]")==0) return BATCHNORM;
    if (strcmp(type, "[soft]")==0
            || strcmp(type, "[softmax]")==0) return SOFTMAX;
    if (strcmp(type, "[route]")==0) return ROUTE;
    if (strcmp(type, "[upsample]")==0) return UPSAMPLE;
    return BLANK;
}



/****************************************************************************
*  func：释放掉section s所指链表的每一个节点，包括头指针                    *
*  args s：欲释放的链表头指针                                               *
****************************************************************************/
void free_section(section *s)
{
	// 释放section的type域
    free(s->type);
	// 得到section的options域所指的第一个结点
    node *n = s->options->front;
	// 依次释放链表中的每一个kvp结点
    while(n){
        kvp *pair = (kvp *)n->val;
        free(pair->key);
        free(pair);
        node *next = n->next;
        free(n);
        n = next;
    }
	// 所有成员信息释放完后，释放本结点所占的空间
    free(s->options);
    free(s);
}




void parse_data(char *data, float *a, int n)
{
    int i;
    if(!data) return;
    char *curr = data;
    char *next = data;
    int done = 0;
    for(i = 0; i < n && !done; ++i){
        while(*++next !='\0' && *next != ',');
        if(*next == '\0') done = 1;
        *next = '\0';
        sscanf(curr, "%g", &a[i]);
        curr = next+1;
    }
}



typedef struct size_params{
    int batch;
    int inputs;
    int h;
    int w;
    int c;
    int index;
    int time_steps;
    network *net;
} size_params;



local_layer parse_local(list *options, size_params params)
{
    int n = option_find_int(options, "filters",1);
    int size = option_find_int(options, "size",1);
    int stride = option_find_int(options, "stride",1);
    int pad = option_find_int(options, "pad",0);
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before local layer must output image.");

    local_layer layer = make_local_layer(batch,h,w,c,n,size,stride,pad,activation);

    return layer;
}

layer parse_deconvolutional(list *options, size_params params)
{
    int n = option_find_int(options, "filters",1);
    int size = option_find_int(options, "size",1);
    int stride = option_find_int(options, "stride",1);

    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before deconvolutional layer must output image.");
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);
    int pad = option_find_int_quiet(options, "pad",0);
    int padding = option_find_int_quiet(options, "padding",0);
    if(pad) padding = size/2;

    layer l = make_deconvolutional_layer(batch,h,w,c,n,size,stride,padding, activation, batch_normalize, params.net->adam);

    return l;
}



/***************************************************************************
*  func：完成一层卷积层的初始化工作，返回初始化后的网络层，其中会指定前向  *
*        和后向传递所调用的函数指针以及初始化网络的一些配置参数，初始化完  *
*        毕后向屏幕输出该层网络信息                                        *
*  args options：sections的节点，里面包含了每一层网络的配置参数            *
***************************************************************************/
convolutional_layer parse_convolutional(list *options, size_params params)
{
	// 注意，src/convolutional_layer.h中有这样一句话：typedef layer convolutional_layer;说明
	// convolutional_layer类型就是include/darknet.h中定义的layer类型
	
	// 获取卷积核个数，若配置文件中没有指定，则设为1，找到后kvp结点的used会被置为1，下同
    int n = option_find_int(options, "filters",1);
	// 获取卷积核尺寸，若配置文件中没有指定，则设为1
    int size = option_find_int(options, "size",1);
	// 获取跨度即卷积步长，若配置文件中没有指定，则设为1
    int stride = option_find_int(options, "stride",1);
	// 是否在输入图像四周补0,若需要补0,值为1；若配置文件中没有指定，则设为0,不补0
    int pad = option_find_int_quiet(options, "pad",0);
	// 四周补0的长读，下面这句代码多余，有if(pad)这句就够了，即强制补0的长度为size/2取整
    int padding = option_find_int_quiet(options, "padding",0);
	// yolov3.cfg文件中并没有指定groups的值，默认为1，此处groups实际用途是啥没弄明白。
    int groups = option_find_int_quiet(options, "groups", 1);
	// 如若需要补0,补0长度为卷积核一半长度（往下取整），这对应same补0策略
    if(pad) padding = size/2;

	// 获取该层使用的激活函数类型，若配置文件中没有指定，则使用logistic激活函数
    char *activation_s = option_find_str(options, "activation", "logistic");
	// 1、ACTIVATION是定义在include/darknet中的一种枚举类型
	// 2、get_activation函数：src/activation.c。返回activation对应的枚举名称，默认返回RELU
    ACTIVATION activation = get_activation(activation_s);

	// h,w,c为上一层的输出的高度/宽度/通道数（第一层的则是输入的图片的尺寸与通道数，也即net.h,net.w,net.c），batch所有层都一样（不变），
    // params.h,params.w,params.c及params.inputs在构建每一层之后都会更新为上一层相应的输出参数（参见parse_network_cfg()）
    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
	// 如果这三个数存在0值，那肯定有问题了，因为上一层（或者输入）必须不为0
    if(!(h && w && c)) error("Layer before convolutional layer must output image.");
	// 是否进行规范化，1表示进行规范化，若配置文件中没有指定，则设为0,即默认不进行规范化
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);
	// 是否对权重进行二值化，1表示进行二值化，若配置文件中没有指定，则设为0,即默认不进行二值化
    int binary = option_find_int_quiet(options, "binary", 0);
	// 是否对权重以及输入进行二值化，1表示是，若配置文件中没有指定，则设为0,即默认不进行二值化
    int xnor = option_find_int_quiet(options, "xnor", 0);

	// 以上已经获取到了构建一层卷积层的所有参数，现在可以用这些参数构建卷积层了
	// make_convolutional_layer函数：src/convolutional_layer.c 
	// 以第一层卷积层为例，实际调用格式为：
	// make_convolutional_layer(64,416,416,3,32,1,3,1,1,“leaky”, 1, 0, 0, 0);
	// 构建一层类型为conv或convolutional的layer并返回，注意细读构建卷积层的函数和理解前向和后向算法
    convolutional_layer layer = make_convolutional_layer(batch,h,w,c,n,groups,size,stride,padding,activation, batch_normalize, binary, xnor, params.net->adam);
	// 卷积层中是否翻转？
    layer.flipped = option_find_int_quiet(options, "flipped", 0);
    layer.dot = option_find_float_quiet(options, "dot", 0);

    return layer;
}

layer parse_crnn(list *options, size_params params)
{
    int output_filters = option_find_int(options, "output_filters",1);
    int hidden_filters = option_find_int(options, "hidden_filters",1);
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

    layer l = make_crnn_layer(params.batch, params.w, params.h, params.c, hidden_filters, output_filters, params.time_steps, activation, batch_normalize);

    l.shortcut = option_find_int_quiet(options, "shortcut", 0);

    return l;
}

layer parse_rnn(list *options, size_params params)
{
    int output = option_find_int(options, "output",1);
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

    layer l = make_rnn_layer(params.batch, params.inputs, output, params.time_steps, activation, batch_normalize, params.net->adam);

    l.shortcut = option_find_int_quiet(options, "shortcut", 0);

    return l;
}

layer parse_gru(list *options, size_params params)
{
    int output = option_find_int(options, "output",1);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

    layer l = make_gru_layer(params.batch, params.inputs, output, params.time_steps, batch_normalize, params.net->adam);
    l.tanh = option_find_int_quiet(options, "tanh", 0);

    return l;
}

layer parse_lstm(list *options, size_params params)
{
    int output = option_find_int(options, "output", 1);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

    layer l = make_lstm_layer(params.batch, params.inputs, output, params.time_steps, batch_normalize, params.net->adam);

    return l;
}

layer parse_connected(list *options, size_params params)
{
    int output = option_find_int(options, "output",1);
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

    layer l = make_connected_layer(params.batch, params.inputs, output, activation, batch_normalize, params.net->adam);
    return l;
}

layer parse_softmax(list *options, size_params params)
{
    int groups = option_find_int_quiet(options, "groups",1);
    layer l = make_softmax_layer(params.batch, params.inputs, groups);
    l.temperature = option_find_float_quiet(options, "temperature", 1);
    char *tree_file = option_find_str(options, "tree", 0);
    if (tree_file) l.softmax_tree = read_tree(tree_file);
    l.w = params.w;
    l.h = params.h;
    l.c = params.c;
    l.spatial = option_find_float_quiet(options, "spatial", 0);
    l.noloss =  option_find_int_quiet(options, "noloss", 0);
    return l;
}



/*******************************************
*  func：解析yolo层的mask参数到数组返回    *
*  args a：mask的字符串                    *
*  args num：该层有多少个anchor            *
*******************************************/
int *parse_yolo_mask(char *a, int *num)
{
    int *mask = 0;
    if(a){
        int len = strlen(a);
        int n = 1;
        int i;
		// 统计a中有几个数，每两个数之间用‘，’隔开
        for(i = 0; i < len; ++i){
            if (a[i] == ',') ++n;
        }
		// 为每一个mask开辟一个整型空间存储
        mask = calloc(n, sizeof(int));
        for(i = 0; i < n; ++i){
            int val = atoi(a);
            mask[i] = val;
            a = strchr(a, ',')+1;
        }
        *num = n;
    }
    return mask;
}



/****************************************************************
*  func：处理yolo层的函数，构建yolo层和初始化                   *
*  args options：yolo层的配置参数链表                           *
*  args params：上一层网络的输出信息                            *
****************************************************************/
layer parse_yolo(list *options, size_params params)
{
	// 类别数，COCO为80类，也可以自定义
    int classes = option_find_int(options, "classes", 20);
	// anchor的总个数，3*3=9，YOLOV3中三个地方需要anchor，具体结合论文理解
    int total = option_find_int(options, "num", 1);
	// num保存当前yolo层的anchor数目
    int num = total;

	// mask指定了该层yolo选用哪几个anchor，如0，1，2代表前3个
    char *a = option_find_str(options, "mask", 0);
	// parse_yolo_mask函数：本文件 解析mask字符串到整型数组，num的值在该函数改变
    int *mask = parse_yolo_mask(a, &num);
	// make_yolo_layer函数：src/yolo_layer.c 调用函数，构造yolo层，以第一个yolo层为例，实际调用为：
	// make_yolo_layer(param.batch,13,13,3,9,mask,80)
    layer l = make_yolo_layer(params.batch, params.w, params.h, num, total, mask, classes);
    assert(l.outputs == params.inputs);

	// 一张图片中最多训练的框的个数
    l.max_boxes = option_find_int_quiet(options, "max",90);
	// 改变训练图片宽高比范围的参数
    l.jitter = option_find_float(options, "jitter", .2);

	// 参与计算的IOU阈值大小.当预测的检测框与ground true的IOU大于ignore_thresh的时候，参与loss的计算，否则，检测框的不参与损失计算。
	// 参数目的和理解：目的是控制参与loss计算的检测框的规模，当ignore_thresh过于大，接近于1的时候，那么参与检测框回归loss的个数就会
	// 比较少，同时也容易造成过拟合；而如果ignore_thresh设置的过于小，那么参与计算的会数量规模就会很大。同时也容易在进行检测框回归的时候造成欠拟合。
    // 参数设置：一般选取0.5-0.7之间的一个值，之前的计算基础都是小尺度（13*13）用的是0.7，（26*26）用的是0.5。这次先将0.5更改为0.7。
	// 参考：https://www.e-learn.cn/content/qita/804953
    l.ignore_thresh = option_find_float(options, "ignore_thresh", .5);
    l.truth_thresh = option_find_float(options, "truth_thresh", 1);
	// 是否开启多尺度训练的参数，0为关闭，1为开启
    l.random = option_find_int_quiet(options, "random", 0);

    char *map_file = option_find_str(options, "map", 0);
    if (map_file) l.map = read_map(map_file);

	// 获取yolo层的anchor
    a = option_find_str(options, "anchors", 0);
    if(a){
        int len = strlen(a);
        int n = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (a[i] == ',') ++n;
        }
        for(i = 0; i < n; ++i){
            float bias = atof(a);
            l.biases[i] = bias;
            a = strchr(a, ',')+1;
        }
    }
    return l;
}

layer parse_iseg(list *options, size_params params)
{
    int classes = option_find_int(options, "classes", 20);
    int ids = option_find_int(options, "ids", 32);
    layer l = make_iseg_layer(params.batch, params.w, params.h, classes, ids);
    assert(l.outputs == params.inputs);
    return l;
}

layer parse_region(list *options, size_params params)
{
    int coords = option_find_int(options, "coords", 4);
    int classes = option_find_int(options, "classes", 20);
    int num = option_find_int(options, "num", 1);

    layer l = make_region_layer(params.batch, params.w, params.h, num, classes, coords);
    assert(l.outputs == params.inputs);

    l.log = option_find_int_quiet(options, "log", 0);
    l.sqrt = option_find_int_quiet(options, "sqrt", 0);

    l.softmax = option_find_int(options, "softmax", 0);
    l.background = option_find_int_quiet(options, "background", 0);
    l.max_boxes = option_find_int_quiet(options, "max",30);
    l.jitter = option_find_float(options, "jitter", .2);
    l.rescore = option_find_int_quiet(options, "rescore",0);

    l.thresh = option_find_float(options, "thresh", .5);
    l.classfix = option_find_int_quiet(options, "classfix", 0);
    l.absolute = option_find_int_quiet(options, "absolute", 0);
    l.random = option_find_int_quiet(options, "random", 0);

    l.coord_scale = option_find_float(options, "coord_scale", 1);
    l.object_scale = option_find_float(options, "object_scale", 1);
    l.noobject_scale = option_find_float(options, "noobject_scale", 1);
    l.mask_scale = option_find_float(options, "mask_scale", 1);
    l.class_scale = option_find_float(options, "class_scale", 1);
    l.bias_match = option_find_int_quiet(options, "bias_match",0);

    char *tree_file = option_find_str(options, "tree", 0);
    if (tree_file) l.softmax_tree = read_tree(tree_file);
    char *map_file = option_find_str(options, "map", 0);
    if (map_file) l.map = read_map(map_file);

    char *a = option_find_str(options, "anchors", 0);
    if(a){
        int len = strlen(a);
        int n = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (a[i] == ',') ++n;
        }
        for(i = 0; i < n; ++i){
            float bias = atof(a);
            l.biases[i] = bias;
            a = strchr(a, ',')+1;
        }
    }
    return l;
}

detection_layer parse_detection(list *options, size_params params)
{
    int coords = option_find_int(options, "coords", 1);
    int classes = option_find_int(options, "classes", 1);
    int rescore = option_find_int(options, "rescore", 0);
    int num = option_find_int(options, "num", 1);
    int side = option_find_int(options, "side", 7);
    detection_layer layer = make_detection_layer(params.batch, params.inputs, num, side, classes, coords, rescore);

    layer.softmax = option_find_int(options, "softmax", 0);
    layer.sqrt = option_find_int(options, "sqrt", 0);

    layer.max_boxes = option_find_int_quiet(options, "max",90);
    layer.coord_scale = option_find_float(options, "coord_scale", 1);
    layer.forced = option_find_int(options, "forced", 0);
    layer.object_scale = option_find_float(options, "object_scale", 1);
    layer.noobject_scale = option_find_float(options, "noobject_scale", 1);
    layer.class_scale = option_find_float(options, "class_scale", 1);
    layer.jitter = option_find_float(options, "jitter", .2);
    layer.random = option_find_int_quiet(options, "random", 0);
    layer.reorg = option_find_int_quiet(options, "reorg", 0);
    return layer;
}

cost_layer parse_cost(list *options, size_params params)
{
    char *type_s = option_find_str(options, "type", "sse");
    COST_TYPE type = get_cost_type(type_s);
    float scale = option_find_float_quiet(options, "scale",1);
    cost_layer layer = make_cost_layer(params.batch, params.inputs, type, scale);
    layer.ratio =  option_find_float_quiet(options, "ratio",0);
    layer.noobject_scale =  option_find_float_quiet(options, "noobj", 1);
    layer.thresh =  option_find_float_quiet(options, "thresh",0);
    return layer;
}

crop_layer parse_crop(list *options, size_params params)
{
    int crop_height = option_find_int(options, "crop_height",1);
    int crop_width = option_find_int(options, "crop_width",1);
    int flip = option_find_int(options, "flip",0);
    float angle = option_find_float(options, "angle",0);
    float saturation = option_find_float(options, "saturation",1);
    float exposure = option_find_float(options, "exposure",1);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before crop layer must output image.");

    int noadjust = option_find_int_quiet(options, "noadjust",0);

    crop_layer l = make_crop_layer(batch,h,w,c,crop_height,crop_width,flip, angle, saturation, exposure);
    l.shift = option_find_float(options, "shift", 0);
    l.noadjust = noadjust;
    return l;
}

layer parse_reorg(list *options, size_params params)
{
    int stride = option_find_int(options, "stride",1);
    int reverse = option_find_int_quiet(options, "reverse",0);
    int flatten = option_find_int_quiet(options, "flatten",0);
    int extra = option_find_int_quiet(options, "extra",0);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before reorg layer must output image.");

    layer layer = make_reorg_layer(batch,w,h,c,stride,reverse, flatten, extra);
    return layer;
}

maxpool_layer parse_maxpool(list *options, size_params params)
{
    int stride = option_find_int(options, "stride",1);
    int size = option_find_int(options, "size",stride);
    int padding = option_find_int_quiet(options, "padding", size-1);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before maxpool layer must output image.");

    maxpool_layer layer = make_maxpool_layer(batch,h,w,c,size,stride,padding);
    return layer;
}

avgpool_layer parse_avgpool(list *options, size_params params)
{
    int batch,w,h,c;
    w = params.w;
    h = params.h;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before avgpool layer must output image.");

    avgpool_layer layer = make_avgpool_layer(batch,w,h,c);
    return layer;
}

dropout_layer parse_dropout(list *options, size_params params)
{
    float probability = option_find_float(options, "probability", .5);
    dropout_layer layer = make_dropout_layer(params.batch, params.inputs, probability);
    layer.out_w = params.w;
    layer.out_h = params.h;
    layer.out_c = params.c;
    return layer;
}

layer parse_normalization(list *options, size_params params)
{
    float alpha = option_find_float(options, "alpha", .0001);
    float beta =  option_find_float(options, "beta" , .75);
    float kappa = option_find_float(options, "kappa", 1);
    int size = option_find_int(options, "size", 5);
    layer l = make_normalization_layer(params.batch, params.w, params.h, params.c, size, alpha, beta, kappa);
    return l;
}

layer parse_batchnorm(list *options, size_params params)
{
    layer l = make_batchnorm_layer(params.batch, params.w, params.h, params.c);
    return l;
}



/**********************************************************************
*  func: 处理残差模块层的相应函数                                     *
*  args options: 该层配置参数的链表                                   *
*  args params: 网络上一层的相关信息                                  *
*  args net：已解析的网络层                                           *
**********************************************************************/
layer parse_shortcut(list *options, size_params params, network *net)
{
	// 链接的上册网络相对索引值，一般为负数，以第一个shortcut层为例，其值为-3
    char *l = option_find(options, "from");
    int index = atoi(l);
	// params.index记录了当前网络层的索引号，从0开始，以第一个shortcut层为例，此时params.index=4
	// index的值就等于：4+（-3）=1，可以对照网络结构输出信息进行验证
    if(index < 0) index = params.index + index;

	// batch会始终等于net->batch
    int batch = params.batch;
	// 此时你可以对照网络层输出信息看，本层的上一层输出尺寸应该和net->layers[index]的尺寸一致
	// 例如第一个res上一层的尺寸为：208*208*64 而索引为1的层的尺寸为：208*208*64（注意，这儿的例子是在cfg的配置为416*416的前提下）
	// 此时from是浅复制net->layers的值
    layer from = net->layers[index];

	// make_shortcut_layer函数：src/shotcut_layer.c 构造一层shotcut层，指定前向后向等参数
    layer s = make_shortcut_layer(batch, index, params.w, params.h, params.c, from.out_w, from.out_h, from.out_c);

    char *activation_s = option_find_str(options, "activation", "linear");
    ACTIVATION activation = get_activation(activation_s);
    s.activation = activation;
    s.alpha = option_find_float_quiet(options, "alpha", 1);
    s.beta = option_find_float_quiet(options, "beta", 1);
    return s;
}


layer parse_l2norm(list *options, size_params params)
{
    layer l = make_l2norm_layer(params.batch, params.inputs);
    l.h = l.out_h = params.h;
    l.w = l.out_w = params.w;
    l.c = l.out_c = params.c;
    return l;
}


layer parse_logistic(list *options, size_params params)
{
    layer l = make_logistic_layer(params.batch, params.inputs);
    l.h = l.out_h = params.h;
    l.w = l.out_w = params.w;
    l.c = l.out_c = params.c;
    return l;
}

layer parse_activation(list *options, size_params params)
{
    char *activation_s = option_find_str(options, "activation", "linear");
    ACTIVATION activation = get_activation(activation_s);

    layer l = make_activation_layer(params.batch, params.inputs, activation);

    l.h = l.out_h = params.h;
    l.w = l.out_w = params.w;
    l.c = l.out_c = params.c;

    return l;
}

layer parse_upsample(list *options, size_params params, network *net)
{

    int stride = option_find_int(options, "stride",2);
    layer l = make_upsample_layer(params.batch, params.w, params.h, params.c, stride);
    l.scale = option_find_float_quiet(options, "scale", 1);
    return l;
}

route_layer parse_route(list *options, size_params params, network *net)
{
    char *l = option_find(options, "layers");
    int len = strlen(l);
    if(!l) error("Route Layer must specify input layers");
    int n = 1;
    int i;
    for(i = 0; i < len; ++i){
        if (l[i] == ',') ++n;
    }

    int *layers = calloc(n, sizeof(int));
    int *sizes = calloc(n, sizeof(int));
    for(i = 0; i < n; ++i){
        int index = atoi(l);
        l = strchr(l, ',')+1;
        if(index < 0) index = params.index + index;
        layers[i] = index;
        sizes[i] = net->layers[index].outputs;
    }
    int batch = params.batch;

    route_layer layer = make_route_layer(batch, n, layers, sizes);

    convolutional_layer first = net->layers[layers[0]];
    layer.out_w = first.out_w;
    layer.out_h = first.out_h;
    layer.out_c = first.out_c;
    for(i = 1; i < n; ++i){
        int index = layers[i];
        convolutional_layer next = net->layers[index];
        if(next.out_w == first.out_w && next.out_h == first.out_h){
            layer.out_c += next.out_c;
        }else{
            layer.out_h = layer.out_w = layer.out_c = 0;
        }
    }

    return layer;
}



/******************************************************************
*  func：判断学习率调整策略的函数                                 *
*  args s: 学习率调整策略的名称                                   *
*  details：返回学习率调整策略或默认值CONSTANT                    *
******************************************************************/
learning_rate_policy get_policy(char *s)
{
    if (strcmp(s, "random")==0) return RANDOM;
    if (strcmp(s, "poly")==0) return POLY;
    if (strcmp(s, "constant")==0) return CONSTANT;
    if (strcmp(s, "step")==0) return STEP;
    if (strcmp(s, "exp")==0) return EXP;
    if (strcmp(s, "sigmoid")==0) return SIG;
    if (strcmp(s, "steps")==0) return STEPS;
    fprintf(stderr, "Couldn't find policy %s, going with constant\n", s);
    return CONSTANT;
}



/**********************************************************************
*  func：解析链表options中的参数，options此时指向[net]块中的详细参数  *
*  args options：待解析的链表                                         *
*  args net：解析的结果赋值给网络net，通过指针引用保存修改结果        *
**********************************************************************/
void parse_net_options(list *options, network *net)
{
	// 从.cfg网络参数配置文件中读入一些通用的网络配置参数，option_find_int()以及option_find_float()
	// 函数的第三个参数都是默认值（如果配置文件中没有设置该参数的值，就取默认值）
    // 稍微提一下batch这个参数，首先读入的net->batch是真实batch值，即每个batch中包含的照片张数
	// ，而后又读入一个subdivisions参数，这个参数有很大的用处，读者可以继续跟进程序以便理解，
    // 总之最终的net->batch = net->batch / net->subdivisions
	// option_find_int函数和option_find_float等函数都定义在src/option_list.c中
	// 如下所示读入的参数，意义简明，不再赘述。每个参数读取完后，会改变其used属性值为1
    net->batch = option_find_int(options, "batch",1);
    net->learning_rate = option_find_float(options, "learning_rate", .001);
    net->momentum = option_find_float(options, "momentum", .9);
    net->decay = option_find_float(options, "decay", .0001);
    int subdivs = option_find_int(options, "subdivisions",1);
	
	// option_find_int_quiet在没有该参数系统使用默认参数的时候不会向屏幕打印使用了默认参数的提示信息
    net->time_steps = option_find_int_quiet(options, "time_steps",1);
    net->notruth = option_find_int_quiet(options, "notruth",0);
    net->batch /= subdivs;
	
	// yolov3.cfg的time_steps并没有指定，故使用默认值1
    net->batch *= net->time_steps;
    net->subdivisions = subdivs;
    net->random = option_find_int_quiet(options, "random", 0);

	// 查看是否使用了adam，yolov3中并没有使用
    net->adam = option_find_int_quiet(options, "adam", 0);
    if(net->adam){
        net->B1 = option_find_float(options, "B1", .9);
        net->B2 = option_find_float(options, "B2", .999);
        net->eps = option_find_float(options, "eps", .0000001);
    }
	
    // 读取网络的weight、height、channels等信息
    net->h = option_find_int_quiet(options, "height",0);
    net->w = option_find_int_quiet(options, "width",0);
    net->c = option_find_int_quiet(options, "channels",0);
	
	// 一张输入图片的元素个数，如果网络配置文件没有指定，则默认值为net->h * net->w * net->c
	// 以下的参数具体意义有待补充。应该是图像增强时使用的配置参数，在读取数据的时候会用到
    net->inputs = option_find_int_quiet(options, "inputs", net->h * net->w * net->c);
    net->max_crop = option_find_int_quiet(options, "max_crop",net->w*2);
    net->min_crop = option_find_int_quiet(options, "min_crop",net->w);
    net->max_ratio = option_find_float_quiet(options, "max_ratio", (float) net->max_crop / net->w);
    net->min_ratio = option_find_float_quiet(options, "min_ratio", (float) net->min_crop / net->w);
    net->center = option_find_int_quiet(options, "center",0);
    net->clip = option_find_float_quiet(options, "clip", 0);
    net->angle = option_find_float_quiet(options, "angle", 0);
    net->aspect = option_find_float_quiet(options, "aspect", 1);
    net->saturation = option_find_float_quiet(options, "saturation", 1);
    net->exposure = option_find_float_quiet(options, "exposure", 1);
    net->hue = option_find_float_quiet(options, "hue", 0);

	// 如果网络的width和height以及channels的任意一个定义有问题，都会抛出错误
    if(!net->inputs && !(net->h && net->w && net->c)) error("No input parameters supplied");

	// yolov3中: policy=steps, steps=400000,450000,  scales=.1,.1
    char *policy_s = option_find_str(options, "policy", "constant");
	// net->policy这个参数是learning_rate_policy枚举类型，定义在include/darknet.h中
	// get_policy函数：本文件。判断policy_s是属于哪一种，返回对应枚举值。都不是返回默认值
	// 此处：net->policy=‘STEPS’
    net->policy = get_policy(policy_s);
	
	// 找到burn_in参数的值，burn_in表示学习率的调整点，当训练次数小于burn_in时为一种调整策略，大于
	// burn_in过后又是另外一种调整策略
    net->burn_in = option_find_int_quiet(options, "burn_in", 0);
    net->power = option_find_float_quiet(options, "power", 4);
	
	// 根据学习率调整策略的不同，继续读入其它参数
    if(net->policy == STEP){
        net->step = option_find_int(options, "step", 1);
        net->scale = option_find_float(options, "scale", 1);
		
    } else if (net->policy == STEPS){
		// yolov3.cfg中的policy就是steps，且steps=400000,450000,  scales=.1,.1
        char *l = option_find(options, "steps");
        char *p = option_find(options, "scales");
        if(!l || !p) error("STEPS policy must have steps and scales in cfg file");

        int len = strlen(l);
        int n = 1;
        int i;
		// 统计steps中有几个调整点
        for(i = 0; i < len; ++i){
            if (l[i] == ',') ++n;
        }
        int *steps = calloc(n, sizeof(int));
        float *scales = calloc(n, sizeof(float));
		// 每次循环取一个值，以‘，’为取值截点
        for(i = 0; i < n; ++i){
            int step    = atoi(l);
            float scale = atof(p);
			// strchr是计算机编程语言的一个函数，原型为extern char *strchr(const char *s,char c)，
			// 可以查找字符串s中首次出现字符c的位置。
            l = strchr(l, ',')+1;
            p = strchr(p, ',')+1;
            steps[i] = step;
            scales[i] = scale;
        }
		// 此时scales中保存的是调整值的数组，steps中保存的是调整步数的数组，num_steps表示调整几次
        net->scales = scales;
        net->steps = steps;
        net->num_steps = n;
		
    } else if (net->policy == EXP){
        net->gamma = option_find_float(options, "gamma", 1);
    } else if (net->policy == SIG){
        net->gamma = option_find_float(options, "gamma", 1);
        net->step = option_find_int(options, "step", 1);
    } else if (net->policy == POLY || net->policy == RANDOM){
    }
	
	// 网络的最大训练次数，控制网络训练自动结束
    net->max_batches = option_find_int(options, "max_batches", 0);
}



/******************************************************************
*  func：判断节点s的type是否为[net]或[network]                    *
*  details：是返回1，不是则返回0                                  *
******************************************************************/
int is_network(section *s)
{
    return (strcmp(s->type, "[net]")==0
            || strcmp(s->type, "[network]")==0);
}



/***********************************************************************
*  func：从神经网络结构参数文件中读入所有神经网络层的结构参数，存储到  *
*        sections中，sections的每个node包含一层神经网络的所有结构参数  *
*        随后调用make_network函数创建网络并初始化参数，依次解析网络的  *
*        每一层类型和功能，返回构造好的网络                            *
*  args filename: 参数文件路径                                         *
*  details：这里比较复杂的是sections中间的内容，一定要理清数据结构，可 *
*           以结合网上的图形化描述加以理解，这儿只是简单描述           *
***********************************************************************/
network *parse_network_cfg(char *filename)
{
	// read_cfg函数：本文件。具体读取参数和解析的函数，返回的section指向链表头节点
	// 现在简要描述sections中的具体数据结构，望读者一定加以理解：
	// 首先将相关的数据结构定义拷贝过来，以便对照查看
	/*
	typedef struct node{
		void *val;
		struct node *next;
		struct node *prev;
	} node;

	typedef struct list{
		int size;
		node *front;
		node *back;
	} list;
	
	typedef struct{
		char *key;
		char *val;
		int used;
	} kvp;
	
	typedef struct{
		char *type;
		list *options;
	}section;	
	sections是指向链表的头结点，sections双向链表的每一个结点都是node类型，表示网络的一层，sections的front指针指向链表的第一个
	结点，back指针指向链表的最后一个节点，size表示当前链表一共有多少个节点，这个计数不包括头节点，插入链表时采用的是
	尾插法。其次分析sections的每一个结点node，该结点的val域为结构体类型section,next指针指向下一个结点，prev指针指向上
	一个结点，当Node为第一个节点时，prev指针为空，为最后一个结点时，next指针为空。现在注意分析Node结点的val域，对于一个
	section结点，其type说明了该层网络的类型，比如[conv]或[yolo]等，这儿要特别注意section的option域，其又是一个指向双向链表头结点
	的指针，而这个双向链表与sections双向链表有所不同，主要体现在链表中每一个结点node的val域上，以及每个node代表的意义，sections的结点
	node的val域已经分析了，再来看options链表结点node的val域，其是一个kvp结构类型的结构体，kvp的key表示诸如 learning_rate=0.001中的
	learning_rate,而kvp的val则表示数值0.00，这个比较好理解，used表示该结点是否使用过，初始化时均为0，表示没使用过。对于node的意义，options
	中每一个node则表示每一层神经网络的每一个参数配置说明，读者可以根据解析自行画图加以理解，多看几遍也能记住
	*/
    list *sections = read_cfg(filename);
		
	// 获取sections的第一个节点，可以查看一下cfg/***.cfg文件，其实第一块参数（以[net]开头）不是某层神经网络的参数，
    // 而是关于整个网络的一些通用参数，比如学习率，衰减率，输入图像宽高，batch大小等，
    // 具体的关于某个网络层的参数是从第二块开始的，如[convolutional],[maxpool]...，
    // 这些层并没有编号，只说明了层的属性，但层的参数都是按顺序在文件中排好的，读入时，
    // sections链表上的顺序就是文件中的排列顺序。
    node *n = sections->front;
	// 如果结点内容为空，说明cfg文件中没有符合规则的配置文件内容
    if(!n) error("Config file has no sections");
	
	// 1、network数据结构定义在include/darknet.h中
	// 2、make_network函数：src/network.c。创建网络结构并动态分配内存：输入网络层数为sections->size - 1，
	// sections的第一段不是网络层，而是通用网络参数，sections->size为sections的节点个数，得到了网络的首地址
    network *net = make_network(sections->size - 1);
	
	// 所用显卡的卡号（gpu_index在cuda.c中用extern关键字声明）
    // 在调用parse_network_cfg()之前，使用了cuda_set_device()设置了gpu_index的值号为当前活跃GPU卡号
    net->gpu_index = gpu_index;
	
	// size_params结构体定义在本文件中
	/*
	typedef struct size_params{
		int batch;
		int inputs;
		int h;
		int w;
		int c;
		int index;
		int time_steps;
		network *net;
	} size_params;
	*/
    size_params params;
	
    // 指针s指向第一个节点node的val域，正如上面所述，每个结点node的val域是section类型的，所以需要强制转化数据类型
    section *s = (section *)n->val;
	// s->options为一个链表，是cfg文件中每个网络块的详细参数链表头节点指针
    list *options = s->options;
	// is_network函数：本文件。判断第一个节点是否为[net]或[network]，根据section.type来判断
    if(!is_network(s)) error("First section must be [net] or [network]");
	// parse_net_options函数：本文件。将链表options中对于[net]中的参数解析并赋值给net，详细注释见函数定义
    parse_net_options(options, net);

	// 此时的net->h / w / c 表示网络输入的width/height/channel，params的成员变量值会在每处理一层网络后作出调整
    params.h = net->h;
    params.w = net->w;
    params.c = net->c;
    params.inputs = net->inputs;
    params.batch = net->batch;
	// yolov3.cfg中并没有指定time_steps，取默认值1
    params.time_steps = net->time_steps;
	// 注意，此处的net是network类型的指针变量，即params.net与net指向同一块地址
    params.net = net;

    size_t workspace_size = 0;
	// 将指针指向实际网络的第一层，以yolov3.cfg为例，此时为：[convolutional]
    n = n->next;
    int count = 0;
	// free_section函数：本文件，释放section类型的结点，此时会连同section型结点的options链表域一并释放
    free_section(s);
	// 此处stderr不是错误提示，而是输出结果提示，提示网络结构
    fprintf(stderr, "layer     filters    size              input                output\n");
	// 此段代码就是我们运行yolov3开始显示的那段花里胡哨的东西，建议细读
	// 循环处理网络结构中的每一层
    while(n){
		// index指示当前处理的网络层次索引
        params.index = count;
		// 此时输出的count就是我们在加载网络的时候看到的网络层编号，右对齐，占5位，从0开始
        fprintf(stderr, "%5d ", count);
		// 重新将s指向section类型的结点，表示网络每一层的配置信息
        s = (section *)n->val;
        options = s->options;
		// 定义网络层， layer的结构体定义在include/darknet.h中，该结构体十分复杂
        layer l = {0};
		// 1、LAYER_TYPE是定义在include/darknet.h中的枚举类型
		// 2、string_to_layer_type函数：本文件。返回s->type所对应的LAYER_TYPE的枚举名称
        LAYER_TYPE lt = string_to_layer_type(s->type);
		
		// 以下的parse_**函数：均定义在本文件。详细注释看函数定义出
        if(lt == CONVOLUTIONAL){
			// 当网络层类型为卷积层时，调用该函数，为其它网络层类型是分别查看相应函数
            l = parse_convolutional(options, params);
        }else if(lt == DECONVOLUTIONAL){
            l = parse_deconvolutional(options, params);
        }else if(lt == LOCAL){
            l = parse_local(options, params);
        }else if(lt == ACTIVE){
            l = parse_activation(options, params);
        }else if(lt == LOGXENT){
            l = parse_logistic(options, params);
        }else if(lt == L2NORM){
            l = parse_l2norm(options, params);
        }else if(lt == RNN){
            l = parse_rnn(options, params);
        }else if(lt == GRU){
            l = parse_gru(options, params);
        }else if (lt == LSTM) {
            l = parse_lstm(options, params);
        }else if(lt == CRNN){
            l = parse_crnn(options, params);
        }else if(lt == CONNECTED){
            l = parse_connected(options, params);
        }else if(lt == CROP){
            l = parse_crop(options, params);
        }else if(lt == COST){
            l = parse_cost(options, params);
        }else if(lt == REGION){
            l = parse_region(options, params);
        }else if(lt == YOLO){
            l = parse_yolo(options, params);
        }else if(lt == ISEG){
            l = parse_iseg(options, params);
        }else if(lt == DETECTION){
            l = parse_detection(options, params);
        }else if(lt == SOFTMAX){
            l = parse_softmax(options, params);
            net->hierarchy = l.softmax_tree;
        }else if(lt == NORMALIZATION){
            l = parse_normalization(options, params);
        }else if(lt == BATCHNORM){
            l = parse_batchnorm(options, params);
        }else if(lt == MAXPOOL){
            l = parse_maxpool(options, params);
        }else if(lt == REORG){
            l = parse_reorg(options, params);
        }else if(lt == AVGPOOL){
            l = parse_avgpool(options, params);
        }else if(lt == ROUTE){
            l = parse_route(options, params, net);
        }else if(lt == UPSAMPLE){
            l = parse_upsample(options, params, net);
        }else if(lt == SHORTCUT){
            l = parse_shortcut(options, params, net);
        }else if(lt == DROPOUT){
            l = parse_dropout(options, params);
            l.output = net->layers[count-1].output;
            l.delta = net->layers[count-1].delta;
#ifdef GPU
            l.output_gpu = net->layers[count-1].output_gpu;
            l.delta_gpu = net->layers[count-1].delta_gpu;
#endif
        }else{
			// 以上类型均不是的时候，输出错误提示信息
            fprintf(stderr, "Type not recognized: %s\n", s->type);
        }
        
		l.clip = net->clip;
		// l.truth保存了图片的标签信息，通常在输出层才会用到
		// 如下的参数读者可以根据自己的需要进行设置并实验
        l.truth = option_find_int_quiet(options, "truth", 0);
        l.onlyforward = option_find_int_quiet(options, "onlyforward", 0);
        l.stopbackward = option_find_int_quiet(options, "stopbackward", 0);
        l.dontsave = option_find_int_quiet(options, "dontsave", 0);
        l.dontload = option_find_int_quiet(options, "dontload", 0);
        l.numload = option_find_int_quiet(options, "numload", 0);
        l.dontloadscales = option_find_int_quiet(options, "dontloadscales", 0);
        l.learning_rate_scale = option_find_float_quiet(options, "learning_rate", 1);
        l.smooth = option_find_float_quiet(options, "smooth", 0);
		// option_unused函数：src/option_list.c 输出options列表中的used=0的节点，也就是没访问过的节点
        option_unused(options);
		// 将所有参数已经解析和处理后的layer l 的地址赋值给net->layers[count]，即将网络层添加进网络
        net->layers[count] = l;
		
		// workspace_size始终保持网络中最大网络层的workspace_size
        if (l.workspace_size > workspace_size) workspace_size = l.workspace_size;
		// free_section函数：本文件。释放node结点的val域
        free_section(s);
		// 指向下一个节点，继续处理
        n = n->next;
        ++count;
		// 构建每一层之后，如果之后还有层，则更新params.h,params.w,params.c及params.inputs为上一层相应的输出参数
        if(n){
            params.h = l.out_h;
            params.w = l.out_w;
            params.c = l.out_c;
            params.inputs = l.outputs;
        }
    }
    // 当所有节点都处理完后，网络信息已经储存在net中，便可以释放sections所指向的链表了
	free_list(sections);
	// get_network_output_layer函数：src/network.c 获取net的输出层out，此处得到的是Yolo中的最后一个yolo层
    layer out = get_network_output_layer(net);
	
    net->outputs = out.outputs;
    net->truths = out.outputs;
	// 程序并不执行该条语句，因为yolo的最后一个yolo层并没有定义truth
    if(net->layers[net->n-1].truths) net->truths = net->layers[net->n-1].truths;
    net->output = out.output;
	// 为网络的输入和标签分配空间，注意要乘以batch
    net->input = calloc(net->inputs*net->batch, sizeof(float));
    net->truth = calloc(net->truths*net->batch, sizeof(float));
#ifdef GPU
    net->output_gpu = out.output_gpu;
    net->input_gpu = cuda_make_array(net->input, net->inputs*net->batch);
    net->truth_gpu = cuda_make_array(net->truth, net->truths*net->batch);
#endif
    if(workspace_size){
        //printf("%ld\n", workspace_size);
#ifdef GPU
        if(gpu_index >= 0){
            net->workspace = cuda_make_array(0, (workspace_size-1)/sizeof(float)+1);
        }else {
            net->workspace = calloc(1, workspace_size);
        }
#else
        net->workspace = calloc(1, workspace_size);
#endif
    }
    return net;
}



/************************************************************************
*  func：读取神经网络结构配置文件（.cfg文件）中的配置数据，将每个神     *
*        经网络层参数读取到每个options结构体中，（每个section是options  *
*        的一个节点）而后全部插入到list结构体options中并返回            *
*  args filename: 神经网络配置文件路径名                                *
*  details：返回从神经网络结构配置文件中读入的所有神经网络层的参数 。具 *
*           体处理时是先建立一个options链表头指针，然后扫描配置文件的每 *
*           一行，每遇到一个[xxx]开始的代码块便开辟一个新的section结构体*
*           来表示神经网络的网络功能块，继续读功能块中的参数配置，这些参*
*           数配置以链表+节点的形式存放在sections的options域            *
************************************************************************/
list *read_cfg(char *filename)
{
	// 打开文件读写指针，打开失败报错退出
    FILE *file = fopen(filename, "r");
    if(file == 0) file_error(filename);
	
    char *line;
    int nu = 0;
	// 动态分配list对象内存，并初始化options所有元素值为0
    // options包含所有的section，也即包含所有的神经网络层参数
    list *options = make_list();
	// section数据结构定义在本文件中。
    section *current = 0;
	// 注意本段的解析方式：扫描每一行cfg的内容，每遇到一个’[‘开头的行，则先建立一个section节点，并插入网络
	// 头指针所指向的链表options中，一个section代表cfg中的一块。结合cfg中的具体类容，比如[net]下面的若干行
	// 都是 key=val 的形式，此时将会把'='号前后的内容提取出来，生成一个个node类型的节点，这些节点构成的链表
	// 的头指针又由每个section节点的options域引用
    while((line=fgetl(file)) != 0){
        ++ nu;
		// 去除line中的空格'\t'和'\n'符号
        strip(line);
        switch(line[0]){
			// 以[开头的行是层的类别说明，比如[net],[maxpool],[convolutional]之类的
			// cfg中每一层的说明先以功能名称开头，所以[net]等类似内容直接作为type记录
            case '[':
				// 动态分配一个section内存给current
                current = malloc(sizeof(section));
				// 将单个section current插入section集合options中 current的类容会作为Node节点类型的val域
                list_insert(options, current);
				// 进一步动态的为current的元素options动态分配内存
                current->options = make_list();
				// 以[开头的是层的类别，赋值给type
                current->type = line;
                break;
			// 以下三种情况是无效行，直接释放内存即可（以#开头的是注释）
            case '\0':
            case '#':
            case ';':
                free(line);
                break;
			// 剩下的才真正是网络结构的数据，调用read_option()函数读取
            // 返回0说明文件中的数据格式有问题，将会提示错误
			// read_option函数：src/option_list.c 将line中的内容进行剥离，以=号分成两部分后调用
			// option_insert,插入到current->options中，有点类似于处理"cfg/coco.data"的数据
            default:
                if(!read_option(line, current->options)){
                    fprintf(stderr, "Config file error line %d, could parse: %s\n", nu, line);
                    free(line);
                }
                break;
        }
    }
	// 关闭打开的文件流，释放资源
    fclose(file);
    return options;
}



void save_convolutional_weights_binary(layer l, FILE *fp)
{
#ifdef GPU
    if(gpu_index >= 0){
        pull_convolutional_layer(l);
    }
#endif
    binarize_weights(l.weights, l.n, l.c*l.size*l.size, l.binary_weights);
    int size = l.c*l.size*l.size;
    int i, j, k;
    fwrite(l.biases, sizeof(float), l.n, fp);
    if (l.batch_normalize){
        fwrite(l.scales, sizeof(float), l.n, fp);
        fwrite(l.rolling_mean, sizeof(float), l.n, fp);
        fwrite(l.rolling_variance, sizeof(float), l.n, fp);
    }
    for(i = 0; i < l.n; ++i){
        float mean = l.binary_weights[i*size];
        if(mean < 0) mean = -mean;
        fwrite(&mean, sizeof(float), 1, fp);
        for(j = 0; j < size/8; ++j){
            int index = i*size + j*8;
            unsigned char c = 0;
            for(k = 0; k < 8; ++k){
                if (j*8 + k >= size) break;
                if (l.binary_weights[index + k] > 0) c = (c | 1<<k);
            }
            fwrite(&c, sizeof(char), 1, fp);
        }
    }
}

void save_convolutional_weights(layer l, FILE *fp)
{
    if(l.binary){
        //save_convolutional_weights_binary(l, fp);
        //return;
    }
#ifdef GPU
    if(gpu_index >= 0){
        pull_convolutional_layer(l);
    }
#endif
    int num = l.nweights;
    fwrite(l.biases, sizeof(float), l.n, fp);
    if (l.batch_normalize){
        fwrite(l.scales, sizeof(float), l.n, fp);
        fwrite(l.rolling_mean, sizeof(float), l.n, fp);
        fwrite(l.rolling_variance, sizeof(float), l.n, fp);
    }
    fwrite(l.weights, sizeof(float), num, fp);
}

void save_batchnorm_weights(layer l, FILE *fp)
{
#ifdef GPU
    if(gpu_index >= 0){
        pull_batchnorm_layer(l);
    }
#endif
    fwrite(l.scales, sizeof(float), l.c, fp);
    fwrite(l.rolling_mean, sizeof(float), l.c, fp);
    fwrite(l.rolling_variance, sizeof(float), l.c, fp);
}

void save_connected_weights(layer l, FILE *fp)
{
#ifdef GPU
    if(gpu_index >= 0){
        pull_connected_layer(l);
    }
#endif
    fwrite(l.biases, sizeof(float), l.outputs, fp);
    fwrite(l.weights, sizeof(float), l.outputs*l.inputs, fp);
    if (l.batch_normalize){
        fwrite(l.scales, sizeof(float), l.outputs, fp);
        fwrite(l.rolling_mean, sizeof(float), l.outputs, fp);
        fwrite(l.rolling_variance, sizeof(float), l.outputs, fp);
    }
}

void save_weights_upto(network *net, char *filename, int cutoff)
{
#ifdef GPU
    if(net->gpu_index >= 0){
        cuda_set_device(net->gpu_index);
    }
#endif
    fprintf(stderr, "Saving weights to %s\n", filename);
    FILE *fp = fopen(filename, "wb");
    if(!fp) file_error(filename);

    int major = 0;
    int minor = 2;
    int revision = 0;
    fwrite(&major, sizeof(int), 1, fp);
    fwrite(&minor, sizeof(int), 1, fp);
    fwrite(&revision, sizeof(int), 1, fp);
    fwrite(net->seen, sizeof(size_t), 1, fp);

    int i;
    for(i = 0; i < net->n && i < cutoff; ++i){
        layer l = net->layers[i];
        if (l.dontsave) continue;
        if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
            save_convolutional_weights(l, fp);
        } if(l.type == CONNECTED){
            save_connected_weights(l, fp);
        } if(l.type == BATCHNORM){
            save_batchnorm_weights(l, fp);
        } if(l.type == RNN){
            save_connected_weights(*(l.input_layer), fp);
            save_connected_weights(*(l.self_layer), fp);
            save_connected_weights(*(l.output_layer), fp);
        } if (l.type == LSTM) {
            save_connected_weights(*(l.wi), fp);
            save_connected_weights(*(l.wf), fp);
            save_connected_weights(*(l.wo), fp);
            save_connected_weights(*(l.wg), fp);
            save_connected_weights(*(l.ui), fp);
            save_connected_weights(*(l.uf), fp);
            save_connected_weights(*(l.uo), fp);
            save_connected_weights(*(l.ug), fp);
        } if (l.type == GRU) {
            if(1){
                save_connected_weights(*(l.wz), fp);
                save_connected_weights(*(l.wr), fp);
                save_connected_weights(*(l.wh), fp);
                save_connected_weights(*(l.uz), fp);
                save_connected_weights(*(l.ur), fp);
                save_connected_weights(*(l.uh), fp);
            }else{
                save_connected_weights(*(l.reset_layer), fp);
                save_connected_weights(*(l.update_layer), fp);
                save_connected_weights(*(l.state_layer), fp);
            }
        }  if(l.type == CRNN){
            save_convolutional_weights(*(l.input_layer), fp);
            save_convolutional_weights(*(l.self_layer), fp);
            save_convolutional_weights(*(l.output_layer), fp);
        } if(l.type == LOCAL){
#ifdef GPU
            if(gpu_index >= 0){
                pull_local_layer(l);
            }
#endif
            int locations = l.out_w*l.out_h;
            int size = l.size*l.size*l.c*l.n*locations;
            fwrite(l.biases, sizeof(float), l.outputs, fp);
            fwrite(l.weights, sizeof(float), size, fp);
        }
    }
    fclose(fp);
}
void save_weights(network *net, char *filename)
{
    save_weights_upto(net, filename, net->n);
}

void transpose_matrix(float *a, int rows, int cols)
{
    float *transpose = calloc(rows*cols, sizeof(float));
    int x, y;
    for(x = 0; x < rows; ++x){
        for(y = 0; y < cols; ++y){
            transpose[y*rows + x] = a[x*cols + y];
        }
    }
    memcpy(a, transpose, rows*cols*sizeof(float));
    free(transpose);
}

void load_connected_weights(layer l, FILE *fp, int transpose)
{
    fread(l.biases, sizeof(float), l.outputs, fp);
    fread(l.weights, sizeof(float), l.outputs*l.inputs, fp);
    if(transpose){
        transpose_matrix(l.weights, l.inputs, l.outputs);
    }
    //printf("Biases: %f mean %f variance\n", mean_array(l.biases, l.outputs), variance_array(l.biases, l.outputs));
    //printf("Weights: %f mean %f variance\n", mean_array(l.weights, l.outputs*l.inputs), variance_array(l.weights, l.outputs*l.inputs));
    if (l.batch_normalize && (!l.dontloadscales)){
        fread(l.scales, sizeof(float), l.outputs, fp);
        fread(l.rolling_mean, sizeof(float), l.outputs, fp);
        fread(l.rolling_variance, sizeof(float), l.outputs, fp);
        //printf("Scales: %f mean %f variance\n", mean_array(l.scales, l.outputs), variance_array(l.scales, l.outputs));
        //printf("rolling_mean: %f mean %f variance\n", mean_array(l.rolling_mean, l.outputs), variance_array(l.rolling_mean, l.outputs));
        //printf("rolling_variance: %f mean %f variance\n", mean_array(l.rolling_variance, l.outputs), variance_array(l.rolling_variance, l.outputs));
    }
#ifdef GPU
    if(gpu_index >= 0){
        push_connected_layer(l);
    }
#endif
}

void load_batchnorm_weights(layer l, FILE *fp)
{
    fread(l.scales, sizeof(float), l.c, fp);
    fread(l.rolling_mean, sizeof(float), l.c, fp);
    fread(l.rolling_variance, sizeof(float), l.c, fp);
#ifdef GPU
    if(gpu_index >= 0){
        push_batchnorm_layer(l);
    }
#endif
}

void load_convolutional_weights_binary(layer l, FILE *fp)
{
    fread(l.biases, sizeof(float), l.n, fp);
    if (l.batch_normalize && (!l.dontloadscales)){
        fread(l.scales, sizeof(float), l.n, fp);
        fread(l.rolling_mean, sizeof(float), l.n, fp);
        fread(l.rolling_variance, sizeof(float), l.n, fp);
    }
    int size = l.c*l.size*l.size;
    int i, j, k;
    for(i = 0; i < l.n; ++i){
        float mean = 0;
        fread(&mean, sizeof(float), 1, fp);
        for(j = 0; j < size/8; ++j){
            int index = i*size + j*8;
            unsigned char c = 0;
            fread(&c, sizeof(char), 1, fp);
            for(k = 0; k < 8; ++k){
                if (j*8 + k >= size) break;
                l.weights[index + k] = (c & 1<<k) ? mean : -mean;
            }
        }
    }
#ifdef GPU
    if(gpu_index >= 0){
        push_convolutional_layer(l);
    }
#endif
}

void load_convolutional_weights(layer l, FILE *fp)
{
    if(l.binary){
        //load_convolutional_weights_binary(l, fp);
        //return;
    }
    if(l.numload) l.n = l.numload;
    int num = l.c/l.groups*l.n*l.size*l.size;
    fread(l.biases, sizeof(float), l.n, fp);
    if (l.batch_normalize && (!l.dontloadscales)){
        fread(l.scales, sizeof(float), l.n, fp);
        fread(l.rolling_mean, sizeof(float), l.n, fp);
        fread(l.rolling_variance, sizeof(float), l.n, fp);
        if(0){
            int i;
            for(i = 0; i < l.n; ++i){
                printf("%g, ", l.rolling_mean[i]);
            }
            printf("\n");
            for(i = 0; i < l.n; ++i){
                printf("%g, ", l.rolling_variance[i]);
            }
            printf("\n");
        }
        if(0){
            fill_cpu(l.n, 0, l.rolling_mean, 1);
            fill_cpu(l.n, 0, l.rolling_variance, 1);
        }
        if(0){
            int i;
            for(i = 0; i < l.n; ++i){
                printf("%g, ", l.rolling_mean[i]);
            }
            printf("\n");
            for(i = 0; i < l.n; ++i){
                printf("%g, ", l.rolling_variance[i]);
            }
            printf("\n");
        }
    }
    fread(l.weights, sizeof(float), num, fp);
    //if(l.c == 3) scal_cpu(num, 1./256, l.weights, 1);
    if (l.flipped) {
        transpose_matrix(l.weights, l.c*l.size*l.size, l.n);
    }
    //if (l.binary) binarize_weights(l.weights, l.n, l.c*l.size*l.size, l.weights);
#ifdef GPU
    if(gpu_index >= 0){
        push_convolutional_layer(l);
    }
#endif
}



/****************************************************************
*  func：将filename所指向的权重文件加载进网络net                *
*  args net: 加载的网络                                         *
*  args filename: 权重文件的路径                                *
*  args start: 开始加载的网络层                                 *
*  args cutoff: 结束加载的网络层                                *
****************************************************************/
void load_weights_upto(network *net, char *filename, int start, int cutoff)
{
#ifdef GPU
    if(net->gpu_index >= 0){
		// cuda_set_device函数：src/cuda.c 设置使用GPU的编号
        cuda_set_device(net->gpu_index);
    }
#endif
    fprintf(stderr, "Loading weights from %s...", filename);
	// 原型:int fflush(FILE *stream) 清除读写缓冲区，需要立即把输出缓冲区的数据进行物理写入时调用
    fflush(stdout);
	// 打开网络权重文件的读写指针
    FILE *fp = fopen(filename, "rb");
    if(!fp) file_error(filename);

    int major;
    int minor;
    int revision;
	
	// fread函数：函数原型为 size_t fread ( void *buffer, size_t size, size_t count, FILE *stream);
	// fread是一个函数，它从文件流中读数据，最多读取count个项，每个项size个字节，如果调用成功返回实
	// 际读取到的项个数（小于或等于count），如果不成功或读到文件末尾返回 0。
    fread(&major, sizeof(int), 1, fp);
    fread(&minor, sizeof(int), 1, fp);
    fread(&revision, sizeof(int), 1, fp);
	
	
    if ((major*10 + minor) >= 2 && major < 1000 && minor < 1000){
        fread(net->seen, sizeof(size_t), 1, fp);
    } else {
        int iseen = 0;
        fread(&iseen, sizeof(int), 1, fp);
        *net->seen = iseen;
    }
    int transpose = (major > 1000) || (minor > 1000);

    int i;
	// 循环中根据网络层类型的不同，调用不同的函数加载权重
    for(i = start; i < net->n && i < cutoff; ++i){
        layer l = net->layers[i];
        if (l.dontload) continue;
        if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
            load_convolutional_weights(l, fp);
        }
        if(l.type == CONNECTED){
            load_connected_weights(l, fp, transpose);
        }
        if(l.type == BATCHNORM){
            load_batchnorm_weights(l, fp);
        }
        if(l.type == CRNN){
            load_convolutional_weights(*(l.input_layer), fp);
            load_convolutional_weights(*(l.self_layer), fp);
            load_convolutional_weights(*(l.output_layer), fp);
        }
        if(l.type == RNN){
            load_connected_weights(*(l.input_layer), fp, transpose);
            load_connected_weights(*(l.self_layer), fp, transpose);
            load_connected_weights(*(l.output_layer), fp, transpose);
        }
        if (l.type == LSTM) {
            load_connected_weights(*(l.wi), fp, transpose);
            load_connected_weights(*(l.wf), fp, transpose);
            load_connected_weights(*(l.wo), fp, transpose);
            load_connected_weights(*(l.wg), fp, transpose);
            load_connected_weights(*(l.ui), fp, transpose);
            load_connected_weights(*(l.uf), fp, transpose);
            load_connected_weights(*(l.uo), fp, transpose);
            load_connected_weights(*(l.ug), fp, transpose);
        }
        if (l.type == GRU) {
            if(1){
                load_connected_weights(*(l.wz), fp, transpose);
                load_connected_weights(*(l.wr), fp, transpose);
                load_connected_weights(*(l.wh), fp, transpose);
                load_connected_weights(*(l.uz), fp, transpose);
                load_connected_weights(*(l.ur), fp, transpose);
                load_connected_weights(*(l.uh), fp, transpose);
            }else{
                load_connected_weights(*(l.reset_layer), fp, transpose);
                load_connected_weights(*(l.update_layer), fp, transpose);
                load_connected_weights(*(l.state_layer), fp, transpose);
            }
        }
        if(l.type == LOCAL){
            int locations = l.out_w*l.out_h;
            int size = l.size*l.size*l.c*l.n*locations;
            fread(l.biases, sizeof(float), l.outputs, fp);
            fread(l.weights, sizeof(float), size, fp);
#ifdef GPU
            if(gpu_index >= 0){
                push_local_layer(l);
            }
#endif
        }
    }
    fprintf(stderr, "Done!\n");
    fclose(fp);
}



/**************************************************
*  func：调用load_weights_upto函数将网络net的权重 *
*        文件filename读取并加载进网络net          *
*  args net：欲加载的网络                         *
*  args filename：权重文件的完整路径              *
**************************************************/
void load_weights(network *net, char *filename)
{
	// load_weights_upto函数：本文件 从第一层开始，加载到网络的最后一层
    load_weights_upto(net, filename, 0, net->n);
}

