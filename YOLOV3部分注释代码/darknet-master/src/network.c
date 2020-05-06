#include <stdio.h>
#include <time.h>
#include <assert.h>
#include "network.h"
#include "image.h"
#include "data.h"
#include "utils.h"
#include "blas.h"

#include "crop_layer.h"
#include "connected_layer.h"
#include "gru_layer.h"
#include "rnn_layer.h"
#include "crnn_layer.h"
#include "local_layer.h"
#include "convolutional_layer.h"
#include "activation_layer.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "yolo_layer.h"
#include "normalization_layer.h"
#include "batchnorm_layer.h"
#include "maxpool_layer.h"
#include "reorg_layer.h"
#include "avgpool_layer.h"
#include "cost_layer.h"
#include "softmax_layer.h"
#include "dropout_layer.h"
#include "route_layer.h"
#include "upsample_layer.h"
#include "shortcut_layer.h"
#include "parser.h"
#include "data.h"



/***************************************
*  func: 获得网络中关于图像增强的参数  *
*  args net: 初始化好的网络            *
***************************************/
load_args get_base_args(network *net)
{
    load_args args = {0};
	// 网络输入的width和height，通过cfg设置
    args.w = net->w;
    args.h = net->h;
	// size具体意义有待补充
    args.size = net->w;

	// 对于图像增强的一些参数赋值
    args.min = net->min_crop;
    args.max = net->max_crop;
    args.angle = net->angle;
    args.aspect = net->aspect;
    args.exposure = net->exposure;
    args.center = net->center;
    args.saturation = net->saturation;
    args.hue = net->hue;
    return args;
}



/*****************************************************************
*  func：调用parse_network_cfg函数解析网络结构，调用load_weights *
*        函数加载网络权重                                        *
*  args cfg：网络结构配置文件，如：cfg/yolov3.cfg                *
*  args weights: 网络权重文件，如：yolov3.weights                *
*  args clear: 是否将网络的global_step置0                        *
*****************************************************************/
network *load_network(char *cfg, char *weights, int clear)
{
	// 1、network数据结构定义在include/darknet.h  注意每一个成员变量的意义，有助于理解程序
	// parse_network_cfg函数：src/parser.c 具体注释见该文件的函数定义处
    network *net = parse_network_cfg(cfg);
    if(weights && weights[0] != 0){
		// load_weights函数：src/parser.c ，加载net的权重文件weights进net，net属于指针类型，改变会被保留
        load_weights(net, weights);
    }
    if(clear) (*net->seen) = 0;
	// 返回初始化好的网络
    return net;
}



/**************************************************************************************
*  func: 计算当前已经读入多少个batch（提醒一下：网络配置.cfg文件中的batch不是指总共有 *
*        多少个batch，而是指每个batch中有多少张图片，tensorflow实战中一般用batch_size *
*        表示一个batch中函数的图片张数，用num_batches表示共有多少个batch，这样更清晰）*
*  args net: 构建的整个神经网络                                                       *
**************************************************************************************/
size_t get_current_batch(network *net)
{
	// net.seen为截至目前已经读入的图片张数，batch*subdivisons为一个batch含有的图片张数，二者一除即可得截至目前已经读入的batch个数
    // net.subdivisions这个参数目前还不知道有什么用，总之net.batch*net.subdivisions等于.cfg中指定的batch值（参看：parser.c中的parse_net_options()函数）
    size_t batch_num = (*net->seen)/(net->batch*net->subdivisions);
    return batch_num;
}

void reset_network_state(network *net, int b)
{
    int i;
    for (i = 0; i < net->n; ++i) {
        #ifdef GPU
        layer l = net->layers[i];
        if(l.state_gpu){
            fill_gpu(l.outputs, 0, l.state_gpu + l.outputs*b, 1);
        }
        if(l.h_gpu){
            fill_gpu(l.outputs, 0, l.h_gpu + l.outputs*b, 1);
        }
        #endif
    }
}

void reset_rnn(network *net)
{
    reset_network_state(net, 0);
}

float get_current_rate(network *net)
{
    size_t batch_num = get_current_batch(net);
    int i;
    float rate;
    if (batch_num < net->burn_in) return net->learning_rate * pow((float)batch_num / net->burn_in, net->power);
    switch (net->policy) {
        case CONSTANT:
            return net->learning_rate;
        case STEP:
            return net->learning_rate * pow(net->scale, batch_num/net->step);
        case STEPS:
            rate = net->learning_rate;
            for(i = 0; i < net->num_steps; ++i){
                if(net->steps[i] > batch_num) return rate;
                rate *= net->scales[i];
            }
            return rate;
        case EXP:
            return net->learning_rate * pow(net->gamma, batch_num);
        case POLY:
            return net->learning_rate * pow(1 - (float)batch_num / net->max_batches, net->power);
        case RANDOM:
            return net->learning_rate * pow(rand_uniform(0,1), net->power);
        case SIG:
            return net->learning_rate * (1./(1.+exp(net->gamma*(batch_num - net->step))));
        default:
            fprintf(stderr, "Policy is weird!\n");
            return net->learning_rate;
    }
}

char *get_layer_string(LAYER_TYPE a)
{
    switch(a){
        case CONVOLUTIONAL:
            return "convolutional";
        case ACTIVE:
            return "activation";
        case LOCAL:
            return "local";
        case DECONVOLUTIONAL:
            return "deconvolutional";
        case CONNECTED:
            return "connected";
        case RNN:
            return "rnn";
        case GRU:
            return "gru";
        case LSTM:
	    return "lstm";
        case CRNN:
            return "crnn";
        case MAXPOOL:
            return "maxpool";
        case REORG:
            return "reorg";
        case AVGPOOL:
            return "avgpool";
        case SOFTMAX:
            return "softmax";
        case DETECTION:
            return "detection";
        case REGION:
            return "region";
        case YOLO:
            return "yolo";
        case DROPOUT:
            return "dropout";
        case CROP:
            return "crop";
        case COST:
            return "cost";
        case ROUTE:
            return "route";
        case SHORTCUT:
            return "shortcut";
        case NORMALIZATION:
            return "normalization";
        case BATCHNORM:
            return "batchnorm";
        default:
            break;
    }
    return "none";
}



/****************************************************************
*  func：新建一个空网络，并为网络中部分指针参数动态分配内存     *
*  args n: 神经网络层数                                         *
*  details：该函数只为网络的三个指针参数动态分配了内存，并没有  *
*        为所有指针参数分配内存                                 *
****************************************************************/
network *make_network(int n)
{
	// 生成net对象，其中的一些属性并未开辟空间和初始化
    network *net = calloc(1, sizeof(network));
	// net->n表示网络一共有多少层
    net->n = n;
	// 为每一层分配内存，net->seen表示网络已经处理了多少张图片，cost表示网络的损失，t的意义有待补充
	// 总共有n层，需要为每一层layers开辟一块存储空间，所以一共开辟了N块
    net->layers = calloc(net->n, sizeof(layer));
    net->seen = calloc(1, sizeof(size_t));
    net->t    = calloc(1, sizeof(int));
    net->cost = calloc(1, sizeof(float));
	// 返回开辟的网络的首地址
    return net;
}



/***************************************************************
*  func: 前向计算网络net每一层的输出                           *
*  args netp：构建好的网络                                     *
*  details：遍历net的每一层网络，从第0层到最后一层，逐层计算   *
*           每层的输出                                         *
***************************************************************/
void forward_network(network *netp)
{
	// 如果有定义GPU，则调用forward_network_gpu函数完成网络的前向过程
	// forward_network_gpu函数：本文件。
#ifdef GPU
    if(netp->gpu_index >= 0){
        forward_network_gpu(netp);   
        return;
    }
#endif
    network net = *netp;
    int i;
	// 遍历所有层，从第一层到最后一层，逐层进行前向传播（网络总共有net.n层）
    for(i = 0; i < net.n; ++i){
		// 置网络当前活跃层为当前层，即第i层
        net.index = i;
		// 获取当前层
        layer l = net.layers[i];
		// 如果当前层的l.delta已经动态分配了内存，则调用fill_cpu()函数，将其所有元素的值初始化为0
        if(l.delta){
			// fill_cpu函数：src/blas.c 初始化l.delta的所有值为0，第一个参数为l.delta的元素个数，第二个参数为初始化值，为0
            fill_cpu(l.outputs * l.batch, 0, l.delta, 1);
        }
		// 前向传播（前向推理）：完成l层的前向推理
        l.forward(l, net);
		// 完成某一层的推理时，置网络的输入为当前层的输出（这将成为下一层网络的输入），要注意的是，此处是直接更改指针变量net.input本身的值，
        // 也就是此处是通过改变指针net.input所指的地址来改变其中所存内容的值，并不是直接改变其所指的内容而指针所指的地址没变，
        // 所以在退出forward_network()函数后，其对net.input的改变都将失效，net.input将回到进入forward_network()之前时的值。
        net.input = l.output;
        if(l.truth) {
            net.truth = l.output;
        }
    }
	// calc_network_cost函数：本文件 计算一次前向的cost
    calc_network_cost(netp);
}

void update_network(network *netp)
{
#ifdef GPU
    if(netp->gpu_index >= 0){
        update_network_gpu(netp);   
        return;
    }
#endif
    network net = *netp;
    int i;
    update_args a = {0};
    a.batch = net.batch*net.subdivisions;
    a.learning_rate = get_current_rate(netp);
    a.momentum = net.momentum;
    a.decay = net.decay;
    a.adam = net.adam;
    a.B1 = net.B1;
    a.B2 = net.B2;
    a.eps = net.eps;
    ++*net.t;
    a.t = *net.t;

    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if(l.update){
            l.update(l, a);
        }
    }
}

/******************************************************************
*  func：得到整个网络跑一次的loss                                 *
*  args netp: 经过了一次forward的网络                             *
******************************************************************/
void calc_network_cost(network *netp)
{
    network net = *netp;
    int i;
    float sum = 0;
    int count = 0;
	// 在yolo v3 中  只有在yolo层计算了cost,所以这儿其实是3个yolo层的cost相加
    for(i = 0; i < net.n; ++i){
        if(net.layers[i].cost){
            sum += net.layers[i].cost[0];
            ++count;
        }
    }
	// 整个网络的cost是三个yolo层的平均cost
    *net.cost = sum/count;
}

int get_predicted_class_network(network *net)
{
    return max_index(net->output, net->outputs);
}

void backward_network(network *netp)
{
#ifdef GPU
    if(netp->gpu_index >= 0){
        backward_network_gpu(netp);   
        return;
    }
#endif
    network net = *netp;
    int i;
    network orig = net;
    for(i = net.n-1; i >= 0; --i){
        layer l = net.layers[i];
        if(l.stopbackward) break;
        if(i == 0){
            net = orig;
        }else{
            layer prev = net.layers[i-1];
            net.input = prev.output;
            net.delta = prev.delta;
        }
        net.index = i;
        l.backward(l, net);
    }
}




float train_network_datum(network *net)
{
	// 更新目前已经处理的图片数量：每次处理一个batch，故直接添加l.batch
    *net->seen += net->batch;
	// 标记处于训练阶段
    net->train = 1;
	// forward_network函数：本文件 前向一遍网络，计算网络的cost
    forward_network(net);
	// backward_network函数：本文件 后向一遍网络
    backward_network(net);
    float error = *net->cost;
    if(((*net->seen)/net->batch)%net->subdivisions == 0) update_network(net);
    return error;
}

float train_network_sgd(network *net, data d, int n)
{
    int batch = net->batch;

    int i;
    float sum = 0;
    for(i = 0; i < n; ++i){
        get_random_batch(d, batch, net->input, net->truth);
        float err = train_network_datum(net);
        sum += err;
    }
    return (float)sum/(n*batch);
}



/*******************************************************************************************
*  func: 训练一个batch（此处所指一个batch含有的图片是配置文件中真实指定的一个batch中含有的 *
*        图片数量，也即图片张数为：net.batch*net.subdivision）                             *
*  args net：已经构建好待训练的整个网络                                                    *
*  args d: 此番训练所用到的所有图片数据（包含net.batch*net.subdivision张图片）             *
*******************************************************************************************/
float train_network(network *net, data d)
{
    // 此处d.X.rows为一个batch的数据，等于net->batch*net->subdivisions
    assert(d.X.rows % net->batch == 0);
    int batch = net->batch;
	// n的值其实为 net->subdivisions
    int n = d.X.rows / batch;

    int i;
    float sum = 0;
	// 将一个batch(cfg中的batch)训练数据分为subdivisions次后向传递
    for(i = 0; i < n; ++i){
		// 从d中读取batch张图片到net.input中，进行训练
        // 第一个参数d包含了net.batch*net.subdivision张图片的数据，第二个参数batch即为每次循环读入到net.input也即参与train_network_datum()
        // 训练的图片张数，第三个参数为在d中的偏移量，第四个参数为网络的输入数据，第五个参数为输入数据net.input对应的标签数据（真实数据）
		// get_next_batch函数：src/data.c 从输入d中深拷贝n张图片的数据与标签信息至net->input与net->truth中：将d.X.vals以及d.y.vals（如有必要）逐行深拷贝至X,y中
        get_next_batch(d, batch, i*batch, net->input, net->truth);
		// 训练网络：本次训练的数据共有net.batch张图片。
        // 训练包括一次前向过程：计算每一层网络的输出并计算cost；一次反向过程：计算敏感度、权重更新值、偏置更新值；适时更新过程：更新权重与偏置
		// train_network_datum函数：本文件 得到本batch网络输出的loss，注意分析其中的计算过程！
        float err = train_network_datum(net);
		// 每个subdivisions的误差进行累加
        sum += err;
    }
	// 返回的误差是一个batch（cfg中的batch）中每张图片的平均误差
    return (float)sum/(n*batch);
}

void set_temp_network(network *net, float t)
{
    int i;
    for(i = 0; i < net->n; ++i){
        net->layers[i].temperature = t;
    }
}


/********************************************************************
*  func：将网络net的batch大小设置为b                                *
*  args net:欲设置的网络                                            *
*  args b: 设置的batch大小                                          *
*  details：根据网络层功能的不同以及是否使用了cuda来调用不同的函数  *
*           进行设置                                                *
********************************************************************/
void set_batch_network(network *net, int b)
{
    net->batch = b;
    int i;
	// 循环设置网络中每一个layer层的batch为b
    for(i = 0; i < net->n; ++i){
        net->layers[i].batch = b;
#ifdef CUDNN
        if(net->layers[i].type == CONVOLUTIONAL){
            cudnn_convolutional_setup(net->layers + i);
        }
        if(net->layers[i].type == DECONVOLUTIONAL){
            layer *l = net->layers + i;
            cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, l->out_h, l->out_w);
            cudnnSetTensor4dDescriptor(l->normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, 1, 1); 
        }
#endif
    }
}



int resize_network(network *net, int w, int h)
{
#ifdef GPU
    cuda_set_device(net->gpu_index);
    cuda_free(net->workspace);
#endif
    int i;
    // if(w == net->w && h == net->h) return 0;
    net->w = w;
    net->h = h;
    int inputs = 0;
    size_t workspace_size = 0;
    // fprintf(stderr, "Resizing to %d x %d...\n", w, h);
    // fflush(stderr);
	// 对网络的每一层都需要重新计算大小，各自的函数在对应层的.c文件中
    for (i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == CONVOLUTIONAL){
            resize_convolutional_layer(&l, w, h);
        }else if(l.type == CROP){
            resize_crop_layer(&l, w, h);
        }else if(l.type == MAXPOOL){
            resize_maxpool_layer(&l, w, h);
        }else if(l.type == REGION){
            resize_region_layer(&l, w, h);
        }else if(l.type == YOLO){
            resize_yolo_layer(&l, w, h);
        }else if(l.type == ROUTE){
            resize_route_layer(&l, net);
        }else if(l.type == SHORTCUT){
            resize_shortcut_layer(&l, w, h);
        }else if(l.type == UPSAMPLE){
            resize_upsample_layer(&l, w, h);
        }else if(l.type == REORG){
            resize_reorg_layer(&l, w, h);
        }else if(l.type == AVGPOOL){
            resize_avgpool_layer(&l, w, h);
        }else if(l.type == NORMALIZATION){
            resize_normalization_layer(&l, w, h);
        }else if(l.type == COST){
            resize_cost_layer(&l, inputs);
        }else{
            error("Cannot resize this type of layer");
        }
        if(l.workspace_size > workspace_size) workspace_size = l.workspace_size;
        if(l.workspace_size > 2000000000) assert(0);
        inputs = l.outputs;
        net->layers[i] = l;
        w = l.out_w;
        h = l.out_h;
        if(l.type == AVGPOOL) break;
    }
    layer out = get_network_output_layer(net);
    net->inputs = net->layers[0].inputs;
    net->outputs = out.outputs;
    net->truths = out.outputs;
    if(net->layers[net->n-1].truths) net->truths = net->layers[net->n-1].truths;
    net->output = out.output;
    free(net->input);
    free(net->truth);
    net->input = calloc(net->inputs*net->batch, sizeof(float));
    net->truth = calloc(net->truths*net->batch, sizeof(float));
#ifdef GPU
    if(gpu_index >= 0){
        cuda_free(net->input_gpu);
        cuda_free(net->truth_gpu);
        net->input_gpu = cuda_make_array(net->input, net->inputs*net->batch);
        net->truth_gpu = cuda_make_array(net->truth, net->truths*net->batch);
        if(workspace_size){
            net->workspace = cuda_make_array(0, (workspace_size-1)/sizeof(float)+1);
        }
    }else {
        free(net->workspace);
        net->workspace = calloc(1, workspace_size);
    }
#else
    free(net->workspace);
    net->workspace = calloc(1, workspace_size);
#endif
    //fprintf(stderr, " Done!\n");
    return 0;
}

layer get_network_detection_layer(network *net)
{
    int i;
    for(i = 0; i < net->n; ++i){
        if(net->layers[i].type == DETECTION){
            return net->layers[i];
        }
    }
    fprintf(stderr, "Detection layer not found!!\n");
    layer l = {0};
    return l;
}

image get_network_image_layer(network *net, int i)
{
    layer l = net->layers[i];
#ifdef GPU
    //cuda_pull_array(l.output_gpu, l.output, l.outputs);
#endif
    if (l.out_w && l.out_h && l.out_c){
        return float_to_image(l.out_w, l.out_h, l.out_c, l.output);
    }
    image def = {0};
    return def;
}

image get_network_image(network *net)
{
    int i;
    for(i = net->n-1; i >= 0; --i){
        image m = get_network_image_layer(net, i);
        if(m.h != 0) return m;
    }
    image def = {0};
    return def;
}

void visualize_network(network *net)
{
    image *prev = 0;
    int i;
    char buff[256];
    for(i = 0; i < net->n; ++i){
        sprintf(buff, "Layer %d", i);
        layer l = net->layers[i];
        if(l.type == CONVOLUTIONAL){
            prev = visualize_convolutional_layer(l, buff, prev);
        }
    } 
}

void top_predictions(network *net, int k, int *index)
{
    top_k(net->output, net->outputs, k, index);
}



/*********************************************************************
*  func：以input作为网络输入，跑一遍net的前向过程，返回网络输出      *
*  args net：构建的网络                                              *
*  args input：输入数据（如果是图片，则其已经调整至网络需要的尺寸）  *
*  details：本函数为整个网络的前向推理函数，调用该函数可完成对某一   *
*           输入数据进行前向推理的过程，可以参考detector.c中         *
*           test_detector()函数调用该函数完成对一张输入图片的检测    *
*********************************************************************/
float *network_predict(network *net, float *input)
{
    network orig = *net;
    net->input = input;
    net->truth = 0;
    net->train = 0;
    net->delta = 0;
	// forward_network函数：本文件。根据是否使用GPU使用不同得forward函数
    forward_network(net);
    float *out = net->output;
    *net = orig;
    return out;
}



/***************************************************************************
*  func：调用yolo_num_detections统计有多少个有效预测框                     *
*  args net：构建得网络                                                    *
*  args thresh：阈值                                                       *
*  details: 当网络类型为[yolo]时，调用yolo_num_detections统计有多少个预测  *
*           框，当网络类型为[detection]或[region]时，预测框个数加上w*h*c   *
***************************************************************************/
int num_detections(network *net, float thresh)
{
    int i;
    int s = 0;
	// 循环判断网络的每一层
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO){
			// yolo_num_detections函数：src/yolo_layer.c 统计该层yolo层的预测框的confidence
			// 值大于阈值thresh的框的个数
            s += yolo_num_detections(l, thresh);
        }
        if(l.type == DETECTION || l.type == REGION){
            s += l.w*l.h*l.n;
        }
    }
    return s;
}



/***************************************************************************
*  func: 为所有输出层的预测框中凡是confidence大于thresh的框分配存储空间    *
*        统计多少个有效框，并为框的目标类别prob开辟存储空间                *
*  args net: 完成了前向传递的网络指针                                      *
*  args thresh: 阈值                                                       *
*  args num: 有效框的个数指针                                              *
***************************************************************************/
detection *make_network_boxes(network *net, float thresh, int *num)
{
	/* detection结构体定义如下：
	typedef struct detection{
		box bbox;
		int classes;
		float *prob;
		float *mask;
		float objectness;
		int sort_class;
	} detection;
	*/
	// 获取网络的最后一层
    layer l = net->layers[net->n - 1];
    int i;
	// num_detections函数：本文件  调用yolo_num_detections统计有多少个预测框的confidence值大于thresh
    int nboxes = num_detections(net, thresh);
    if(num) *num = nboxes;
	// 为有效框开辟存储空间，这儿的有效框指预测框的confidence值大于thresh的框
    detection *dets = calloc(nboxes, sizeof(detection));
    for(i = 0; i < nboxes; ++i){
		// 为每一个有效框的prob域开辟存储空间，存储类别概率
        dets[i].prob = calloc(l.classes, sizeof(float));
		// l.coords这个值不知道我是在Yolov3中没找到还是怎么？总之没见其它地方有过赋值，所以其值为初始值0
        if(l.coords > 4){
            dets[i].mask = calloc(l.coords-4, sizeof(float));
        }
    }
    return dets;
}



/********************************************************************************************
*  func：根据网络层类型的不同，调用不同的函数，将边框信息进行转化，具体见函数定义处         *
*  args net：完成了前向后的网络，包含了预测框的信息在yolo层中                               *
*  args w: 原始图片的width                                                                  *
*  args h: 原始图片的height                                                                 *
*  args thresh: 姑且当作是否为有效框的阈值                                                  *
*  args hier: 这个参数在yolov3中没有使用                                                    *
*  args map: yolov3中没有使用                                                               *
*  args relative: 
*  args dets: 预测框的指针，引用传递                                                        *
********************************************************************************************/
void fill_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, detection *dets)
{
    int j;
    for(j = 0; j < net->n; ++j){
        layer l = net->layers[j];
        if(l.type == YOLO){
			// 当网络层类型为yolo时，调用get_yolo_detection函数，定义在src/yolo_layer.c中，得到预测边框进行转化后的信息
            int count = get_yolo_detections(l, w, h, net->w, net->h, thresh, map, relative, dets);
			// 每操作完一个yolo层，使指针进行偏移
            dets += count;
        }
        if(l.type == REGION){
            get_region_detections(l, w, h, net->w, net->h, thresh, map, hier, relative, dets);
            dets += l.w*l.h*l.n;
        }
        if(l.type == DETECTION){
            get_detection_detections(l, w, h, thresh, dets);
            dets += l.w*l.h*l.n;
        }
    }
}



/******************************************************************************************************************
*  func：调用函数make_network_boxes 统计有多少个预测框的confidence值大于thresh,并分配一定空间，再调用             *
*        fill_network_boxes函数获取有效框的边框信息和类别概率等，随后进行相对值的转化                             *
*  args net: 完成了前向后的网络，包含了预测框的信息在yolo层中                                                     *
*  args w: 原始图片的width                                                                                        *
*  args h: 原始图片的height                                                                                       *
*  args thresh: 姑且当作是否为有效框的阈值                                                                        *
*  args hier: 这个参数在yolov3中没有使用                                                                          *
*  args map: yolov3中没有使用                                                                                     *
*  args relative: 指示边框信息中的x,y,w,h是否是相对值，即归一化后的值，默认为1                                    *
*  args num: 有效框的个数指针，引用传递                                                                           *
*  details: 调用原型：get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes) thresh为输入值，     *
*           hier_thresh默认0.5                                                                                    *
******************************************************************************************************************/
detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num)
{
	// make_network_boxes函数：本文件。 统计有多少个预测框的confidence值大于thresh,为有效的预测框分配空间
	// 同时为每一个有效预测框的类别概率分配空间
    detection *dets = make_network_boxes(net, thresh, num);
	// fill_network_boxes函数：本文件。获取有效框的边框信息和类别概率等，随后进行相对值的转化，转化到相对于原图的比例
    fill_network_boxes(net, w, h, thresh, hier, map, relative, dets);
	// 返回预测框集合
    return dets;
}


/***************************
*  func: 释放检测框集合    *
***************************/
void free_detections(detection *dets, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        free(dets[i].prob);
        if(dets[i].mask) free(dets[i].mask);
    }
    free(dets);
}



/*********************************************************************
*  func：
*********************************************************************/
float *network_predict_image(network *net, image im)
{
    image imr = letterbox_image(im, net->w, net->h);
    set_batch_network(net, 1);
    float *p = network_predict(net, imr.data);
    free_image(imr);
    return p;
}

int network_width(network *net){return net->w;}
int network_height(network *net){return net->h;}

matrix network_predict_data_multi(network *net, data test, int n)
{
    int i,j,b,m;
    int k = net->outputs;
    matrix pred = make_matrix(test.X.rows, k);
    float *X = calloc(net->batch*test.X.rows, sizeof(float));
    for(i = 0; i < test.X.rows; i += net->batch){
        for(b = 0; b < net->batch; ++b){
            if(i+b == test.X.rows) break;
            memcpy(X+b*test.X.cols, test.X.vals[i+b], test.X.cols*sizeof(float));
        }
        for(m = 0; m < n; ++m){
            float *out = network_predict(net, X);
            for(b = 0; b < net->batch; ++b){
                if(i+b == test.X.rows) break;
                for(j = 0; j < k; ++j){
                    pred.vals[i+b][j] += out[j+b*k]/n;
                }
            }
        }
    }
    free(X);
    return pred;   
}

matrix network_predict_data(network *net, data test)
{
    int i,j,b;
    int k = net->outputs;
    matrix pred = make_matrix(test.X.rows, k);
    float *X = calloc(net->batch*test.X.cols, sizeof(float));
    for(i = 0; i < test.X.rows; i += net->batch){
        for(b = 0; b < net->batch; ++b){
            if(i+b == test.X.rows) break;
            memcpy(X+b*test.X.cols, test.X.vals[i+b], test.X.cols*sizeof(float));
        }
        float *out = network_predict(net, X);
        for(b = 0; b < net->batch; ++b){
            if(i+b == test.X.rows) break;
            for(j = 0; j < k; ++j){
                pred.vals[i+b][j] = out[j+b*k];
            }
        }
    }
    free(X);
    return pred;   
}

void print_network(network *net)
{
    int i,j;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        float *output = l.output;
        int n = l.outputs;
        float mean = mean_array(output, n);
        float vari = variance_array(output, n);
        fprintf(stderr, "Layer %d - Mean: %f, Variance: %f\n",i,mean, vari);
        if(n > 100) n = 100;
        for(j = 0; j < n; ++j) fprintf(stderr, "%f, ", output[j]);
        if(n == 100)fprintf(stderr,".....\n");
        fprintf(stderr, "\n");
    }
}

void compare_networks(network *n1, network *n2, data test)
{
    matrix g1 = network_predict_data(n1, test);
    matrix g2 = network_predict_data(n2, test);
    int i;
    int a,b,c,d;
    a = b = c = d = 0;
    for(i = 0; i < g1.rows; ++i){
        int truth = max_index(test.y.vals[i], test.y.cols);
        int p1 = max_index(g1.vals[i], g1.cols);
        int p2 = max_index(g2.vals[i], g2.cols);
        if(p1 == truth){
            if(p2 == truth) ++d;
            else ++c;
        }else{
            if(p2 == truth) ++b;
            else ++a;
        }
    }
    printf("%5d %5d\n%5d %5d\n", a, b, c, d);
    float num = pow((abs(b - c) - 1.), 2.);
    float den = b + c;
    printf("%f\n", num/den); 
}

float network_accuracy(network *net, data d)
{
    matrix guess = network_predict_data(net, d);
    float acc = matrix_topk_accuracy(d.y, guess,1);
    free_matrix(guess);
    return acc;
}

float *network_accuracies(network *net, data d, int n)
{
    static float acc[2];
    matrix guess = network_predict_data(net, d);
    acc[0] = matrix_topk_accuracy(d.y, guess, 1);
    acc[1] = matrix_topk_accuracy(d.y, guess, n);
    free_matrix(guess);
    return acc;
}



/************************************************************
*  func：获取网络net的输出层，判断依据是从最后一层向前找，只*
*        要该层的type不是COST则第一个层就是输出层           *
*  args net：寻找的网络net                                  *
************************************************************/
layer get_network_output_layer(network *net)
{
    int i;
    for(i = net->n - 1; i >= 0; --i){
        if(net->layers[i].type != COST) break;
    }
    return net->layers[i];
}

float network_accuracy_multi(network *net, data d, int n)
{
    matrix guess = network_predict_data_multi(net, d, n);
    float acc = matrix_topk_accuracy(d.y, guess,1);
    free_matrix(guess);
    return acc;
}

void free_network(network *net)
{
    int i;
    for(i = 0; i < net->n; ++i){
        free_layer(net->layers[i]);
    }
    free(net->layers);
    if(net->input) free(net->input);
    if(net->truth) free(net->truth);
#ifdef GPU
    if(net->input_gpu) cuda_free(net->input_gpu);
    if(net->truth_gpu) cuda_free(net->truth_gpu);
#endif
    free(net);
}

// Some day...
// ^ What the hell is this comment for?


layer network_output_layer(network *net)
{
    int i;
    for(i = net->n - 1; i >= 0; --i){
        if(net->layers[i].type != COST) break;
    }
    return net->layers[i];
}

int network_inputs(network *net)
{
    return net->layers[0].inputs;
}

int network_outputs(network *net)
{
    return network_output_layer(net).outputs;
}

float *network_output(network *net)
{
    return network_output_layer(net).output;
}





#ifdef GPU



/******************************************************************
*
******************************************************************/
void forward_network_gpu(network *netp)
{
    network net = *netp;
	// 设定GPU编号
    cuda_set_device(net.gpu_index);
    cuda_push_array(net.input_gpu, net.input, net.inputs*net.batch);
    if(net.truth){
        cuda_push_array(net.truth_gpu, net.truth, net.truths*net.batch);
    }

    int i;
    for(i = 0; i < net.n; ++i){
        net.index = i;
        layer l = net.layers[i];
        if(l.delta_gpu){
            fill_gpu(l.outputs * l.batch, 0, l.delta_gpu, 1);
        }
        l.forward_gpu(l, net);
        net.input_gpu = l.output_gpu;
        net.input = l.output;
        if(l.truth) {
            net.truth_gpu = l.output_gpu;
            net.truth = l.output;
        }
    }
    pull_network_output(netp);
    calc_network_cost(netp);
}

void backward_network_gpu(network *netp)
{
    int i;
    network net = *netp;
    network orig = net;
    cuda_set_device(net.gpu_index);
    for(i = net.n-1; i >= 0; --i){
        layer l = net.layers[i];
        if(l.stopbackward) break;
        if(i == 0){
            net = orig;
        }else{
            layer prev = net.layers[i-1];
            net.input = prev.output;
            net.delta = prev.delta;
            net.input_gpu = prev.output_gpu;
            net.delta_gpu = prev.delta_gpu;
        }
        net.index = i;
        l.backward_gpu(l, net);
    }
}

void update_network_gpu(network *netp)
{
    network net = *netp;
    cuda_set_device(net.gpu_index);
    int i;
    update_args a = {0};
    a.batch = net.batch*net.subdivisions;
    a.learning_rate = get_current_rate(netp);
    a.momentum = net.momentum;
    a.decay = net.decay;
    a.adam = net.adam;
    a.B1 = net.B1;
    a.B2 = net.B2;
    a.eps = net.eps;
    ++*net.t;
    a.t = (*net.t);

    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if(l.update_gpu){
            l.update_gpu(l, a);
        }
    }
}

void harmless_update_network_gpu(network *netp)
{
    network net = *netp;
    cuda_set_device(net.gpu_index);
    int i;
    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if(l.weight_updates_gpu) fill_gpu(l.nweights, 0, l.weight_updates_gpu, 1);
        if(l.bias_updates_gpu) fill_gpu(l.nbiases, 0, l.bias_updates_gpu, 1);
        if(l.scale_updates_gpu) fill_gpu(l.nbiases, 0, l.scale_updates_gpu, 1);
    }
}

typedef struct {
    network *net;
    data d;
    float *err;
} train_args;



/***********************************************************************
*  func: 调用train_network函数训练一个batch的数据                      *
*  args ptr：包含了训练所需的网络、数据和err                           *
***********************************************************************/
void *train_thread(void *ptr)
{
	// 深复制ptr的内容，但是指针变量指向同一区域
    train_args args = *(train_args*)ptr;
    free(ptr);
    cuda_set_device(args.net->gpu_index);
	// train_network函数：本文件 具体注释见该文件的函数定义处
    *args.err = train_network(args.net, args.d);
    return 0;
}



/************************************************************************
*  func: 开辟训练线程，进行网络的训练                                   *
*  args net：初始化好的网络，即将训练的网络                             *
*  args d: 训练数据的结构体                                             *
*  args err: 训练误差                                                   *
************************************************************************/
pthread_t train_network_in_thread(network *net, data d, float *err)
{
    pthread_t thread;
	// train_args数据结构定义在本文件
    train_args *ptr = (train_args *)calloc(1, sizeof(train_args));
	// 训练使用的网络
    ptr->net = net;
	// 训练使用的数据
    ptr->d = d;
	// 这批数据的误差
    ptr->err = err;
	// train_thread函数：本文件
    if(pthread_create(&thread, 0, train_thread, ptr)) error("Thread creation failed");
    return thread;
}

void merge_weights(layer l, layer base)
{
    if (l.type == CONVOLUTIONAL) {
        axpy_cpu(l.n, 1, l.bias_updates, 1, base.biases, 1);
        axpy_cpu(l.nweights, 1, l.weight_updates, 1, base.weights, 1);
        if (l.scales) {
            axpy_cpu(l.n, 1, l.scale_updates, 1, base.scales, 1);
        }
    } else if(l.type == CONNECTED) {
        axpy_cpu(l.outputs, 1, l.bias_updates, 1, base.biases, 1);
        axpy_cpu(l.outputs*l.inputs, 1, l.weight_updates, 1, base.weights, 1);
    }
}

void scale_weights(layer l, float s)
{
    if (l.type == CONVOLUTIONAL) {
        scal_cpu(l.n, s, l.biases, 1);
        scal_cpu(l.nweights, s, l.weights, 1);
        if (l.scales) {
            scal_cpu(l.n, s, l.scales, 1);
        }
    } else if(l.type == CONNECTED) {
        scal_cpu(l.outputs, s, l.biases, 1);
        scal_cpu(l.outputs*l.inputs, s, l.weights, 1);
    }
}


void pull_weights(layer l)
{
    if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
        cuda_pull_array(l.biases_gpu, l.bias_updates, l.n);
        cuda_pull_array(l.weights_gpu, l.weight_updates, l.nweights);
        if(l.scales) cuda_pull_array(l.scales_gpu, l.scale_updates, l.n);
    } else if(l.type == CONNECTED){
        cuda_pull_array(l.biases_gpu, l.bias_updates, l.outputs);
        cuda_pull_array(l.weights_gpu, l.weight_updates, l.outputs*l.inputs);
    }
}

void push_weights(layer l)
{
    if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
        cuda_push_array(l.biases_gpu, l.biases, l.n);
        cuda_push_array(l.weights_gpu, l.weights, l.nweights);
        if(l.scales) cuda_push_array(l.scales_gpu, l.scales, l.n);
    } else if(l.type == CONNECTED){
        cuda_push_array(l.biases_gpu, l.biases, l.outputs);
        cuda_push_array(l.weights_gpu, l.weights, l.outputs*l.inputs);
    }
}

void distribute_weights(layer l, layer base)
{
    if (l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL) {
        cuda_push_array(l.biases_gpu, base.biases, l.n);
        cuda_push_array(l.weights_gpu, base.weights, l.nweights);
        if (base.scales) cuda_push_array(l.scales_gpu, base.scales, l.n);
    } else if (l.type == CONNECTED) {
        cuda_push_array(l.biases_gpu, base.biases, l.outputs);
        cuda_push_array(l.weights_gpu, base.weights, l.outputs*l.inputs);
    }
}


/*

   void pull_updates(layer l)
   {
   if(l.type == CONVOLUTIONAL){
   cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.n);
   cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
   if(l.scale_updates) cuda_pull_array(l.scale_updates_gpu, l.scale_updates, l.n);
   } else if(l.type == CONNECTED){
   cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
   cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.outputs*l.inputs);
   }
   }

   void push_updates(layer l)
   {
   if(l.type == CONVOLUTIONAL){
   cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.n);
   cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
   if(l.scale_updates) cuda_push_array(l.scale_updates_gpu, l.scale_updates, l.n);
   } else if(l.type == CONNECTED){
   cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
   cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.outputs*l.inputs);
   }
   }

   void update_layer(layer l, network net)
   {
   int update_batch = net.batch*net.subdivisions;
   float rate = get_current_rate(net);
   l.t = get_current_batch(net);
   if(l.update_gpu){
   l.update_gpu(l, update_batch, rate*l.learning_rate_scale, net.momentum, net.decay);
   }
   }
   void merge_updates(layer l, layer base)
   {
   if (l.type == CONVOLUTIONAL) {
   axpy_cpu(l.n, 1, l.bias_updates, 1, base.bias_updates, 1);
   axpy_cpu(l.nweights, 1, l.weight_updates, 1, base.weight_updates, 1);
   if (l.scale_updates) {
   axpy_cpu(l.n, 1, l.scale_updates, 1, base.scale_updates, 1);
   }
   } else if(l.type == CONNECTED) {
   axpy_cpu(l.outputs, 1, l.bias_updates, 1, base.bias_updates, 1);
   axpy_cpu(l.outputs*l.inputs, 1, l.weight_updates, 1, base.weight_updates, 1);
   }
   }

   void distribute_updates(layer l, layer base)
   {
   if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
   cuda_push_array(l.bias_updates_gpu, base.bias_updates, l.n);
   cuda_push_array(l.weight_updates_gpu, base.weight_updates, l.nweights);
   if(base.scale_updates) cuda_push_array(l.scale_updates_gpu, base.scale_updates, l.n);
   } else if(l.type == CONNECTED){
   cuda_push_array(l.bias_updates_gpu, base.bias_updates, l.outputs);
   cuda_push_array(l.weight_updates_gpu, base.weight_updates, l.outputs*l.inputs);
   }
   }
 */

/*
   void sync_layer(network *nets, int n, int j)
   {
   int i;
   network net = nets[0];
   layer base = net.layers[j];
   scale_weights(base, 0);
   for (i = 0; i < n; ++i) {
   cuda_set_device(nets[i].gpu_index);
   layer l = nets[i].layers[j];
   pull_weights(l);
   merge_weights(l, base);
   }
   scale_weights(base, 1./n);
   for (i = 0; i < n; ++i) {
   cuda_set_device(nets[i].gpu_index);
   layer l = nets[i].layers[j];
   distribute_weights(l, base);
   }
   }
 */

void sync_layer(network **nets, int n, int j)
{
    int i;
    network *net = nets[0];
    layer base = net->layers[j];
    scale_weights(base, 0);
    for (i = 0; i < n; ++i) {
        cuda_set_device(nets[i]->gpu_index);
        layer l = nets[i]->layers[j];
        pull_weights(l);
        merge_weights(l, base);
    }
    scale_weights(base, 1./n);
    for (i = 0; i < n; ++i) {
        cuda_set_device(nets[i]->gpu_index);
        layer l = nets[i]->layers[j];
        distribute_weights(l, base);
    }
}

typedef struct{
    network **nets;
    int n;
    int j;
} sync_args;

void *sync_layer_thread(void *ptr)
{
    sync_args args = *(sync_args*)ptr;
    sync_layer(args.nets, args.n, args.j);
    free(ptr);
    return 0;
}

pthread_t sync_layer_in_thread(network **nets, int n, int j)
{
    pthread_t thread;
    sync_args *ptr = (sync_args *)calloc(1, sizeof(sync_args));
    ptr->nets = nets;
    ptr->n = n;
    ptr->j = j;
    if(pthread_create(&thread, 0, sync_layer_thread, ptr)) error("Thread creation failed");
    return thread;
}

void sync_nets(network **nets, int n, int interval)
{
    int j;
    int layers = nets[0]->n;
    pthread_t *threads = (pthread_t *) calloc(layers, sizeof(pthread_t));

    *(nets[0]->seen) += interval * (n-1) * nets[0]->batch * nets[0]->subdivisions;
    for (j = 0; j < n; ++j){
        *(nets[j]->seen) = *(nets[0]->seen);
    }
    for (j = 0; j < layers; ++j) {
        threads[j] = sync_layer_in_thread(nets, n, j);
    }
    for (j = 0; j < layers; ++j) {
        pthread_join(threads[j], 0);
    }
    free(threads);
}



/*******************************************************************************************
*  func: 训练一个batch（此处所指一个batch含有的图片是配置文件中真实指定的一个batch中含有的 *
*        图片数量，也即图片张数为：net.batch*net.subdivision）                             *
*  args n: 使用GPU的块数                                                                   *
*  args net：已经构建好待训练的整个网络                                                    *
*  args d: 此番训练所用到的所有图片数据（包含net.batch*net.subdivision张图片）             *
*  details：train_network()函数对应处理ngpu=1的情况，如果有多个gpu，那么会首先调用         *
*           train_networks()函数，在其之中会调用get_data_part()将在detector.c              *
*           train_detector()函数中读入的net.batch*net.subdivision*ngpu张图片平分至每个gpu  *
*   		上，而后再调用train_network()，总之最后train_work()的输入d中只包含             *
*           net.batch*net.subdivision张图片                                                *
*******************************************************************************************/
float train_networks(network **nets, int n, data d, int interval)
{
    int i;
    int batch = nets[0]->batch;
    int subdivisions = nets[0]->subdivisions;
	// 事实上对于图像检测而言，d.X.rows/net.batch=net.subdivision，因此恒有d.X.rows % net.batch == 0，且下面的n就等于net.subdivision
    // （可以参看detector.c中的train_detector()），因此对于图像检测而言，下面三句略有冗余，但对于其他种情况（比如其他应用，非图像检测甚至非视觉情况），
    // 不知道是不是这样
    assert(batch * subdivisions * n == d.X.rows);
    pthread_t *threads = (pthread_t *) calloc(n, sizeof(pthread_t));
	// errors用来统计loss
    float *errors = (float *) calloc(n, sizeof(float));

    float sum = 0;
    for(i = 0; i < n; ++i){
        // get_data_part函数：src/data.c 从d中读取batch张图片到net.input中，进行训练
        data p = get_data_part(d, i, n);
		// train_network_in_thread函数：本文件 
        threads[i] = train_network_in_thread(nets[i], p, errors + i);
    }
    for(i = 0; i < n; ++i){
        pthread_join(threads[i], 0);
        //printf("%f\n", errors[i]);
        sum += errors[i];
    }
    //cudaDeviceSynchronize();
    if (get_current_batch(nets[0]) % interval == 0) {
        printf("Syncing... ");
        fflush(stdout);
        sync_nets(nets, n, interval);
        printf("Done!\n");
    }
    //cudaDeviceSynchronize();
    free(threads);
    free(errors);
    return (float)sum/(n);
}

void pull_network_output(network *net)
{
    layer l = get_network_output_layer(net);
    cuda_pull_array(l.output_gpu, l.output, l.outputs*l.batch);
}

#endif
