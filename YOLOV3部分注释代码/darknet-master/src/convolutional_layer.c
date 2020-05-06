#include "convolutional_layer.h"
#include "utils.h"
#include "batchnorm_layer.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>
#include <time.h>

#ifdef AI2
#include "xnor_layer.h"
#endif

void swap_binary(convolutional_layer *l)
{
    float *swap = l->weights;
    l->weights = l->binary_weights;
    l->binary_weights = swap;

#ifdef GPU
    swap = l->weights_gpu;
    l->weights_gpu = l->binary_weights_gpu;
    l->binary_weights_gpu = swap;
#endif
}

void binarize_weights(float *weights, int n, int size, float *binary)
{
    int i, f;
    for(f = 0; f < n; ++f){
        float mean = 0;
        for(i = 0; i < size; ++i){
            mean += fabs(weights[f*size + i]);
        }
        mean = mean / size;
        for(i = 0; i < size; ++i){
            binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
        }
    }
}

void binarize_cpu(float *input, int n, float *binary)
{
    int i;
    for(i = 0; i < n; ++i){
        binary[i] = (input[i] > 0) ? 1 : -1;
    }
}

void binarize_input(float *input, int n, int size, float *binary)
{
    int i, s;
    for(s = 0; s < size; ++s){
        float mean = 0;
        for(i = 0; i < n; ++i){
            mean += fabs(input[i*size + s]);
        }
        mean = mean / n;
        for(i = 0; i < n; ++i){
            binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
        }
    }
}



/*****************************************************
*  func：这两个函数分别得到卷积后的height和width     *
*  args l：需要计算的网络层，里面包含下面公式中需要  *
*          的各个参数值                              *
*****************************************************/
int convolutional_out_height(convolutional_layer l)
{
    return (l.h + 2*l.pad - l.size) / l.stride + 1;
}
int convolutional_out_width(convolutional_layer l)
{
    return (l.w + 2*l.pad - l.size) / l.stride + 1;
}




image get_convolutional_image(convolutional_layer l)
{
    return float_to_image(l.out_w,l.out_h,l.out_c,l.output);
}

image get_convolutional_delta(convolutional_layer l)
{
    return float_to_image(l.out_w,l.out_h,l.out_c,l.delta);
}

static size_t get_workspace_size(layer l){
#ifdef CUDNN
    if(gpu_index >= 0){
        size_t most = 0;
        size_t s = 0;
        cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle(),
                l.srcTensorDesc,
                l.weightDesc,
                l.convDesc,
                l.dstTensorDesc,
                l.fw_algo,
                &s);
        if (s > most) most = s;
        cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle(),
                l.srcTensorDesc,
                l.ddstTensorDesc,
                l.convDesc,
                l.dweightDesc,
                l.bf_algo,
                &s);
        if (s > most) most = s;
        cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle(),
                l.weightDesc,
                l.ddstTensorDesc,
                l.convDesc,
                l.dsrcTensorDesc,
                l.bd_algo,
                &s);
        if (s > most) most = s;
        return most;
    }
#endif
    // 按道理不管是使用CPU还是GPU，一种方法算出来的大小都是有效的
    return (size_t)l.out_h*l.out_w*l.size*l.size*l.c/l.groups*sizeof(float);
}

#ifdef GPU
#ifdef CUDNN
void cudnn_convolutional_setup(layer *l)
{
    cudnnSetTensor4dDescriptor(l->dsrcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->c, l->h, l->w); 
    cudnnSetTensor4dDescriptor(l->ddstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w); 

    cudnnSetTensor4dDescriptor(l->srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->c, l->h, l->w); 
    cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w); 
    cudnnSetTensor4dDescriptor(l->normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, 1, 1); 

    cudnnSetFilter4dDescriptor(l->dweightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, l->n, l->c/l->groups, l->size, l->size); 
    cudnnSetFilter4dDescriptor(l->weightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, l->n, l->c/l->groups, l->size, l->size); 
    #if CUDNN_MAJOR >= 6
    cudnnSetConvolution2dDescriptor(l->convDesc, l->pad, l->pad, l->stride, l->stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    #else
    cudnnSetConvolution2dDescriptor(l->convDesc, l->pad, l->pad, l->stride, l->stride, 1, 1, CUDNN_CROSS_CORRELATION);
    #endif

    #if CUDNN_MAJOR >= 7
    cudnnSetConvolutionGroupCount(l->convDesc, l->groups);
    #else
    if(l->groups > 1){
        error("CUDNN < 7 doesn't support groups, please upgrade!");
    }
    #endif

    cudnnGetConvolutionForwardAlgorithm(cudnn_handle(),
            l->srcTensorDesc,
            l->weightDesc,
            l->convDesc,
            l->dstTensorDesc,
            CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
            2000000000,
            &l->fw_algo);
    cudnnGetConvolutionBackwardDataAlgorithm(cudnn_handle(),
            l->weightDesc,
            l->ddstTensorDesc,
            l->convDesc,
            l->dsrcTensorDesc,
            CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
            2000000000,
            &l->bd_algo);
    cudnnGetConvolutionBackwardFilterAlgorithm(cudnn_handle(),
            l->srcTensorDesc,
            l->ddstTensorDesc,
            l->convDesc,
            l->dweightDesc,
            CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
            2000000000,
            &l->bf_algo);
}
#endif
#endif



/**************************************************************************************************************
*  func：构建一层类型为conv或convolutional的layer并返回，执行此函数时以构建第一层网络为例，执行的实际指令为： *
*        make_convolutional_layer(64,416,416,3,32,1,3,1,1,“leaky”, 1, 0, 0, 0)                                *
*  details：具体的参数意义这里不再赘述，看函数调用处                                                          *
**************************************************************************************************************/
convolutional_layer make_convolutional_layer(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam)
{
	/*
        batch           每个batch含有的图片数
        h               图片高度（行数）
        w               图片宽度（列数）
        c               输入图片通道数
        n               卷积核个数
        size            卷积核尺寸
        stride          卷积的跨度
        padding         四周补0长度
        activation      激活函数类别
        batch_normalize 是否进行BN(规范化)
        binary          是否对权重进行二值化
        xnor            是否对权重以及输入进行二值化
        adam            是否使用adam,yolov3.cfg中没指定，adam=0
    */
    int i;
	// convolutional_layer类型就是 layer类型
    convolutional_layer l = {0};
	
	// 本函数处理的网络层类型自然为 CONVOLUTIONAL
    l.type = CONVOLUTIONAL;
    
	// 为layer中的若干属性赋值，groups的值为1，对程序没影响，有明白的读者可以给我讲讲
    l.groups = groups;
    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n;
    l.binary = binary;
    l.xnor = xnor;
    l.batch = batch;
    l.stride = stride;
    l.size = size;
    l.pad = padding;
    l.batch_normalize = batch_normalize;

	// 该卷积层总的权重元素（卷积核元素）个数=输入图像通道数*卷积核个数*卷积核尺寸（这儿的尺寸应该是二维的尺寸都要乘）
    // （因为一个卷积核要作用在输入图片的所有通道上，所以说是一个卷积核，实际含有的卷积核参数个数需要乘以输入图片的通道数）
    l.weights = calloc(c/groups*n*size*size, sizeof(float));	
	// 权重的跟新量，大小和weights相同
    l.weight_updates = calloc(c/groups*n*size*size, sizeof(float));

	// bias就是Wx+b中的b（上面的weights就是W），有多少个卷积核，就有多少个b（与W的个数一一对应，每个W的元素个数为c*size*size）
    l.biases = calloc(n, sizeof(float));	
	// 偏置的跟新量，与偏置个数一致
    l.bias_updates = calloc(n, sizeof(float));

	// 该卷积层总的权重元素个数（权重元素个数等于输入数据的通道数*卷积核个数*卷积核的二维尺寸，注意因为每一个卷积核是同时作用于输入数据
    // 的多个通道上的，因此实际上卷积核是三维的，包括两个维度的平面尺寸，以及输入数据通道数这个维度，每个通道上的卷积核参数都是独立的训练参数）
	// 这块不是很清楚的童鞋建议好好看看卷积的原理，卷积核应该是一个思维的矩阵 size*size是在平面上的显示，size*size*输入数据通道是对每一个通道
	// 提取一组特征，size*size*c*输出通道数是为了提取多组特征
	// 其实这段代码应该放在给权重和偏置初始化语句的上面，这样就不用重复计算，效率提升一定要做到极致！
    l.nweights = c/groups*n*size*size;
    l.nbiases = n;

    // float scale = 1./sqrt(size*size*c);
    float scale = sqrt(2./(size*size*c/l.groups));
    // printf("convscale %f\n", scale);
    // scale = .02;
    // for(i = 0; i < c*n*size*size; ++i) l.weights[i] = scale*rand_uniform(-1, 1);
	// 初始化权重：缩放因子*标准正态分布随机数，缩放因子等于sqrt(2./(size*size*c))，为什么取这个值呢？？
    // 此处初始化权重为正态分布，而在全连接层make_connected_layer()中初始化权重是均匀分布的。
    // TODO：个人感觉，这里应该加一个if条件语句：if(weightfile)，因为如果导入了预训练权重文件，就没有必要这样初始化了（事实上在detector.c的train_detector()函数中，
    // 紧接着parse_network_cfg()函数之后，就添加了if(weightfile)语句判断是否导入权重系数文件，如果导入了权重系数文件，也许这里初始化的值也会覆盖掉，
    // 总之这里的权重初始化的处理方式还是值得思考的，也许更好的方式是应该设置专门的函数进行权重的初始化，同时偏置也是，不过这里似乎没有考虑偏置的初始化，在make_connected_layer()中倒是有。。。）
	// rand_normal函数定义在src/utils.c 生成标准正态分布随机数（float）
    for(i = 0; i < l.nweights; ++i) l.weights[i] = scale*rand_normal();
	
	// 根据该层输入图像的尺寸、卷积核尺寸以及跨度计算输出特征图的宽度和高度
	// convolutional_out_width函数和convolutional_out_height函数：本文件定义
    int out_w = convolutional_out_width(l);
    int out_h = convolutional_out_height(l);
	
	// 输出图像通道（等于卷积核个数，有多少个卷积核，最终就得到多少张特征图，每张图是一个通道）
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = n;
	
	// 对应每张输入图片的所有输出特征图的总元素个数（每张输入图片会得到n也即l.out_c张特征图），这
	// 一层的输出元素总个数对应着下一层的输入元素总个数
    l.outputs = l.out_h * l.out_w * l.out_c;
	// mini-batch中每张输入图片的像素元素个数
	// 一个mini-batch中有多张图片，每张图片可能有多个通道（彩色图有三通道），l.inputs是每张输入图片所有通道的总元素个数，
    // 而每张输入图片会有n个卷积核对其进行卷积操作，因此一张输入图片会输出n张特征图，这n张特征图的总元素个数就为l.outputs
    l.inputs = l.w * l.h * l.c;

	// l.output为该层所有的输出（包括mini-batch所有输入图片的输出）,有多少个batch就有多少张图片
    l.output = calloc(l.batch*l.outputs, sizeof(float));
	// delta 存储每一层的敏感度图
    l.delta  = calloc(l.batch*l.outputs, sizeof(float));

	// 卷积层三种指针函数，对应三种计算：前向，反向，更新
	/* 声明如下
		void (*forward)   (struct layer, struct network);
		void (*backward)  (struct layer, struct network);
		void (*update)    (struct layer, update_args);
		void (*forward_gpu)   (struct layer, struct network);
		void (*backward_gpu)  (struct layer, struct network);
		void (*update_gpu)    (struct layer, update_args);
	*/
	// 将函数指针指向具体函数，如下三个函数均在本文件定义，涉及到具体运算，读者可以先不推理，只了解函数功能即可
    l.forward = forward_convolutional_layer;
    l.backward = backward_convolutional_layer;
    l.update = update_convolutional_layer;
	
	// 其实如下部分的代码应该是和 是否使用了GPU的关系为 二选一的关系，但这样设计在使用了GPU的情况下，这些代码是否冗余？
	// 是否进行权重二值化
    if(binary){
        l.binary_weights = calloc(l.nweights, sizeof(float));
        l.cweights = calloc(l.nweights, sizeof(char));
        l.scales = calloc(n, sizeof(float));
    }
	// 是否对权重和输入二值化
    if(xnor){
        l.binary_weights = calloc(l.nweights, sizeof(float));
        l.binary_input = calloc(l.inputs*l.batch, sizeof(float));
    }

	// 是否规范化，通常都会进行规范化
    if(batch_normalize){
        l.scales = calloc(n, sizeof(float));
        l.scale_updates = calloc(n, sizeof(float));
        for(i = 0; i < n; ++i){
            l.scales[i] = 1;
        }

        l.mean = calloc(n, sizeof(float));
        l.variance = calloc(n, sizeof(float));

        l.mean_delta = calloc(n, sizeof(float));
        l.variance_delta = calloc(n, sizeof(float));

        l.rolling_mean = calloc(n, sizeof(float));
        l.rolling_variance = calloc(n, sizeof(float));
        l.x = calloc(l.batch*l.outputs, sizeof(float));
        l.x_norm = calloc(l.batch*l.outputs, sizeof(float));
    }
    if(adam){
        l.m = calloc(l.nweights, sizeof(float));
        l.v = calloc(l.nweights, sizeof(float));
        l.bias_m = calloc(n, sizeof(float));
        l.scale_m = calloc(n, sizeof(float));
        l.bias_v = calloc(n, sizeof(float));
        l.scale_v = calloc(n, sizeof(float));
    }

#ifdef GPU
	// 卷积层三种指针函数，对应三种计算：前向，反向，更新，此时是使用GPU时候的调用方式
	// 这三个函数定义在：src/convolutional_kernels.cu 中
    l.forward_gpu = forward_convolutional_layer_gpu;
    l.backward_gpu = backward_convolutional_layer_gpu;
    l.update_gpu = update_convolutional_layer_gpu;

    if(gpu_index >= 0){
        if (adam) {
            l.m_gpu = cuda_make_array(l.m, l.nweights);
            l.v_gpu = cuda_make_array(l.v, l.nweights);
            l.bias_m_gpu = cuda_make_array(l.bias_m, n);
            l.bias_v_gpu = cuda_make_array(l.bias_v, n);
            l.scale_m_gpu = cuda_make_array(l.scale_m, n);
            l.scale_v_gpu = cuda_make_array(l.scale_v, n);
        }

        l.weights_gpu = cuda_make_array(l.weights, l.nweights);
        l.weight_updates_gpu = cuda_make_array(l.weight_updates, l.nweights);

        l.biases_gpu = cuda_make_array(l.biases, n);
        l.bias_updates_gpu = cuda_make_array(l.bias_updates, n);

        l.delta_gpu = cuda_make_array(l.delta, l.batch*out_h*out_w*n);
        l.output_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);

        if(binary){
            l.binary_weights_gpu = cuda_make_array(l.weights, l.nweights);
        }
        if(xnor){
            l.binary_weights_gpu = cuda_make_array(l.weights, l.nweights);
            l.binary_input_gpu = cuda_make_array(0, l.inputs*l.batch);
        }

        if(batch_normalize){
            l.mean_gpu = cuda_make_array(l.mean, n);
            l.variance_gpu = cuda_make_array(l.variance, n);

            l.rolling_mean_gpu = cuda_make_array(l.mean, n);
            l.rolling_variance_gpu = cuda_make_array(l.variance, n);

            l.mean_delta_gpu = cuda_make_array(l.mean, n);
            l.variance_delta_gpu = cuda_make_array(l.variance, n);

            l.scales_gpu = cuda_make_array(l.scales, n);
            l.scale_updates_gpu = cuda_make_array(l.scale_updates, n);

            l.x_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
            l.x_norm_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
        }
#ifdef CUDNN
        cudnnCreateTensorDescriptor(&l.normTensorDesc);
        cudnnCreateTensorDescriptor(&l.srcTensorDesc);
        cudnnCreateTensorDescriptor(&l.dstTensorDesc);
        cudnnCreateFilterDescriptor(&l.weightDesc);
        cudnnCreateTensorDescriptor(&l.dsrcTensorDesc);
        cudnnCreateTensorDescriptor(&l.ddstTensorDesc);
        cudnnCreateFilterDescriptor(&l.dweightDesc);
        cudnnCreateConvolutionDescriptor(&l.convDesc);
        cudnn_convolutional_setup(&l);
#endif
    }
#endif
    // get_workspace_size函数：本文件 
    l.workspace_size = get_workspace_size(l);
    l.activation = activation;

	// 向屏幕输出该层网络的名称、filters数目，size、输入和输出信息
    fprintf(stderr, "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f BFLOPs\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c, (2.0 * l.n * l.size*l.size*l.c/l.groups * l.out_h*l.out_w)/1000000000.);

    return l;
}

void denormalize_convolutional_layer(convolutional_layer l)
{
    int i, j;
    for(i = 0; i < l.n; ++i){
        float scale = l.scales[i]/sqrt(l.rolling_variance[i] + .00001);
        for(j = 0; j < l.c/l.groups*l.size*l.size; ++j){
            l.weights[i*l.c/l.groups*l.size*l.size + j] *= scale;
        }
        l.biases[i] -= l.rolling_mean[i] * scale;
        l.scales[i] = 1;
        l.rolling_mean[i] = 0;
        l.rolling_variance[i] = 1;
    }
}

/*
void test_convolutional_layer()
{
    convolutional_layer l = make_convolutional_layer(1, 5, 5, 3, 2, 5, 2, 1, LEAKY, 1, 0, 0, 0);
    l.batch_normalize = 1;
    float data[] = {1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3};
    //net.input = data;
    //forward_convolutional_layer(l);
}
*/

void resize_convolutional_layer(convolutional_layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    int out_w = convolutional_out_width(*l);
    int out_h = convolutional_out_height(*l);

    l->out_w = out_w;
    l->out_h = out_h;

    l->outputs = l->out_h * l->out_w * l->out_c;
    l->inputs = l->w * l->h * l->c;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta  = realloc(l->delta,  l->batch*l->outputs*sizeof(float));
    if(l->batch_normalize){
        l->x = realloc(l->x, l->batch*l->outputs*sizeof(float));
        l->x_norm  = realloc(l->x_norm, l->batch*l->outputs*sizeof(float));
    }

#ifdef GPU
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu =  cuda_make_array(l->delta,  l->batch*l->outputs);
    l->output_gpu = cuda_make_array(l->output, l->batch*l->outputs);

    if(l->batch_normalize){
        cuda_free(l->x_gpu);
        cuda_free(l->x_norm_gpu);

        l->x_gpu = cuda_make_array(l->output, l->batch*l->outputs);
        l->x_norm_gpu = cuda_make_array(l->output, l->batch*l->outputs);
    }
#ifdef CUDNN
    cudnn_convolutional_setup(l);
#endif
#endif
    l->workspace_size = get_workspace_size(*l);
}

void add_bias(float *output, float *biases, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] += biases[i];
            }
        }
    }
}

void scale_bias(float *output, float *scales, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] *= scales[i];
            }
        }
    }
}

void backward_bias(float *bias_updates, float *delta, int batch, int n, int size)
{
    int i,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            bias_updates[i] += sum_array(delta+size*(i+b*n), size);
        }
    }
}



/********************************************************************************************
*  func：实现卷积层的前向传输，会利用该卷积层的参数进行计算，该层的输出会通过l的output保留  *
*  args l：该层卷积层的结构体，包括了权重和偏置等                                           *
*  args net: 这一个参数主要是在调用forward的时候填充                                        *
********************************************************************************************/
void forward_convolutional_layer(convolutional_layer l, network net)
{
    int i, j;
	// l.outputs = l.out_h * l.out_w * l.out_c在make各网络层函数中赋值（比如make_convolutional_layer()），
    // 对应每张输入图片的所有输出特征图的总元素个数（每张输入图片会得到n也即l.out_c张特征图）
    // 初始化输出l.output全为0.0；输入l.outputs*l.batch为输出的总元素个数，其中l.outputs为batch
    // 中一个输入对应的输出的所有元素的个数，l.batch为一个batch输入包含的图片张数；0表示初始化所有输出为0；
    // fill_cpu函数：src/blas.c 以步长1初始化l.output数组所有元素的值为0
    fill_cpu(l.outputs*l.batch, 0, l.output, 1);

	// 是否进行二值化操作（这个操作应该只有第一个卷积层使用吧？因为下面直接对net.input操作，这个理解是错误的，因为在forward_network()含中，
    // 每进行一层都会将net.input = l.output，即下一层的输入被设置为当前层的输出）
    if(l.xnor){
        binarize_weights(l.weights, l.n, l.c/l.groups*l.size*l.size, l.binary_weights);
        swap_binary(&l);
        binarize_cpu(net.input, l.c*l.h*l.w*l.batch, l.binary_input);
        net.input = l.binary_input;
    }

	// 该层卷积核个数
    int m = l.n/l.groups;
	// 该层每个卷积核的参数元素个数，注意，这儿是一个卷积核中的元素个数！
    int k = l.size*l.size*l.c/l.groups;
	
	// 该层每个特征图的尺寸（元素个数）
    int n = l.out_w*l.out_h;
	
	
	// l.weights;       所有卷积核（也即权重），元素个数为l.n*l.c*l.size*l.size，按行存储，共有l.n行，l.c*l.size*l.size列
    // net.workspace;   对输入图像进行重排之后的图像数据
    // l.output;        存储一张输入图片（多通道）所有的输出特征图（输入图片是多通道的，输出图片也是多通道的，有多少个卷积核就有多少个通道，每个卷积核得到一张特征图即为一个通道）
	
	// 该循环即为卷积计算核心代码：所有卷积核对batch中每张图片进行卷积运算！！
    // 可以参考：https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/
    // 进行辅助理解（主要是辅助理解，实际执行并不一样）。
    // 每次循环处理一张输入图片（所有卷积核对batch中一张图片做卷积运算）
    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
			// a为权重的起始指针
            float *a = l.weights + j*l.nweights/l.groups;
            float *b = net.workspace;
            float *c = l.output + (i*l.groups + j)*n*m;
            float *im =  net.input + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

            if (l.size == 1) {
                b = im;
            } else {
				// 将多通道二维图像net.input变成按一定存储规则排列的数组b，以方便、高效地进行矩阵（卷积）计算，详细查看该函数注释（比较复杂）
				// 注意net.input包含batch中所有图片的数据，但是每次循环只处理一张（本循环最后一句对net.input进行了移位），因此在im2col_cpu仅会对其中一张图片
				// 进行重排，l.c为每张图片的通道数，l.h为每张图片的高度，l.w为每张图片的宽度，l.size为卷积核尺寸，l.stride为跨度
				// 得到的b为一张图片重排后的结果，也是按行存储的一维数组（共有l.c*l.size*l.size行，l.out_w*l.out_h列），
                im2col_cpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
            }
			// GEneral Matrix to Matrix Multiplication
			// 此处在im2col_cpu操作基础上，利用矩阵乘法c=alpha*a*b+beta*c完成对图像卷积的操作
			// 0,0表示不对输入a,b进行转置，
			// m是输入a,c的行数，具体含义为每个卷积核的个数，
			// n是输入b,c的列数，具体含义为每个输出特征图的元素个数(out_h*out_w)，
			// k是输入a的列数也是b的行数，具体含义为卷积核元素个数乘以输入图像的通道数（l.size*l.size*l.c），
			// a,b,c即为三个参与运算的矩阵（用一维数组存储）,alpha=beta=1为常系数，
			// a为所有卷积核集合,元素个数为l.n*l.c*l.size*l.size，按行存储，共有l*n行，l.c*l.size*l.size列，
			// 即a中每行代表一个可以作用在3通道上的卷积核，
			// b为一张输入图像经过im2col_cpu重排后的图像数据（共有l.c*l.size*l.size行，l.out_w*l.out_h列），
			// c为gemm()计算得到的值，包含一张输入图片得到的所有输出特征图（每个卷积核得到一张特征图），c中一行代表一张特征图，
			// 各特征图铺排开成一行后，再将所有特征图并成一大行，存储在c中，因此c可视作有l.n行，l.out_h*l.out_w列。
			// 详细查看该函数注释（比较复杂）
            gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
        }
    }
    // 如需要规范化（BN在非线性激活函数处理之前完成）
    if(l.batch_normalize){
		// forward_batchnorm_layer函数：src/batchnorm_layer.c
        forward_batchnorm_layer(l, net);
    } else {
		// add_bias函数：本文件。
        add_bias(l.output, l.biases, l.batch, l.n, l.out_h*l.out_w);
    }
    // activate_array函数：src/activation.c
    activate_array(l.output, l.outputs*l.batch, l.activation);
    if(l.binary || l.xnor) swap_binary(&l);
}



/*
** 卷积神经网络反向传播核心函数
** 主要流程：1） 调用gradient_array()计算当前层l所有输出元素关于加权输入的导数值（也即激活函数关于输入的导数值），
**             并乘上上一次调用backward_convolutional_layer()还没计算完的l.delta，得到当前层最终的敏感度图；
**          2） 如果网络进行了BN，则；
**          3） 如果网络没有进行BN，则直接调用 backward_bias()计算当前层所有卷积核的偏置更新值；
**          4） 依次调用im2col_cpu()，gemm_nt()函数计算当前层权重系数更新值；
**          5） 如果上一层的delta已经动态分配了内存，则依次调用gemm_tn(), col2im_cpu()计算上一层的敏感度图（并未完成所有计算，还差一个步骤）；
** 强调：每次调用本函数会计算完成当前层的敏感度计算，同时计算当前层的偏置、权重更新值，除此之外，还会计算上一层的敏感度图，但是要注意的是，
**      并没有完全计算完，还差一步：乘上激活函数对加权输入的导数值。这一步在下一次调用本函数时完成。
*/
void backward_convolutional_layer(convolutional_layer l, network net)
{
    int i, j;
    int m = l.n/l.groups;
    int n = l.size*l.size*l.c/l.groups;
    int k = l.out_w*l.out_h;

    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);

    if(l.batch_normalize){
        backward_batchnorm_layer(l, net);
    } else {
        backward_bias(l.bias_updates, l.delta, l.batch, l.n, k);
    }

    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            float *a = l.delta + (i*l.groups + j)*m*k;
            float *b = net.workspace;
            float *c = l.weight_updates + j*l.nweights/l.groups;

            float *im  = net.input + (i*l.groups + j)*l.c/l.groups*l.h*l.w;
            float *imd = net.delta + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

            if(l.size == 1){
                b = im;
            } else {
                im2col_cpu(im, l.c/l.groups, l.h, l.w, 
                        l.size, l.stride, l.pad, b);
            }

            gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);

            if (net.delta) {
                a = l.weights + j*l.nweights/l.groups;
                b = l.delta + (i*l.groups + j)*m*k;
                c = net.workspace;
                if (l.size == 1) {
                    c = imd;
                }

                gemm(1,0,n,k,m,1,a,n,b,k,0,c,k);

                if (l.size != 1) {
                    col2im_cpu(net.workspace, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, imd);
                }
            }
        }
    }
}

void update_convolutional_layer(convolutional_layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;

    axpy_cpu(l.n, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.n, momentum, l.bias_updates, 1);

    if(l.scales){
        axpy_cpu(l.n, learning_rate/batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.n, momentum, l.scale_updates, 1);
    }

    axpy_cpu(l.nweights, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(l.nweights, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(l.nweights, momentum, l.weight_updates, 1);
}


image get_convolutional_weight(convolutional_layer l, int i)
{
    int h = l.size;
    int w = l.size;
    int c = l.c/l.groups;
    return float_to_image(w,h,c,l.weights+i*h*w*c);
}

void rgbgr_weights(convolutional_layer l)
{
    int i;
    for(i = 0; i < l.n; ++i){
        image im = get_convolutional_weight(l, i);
        if (im.c == 3) {
            rgbgr_image(im);
        }
    }
}

void rescale_weights(convolutional_layer l, float scale, float trans)
{
    int i;
    for(i = 0; i < l.n; ++i){
        image im = get_convolutional_weight(l, i);
        if (im.c == 3) {
            scale_image(im, scale);
            float sum = sum_array(im.data, im.w*im.h*im.c);
            l.biases[i] += sum*trans;
        }
    }
}

image *get_weights(convolutional_layer l)
{
    image *weights = calloc(l.n, sizeof(image));
    int i;
    for(i = 0; i < l.n; ++i){
        weights[i] = copy_image(get_convolutional_weight(l, i));
        normalize_image(weights[i]);
        /*
           char buff[256];
           sprintf(buff, "filter%d", i);
           save_image(weights[i], buff);
         */
    }
    //error("hey");
    return weights;
}

image *visualize_convolutional_layer(convolutional_layer l, char *window, image *prev_weights)
{
    image *single_weights = get_weights(l);
    show_images(single_weights, l.n, window);

    image delta = get_convolutional_image(l);
    image dc = collapse_image_layers(delta, 1);
    char buff[256];
    sprintf(buff, "%s: Output", window);
    //show_image(dc, buff);
    //save_image(dc, buff);
    free_image(dc);
    return single_weights;
}

