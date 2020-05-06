#include "yolo_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "dark_cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>


/**************************************************************************************
*  func：构建yolo层，初始化yolo层的相关参数，返回层指针                               *
*  args batch：该层的batch，与网络batch大小一致                                       *
*  args w：上一层网络的输出，本层网络的输入width                                      *
*  args h: 本层网络的输入height                                                       *
*  args n: 本层anchor的个数，yolo v3中为3个                                           *
*  args total：anchor的总个数                                                         *
*  args mask：保存本层使用anchor的索引                                                *
*  args classes: 类别数目                                                             *
*  args max_boxes：最大检测框个数                                                     *
**************************************************************************************/
layer make_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes, int max_boxes)
{
    int i;
    layer l = { (LAYER_TYPE)0 };
    l.type = YOLO;

    // 一个cell（网格）中预测多少个矩形框（box）
    l.n = n;
    l.total = total;
    l.batch = batch;
    l.h = h;
    l.w = w;
    // 在yolov3中，预测值也就是输出值的每一维包括：每个类别的概率值+4个BOX的信息+1个confidence
    // 每个cell有3个anchor，这里为参数n，所以输出的总维度为：n*(classes + 4 + 1)
    // yolo_layer层的输入和输出尺寸一致，通道数也一样，也就是这一层并不改变输入数据的维度
    l.c = n*(classes + 4 + 1);
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    // cost保存目标函数值，为单精度浮点型指针
    l.cost = (float*)calloc(1, sizeof(float));
    // 偏置参数，其实用来存储anchor
    l.biases = (float*)calloc(total * 2, sizeof(float));
    if(mask) l.mask = mask;
    // 如果不指定mask 则mask 值为1到n-1
    else{
        l.mask = (int*)calloc(n, sizeof(int));
        for(i = 0; i < n; ++i){
            l.mask[i] = i;
        }
    }
    l.bias_updates = (float*)calloc(n * 2, sizeof(float));
    // 一张训练图片经过yolo_layer层后得到的输出元素个数（等于网格数*每个网格预测的矩形框数*每个矩形框的参数个数）
    l.outputs = h*w*n*(classes + 4 + 1);
    // 一张训练图片输入到yolo_layer层的元素个数（注意是一张图片，对于yolo_layer，输入和输出的元素个数相等）
    l.inputs = l.outputs;
    l.max_boxes = max_boxes;
    /*
      每张图片含有的真实矩形框参数的个数（90表示一张图片中最多有90个ground truth矩形框，每个真实矩形框有
      5个参数，包括x,y,w,h四个定位参数，以及物体类别）,注意90是darknet程序内写死的，实际上每张图片可能
      并没有90个真实矩形框，也能没有这么多参数，但为了保持一致性，还是会留着这么大的存储空间，只是其中的
      值未空而已.
    */
    l.truths = l.max_boxes*(4 + 1);    // 90*(4 + 1);
    l.delta = (float*)calloc(batch * l.outputs, sizeof(float));

    /*
     * yolo_layer的输出维度为l.out_w*l.out_h，等于输入的维度，输出通道数为l.out_c，等于输入通道数，
     * 且通道数等于n*(classes+4+1)。那yolo_layer的输出l.output中到底存储了什么呢？存储了
     * 所有网格（grid cell）中预测矩形框（box）的所有信息。看Yolo论文就知道，Yolo检测模型最终将图片
     * 划分成了S*S（论文中为13*13）个网格，每个网格中预测B个（论文中B=3）矩形框，最后一层输出的就是这些
     * 网格中所包含的所有预测矩形框信息。目标检测模型中，作者用矩形框来表示并定位检测到的物体，每个矩形框中
     * 包含了矩形框定位信息x,y,w,h，含有物体的自信度信息c，以及属于各类的概率（如果有20类，那么就有矩形框
     * 中所包含物体属于这20类的概率）。注意了，这里的实现与论文中的描述有不同，首先参数固然可能不同（比如
     * 并不像论文中那样每个网格预测3个box，也有可能更多），更为关键的是，输出维度的计算方式不同，论文中提到
     * 最后一层输出的维度为一个S_w*S_c*(B*5+C)的tensor（作者在论文中是S*S，这里我写成S_w，S_c是考虑到
     * 网格划分维度不一定S_w=S_c=S，不过貌似作者用的都是S_w=S_c的，比如7*7,13*13，总之明白就可以了），
     * 实际上，这里有点不同，输出的维度应该为S_w*S_c*B*(5+C),C为类别数目，比如共有20类；5是因为有4个定位
     * 信息，外加一个自信度信息c，共有5个参数。也即每个矩形框都包含一个属于各类的概率，并不是所有矩形框共有
     * 一组属于各类的概率，这点可以从l.outputs的计算方式中看出（可以对应上，l.out_w = S_w, l.out_c = S_c,
     * l.out_c = B*(5+C)）。知道输出到底存储什么之后，接下来要搞清是怎么存储的，毕竟输出的是一个三维张量，
     * 但实现中是用一个一维数组来存储的，详细的注释可以参考下面forward_yolo_layer()以及entry_index()
     * 函数的注释，这个东西仅用文字还是比较难叙述的，应该借助图来说明～
    */
    l.output = (float*)calloc(batch * l.outputs, sizeof(float));
    // 为什么在这要做这么一步初始化anchor值的工作？在parse_yolo中不是会做相应的处理吗？
    for(i = 0; i < total*2; ++i){
        l.biases[i] = .5;
    }

    l.forward = forward_yolo_layer;
    l.backward = backward_yolo_layer;
#ifdef GPU
    l.forward_gpu = forward_yolo_layer_gpu;
    l.backward_gpu = backward_yolo_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);

    free(l.output);
    if (cudaSuccess == cudaHostAlloc(&l.output, batch*l.outputs*sizeof(float), cudaHostRegisterMapped)) l.output_pinned = 1;
    else {
        cudaGetLastError(); // reset CUDA-error
        l.output = (float*)calloc(batch * l.outputs, sizeof(float));
    }

    free(l.delta);
    if (cudaSuccess == cudaHostAlloc(&l.delta, batch*l.outputs*sizeof(float), cudaHostRegisterMapped)) l.delta_pinned = 1;
    else {
        cudaGetLastError(); // reset CUDA-error
        l.delta = (float*)calloc(batch * l.outputs, sizeof(float));
    }
#endif

    fprintf(stderr, "yolo\n");
    srand(time(0));

    return l;
}

void resize_yolo_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;

    l->outputs = h*w*l->n*(l->classes + 4 + 1);
    l->inputs = l->outputs;

    if (!l->output_pinned) l->output = (float*)realloc(l->output, l->batch*l->outputs * sizeof(float));
    if (!l->delta_pinned) l->delta = (float*)realloc(l->delta, l->batch*l->outputs*sizeof(float));

#ifdef GPU
    if (l->output_pinned) {
        cudaFreeHost(l->output);
        if (cudaSuccess != cudaHostAlloc(&l->output, l->batch*l->outputs * sizeof(float), cudaHostRegisterMapped)) {
            cudaGetLastError(); // reset CUDA-error
            l->output = (float*)realloc(l->output, l->batch * l->outputs * sizeof(float));
            l->output_pinned = 0;
        }
    }

    if (l->delta_pinned) {
        cudaFreeHost(l->delta);
        if (cudaSuccess != cudaHostAlloc(&l->delta, l->batch*l->outputs * sizeof(float), cudaHostRegisterMapped)) {
            cudaGetLastError(); // reset CUDA-error
            l->delta = (float*)realloc(l->delta, l->batch * l->outputs * sizeof(float));
            l->delta_pinned = 0;
        }
    }

    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu =     cuda_make_array(l->delta, l->batch*l->outputs);
    l->output_gpu =    cuda_make_array(l->output, l->batch*l->outputs);
#endif
}

/************************************************************************************************************
*  func: 获取某个矩形框的4个定位信息（根据输入的矩形框索引从l.output中获取该矩形框的定位信息x,y,w,h）.      *
*  args x：yolo_layer的输出，即l.output，包含所有batch预测得到的矩形框信息                                  *
*  args biases: 这个参数千万别被名称误导，里面存放的是所有的anchor,yolo3的话有2*9=18个值                    *
*  args n: 这是每个yolo层中anchor的索引，拿最后一个yolo层举例，其mask=0，1，2 此处的n就表示这个0,1,2的索引  *
*  args index: 矩形框的首地址（索引，矩形框中存储的首个参数x在l.output中的索引）                            *
*  args i: 列索引                                                                                           *
*  args j: 行索引                                                                                           *
*  args lw: 该层的width                                                                                     *
*  args lh: 该层的height                                                                                    *
*  args w: 网络的width                                                                                      *
*  args h: 网络的height                                                                                     *
*  args stride: 这个参数实际需要？意思是每张图片总共有多少个cell? 值为：l.w*l.h                             *
************************************************************************************************************/
box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride)
{
    box b;
    // ln - natural logarithm (base = e)
    // x` = t.x * lw - i;   // x = ln(x`/(1-x`))   // x - output of previous conv-layer
    // y` = t.y * lh - i;   // y = ln(y`/(1-y`))   // y - output of previous conv-layer
                            // w = ln(t.w * net.w / anchors_w); // w - output of previous conv-layer
                            // h = ln(t.h * net.h / anchors_h); // h - output of previous conv-layer
    // x[index+0*stride]为预测的x的值，加上i是因为预测值是相对于cell的索引进行的，例如，当前cell为第三行，第四列的cell
    // 则此时i=3,j=2，则x加上3表示现在x在特征图上的绝对值，随后除以特征图的宽度即得到了相对值
    b.x = (i + x[index + 0*stride]) / lw;
    // 此处相当于知道了X的index,要找Y的index,向后偏移l.w*l.h个索引
    b.y = (j + x[index + 1*stride]) / lh;
    // 此处exp函数可以验证w和h为啥不用logis函数激活？
    // 首先x[index + 2*stride]取出对应的W值，再做exp函数处理，biases[2*n]得到的是其对应anchor的w，最后除以网络的宽度得到其相对于网络的比例
    // 此处计算b.w要结合论文的公式理解，参考 https://blog.csdn.net/hrsstudy/article/details/70305791 讲得贼清楚
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    return b;
}

// 功能：box delta这里没什么可说的，就是square error的求导
float delta_yolo_box(box truth, float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, float *delta, float scale, int stride)
{
    // 得到预测空的边框信息
    box pred = get_yolo_box(x, biases, n, index, i, j, lw, lh, w, h, stride);    
    float iou = box_iou(pred, truth);

    // 构建标签中的x,y,w,h信息，使之与预测的一样
    // truth.x和truth.y是一个相对值，所以直接可以乘以特征图得尺寸得到在特征图上的x和y，然后减去cell的偏移
    float tx = (truth.x*lw - i);
    float ty = (truth.y*lh - j);
    // 对于w和h的处理，先映射到相对于网络输入的width和height的尺寸，再转化为相对于anchor的缩放值
    float tw = log(truth.w*w / biases[2*n]);
    float th = log(truth.h*h / biases[2*n + 1]);
    // 分别计算位置损失 
    delta[index + 0*stride] = scale * (tx - x[index + 0*stride]);
    delta[index + 1*stride] = scale * (ty - x[index + 1*stride]);
    delta[index + 2*stride] = scale * (tw - x[index + 2*stride]);
    delta[index + 3*stride] = scale * (th - x[index + 3*stride]);
    return iou;
}

// 功能：计算类别的损失函数
// 参数：输出数据的指针，损失保存的指针，index指示了类别概率在output中的起始存储位置,classes表示了预测的类别
//       stride表示偏移量，也就是一个特征图的w*h，avg_cat传递进来的值为0，最后一个参数表示是否使用了focal_loss
void delta_yolo_class(float *output, float *delta, int index, int class_id, int classes, int stride, float *avg_cat, int focal_loss)
{
    int n;
    // 话说，什么样的情况下不为0呢？
    if (delta[index + stride*class_id]){
        delta[index + stride*class_id] = 1 - output[index + stride*class_id];
        if(avg_cat) *avg_cat += output[index + stride*class_id];
        return;
    }
    // Focal loss
    // 如果定义了focal_loss则执行此
    if (focal_loss) {
        // Focal Loss
        float alpha = 0.5;    // 0.25 or 0.5
        //float gamma = 2;    // hardcoded in many places of the grad-formula

        int ti = index + stride*class_id;
        float pt = output[ti] + 0.000000000000001F;
        // http://fooplot.com/#W3sidHlwZSI6MCwiZXEiOiItKDEteCkqKDIqeCpsb2coeCkreC0xKSIsImNvbG9yIjoiIzAwMDAwMCJ9LHsidHlwZSI6MTAwMH1d
        float grad = -(1 - pt) * (2 * pt*logf(pt) + pt - 1);    // http://blog.csdn.net/linmingan/article/details/77885832
        //float grad = (1 - pt) * (2 * pt*logf(pt) + pt - 1);    // https://github.com/unsky/focal-loss

        for (n = 0; n < classes; ++n) {
            delta[index + stride*n] = (((n == class_id) ? 1 : 0) - output[index + stride*n]);

            delta[index + stride*n] *= alpha*grad;

            if (n == class_id) *avg_cat += output[index + stride*n];
        }
    }
    else {
        // default
        for (n = 0; n < classes; ++n) {
            // 将类别对应处的损失设置为1-其类别得分。其它位置设置为0-类别得分，这也是LOGISTIC求导后公式的应用
            delta[index + stride*n] = ((n == class_id) ? 1 : 0) - output[index + stride*n];
            if (n == class_id && avg_cat) *avg_cat += output[index + stride*n];
        }
    }
}

/****************************************************************************************************************************************
*  func：计算某个矩形框中某个参数在l.output中的索引。一个矩形框包含了x,y,w,h,c,C1,C2...,Cn信息，前四个用于定位，第五个为矩形框含有物体  *
*        的自信度信息c，即矩形框中存在物体的概率为多大，而C1到Cn为矩形框中所包含的物体分别属于这n类物体的概率。本函数负责获取该矩形框   *
*        首个定位信息也即x值在l.output中索引、获取该矩形框自信度信息c在l.output中的索引、获取该矩形框分类所属概率的首个概率也即C1值的   *
*        索引，具体是获取矩形框哪个参数的索引，取决于输入参数entry的值，这些在forward_yolo_layer()函数中都有用到，由于l.output的存储    *
*        方式，当entry=0时，就是获取矩形框x参数在l.output中的索引；当entry=4时，就是获取矩形框自信度信息c在l.output中的索引；当entry=5  *
*        时，就是获取矩形框首个所属概率C1在l.output中的索引，具体可以参考forward_yolo_layer()中调用本函数时的注释.                      *                                                                                                      *
*  args l: 当前yolo_layer                                                                                                               *
*  args batch：当前照片是整个batch中的第几张，因为l.output中包含整个batch的输出，所以要定位某张训练图片输出的众多网格中的某个矩形框，当 *
*              然需要该参数.                                                                                                            *
*  args location：这个参数，说实话，感觉像鸡肋参数，函数中用这个参数获取n和loc的值，这个n就是表示网格中的第几个预测矩形框（比如每个网格 *
*                 预测5个矩形框，那么n取值范围就是从0~4），loc就是某个通道上的元素偏移（yolo_layer输出的通道数为                        *
*                 l.out_c = (classes + 4 + 1)）,这样说可能没有说明白，这都与l.output的存储结构相关，见下面详细注释以及其他说明。总之，  *                                *
*                 查看一下调用本函数的父函数forward_yolo_layer()就知道了，可以直接输入n和j*l.w+i的，没有必要输入location，这样还得重新  *
*                 计算一次n和loc.                                                                                                       *
*  args entry：切入点偏移系数，关于这个参数，就又要扯到l.output的存储结构了，见下面详细注释以及其他说明                                 *
*  details：l.output这个参数的存储内容以及存储方式已经在多个地方说明了，再多的文字都不及图文说明，此处再                                *
*           简要罗嗦几句，更为具体的参考图文说明。l.output中存储了整个batch的训练输出，每张训练图片都会输出                             *
*           l.out_w*l.out_h个网格，每个网格会预测l.n个矩形框，每个矩形框含有l.classes+4+1个参数，                                       *
*           而最后一层的输出通道数为l.n*(l.classes+4+1)，可以想象下最终输出的三维张量是个什么样子的。                                   *
*           展成一维数组存储时，l.output可以首先分成batch个大段，每个大段存储了一张训练图片的所有输出；进一步细分，                     *
*           取其中第一大段分析，该大段中存储了第一张训练图片所有输出网格预测的矩形框信息，每个网格预测了l.n个矩形框，                   *
*           存储时，l.n个矩形框是分开存储的，也就是先存储所有网格中的第一个矩形框，而后存储所有网格中的第二个矩形框，                   *
*           依次类推，如果每个网格中预测5个矩形框，则可以继续把这一大段分成5个中段。继续细分，5个中段中取第                             *
*           一个中段来分析，这个中段中按行（有l.out_w*l.out_h个网格，按行存储）依次存储了这张训练图片所有输出网格中                     *
*           的第一个矩形框信息，要注意的是，这个中段存储的顺序并不是挨个挨个存储每个矩形框的所有信息，                                  *
*           而是先存储所有矩形框的x，而后是所有的y,然后是所有的w,再是h，c，最后的的概率数组也是拆分进行存储，                           *
*           并不是一下子存储完一个矩形框所有类的概率，而是先存储所有网格所属第一类的概率，再存储所属第二类的概率，                      *
*           具体来说这一中段首先存储了l.out_w*l.out_h个x，然后是l.out_w*l.out_h个y，依次下去，                                          *
*           最后是l.out_w*l.out_h个C1（属于第一类的概率，用C1表示，下面类似），l.out_w*l.out_h个C2,...,                                 *
*           l.out_w*l.out_h*Cn（假设共有n类），所以可以继续将中段分成几个小段，依次为x,y,w,h,c,C1,C2,...Cn                              *
*           小段，每小段的长度都为l.out_w*l.out_h.现在回过来看本函数的输入参数，batch就是大段的偏移数（从第几个大段开始，对应是第几     *
*           张训练图片），由location计算得到的n就是中段的偏移数（从第几个中段开始，对应是第几个矩形框），                               *
*           entry就是小段的偏移数（从几个小段开始，对应具体是那种参数，x,c还是C1），而loc则是最后的定位，                               *
*           前面确定好第几大段中的第几中段中的第几小段的首地址，loc就是从该首地址往后数loc个元素，得到最终定位                          *
*           某个具体参数（x或c或C1）的索引值，比如l.output中存储的数据如下所示（这里假设只存了一张训练图片的输出，                      *
*           因此batch只能为1；并假设l.out_w=l.out_h=2,l.classes=2）：                                                                   *
*           xxxxyyyywwwwhhhhccccC1C1C1C1C2C2C2C2-#-xxxxyyyywwwwhhhhccccC1C1C1C1C2C2C2C2，                                               *
*           n=0则定位到-#-左边的首地址（表示每个网格预测的第一个矩形框），n=1则定位到-#-右边的首地址（表示每个网格预测的第二个矩形框）  *
*           entry=0,loc=0获取的是x的索引，且获取的是第一个x也即l.out_w*l.out_h个网格中第一个网格中第一个矩形框x参数的索引；             *
*           entry=4,loc=1获取的是c的索引，且获取的是第二个c也即l.out_w*l.out_h个网格中第二个网格中第一个矩形框c参数的索引；             *
*           entry=5,loc=2获取的是C1的索引，且获取的是第三个C1也即l.out_w*l.out_h个网格中第三个网格中第一个矩形框C1参数的索引；          *
*           如果要获取第一个网格中第一个矩形框w参数的索引呢？如果已经获取了其x值的索引，显然用x的索引加上3*l.out_w*l.out_h即可获取到，  *
*           如果要获取第三个网格中第一个矩形框C2参数的索引呢？如果已经获取了其C1值的索引，显然用C1的索引加上l.out_w*l.out_h即可获取到， *
*           由上可知，entry=0时,即偏移0个小段，是获取x的索引；entry=4,是获取自信度信息c的索引；entry=5，是获取C1的索引.                 *
*           l.output的存储方式大致就是这样，个人觉得说的已经很清楚了，但可视化效果终究不如图文说明～                                    *
****************************************************************************************************************************************/
static int entry_index(layer l, int batch, int location, int entry)
{
    int n =   location / (l.w*l.h);
    int loc = location % (l.w*l.h);
    // 如下公式解释：batch*l.outputs l.outputs表示的是本层一张图片的输出维度，乘以batch则表示是第几张图片的偏移
    // n*l.w*l.h*(4+l.classes+1) 其中n表示第几个框，以yolo3为例，假设n=2则此时得到的是第二类框的偏移量
    // entry*l.w*l.h 每种信息占的位数为l.w*l.h 你要求的第几类信息的偏移量就乘以长度
    // loc表示最终的小段偏移，比如所有x中的第几个x
    // 若再不理解，可以结合forward_yolo_layer中四层循环处的说明理解
    return batch*l.outputs + n*l.w*l.h*(4+l.classes+1) + entry*l.w*l.h + loc;
}

/***************************************************************************************************************
*  func: 从存储矩形框信息的大数组中，提取某一个矩形框定位信息并返回（f是这个大数组中某一起始地址）.            *
*  args f：矩形框信息（每个矩形框有5个参数值，此函数仅提取其中4个用于定位的参数x,y,w,h，不包含物体类别编号）   *
*  args stride: 跨度，按倍数跳越取值，此处要结合net.truth的存储结构理解                                        *
***************************************************************************************************************/
static box float_to_box_stride(float *f, int stride)
{
    // f中存储每一个矩形框信息的顺序为: x, y, w, h, class_index，这里仅提取前四个，
    // 也即矩形框的定位信息，最后一个物体类别编号信息不在此处提取
    box b = { 0 };
    b.x = f[0];
    b.y = f[1 * stride];
    b.w = f[2 * stride];
    b.h = f[3 * stride];
    return b;
}


/*****************************************************************************************************************************
*  func：完成yolo层的前向传递                                                                                                *
*  args l : 本层的配置参数指针                                                                                               *
*  args state：上一层的相关信息                                                                                              *
*  details：本函数多次调用了entry_index()函数，且使用的参数不尽相同，尤其是最后一个参数，通过最后一个参数，                  *
*           可以确定出yolo_layer输出l.output的数据存储方式。为方便叙述，假设本层输出参数l.w = 2, l.h= 3,                     *
*           l.n = 2, l.classes = 2, l.c = l.n * (4 + l.classes + 1) = 14,                                                    *
*           l.output中存储了所有矩形框的信息参数，每个矩形框包括4条定位信息参数x,y,w,h，一条自信度（confidience）            *
*           参数c，以及所有类别的概率C1,C2（本例中，假设就只有两个类别，l.classes=2），那么一张样本图片最终会有              *
*           l.w*l.h*l.n个矩形框（l.w*l.h即为最终图像划分层网格的个数，每个网格预测l.n个矩形框），那么                        *
*           l.output中存储的元素个数共有l.w*l.h*l.n*(4 + 1 + l.classes)，这些元素全部拉伸成一维数组                          *
*           的形式存储在l.output中，存储的顺序为：                                                                           *
*           xxxxxx-yyyyyy-wwwwww-hhhhhh-cccccc-C1C1C1C1C1C1C2C2C2C2C2C2-                                                     *
*           ##-xxxxxx-yyyyyy-wwwwww-hhhhhh-cccccc-C1C2C1C2C1C2C1C2C1C2C1C2                                                   *
*           文字说明如下：-##-隔开分成两段，左右分别是代表所有网格的第1个box和第2个box（因为l.n=2，表示每个网格预测两个box） *
*           总共有l.w*l.h个网格，且存储时，把所有网格的x,y,w,h,c信息聚到一起再拼接起来，因此xxxxxx及其他信息都有l.w*l.h=6个，*
*           因为每个有l.classes个物体类别，而且也是和xywh一样，每一类都集中存储，先存储l.w*l.h=6个C1类，而后存储6个C2类，    *
*           更为具体的注释可以函数中的语句注释（注意不是C1C2C1C2C1C2C1C2C1C2C1C2的模式，而是将所有的类别拆开分别集中存储）。 *
*           自信度参数c表示的是该矩形框内存在物体的概率，而C1，C2分别表示矩形框内存在物体时属于物体1和物体2的概率，          *
*           因此c*C1即得矩形框内存在物体1的概率，c*C2即得矩形框内存在物体2的概率                                             *
*****************************************************************************************************************************/
void forward_yolo_layer(const layer l, network_state state)
{
    int i,j,b,t,n;
    // memccpy函数原型：void *memcpy(void *dest, const void *src, size_t n);
    // memcpy指的是c和c++使用的内存拷贝函数，memcpy函数的功能是从源src所指的内存地址的起始位置开始拷贝n个字节到目标dest所指的内存地址的起始位置中。
    // 将net.input中的元素全部拷贝至l.output中
    // 这儿state.input是网络上一层的输出,l.output是本层网络的输出,第三个参数是拷贝元素的长度（注意与个数的区别）
    memcpy(l.output, state.input, l.outputs*l.batch*sizeof(float));

    // 这个#ifndef预编译指令没有必要用的，因为forward_yolo_layer()函数本身就对应没有定义gpu版的，所以肯定会执行其中的语句,
    // 其中的语句的作用是为了计算yolo_layer层的输出l.output
    // 这一段总结起来的作用就是对l.output中的xy c和C进行了LOGISTIC激活处理
#ifndef GPU
    // 遍历batch中的每张图片（l.output含有整个batch训练图片对应的输出）
    for (b = 0; b < l.batch; ++b){
        // 注意yolo_layer层中的l.n含义是每个cell grid（网格）中预测的矩形框个数（不是卷积层中卷积核的个数）,也就是每个yolo层anchor得个数
        for(n = 0; n < l.n; ++n){
            // 获取 某一中段首个x的地址（中段的含义参考entry_idnex()函数的注释），此处仅用两层循环处理所有的输入，直观上应该需要四层的，
            // 即还需要两层遍历l.w和l.h（遍历每一个网格），但实际上并不需要，因为每次循环，其都会处理一个中段内某一小段的数据，这一小段数据
            // 就包含所有网格的数据。比如处理第1个中段内所有x和y（分别有l.w*l.h个x和y）.
            int index = entry_index(l, b, n*l.w*l.h, 0);
            // 注意第二个参数是2*l.w*l.h，也就是从l.output的index所引处开始，对之后2*l.w*l.h个元素进行logistic激活函数处理，也就是对
            // 一个中段内所有的x,y进行logistic函数处理，之所以只对x,y做激活，而不对w,h作激活主要与w和h的计算公式有关？
            // activate_array函数：src/activation.c 依次对l.output中从index索引起的2*l.w*l.h个元素进行激活处理，激活函数为LOGISTIC
            activate_array(l.output + index, 2*l.w*l.h, LOGISTIC);

            // 和上面一样，此处是获取一个中段内首个自信度信息c值的地址，而后对该中段内所有的c值（该中段内共有l.w*l.h个c值）进行logistic激活函数处理
            index = entry_index(l, b, n*l.w*l.h, 4);
            activate_array(l.output + index, (1+l.classes)*l.w*l.h, LOGISTIC);
        }
    }
#endif

    // memset函数原型：void *memset(void *s, int ch, size_t n);
    // 将s中当前位置后面的n个字节 （typedef unsigned int size_t ）用 ch 替换并返回 s 。
    // memset：作用是在一段内存块中填充某个给定的值，它是对较大的结构体或数组进行清零操作的一种最快方法
    // 此处是数组初始化：将l.delta中所有元素（共有l.outputs*l.batch个元素，每个元素sizeof(float)个字节）清零
    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
    // 如果不是训练过程，则返回不再执行下面的语句（前向推理即检测过程也会调用这个函数，这时就不需要执行下面训练时才会用到的语句了）
    if(!state.train) return;


    float avg_iou = 0;
    float recall = 0;
    float recall75 = 0;
    float avg_cat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;    // 一张训练图片所有预测矩形框的平均自信度（矩形框中含有物体的概率），该参数没有实际用处，仅用于输出打印
    int count = 0;
    int class_count = 0;
    // 首先将loss置0
    *(l.cost) = 0;

    // 外层循环遍历每张图片
    for (b = 0; b < l.batch; ++b) {
        // 本层循环处理每张图片的所有行
        for (j = 0; j < l.h; ++j) {
            // 处理一张图片的所有列
            for (i = 0; i < l.w; ++i) {
                // 此处的l.n同上面提到的一样，表示每个cell预测的框的个数，对yolov3来说是3个.在这儿依次处理每一个框
                for (n = 0; n < l.n; ++n) {
                    // 得到第b个batch中，行为j,列为i的第n类（总共3类）框的x的索引，这儿可能比较晦涩，我详细解释，不知道对不对？好吧，相信自己
                    // 结合entry_index函数，其中 n=location/(w*h)  (从这起我的l.w和l.h就简写了)，那此时带入公式时得到的n的值就和i相等，也正是第几类
                    // 框的偏移。再看loc,loc=location%(w*h) ,将实参带入公式有：loc=j*w+i,这不正是一张图片二维像素（一个通道）中行为j，列为i的偏移吗？
                    // 将这个loc结合index的公式和l.output的存储模式分析，index=batch*l.outputs + n*l.w*l.h*(4+l.classes+1) + entry*l.w*l.h + loc;
                    // batch*l.outputs + n*l.w*l.h*(4+l.classes+1) + entry*l.w*l.h得到了所有w*h个X的起始索引（因为这儿最后一个参数为0，说明是获得X的起始索引），
                    // 最后再加上loc就刚好是w*h个X中的第loc个，不知道这样有没有讲清楚，前面也提过，要理解entry_index函数中的处理必须结合实参研究
                    int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                    // box数据结构定义在include/darknet.h中 其定义为：
                    // typedef struct{
                    //	  float x, y, w, h;
                    // } box;
                    // get_yolo_box函数：本文件 得到一个yolo特征层的第j行，第i列的cell所预测的第n个框的BBox坐标
                    box pred = get_yolo_box(l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.w*l.h);
                    // 最高IoU，赋初值0
                    float best_iou = 0;
                    int best_t = 0;
                    // l.max_boxes值为90，查看darknet.h中关于max_boxes变量的注释就知道了，每张图片最多能够有90真实个框;
                    // 还要说明的一点是，此处所说最多处理90个矩形框，是指真实值中，一张图片含有的最大真实物体标签数，
                    // 也即真实的物体矩形框个数最大为90,并不是模型预测限制最多就90个，如上注释，一个图片如果分成7*7个网格，
                    // 每个网格预测两个矩形框，那就有90个了，所以不是指模型只能预测90个。模型你可以尽管预测多的矩形框，
                    // 只是我默认一张图片中最多就打了90个物体的标签，所以之后会有滤除过程。滤除过程有两层，首先第一层就是
                    // 下面的for循环了，
                    // 执行完这层循环后，有一件事可以肯定：best_iou中记录的是一个cell的1个anchor中预测的框与所有Ground truth之间最大的IOU，best_t记录的
                    // 是这个Ground truth的编号（索引）
                    for(t = 0; t < l.max_boxes; ++t){
                        // 通过移位来获取每一个真实矩形框的信息，net.truth存储了网络吞入的所有图片的真实矩形框信息（一次吞入一个batch的训练图片），
                        // net.truth作为这一个大数组的首地址，l.truths参数是每一张图片含有的真实值参数个数（可参考darknet.h中的truths参数中的注释），
                        // b是batch中已经处理完图片的图片的张数，5是每个真实矩形框需要5个参数值（也即每条矩形框真值有5个参数），t是本张图片已经处理
                        // 过的矩形框的个数（每张图片最多处理90个框），明白了上面的参数之后对于下面的移位获取对应矩形框真实值的代码就不难了。
                        // float_to_box_stride函数从net.truth中取出标签的信息 b*l.truth为图片的偏移，t*(4+1)为框的偏移
                        box truth = float_to_box_stride(state.truth + t*(4 + 1) + b*l.truths, 1);
                        // 获取真实矩形框对应的类别信息
                        int class_id = state.truth[t*(4 + 1) + b*l.truths + 4];
                        // 简单的验错
                        if (class_id >= l.classes) {
                            printf(" Warning: in txt-labels class_id=%d >= classes=%d in cfg-file. In txt-labels class_id should be [from 0 to %d] \n", class_id, l.classes, l.classes - 1);
                            getchar();
                            continue; // if label contains class_id more than number of classes in the cfg-file
                        }
                        // 这个if语句是用来判断一下是否有读到真实矩形框值（每个矩形框有5个参数,float_to_box_stride只读取其中的4个定位参数，
                        // 只要验证x的值不为0,那肯定是4个参数值都读取到了，要么全部读取到了，要么一个也没有），另外，因为程序中写死了每张图片处理90个矩形框，
                        // 那么有些图片没有这么多矩形框，就会出现没有读到的情况，此时的x为0，就会跳过处理。
                        if(!truth.x) break;  // continue;
                        // box_iou函数：src/box.c 获取完真实标签矩形定位坐标后，与模型检测出的矩形框求IoU，具体参考box_iou()函数注释
                        float iou = box_iou(pred, truth);
                        // 找出最大的IoU值和对应框在标签中的索引
                        if (iou > best_iou) {
                            best_iou = iou;
                            best_t = t;
                        }
                    }
                    // 获取当前遍历cell含有物体的自信度信息c（该矩形框中的确存在物体的概率）在l.output中的索引值
                    int obj_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4);
                    // 叠加每个预测矩形框的自信度c（也即每个矩形框中含有物体的概率）
                    avg_anyobj += l.output[obj_index];

                    // 计算梯度，delta中保存的就是本层的梯度值
                    // 如果大于ignore_thresh, 那么忽略，意思是检测正确了，其置信度的梯度为0
                    // 如果小于ignore_thresh，target = 0
                    // diff = -gradient = target - output，其中gradient=output-target
                    // 为什么是上式，参考：https://xmfbit.github.io/2018/04/01/paper-yolov3/
                    l.delta[obj_index] = 0 - l.output[obj_index];
                    // 上面90次循环使得本矩形框已经与训练图片中所有90个（90个只是最大值，可能没有这么多）真实矩形标签进行了对比，只要在这90个中
                    // 找到一个真实矩形标签与该预测矩形框的iou大于指定的阈值，则判定该框检测正确了，则置信度的梯度为0
                    if (best_iou > l.ignore_thresh) {
                        l.delta[obj_index] = 0;
                    }
                    // best_iou不会大于1 ，所以此段代码如果不调整l.truth_thresh的值的话是永不会执行的，现假设该值为0.9，给它一个执行的机会，进去跟踪具体操作
                    // 此处是模仿双IOU阈值？
                    if (best_iou > l.truth_thresh) {
                        // 此时，best_iou满足大于0.9，说明该检测框与某个真实框之间的重叠度很高很高，说明这个框极有可能是目标而不是对象
                        // 则此处的自信度信息的损失应该很小，但感觉也不至于像前面那样置为0，所以按公式target-output 计算，即此时target当作1
                        l.delta[obj_index] = 1 - l.output[obj_index];

                        int class_id = state.truth[best_t*(4 + 1) + b*l.truths + 4];

                        if (l.map) class_id = l.map[class_id];
                        // 找到类别信息的起始存储位置
                        int class_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4 + 1);
                        // 计算类别损失函数，和置信度损失类似，只是此处要为多个位置赋值
                        delta_yolo_class(l.output, l.delta, class_index, class_id, l.classes, l.w*l.h, 0, l.focal_loss);
                        // 获得与最高IOU的标签框的信息，用于计算位置损失
                        box truth = float_to_box_stride(state.truth + best_t*(4 + 1) + b*l.truths, 1);
                        // 计算位置损失，为何scale为（2-truth.w*truth.h）
                        delta_yolo_box(truth, l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.delta, (2-truth.w*truth.h), l.w*l.h);
                    }
                }
            }
        }

        // 前面循环是针对每一个预测框计算了损失，本轮训话针对每个标签框计算损失
        for(t = 0; t < l.max_boxes; ++t){
            // truth中保存了标签框的位置信息
            box truth = float_to_box_stride(state.truth + t*(4 + 1) + b*l.truths, 1);
            // 得到标签框的类别信息
            int class_id = state.truth[t*(4 + 1) + b*l.truths + 4];
            // 类别错误的时候跳过处理
            if (class_id >= l.classes) continue; // if label contains class_id more than number of classes in the cfg-file
            // 注释同上
            if(!truth.x) break;  // continue;
            float best_iou = 0;
            int best_n = 0;
            // i的值为标签框的w在特征图上的绝对值，j为h的绝对值
            i = (truth.x * l.w);
            j = (truth.y * l.h);
            box truth_shift = truth;

            truth_shift.x = truth_shift.y = 0;
            // 这个循环貌似在找出与真实框形状最接近的anchor
            for(n = 0; n < l.total; ++n){
                box pred = {0};
                // 取出对应anchor的w和h
                pred.w = l.biases[2*n]/ state.net.w;
                pred.h = l.biases[2*n+1]/ state.net.h;
                float iou = box_iou(pred, truth_shift);
                if (iou > best_iou){
                    best_iou = iou;
                    best_n = n;
                }
            }
            // int_index函数：util.c 找出整型数组a中的元素val所在的位置，返回其索引值,在这返回的是
            // 在该yolo层中，上面找到的与真实框形状最接近的anchor的索引,这个是相对索引，最大值为l.n-1
            int mask_n = int_index(l.mask, best_n, l.n);
            // 如果对应anchor存在，则mask_n>=0
            if(mask_n >= 0){
                //  找到对应anchor所输出的值的x的起始坐标
                int box_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 0);
                // 计算anchor和标签框之间的梯度
                float iou = delta_yolo_box(truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.delta, (2-truth.w*truth.h), l.w*l.h);
                // 得到类别置信度
                int obj_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4);
                avg_obj += l.output[obj_index];
                l.delta[obj_index] = 1 - l.output[obj_index];

                int class_id = state.truth[t*(4 + 1) + b*l.truths + 4];
                if (l.map) class_id = l.map[class_id];
                int class_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4 + 1);
                delta_yolo_class(l.output, l.delta, class_index, class_id, l.classes, l.w*l.h, &avg_cat, l.focal_loss);

                ++count;
                ++class_count;
                if(iou > .5) recall += 1;
                if(iou > .75) recall75 += 1;
                avg_iou += iou;
            }
        }
    }
    // mag_array函数：求l.delta种每个元素的平方和，再开方
    *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    printf("Region %d Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f,  count: %d\n", state.index, avg_iou/count, avg_cat/class_count, avg_obj/count, avg_anyobj/(l.w*l.h*l.n*l.batch), recall/count, recall75/count, count);
}

void backward_yolo_layer(const layer l, network_state state)
{
   axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, state.delta, 1);
}

/*****************************************************************************************
*  func: 将以网络输入的width和height为size进行预测的边框信息x，y,w,h转化到以resize过后   *
*        的letter_box的中间图的width和height的size为基准的边框信息，因为预测值x,y,w,h    *
*        均相对于网络输入的width和height进行了归一化，而且letter_box中图片的width和height*
*        相对于网络输入上下（w>h）或左右（w<h）有像素值为0.5的补边，需要进行转化         *
*  args dets：需要转化的边框列表                                                         *
*  args n: 框的总个数                                                                    *
*  args w/h: 原始图片的width和height                                                     *
*  args netw/neth: 网络输入的width和height                                               *
*  args relative: 指示边框信息中的x,y,w,h是否是相对值，即归一化后的值，默认为1           *
*****************************************************************************************/
void correct_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative, int letter)
{
    int i;
    // 此处new_w表示输入图片经压缩后在网络输入大小的letter_box中的width,new_h表示在leitter_box中的
    // height,以1280*720的输入图片为例，在进行letter_box的过程中，因为w>h（1280>720），所以原图经
    // resize后的width以网络的输入width为标准，这里假设为416，那么resize后的对应height为720*416/1280
    // 所以height为234，而超过234的上下空余部分在作为网络输入之前填充了浮点值0.5，这儿一定要明白new_w
    // 和new_h的实际意义   当然，这儿得看指示参数letter指示是否进行了嵌入
    int new_w=0;
    int new_h=0;
    if (letter) {
        // 如果w>h说明resize的时候是以width/图像的width为resize比例的，先得到中间图的width,再根据比例得到height
        if (((float)netw / w) < ((float)neth / h)) {
            new_w = netw;
            new_h = (h * netw) / w;
        }
        else {
            new_h = neth;
            new_w = (w * neth) / h;
        }
    }
    else {
        // 如果指示没有嵌入，则为网络输入得width和 height，可能这是AB得作者考虑到不嵌入对系统也没什么影响得原因吧
        new_w = netw;
        new_h = neth;
    }
    for (i = 0; i < n; ++i){
        box b = dets[i].bbox;
        // 此处的公式很不好理解还是接着上面的例子，现有new_w=416,new_h=234,因为resize是以w为长边压缩的
        // 所以x相对于width的比例不变，而b.y表示y相对于图像高度的比例，在进行这一步的转化之前，b.y表示
        // 的是预测框的y坐标相对于网络height的比值，要转化到相对于letter_box中图像的height的比值时，需要先
        // 计算出y在letter_box中的相对坐标，即(b.y - (neth - new_h)/2./neth)，再除以比例
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw);
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth);
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;
        // 如果标签框得值不是相对值，就需要转化为绝对值
        if(!relative){
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}

/******************************************************************
*  func: 统计yolo层输出框中目标的confidence值比thresh大的框个数   *
*  args l: yolo层的指针                                           *
*  args thresh：阈值                                              *
******************************************************************/
int yolo_num_detections(layer l, float thresh)
{
    int i, n;
    int count = 0;
    // 此处l.w和l.h表示该层输入的维度，与该层输出的维度一致
    for (i = 0; i < l.w*l.h; ++i){
        // l.n表示该层每个cell预测多少个框，对于yolov3来说为3
        for(n = 0; n < l.n; ++n){
            // entry_index函数：本文件 定位目标confidence的起始索引
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
            if(l.output[obj_index] > thresh){
                // 当目标的confidence值比thresh高时，则计数器加1
                ++count;
            }
        }
    }
    return count;
}

void avg_flipped_yolo(layer l)
{
    int i,j,n,z;
    float *flip = l.output + l.outputs;
    for (j = 0; j < l.h; ++j) {
        for (i = 0; i < l.w/2; ++i) {
            for (n = 0; n < l.n; ++n) {
                for(z = 0; z < l.classes + 4 + 1; ++z){
                    int i1 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + i;
                    int i2 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + (l.w - i - 1);
                    float swap = flip[i1];
                    flip[i1] = flip[i2];
                    flip[i2] = swap;
                    if(z == 0){
                        flip[i1] = -flip[i1];
                        flip[i2] = -flip[i2];
                    }
                }
            }
        }
    }
    for(i = 0; i < l.outputs; ++i){
        l.output[i] = (l.output[i] + flip[i])/2.;
    }
}

/******************************************************************************************
*  func: 获取预测框的confidence=objectness,边框信息和类别概率，类别概率需要乘以abjectness *
*        随后调用函数correct_yolo_boxes对边框信息进行纠正，详细注释见定义处               *
*  args l：待处理的yolo层，其output包含了预测框和类别概率等信息                           *
*  args w: 原始图片的width                                                                *
*  args h: 原始图片的height                                                               *
*  args netw/neth: 网络输入的width和height                                                *
*  args thresh: 姑且当作是否为有效框的阈值                                                *
*  args map: yolov3中没有使用                                                             *
*  args relative: 指示边框信息中的x,y,w,h是否是相对值，即归一化后的值，默认为1            *
*  args dets: 预测框的指针，引用传递,此处处理后的框通过dets传出                           *
******************************************************************************************/
int get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets, int letter)
{
    //printf("\n l.batch = %d, l.w = %d, l.h = %d, l.n = %d \n", l.batch, l.w, l.h, l.n);
    int i,j,n;
    float *predictions = l.output;
    // 测试得时候batch都设置为1，难道可以设置成2会有奇效？
    if (l.batch == 2) avg_flipped_yolo(l);
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i){
        int row = i / l.w;
        int col = i % l.w;
        for(n = 0; n < l.n; ++n){
            // entry_index函数：本文件 找到预测框的confidence的起始索引
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
            // 获取置信度
            float objectness = predictions[obj_index];
            //if(objectness <= thresh) continue;    // incorrect behavior for Nan values
            if (objectness > thresh) {
                //printf("\n objectness = %f, thresh = %f, i = %d, n = %d \n", objectness, thresh, i, n);
                // 得到每个预测框的BBox信息的起始索引
                int box_index = entry_index(l, 0, n*l.w*l.h + i, 0);
                // get_yolo_box函数：本文件 获取某个矩形框的4个定位信息（根据输入的矩形框索引从l.output中获取该矩形框的定位信息x,y,w,h）
                dets[count].bbox = get_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw, neth, l.w*l.h);
                dets[count].objectness = objectness;
                dets[count].classes = l.classes;
                for (j = 0; j < l.classes; ++j) {
                    // 此处这么获取每个类别的分类得分是不是不太高效？
                    int class_index = entry_index(l, 0, n*l.w*l.h + i, 4 + 1 + j);
                    // prob验证论文公式中的confidence*类别得分
                    float prob = objectness*predictions[class_index];
                    // 最终赋值给dets的prob还需要和阈值进行比较
                    dets[count].prob[j] = (prob > thresh) ? prob : 0;
                }
                ++count;
            }
        }
    }
    // correct_yolo_boxes函数：本文件 将dets中的边框信息进行转化，详情见函数定义处
    // 转化过后，dets中保留了所有物体类别概率大于thresh，且其边界框坐标已经转化到相对于其输入得图片坐标值
    correct_yolo_boxes(dets, count, w, h, netw, neth, relative, letter);
    return count;
}

#ifdef GPU

void forward_yolo_layer_gpu(const layer l, network_state state)
{
    //copy_ongpu(l.batch*l.inputs, state.input, 1, l.output_gpu, 1);
    simple_copy_ongpu(l.batch*l.inputs, state.input, l.output_gpu);
    int b, n;
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            // y = 1./(1. + exp(-x))
            // x = ln(y/(1-y))  // ln - natural logarithm (base = e)
            // if(y->1) x -> inf
            // if(y->0) x -> -inf
            activate_array_ongpu(l.output_gpu + index, 2*l.w*l.h, LOGISTIC);    // x,y
            index = entry_index(l, b, n*l.w*l.h, 4);
            activate_array_ongpu(l.output_gpu + index, (1+l.classes)*l.w*l.h, LOGISTIC); // classes and objectness
        }
    }
    if(!state.train || l.onlyforward){
        //cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
        cuda_pull_array_async(l.output_gpu, l.output, l.batch*l.outputs);
        CHECK_CUDA(cudaPeekAtLastError());
        return;
    }

    float *in_cpu = (float *)calloc(l.batch*l.inputs, sizeof(float));
    cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
    memcpy(in_cpu, l.output, l.batch*l.outputs*sizeof(float));
    float *truth_cpu = 0;
    if (state.truth) {
        int num_truth = l.batch*l.truths;
        truth_cpu = (float *)calloc(num_truth, sizeof(float));
        cuda_pull_array(state.truth, truth_cpu, num_truth);
    }
    network_state cpu_state = state;
    cpu_state.net = state.net;
    cpu_state.index = state.index;
    cpu_state.train = state.train;
    cpu_state.truth = truth_cpu;
    cpu_state.input = in_cpu;
    forward_yolo_layer(l, cpu_state);
    //forward_yolo_layer(l, state);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
    free(in_cpu);
    if (cpu_state.truth) free(cpu_state.truth);
}

void backward_yolo_layer_gpu(const layer l, network_state state)
{
    axpy_ongpu(l.batch*l.inputs, 1, l.delta_gpu, 1, state.delta, 1);
}
#endif
