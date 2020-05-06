#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "demo.h"
#include <sys/time.h>

#define DEMO 1

#ifdef OPENCV

static char **demo_names;
static image **demo_alphabet;
static int demo_classes;

static network *net;
static image buff [3];
static image buff_letter[3];
static int buff_index = 0;
static CvCapture * cap;
static IplImage  * ipl;
static float fps = 0;
static float demo_thresh = 0;
static float demo_hier = .5;
static int running = 0;

static int demo_frame = 3;
static int demo_index = 0;
static float **predictions;
static float *avg;
static int demo_done = 0;
static int demo_total = 0;
double demo_time;

detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num);



/***************************************************************************************
*  func: 统计网络中输出层所占总空间的大小，此处还是以Yolov3为例，采用yolov3.cfg        *
*  args net：使用的网络，已经初始化各层的参数                                          *
*  details: 在yolov3中，net->n表示网络一共有多少层，循环内部首先判别net中每个网络层    *
*           的类别，当类别满足要求时（yolov3中为yolo层，V2中为region层），则累加该层   *
*           的输出维度，其中l.outputs表示了一张图片的总的输出维度，大小为：            *
*           w*h*n*(classes+5)，其中n为该层对应anchor数目，yolov3中为3个，5表示一个     *
*           bbox所对应的5个信息，x,y,w,h和confidene                                    *
***************************************************************************************/
int size_network(network *net)
{
    int i;
    int count = 0;
	// 遍历网络的每一层
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
		// 在yolov3的配置文件中，yolo层作为输出层，没有region和detection层，其中yolo层的
		// outputs参数表示输出数据量，output存储了具体输出数据（预测框信息）
        if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
			// yolov3中一共有3个yolo层，所以这个cout应该为3个数的和，更为具体点就是：
			// count=13*13*3*85+26*26*3*85+52*52*3*85
            count += l.outputs;
        }
    }
    return count;
}



/****************************************************************************
*  func：获取网络net中所有yolo层或region或detection层的输出信息，获取的信息 *
*        放入二维数组predictions中                                          *
*  args net: 完成了一次前向后的网络                                         * 
****************************************************************************/
void remember_network(network *net)
{
    int i;
    int count = 0;
	// 遍历网络net的每一层
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
		// yolov3中为yolo层，V2中为region层
        if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
			// memcpy函数原型：void *memcpy(void *dest, const void *src, size_t n);
			// memcpy指的是c和c++使用的内存拷贝函数，memcpy函数的功能是从源src所指
			// 的内存地址的起始位置开始拷贝n个字节到目标dest所指的内存地址的起始位置中。
			// demo_index的初始值为0，在detect_in_thread函数中维护其值
            memcpy(predictions[demo_index] + count, net->layers[i].output, sizeof(float) * l.outputs);
            count += l.outputs;
        }
    }
}



/****************************************************************************
*  func: 将本帧的预测框信息和前面若干帧的框信息进行加权平均以防止视频中框   *
*        的抖动，默认与前面2帧进行平均                                      *
*  args net: 完成了前向过程的网络，包含预测信息                             *
*  args nboxes：有效框个数指针，引用传递                                    *
****************************************************************************/
detection *avg_predictions(network *net, int *nboxes)
{
    int i, j;
    int count = 0;
	// fill_cpu函数：src/blas.c 将avg数组中的demo_total个元素初始化为0，初始步长为1
    fill_cpu(demo_total, 0, avg, 1);
	// demo_frame的值为3
    for(j = 0; j < demo_frame; ++j){
		// axpy_cpu函数：src/blas.c 将avg中的元素+= predictions[j]*1/demo_frame
		// 注意，累加之后的值存储在avg之中，并没有改变prediction中的值
        axpy_cpu(demo_total, 1./demo_frame, predictions[j], 1, avg, 1);
    }
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
		// 将平均后的预测值重新赋值给网络yolo层的output
        if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
			// memcpy函数原型：void *memcpy(void *dest, const void *src, size_t n);
			// memcpy指的是c和c++使用的内存拷贝函数，memcpy函数的功能是从源src所指
			// 的内存地址的起始位置开始拷贝n个字节到目标dest所指的内存地址的起始位置中。
            memcpy(l.output, avg + count, sizeof(float) * l.outputs);
            count += l.outputs;
        }
    }
	// get_network_boxes函数：src/network.c 从加权平均处理后的l.output中提取出有效框的集合
	// 其中会对有效框的边框信息从相对于网络size的值转化到相对于原图size的值
    detection *dets = get_network_boxes(net, buff[0].w, buff[0].h, demo_thresh, demo_hier, 0, 1, nboxes);
    return dets;
}



/*****************************************************************
*  func: 调用remember函数将3层yolo层的输出信息汇集到predictions  *
*        的一行，再调用avg_predictions函数将本帧的预测框与前两   *
*        帧的预测信息进行加权平均，并从中获得有效框信息，最后调  *
*        用do_nms_obj函数对框进行非极大值抑制处理，然后输出结果  *
*  details: 本函数为处理检测的函数，作为线程函数被调用           *
*****************************************************************/
void *detect_in_thread(void *ptr)
{
	// 此处这个running有任何意义？
    running = 1;
	// nms阈值
    float nms = .4;

    layer l = net->layers[net->n-1];
    float *X = buff_letter[(buff_index+2)%3].data;
	// network_predict函数：src/network.c 以X作为网络的输入数据，跑一遍net网络的前向过程
    network_predict(net, X);
	   
	// remember_network函数：本文件 将net网络中的yolo层的输出信息汇总到predictions中的第demo_index行
    remember_network(net);
	
	// detection 数据结构定义在 include/darknet.h中，具体定义如下：
	/*
		typedef struct detection{
			box bbox;
			int classes;
			float *prob;
			float *mask;
			float objectness;
			int sort_class;
		} detection;
	*/
    detection *dets = 0;
    int nboxes = 0;
	// avg_predictions函数：本文件 将网络处理的这一帧的结果与前面若干帧的结果求平均，能有效防止视频中的框抖动
	// 默认是与前两帧求加权平均，随后调用get_network_boxes函数从平均后的边框信息中筛选有效框，返回有效框集合
    dets = avg_predictions(net, &nboxes);

	// do_nms_obj函数：src/box.c 对有效框进行NMS，保留最后的显示框，其objectness值不为0，则表示该框最终有意义
	// 注意此处的nboxes为值传递，所以在函数内部并不能修改Nboxes的值
    if (nms > 0) do_nms_obj(dets, nboxes, l.classes, nms);

	// printf("\033[ 颜色特效控制，参考https://blog.csdn.net/mao834099514/article/details/52303074
	// 以及 https://blog.csdn.net/wilson1068/article/details/42970551
    printf("\033[2J");             // 清除屏幕
    printf("\033[1;1H");           // 不晓得是在搞啥子
    printf("\nFPS:%.1f\n",fps);    // 输出fps
    printf("Objects:\n\n");
    image display = buff[(buff_index+2) % 3];
	// draw_detections函数：src/image.c 把dets中得所有框和对应标签画到display上
    draw_detections(display, dets, nboxes, demo_thresh, demo_names, demo_alphabet, demo_classes);
	// free_detections函数：src/network.c 释放检测框集合
    free_detections(dets, nboxes);

    demo_index = (demo_index + 1)%demo_frame;
    running = 0;
    return 0;
}



/***************************************************************
*  func: 获取数据的线程,从cap中获取一帧视频图像放到指定位置    *
*        同时完成图像的赋值和嵌入等工作                        *
***************************************************************/
void *fetch_in_thread(void *ptr)
{
	// fill_image_from_stream函数：src/image.c 从cap中获取一帧图片后写入buff[buff_index]的data域 返回读写状态 成功为1，否则为0
	// 注意，此处读入的帧图片数据会处理好格式后写入buff[buff_index]的data域
    int status = fill_image_from_stream(cap, buff[buff_index]);
	// letterbox_image_into函数：src/image.c 将buff[buff_index]图像的内容resize到中间图再嵌入到buff_letter[buff_index]中
	// 其中buff[buff_index]的图像并未改变，buff_letter[buff_index]保留了嵌入结果
    letterbox_image_into(buff[buff_index], net->w, net->h, buff_letter[buff_index]);
	// status=0 表示没读取到内容 说明已经处理完毕
    if(status == 0) demo_done = 1;
    return 0;
}

void *display_in_thread(void *ptr)
{
    show_image_cv(buff[(buff_index + 1)%3], "Demo", ipl);
    int c = cvWaitKey(1);
    if (c != -1) c = c%256;
    if (c == 27) {
        demo_done = 1;
        return 0;
    } else if (c == 82) {
        demo_thresh += .02;
    } else if (c == 84) {
        demo_thresh -= .02;
        if(demo_thresh <= .02) demo_thresh = .02;
    } else if (c == 83) {
        demo_hier += .02;
    } else if (c == 81) {
        demo_hier -= .02;
        if(demo_hier <= .0) demo_hier = .0;
    }
    return 0;
}

void *display_loop(void *ptr)
{
    while(1){
        display_in_thread(0);
    }
}

void *detect_loop(void *ptr)
{
    while(1){
        detect_in_thread(0);
    }
}



/****************************************************
*  func: 处理视频或者接受摄像头输入的demo           *
*  args cfgfile: 网络配置文件，例如cfg/yolov3.cfg   *
*  args weightfile: 权重文件路径，如yolov3.weights  *
*  args thresh: confidence阈值 例如0.5              *
*  args cam_index: 摄像头编号 例如0/1.              *
*  args filename: 视频文件路径和名 如 ./test.mp4    *
*  args names: 类别标签列表，一行为一个标签名       *
*  args classes: 类别数目，coco为80个类别           *
*  args delay: 本函数并未使用该参数                 *
*  args prefix：意义有待补充                        *
*  args avg_frames: 本函数并未使用该参数            *
*  args hier: 意义有待补充，yolov3中并未使用        *
*  args w: 视频流的帧宽度（只对摄像头有效）         *
*  args h: 视频流的帧高度（只对摄像头有效）         *
*  args frames: 设置摄像头的帧率                    *
*  args fullscreen：是否全屏显示demo                *
*  details: 其实这些参数的实际意义大家只要用一个    *
*           测试视频跑一下便知道了                  *
****************************************************/
void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen)
{
	// 1、image类型定义：include/darknet.h 结构体，包含：w,h,c和float *data
	// 2、load_alphabet函数：src/image.c  加载data/labels/文件夹中所有的字符标签图片
	// 现在alphabet中的每一个元素均为image的结构体对象
    image **alphabet = load_alphabet();
	
	// 将局部变量的值赋值给全局变量，以便在其它函数内部使用
    demo_names = names;
    demo_alphabet = alphabet;
    demo_classes = classes;
    demo_thresh = thresh;
    demo_hier = hier;
    printf("Demo\n");
	
	// load_network函数：src/network.c 将配置文件cfg中的网络加载并初始化权重weightfile所指权重文件，详细注释见该文档，十分复杂！
    net = load_network(cfgfile, weightfile, 0);
	// set_batch_network函数：src/network.c 设置网络的batchsize大小为1，可以尝试把batch设置成非1的正整数，修改相应处理代码
    set_batch_network(net, 1);
	// pthread_t声明两个线程,一个获取图像帧，一个处理图像
    pthread_t detect_thread;
    pthread_t fetch_thread;

	// 作者很开心使用随机种子，关键还很2
    srand(2222222);

    int i;
	// size_network函数：本文件 统计网络中输出层所占总空间的大小，yolov3中一共有3个输出层
    demo_total = size_network(net);
	// predictions为全局float型双重指针，此处demo_frame的值为3，此处的calloc其实是开辟了一个二维数组的3行
    predictions = calloc(demo_frame, sizeof(float*));
    for (i = 0; i < demo_frame; ++i){
		// 为二维数组的每列动态开辟空间，一共有demo_total列
        predictions[i] = calloc(demo_total, sizeof(float));
    }
	// 
    avg = calloc(demo_total, sizeof(float));

    if(filename){
        printf("video file: %s\n", filename);
		// CvCapture类型的cap 是一个结构体，用来保存图像捕获所需要的信息。
	    // opencv提供两种方式从外部捕获图像，一种是从摄像头中，一种
	    // 是通过解码视频得到图像。两种方式都必须从第一帧开始一帧一帧
	    // 的按顺序获取，因此每获取一帧后都要保存相应的状态和参数。
	    // 比如从视频文件中获取，需要保存视频文件的文件名，相应的******
	    // 类型，下一次如果要获取将需要解码哪一帧等。 这些信息都保存在
	    // CvCapture结构中，每获取一帧后，这些信息都将被更新，获取下一帧
	    // 需要将新信息传给获取的api接口
	
	    // 初始化一个视频捕获操作。
	    // 告诉底层的捕获api我想从 Capture1.avi中捕获图片，
	    // 底层api将检测并选择相应的******并做好准备工作
        cap = cvCaptureFromFile(filename);
    }else{
		/*
			CvCaptureFromCam是一个函数。OpenCV中一个函数。初始化从摄像头中获取视频
			CvCapture*cvCaptureFromCAM( int index );
			index要使用的摄像头索引。如果只有一个摄像头或者用哪个摄像头也无所谓，那使用参数-1应该便可以。
			一般index=0
		*/
        cap = cvCaptureFromCAM(cam_index);
        /*
			cvSetCaptureProperty
			设置视频获取属性
			int cvSetCaptureProperty( CvCapture* capture, int property_id, double value );
			注意此方法定位并不准确。
			capture 视频获取结构。
			property_id 属性标识符。可以是下面之一：
			CV_CAP_PROP_POS_MSEC - 从文件开始的位置，单位为毫秒
			CV_CAP_PROP_POS_FRAMES - 单位为帧数的位置（只对视频文件有效）
			CV_CAP_PROP_POS_AVI_RATIO - 视频文件的相对位置（0 - 影片的开始，1 - 影片的结尾)
			CV_CAP_PROP_FRAME_WIDTH - 视频流的帧宽度（只对摄像头有效）
			CV_CAP_PROP_FRAME_HEIGHT - 视频流的帧高度（只对摄像头有效）
			CV_CAP_PROP_FPS - 帧率（只对摄像头有效）
			CV_CAP_PROP_FOURCC - 表示codec的四个字符（只对摄像头有效）
			value 属性的值。
			函数cvSetCaptureProperty设置指定视频获取的属性。目前这个函数对视频文件只支持： CV_CAP_PROP_POS_MSEC, CV_CAP_PROP_POS_FRAMES, CV_CAP_PROP_POS_AVI_RATIO
        */
        if(w){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH, w);
        }
        if(h){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT, h);
        }
        if(frames){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FPS, frames);
        }
    }
    // 如果cap结构体未初始化说明指定文件错误且摄像头也未初始化成功
    if(!cap) error("Couldn't connect to webcam.\n");

	// get_image_from_stream函数：src/image.c 从cap中获取一帧图片后转化为Image类型后返回
    buff[0] = get_image_from_stream(cap);
	// copy_image函数：src/image.c 将buff[0]的内容拷贝至buff[1]，此处是深拷贝
    buff[1] = copy_image(buff[0]);
    buff[2] = copy_image(buff[0]);
	
	// letterbox_image函数：src/image.c 按照神经网络能够接受处理的图片尺寸对输入图片进行尺寸调整
	//（主要包括插值缩放与嵌入两个步骤）
    buff_letter[0] = letterbox_image(buff[0], net->w, net->h);
    buff_letter[1] = letterbox_image(buff[0], net->w, net->h);
    buff_letter[2] = letterbox_image(buff[0], net->w, net->h);
	
	// cvCreateImage函数原型：IplImage* cvCreateImage( CvSize size, int depth, int channels );
	// 创建首地址并分配存储空间
	/*
		Cvsize 图像宽、高.
		depth 图像元素的位深度，可以是下面的其中之一：
		IPL_DEPTH_8U - 无符号8位整型
		IPL_DEPTH_8S - 有符号8位整型
		IPL_DEPTH_16U - 无符号16位整型
		IPL_DEPTH_16S - 有符号16位整型
		IPL_DEPTH_32S - 有符号32位整型
		IPL_DEPTH_32F - 单精度浮点数
		IPL_DEPTH_64F - 双精度浮点数
		channels 每个元素（像素）通道数.可以是 1, 2, 3 或 4.通道是交叉存取的，例如通常的彩色图
		像数据排列是：b0 g0 r0 b1 g1 r1 ... 虽然通常 IPL 图象格式可以存贮非交叉存取的图像，并且
		一些OpenCV 也能处理他, 但是这个函数只能创建交叉存取图像.
	*/
    ipl = cvCreateImage(cvSize(buff[0].w,buff[0].h), IPL_DEPTH_8U, buff[0].c);

	// 设置CvWindows的若干属性，意义很好理解，仿用就好
    int count = 0;
	// 设置了prefix的值后，就不会正常显示demo窗口
    if(!prefix){
        cvNamedWindow("Demo", CV_WINDOW_NORMAL); 
        if(fullscreen){
            cvSetWindowProperty("Demo", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
        } else {
            cvMoveWindow("Demo", 0, 0);
            cvResizeWindow("Demo", 1352, 1013);
        }
    }

    demo_time = what_time_is_it_now();

	// demo_done的初始值为0，标识是否处理结束 
    while(!demo_done){
		// buff_index为全局变量，初始值为0
        buff_index = (buff_index + 1) %3;
		// 创建相应线程，并绑定load_threads()函数到该线程上，第二参数是线程的属性，这里设置为0（即NULL）,第四个参数ptr就是load_threads()的输入参数
		/*
			函数声明:
				int pthread_create(pthread_t *restrict tidp,const pthread_attr_t *restrict_attr,void*（*start_rtn)(void*),void *restrict arg);
			返回值:
	　　		若成功则返回0，否则返回出错编号
			参数:
	　　		第一个参数为指向线程标识符的指针。
	　　		第二个参数用来设置线程属性。
	　　		第三个参数是线程运行函数的地址。
	　　		最后一个参数是运行函数的参数。
			注意:
	　　		在编译时注意加上-lpthread参数，以调用静态链接库。因为pthread并非Linux系统的默认库。 
		*/
		// fetch_in_thread函数：本文件 获取下一帧图像并resize等操作
        if(pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
		// detect_in_thread函数：本文件 
        if(pthread_create(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");
        if(!prefix){
            fps = 1./(what_time_is_it_now() - demo_time);
            demo_time = what_time_is_it_now();
            display_in_thread(0);
        }else{
            char name[256];
            sprintf(name, "%s_%08d", prefix, count);
            save_image(buff[(buff_index + 1)%3], name);
        }
        pthread_join(fetch_thread, 0);
        pthread_join(detect_thread, 0);
        ++count;
    }
}

/*
   void demo_compare(char *cfg1, char *weight1, char *cfg2, char *weight2, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen)
   {
   demo_frame = avg_frames;
   predictions = calloc(demo_frame, sizeof(float*));
   image **alphabet = load_alphabet();
   demo_names = names;
   demo_alphabet = alphabet;
   demo_classes = classes;
   demo_thresh = thresh;
   demo_hier = hier;
   printf("Demo\n");
   net = load_network(cfg1, weight1, 0);
   set_batch_network(net, 1);
   pthread_t detect_thread;
   pthread_t fetch_thread;

   srand(2222222);

   if(filename){
   printf("video file: %s\n", filename);
   cap = cvCaptureFromFile(filename);
   }else{
   cap = cvCaptureFromCAM(cam_index);

   if(w){
   cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH, w);
   }
   if(h){
   cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT, h);
   }
   if(frames){
   cvSetCaptureProperty(cap, CV_CAP_PROP_FPS, frames);
   }
   }

   if(!cap) error("Couldn't connect to webcam.\n");

   layer l = net->layers[net->n-1];
   demo_detections = l.n*l.w*l.h;
   int j;

   avg = (float *) calloc(l.outputs, sizeof(float));
   for(j = 0; j < demo_frame; ++j) predictions[j] = (float *) calloc(l.outputs, sizeof(float));

   boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
   probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
   for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes+1, sizeof(float));

   buff[0] = get_image_from_stream(cap);
   buff[1] = copy_image(buff[0]);
   buff[2] = copy_image(buff[0]);
   buff_letter[0] = letterbox_image(buff[0], net->w, net->h);
   buff_letter[1] = letterbox_image(buff[0], net->w, net->h);
   buff_letter[2] = letterbox_image(buff[0], net->w, net->h);
   ipl = cvCreateImage(cvSize(buff[0].w,buff[0].h), IPL_DEPTH_8U, buff[0].c);

   int count = 0;
   if(!prefix){
   cvNamedWindow("Demo", CV_WINDOW_NORMAL); 
   if(fullscreen){
   cvSetWindowProperty("Demo", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
   } else {
   cvMoveWindow("Demo", 0, 0);
   cvResizeWindow("Demo", 1352, 1013);
   }
   }

   demo_time = what_time_is_it_now();

   while(!demo_done){
buff_index = (buff_index + 1) %3;
if(pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
if(pthread_create(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");
if(!prefix){
    fps = 1./(what_time_is_it_now() - demo_time);
    demo_time = what_time_is_it_now();
    display_in_thread(0);
}else{
    char name[256];
    sprintf(name, "%s_%08d", prefix, count);
    save_image(buff[(buff_index + 1)%3], name);
}
pthread_join(fetch_thread, 0);
pthread_join(detect_thread, 0);
++count;
}
}
*/
#else
void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg, float hier, int w, int h, int frames, int fullscreen)
{
    fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
#endif

