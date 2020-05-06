#include "box.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>



/***********************************************
*  func: 比较两个对象的abjectness值，pa>pb返回 *
*        -1，相等返回0，否则返回1              *
***********************************************/
int nms_comparator(const void *pa, const void *pb)
{
    detection a = *(detection *)pa;
    detection b = *(detection *)pb;
    float diff = 0;
    if(b.sort_class >= 0){
        diff = a.prob[b.sort_class] - b.prob[b.sort_class];
    } else {
        diff = a.objectness - b.objectness;
    }
    if(diff < 0) return 1;
    else if(diff > 0) return -1;
    return 0;
}



/***********************************************************************
*
*  args dets：待处理的边框集合                                         *
*  args total: dets中框的个数                                          *
*  args classes: 类别总数                                              *
*  args thresh: nms的阈值                                              *
***********************************************************************/
void do_nms_obj(detection *dets, int total, int classes, float thresh)
{
    int i, j, k;
    k = total-1;
    for(i = 0; i <= k; ++i){
        if(dets[i].objectness == 0){
			// 如果有框的confidence值为0，则将该框与最后一个框交换，并减少有效框的个数
			// 此处这样处理时不是有点低效率，难道不可以在yolo层的时候判断一下confidence，
			// 为0则不放入到dets中吗？
            detection swap = dets[i];
            dets[i] = dets[k];
            dets[k] = swap;
            --k;
            --i;
        }
    }
	// 跟新Total的值为有效框个数
    total = k+1;

	// 从效率角度讲，这个初始化放在上面循环中一并完成得了，这样做可能是程序得模块化思想
    for(i = 0; i < total; ++i){
        dets[i].sort_class = -1;
    }

	// qsort函数原型：void qsort(void*base,size_t num,size_t width,int(__cdecl*compare)(const void*,const void*));
	// 各参数：1 待排序数组首地址 2 数组中待排序元素数量 3 各元素的占用空间大小 4 指向函数的指针
	// 使用快速排序例程进行排序，头文件：stdlib.h 注意函数得返回值决定排序顺序
	/*
		Compare函数的返回值             描述
		       < 0              elem1将被排在elem2前面
		        0                 elem1 等于 elem2
		       > 0              elem1 将被排在elem2后面
	*/
	// nms_comparator函数：本文件 在这儿结合起来相当于对dets中的元素按照其objectness的值进行了一次降序排序
    qsort(dets, total, sizeof(detection), nms_comparator);
	
	// 下面循环是按照NMS算法流程进行设计的，NMS的流程如下：
	/*
	    先假设有6个候选框，根据分类器类别分类概率（这儿指是否是目标的概率）做排序，从小到大分别属于车辆的概率分别为A、B、C、D、E、F。
		1、从最大概率矩形框F开始，分别判断A~E与F的重叠度IOU是否大于某个设定的阈值;
		2、假设B、D与F的重叠度超过阈值，那么就扔掉B、D；并标记第一个矩形框F，是我们保留下来的。
		3、从剩下的矩形框A、C、E中，选择概率最大的E，然后判断E与A、C的重叠度，重叠度大于一定的阈值，那么就扔掉；并标记E是我们保留下来的第二个矩形框。
		4、一直重复这个过程，找到所有曾经被保留下来的矩形框
	*/
    for(i = 0; i < total; ++i){
		// 这儿还有必要判断objectness的值吗？前面已经去除为0 的了。不，有必要，因为在后面进行NMS的IOU比较后
		// 对于IOU超过阈值的框会将其objecness和每个类别的类别概率都置为0，从而表示该框为重复框
        if(dets[i].objectness == 0) continue;
		// 从objectness最大值的框开始
        box a = dets[i].bbox;
        for(j = i+1; j < total; ++j){
            if(dets[j].objectness == 0) continue;
            box b = dets[j].bbox;
			// 依次和后面的每个框进行IOU计算并决定是否舍弃
			// box_iou函数：本文件  计算框a和框b的IOU
            if (box_iou(a, b) > thresh){
				// 将IOU大于阈值的框的objectness和类别概率都置为0
                dets[j].objectness = 0;
                for(k = 0; k < classes; ++k){
                    dets[j].prob[k] = 0;
                }
            }
        }
    }
}


void do_nms_sort(detection *dets, int total, int classes, float thresh)
{
    int i, j, k;
    k = total-1;
    for(i = 0; i <= k; ++i){
        if(dets[i].objectness == 0){
            detection swap = dets[i];
            dets[i] = dets[k];
            dets[k] = swap;
            --k;
            --i;
        }
    }
    total = k+1;

    for(k = 0; k < classes; ++k){
        for(i = 0; i < total; ++i){
            dets[i].sort_class = k;
        }
        qsort(dets, total, sizeof(detection), nms_comparator);
        for(i = 0; i < total; ++i){
            if(dets[i].prob[k] == 0) continue;
            box a = dets[i].bbox;
            for(j = i+1; j < total; ++j){
                box b = dets[j].bbox;
                if (box_iou(a, b) > thresh){
                    dets[j].prob[k] = 0;
                }
            }
        }
    }
}



/***************************************************************************************************************
*  func: 从存储矩形框信息的大数组中，提取某一个矩形框定位信息并返回（f是这个大数组中某一起始地址）.            *
*  args f：矩形框信息（每个矩形框有5个参数值，此函数仅提取其中4个用于定位的参数x,y,w,h，不包含物体类别编号）   *
*  args stride: 跨度，按倍数跳越取值，此处要结合net.truth的存储结构理解                                        *
***************************************************************************************************************/
box float_to_box(float *f, int stride)
{
	// f中存储每一个矩形框信息的顺序为: x, y, w, h, class_index，这里仅提取前四个，
    // 也即矩形框的定位信息，最后一个物体类别编号信息不在此处提取
    box b = {0};
    b.x = f[0];
    b.y = f[1*stride];
    b.w = f[2*stride];
    b.h = f[3*stride];
    return b;
}

dbox derivative(box a, box b)
{
    dbox d;
    d.dx = 0;
    d.dw = 0;
    float l1 = a.x - a.w/2;
    float l2 = b.x - b.w/2;
    if (l1 > l2){
        d.dx -= 1;
        d.dw += .5;
    }
    float r1 = a.x + a.w/2;
    float r2 = b.x + b.w/2;
    if(r1 < r2){
        d.dx += 1;
        d.dw += .5;
    }
    if (l1 > r2) {
        d.dx = -1;
        d.dw = 0;
    }
    if (r1 < l2){
        d.dx = 1;
        d.dw = 0;
    }

    d.dy = 0;
    d.dh = 0;
    float t1 = a.y - a.h/2;
    float t2 = b.y - b.h/2;
    if (t1 > t2){
        d.dy -= 1;
        d.dh += .5;
    }
    float b1 = a.y + a.h/2;
    float b2 = b.y + b.h/2;
    if(b1 < b2){
        d.dy += 1;
        d.dh += .5;
    }
    if (t1 > b2) {
        d.dy = -1;
        d.dh = 0;
    }
    if (b1 < t2){
        d.dy = 1;
        d.dh = 0;
    }
    return d;
}



/******************************************************************************************************************************
*  func: 计算两个矩形框相交部分矩形的某一边的边长（视调用情况，可能是相交部分矩形的高，也可能是宽）.                          *
*  args x1：第一个矩形框的x坐标（或者y坐标，视调用情况，如果计算的是相交部分矩形的宽，则输入的是x坐标)                        *
*  args w1: 第一个矩形框的宽（而如果要计算相交部分矩形的高，则为y坐标，下面凡是说x坐标的，都可能为y坐标，当然，对应宽变为高)  *
*  args x2：第二个矩形框的x坐标                                                                                               *
*  args w2: 第二个矩形框的宽                                                                                                  *
*  details: 在纸上画一下两个矩形，自己想一下如何计算交集的面积就很清楚下面的代码了：首先计算两个框左边的x坐标，比较大小，     *
*           取其大者，记为left；而后计算两个框右边的x坐标，取其小者，记为right，right-left即得相交部分矩形的宽。返回两个矩形  *
*           框相交部分矩形的宽或者高                                                                                          *
******************************************************************************************************************************/
float overlap(float x1, float w1, float x2, float w2)
{
	// 结合下图进行理解，画得不好，不要嫌弃
	/*
	         l1 *************************
				*             right-left*
				*            |---------|*
				*         l2 ***************************
				*            *          *              *
				*            *          *              *
				*            *          *              *
				************************* r1           *
				             *                         *
							 *                         *
							 *************************** r2
	*/
	// 第一个框的left值（假设传入的参数是x和w，如果是y和h则l1表示第一个框的top,下同，不再赘述）
    float l1 = x1 - w1/2;
	// 第二个框的left值
    float l2 = x2 - w2/2;
	// 相交区域的left当然是取两个left中的较大者
    float left = l1 > l2 ? l1 : l2;
	// 第一个框的right
    float r1 = x1 + w1/2;
	// 第二个框的right
    float r2 = x2 + w2/2;
	// 相交区域的right取两个right中的较小者
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}



/************************************************************************
*  func: 两个矩形框求交：计算两个矩形框a,b相交部分的面积.               *
*  args a: 保存了a框的x,y,w,h信息                                       *
*  args b: 保存了b框的x,y,w,h信息                                       *
*  details: 当两个矩形不相交的时候，返回的值为0（此时计算得到的w,h将小  *
*           于0,w,h是按照上面overlap()函数的方式计算得到的，在纸上比划  *
*           一下就知道为什么会小于0了）                                 *
************************************************************************/
float box_intersection(box a, box b)
{
	// overlap函数：本文件 
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}



/************************************************************************
*  func: 两个矩形框求交：计算两个矩形框a,b相并后的面积.                 *
*  args a: 保存了a框的x,y,w,h信息                                       *
*  args b: 保存了b框的x,y,w,h信息                                       *
************************************************************************/
float box_union(box a, box b)
{
	// a与b相并区域的面积等于a的面积+b的面积-相交部分的面积
    float i = box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}



/**************************************************************************************************************
*  func: 计算box a 与box b 的IoU值.                                                                           *
*  args a: 保存了a框的x,y,w,h信息                                                                             *
*  args b: 保存了b框的x,y,w,h信息                                                                             *
*  details：IoU值，是目标检测精确度的一个评判指标，全称是intersection over union，翻译成中文就是交比并值，    *
*           字面上的意思很直接，就是两个矩形相交部分的面积比两个矩形求并之后的总面积，用来做检测评判指标时，  *
*           含义为模型检测到的矩形框与GroundTruth标记的矩形框之间的交比并值（即可反映检测到的矩形框与         *
*           GroundTruth之间的重叠度），当两个矩形框完全重叠时，值为1；完全不相交时，值为0。                   *
**************************************************************************************************************/
float box_iou(box a, box b)
{
	// box_intersection计算a与b相交部分的面积 box_union计算a与b相并的面积，均在本文件定义
    return box_intersection(a, b)/box_union(a, b);
}

float box_rmse(box a, box b)
{
    return sqrt(pow(a.x-b.x, 2) + 
                pow(a.y-b.y, 2) + 
                pow(a.w-b.w, 2) + 
                pow(a.h-b.h, 2));
}

dbox dintersect(box a, box b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    dbox dover = derivative(a, b);
    dbox di;

    di.dw = dover.dw*h;
    di.dx = dover.dx*h;
    di.dh = dover.dh*w;
    di.dy = dover.dy*w;

    return di;
}

dbox dunion(box a, box b)
{
    dbox du;

    dbox di = dintersect(a, b);
    du.dw = a.h - di.dw;
    du.dh = a.w - di.dh;
    du.dx = -di.dx;
    du.dy = -di.dy;

    return du;
}


void test_dunion()
{
    box a = {0, 0, 1, 1};
    box dxa= {0+.0001, 0, 1, 1};
    box dya= {0, 0+.0001, 1, 1};
    box dwa= {0, 0, 1+.0001, 1};
    box dha= {0, 0, 1, 1+.0001};

    box b = {.5, .5, .2, .2};
    dbox di = dunion(a,b);
    printf("Union: %f %f %f %f\n", di.dx, di.dy, di.dw, di.dh);
    float inter =  box_union(a, b);
    float xinter = box_union(dxa, b);
    float yinter = box_union(dya, b);
    float winter = box_union(dwa, b);
    float hinter = box_union(dha, b);
    xinter = (xinter - inter)/(.0001);
    yinter = (yinter - inter)/(.0001);
    winter = (winter - inter)/(.0001);
    hinter = (hinter - inter)/(.0001);
    printf("Union Manual %f %f %f %f\n", xinter, yinter, winter, hinter);
}
void test_dintersect()
{
    box a = {0, 0, 1, 1};
    box dxa= {0+.0001, 0, 1, 1};
    box dya= {0, 0+.0001, 1, 1};
    box dwa= {0, 0, 1+.0001, 1};
    box dha= {0, 0, 1, 1+.0001};

    box b = {.5, .5, .2, .2};
    dbox di = dintersect(a,b);
    printf("Inter: %f %f %f %f\n", di.dx, di.dy, di.dw, di.dh);
    float inter =  box_intersection(a, b);
    float xinter = box_intersection(dxa, b);
    float yinter = box_intersection(dya, b);
    float winter = box_intersection(dwa, b);
    float hinter = box_intersection(dha, b);
    xinter = (xinter - inter)/(.0001);
    yinter = (yinter - inter)/(.0001);
    winter = (winter - inter)/(.0001);
    hinter = (hinter - inter)/(.0001);
    printf("Inter Manual %f %f %f %f\n", xinter, yinter, winter, hinter);
}

void test_box()
{
    test_dintersect();
    test_dunion();
    box a = {0, 0, 1, 1};
    box dxa= {0+.00001, 0, 1, 1};
    box dya= {0, 0+.00001, 1, 1};
    box dwa= {0, 0, 1+.00001, 1};
    box dha= {0, 0, 1, 1+.00001};

    box b = {.5, 0, .2, .2};

    float iou = box_iou(a,b);
    iou = (1-iou)*(1-iou);
    printf("%f\n", iou);
    dbox d = diou(a, b);
    printf("%f %f %f %f\n", d.dx, d.dy, d.dw, d.dh);

    float xiou = box_iou(dxa, b);
    float yiou = box_iou(dya, b);
    float wiou = box_iou(dwa, b);
    float hiou = box_iou(dha, b);
    xiou = ((1-xiou)*(1-xiou) - iou)/(.00001);
    yiou = ((1-yiou)*(1-yiou) - iou)/(.00001);
    wiou = ((1-wiou)*(1-wiou) - iou)/(.00001);
    hiou = ((1-hiou)*(1-hiou) - iou)/(.00001);
    printf("manual %f %f %f %f\n", xiou, yiou, wiou, hiou);
}

dbox diou(box a, box b)
{
    float u = box_union(a,b);
    float i = box_intersection(a,b);
    dbox di = dintersect(a,b);
    dbox du = dunion(a,b);
    dbox dd = {0,0,0,0};

    if(i <= 0 || 1) {
        dd.dx = b.x - a.x;
        dd.dy = b.y - a.y;
        dd.dw = b.w - a.w;
        dd.dh = b.h - a.h;
        return dd;
    }

    dd.dx = 2*pow((1-(i/u)),1)*(di.dx*u - du.dx*i)/(u*u);
    dd.dy = 2*pow((1-(i/u)),1)*(di.dy*u - du.dy*i)/(u*u);
    dd.dw = 2*pow((1-(i/u)),1)*(di.dw*u - du.dw*i)/(u*u);
    dd.dh = 2*pow((1-(i/u)),1)*(di.dh*u - du.dh*i)/(u*u);
    return dd;
}


void do_nms(box *boxes, float **probs, int total, int classes, float thresh)
{
    int i, j, k;
    for(i = 0; i < total; ++i){
        int any = 0;
        for(k = 0; k < classes; ++k) any = any || (probs[i][k] > 0);
        if(!any) {
            continue;
        }
        for(j = i+1; j < total; ++j){
            if (box_iou(boxes[i], boxes[j]) > thresh){
                for(k = 0; k < classes; ++k){
                    if (probs[i][k] < probs[j][k]) probs[i][k] = 0;
                    else probs[j][k] = 0;
                }
            }
        }
    }
}

box encode_box(box b, box anchor)
{
    box encode;
    encode.x = (b.x - anchor.x) / anchor.w;
    encode.y = (b.y - anchor.y) / anchor.h;
    encode.w = log2(b.w / anchor.w);
    encode.h = log2(b.h / anchor.h);
    return encode;
}

box decode_box(box b, box anchor)
{
    box decode;
    decode.x = b.x * anchor.w + anchor.x;
    decode.y = b.y * anchor.h + anchor.y;
    decode.w = pow(2., b.w) * anchor.w;
    decode.h = pow(2., b.h) * anchor.h;
    return decode;
}
