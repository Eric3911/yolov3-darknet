#include "image.h"
#include "utils.h"
#include "blas.h"
#include "cuda.h"
#include <stdio.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int windows = 0;

float colors[6][3] = { {1,0,1}, {0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0} };



/***************************************************************
*  func: 得到三通道中任一通道的色调值                          *
***************************************************************/
float get_color(int c, int x, int max)
{
	
    float ratio = ((float)x/max)*5;
    int i = floor(ratio);
    int j = ceil(ratio);
    ratio -= i;
    float r = (1-ratio) * colors[i][c] + ratio*colors[j][c];
    //printf("%f\n", r);
    return r;
}

image mask_to_rgb(image mask)
{
    int n = mask.c;
    image im = make_image(mask.w, mask.h, 3);
    int i, j;
    for(j = 0; j < n; ++j){
        int offset = j*123457 % n;
        float red = get_color(2,offset,n);
        float green = get_color(1,offset,n);
        float blue = get_color(0,offset,n);
        for(i = 0; i < im.w*im.h; ++i){
            im.data[i + 0*im.w*im.h] += mask.data[j*im.h*im.w + i]*red;
            im.data[i + 1*im.w*im.h] += mask.data[j*im.h*im.w + i]*green;
            im.data[i + 2*im.w*im.h] += mask.data[j*im.h*im.w + i]*blue;
        }
    }
    return im;
}



/********************************************************
*  func：获取图片m的w=x,h=y,c=c处的像素值               *
*  args m：要获取像素的图片                             *
*  args x：图像的width索引                              *
*  args y：图像的height索引                             *
*  args c：图像的channel索引                            *
*  details：assert宏的原型定义在<assert.h>中，其作用是  *
*           如果它的条件返回错误，则终止程序执行        *
********************************************************/
static float get_pixel(image m, int x, int y, int c)
{
    assert(x < m.w && y < m.h && c < m.c);
    return m.data[c*m.h*m.w + y*m.w + x];
}


/**********************************************************************************
*  func：和get_pixel()一样，只不过多了超出边界坐标的处理，凡是超过图像边界坐标的，*
*        设为边界坐标，但在yolo v3的版本中，已经将超过边界处理的注释起来了，其作  *
*        用就和get_pixel()完全一样了，那么这段代码就可以不必嵌套调用了            *
**********************************************************************************/
static float get_pixel_extend(image m, int x, int y, int c)
{
    if(x < 0 || x >= m.w || y < 0 || y >= m.h) return 0;
    /*
    if(x < 0) x = 0;
    if(x >= m.w) x = m.w-1;
    if(y < 0) y = 0;
    if(y >= m.h) y = m.h-1;
    */
    if(c < 0 || c >= m.c) return 0;
    return get_pixel(m, x, y, c);
}



/********************************************************
*  func：设置图片m的w=x,h=y,c=c处的像素值               *
*  args m：要获取像素的图片                             *
*  args x：图像的width索引                              *
*  args y：图像的height索引                             *
*  args c：图像的channel索引                            *
*  details：assert宏的原型定义在<assert.h>中，其作用是  *
*           如果它的条件返回错误，则终止程序执行        *
********************************************************/
static void set_pixel(image m, int x, int y, int c, float val)
{
    if (x < 0 || y < 0 || c < 0 || x >= m.w || y >= m.h || c >= m.c) return;
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c*m.h*m.w + y*m.w + x] = val;
}



/*************************************************************************************
*  func：将输入图片m的c通道y行x列像素值叠加val（m的data是堆内存，因此虽然m是按值传递,*
*        但是函数内对data的改动在退出add_pixel函数后依然有效，虽然如此，个人感觉形参 *
*        不用按值传递用指针应该更清晰些）                                            *
*  details：m中的像素按行存储（各通道所有行并成一行，然后所有通道再并成一行）        *
*************************************************************************************/
static void add_pixel(image m, int x, int y, int c, float val)
{
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c*m.h*m.w + y*m.w + x] += val;
}


/***********************************************
*  func: 双线性插值得到(x,y)处的像素值         *
*  args im: 输入图像                           *
*  args x: 像素坐标（列），亚像素              *
*  args y: 像素坐标（行），亚像素              *
*  args c: 所在通道数                          *
***********************************************/
static float bilinear_interpolate(image im, float x, float y, int c)
{
	// 往下取整得到整数坐标
    int ix = (int) floorf(x);
    int iy = (int) floorf(y);

    float dx = x - ix;
    float dy = y - iy;
    // get_pixel_extend函数：本文件 和get_pixel()一样，只不过多了超出边界坐标的处理：凡是超过图像边界坐标的，设为边界坐标
    float val = (1-dy) * (1-dx) * get_pixel_extend(im, ix, iy, c) + 
        dy     * (1-dx) * get_pixel_extend(im, ix, iy+1, c) + 
        (1-dy) *   dx   * get_pixel_extend(im, ix+1, iy, c) +
        dy     *   dx   * get_pixel_extend(im, ix+1, iy+1, c);
    return val;
}


/**************************************************************************
*  func: 将图像source融入到dest中，偏移分别为dx和dy                       *
**************************************************************************/
void composite_image(image source, image dest, int dx, int dy)
{
    int x,y,k;
    for(k = 0; k < source.c; ++k){
        for(y = 0; y < source.h; ++y){
            for(x = 0; x < source.w; ++x){
				// get_pixel函数：本文件。分别得到图像source在索引x，y处通道k的值
                float val = get_pixel(source, x, y, k);
				// get_pixel_extend函数：本文件  其实和上一个函数一样
                float val2 = get_pixel_extend(dest, dx+x, dy+y, k);
				// set_pixel函数：本文件 设置dest的 dx+x,dy+y,在通道k的像素值为val*val2
                set_pixel(dest, dx+x, dy+y, k, val * val2);
            }
        }
    }
}
/**********************************************************
*  func: 给标签周围画上一个像素值为1得边框                *
**********************************************************/
image border_image(image a, int border)
{
    image b = make_image(a.w + 2*border, a.h + 2*border, a.c);
    int x,y,k;
    for(k = 0; k < b.c; ++k){
        for(y = 0; y < b.h; ++y){
            for(x = 0; x < b.w; ++x){
                float val = get_pixel_extend(a, x - border, y - border, k);
                if(x - border < 0 || x - border >= a.w || y - border < 0 || y - border >= a.h) val = 1;
                set_pixel(b, x, y, k, val);
            }
        }
    }
    return b;
}
/********************************************************
*  func: 将a图与b图拼接后返回，b在a右边dx像素           *
********************************************************/
image tile_images(image a, image b, int dx)
{
	// 当标签的第一个字母和空图拼接时，直接将第一个字母的像素数据深复制一份返回
	// copy_image函数：本文件 深复制b的内容，返回复制结果
    if(a.w == 0) return copy_image(b);
	// make_image：本文件  此处还是比较好玩的，不难理解
    image c = make_image(a.w + b.w + dx, (a.h > b.h) ? a.h : b.h, (a.c > b.c) ? a.c : b.c);
	// 先将中间图c的所有像素值置1
    fill_cpu(c.w*c.h*c.c, 1, c.data, 1);
	// embed_image函数：本文件。 将图片a嵌入到图片c中
    embed_image(a, c, 0, 0); 
	// composite_image函数：本文件 融合a到c中
    composite_image(b, c, a.w + dx, 0);
    return c;
}



/******************************************************************
*  func： 得到string得标签图相框                                  *
******************************************************************/
image get_label(image **characters, char *string, int size)
{
	// size是根据图像高度计算的size=im.h*0.03,拿1280*720举例，此时size=22,size/=10,所以size=2
	// 此处size表示会根据这个大小去对应的characters 库中找对应的图片
    size = size/10;
	// 总共就8种规格大小的ascii样本，所以最大取7
    if(size > 7) size = 7;
	// make_empty_image函数：本文件。生成image类型节点，data为0
    image label = make_empty_image(0,0,0);
	// 循环处理label中的每一个字符
    while(*string){
		// characters中行索引控制字符大小，列索引控制字符种类，列索引号对应字符的ascii
		// 此处是按照size的大小去取对用字母的图片至l中
        image l = characters[size][(int)*string];
		// tile_images函数：
        image n = tile_images(label, l, -size - 1 + (size+1)/2);
        free_image(label);
        label = n;
		// 指针后移
        ++string;
    }
	// border_image函数：本文件  为label图像加一个宽度为label.h*0.25的边框
    image b = border_image(label, label.h*.25);
    free_image(label);
	// 返回标签图片
    return b;
}

/********************************************************************
*  func：将标签图片label画到图片a上                                 *
********************************************************************/
void draw_label(image a, int r, int c, image label, const float *rgb)
{
    int w = label.w;
    int h = label.h;
    if (r - h >= 0) r = r - h;

    int i, j, k;
    for(j = 0; j < h && j + r < a.h; ++j){
        for(i = 0; i < w && i + c < a.w; ++i){
            for(k = 0; k < label.c; ++k){
                float val = get_pixel(label, i, j, k);
                set_pixel(a, i+c, j+r, k, rgb[k] * val);
            }
        }
    }
}



/************************************************************************************
*  func：自定义函数在图像a上画x1,y1,x2,y2所代表的矩形框，三通道值为r,g,b            *
************************************************************************************/
void draw_box(image a, int x1, int y1, int x2, int y2, float r, float g, float b)
{
    //normalize_image(a);
    int i;
    if(x1 < 0) x1 = 0;
    if(x1 >= a.w) x1 = a.w-1;
    if(x2 < 0) x2 = 0;
    if(x2 >= a.w) x2 = a.w-1;

    if(y1 < 0) y1 = 0;
    if(y1 >= a.h) y1 = a.h-1;
    if(y2 < 0) y2 = 0;
    if(y2 >= a.h) y2 = a.h-1;

	// 本循环画矩形框的水平线，需要结合Image中data的存储方式理解，这儿不赘述了
    for(i = x1; i <= x2; ++i){
        a.data[i + y1*a.w + 0*a.w*a.h] = r;
        a.data[i + y2*a.w + 0*a.w*a.h] = r;

        a.data[i + y1*a.w + 1*a.w*a.h] = g;
        a.data[i + y2*a.w + 1*a.w*a.h] = g;

        a.data[i + y1*a.w + 2*a.w*a.h] = b;
        a.data[i + y2*a.w + 2*a.w*a.h] = b;
    }
	// 本循环画矩形框的铅垂线
    for(i = y1; i <= y2; ++i){
        a.data[x1 + i*a.w + 0*a.w*a.h] = r;
        a.data[x2 + i*a.w + 0*a.w*a.h] = r;

        a.data[x1 + i*a.w + 1*a.w*a.h] = g;
        a.data[x2 + i*a.w + 1*a.w*a.h] = g;

        a.data[x1 + i*a.w + 2*a.w*a.h] = b;
        a.data[x2 + i*a.w + 2*a.w*a.h] = b;
    }
}


/************************************************************************
*  func：自定义函数在a上由外向内画宽度为w（线条宽度）的矩形框           *
************************************************************************/
void draw_box_width(image a, int x1, int y1, int x2, int y2, int w, float r, float g, float b)
{
    int i;
    for(i = 0; i < w; ++i){
		// draw_box函数：自定义函数画矩形框，好朴素的方法
        draw_box(a, x1+i, y1+i, x2-i, y2-i, r, g, b);
    }
}

void draw_bbox(image a, box bbox, int w, float r, float g, float b)
{
    int left  = (bbox.x-bbox.w/2)*a.w;
    int right = (bbox.x+bbox.w/2)*a.w;
    int top   = (bbox.y-bbox.h/2)*a.h;
    int bot   = (bbox.y+bbox.h/2)*a.h;

    int i;
    for(i = 0; i < w; ++i){
        draw_box(a, left+i, top+i, right-i, bot-i, r, g, b);
    }
}

/*********************************************************************************************************
*  func：加载data/labels/文件夹中所有的标签图片。所谓标签图片，就是仅含有单个字符的小图片，各标签图片组  *
*        合在一起，就可以得到完成的类别标签，data/labels含有8套美国标准ASCII码32～127号字符，每套间仅大  *
*        小（尺寸）不同，以应对不同大小的图片返回：image**，二维数组，8行128列，实际有效为8行96列，因为  *
*        前32列为0；每行包括一套32号～127号的ASCII标准码字符标签图片注意：image**实际有效值为后面的96列，*
*        前32列为0指针，之所以还要保留前32列，是保持秩序统一，便于之后的访问，访问时，直接将ASCII码转为  *
*        整型值即可得到在image**中的索引号，利于查找定位                                                 *
*********************************************************************************************************/
image **load_alphabet()
{
    int i, j;
    const int nsize = 8;
	// alphabets是一个二维image指针，每行代表一套，包含32~127个字符图片指针
	// 为每行第一个字符标签图片动态分配内存。共有8行，每行的首个元素都是存储image的指针，
    // 因此calloc(8, sizeof(image))（为指针分配内存不是为指针变量本身分配内存，而是为其所指的内存块分配内存）
    // calloc()这个动态分配函数会初始化元素值为0指针（malloc不会，之前内存中遗留了什么就是什么，不会重新初始化为0,
    // 而这里需要重新初始化为0,因此这里用的是calloc）
    image **alphabets = calloc(nsize, sizeof(image));
    for(j = 0; j < nsize; ++j){
		// 外循环8次，读入8套字符标签图片
		// 为每列动态分配内存，这里其实没有128个元素，只有96个元素，但是依然分配了128个元素，
        // 是为了便于之后的访问：直接将ASCII码转为整型值就可以得到在image中的索引，定位到相应的字符标签图片
        alphabets[j] = calloc(128, sizeof(image));
        for(i = 32; i < 127; ++i){
			// 内循环从32开始，读入32号~127号字符标签图片
            char buff[256];
			// int sprintf( char *buffer, const char *format, ... );
            // 按照指定格式将字符串输出至字符数组buffer中，得到每个字符标签图片的完整名称
            sprintf(buff, "data/labels/%d_%d.png", i, j);
			// load_image_color函数：本文件。视情况调用函数读取图片进image,imgae的w,h,c分别表示width,height和
			// channel，data域存放rrr...ggg...bbb...格式的图像数据，现在alphabets的每一个元素均为image对象
            alphabets[j][i] = load_image_color(buff, 0, 0);
        }
    }
    return alphabets;
}


/***************************************************************************************************************
*  func：检测最后一步：遍历得到的所有检测框，找出其最大所属概率，判断其是否大于指定阈值，如果大于则在输入图片  *
*                      上绘制检测信息，同时在终端输出检测结果（如果小于，则不作为）.                           *
*  args im: 输入图片，将在im上绘制检测结果，绘制内容包括定位用矩形框，由26个单字母拼接而成的类别标签           *
*  args dets: 包含了所有的有效框集合，其中每一个元素为detection结构体                                          *
*  args num：整张图片所拥有的box（物体检测框）的有效框个数                                                     *
*  args thresh: 概率阈值，每个box最大的所属类别概率必须大于这个阈值才接受这个检测结果，否则该box将作废         *
*              （即不认为其包含物体）                                                                          *
*  args names: 可以理解成一个二维字符数组（C中没有字符串的概念，用C字符数组实现字符串），每行存储一个物体的    *
*              名称，共有classes行                                                                             *
*  args alphabet: alphabet中的每一个元素均为image的结构体对象,具体存储大小不同的ascii为32-127的图片数据        *  
*  args classes: 物体类别总数                                                                                  *
***************************************************************************************************************/
void draw_detections(image im, detection *dets, int num, float thresh, char **names, image **alphabet, int classes)
{
    int i,j;

	// 遍历dets中的每一个框
    for(i = 0; i < num; ++i){
        char labelstr[4096] = {0};
        int class = -1;
        for(j = 0; j < classes; ++j){
			// 如果一个类别概率的得分值超过了阈值，则该框有意义，如果有多个类别概率大于阈值，则说明是多标签
            if (dets[i].prob[j] > thresh){
                if (class < 0) {
					// 将类别名称赋值给labelstr，并记录类别编号，注意字符串赋值只能用strcat哦
                    strcat(labelstr, names[j]);
                    class = j;
                } else {
                    strcat(labelstr, ", ");
                    strcat(labelstr, names[j]);
                }
				// 输出类别名称和其对应的类别概率值
                printf("%s: %.0f%%\n", names[j], dets[i].prob[j]*100);
            }
        }
		// class>=0 说明该框至少有一个目标类别概率大于阈值，则为有意义的框，需要画在im上
        if(class >= 0){
			// 这儿以1280*720的原图为例，假如有三个类别 person、car、dog，现假本框的类别为dog，则：
			// width=720*0.006=4.32=4  其实此处width只是代表框 线条宽度，以像素计
			// offset=2*123457%3=2 这个计算有必要么？
            int width = im.h * .006;
            int offset = class*123457 % classes;
			// get_color函数：本文件 应该是按照类别的不同获得RGB三通道不同的色调值，同类别得到的值相同
			// 也就是给不同类别画不同颜色的框，同类别画同颜色的框
            float red = get_color(2,offset,classes);
            float green = get_color(1,offset,classes);
            float blue = get_color(0,offset,classes);
            float rgb[3];

            //width = prob*20+2;

            rgb[0] = red;
            rgb[1] = green;
            rgb[2] = blue;
            box b = dets[i].bbox;
            //printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);

			// 根据xywh计算left/right/top/bot
            int left  = (b.x-b.w/2.)*im.w;
            int right = (b.x+b.w/2.)*im.w;
            int top   = (b.y-b.h/2.)*im.h;
            int bot   = (b.y+b.h/2.)*im.h;

			// 约束框到图像内
            if(left < 0) left = 0;
            if(right > im.w-1) right = im.w-1;
            if(top < 0) top = 0;
            if(bot > im.h-1) bot = im.h-1;

			// draw_box_width函数：自定义函数画矩形框
            draw_box_width(im, left, top, right, bot, width, red, green, blue);
            if (alphabet) {
				// get_label函数：本文件 得到labelstr得标签图片框
                image label = get_label(alphabet, labelstr, (im.h*.03));
				// draw_label函数：本文件 
                draw_label(im, top + width, left, label, rgb);
                free_image(label);
            }
            if (dets[i].mask){
                image mask = float_to_image(14, 14, 1, dets[i].mask);
                image resized_mask = resize_image(mask, b.w*im.w, b.h*im.h);
                image tmask = threshold_image(resized_mask, .5);
                embed_image(tmask, im, left, top);
                free_image(mask);
                free_image(resized_mask);
                free_image(tmask);
            }
        }
    }
}

void transpose_image(image im)
{
    assert(im.w == im.h);
    int n, m;
    int c;
    for(c = 0; c < im.c; ++c){
        for(n = 0; n < im.w-1; ++n){
            for(m = n + 1; m < im.w; ++m){
                float swap = im.data[m + im.w*(n + im.h*c)];
                im.data[m + im.w*(n + im.h*c)] = im.data[n + im.w*(m + im.h*c)];
                im.data[n + im.w*(m + im.h*c)] = swap;
            }
        }
    }
}

void rotate_image_cw(image im, int times)
{
    assert(im.w == im.h);
    times = (times + 400) % 4;
    int i, x, y, c;
    int n = im.w;
    for(i = 0; i < times; ++i){
        for(c = 0; c < im.c; ++c){
            for(x = 0; x < n/2; ++x){
                for(y = 0; y < (n-1)/2 + 1; ++y){
                    float temp = im.data[y + im.w*(x + im.h*c)];
                    im.data[y + im.w*(x + im.h*c)] = im.data[n-1-x + im.w*(y + im.h*c)];
                    im.data[n-1-x + im.w*(y + im.h*c)] = im.data[n-1-y + im.w*(n-1-x + im.h*c)];
                    im.data[n-1-y + im.w*(n-1-x + im.h*c)] = im.data[x + im.w*(n-1-y + im.h*c)];
                    im.data[x + im.w*(n-1-y + im.h*c)] = temp;
                }
            }
        }
    }
}



/******************************************************
*  func: 左右(水平)翻转图片a（本地翻转，即a也是输出） *
*  args a: 待翻转的图片                               *
******************************************************/
void flip_image(image a)
{
    int i,j,k;
    for(k = 0; k < a.c; ++k){
        for(i = 0; i < a.h; ++i){
			// 水平翻转的时候，水平索引只需要遍历到一半
            for(j = 0; j < a.w/2; ++j){
				// 看索引的时候可以将括号拆开了看更容易理解，这儿不赘述
                int index = j + a.w*(i + a.h*(k));
                int flip = (a.w - j - 1) + a.w*(i + a.h*(k));
				// 交换对应索引处的值
                float swap = a.data[flip];
                a.data[flip] = a.data[index];
                a.data[index] = swap;
            }
        }
    }
}

image image_distance(image a, image b)
{
    int i,j;
    image dist = make_image(a.w, a.h, 1);
    for(i = 0; i < a.c; ++i){
        for(j = 0; j < a.h*a.w; ++j){
            dist.data[j] += pow(a.data[i*a.h*a.w+j]-b.data[i*a.h*a.w+j],2);
        }
    }
    for(j = 0; j < a.h*a.w; ++j){
        dist.data[j] = sqrt(dist.data[j]);
    }
    return dist;
}

void ghost_image(image source, image dest, int dx, int dy)
{
    int x,y,k;
    float max_dist = sqrt((-source.w/2. + .5)*(-source.w/2. + .5));
    for(k = 0; k < source.c; ++k){
        for(y = 0; y < source.h; ++y){
            for(x = 0; x < source.w; ++x){
                float dist = sqrt((x - source.w/2. + .5)*(x - source.w/2. + .5) + (y - source.h/2. + .5)*(y - source.h/2. + .5));
                float alpha = (1 - dist/max_dist);
                if(alpha < 0) alpha = 0;
                float v1 = get_pixel(source, x,y,k);
                float v2 = get_pixel(dest, dx+x,dy+y,k);
                float val = alpha*v1 + (1-alpha)*v2;
                set_pixel(dest, dx+x, dy+y, k, val);
            }
        }
    }
}

void blocky_image(image im, int s)
{
    int i,j,k;
    for(k = 0; k < im.c; ++k){
        for(j = 0; j < im.h; ++j){
            for(i = 0; i < im.w; ++i){
                im.data[i + im.w*(j + im.h*k)] = im.data[i/s*s + im.w*(j/s*s + im.h*k)];
            }
        }
    }
}

void censor_image(image im, int dx, int dy, int w, int h)
{
    int i,j,k;
    int s = 32;
    if(dx < 0) dx = 0;
    if(dy < 0) dy = 0;

    for(k = 0; k < im.c; ++k){
        for(j = dy; j < dy + h && j < im.h; ++j){
            for(i = dx; i < dx + w && i < im.w; ++i){
                im.data[i + im.w*(j + im.h*k)] = im.data[i/s*s + im.w*(j/s*s + im.h*k)];
                //im.data[i + j*im.w + k*im.w*im.h] = 0;
            }
        }
    }
}



/************************************************************************************************************
*  func：将输入source图片的像素值嵌入到目标图片dest中.嵌入的列偏移和行偏移分别为dx,dy                       *
*  args source：源图片                                                                                      *
*  args dest: 目标图片（相当于该函数的输出）                                                                *
*  args dx：列偏移（dx=(source.w-dest.w)/2，因为源图尺寸一般小于目标图片尺寸，所以要将源图嵌入到目标图中心，*
*           源图需要在目标图上偏移dx开始插入，如下图）                                                      *
*  args dy: 行偏移（dy=(source.h-dest.h)/2）                                                                *
*  details: 下图所示，外层为目标图(dest),内层为源图(source),源图尺寸不超过目标图，源图嵌入到目标图中心      *
*                        ###############                                                                    *
*                        #             #                                                                    *
*                        #<-->######   #                                                                    *
*                        # dx #    #   #                                                                    *
*                        #    ######   #                                                                    *
*                        #      dy     #                                                                    *
*                        ###############                                                                    *
*           此函数是将源图片的嵌入到目标图片中心，意味着源图片的尺寸（宽高）不大于目标图片的尺寸，          *
*           目标图片在输入函数之前已经有初始化值了，因此如果源图片的尺寸小于目标图片尺寸，                  *
*           那么源图片会覆盖掉目标图片中新区域的像素值，而目标图片四周多出的像素值则保持为初始化值.         *
************************************************************************************************************/
void embed_image(image source, image dest, int dx, int dy)
{
    int x,y,k;
    for(k = 0; k < source.c; ++k){
        for(y = 0; y < source.h; ++y){
            for(x = 0; x < source.w; ++x){
				// 获取源图中k通道y行x列处的像素值
                float val = get_pixel(source, x,y,k);
				// 设置目标图中k通道dy+y行dx+x列处的像素值为val
                set_pixel(dest, dx+x, dy+y, k, val);
            }
        }
    }
}



image collapse_image_layers(image source, int border)
{
    int h = source.h;
    h = (h+border)*source.c - border;
    image dest = make_image(source.w, h, 1);
    int i;
    for(i = 0; i < source.c; ++i){
        image layer = get_image_layer(source, i);
        int h_offset = i*(source.h+border);
        embed_image(layer, dest, 0, h_offset);
        free_image(layer);
    }
    return dest;
}



/************************************************************************
*  func: 将图像的im所有通道的像素值严格限制在0.0~1.0内（像素值已经归一  *
*        化）（小于0的设为0,大于1的设为1,其他保持不变）                 *
*  args im: 待处理的图片                                                *
************************************************************************/
void constrain_image(image im)
{
    int i;
    for(i = 0; i < im.w*im.h*im.c; ++i){
        if(im.data[i] < 0) im.data[i] = 0;
        if(im.data[i] > 1) im.data[i] = 1;
    }
}

void normalize_image(image p)
{
    int i;
    float min = 9999999;
    float max = -999999;

    for(i = 0; i < p.h*p.w*p.c; ++i){
        float v = p.data[i];
        if(v < min) min = v;
        if(v > max) max = v;
    }
    if(max - min < .000000001){
        min = 0;
        max = 1;
    }
    for(i = 0; i < p.c*p.w*p.h; ++i){
        p.data[i] = (p.data[i] - min)/(max-min);
    }
}

void normalize_image2(image p)
{
    float *min = calloc(p.c, sizeof(float));
    float *max = calloc(p.c, sizeof(float));
    int i,j;
    for(i = 0; i < p.c; ++i) min[i] = max[i] = p.data[i*p.h*p.w];

    for(j = 0; j < p.c; ++j){
        for(i = 0; i < p.h*p.w; ++i){
            float v = p.data[i+j*p.h*p.w];
            if(v < min[j]) min[j] = v;
            if(v > max[j]) max[j] = v;
        }
    }
    for(i = 0; i < p.c; ++i){
        if(max[i] - min[i] < .000000001){
            min[i] = 0;
            max[i] = 1;
        }
    }
    for(j = 0; j < p.c; ++j){
        for(i = 0; i < p.w*p.h; ++i){
            p.data[i+j*p.h*p.w] = (p.data[i+j*p.h*p.w] - min[j])/(max[j]-min[j]);
        }
    }
    free(min);
    free(max);
}

void copy_image_into(image src, image dest)
{
    memcpy(dest.data, src.data, src.h*src.w*src.c*sizeof(float));
}



/***********************************************************
*  func: 拷贝p的data域至新的内存空间并返回                 *
*  args p: 待拷贝的图像数据                                *
***********************************************************/
image copy_image(image p)
{
    image copy = p;
    copy.data = calloc(p.h*p.w*p.c, sizeof(float));
	// memcpy函数原型：void *memcpy(void *dest, const void *src, size_t n);
	// memcpy指的是c和c++使用的内存拷贝函数，memcpy函数的功能是从源src所指
	// 的内存地址的起始位置开始拷贝n个字节到目标dest所指的内存地址的起始位置中。
    memcpy(copy.data, p.data, p.h*p.w*p.c*sizeof(float));
    return copy;
}



/***********************************************************
*  func：将im中data域的bbb...ggg...rrr组织的数据抓换为最终 *
*        的rrr...ggg...bbb格式，循环交换元素               *
*  args im：待调整的图片                                   *
*  details：其实也可以理解为广义的1、3通道交换bgr<->rgb互换*
***********************************************************/
void rgbgr_image(image im)
{
    int i;
	// 常规的交换元素的循环，类似冒泡排序的交换
    for(i = 0; i < im.w*im.h; ++i){
        float swap = im.data[i];
        im.data[i] = im.data[i+im.w*im.h*2];
        im.data[i+im.w*im.h*2] = swap;
    }
}



#ifdef OPENCV
void show_image_cv(image p, const char *name, IplImage *disp)
{
    int x,y,k;
    if(p.c == 3) rgbgr_image(p);
    //normalize_image(copy);

    char buff[256];
    //sprintf(buff, "%s (%d)", name, windows);
    sprintf(buff, "%s", name);

    int step = disp->widthStep;
    cvNamedWindow(buff, CV_WINDOW_NORMAL); 
    //cvMoveWindow(buff, 100*(windows%10) + 200*(windows/10), 100*(windows%10));
    ++windows;
    for(y = 0; y < p.h; ++y){
        for(x = 0; x < p.w; ++x){
            for(k= 0; k < p.c; ++k){
                disp->imageData[y*step + x*p.c + k] = (unsigned char)(get_pixel(p,x,y,k)*255);
            }
        }
    }
    if(0){
        int w = 448;
        int h = w*p.h/p.w;
        if(h > 1000){
            h = 1000;
            w = h*p.w/p.h;
        }
        IplImage *buffer = disp;
        disp = cvCreateImage(cvSize(w, h), buffer->depth, buffer->nChannels);
        cvResize(buffer, disp, CV_INTER_LINEAR);
        cvReleaseImage(&buffer);
    }
    cvShowImage(buff, disp);
}
#endif

int show_image(image p, const char *name, int ms)
{
#ifdef OPENCV
    IplImage *disp = cvCreateImage(cvSize(p.w,p.h), IPL_DEPTH_8U, p.c);
    image copy = copy_image(p);
    constrain_image(copy);
    show_image_cv(copy, name, disp);
    free_image(copy);
    cvReleaseImage(&disp);
    int c = cvWaitKey(ms);
    if (c != -1) c = c%256;
    return c;
#else
    fprintf(stderr, "Not compiled with OpenCV, saving to %s.png instead\n", name);
    save_image(p, name);
    return 0;
#endif
}

#ifdef OPENCV



/****************************************************************
*  func：将IplImage类型图片的imageData信息写入image的data域     *
*  args src: 欲操作的IplImage图片                               *
*  args im: 欲写入的image                                       *
****************************************************************/
void ipl_into_image(IplImage* src, image im)
{
	/*  示例代码如下,此处对应单通道读写：
		IplImage* imgSrc = cvLoadImage("./inputData\\shuke1.jpg",0);
		uchar* pixel = new uchar;
		for (int i = 0; i < imgSrc->height; i++)
		{
			for (int j = 0; j < imgSrc->width; j++)
			{
				pixel = (uchar*)(imgSrc->imageData + i*imgSrc->widthStep+j);
				cout << "pixel=" <<(*pixel)+0<< endl;//+0隐式转换为整型，否则会打印出字符
			}
		}
	   代码说明：*/
	// IplImage的imageData以unsigned char类型存放是为了节约存储空间？
	// imgSrc->imageData指向图像第一行的首地址，i是指当前像素点所在的行,widthStep
	// 是指图像每行所占的字节数；所以imgSrc->imageData + i*imgSrc->widthStep表示
	// 该像素点所在行的首地址；j表示当前像素点所在列，所以
	// imgSrc->imageData + i*imgSrc->widthStep+j即表示该像素点的地址。而因为
	// IplImage->ImageData 的默认类型是 char 类型，所以再对图像像素值进行操作时，
	// 要使用强制类型转换为unsigned char，再对其进行处理。否则，图像像素值中，会有负值出现。
    // widthStep表示存储一行像素需要的字节数
	// 因为opencv分配的内存是按4字节对齐的，所以widthStep必须是4的倍数，如果8U图像宽度为3，
	// 那么widthStep是4，加一个字节补齐。这个图像的一行需要4个字节，只使用前3个，最后一个
	// 空在那儿不用。也就是一个宽3高3的图像的imageData数据大小为4*3=12字节。
	/*  如下代码对应多通道的像素访问：
		IplImage* imgSrc = cvLoadImage("./inputData\\shuke1.jpg");
		uchar* b_pixel = new uchar;
		uchar* g_pixel = new uchar;
		uchar* r_pixel = new uchar;
		for (int i = 0; i < imgSrc->height; i++)
		{
			for (int j = 0; j < imgSrc->width; j++)
			{
				b_pixel = (uchar*)(imgSrc->imageData + i*imgSrc->widthStep + (j*imgSrc->nChannels + 0));
				g_pixel = (uchar*)(imgSrc->imageData + i*imgSrc->widthStep + (j*imgSrc->nChannels + 1));
				r_pixel=(uchar*)(imgSrc->imageData + i*imgSrc->widthStep + (j*imgSrc->nChannels + 2));
			}
		}
	*/
    unsigned char *data = (unsigned char *)src->imageData;
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    int step = src->widthStep;
    int i, j, k;

	// 嵌套循环，将IplImage.data按bgrbgr...bgrbgr组织的数据转化为bbb...ggg...rrr...
    for(i = 0; i < h; ++i){                // height
        for(k= 0; k < c; ++k){             // channel
            for(j = 0; j < w; ++j){        // width
			    // 这个转化关系可以详细推导，也可以记住，注意im.data[0]=src.imageData[0]，从而说明
				// image.data的组织方式是bbb...ggg...rrr...，而不是最后的rrr...ggg...bbb...
                im.data[k*w*h + i*w + j] = data[i*step + j*c + k]/255.;
            }
        }
    }
}



/****************************************************************
*  func：将以opencv的方式读取的IplImage类型图片转化为image类型  *
*  args src: 欲转化的IplImage图片                               *
*  details：调用 ipl_into_image函数将src.data转化为image.data   *                        *
****************************************************************/
image ipl_to_image(IplImage* src)
{
	// IplImage类型具有如下属性：
	/*
	    typedef struct _IplImage  
		{  
			int  nSize;         			IplImage大小  
			int  ID;            			版本 (=0) 
			int  nChannels;     			大多数OPENCV函数支持1,2,3 或 4 个通道  
			int  alphaChannel;  			被OpenCV忽略   
			int  depth;         			像素的位深度: IPL_DEPTH_8U, IPL_DEPTH_8S, IPL_DEPTH_16U, 
											IPL_DEPTH_16S, IPL_DEPTH_32S, IPL_DEPTH_32F and IPL_DEPTH_64F 可支持  
			char colorModel[4]; 			被OpenCV忽略 
			char channelSeq[4]; 			同上   
			int  dataOrder;    				0 - 交叉存取颜色通道, 
											1 - 分开的颜色通道. 
											cvCreateImage只能创建交叉存取图像  
			int  origin;       				0 - 顶—左结构, 
											1 - 底—左结构 (Windows bitmaps 风格)  
			int  align;         			图像行排列 (4 or 8). OpenCV 忽略它，使用 widthStep 代替 
			int  width;         			图像宽像素数 
			int  height;        			图像高像素数  
			struct _IplROI *roi;			图像感兴趣区域. 当该值非空只对该区域进行处理 
			struct _IplImage *maskROI;  	在 OpenCV中必须置NULL   
			void  *imageId;    				同上 
			struct _IplTileInfo *tileInfo;  同上 
			int  imageSize;     			图像数据大小(在交叉存取格式下imageSize=image->height*image->widthStep），单位字节  
			char *imageData;  				指向排列的图像数据  
			int  widthStep;   				列的图像行大小，以字节为单位 
			int  BorderMode[4]; 			边际结束模式, 被OpenCV忽略   
			int  BorderConst[4];  			同上 
			char *imageDataOrigin;  		指针指向一个不同的图像数据结构（不是必须排列的），是为了纠正图像内存分配准备的  
		}IplImage; 
    */ 
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
	// make_image函数：本文件。生成w*h*c大小的全0image的data域图片
    image out = make_image(w, h, c);
	// ipl_into_image函数：本文件。将IplImage类型图片的imageData信息写入image的data域
    ipl_into_image(src, out);
    return out;
}



/****************************************************************
*  func：以opencv的方式读取图片，返回图片的image类型信息        *
*  args filename: 欲读取的图片全称                              *
*  args channels: 欲读取图片的通道数                            *
*  details：会
****************************************************************/
image load_image_cv(char *filename, int channels)
{
	// IplImage:是opencv打开图片后得到的一种数据类型，详细结构定义见
	// https://www.cnblogs.com/codingmengmeng/p/6559724.html
	// https://baike.baidu.com/item/IplImage/5239886
    IplImage* src = 0;
	// flag标记欲读取图片通道数与cvLoadImage第二个参数之间的转化
    int flag = -1;
    if (channels == 0) flag = -1;
    else if (channels == 1) flag = 0;
    else if (channels == 3) flag = 1;
    else {
        fprintf(stderr, "OpenCV can't force load with %d channels\n", channels);
    }
	
	// 函数原型：IplImage* cvLoadImage( const char* filename, int flags=CV_LOAD_IMAGE_COLOR );
	// filename ：要被读入的文件的文件名(包括后缀)；
    // flags ：指定读入图像的颜色和深度：指定的颜色可以将输入的图片转为3信道(CV_LOAD_IMAGE_COLOR),
    // 单信道(CV_LOAD_IMAGE_GRAYSCALE), 或者保持不变(CV_LOAD_IMAGE_ANYCOLOR)。深度指定输入的图像
	// 是否转为每个颜色信道每象素8位，（OpenCV的早期版本一样），或者同输入的图像一样保持不变。
    // 选中CV_LOAD_IMAGE_ANYDEPTH，则输入图像格式可以为8位无符号，16位无符号，32位有符号或者32位浮点型。
    // 如果输入有冲突的标志，将采用较小的数字值。比如CV_LOAD_IMAGE_COLOR | CV_LOAD_IMAGE_ANYCOLOR 
	// 将载入3信道图。CV_LOAD_IMAGE_ANYCOLOR和CV_LOAD_IMAGE_UNCHANGED是等值的。但是，
	// CV_LOAD_IMAGE_ANYCOLOR有着可以和CV_LOAD_IMAGE_ANYDEPTH同时使用的优点，所以CV_LOAD_IMAGE_UNCHANGED不再使用了。
    // 如果想要载入最真实的图像，选择CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR。函数cvLoadImage
	// 从指定文件读入图像，返回读入图像的指针。实例如下：
	// cvLoadImage( filename, -1 ); 默认读取图像的原通道数
    // cvLoadImage( filename, 0 ); 强制转化读取图像为灰度图
    // cvLoadImage( filename, 1 ); 读取彩色图
    if( (src = cvLoadImage(filename, flag)) == 0 )
    {
        fprintf(stderr, "Cannot load image \"%s\"\n", filename);
        char buff[256];
        sprintf(buff, "echo %s >> bad.list", filename);
	    // 处理错误文件的代码，调用system运行命令行将错误文件加入bad.list文件中
	    // system(str)，相当于直接在中断输入str并按回车执行,此处echo %s >> bad.list 相当于把%s加入bad.list文件
        system(buff);
		// make_image函数：本文件。生成一个image类型节点，w=10,h=10,c=3,data域为大小为w*h*c的全0
        return make_image(10,10,3);
        // exit(0);
    }
	// ipl_to_image函数：本文件。将IplImage图片的imageData转化为image的data
    image out = ipl_to_image(src);
	// 释放src所占的内存空间，cvLoadImage是动态申请空间的，需要手动释放
    cvReleaseImage(&src);
	// rgbgr_image函数：本文件。交换out.data的1、3通道
    rgbgr_image(out);
    return out;
}

void flush_stream_buffer(CvCapture *cap, int n)
{
    int i;
    for(i = 0; i < n; ++i) {
        cvQueryFrame(cap);
    }
}



/*******************************************************
*  func：获取视频或者摄像头的一帧图像，处理后并返回    *
*  args cap：摄像头或视频指针                          *
*******************************************************/
image get_image_from_stream(CvCapture *cap)
{
	// 函数原型：IplImage* cvQueryFrame( CvCapture* capture );
	// 函数cvQueryFrame从摄像头或者文件中抓取一帧，然后解压并返回这一帧。 这个函数仅仅是函数cvGrabFrame和
    // 函数cvRetrieveFrame在一起调用的组合。 返回的图像不可以被用户释放或者修改。
	// cvQueryFrame的参数为CvCapture结构的指针。用来将下一帧视频文件载入内存，返回一个对应当前帧的指针。
	// 与cvLoadImage不同的是cvLoadImage为图像分配内存空间，而cvQueryFrame使用已经在CvCapture结构中分配好的内存。
	// 这样的话，就没有必要通过cvReleaseImage()对这个返回的图像指针进行释放，当CvCapture结构被释放后，
	// 每一帧图像所对应的内存空间即会被释放。
    IplImage* src = cvQueryFrame(cap);
	// 如果没有获取到图像帧则生成空图片返回
    if (!src) return make_empty_image(0,0,0);
	// ipl_to_image函数：本文件。将以opencv的方式读取的IplImage类型图片转化为image类型
    image im = ipl_to_image(src);
	// rgbgr_image函数：本文件。交换out.data的1、3通道
    rgbgr_image(im);
    return im;
}


/**************************************************************
*  func: 从cap中获取一帧图像数据，处理后写入im的data域        *
*  args cap: 获取图像帧的指针                                 *
*  args im: 写入图像的im指针                                  *
**************************************************************/
int fill_image_from_stream(CvCapture *cap, image im)
{
	// 函数原型：IplImage* cvQueryFrame( CvCapture* capture );
	// 函数cvQueryFrame从摄像头或者文件中抓取一帧，然后解压并返回这一帧。 这个函数仅仅是函数cvGrabFrame和
    // 函数cvRetrieveFrame在一起调用的组合。 返回的图像不可以被用户释放或者修改。
	// cvQueryFrame的参数为CvCapture结构的指针。用来将下一帧视频文件载入内存，返回一个对应当前帧的指针。
	// 与cvLoadImage不同的是cvLoadImage为图像分配内存空间，而cvQueryFrame使用已经在CvCapture结构中分配好的内存。
	// 这样的话，就没有必要通过cvReleaseImage()对这个返回的图像指针进行释放，当CvCapture结构被释放后，
	// 每一帧图像所对应的内存空间即会被释放。
    IplImage* src = cvQueryFrame(cap);
    if (!src) return 0;
	// ipl_into_image函数：本文件。将IplImage类型图片的imageData信息写入image的data域
    ipl_into_image(src, im);
    rgbgr_image(im);
    return 1;
}

void save_image_jpg(image p, const char *name)
{
    image copy = copy_image(p);
    if(p.c == 3) rgbgr_image(copy);
    int x,y,k;

    char buff[256];
    sprintf(buff, "%s.jpg", name);

    IplImage *disp = cvCreateImage(cvSize(p.w,p.h), IPL_DEPTH_8U, p.c);
    int step = disp->widthStep;
    for(y = 0; y < p.h; ++y){
        for(x = 0; x < p.w; ++x){
            for(k= 0; k < p.c; ++k){
                disp->imageData[y*step + x*p.c + k] = (unsigned char)(get_pixel(copy,x,y,k)*255);
            }
        }
    }
    cvSaveImage(buff, disp,0);
    cvReleaseImage(&disp);
    free_image(copy);
}
#endif

void save_image_png(image im, const char *name)
{
    char buff[256];
    //sprintf(buff, "%s (%d)", name, windows);
    sprintf(buff, "%s.png", name);
    unsigned char *data = calloc(im.w*im.h*im.c, sizeof(char));
    int i,k;
    for(k = 0; k < im.c; ++k){
        for(i = 0; i < im.w*im.h; ++i){
            data[i*im.c+k] = (unsigned char) (255*im.data[i + k*im.w*im.h]);
        }
    }
    int success = stbi_write_png(buff, im.w, im.h, im.c, data, im.w*im.c);
    free(data);
    if(!success) fprintf(stderr, "Failed to write image %s\n", buff);
}

void save_image(image im, const char *name)
{
#ifdef OPENCV
    save_image_jpg(im, name);
#else
    save_image_png(im, name);
#endif
}


void show_image_layers(image p, char *name)
{
    int i;
    char buff[256];
    for(i = 0; i < p.c; ++i){
        sprintf(buff, "%s - Layer %d", name, i);
        image layer = get_image_layer(p, i);
        show_image(layer, buff, 1);
        free_image(layer);
    }
}

void show_image_collapsed(image p, char *name)
{
    image c = collapse_image_layers(p, 1);
    show_image(c, name, 1);
    free_image(c);
}

/****************************************************************
*  func：生成一个w,h,c，data为0的image类型节点并返回            *
*  args w: 欲生成图片的width                                    *
*  args h: 欲生成图片的height                                   *
*  args c：欲生成图片的通道数                                   *
****************************************************************/
image make_empty_image(int w, int h, int c)
{
    image out;
	// image.data域为float的指针型，当作为普通浮点型时，可以先不开辟存储空间
    out.data = 0;
    out.h = h;
    out.w = w;
    out.c = c;
    return out;
}



/****************************************************************
*  func：调用make_empty_image函数，生成w,h,c的空白图片并返回    *
*  args w: 欲生成图片的width                                    *
*  args h: 欲生成图片的height                                   *
*  args c：欲生成图片的通道数                                   *
****************************************************************/
image make_image(int w, int h, int c)
{
	// make_empty_image函数：本文件。生成image类型节点，data为0 
    image out = make_empty_image(w,h,c);
	// 为image类型的out的data域开辟新的空间，大小为h*w*c*(sizeof(float))
	// 采用calloc开辟空间，空间内的值全部初始化为0
    out.data = calloc(h*w*c, sizeof(float));
    return out;
}



image make_random_image(int w, int h, int c)
{
    image out = make_empty_image(w,h,c);
    out.data = calloc(h*w*c, sizeof(float));
    int i;
    for(i = 0; i < w*h*c; ++i){
        out.data[i] = (rand_normal() * .25) + .5;
    }
    return out;
}

image float_to_image(int w, int h, int c, float *data)
{
    image out = make_empty_image(w,h,c);
    out.data = data;
    return out;
}



/*******************************************************************************************************
*  func: 先用双线性插值对输入图像im进行重排得到一个虚拟中间图（之所以称为虚拟，是因为这个中间图并不是  *
*        一个真实存在的变量），而后将中间图嵌入到canvas中（中间图尺寸比canvas小）或者将canvas当作一个  *
*        mask在im上抠图（canvas尺寸小于中间图尺寸）（canvas是帆布/画布的意思）                         *
*  args im: 源图                                                                                       *
*  args w: 中间图的宽度                                                                                *
*  args h: 中间图的高度                                                                                *
*  args dx: 中间图插入到canvas的x方向上的偏移                                                          *
*  args dy: 中间图插入到canvas的y方向上的偏移                                                          *
*  args canvas: 目标图（在传入到本函数之前，所有像素值已经初始化为某个值，比如0.5）                    *
*  details：此函数是实现图像数据增强手段中的一种：平移（不含旋转等其他变换）。先用双线性插值将源图im   *
*           重排至一个中间图，im与中间图的长宽比不一定一样，而后将中间图放入到输出图canvas中（这两者   *
*           的长宽比也不一定一样），分两种情况，如果中间图的尺寸小于输出图canvas，显然，中间图是填不   *
*           满canvas的，那么就将中间图随机的嵌入到canvas的某个位置（dx,dy就是起始位置，此时二者大于0,  *
*           相对canvas的起始位置，这两个数是在函数外随机生成的），因为canvas已经在函数外初始化了（比   *
*           如所有像素值初始化为0.5），剩下没填满的就保持为初始化值；如果canvas的尺寸小于中间图尺寸，  *
*           那么就将canvas当作一个mask，在中间图上随机位置（dx,dy就是起始位置，此时二者小于0,相对中间  *
*           图的起始位置）抠图。因为canvas与中间图的长宽比不一样，因此，有可能在一个方向是嵌入情况，而 *
*           在另一个方向上是mask情况，总之理解就可以了。可以参考一下：                                 *
*           https://medium.com/@vivek.yadav/dealing-with-unbalanced-data-generating-additional-data-   *
*           by-jittering-the-original-image-7497fe2119c3                                               *
*          上面网址给出了100张对原始图像进行增强之后的图片，可以看到很多图片有填满的，也有未填满的（无 *
*          黑区），且位置随机。(当然，网址中给出的图片包含了多种用于图片数据增强的变换，此函数仅仅完成 *
*          最简单的一种：平移)                                                                         *
*******************************************************************************************************/
void place_image(image im, int w, int h, int dx, int dy, image canvas)
{

    int x, y, c;
    for(c = 0; c < im.c; ++c){
		// 中循环和内循环的循环次数分别为中间图的行数与列数，这两个循环下来，实际可以得到中间图的所有像素值（当然，此处并没有形成一个真实的中间图变量）
        for(y = 0; y < h; ++y){
            for(x = 0; x < w; ++x){
				// x为中间图的列坐标，x/w*im.w得到中间图对应在源图上的列坐标（按比例得到，亚像素坐标）
                float rx = ((float)x / w) * im.w;
				// y为中间图的行坐标，y/h*im.h得到中间图对应在源图上的行坐标（TODO:这里代码实现显然存在不足，应该放入中循环，以减小没必要的计算）
                float ry = ((float)y / h) * im.h;
				// 利用源图进行双线性插值得到中间图在c通道y行x列处的像素值
				// bilinear_interpolate函数：本文件 得到im中rx列，ry行，通道c的像素值
                float val = bilinear_interpolate(im, rx, ry, c);
				// 设置canvas中c通道y+dy行x+dx列的像素值为val
                // dx,dy可大于0也可以小于0,大于0的情况很好理解，对于小于0的情况，x+dx以及y+dy会有一段小于0的，这时
                // set_pixel()函数无作为，直到x+dx,y+dy大于0时，才有作为，这样canvas相当于起到一个mask的作用
                set_pixel(canvas, x + dx, y + dy, c, val);
            }
        }
    }
}

image center_crop_image(image im, int w, int h)
{
    int m = (im.w < im.h) ? im.w : im.h;   
    image c = crop_image(im, (im.w - m) / 2, (im.h - m)/2, m, m);
    image r = resize_image(c, w, h);
    free_image(c);
    return r;
}

image rotate_crop_image(image im, float rad, float s, int w, int h, float dx, float dy, float aspect)
{
    int x, y, c;
    float cx = im.w/2.;
    float cy = im.h/2.;
    image rot = make_image(w, h, im.c);
    for(c = 0; c < im.c; ++c){
        for(y = 0; y < h; ++y){
            for(x = 0; x < w; ++x){
                float rx = cos(rad)*((x - w/2.)/s*aspect + dx/s*aspect) - sin(rad)*((y - h/2.)/s + dy/s) + cx;
                float ry = sin(rad)*((x - w/2.)/s*aspect + dx/s*aspect) + cos(rad)*((y - h/2.)/s + dy/s) + cy;
                float val = bilinear_interpolate(im, rx, ry, c);
                set_pixel(rot, x, y, c, val);
            }
        }
    }
    return rot;
}

image rotate_image(image im, float rad)
{
    int x, y, c;
    float cx = im.w/2.;
    float cy = im.h/2.;
    image rot = make_image(im.w, im.h, im.c);
    for(c = 0; c < im.c; ++c){
        for(y = 0; y < im.h; ++y){
            for(x = 0; x < im.w; ++x){
                float rx = cos(rad)*(x-cx) - sin(rad)*(y-cy) + cx;
                float ry = sin(rad)*(x-cx) + cos(rad)*(y-cy) + cy;
                float val = bilinear_interpolate(im, rx, ry, c);
                set_pixel(rot, x, y, c, val);
            }
        }
    }
    return rot;
}



/****************************************************************
*  func：将m图像的所有像素置为s                                 *
*  args m: 设置图片的指针                                       *    
*  args s: 设置的像素值大小                                     *
****************************************************************/
void fill_image(image m, float s)
{
    int i;
    for(i = 0; i < m.h*m.w*m.c; ++i) m.data[i] = s;
}



void translate_image(image m, float s)
{
    int i;
    for(i = 0; i < m.h*m.w*m.c; ++i) m.data[i] += s;
}

void scale_image(image m, float s)
{
    int i;
    for(i = 0; i < m.h*m.w*m.c; ++i) m.data[i] *= s;
}

image crop_image(image im, int dx, int dy, int w, int h)
{
    image cropped = make_image(w, h, im.c);
    int i, j, k;
    for(k = 0; k < im.c; ++k){
        for(j = 0; j < h; ++j){
            for(i = 0; i < w; ++i){
                int r = j + dy;
                int c = i + dx;
                float val = 0;
                r = constrain_int(r, 0, im.h-1);
                c = constrain_int(c, 0, im.w-1);
                val = get_pixel(im, c, r, k);
                set_pixel(cropped, i, j, k, val);
            }
        }
    }
    return cropped;
}

int best_3d_shift_r(image a, image b, int min, int max)
{
    if(min == max) return min;
    int mid = floor((min + max) / 2.);
    image c1 = crop_image(b, 0, mid, b.w, b.h);
    image c2 = crop_image(b, 0, mid+1, b.w, b.h);
    float d1 = dist_array(c1.data, a.data, a.w*a.h*a.c, 10);
    float d2 = dist_array(c2.data, a.data, a.w*a.h*a.c, 10);
    free_image(c1);
    free_image(c2);
    if(d1 < d2) return best_3d_shift_r(a, b, min, mid);
    else return best_3d_shift_r(a, b, mid+1, max);
}

int best_3d_shift(image a, image b, int min, int max)
{
    int i;
    int best = 0;
    float best_distance = FLT_MAX;
    for(i = min; i <= max; i += 2){
        image c = crop_image(b, 0, i, b.w, b.h);
        float d = dist_array(c.data, a.data, a.w*a.h*a.c, 100);
        if(d < best_distance){
            best_distance = d;
            best = i;
        }
        printf("%d %f\n", i, d);
        free_image(c);
    }
    return best;
}

void composite_3d(char *f1, char *f2, char *out, int delta)
{
    if(!out) out = "out";
    image a = load_image(f1, 0,0,0);
    image b = load_image(f2, 0,0,0);
    int shift = best_3d_shift_r(a, b, -a.h/100, a.h/100);

    image c1 = crop_image(b, 10, shift, b.w, b.h);
    float d1 = dist_array(c1.data, a.data, a.w*a.h*a.c, 100);
    image c2 = crop_image(b, -10, shift, b.w, b.h);
    float d2 = dist_array(c2.data, a.data, a.w*a.h*a.c, 100);

    if(d2 < d1 && 0){
        image swap = a;
        a = b;
        b = swap;
        shift = -shift;
        printf("swapped, %d\n", shift);
    }
    else{
        printf("%d\n", shift);
    }

    image c = crop_image(b, delta, shift, a.w, a.h);
    int i;
    for(i = 0; i < c.w*c.h; ++i){
        c.data[i] = a.data[i];
    }
#ifdef OPENCV
    save_image_jpg(c, out);
#else
    save_image(c, out);
#endif
}



/**********************************************************************
*  func: 将im中的图像resize倒中间图后嵌入到boxed中，原图不改变，改变  *
*        后的图保留在boxed里面，这个boxed图注意理解，一般上下或者左   *
*        右会补全像素                                                 *
*  args im：待调整的图片，此处不会更改原始图片                        *
*  args w: 调整后的width                                              *
*  args h: 调整后的height                                             *
*  args boxed: 调整后的图片存储位置                                   *
**********************************************************************/
void letterbox_image_into(image im, int w, int h, image boxed)
{
    int new_w = im.w;
    int new_h = im.h;
	// 当im的w比h大时，中间图片的宽度为w,高度按照原图宽高比同比压缩
    if (((float)w/im.w) < ((float)h/im.h)) {
        new_w = w;
        new_h = (im.h * w)/im.w;
	// 当im的w比h小时，中间图片的高度为h,宽度按照原图宽高比同比压缩
    } else {
        new_h = h;
        new_w = (im.w * h)/im.h;
    }
	// resize_image函数：本文件。按照w和h的大小重排图片的尺寸，不会改变im,返回的是新建的image
    image resized = resize_image(im, new_w, new_h);
	// embed_image函数：本文件。将输入resized图片的像素值嵌入到目标图片boxed中.嵌入的列偏移和行偏移分别为dx,dy 
    embed_image(resized, boxed, (w-new_w)/2, (h-new_h)/2); 
    free_image(resized);
}



/**********************************************************************************************************
*  func：按照神经网络能够接受处理的图片尺寸对输入图片进行尺寸调整（主要包括插值缩放与嵌入两个步骤）       *
*  args im: 输入图片（读入的原始图片）                                                                    *
*  args w: 网络处理的标准图片宽度（列）                                                                   *
*  args h: 网络处理的标准图片高度（行）                                                                   *
*  details:返回 boxed，image类型，其尺寸为神经网络能够处理的标准图片尺寸此函数常用于将图片输入卷积神经网  *
*          络之前，因为构建神经网络时，一般会指定神经网络第一层接受的输入图片尺寸，比如yolo.cfg神经网络配 *
*          置文件中，就指定了height=416,width=416，也就是说图片输入进神经网络之前，需要标准化图片的尺寸为 *
*          416,416（此时w=416,h=416）.流程主要包括两步：                                                  *
*          1）利用插值等比例缩放图片尺寸，缩放后图片resized的尺寸与原图尺寸比例为w/im.w与h/im.h中的较小值 *
*          2）等比例缩放后的图片resized，还不是神经网络能够处理的标准尺寸（但是resized宽、高二者有一个等于*
*          标准尺寸），第二步进一步将缩放后的图片resized嵌入到标准尺寸图片boxed中并返回                   *
**********************************************************************************************************/
image letterbox_image(image im, int w, int h)
{
    int new_w = im.w;
    int new_h = im.h;
	// 确认缩放后图片（resized）的尺寸大小
	// 缩放后的图片的尺寸与原图成等比例关系，比例值为w/im.w与h/im.h的较小者,
    // 总之最后的结果有两种：1）new_w=w,new_h=im.h*w/im.w；2）new_w=im.w*h/im.h,new_h=h,
    // 也即resized的宽高有一个跟标准尺寸w或h相同，另一个按照比例确定.
    if (((float)w/im.w) < ((float)h/im.h)) {
        new_w = w;
        new_h = (im.h * w)/im.w;
    } else {
        new_h = h;
        new_w = (im.w * h)/im.h;
    }
	// 第一步：缩放图片，使缩放后的图片尺寸为new_w,new_h
    image resized = resize_image(im, new_w, new_h);
	// 调用make_image创建标准尺寸图片，使用fill_image初始化所有像素值为0.5,两函数均在本文件定义
    image boxed = make_image(w, h, im.c);
    fill_image(boxed, .5);
    // int i;
    // for(i = 0; i < boxed.w*boxed.h*boxed.c; ++i) boxed.data[i] = 0;
	// 第二步：将缩放后的图片嵌入到标准尺寸图片中
	// embed_image函数：本文件。
    embed_image(resized, boxed, (w-new_w)/2, (h-new_h)/2); 
	// 切记释放没用的resized的内存，然后返回最终的标准尺寸图片
    free_image(resized);
    return boxed;
}



image resize_max(image im, int max)
{
    int w = im.w;
    int h = im.h;
    if(w > h){
        h = (h * max) / w;
        w = max;
    } else {
        w = (w * max) / h;
        h = max;
    }
    if(w == im.w && h == im.h) return im;
    image resized = resize_image(im, w, h);
    return resized;
}

image resize_min(image im, int min)
{
    int w = im.w;
    int h = im.h;
    if(w < h){
        h = (h * min) / w;
        w = min;
    } else {
        w = (w * min) / h;
        h = min;
    }
    if(w == im.w && h == im.h) return im;
    image resized = resize_image(im, w, h);
    return resized;
}

image random_crop_image(image im, int w, int h)
{
    int dx = rand_int(0, im.w - w);
    int dy = rand_int(0, im.h - h);
    image crop = crop_image(im, dx, dy, w, h);
    return crop;
}

augment_args random_augment_args(image im, float angle, float aspect, int low, int high, int w, int h)
{
    augment_args a = {0};
    aspect = rand_scale(aspect);
    int r = rand_int(low, high);
    int min = (im.h < im.w*aspect) ? im.h : im.w*aspect;
    float scale = (float)r / min;

    float rad = rand_uniform(-angle, angle) * TWO_PI / 360.;

    float dx = (im.w*scale/aspect - w) / 2.;
    float dy = (im.h*scale - w) / 2.;
    //if(dx < 0) dx = 0;
    //if(dy < 0) dy = 0;
    dx = rand_uniform(-dx, dx);
    dy = rand_uniform(-dy, dy);

    a.rad = rad;
    a.scale = scale;
    a.w = w;
    a.h = h;
    a.dx = dx;
    a.dy = dy;
    a.aspect = aspect;
    return a;
}

image random_augment_image(image im, float angle, float aspect, int low, int high, int w, int h)
{
    augment_args a = random_augment_args(im, angle, aspect, low, high, w, h);
    image crop = rotate_crop_image(im, a.rad, a.scale, a.w, a.h, a.dx, a.dy, a.aspect);
    return crop;
}



/*************************************************************
*  func: 如下两函数分别返回a,b,c中的最大值和最小值           *
*************************************************************/
float three_way_max(float a, float b, float c)
{
    return (a > b) ? ( (a > c) ? a : c) : ( (b > c) ? b : c) ;
}
float three_way_min(float a, float b, float c)
{
    return (a < b) ? ( (a < c) ? a : c) : ( (b < c) ? b : c) ;
}



void yuv_to_rgb(image im)
{
    assert(im.c == 3);
    int i, j;
    float r, g, b;
    float y, u, v;
    for(j = 0; j < im.h; ++j){
        for(i = 0; i < im.w; ++i){
            y = get_pixel(im, i , j, 0);
            u = get_pixel(im, i , j, 1);
            v = get_pixel(im, i , j, 2);

            r = y + 1.13983*v;
            g = y + -.39465*u + -.58060*v;
            b = y + 2.03211*u;

            set_pixel(im, i, j, 0, r);
            set_pixel(im, i, j, 1, g);
            set_pixel(im, i, j, 2, b);
        }
    }
}

void rgb_to_yuv(image im)
{
    assert(im.c == 3);
    int i, j;
    float r, g, b;
    float y, u, v;
    for(j = 0; j < im.h; ++j){
        for(i = 0; i < im.w; ++i){
            r = get_pixel(im, i , j, 0);
            g = get_pixel(im, i , j, 1);
            b = get_pixel(im, i , j, 2);

            y = .299*r + .587*g + .114*b;
            u = -.14713*r + -.28886*g + .436*b;
            v = .615*r + -.51499*g + -.10001*b;

            set_pixel(im, i, j, 0, y);
            set_pixel(im, i, j, 1, u);
            set_pixel(im, i, j, 2, v);
        }
    }
}


/**************************************************
*  func: 将彩色图像im由RGB空间转化到HSV空间       *
*  args im：待转化的图片                          *
**************************************************/
void rgb_to_hsv(image im)
{
	// http://www.cs.rit.edu/~ncs/color/t_convert.html
	// HSV(Hue, Saturation, Value)是根据颜色的直观特性由A. R. Smith在1978年创建的一种颜色空间, 也称六角锥体模型(Hexcone Model)。
    // 这个模型中颜色的参数分别是：色调（H），饱和度（S），明度（V）。
	// https://www.cnblogs.com/zl1991/p/4913450.html
	// 一般的3D编程只需要使用RGB颜色空间就好了，但其实美术人员更多的是使用HSV(HSL)，因为可以方便的调整饱和度和亮度。
    // 有时候美术需要程序帮助调整饱和度来达到特定风格的渲染效果，这时候就需要转换颜色空间了
	/* RGB转化到HSV的算法: https://baike.baidu.com/item/HSV/547122?fromtitle=HSV%E9%A2%9C%E8%89%B2%E7%A9%BA%E9%97%B4&fromid=12630604&fr=aladdin
		max=max(R,G,B)；
		min=min(R,G,B)；
		V=max(R,G,B)；
		S=(max-min)/max；
		HSV颜色空间模型（圆锥模型）
		HSV颜色空间模型（圆锥模型） [2]
		if (R = max) H =(G-B)/(max-min)* 60；
		if (G = max) H = 120+(B-R)/(max-min)* 60；
		if (B = max) H = 240 +(R-G)/(max-min)* 60；
		if (H < 0) H = H+ 360；
    */
    assert(im.c == 3);
    int i, j;
    float r, g, b;
    float h, s, v;
    for(j = 0; j < im.h; ++j){
        for(i = 0; i < im.w; ++i){
			// get_pixel函数：本文件 分别得到图像m在索引i,j处三通道的值
            r = get_pixel(im, i , j, 0);
            g = get_pixel(im, i , j, 1);
            b = get_pixel(im, i , j, 2);
			// three_way_max函数：本文件  返回r,g,b中的最大值
            float max = three_way_max(r,g,b);
			// three_way_min函数：本文件  返回r,g,b中的最小值
            float min = three_way_min(r,g,b);
            float delta = max - min;
            v = max;
            if(max == 0){
                s = 0;
                h = 0;
            }else{
                s = delta/max;
                if(r == max){
                    h = (g - b) / delta;
                } else if (g == max) {
                    h = 2 + (b - r) / delta;
                } else {
                    h = 4 + (r - g) / delta;
                }
                if (h < 0) h += 6;
                h = h/6.;
            }
			// 将计算出来的值调用set_pixel函数插入到对应处
            set_pixel(im, i, j, 0, h);
            set_pixel(im, i, j, 1, s);
            set_pixel(im, i, j, 2, v);
        }
    }
}



/**************************************************
*  func: 将彩色图像im由HSV空间转化到RGB空间       *
*  args im：待转化的图片                          *
**************************************************/
void hsv_to_rgb(image im)
{
	/* 具体注释可以对照rgb_to_hsv看，这里不在赘述
		HSV转化到RGB的算法:
		if (s = 0)
		R=G=B=V;
		else
		H /= 60;
		i = INTEGER(H);
		f = H - i;
		a = V * ( 1 - s );
		b = V * ( 1 - s * f );
		c = V * ( 1 - s * (1 - f ) );
		switch(i)
		case 0: R = V; G = c; B = a;
		case 1: R = b; G = v; B = a;
		case 2: R = a; G = v; B = c;
		case 3: R = a; G = b; B = v;
		case 4: R = c; G = a; B = v;
		case 5: R = v; G = a; B = b;
	*/
    assert(im.c == 3);
    int i, j;
    float r, g, b;
    float h, s, v;
    float f, p, q, t;
    for(j = 0; j < im.h; ++j){
        for(i = 0; i < im.w; ++i){
            h = 6 * get_pixel(im, i , j, 0);
            s = get_pixel(im, i , j, 1);
            v = get_pixel(im, i , j, 2);
            if (s == 0) {
                r = g = b = v;
            } else {
                int index = floor(h);
                f = h - index;
                p = v*(1-s);
                q = v*(1-s*f);
                t = v*(1-s*(1-f));
                if(index == 0){
                    r = v; g = t; b = p;
                } else if(index == 1){
                    r = q; g = v; b = p;
                } else if(index == 2){
                    r = p; g = v; b = t;
                } else if(index == 3){
                    r = p; g = q; b = v;
                } else if(index == 4){
                    r = t; g = p; b = v;
                } else {
                    r = v; g = p; b = q;
                }
            }
            set_pixel(im, i, j, 0, r);
            set_pixel(im, i, j, 1, g);
            set_pixel(im, i, j, 2, b);
        }
    }
}

void grayscale_image_3c(image im)
{
    assert(im.c == 3);
    int i, j, k;
    float scale[] = {0.299, 0.587, 0.114};
    for(j = 0; j < im.h; ++j){
        for(i = 0; i < im.w; ++i){
            float val = 0;
            for(k = 0; k < 3; ++k){
                val += scale[k]*get_pixel(im, i, j, k);
            }
            im.data[0*im.h*im.w + im.w*j + i] = val;
            im.data[1*im.h*im.w + im.w*j + i] = val;
            im.data[2*im.h*im.w + im.w*j + i] = val;
        }
    }
}

image grayscale_image(image im)
{
    assert(im.c == 3);
    int i, j, k;
    image gray = make_image(im.w, im.h, 1);
    float scale[] = {0.299, 0.587, 0.114};
    for(k = 0; k < im.c; ++k){
        for(j = 0; j < im.h; ++j){
            for(i = 0; i < im.w; ++i){
                gray.data[i+im.w*j] += scale[k]*get_pixel(im, i, j, k);
            }
        }
    }
    return gray;
}

image threshold_image(image im, float thresh)
{
    int i;
    image t = make_image(im.w, im.h, im.c);
    for(i = 0; i < im.w*im.h*im.c; ++i){
        t.data[i] = im.data[i]>thresh ? 1 : 0;
    }
    return t;
}

image blend_image(image fore, image back, float alpha)
{
    assert(fore.w == back.w && fore.h == back.h && fore.c == back.c);
    image blend = make_image(fore.w, fore.h, fore.c);
    int i, j, k;
    for(k = 0; k < fore.c; ++k){
        for(j = 0; j < fore.h; ++j){
            for(i = 0; i < fore.w; ++i){
                float val = alpha * get_pixel(fore, i, j, k) + 
                    (1 - alpha)* get_pixel(back, i, j, k);
                set_pixel(blend, i, j, k, val);
            }
        }
    }
    return blend;
}



/******************************************************************
*  func: 将图像c通道的所有元素值乘以一个因子v（放大或者缩小）     *
*  args im：输入图片                                              *
*  args c: 要进行元素值缩放的通道编号                             *
*  args v：缩放因子                                               *
*  details：比如用于数据增强，对图像的饱和度以及明度进行缩放      *
******************************************************************/
void scale_image_channel(image im, int c, float v)
{
    int i, j;
    for(j = 0; j < im.h; ++j){
        for(i = 0; i < im.w; ++i){
			// 获取im图像c通道j行i列的元素值，而后乘以因子v进行缩放，然后更新当前像素值
			// 跟新值就是简单得缩放
            float pix = get_pixel(im, i, j, c);
            pix = pix*v;
            set_pixel(im, i, j, c, pix);
        }
    }
}

void translate_image_channel(image im, int c, float v)
{
    int i, j;
    for(j = 0; j < im.h; ++j){
        for(i = 0; i < im.w; ++i){
            float pix = get_pixel(im, i, j, c);
            pix = pix+v;
            set_pixel(im, i, j, c, pix);
        }
    }
}

image binarize_image(image im)
{
    image c = copy_image(im);
    int i;
    for(i = 0; i < im.w * im.h * im.c; ++i){
        if(c.data[i] > .5) c.data[i] = 1;
        else c.data[i] = 0;
    }
    return c;
}

void saturate_image(image im, float sat)
{
    rgb_to_hsv(im);
    scale_image_channel(im, 1, sat);
    hsv_to_rgb(im);
    constrain_image(im);
}

void hue_image(image im, float hue)
{
    rgb_to_hsv(im);
    int i;
    for(i = 0; i < im.w*im.h; ++i){
        im.data[i] = im.data[i] + hue;
        if (im.data[i] > 1) im.data[i] -= 1;
        if (im.data[i] < 0) im.data[i] += 1;
    }
    hsv_to_rgb(im);
    constrain_image(im);
}

void exposure_image(image im, float sat)
{
    rgb_to_hsv(im);
    scale_image_channel(im, 2, sat);
    hsv_to_rgb(im);
    constrain_image(im);
}



/*************************************************************************************
*  func: 先将输入的彩色图片im由RGB颜色空间转换至hsv空间，而后在hsv空间的h,s,v        *
*        三通道上在添加噪声，以实现数据增强                                          *
*  args im: 读入的彩色图片                                                           *
*  args hue: 色调偏差值                                                              *
*  args sat: 色彩饱和度（取值范围0~1）缩放因子                                       *
*  args val: 明度（色彩明亮程度，0~1）缩放因子                                       *
*************************************************************************************/
void distort_image(image im, float hue, float sat, float val)
{
	// rgb_to_hsv函数：本文件 将图像由rgb空间转至hsv空间
    rgb_to_hsv(im);
	// scale_image_channel函数：本文件 缩放第二通道即s通道的值以进行图像jitter（绕动，或者说添加噪声），进而实现数据增强
    scale_image_channel(im, 1, sat);
	// 缩放第三通道即v通道的值以进行图像jitter（绕动，或者说添加噪声），进而实现数据增强
    scale_image_channel(im, 2, val);
    int i;
	// 对于第0通道即h通道，直接添加指定偏差值来实现图像jitter
    for(i = 0; i < im.w*im.h; ++i){
        im.data[i] = im.data[i] + hue;
        if (im.data[i] > 1) im.data[i] -= 1;
        if (im.data[i] < 0) im.data[i] += 1;
    }
	// hsv_to_rgb函数：本文件 将图像变换回rgb颜色空间
    hsv_to_rgb(im);
	// constrain_image函数：本文件 可能之前在hsv空间添加的绕动使得变换回RGB空间之后，其像素值不在合理范围内（像素值已经归一化至0~1），
    // 通过constrain_image()函数将严格限制在0~1范围内（小于0的设为0,大于1的设为1,其他保持不变）
    constrain_image(im);
}



/********************************************************************************************************************
*  func: 此函数先将输入的颜色图片由RGB颜色空间转换至hsv空间，而后在hsv空间的h,s,v三通道上在添加噪声，以实现数据增强 *
*  args im：读入的彩色图片                                                                                          *
*  args hue: 色调（取值0度到360度）偏差最大值，实际色调偏差为-hue~hue之间的随机值                                   *
*  args saturation：色彩饱和度（取值范围0~1）缩放最大值，实际为范围内的随机值                                       *
*  args exposure: 明度（色彩明亮程度，0~1）缩放最大值，实际为范围内的随机值                                         *
*  details：色调正常都是0度到360度的，但是这里并没有乘以60，所以范围正常为0~6，此外，rgb_to_hsv()函数最后还除以了6  *
*           进行了类似归一化的操作，不知道是何意，总之不管怎样，在hsv_to_rgb()函数中将hsv转回至rgb配套上就可以了    *
********************************************************************************************************************/
void random_distort_image(image im, float hue, float saturation, float exposure)
{
	// 下面依次随机产生具体的色调，饱和度与明读的偏差值，三函数均定义在src/utils.c
    float dhue = rand_uniform(-hue, hue);
	// 得到1-saturation之间均匀分布的值或其倒数值
    float dsat = rand_scale(saturation);
    float dexp = rand_scale(exposure);
	// distort_image函数：本文件  在hsv空间h,s,v三个通道上，为图像增加噪声（对于h是添加偏差，对于s,v是缩放），以进行图像数据增强
    distort_image(im, dhue, dsat, dexp);
}

void saturate_exposure_image(image im, float sat, float exposure)
{
    rgb_to_hsv(im);
    scale_image_channel(im, 1, sat);
    scale_image_channel(im, 2, exposure);
    hsv_to_rgb(im);
    constrain_image(im);
}

/**********************************************************************************************************
*  func：按照w和h的大小重排图片的尺寸                                                                     *
*  args im：待重排的图片                                                                                  *
*  args w:  新的图片宽度                                                                                  *
*  args h:  新的图片高度                                                                                  *
*  details：目标图片（缩放后的图片）resized                                                               *
*  1）此处重排并不是保持像素个数意义上的重排，像素个数是可以变的，即图像缩放，这就涉及到像素插值操作      *
*  2）重排操作分两步完成，第一步图像行数不变，仅缩放图像的列数，而后在水平方向上（x方向）进行像素线性插值 *
*  3）第二步，在第一步基础上，保持列数不变（列在第一步已经缩放完成），缩放行数，在竖直方向（y方向）进行   *
*     像素线性插值，两步叠加在一起，其实就相当于是双线性插值                                              *
**********************************************************************************************************/
image resize_image(image im, int w, int h)
{
	// 创建目标图像：宽高分别为w、h（为最终目标图像的尺寸），通道数和原图保持一样
    image resized = make_image(w, h, im.c);
	// 创建中间图像：宽为w，高保持与原图一样，中间图像为第一步缩放插值之后的结果	
    image part = make_image(w, im.h, im.c);
    int r, c, k;
	// 计算目标图像与原图宽高比例（分子分母都减了1,因为四周边缘上的像素不需要插值，直接等于原图四周边缘像素值就可以）
    float w_scale = (float)(im.w - 1) / (w - 1);
    float h_scale = (float)(im.h - 1) / (h - 1);
	// 第一步：缩放宽，保持高度不变
    // 遍历所有通道（注意不管是通道还是行、列，都是按中间图像尺寸遍历，因为要获取中间图像每个像素的值）
    for(k = 0; k < im.c; ++k){
		// 遍历所有行
        for(r = 0; r < im.h; ++r){
			// 遍历所有列
            for(c = 0; c < w; ++c){
                float val = 0;
				// 对于中间图像右边缘上的点，其直接就等于原图对应行右边缘上的点的像素值，不需要进行线性插值（没法插值，右边没有像素了），
                // 如果原图的宽度就为1,那么也不用插值了，不管怎么缩放，中间图像所有像素值就为原图对应行的像素值
                // 是不是少考虑中间图像左边缘上的像素？左边缘上的像素也不用插值，直接等于原图对应行左边缘像素值就可以了，
                // 只不过这里没有放到if语句中（其实也可以的），并到else中了（简单分析一下就可知道else中包括了这种情况的处理），
                // 个人感觉此处处理不是最佳的，直接在if语句中考虑左右两边的像素更为清晰
                if(c == w-1 || im.w == 1){
					//get_pixel函数：本文件。取出im图像中w,r,k处的像素值
                    val = get_pixel(im, im.w-1, r, k);
                } else {
					// 正常情况，都是需要插值的，c*w_scale为中间图c列对应在原图中的列数
                    float sx = c*w_scale;
                    int ix = (int) sx;
					// sx一般不为整数，所以需要用ix及ix+1列处两个像素进行线性插值获得中间图c列（r行）处的像素值
                    float dx = sx - ix;
					// 线性插值：像素值的权重与到两个像素的坐标距离成反比（分别获取原图im的k通道r行ix和ix+1列处的像素值）
                    val = (1 - dx) * get_pixel(im, ix, r, k) + dx * get_pixel(im, ix+1, r, k);
                }
				// set_pixel函数：本文件。设置中间图part的k通道r行，c列处的像素值为val
                set_pixel(part, c, r, k, val);
            }
        }
    }
	// 第二步：在第一步的基础上，保持列数不变，缩放宽
    // 遍历所有通道（注意不管是通道还是行、列，都是按最终目标图像尺寸遍历，因为要获取目标图像每个像素的值）
    for(k = 0; k < im.c; ++k){
        for(r = 0; r < h; ++r){
			// 过程和第一步类似
            // 获取目标图片中第r行对应在中间图中的行数
            float sy = r*h_scale;
            int iy = (int) sy;
            float dy = sy - iy;
			// 这里分了两个for循环来完成插值，个人感觉不可取，直接用一个循环，
            // 加一个if语句单独考虑边缘像素的插值情况以及原图高为1的情况，既清晰，又提高了效率
            for(c = 0; c < w; ++c){
				// 获取中间图像part的k通道iy行c列处的像素值
                float val = (1-dy) * get_pixel(part, c, iy, k);
                set_pixel(resized, c, r, k, val);
            }
			// 如果是下边缘上的像素，或者原图的高度等于1,那么就没有必要插值了，直接跳过
            // 至于上边缘，在下面的for循环中考虑了（此时sy=0,iy=0,dy=0）
            if(r == h-1 || im.h == 1) continue;
			// 正常情况，需要进行线性像素插值：叠加上下一行的加权像素值
            for(c = 0; c < w; ++c){
                float val = dy * get_pixel(part, c, iy+1, k);
				// add_pixel函数：本文件。将resized图像k通道r行c列的像素值叠加val
                add_pixel(resized, c, r, k, val);
            }
        }
    }
    // 释放中间图片的空间
    free_image(part);
    return resized;
}



void test_resize(char *filename)
{
    image im = load_image(filename, 0,0, 3);
    float mag = mag_array(im.data, im.w*im.h*im.c);
    printf("L2 Norm: %f\n", mag);
    image gray = grayscale_image(im);

    image c1 = copy_image(im);
    image c2 = copy_image(im);
    image c3 = copy_image(im);
    image c4 = copy_image(im);
    distort_image(c1, .1, 1.5, 1.5);
    distort_image(c2, -.1, .66666, .66666);
    distort_image(c3, .1, 1.5, .66666);
    distort_image(c4, .1, .66666, 1.5);


    show_image(im,   "Original", 1);
    show_image(gray, "Gray", 1);
    show_image(c1, "C1", 1);
    show_image(c2, "C2", 1);
    show_image(c3, "C3", 1);
    show_image(c4, "C4", 1);
#ifdef OPENCV
    while(1){
        image aug = random_augment_image(im, 0, .75, 320, 448, 320, 320);
        show_image(aug, "aug", 1);
        free_image(aug);


        float exposure = 1.15;
        float saturation = 1.15;
        float hue = .05;

        image c = copy_image(im);

        float dexp = rand_scale(exposure);
        float dsat = rand_scale(saturation);
        float dhue = rand_uniform(-hue, hue);

        distort_image(c, dhue, dsat, dexp);
        show_image(c, "rand", 1);
        printf("%f %f %f\n", dhue, dsat, dexp);
        free_image(c);
        cvWaitKey(0);
    }
#endif
}



/*******************************************************************************************************
*  func：调用开源库stb_image.h中的函数stbi_load()读入指定图片，并转为darkenet的image类型后返回image变量*
*        stbi_load()返回的值是unsigned char*类型，且数据存储方式是rgbrgb...格式（只有一行），          *
*        而darknet中的image是三个通道分开存储的（但还是只有一行），类似这种形式：rrr...ggg...bbb...    *
*        本函数完成了类型以及存储格式的转换；第二个参数channels是期待的图片的通道数，                  *
*        如果读入图片通道数不等于channels，会进行强转（在stbi_load()函数内部完成转换），               *
*        这和opencv中的图片读入函数类似，同样可以指定读入图片的通道，比如即使图片是彩色图，            *
*        也可以通过指定通道数读入灰度图                                                                *
*  args filename: 欲读取的图片全称                                                                     *
*  args hannels: 期待图片的通道数                                                                      *
*  args details：返回image类型变量，灰度值被归一化至0～1,灰度值存储方式为rrr...ggg...bbb...（只有一行）*
*                三通道分开存储，每通道的二维数据按行存储（所有行并成一行），而后三通道再并成一行存储  *
*******************************************************************************************************/
image load_image_stb(char *filename, int channels)
{
    int w, h, c;
	// stbi_load()是开源库stb_image.h中的函数，读入指定图片，返回unsigned char*类型数据
    // 该库是用C语言写的专门用来读取图片数据的（非常复杂），读入二维图片按行存储到data中（所有行并成一行），
    // 此处w,h,c为读入图片的宽，高，通道，是读入图片后，在stbi_load中赋值的，c是图片的真实通道数，彩色图为3通道，灰度图为单通道
    // 而channels是期待的图像通道（也是输出的图的通道数，转换之后的），因此如果c!=channels，stbi_load()会进行通道转换，
    // 比如图片是彩色图，那么c=3，而如果channels指定为1，说明只想读入灰度图，
    // stbi_load()函数会完成这一步转换，最终输出通道为channels=1的灰度图像数据
    // 从下面的代码可知，如果data是三通道的，那么data存储方式是三通道杂揉存储的即rgbrgb...方式，且全部都存储在一行中
    // 注意： channels的取值必须取1,2,3,4中的某个，如果不是，会发生断言，中断程序
    unsigned char *data = stbi_load(filename, &w, &h, &c, channels);
    if (!data) {
        fprintf(stderr, "Cannot load image \"%s\"\nSTB Reason: %s\n", filename, stbi_failure_reason());
        exit(0);
    }
	// stbi_load()函数读入的图片数据格式为unsigned char*类型，
    // 接下来要转换为darknet中的image类型
    // 这个if语句没有必要，因子stbi_load()中要求channels必须为1,2,3,4，否则会发生断言，
    // 且stbi_load()函数会判断c是否等于channels，如果不相等，则会进行通道转换（灰度转换），
    // 所以直接令c = channels即可
    if(channels) c = channels;
    int i,j,k;
	// 创建一张图片，并分配必要的内存
    image im = make_image(w, h, c);
	// 将图片像素值存入im中
    for(k = 0; k < c; ++k){
        for(j = 0; j < h; ++j){
            for(i = 0; i < w; ++i){
				// 转完之后的像素索引：dst_index =i+w*j+w*h*k，其中i表示在im.data中的列偏移，
				// w*j表示换行偏移，w*h*k表示换通道偏移，因此转完后得到的im.data是三通道分开
				// 存储的，且每通道都是将二维数据按行存储（所有行并成一行），然后三通道再并成一行
                int dst_index = i + w*j + w*h*k;
				// 在data中的存储方式是三通道杂揉在一起的：rgbrgbrgb...，因此，
                // src_index = k + c(i+w*j)中，i+w*j表示单通道的偏移，乘以c则包括总共3通道的偏移，
                // 加上w表示要读取w通道的灰度值。
                // 比如，图片原本是颜色图，因此data原本应该是rgbrgbrgb...类型的数据，
                // 但如果指定的channels=1,data将是经过转换后通道数为1的图像数据，这时k=0，只能读取一个通道的数据;
                // 如果channels=3，那么data保持为rgbrgbrgb...存储格式，这时w=0将读取所有r通道的数据，
                // w=1将读取所有g通道的数据，w=2将读取所有b通道的数据
                int src_index = k + c*i + c*w*j;
				// 图片的灰度值转换为0～1（强转为float型）
                im.data[dst_index] = (float)data[src_index]/255.;
            }
        }
    }
	// 及时释放已经用完的data的内存
    free(data);
    return im;
}



/**************************************************************************************
*  func：根据是否使用了opencv来加载图片，返回加载结果                                 *
*  args filename: 欲读取的图片全称                                                    *
*  args w: 期待图片的width                                                            *
*  args h: 期待图片的height                                                           *
*  args c：期待图片的通道数                                                           *
*  details：w,h是期待的图片宽，高，如果指定为0,0,那么这两个参数根本没有用到，         * 
*           如果两个都指定为不为0的值，那么会与读入图片的尺寸进行比较，如果不相等，   *
*           会按照指定的大小对图像大小进行重排，这样就可以得到期待大小的图片尺寸      *
**************************************************************************************/
image load_image(char *filename, int w, int h, int c)
{
	// 如果在makefile中使用了opencv，则调用load_image_cv加载图片
	// 当图片错误打不开或图片路径有误等原因导致不能正确加载图片时，out的值会是一个data域
	// 为10*10*3大小的全为0的image类型
#ifdef OPENCV
    image out = load_image_cv(filename, c);
#else
	// load_image_stb()将调用开源的用C语言编写的读入图片的函数（此函数非常复杂），
    // 并完成数据类型以及图片灰度存储格式的转换
    // 输出的out灰度值被归一化到0~1之间，只有一行，且按照rrr...ggg...bbb...方式存储（如果是3通道）
	// load_image_stb函数：此文件。
    image out = load_image_stb(filename, c);
#endif

    // 比较读入图片的尺寸是否与期望尺寸相等，如不等，调用resize_image函数按指定尺寸重排
	// 当h和w为0时，就不调整图片大小
    if((h && w) && (h != out.h || w != out.w)){
		//resize_image函数：本文件。按照w和h的大小重排图片的尺寸
        image resized = resize_image(out, w, h);
        free_image(out);
        out = resized;
    }
    return out;
}



/****************************************************************
*  func：读取彩色（三通道图片），返回图片的image类型信息        *
*  args filename:欲读取的图片全称                               *
*  args w: 图片的width                                          *
*  args h: 图片的height                                         *
*  details：调用load_image函数读取具体图片                      *
****************************************************************/
image load_image_color(char *filename, int w, int h)
{
	// 调用load_image()函数，读入图片，该函数视是否使用opencv调用不同的函数读入图片
    // 最后一个参数3指定通道数为3（彩色图）
    return load_image(filename, w, h, 3);
}



image get_image_layer(image m, int l)
{
    image out = make_image(m.w, m.h, 1);
    int i;
    for(i = 0; i < m.h*m.w; ++i){
        out.data[i] = m.data[i+l*m.h*m.w];
    }
    return out;
}
void print_image(image m)
{
    int i, j, k;
    for(i =0 ; i < m.c; ++i){
        for(j =0 ; j < m.h; ++j){
            for(k = 0; k < m.w; ++k){
                printf("%.2lf, ", m.data[i*m.h*m.w + j*m.w + k]);
                if(k > 30) break;
            }
            printf("\n");
            if(j > 30) break;
        }
        printf("\n");
    }
    printf("\n");
}

image collapse_images_vert(image *ims, int n)
{
    int color = 1;
    int border = 1;
    int h,w,c;
    w = ims[0].w;
    h = (ims[0].h + border) * n - border;
    c = ims[0].c;
    if(c != 3 || !color){
        w = (w+border)*c - border;
        c = 1;
    }

    image filters = make_image(w, h, c);
    int i,j;
    for(i = 0; i < n; ++i){
        int h_offset = i*(ims[0].h+border);
        image copy = copy_image(ims[i]);
        //normalize_image(copy);
        if(c == 3 && color){
            embed_image(copy, filters, 0, h_offset);
        }
        else{
            for(j = 0; j < copy.c; ++j){
                int w_offset = j*(ims[0].w+border);
                image layer = get_image_layer(copy, j);
                embed_image(layer, filters, w_offset, h_offset);
                free_image(layer);
            }
        }
        free_image(copy);
    }
    return filters;
} 

image collapse_images_horz(image *ims, int n)
{
    int color = 1;
    int border = 1;
    int h,w,c;
    int size = ims[0].h;
    h = size;
    w = (ims[0].w + border) * n - border;
    c = ims[0].c;
    if(c != 3 || !color){
        h = (h+border)*c - border;
        c = 1;
    }

    image filters = make_image(w, h, c);
    int i,j;
    for(i = 0; i < n; ++i){
        int w_offset = i*(size+border);
        image copy = copy_image(ims[i]);
        //normalize_image(copy);
        if(c == 3 && color){
            embed_image(copy, filters, w_offset, 0);
        }
        else{
            for(j = 0; j < copy.c; ++j){
                int h_offset = j*(size+border);
                image layer = get_image_layer(copy, j);
                embed_image(layer, filters, w_offset, h_offset);
                free_image(layer);
            }
        }
        free_image(copy);
    }
    return filters;
} 

void show_image_normalized(image im, const char *name)
{
    image c = copy_image(im);
    normalize_image(c);
    show_image(c, name, 1);
    free_image(c);
}

void show_images(image *ims, int n, char *window)
{
    image m = collapse_images_vert(ims, n);
    /*
       int w = 448;
       int h = ((float)m.h/m.w) * 448;
       if(h > 896){
       h = 896;
       w = ((float)m.w/m.h) * 896;
       }
       image sized = resize_image(m, w, h);
     */
    normalize_image(m);
    save_image(m, window);
    show_image(m, window, 1);
    free_image(m);
}

void free_image(image m)
{
    if(m.data){
        free(m.data);
    }
}
