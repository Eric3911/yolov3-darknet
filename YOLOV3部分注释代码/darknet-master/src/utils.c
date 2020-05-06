#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <float.h>
#include <limits.h>
#include <time.h>
#include <sys/time.h>

#include "utils.h"

/// 最后修改时间：二〇一八年八月二十四日 09:56:47
/// 修改人:向国徽
/// 联系信息：17353226974 534395828(QQ)
/// 参考资料：无

/****************************************************************
*  func：获取系统的当前时间，转化为以秒为单位并返回             *
****************************************************************/
double what_time_is_it_now()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}



int *read_intlist(char *gpu_list, int *ngpus, int d)
{
    int *gpus = 0;
    if(gpu_list){
        int len = strlen(gpu_list);
        *ngpus = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (gpu_list[i] == ',') ++*ngpus;
        }
        gpus = calloc(*ngpus, sizeof(int));
        for(i = 0; i < *ngpus; ++i){
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',')+1;
        }
    } else {
        gpus = calloc(1, sizeof(float));
        *gpus = d;
        *ngpus = 1;
    }
    return gpus;
}



int *read_map(char *filename)
{
    int n = 0;
    int *map = 0;
    char *str;
    FILE *file = fopen(filename, "r");
    if(!file) file_error(filename);
    while((str=fgetl(file))){
        ++n;
        map = realloc(map, n*sizeof(int));
        map[n-1] = atoi(str);
    }
    return map;
}



void sorta_shuffle(void *arr, size_t n, size_t size, size_t sections)
{
    size_t i;
    for(i = 0; i < sections; ++i){
        size_t start = n*i/sections;
        size_t end = n*(i+1)/sections;
        size_t num = end-start;
        shuffle(arr+(start*size), num, size);
    }
}



void shuffle(void *arr, size_t n, size_t size)
{
    size_t i;
    void *swp = calloc(1, size);
    for(i = 0; i < n-1; ++i){
        size_t j = i + rand()/(RAND_MAX / (n-i)+1);
        memcpy(swp,          arr+(j*size), size);
        memcpy(arr+(j*size), arr+(i*size), size);
        memcpy(arr+(i*size), swp,          size);
    }
}



int *random_index_order(int min, int max)
{
    int *inds = calloc(max-min, sizeof(int));
    int i;
    for(i = min; i < max; ++i){
        inds[i] = i;
    }
    for(i = min; i < max-1; ++i){
        int swap = inds[i];
        int index = i + rand()%(max-i);
        inds[i] = inds[index];
        inds[index] = swap;
    }
    return inds;
}



/****************************************************************
*  func：删除argv数组中指定索引为index的元素                    *
*  args argc：元素个数                                          *
*  args argv: 元素列表，数组名                                  *
*  args index: 需要删除元素的索引                               *
*  details：删除即是将索引所指元素后面的所有元素依次前移        *
****************************************************************/
void del_arg(int argc, char **argv, int index)
{
    int i;
	//循环前移数组元素实现删除，最后一位元素置0
    for(i = index; i < argc-1; ++i) argv[i] = argv[i+1];
    argv[i] = 0;
}



/******************************************************************
*  func：判断数组argv中是否存在参数arg，存在返回1并删除，否则返回0*
*  args argc：参数个数                                            *
*  args argv: 参数列表                                            *
*  args arg: 需要判断的参数名称                                   *
*  details：此函数只是判断是否存在该参数，不读取其后对应的值      *
******************************************************************/
int find_arg(int argc, char* argv[], char *arg)
{
    int i;
    for(i = 0; i < argc; ++i) {
        if(!argv[i]) continue;
        if(0==strcmp(argv[i], arg)) {
			// 找到arg参数后从argv中删除，并返回1
            del_arg(argc, argv, i);
            return 1;
        }
    }
    return 0;
}



/**********************************************************************
*  func：找到参数arg所对应的整型值，如：./darknet detect -i 3,当*arg  *
*        为-i时，则返回的是3，找到返回找到的值，没有则返回默认值def   *
*  args argc：参数个数                                                *
*  args argv: 参数列表                                                *
*  args arg: 需要找的参数                                             *
*  args def: 参数的默认值，通过函数调用传递过来                       *
*  detail：将查找到的参数值转化为整型，找到后会删除arg及其对应值      *
**********************************************************************/
int find_int_arg(int argc, char **argv, char *arg, int def)
{
    int i;
    for(i = 0; i < argc-1; ++i){
        if(!argv[i]) continue;
		// 查找arg所在的数组索引，找到后再取值和删除
        if(0==strcmp(argv[i], arg)){
			// atoi:ascii_to_int的缩写，函数将字符串转化为整型
            def = atoi(argv[i+1]);
			// del_arg函数：此文件 从argv中删除arg和其对应值
            del_arg(argc, argv, i);
            del_arg(argc, argv, i);
            break;
        }
    }
    return def;
}



/**********************************************************************
*  func：找到参数arg所对应的浮点型值并返回，没有则返回默认值def       *
*  args argc：参数个数                                                *
*  args argv: 参数列表                                                *
*  args arg: 需要找的参数                                             *
*  args def: 参数的默认值，通过函数调用传递过来                       *
*  details：将查找到的参数值转化为浮点型，找到后会删除arg及其对应值   *
**********************************************************************/
float find_float_arg(int argc, char **argv, char *arg, float def)
{
    int i;
	// 其它注释见此文档的find_int_arg函数
    for(i = 0; i < argc-1; ++i){
        if(!argv[i]) continue;
        if(0==strcmp(argv[i], arg)){
			// atof函数：ascii_to_float缩写，望文生义
            def = atof(argv[i+1]);
            del_arg(argc, argv, i);
            del_arg(argc, argv, i);
            break;
        }
    }
    return def;
}



/**********************************************************************
*  func：找到参数arg所对应的字符型值并返回，没有则返回默认值def       *
*  args argc：参数个数                                                *
*  args argv: 参数列表                                                *
*  args arg: 需要找的参数                                             *
*  args def: 参数的默认值，通过函数调用传递过来                       *
*  details：将查找到的参数值直接返回，找到后会删除arg及其对应值       *
**********************************************************************/
char *find_char_arg(int argc, char **argv, char *arg, char *def)
{
    int i;
	// 详细注释和上述三函数类似
    for(i = 0; i < argc-1; ++i){
        if(!argv[i]) continue;
        if(0==strcmp(argv[i], arg)){
            def = argv[i+1];
            del_arg(argc, argv, i);
            del_arg(argc, argv, i);
            break;
        }
    }
    return def;
}



/**********************************************************************
*  func：从文件全路径字符串cfgfile中提取主要信息，比如从cfg/yolo.cfg  *
*        中，提取出yolo，返回提取的字符串                             *
*  args cfgfile：欲提取的字符串路径                                   *
*  details：函数中主要用到strchr()函数定位'/'以及'.'符号，此函数没有  *
*           实质用处，一般用来提取字符串，以打印出主要信息（比如      *
*           cfg/yolov3.cfg，主要信息就是yolov3，表示是yolov3网络）    *
**********************************************************************/
char *basecfg(char *cfgfile)
{
    char *c = cfgfile;
    char *next;
	// strchr是计算机编程语言的一个函数，原型为extern char *strchr(const char *s,char c)，
	// 可以查找字符串s中首次出现字符c的位置。
	// 定位到'/'字符，让c等于'/'之后的字符，丢掉之前的字符，
    // 比如c='cfg/yolo.cfg'->c='yolo.cfg'
    while((next = strchr(c, '/')))
    {
        c = next+1;
    }
	// copy_string(c)不会改变c的值，但在函数内会重新给c分配一段地址，使得c与cfgfile不再关联，
    // 这样，下面改动c也不会影响cfgfile的值
    c = copy_string(c);
    next = strchr(c, '.');
	// 接着上面的例子，此时c="yolo.cfg"，为了提取yolo，需要有标识符隔开yolo与cfg，
    // 因此需要识别出'.'，并将其代替为'\0'（'\0'的ASCII值就为0），这样，就可以隔开两段内容，
    // 且前一段为c风格字符串，会被系统自动识别断开（严格来说不是真正的提取出来，而只是将两段分开而已，这里理解就行）
    if (next) *next = 0;
    return c;
}



/*********************************************************
*  func：将字符c转化为其对应的整型                       *
*********************************************************/
int alphanum_to_int(char c)
{
    return (c < 58) ? c - 48 : c-87;
}
char int_to_alphanum(int i)
{
    if (i == 36) return '.';
    return (i < 10) ? i + 48 : i + 87;
}

void pm(int M, int N, float *A)
{
    int i,j;
    for(i =0 ; i < M; ++i){
        printf("%d ", i+1);
        for(j = 0; j < N; ++j){
            printf("%2.4f, ", A[i*N+j]);
        }
        printf("\n");
    }
    printf("\n");
}



/*********************************************************************************************
*  func：在字符串str中查找指定字符串orig，如果没有找到，则直接令output等于str输出（此时本    *
*        函数相当于没有执行），如果找到了，即orig是str字符串的一部分，那么用rep替换掉str中的 *
*        orig，然后再赋给output返回                                                          *
*  args str：原始字符串                                                                      *
*  args orig: 查找字符串                                                                     *
*  args rep: 替代字符串                                                                      *
*  args output: 输出字符串                                                                   *
*  details: 在读入训练数据时，只给程序输入了图片所在路径，而标签数据的路径并没有直接给，是通 *
*           过对图片路径进行修改得到的，比如在训练voc数据时，输入的train.txt文件中只包含所有 *
*           图片的具体路径，如：                                                             *
*           /home/xgh/Downloads/darknet_dataset/VOCdevkit/VOC2007/JPEGImages/000001.jpg      *
*           而000001.jpg的标签并没有给程序，是通过该函数替换掉图片路径中的JPEGImages为labels,*
*           并替换掉后缀.jpg为.txt得到的，最终得到：                                         *
*           /home/xgh/Downloads/darknet_dataset/VOCdevkit/VOC2007/labels/000001.txt          *
*           这种替换的前提是，标签数据文件夹labels与图片数据文件夹JPEGImages具有相同的父目录 *
*********************************************************************************************/
void find_replace(char *str, char *orig, char *rep, char *output)
{
    char buffer[4096] = {0};
    char *p;

    sprintf(buffer, "%s", str);
	// strstr()用来判断orig是否是buffer字符串的一部分
    // 如果orig不是buffer的字串，则返回的p为NULL指针；如果是buffer的字串，
    // 则返回orig在buffer的首地址（char *类型，注意是buffer中的首地址）
    if(!(p = strstr(buffer, orig))){  // Is 'orig' even in 'str'?
	    // 如果不是，则直接将str赋给output返回，此时本函数相当于什么也没做
        sprintf(output, "%s", str);
        return;
    }
    *p = '\0';
    // 运行到这，说明orig是buffer的字串，则首先把orig字符串抹掉，具体做法是将p处（p是orig在buffer的首地址）的字符置为c风格
    // 字符串终止符'\0'，而后，再往buffer添加待替换的字符串rep（会自动识别到'\0'处），以及原本buffer中orig之后的字符串
    sprintf(output, "%s%s%s", buffer, rep, p+strlen(orig));
}




float sec(clock_t clocks)
{
    return (float)clocks/CLOCKS_PER_SEC;
}

void top_k(float *a, int n, int k, int *index)
{
    int i,j;
    for(j = 0; j < k; ++j) index[j] = -1;
    for(i = 0; i < n; ++i){
        int curr = i;
        for(j = 0; j < k; ++j){
            if((index[j] < 0) || a[curr] > a[index[j]]){
                int swap = curr;
                curr = index[j];
                index[j] = swap;
            }
        }
    }
}



/******************************************************************************
*  func：输出错误信息并调用assert函数中断程序运行                             *
*  args s：想要显示的错误信息                                                 *
******************************************************************************/
void error(const char *s)
{
	// C 库函数 void perror(const char *str) 把一个描述性错误消息输出到标准错误
    // stderr。首先输出字符串 str，后跟一个冒号，然后是一个空格.
    perror(s);
	// 1. 捕捉逻辑错误。可以在程序逻辑必须为真的条件上设置断言。除非发生逻辑错误，
	// 否则断言对程序无任何影响。即预防性的错误检查，在认为不可能的执行到的情况
	// 下加一句ASSERT(0)，如果运行到此，代码逻辑或条件就可能有问题。 
    // 2. 程序没写完的标识，放个assert(0)调试运行时执行到此为报错中断，好知道成员函数还没写完。
    assert(0);
    exit(-1);
}




unsigned char *read_file(char *filename)
{
    FILE *fp = fopen(filename, "rb");
    size_t size;

    fseek(fp, 0, SEEK_END); 
    size = ftell(fp);
    fseek(fp, 0, SEEK_SET); 

    unsigned char *text = calloc(size+1, sizeof(char));
    fread(text, 1, size, fp);
    fclose(fp);
    return text;
}



/****************************************************************
*  func：输出内存分配错误的报错信息                             *
*  details：输出错误信息后会退出程序exit(-1)                    *
****************************************************************/
void malloc_error()
{
    fprintf(stderr, "Malloc error\n");
    exit(-1);
}


/****************************************************************
*  func：输出XX文件不能打开的报错信息                           *
*  args s: 具体文件名                                           *
*  details：输出错误信息后会退出程序exit(0)                     *
****************************************************************/
void file_error(char *s)
{
    fprintf(stderr, "Couldn't open file: %s\n", s);
    exit(0);
}



/******************************************************************
*  func：将字符串s按字符delim进行切分后作为节点放入链表中         *
*  args s: 待处理的字符串                                         *
*  args delim：切割字符
*  details：扫描s，每遇到待切割字符则计数，并将后续元素前移       *
******************************************************************/
list *split_str(char *s, char delim)
{
    size_t i;
    size_t len = strlen(s);
    list *l = make_list();
    list_insert(l, s);
    for(i = 0; i < len; ++i){
        if(s[i] == delim){
            s[i] = '\0';
            list_insert(l, &(s[i+1]));
        }
    }
    return l;
}



/******************************************************************
*  func：去除字符串中的空格制表和换行符，去除的字符串通过指针修改 *
*  args s: 待处理的字符串                                         *
*  details：扫描s，每遇到待去除字符则计数，并将后续元素前移       *
******************************************************************/
void strip(char *s)
{
    size_t i;
    size_t len = strlen(s);
    size_t offset = 0;
	// offset为要剔除的字符数，比如offset=2，说明到此时需要剔除2个空白符，
    // 剔除完两个空白符之后，后面的要往前补上，不能留空
    for(i = 0; i < len; ++i){
        char c = s[i];
        if(c==' '||c=='\t'||c=='\n') ++offset;
        else s[i-offset] = c;
    }
	//修改前移后字符数组的结束符位置
    s[len-offset] = '\0';
}



/******************************************************************
*  func：去除字符串中的bad字符，去除的字符串通过指针修改          *
*  args s: 待处理的字符串                                         *
*  args bad：待去除的字符                                         *
*  details：扫描s，每遇到待去除字符则计数，并将后续元素前移       *
******************************************************************/
void strip_char(char *s, char bad)
{
    size_t i;
    size_t len = strlen(s);
    size_t offset = 0;
	// 详细操作细节同函数void strip(char *s)
    for(i = 0; i < len; ++i){
        char c = s[i];
        if(c==bad) ++offset;
        else s[i-offset] = c;
    }
    s[len-offset] = '\0';
}



/******************99999999999*************************************
*  func：去除字符串中的空格制表和换行符，去除的字符串通过指针修改 *
*  args s: 待处理的字符串                                         *
*  details：扫描s，每遇到待去除字符则计数，并将后续元素前移       *
******************************************************************/
void free_ptrs(void **ptrs, int n)
{
    int i;
    for(i = 0; i < n; ++i) free(ptrs[i]);
    free(ptrs);
}




/******************************************************************
*  func：从文件fp中读取一行内容，返回读取到的内容其读取的结果中不 *
*        再有换行符或者eof，这两个符号在返回之前被处理掉了        *
*  args fp: 操作文件的指针                                        *
*  details：每次读取一行，读取的内容通过字符数组返回，动态管理大小*
******************************************************************/
char *fgetl(FILE *fp)
{
	// feof函数：feof是C语言标准库函数，其原型在stdio.h中，其功能是
	// 检测流上的文件结束符，如果文件结束，则返回非0值，否则返回0，
	// 文件结束符只能被clearerr()清除。
    if(feof(fp)) return 0;
	// size_t 类型定义在cstddef头文件中，该文件是C标准库的头文件
	// stddef.h的C++版。它是一个与机器相关的unsigned类型，其大小足
	// 以保证存储内存中对象的大小。它是sizeof操作符返回的结果类型，
    // 该类型的大小可选择。
	// 默认一行的字符数目最大为512，如果不够，下面会有应对方法
    size_t size = 512;
	// 开辟size大小的一块内存空间
    char *line = malloc(size*sizeof(char));
	// fgets函数：char *fgets(char *buf, int bufsize, FILE *stream);
	// 具体说明：从文件结构体指针stream中读取数据，每次读取一行。读
	// 取的数据保存在buf指向的字符数组中，每次最多读取bufsize-1个字符
	// （第bufsize个字符赋'\0'），如果文件中的该行，不足bufsize-1个字
	// 符，则读完该行就结束。如若该行（包括最后一个换行符）的字符数超
	// 过bufsize-1，则fgets只返回一个不完整的行，但是，缓冲区总是以NULL
	// 字符结尾，对fgets的下一次调用会继续读该行。函数成功将返回buf，失
	// 败或读到文件结尾返回NULL。因此我们不能直接通过fgets的返回值来判
	// 断函数是否是出错而终止的，应该借助feof函数或者ferror函数来判断。
    if(!fgets(line, size, fp)){
        free(line);
        return 0;
    }

    size_t curr = strlen(line);
    // line[curr-1]！=’\n‘则说明上一次的一行还未读完，则继续读
	// 终止符（换行符号以及eof）也会存储到str中，所以可用作while终止条件
    // 如果一整行数据顺利读入到line中，那么line[curr-1]应该会是换行符或者eof，
    // 这样就可以绕过while循环中的处理；否则说明这行数据未读完，line空间不够，需要重新分配，重新读取
    while((line[curr-1] != '\n') && !feof(fp)){
        if(curr == size-1){
            size *= 2;
			// 扩展先前开辟的空间
            line = realloc(line, size*sizeof(char));
			// 如果动态分配内存失败，就会返回空指针，注意防护
            if(!line) {
                printf("%ld\n", size);
                malloc_error();
            }
        }
		
		// 之前因为line空间不够，没有读完一整行，此处不是从头开始读，
        // 而是接着往下读，并接着往下存（fgets会记着上一次停止读数据的地方）
        size_t readsize = size-curr;
        if(readsize > INT_MAX) readsize = INT_MAX-1;
        fgets(&line[curr], readsize, fp);
        curr = strlen(line);
    }
	//将行尾换行符替换为字符串结束符
    if(line[curr-1] == '\n') line[curr-1] = '\0';

    return line;
}




int read_int(int fd)
{
    int n = 0;
    int next = read(fd, &n, sizeof(int));
    if(next <= 0) return -1;
    return n;
}

void write_int(int fd, int n)
{
    int next = write(fd, &n, sizeof(int));
    if(next <= 0) error("read failed");
}

int read_all_fail(int fd, char *buffer, size_t bytes)
{
    size_t n = 0;
    while(n < bytes){
        int next = read(fd, buffer + n, bytes-n);
        if(next <= 0) return 1;
        n += next;
    }
    return 0;
}

int write_all_fail(int fd, char *buffer, size_t bytes)
{
    size_t n = 0;
    while(n < bytes){
        size_t next = write(fd, buffer + n, bytes-n);
        if(next <= 0) return 1;
        n += next;
    }
    return 0;
}

void read_all(int fd, char *buffer, size_t bytes)
{
    size_t n = 0;
    while(n < bytes){
        int next = read(fd, buffer + n, bytes-n);
        if(next <= 0) error("read failed");
        n += next;
    }
}

void write_all(int fd, char *buffer, size_t bytes)
{
    size_t n = 0;
    while(n < bytes){
        size_t next = write(fd, buffer + n, bytes-n);
        if(next <= 0) error("write failed");
        n += next;
    }
}



/*********************************************************
*  func：将字符串s中的字符连同'/0'一并复制到copy中       *
*  args s: 欲复制的字符串                                *
*  details：调用strncpy库函数实现复制，返回复制的字符串  *
*********************************************************/
char *copy_string(char *s)
{
	// 拷贝后s与copy指向不同的地址，二者再无瓜葛
	// strlen(s)+1是想连同'/0'一起复制
	// strncpy是c标准库函数，定义在string.h中，表示将首地址为s的strlen(s)+1个字符拷贝至copy中，
    // 之所以加1是因为c风格字符串默认以'\0'结尾，但并不记录strlen统计中，如果不将'\0'拷贝过去，
    // 那么得到的copy不会自动以'\0'结尾（除非是这种赋值格式：
	// char *c="happy"，直接以字符串初始化的字符数组会自动添加'\0'）,
    // 就不是严格的c风格字符串（注意：c字符数组不一定要以'\0'结尾，但c风格字符串需要）
    char *copy = malloc(strlen(s)+1);
    strncpy(copy, s, strlen(s)+1);
    return copy;
}




list *parse_csv_line(char *line)
{
    list *l = make_list();
    char *c, *p;
    int in = 0;
    for(c = line, p = line; *c != '\0'; ++c){
        if(*c == '"') in = !in;
        else if(*c == ',' && !in){
            *c = '\0';
            list_insert(l, copy_string(p));
            p = c+1;
        }
    }
    list_insert(l, copy_string(p));
    return l;
}

int count_fields(char *line)
{
    int count = 0;
    int done = 0;
    char *c;
    for(c = line; !done; ++c){
        done = (*c == '\0');
        if(*c == ',' || done) ++count;
    }
    return count;
}

float *parse_fields(char *line, int n)
{
    float *field = calloc(n, sizeof(float));
    char *c, *p, *end;
    int count = 0;
    int done = 0;
    for(c = line, p = line; !done; ++c){
        done = (*c == '\0');
        if(*c == ',' || done){
            *c = '\0';
            field[count] = strtod(p, &end);
            if(p == c) field[count] = nan("");
            if(end != c && (end != c-1 || *end != '\r')) field[count] = nan(""); //DOS file formats!
            p = c+1;
            ++count;
        }
    }
    return field;
}



/************************************************************
*  func：计算数组a中所有元素的和，返回计算结果              *
*  args a：一维数组                                         *
*  args n：a中元素的个数                                    *
************************************************************/
float sum_array(float *a, int n)
{
    int i;
    float sum = 0;
    for(i = 0; i < n; ++i) sum += a[i];
    return sum;
}



/************************************************************
*  func：计算数组a中所有元素的均值，返回计算结果            *
*  args a：一维数组                                         *
*  args n：a中元素的个数                                    *
************************************************************/
float mean_array(float *a, int n)
{
    return sum_array(a,n)/n;
}



/************************************************************
*  func：计算二维数组a中各列元素的均值，指针回传计算结果    *
*  args a：二维数组                                         *
*  args n：a中元素的行数                                    *
*  args els：a中元素的列数                                  *
*  args avg：均值结果的指针                                 *
************************************************************/
void mean_arrays(float **a, int n, int els, float *avg)
{
    int i;
    int j;
    memset(avg, 0, els*sizeof(float));
	// 注意理解这种计算方式的表达式
    for(j = 0; j < n; ++j){
        for(i = 0; i < els; ++i){
            avg[i] += a[j][i];
        }
    }
    for(i = 0; i < els; ++i){
        avg[i] /= n;
    }
}



/************************************************************
*  func：输出数组a的统计信息：均值和方差                    *
*  args a：一维数组                                         *
*  args n：a中元素的个数                                    *
************************************************************/
void print_statistics(float *a, int n)
{
    float m = mean_array(a, n);
    float v = variance_array(a, n);
    printf("MSE: %.6f, Mean: %.6f, Variance: %.6f\n", mse_array(a, n), m, v);
}



/************************************************************
*  func：计算数组a中所有元素的方差，返回计算结果            *
*  args a：一维数组                                         *
*  args n：a中元素的个数                                    *
************************************************************/
float variance_array(float *a, int n)
{
    int i;
    float sum = 0;
    float mean = mean_array(a, n);
    for(i = 0; i < n; ++i) sum += (a[i] - mean)*(a[i]-mean);
    float variance = sum/n;
    return variance;
}



/************************************************************
*  func：严格限制输入a的值在min~max之间：对a进行边界值检查，*
*        如果小于min或者大于max，都直接置为边界值           *
*  details：返回a的值(若越界，则会改变，若未越界，保持不变) *
************************************************************/
int constrain_int(int a, int min, int max)
{
    if (a < min) return min;
    if (a > max) return max;
    return a;
}



/************************************************************
*  func：严格限制输入a的值在min~max之间：对a进行边界值检查，*
*        如果小于min或者大于max，都直接置为边界值           *
*  details：返回a的值(若越界，则会改变，若未越界，保持不变) *
************************************************************/
float constrain(float min, float max, float a)
{
    if (a < min) return min;
    if (a > max) return max;
    return a;
}



/************************************************************
*  func：计算数组a中元素与数组b中元素的距离，欧式距离       *
*  args a：一维数组                                         *
*  args b：一维数组                                         *
*  args n：a，b中元素的个数                                 *
*  args sub：每次跳过元素的个数                             *
************************************************************/
float dist_array(float *a, float *b, int n, int sub)
{
    int i;
    float sum = 0;
    for(i = 0; i < n; i += sub) sum += pow(a[i]-b[i], 2);
    return sqrt(sum);
}



/************************************************************
*  func：计算数组a中所有元素平方的均值再开方，返回计算结果  *
*  args a：一维数组                                         *
*  args n：a中元素的个数                                    * 
************************************************************/
float mse_array(float *a, int n)
{
    int i;
    float sum = 0;
    for(i = 0; i < n; ++i) sum += a[i]*a[i];
    return sqrt(sum/n);
}



/************************************************************
*  func：将数组a中所有元素标准化，减去均值，除以标准差      *
*  args a：一维数组                                         *
*  args n：a中元素的个数                                    *
************************************************************/
void normalize_array(float *a, int n)
{
    int i;
    float mu = mean_array(a,n);
	// 方差开方得到标准差
    float sigma = sqrt(variance_array(a,n));
    for(i = 0; i < n; ++i){
        a[i] = (a[i] - mu)/sigma;
    }
    mu = mean_array(a,n);
    sigma = sqrt(variance_array(a,n));
}



/************************************************************
*  func：将数组a中所有元素加上s，结果通过指针改变           *
*  args a：一维数组                                         *
*  args n：a中元素的个数                                    *
************************************************************/
void translate_array(float *a, int n, float s)
{
    int i;
    for(i = 0; i < n; ++i){
        a[i] += s;
    }
}



/************************************************************
*  func：计算数组a中所有元素的平方和再开方，返回计算结果    *
*  args a：一维数组                                         *
*  args n：a中元素的个数                                    *
************************************************************/
float mag_array(float *a, int n)
{
    int i;
    float sum = 0;
    for(i = 0; i < n; ++i){
        sum += a[i]*a[i];   
    }
    return sqrt(sum);
}



/************************************************************
*  func：放缩数组a中所有元素，所有元素乘以比例s             *
*  args a：一维数组                                         *
*  args n：a中元素的个数                                    *
*  args s：放缩比例s                                        *
************************************************************/
void scale_array(float *a, int n, float s)
{
    int i;
    for(i = 0; i < n; ++i){
        a[i] *= s;
    }
}




int sample_array(float *a, int n)
{
    float sum = sum_array(a, n);
    scale_array(a, n, 1./sum);
    float r = rand_uniform(0, 1);
    int i;
    for(i = 0; i < n; ++i){
        r = r - a[i];
        if (r <= 0) return i;
    }
    return n-1;
}

/***********************************************************************
*  func：找出整型数组a中的最大元素，返回其索引值                       *
*  args a：一维数组，比如检测模型中，可以是包含属于各类概率的数组      *
*         （数组中最大元素即为物体最有可能所属的类别）                 *
*  args n：a中元素的个数，比如检测模型中，a中包含所有的物体类别，      *
*          此时n为物体类别总数                                         *
***********************************************************************/
int max_int_index(int *a, int n)
{
	// 如果a为空，返回-1
    if(n <= 0) return -1;
    int i, max_i = 0;
    int max = a[0];
	// max_i为最大元素的索引，初始为第一个元素，而后遍历整个数组，找出最大的元素，返回其索引值
    for(i = 1; i < n; ++i){
        if(a[i] > max){
            max = a[i];
            max_i = i;
        }
    }
    return max_i;
}



/***********************************************************************
*  func：找出浮点型数组a中的最大元素，返回其索引值                     *
*  args a：一维数组，比如检测模型中，可以是包含属于各类概率的数组      *
*         （数组中最大元素即为物体最有可能所属的类别）                 *
*  args n：a中元素的个数，比如检测模型中，a中包含所有的物体类别，      *
*          此时n为物体类别总数                                         *
***********************************************************************/
int max_index(float *a, int n)
{
	// 其他注释同上
    if(n <= 0) return -1;
    int i, max_i = 0;
    float max = a[0];
    for(i = 1; i < n; ++i){
        if(a[i] > max){
            max = a[i];
            max_i = i;
        }
    }
    return max_i;
}



/***********************************************************************
*  func：找出整型数组a中的元素val所在的位置，返回其索引值              *
*  args a：一维数组                                                    *
*  args val：欲查找的元素                                              *
*  args n：a中元素的个数                                               *
***********************************************************************/
int int_index(int *a, int val, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        if(a[i] == val) return i;
    }
    return -1;
}



/***********************************************************************
*  func：生成min到max之间的整型随机数，返回生成的数                    *
*  args min：生成随机数的下限                                          *
*  args max：生成随机数的上限                                          *
*  details：如果max<min的话先交换值，保证空间正确                      *
***********************************************************************/
int rand_int(int min, int max)
{
    if (max < min){
        int s = min;
        min = max;
        max = s;
    }
    int r = (rand()%(max - min + 1)) + min;
    return r;
}




// From http://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
/****************************************************************
*  func：生成标准正态分布随机数（float）并返回                  *
****************************************************************/
float rand_normal()
{
    static int haveSpare = 0;
    static double rand1, rand2;

	// z0和z1都用了，并不是只用z0或只用z1
    if(haveSpare)
    {
        haveSpare = 0;
        return sqrt(rand1) * sin(rand2);
    }

    haveSpare = 1;

	// 产生0~1的随机数
    rand1 = rand() / ((double) RAND_MAX);
    if(rand1 < 1e-100) rand1 = 1e-100;
    rand1 = -2 * log(rand1);
	// 产生0~2*PI之间的随机数
    rand2 = (rand() / ((double) RAND_MAX)) * TWO_PI;

    return sqrt(rand1) * cos(rand2);
}

/*
   float rand_normal()
   {
   int n = 12;
   int i;
   float sum= 0;
   for(i = 0; i < n; ++i) sum += (float)rand()/RAND_MAX;
   return sum-n/2.;
   }
 */

size_t rand_size_t()
{
    return  ((size_t)(rand()&0xff) << 56) | 
        ((size_t)(rand()&0xff) << 48) |
        ((size_t)(rand()&0xff) << 40) |
        ((size_t)(rand()&0xff) << 32) |
        ((size_t)(rand()&0xff) << 24) |
        ((size_t)(rand()&0xff) << 16) |
        ((size_t)(rand()&0xff) << 8) |
        ((size_t)(rand()&0xff) << 0);
}



/***********************************************************************
*  func：生成一个服从min到max之间均匀分布的数                          *
*  args min：生成随机数的下限                                          *
*  args max：生成随机数的上限                                          *
*  details：如果max<min的话先交换值，保证空间正确                      *
***********************************************************************/
float rand_uniform(float min, float max)
{
    if(max < min){
        float swap = min;
        min = max;
        max = swap;
    }
	// RAND_MAX [1]  指的是 C 语言标准库 <stdlib.h> 中定义的一个宏。
	// 经预编译阶段处理后，它展开为一个整数类型的常量表达式。RAND_MAX
    // 是 <stdlib.h> 中伪随机数生成函数 rand 所能返回的最大数值。
    return ((float)rand()/RAND_MAX * (max - min)) + min;
}



/********************************************
*  func: 得到1-s之间均匀分布的值或其倒数值  *
********************************************/
float rand_scale(float s)
{
    float scale = rand_uniform(1, s);
    if(rand()%2) return scale;
    return 1./scale;
}

float **one_hot_encode(float *a, int n, int k)
{
    int i;
    float **t = calloc(n, sizeof(float*));
    for(i = 0; i < n; ++i){
        t[i] = calloc(k, sizeof(float));
        int index = (int)a[i];
        t[i][index] = 1;
    }
    return t;
}

