#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "option_list.h"
#include "utils.h"

/// 最后修改时间：二〇一八年八月二十七日 18:26:47
/// 修改人:向国徽
/// 联系信息：17353226974 534395828(QQ)
/// 参考资料：https://github.com/hgpvision/darknet/blob/master/src/option_list.c

/*****************************************************************************************
*  文件类型：                                     对配置文件进行读取和解析               *
*  函数列表                                       功能简述                               *
*  list *read_data_cfg(char *filename)            解析配置文件，返回结果链表             *
*  metadata get_metadata(char *file)              解析配置文件的类别数和label组成结构体  *
*  int read_option(char *s, list *options)        解析‘=’前后内容，成功返回1否则返回0    *
*  void option_insert(list, char, char)           将‘=’前后的内容组成node插入链表        *
*  void option_unused(list *l)                    找到used标记为0的节点并输出            *
*  char *option_find(list *l, char *key)          返回‘=’后面的val值，找不到返回0        *
*  char *option_find_str(list, char, char)        返回‘=’后面val值的字符型，找不到返回def*
*  int option_find_int(list, char, int)           返回‘=’后面val值的整型，找不到返回def  *
*  int option_find_int_quiet(list, char, int)     同option_find_int函数，不输出用def信息 *
*  float option_find_float_quiet(list,char,float) 同上，返回浮点型，不提示使用了def信息  *
*  float option_find_float(list, char, float)     同上，返回浮点型，提示使用了def的信息  *
*****************************************************************************************/


/****************************************************************
*  func：读取配置文件中的内容，将配置文件按行组织成链表返回     *
*  args filename:配置文件的完整路径                             *
*  details：每次读取一行内容，找到=所在位置，将关键字(=号前面)  *
*           和内容(=号后面)组织成链表，返回：list指针，包含所有 *
*           数据信息。函数中会创建options变量，并返回其指针（若 *
*           文件打开失败，将直接退出程序，不会返空指针）        *
****************************************************************/
list *read_data_cfg(char *filename)
{
	// 创建文件指针，只读模式打开对应文件
    FILE *file = fopen(filename, "r");
	// 如果文件不存在，打印文件打开错误的信息。file_error:src/utils.c
	// 退出程序运行，exit(0)
    if(file == 0) file_error(filename);
	// line保存读取内容，以行为单位。nu记录当前行号
    char *line;
    int nu = 0;
	// make_list函数：src/list.c 创建双向链表的头节点，返回头指针
    list *options = make_list();
	// fgetl函数：src/utils.c 每次循环从file文件指针中读取一行内容，
	// 直到file所指的文件读完为止
    while((line=fgetl(file)) != 0){
        ++ nu;
		// strip函数：src/utils.c 去除line中的空格，制表符和换行符
        strip(line);
        switch(line[0]){
			// 空行、以#或；开头的行直接忽略，释放内存
            case '\0':
            case '#':
            case ';':
			// 原型: void free(void *ptr)
            // 功 能: 释放ptr指向的存储空间。被释放的空间通常被送入可用
			// 存储区池，以后可在调用malloc、realloc以及calloc函数来再分配。
                free(line);
                break;
            default:
			// read_option函数：本文件。将line中的内容处理到options所指链表
			// 中，成功返回1，失败返回0.
                if(!read_option(line, options)){
                    fprintf(stderr, "Config file error line %d, could parse: %s\n", nu, line);
                    free(line);
                }
                break;
        }
    }
    fclose(file);
    return options;
}



/****************************************************************
*  func：解析file所指向的配置文件，得到类别数和label列表，返回  *
*        metadata类型的结构体                                   *
*  args file：待解析的文件路径                                  *
****************************************************************/
metadata get_metadata(char *file)
{
	/*
	typedef struct{
		int classes;
		char **names;
	} metadata;
	*/
	// metadata 数据类型定义在include/darknet.h中
    metadata m = {0};
	// read_data_cfg函数。本文件定义
    list *options = read_data_cfg(file);
    // option_find_str函数。本文件定义
    char *name_list = option_find_str(options, "names", 0);
    if(!name_list) name_list = option_find_str(options, "labels", 0);
    if(!name_list) {
        fprintf(stderr, "No names or labels found\n");
    } else {
        m.names = get_labels(name_list);
    }
    m.classes = option_find_int(options, "classes", 2);
    free_list(options);
    return m;
}



/*****************************************************************
*  func：解析s中的内容，以=号分成两部分后调用option_insert       *
*        进行后续处理，=号前面为key，后面为对应value             *
*  args s: 待处理的字符串                                        *
*  args list: 双向链表头指针                                     *
*  details：成功则返回1，若不是正确格式则返回0                   *
*****************************************************************/
int read_option(char *s, list *options)
{
    size_t i;
    size_t len = strlen(s);
    char *val = 0;
	// 从配置（.data或者.cfg，不管是数据配置文件还是神经网络结构数据文件，其
	// 读取都需要调用这个函数）文件中读入的每行数据包括两部分，第一部分为变
	// 量名称，如learning_rate，第二部分为值，如0.01，两部分由=隔开，因此，要
	// 分别读入两部分的值，首先识别出等号，获取等号所在的指针，并将等号替换为
	// terminating null-characteristic '\0'，这样第一部分会自动识别到'\0'停止，
	// 而第二部分则从等号下一个地方开始
    for(i = 0; i < len; ++i){
        if(s[i] == '='){
			// 此语句表示将s从“=”处截断，以后s便只能读取到'\0'的位置
            s[i] = '\0';
			// val所指内容为=后面的内容，注意指针+的用法，代表指针偏移
            val = s+i+1;
            break;
        }
    }
	// 如果i==len-1，说明没有找到等号这个符号，那么就直接返回0（文件中还有一些
	// 注释，此外还有用[]括起来的字符，这些是网络层的类别或者名字，比如
	// [maxpool]表示这些是池化层的参数）
    if(i == len-1) return 0;
    char *key = s;
	// option_insert函数：本文件。将key作为options所指链表节点的key域，val作为
	// val插入链表options，采用尾插法
    option_insert(options, key, val);
    return 1;
}



/*****************************************************************
*  func：将key和val赋值给kvp节点，调用list_insert尾插法插入链表l *
*  args l:链表的头指针                                           *
*  args key:节点对应的key值                                      *
*  args val:节点对应val值                                        *
*****************************************************************/
void option_insert(list *l, char *key, char *val)
{
	// kvp数据结构定义在src/option_list.h中
	// 当指针需要访问结构体的成员变量时就必须用->的方法
    kvp *p = malloc(sizeof(kvp));
    p->key = key;
    p->val = val;
    p->used = 0;
	// list_insert函数：src/list.c 将p节点的内容组织后插入链表l中
    list_insert(l, p);
}


/*****************************************************************
*  func：找到并显示l链表中node的val域的used标志为0的节点         *
*  args l：欲操作的链表指针                                      *
*****************************************************************/
void option_unused(list *l)
{
    node *n = l->front;
    while(n){
        kvp *p = (kvp *)n->val;
		// 如果node->val->used=0，输出下列信息
        if(!p->used){
            fprintf(stderr, "Unused field: '%s = %s'\n", p->key, p->val);
        }
		// 循环判断下一个节点
        n = n->next;
    }
}



/*****************************************************************
*  func：在l所指的链表中查找key为*key的node所对应的val           *
*  args l: 链表的头指针                                          *
*  args key: 节点对应的key值                                     *
*  details：找到则返回对应的val值，找不到返回0                   *
*****************************************************************/
char *option_find(list *l, char *key)
{
	// 从头节点所指的第一个节点开始查找，l能指向front和back
	// 一个节点的值包含了一条项目的信息：键值与值两部分，
    // 比如classes=80，键值即项目名称classes，值为80，表示该数据集共包含80类物体）
    node *n = l->front;
	// 遍历l中的所有节点，找出节点值的键值等于指定键值key的节点，获取其值的值并返回
    // 这里的注释有些拗口，注意node中含有一个void* val，所以称作节点的值，
    // 而此处节点的val的具体类型为kvp*，该数据中又包含一个key*，一个val*，
    // 因此才称作节点值的值，节点值的键值
    while(n){
		// 获取该节点的值，注意存储到node中的val是void*指针类型，需要强转为kvp*
        kvp *p = (kvp *)n->val;
        if(strcmp(p->key, key) == 0){
			//找到后，使用标识置为1，返回对应val
            p->used = 1;
            return p->val;
        }
		// 如当前节点不是指定键值对应的节点，继续查找下一个节点
        n = n->next;
    }
	// 若遍历完l都没找到指定键值的节点的值的值，则返回0指针
    return 0;
}



/*****************************************************************
*  func：调用option_find函数在l所指的链表中查找key为*key的node   *
*        所对应的val,此时找到的值直接返回                        *
*  args l: 链表的头指针                                          *
*  args key: 节点对应的key值                                     *
*  args def: 默认值，如果l中没有找到对应key的值，则直接返回def   *
*  details：key=*key这个节点存在，返回对应val，否则提示并返回默  *
*           认值，C语言不像C++，声明函数时，不可以设置默认参数， *
*           但可以在调用时候，指定第三个参数def为字面字符数组，  *
*           这就等同于指定了默认的参数了，就像detectr.c中        *
*           test_detector()函数调用option_find_str()的那样       *
*****************************************************************/
char *option_find_str(list *l, char *key, char *def)
{
	// option_find函数：本文件。找到指定key节点的val
    char *v = option_find(l, key);
    if(v) return v;
	// 若没找到，v为空指针，则返回def（默认值），并在屏幕上提示使用默认值
    if(def) fprintf(stderr, "%s: Using default '%s'\n", key, def);
    return def;
}



/*****************************************************************
*  func：按给定键值key从l中查找对应的参数值，主要调用option_find *
*        函数。功能和上一个函数option_find_str()基本一样，只不过 *
*        多了一步处理，将找到的参数值转为了int之后再返回         *
*  args l: 链表的头指针                                          *
*  args key: 节点对应的key值                                     *
*  args def: val的默认值                                         *
*****************************************************************/
int option_find_int(list *l, char *key, int def)
{
    char *v = option_find(l, key);
	// 不为空，则调用atoi()函数将v转为整形并返回
    if(v) return atoi(v);
	// 若未空，说明未找到，返回默认值，并输出提示信息
    fprintf(stderr, "%s: Using default '%d'\n", key, def);
    return def;
}



/*****************************************************************
*  func：与上面的option_find_int()函数基本一样，唯一的区别就是使 *
*        用默认值时，没有在屏幕上输出Using default...字样的提示，*
*        因此叫做quiet（就是安安静静的使用默认值，没有提示）     *
*  args l: 链表的头指针                                          *
*  args key: 节点对应的key值                                     *
*  args def: val的默认值                                         *
*****************************************************************/
int option_find_int_quiet(list *l, char *key, int def)
{
    char *v = option_find(l, key);
    if(v) return atoi(v);
    return def;
}



/*****************************************************************
*  func：与下面的option_find_float()函数基本一样，唯一的区别就是 *
*        使用默认值时，没有在屏幕上输出 Using default...字样的提 *
*        示，因此叫做quiet（就是安安静静的使用默认值，没有提示） *
*  args l: 链表的头指针                                          *
*  args key: 节点对应的key值                                     *
*  args def: val的默认值                                         *
*****************************************************************/
float option_find_float_quiet(list *l, char *key, float def)
{
    char *v = option_find(l, key);
    if(v) return atof(v);
    return def;
}



/*****************************************************************
*  func：按给定键值key从l中查找对应的参数值，主要调用option_find *
*        函数功能和函数option_find_int()基本一样，只不过最后调用 *
*   	 atof()函数将字符串转为了float类型数据，而不是整型数据   *
*  args l：list指针，实际为section结构体中的options元素，包含该层*
*          神经网络的所有配置参数                                *
*  args key：键值，即参数名称，比如卷积核尺寸size，卷积核个数    *
*            filters，跨度stride等等                             *
*  args def: 默认值，如果没有找到对应键值的参数值，则作为默认值  *
*****************************************************************/
float option_find_float(list *l, char *key, float def)
{
    char *v = option_find(l, key);
    if(v) return atof(v);
	// 若未空，说明未找到，返回默认值，并输出提示信息
    fprintf(stderr, "%s: Using default '%lf'\n", key, def);
    return def;
}
