#include <stdlib.h>
#include <string.h>
#include "list.h"

/// 最后修改时间：二〇一八年八月二十四日 09:56:47
/// 修改人:向国徽
/// 联系信息：17353226974 534395828(QQ)
/// 参考资料：无

/*****************************************************************************************
*  文件类型：                                     对链表操作的自定义函数脚本             *
*  函数列表                                       功能简述                               *
*  list *make_list()                              生成链表并初始化                       *
*  void *list_pop(list *l)                        弹出链表最后一元素                     *
*  void list_insert(list *l, void *val)           将val作为node的值域插入l之中           *
*  void free_node(node *n)                        释放从n节点开始的节点                  *
*  void free_list(list *l)                        释放链表l及其所有节点                  *
*  void free_list_contents(list *l)               释放链表所有节点的val域                *
*  void **list_to_array(list *l)                  将链表所有节点的val提取成数组          *
*****************************************************************************************/



/**************************************************************
*  func：创建双向链表头节点，返回链表头指针                   *
*  details：完成头节点的初始化，节点个数为0，L的front指向第   *
*           一个节点，back指向最后一个节点，初始化指向空节点  *
**************************************************************/
list *make_list()
{
	list *l = malloc(sizeof(list));
	// l->front和l->back均为node类型，初始化不指向任何节点
	l->size = 0;
	l->front = 0;
	l->back = 0;
	return l;
}



/***********************************************************
*  func：取出链表l中的最后一个节点，并删除最后一个节点     *
*  args l: pop操作的链表头指针                             *
*  details：链表有节点时返回最后一个节点的val，没有则返回0 *
***********************************************************/
void *list_pop(list *l){
	// l->back为空，说明链表中没有任何节点
    if(!l->back) return 0;
    node *b = l->back;
    void *val = b->val;
    l->back = b->prev;
    if(l->back) l->back->next = 0;
    free(b);
	// 弹出成功后，链表l的长度减1
    --l->size;   
    return val;
}



/**************************************************************
*  func：开辟新节点node，将val作为节点node的val域，以尾插法   *
*        插入双向链表l中，l作为引用传递改变被保留             *
*  args l: 双向链表头指针                                     *
*  args val: node节点的val域值，数据类型灵活变换              *
*  details：每次插入完成后，l的节点数量会增1                  *
**************************************************************/
void list_insert(list *l, void *val)
{
	// node数据结构定义在include/darknet.h中
	node *new = malloc(sizeof(node));
	new->val = val;
	new->next = 0;
	// 当链表中没有节点时，第一个插入节点需要特殊处理
	if(!l->back){
		l->front = new;
		new->prev = 0;
	}else{
		l->back->next = new;
		new->prev = l->back;
	}
	l->back = new;
	++l->size;
}



/**************************************************
*  func：释放链表中的所有节点                     *
*  args n: 开始释放的节点指针                     *
*  details：可以从链表的中间节点开始释放，据n而定 *
**************************************************/
void free_node(node *n)
{
	node *next;
	// 循环释放节点
	while(n) {
		next = n->next;
		// free函数为系统库函数，释放由malloc或alloc等动态
		// 申请的内存等空间
		free(n);
		n = next;
	}
}



/**********************************************
*  func：释放链表l，调用free_node释放所有节点 *
*  args l: 欲释放的链表头指针                 *
**********************************************/
void free_list(list *l)
{
	free_node(l->front);
	free(l);
}



/*******************************************
*  func：释放链表l中所有node val域的空间   *
*  args l: 欲操作的链表头指针              *
*******************************************/
void free_list_contents(list *l)
{
	node *n = l->front;
	while(n){
		free(n->val);
		n = n->next;
	}
}



/***********************************************
*  func：将链表l中的节点的val提取组织成数组    *
*  args l: 欲提取的链表头指针                  *
*  details：node的val域类型为void，根据实际而定*
***********************************************/
void **list_to_array(list *l)
{
	// calloc申请的空间的值会被初始化为0
    void **a = calloc(l->size, sizeof(void*));
    int count = 0;
    node *n = l->front;
    while(n){
        a[count++] = n->val;
        n = n->next;
    }
    return a;
}
