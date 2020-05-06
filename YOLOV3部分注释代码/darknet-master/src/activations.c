#include "activations.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *get_activation_string(ACTIVATION a)
{
    switch(a){
        case LOGISTIC:
            return "logistic";
        case LOGGY:
            return "loggy";
        case RELU:
            return "relu";
        case ELU:
            return "elu";
        case RELIE:
            return "relie";
        case RAMP:
            return "ramp";
        case LINEAR:
            return "linear";
        case TANH:
            return "tanh";
        case PLSE:
            return "plse";
        case LEAKY:
            return "leaky";
        case STAIR:
            return "stair";
        case HARDTAN:
            return "hardtan";
        case LHTAN:
            return "lhtan";
        default:
            break;
    }
    return "relu";
}



/*****************************************************************
*  func：返回s对应ACTIVATION的枚举名称,默认返回RELU              *
*  args s：待获取的名称                                          *
*****************************************************************/
ACTIVATION get_activation(char *s)
{
    if (strcmp(s, "logistic")==0) return LOGISTIC;
    if (strcmp(s, "loggy")==0) return LOGGY;
    if (strcmp(s, "relu")==0) return RELU;
    if (strcmp(s, "elu")==0) return ELU;
    if (strcmp(s, "relie")==0) return RELIE;
    if (strcmp(s, "plse")==0) return PLSE;
    if (strcmp(s, "hardtan")==0) return HARDTAN;
    if (strcmp(s, "lhtan")==0) return LHTAN;
    if (strcmp(s, "linear")==0) return LINEAR;
    if (strcmp(s, "ramp")==0) return RAMP;
    if (strcmp(s, "leaky")==0) return LEAKY;
    if (strcmp(s, "tanh")==0) return TANH;
    if (strcmp(s, "stair")==0) return STAIR;
    fprintf(stderr, "Couldn't find activation function %s, going with ReLU\n", s);
    return RELU;
}



/**********************************************************************
*  func: 根据不同的激活函数类型，调用不同的激活函数处理单个输入元素x  *
*  args x：待处理的元素（单个）                                       *
*  args a: 激活函数类型                                               *
*  details：其实此处有一定的优化空间，因为每次要激活的是一个数组的所有*
*           元素，可以一次性选择激活函数类型后循环处理，而不是在这每次*
*           都重复switch函数类型                                      *
**********************************************************************/
float activate(float x, ACTIVATION a)
{
	// 这些激活函数的定义都在activations.h中，并用static inline 修饰，提高访问效率
    switch(a){
        case LINEAR:
            return linear_activate(x);
        case LOGISTIC:
            return logistic_activate(x);
        case LOGGY:
            return loggy_activate(x);
        case RELU:
            return relu_activate(x);
        case ELU:
            return elu_activate(x);
        case RELIE:
            return relie_activate(x);
        case RAMP:
            return ramp_activate(x);
        case LEAKY:
            return leaky_activate(x);
        case TANH:
            return tanh_activate(x);
        case PLSE:
            return plse_activate(x);
        case STAIR:
            return stair_activate(x);
        case HARDTAN:
            return hardtan_activate(x);
        case LHTAN:
            return lhtan_activate(x);
    }
    return 0;
}


/***********************************************************************************************************
*  func：用激活函数处理输入x中的每一个元素                                                                 *
*  args x：待处理的数组，一般为网络层每个神经元的加权输入Wx+b，在本函数中也相当于是输出（本地操作～）      *
*  args n：x中含有多少个元素                                                                               *
*  args a：激活函数类型                                                                                    *
*  details：该函数会逐个处理x中的元素，注意是逐个；该函数一般用于每一层网络的前向传播函数中，比如          *
*           forward_connected_layer()等，用在最后一步，该函数的输出即为每一层网络的输出                    *
***********************************************************************************************************/
void activate_array(float *x, const int n, const ACTIVATION a)
{
    int i;
    for(i = 0; i < n; ++i){
		// activate函数：本文件，根据a类型的不同，调用不同的激活函数计算x[i]的值并返回
        x[i] = activate(x[i], a);
    }
}

float gradient(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_gradient(x);
        case LOGISTIC:
            return logistic_gradient(x);
        case LOGGY:
            return loggy_gradient(x);
        case RELU:
            return relu_gradient(x);
        case ELU:
            return elu_gradient(x);
        case RELIE:
            return relie_gradient(x);
        case RAMP:
            return ramp_gradient(x);
        case LEAKY:
            return leaky_gradient(x);
        case TANH:
            return tanh_gradient(x);
        case PLSE:
            return plse_gradient(x);
        case STAIR:
            return stair_gradient(x);
        case HARDTAN:
            return hardtan_gradient(x);
        case LHTAN:
            return lhtan_gradient(x);
    }
    return 0;
}

void gradient_array(const float *x, const int n, const ACTIVATION a, float *delta)
{
    int i;
    for(i = 0; i < n; ++i){
        delta[i] *= gradient(x[i], a);
    }
} 

