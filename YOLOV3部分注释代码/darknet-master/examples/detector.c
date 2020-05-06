#include "darknet.h"

static int coco_ids[] = {1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90};



/**************************************************************************************************
*  func: 图像检测网络训练函数（针对图像检测的网络训练）                                           *
*  args datacfg: 训练数据描述信息文件路径及名称，如coco.data                                      *
*  args cfgfile: 神经网络结构配置文件路径及名称，如yolov3.cfg                                     *
*  args weightfile：预训练参数文件路径及名称                                                      *
*  args gpus:  GPU卡号集合（比如使用1块GPU，那么里面只含0元素，默认使用0卡号GPU；如果使用4块GPU， *
*              那么含有0,1,2,3四个元素；如果不使用GPU，那么为空指针）                             *
*  args ngpus: 使用GPUS块数，使用一块GPU和不使用GPU时，nqpus都等于1                               *
*  args clear: 是否清空global_step，该参数在load_network中会使用                                  *
*  details：train_detector的实际调用为：                                                          *
*           train_detector("cfg/coco.data","cfg/yolov3.cfg","darknet53.conv.74",gpus,4,0)         *
**************************************************************************************************/
void train_detector(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear)
{	
	// read_data_cfg函数：src/option_list  读入数据配置文件信息，返回配置信息的链表文件
    list *options = read_data_cfg(datacfg);
	
	// 从options找出训练图片路径信息，如果没找到，默认使用"data/train.list"路径下的图片信息
	//（train.list含有标准的信息格式：<object-class> <x> <y> <width> <height>），
    // 该文件可以由darknet提供的scripts/voc_label.py根据自行在网上下载的voc数据集生成，
	// 所以说是默认路径，其实也需要使用者自行调整，也可以任意命名，不一定要为train.list，
    // 甚至可以不用voc_label.py生成，可以自己不厌其烦的制作一个（当然规模应该是很小的，不然太累了。。。）
    // 读入后，train_images将含有训练图片中所有图片的标签以及定位信息
	// option_find_XX函数：从链表中找node的val域等于第二个参数，如“train”的结点，找到后根据需要决定是否进行
	// 类型转换，然后设置该节点的unused=0,找不到则返回该函数的第三个参数作为默认值，查看该函数在src/option_list.c
    char *train_images = option_find_str(options, "train", "data/train.list");	
	// 获取训练权重保存的路径，否则为默认路径"./backup/"
    char *backup_directory = option_find_str(options, "backup", "/backup/");

	// srand函数是随机数发生器的初始化函数。原型：void srand(unsigned int seed);srand和rand()配合使用产生伪随机数序列。
	// rand函数在产生随机数前，需要系统提供的生成伪随机数序列的种子，rand根据这个种子的值产生一系列随机数。
	// 如果系统提供的种子没有变化，每次调用rand函数生成的伪随机数序列都是一样的。通常设置当前时间为种子
    srand(time(0));
	
	// 提取配置文件名称中的主要信息，用于输出打印,再就是作为模型参数名的前缀，比如提取cfg/yolo.cfg中的yolo，用于下面的输出打印
	// basecfg函数：位于src/utils.c 具体注释见该函数
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
	
	// avg_loss为训练网络时的平均loss，一般会在一个batch处理完后，一个step的最后一行输出跟新后的avg_loss，可以根据
	// avg_loss的值判断网络的收敛情况和是否应该调整学习率，如果该值在一个数的上下波动很大，说明学习率过大，应该适当降低
	// 学习率，如果该值长期没有变化，或者变化非常非常小，则网络有可能收敛了，或者就是陷入了局部极小值，此时可以适当增大
	// 学习率看看网络训练情况，也可以更改batch来实验，具体的调整策略还有望读者自行研究，这是一门学问
    float avg_loss = -1;
	
	// 构建网络：用多少块GPU，就会构建多少个相同的网络（不使用GPU时，ngpus=1），因为有时需要修改nets所指向得地址，用双重指针
    network **nets = calloc(ngpus, sizeof(network));

	// 两个srand(time(0))是干啥？
    srand(time(0));
	// 产生随机数，这儿产生了随机数后面好像也没有具体的应用，有什么意义呢？如果你搞明白了，可以给我讲讲
    int seed = rand();
    int i;
	// for循环次数为ngpus，使用多少块GPU，就循环多少次（不使用GPU时，ngpus=1，也会循环一次）
    // 这里每一次循环都会构建一个相同的神经网络，如果提供了初始训练参数，也会为每个网络导入相同的初始训练参数
    for(i = 0; i < ngpus; ++i){
		// 再次设置随机种子，这样反反复复的设置随机种子是有什么深意吗？
        srand(seed);
#ifdef GPU
        // 设置当前活跃GPU卡号（即设置gpu_index=n，同时调用cudaSetDevice函数设置当前活跃的GPU卡号）
        cuda_set_device(gpus[i]);
#endif
        // load_network函数：src/network.c 详细注释见该文档，加载网络这条线一定要分析透彻，十分复杂！
        nets[i] = load_network(cfgfile, weightfile, clear);
		// 多GPU训练的时候为什么学习率要进行这种改变？
        nets[i]->learning_rate *= ngpus;
    }
    srand(time(0));
	
	// 每个网络初始化出来都一样，随便选一个网络获取其参数
    network *net = nets[0];

	// 一次加载进内存的图片数量 注意此处几个值得注意，现假设cfg中batch=16,subdivisions=3,则：
	// net->batch=5(因为16/3=5)，net->subdivisions=3,ngpus为使用GPU得个数，所以imgs=15,注意不是16 关键看能不能整除，一般情况是能整除的
    int imgs = net->batch * net->subdivisions * ngpus;
	// 这条信息显示在权重加载完毕后的第一行
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
	
	// data数据结构定义在 include/darknet中，具体如下：
	/*
		typedef struct{
			int w, h;
			matrix X;
			matrix y;
			int shallow;
			int *num_boxes;
			box **boxes;
		} data;
    */
    data train, buffer;

	// l指向网络的最后一层，因为索引是从0 开始，总共层数为net->n
    layer l = net->layers[net->n - 1];

	// classes表示类别数，jitter表示是否抖动产生新数据，相当于crop
    int classes = l.classes;
    float jitter = l.jitter;

	// get_paths函数：src/data.c 将train_images文件中的内容按行组织成节点插入双向链表中
    list *plist = get_paths(train_images);
    // int N = plist->size;
	// list_to_array函数：src/list.c 将链表plist中每个节点node的val中的值放到数组paths中
    char **paths = (char **)list_to_array(plist);

	// load_args数据类型定义在 include/darknet中 
	// get_base_args函数：src/network.c 将net中关于图像增强的参数赋值给args
    load_args args = get_base_args(net);
	// yolov3的coords在哪儿赋值的呢？我怎么没找到 没有赋值则该值不会用到
    args.coords = l.coords;
    args.paths = paths;
    args.n = imgs;
	
	// plist->size表示有多少个训练样本
    args.m = plist->size;
    args.classes = classes;
    args.jitter = jitter;
	
	// num_boxes表示一张图片最多的框的个数
    args.num_boxes = l.max_boxes;
	// args.d与buffer指向了同一块地址空间
    args.d = &buffer;
    args.type = DETECTION_DATA;
    // args.type = INSTANCE_DATA;
    args.threads = 64;

	// pthread_t用于声明线程ID 
	// load_data函数：src/data.c 开辟线程，读入一次迭代所需的所有图片数据，返回的是读数据的线程ID
    pthread_t load_thread = load_data(args);
	
    double time;
    int count = 0;
    // while(i*imgs < N*120){
	// get_current_batch函数：src/network.c 计算当前已经读入多少个batch
    while(get_current_batch(net) < net->max_batches){
		
		// 此处有一点疑问：如果要进行多尺度调整的那一轮，下面的代码是不是有些部分会被重复执行？
        if(l.random && count++%10 == 0){
			// 开启随机多尺度训练，开启的条件是random=1且每10个batch进行一次
            printf("Resizing\n");
			// 训练的尺度规定在 320-608之间随机取值
            int dim = (rand() % 10 + 10) * 32;
			// 达到500000次训练后就设置 dim=608再训练200次
            if (get_current_batch(net)+200 > net->max_batches) dim = 608;
            // int dim = (rand() % 4 + 16) * 32;
            printf("%d\n", dim);
			// 直接设置训练尺度的w和h等于dim，是否可以设置w不等于h呢？设置了似乎很麻烦，
			// 既要等比例减少，又要保证是32的倍数，这也许就是w=h的原因
            args.w = dim;
            args.h = dim;

			// 阻塞线程，等待数据加载完毕
            pthread_join(load_thread, 0);
			// 将加载的训练数据赋值给train
            train = buffer;
            free_data(train);
            load_thread = load_data(args);

            #pragma omp parallel for
			// resize_network函数：src/network.c 调整网络的输入大小
            for(i = 0; i < ngpus; ++i){
                resize_network(nets[i], dim, dim);
            }
            net = nets[0];
        }
        time=what_time_is_it_now();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);

        /*
           int k;
           for(k = 0; k < l.max_boxes; ++k){
           box b = float_to_box(train.y.vals[10] + 1 + k*5);
           if(!b.x) break;
           printf("loaded: %f %f %f %f\n", b.x, b.y, b.w, b.h);
           }
         */
        /*
           int zz;
           for(zz = 0; zz < train.X.cols; ++zz){
           image im = float_to_image(net->w, net->h, 3, train.X.vals[zz]);
           int k;
           for(k = 0; k < l.max_boxes; ++k){
           box b = float_to_box(train.y.vals[zz] + k*5, 1);
           printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);
           draw_bbox(im, b, 1, 1,0,0);
           }
           show_image(im, "truth11");
           cvWaitKey(0);
           save_image(im, "truth11");
           }
         */

        printf("Loaded: %lf seconds\n", what_time_is_it_now()-time);

        time=what_time_is_it_now();
        float loss = 0;
#ifdef GPU
        if(ngpus == 1){
            loss = train_network(net, train);
        } else {
			// 在使用了GPU的情况下，调用train_networks开始训练
			// train_networks函数：
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_network(net, train);
#endif
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        i = get_current_batch(net);
        printf("%ld: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), loss, avg_loss, get_current_rate(net), what_time_is_it_now()-time, i*imgs);
        if(i%100==0){
#ifdef GPU
            if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s.backup", backup_directory, base);
            save_weights(net, buff);
        }
        if(i%10000==0 || (i < 1000 && i%100 == 0)){
#ifdef GPU
            if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
        }
        free_data(train);
    }
#ifdef GPU
    if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
}


static int get_coco_image_id(char *filename)
{
    char *p = strrchr(filename, '/');
    char *c = strrchr(filename, '_');
    if(c) p = c;
    return atoi(p+1);
}

static void print_cocos(FILE *fp, char *image_path, detection *dets, int num_boxes, int classes, int w, int h)
{
    int i, j;
    int image_id = get_coco_image_id(image_path);
    for(i = 0; i < num_boxes; ++i){
        float xmin = dets[i].bbox.x - dets[i].bbox.w/2.;
        float xmax = dets[i].bbox.x + dets[i].bbox.w/2.;
        float ymin = dets[i].bbox.y - dets[i].bbox.h/2.;
        float ymax = dets[i].bbox.y + dets[i].bbox.h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        float bx = xmin;
        float by = ymin;
        float bw = xmax - xmin;
        float bh = ymax - ymin;

        for(j = 0; j < classes; ++j){
            if (dets[i].prob[j]) fprintf(fp, "{\"image_id\":%d, \"category_id\":%d, \"bbox\":[%f, %f, %f, %f], \"score\":%f},\n", image_id, coco_ids[j], bx, by, bw, bh, dets[i].prob[j]);
        }
    }
}

void print_detector_detections(FILE **fps, char *id, detection *dets, int total, int classes, int w, int h)
{
    int i, j;
    for(i = 0; i < total; ++i){
        float xmin = dets[i].bbox.x - dets[i].bbox.w/2. + 1;
        float xmax = dets[i].bbox.x + dets[i].bbox.w/2. + 1;
        float ymin = dets[i].bbox.y - dets[i].bbox.h/2. + 1;
        float ymax = dets[i].bbox.y + dets[i].bbox.h/2. + 1;

        if (xmin < 1) xmin = 1;
        if (ymin < 1) ymin = 1;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(j = 0; j < classes; ++j){
            if (dets[i].prob[j]) fprintf(fps[j], "%s %f %f %f %f %f\n", id, dets[i].prob[j],
                    xmin, ymin, xmax, ymax);
        }
    }
}

void print_imagenet_detections(FILE *fp, int id, detection *dets, int total, int classes, int w, int h)
{
    int i, j;
    for(i = 0; i < total; ++i){
        float xmin = dets[i].bbox.x - dets[i].bbox.w/2.;
        float xmax = dets[i].bbox.x + dets[i].bbox.w/2.;
        float ymin = dets[i].bbox.y - dets[i].bbox.h/2.;
        float ymax = dets[i].bbox.y + dets[i].bbox.h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(j = 0; j < classes; ++j){
            int class = j;
            if (dets[i].prob[class]) fprintf(fp, "%d %d %f %f %f %f %f\n", id, j+1, dets[i].prob[class],
                    xmin, ymin, xmax, ymax);
        }
    }
}

void validate_detector_flip(char *datacfg, char *cfgfile, char *weightfile, char *outfile)
{
    int j;
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/train.list");
    char *name_list = option_find_str(options, "names", "data/names.list");
    char *prefix = option_find_str(options, "results", "results");
    char **names = get_labels(name_list);
    char *mapf = option_find_str(options, "map", 0);
    int *map = 0;
    if (mapf) map = read_map(mapf);

    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 2);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    srand(time(0));

    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);

    layer l = net->layers[net->n-1];
    int classes = l.classes;

    char buff[1024];
    char *type = option_find_str(options, "eval", "voc");
    FILE *fp = 0;
    FILE **fps = 0;
    int coco = 0;
    int imagenet = 0;
    if(0==strcmp(type, "coco")){
        if(!outfile) outfile = "coco_results";
        snprintf(buff, 1024, "%s/%s.json", prefix, outfile);
        fp = fopen(buff, "w");
        fprintf(fp, "[\n");
        coco = 1;
    } else if(0==strcmp(type, "imagenet")){
        if(!outfile) outfile = "imagenet-detection";
        snprintf(buff, 1024, "%s/%s.txt", prefix, outfile);
        fp = fopen(buff, "w");
        imagenet = 1;
        classes = 200;
    } else {
        if(!outfile) outfile = "comp4_det_test_";
        fps = calloc(classes, sizeof(FILE *));
        for(j = 0; j < classes; ++j){
            snprintf(buff, 1024, "%s/%s%s.txt", prefix, outfile, names[j]);
            fps[j] = fopen(buff, "w");
        }
    }

    int m = plist->size;
    int i=0;
    int t;

    float thresh = .005;
    float nms = .45;

    int nthreads = 4;
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
    pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

    image input = make_image(net->w, net->h, net->c*2);

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    //args.type = IMAGE_DATA;
    args.type = LETTERBOX_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    double start = what_time_is_it_now();
    for(i = nthreads; i < m+nthreads; i += nthreads){
        fprintf(stderr, "%d\n", i);
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for(t = 0; t < nthreads && i+t < m; ++t){
            args.path = paths[i+t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            char *path = paths[i+t-nthreads];
            char *id = basecfg(path);
            copy_cpu(net->w*net->h*net->c, val_resized[t].data, 1, input.data, 1);
            flip_image(val_resized[t]);
            copy_cpu(net->w*net->h*net->c, val_resized[t].data, 1, input.data + net->w*net->h*net->c, 1);

            network_predict(net, input.data);
            int w = val[t].w;
            int h = val[t].h;
            int num = 0;
            detection *dets = get_network_boxes(net, w, h, thresh, .5, map, 0, &num);
            if (nms) do_nms_sort(dets, num, classes, nms);
            if (coco){
                print_cocos(fp, path, dets, num, classes, w, h);
            } else if (imagenet){
                print_imagenet_detections(fp, i+t-nthreads+1, dets, num, classes, w, h);
            } else {
                print_detector_detections(fps, id, dets, num, classes, w, h);
            }
            free_detections(dets, num);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    for(j = 0; j < classes; ++j){
        if(fps) fclose(fps[j]);
    }
    if(coco){
        fseek(fp, -2, SEEK_CUR); 
        fprintf(fp, "\n]\n");
        fclose(fp);
    }
    fprintf(stderr, "Total Detection Time: %f Seconds\n", what_time_is_it_now() - start);
}


void validate_detector(char *datacfg, char *cfgfile, char *weightfile, char *outfile)
{
    int j;
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/train.list");
    char *name_list = option_find_str(options, "names", "data/names.list");
    char *prefix = option_find_str(options, "results", "results");
    char **names = get_labels(name_list);
    char *mapf = option_find_str(options, "map", 0);
    int *map = 0;
    if (mapf) map = read_map(mapf);

    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    srand(time(0));

    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);

    layer l = net->layers[net->n-1];
    int classes = l.classes;

    char buff[1024];
    char *type = option_find_str(options, "eval", "voc");
    FILE *fp = 0;
    FILE **fps = 0;
    int coco = 0;
    int imagenet = 0;
    if(0==strcmp(type, "coco")){
        if(!outfile) outfile = "coco_results";
        snprintf(buff, 1024, "%s/%s.json", prefix, outfile);
        fp = fopen(buff, "w");
        fprintf(fp, "[\n");
        coco = 1;
    } else if(0==strcmp(type, "imagenet")){
        if(!outfile) outfile = "imagenet-detection";
        snprintf(buff, 1024, "%s/%s.txt", prefix, outfile);
        fp = fopen(buff, "w");
        imagenet = 1;
        classes = 200;
    } else {
        if(!outfile) outfile = "comp4_det_test_";
        fps = calloc(classes, sizeof(FILE *));
        for(j = 0; j < classes; ++j){
            snprintf(buff, 1024, "%s/%s%s.txt", prefix, outfile, names[j]);
            fps[j] = fopen(buff, "w");
        }
    }


    int m = plist->size;
    int i=0;
    int t;

    float thresh = .005;
    float nms = .45;

    int nthreads = 4;
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
    pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    //args.type = IMAGE_DATA;
    args.type = LETTERBOX_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    double start = what_time_is_it_now();
    for(i = nthreads; i < m+nthreads; i += nthreads){
        fprintf(stderr, "%d\n", i);
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for(t = 0; t < nthreads && i+t < m; ++t){
            args.path = paths[i+t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            char *path = paths[i+t-nthreads];
            char *id = basecfg(path);
            float *X = val_resized[t].data;
            network_predict(net, X);
            int w = val[t].w;
            int h = val[t].h;
            int nboxes = 0;
            detection *dets = get_network_boxes(net, w, h, thresh, .5, map, 0, &nboxes);
            if (nms) do_nms_sort(dets, nboxes, classes, nms);
            if (coco){
                print_cocos(fp, path, dets, nboxes, classes, w, h);
            } else if (imagenet){
                print_imagenet_detections(fp, i+t-nthreads+1, dets, nboxes, classes, w, h);
            } else {
                print_detector_detections(fps, id, dets, nboxes, classes, w, h);
            }
            free_detections(dets, nboxes);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    for(j = 0; j < classes; ++j){
        if(fps) fclose(fps[j]);
    }
    if(coco){
        fseek(fp, -2, SEEK_CUR); 
        fprintf(fp, "\n]\n");
        fclose(fp);
    }
    fprintf(stderr, "Total Detection Time: %f Seconds\n", what_time_is_it_now() - start);
}

void validate_detector_recall(char *cfgfile, char *weightfile)
{
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    srand(time(0));

    list *plist = get_paths("data/coco_val_5k.list");
    char **paths = (char **)list_to_array(plist);

    layer l = net->layers[net->n-1];

    int j, k;

    int m = plist->size;
    int i=0;

    float thresh = .001;
    float iou_thresh = .5;
    float nms = .4;

    int total = 0;
    int correct = 0;
    int proposals = 0;
    float avg_iou = 0;

    for(i = 0; i < m; ++i){
        char *path = paths[i];
        image orig = load_image_color(path, 0, 0);
        image sized = resize_image(orig, net->w, net->h);
        char *id = basecfg(path);
        network_predict(net, sized.data);
        int nboxes = 0;
        detection *dets = get_network_boxes(net, sized.w, sized.h, thresh, .5, 0, 1, &nboxes);
        if (nms) do_nms_obj(dets, nboxes, 1, nms);

        char labelpath[4096];
        find_replace(path, "images", "labels", labelpath);
        find_replace(labelpath, "JPEGImages", "labels", labelpath);
        find_replace(labelpath, ".jpg", ".txt", labelpath);
        find_replace(labelpath, ".JPEG", ".txt", labelpath);

        int num_labels = 0;
        box_label *truth = read_boxes(labelpath, &num_labels);
        for(k = 0; k < nboxes; ++k){
            if(dets[k].objectness > thresh){
                ++proposals;
            }
        }
        for (j = 0; j < num_labels; ++j) {
            ++total;
            box t = {truth[j].x, truth[j].y, truth[j].w, truth[j].h};
            float best_iou = 0;
            for(k = 0; k < l.w*l.h*l.n; ++k){
                float iou = box_iou(dets[k].bbox, t);
                if(dets[k].objectness > thresh && iou > best_iou){
                    best_iou = iou;
                }
            }
            avg_iou += best_iou;
            if(best_iou > iou_thresh){
                ++correct;
            }
        }

        fprintf(stderr, "%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, correct, total, (float)proposals/(i+1), avg_iou*100/total, 100.*correct/total);
        free(id);
        free_image(orig);
        free_image(sized);
    }
}



/******************************************************************************************************
*  func：完成一张或多张图片的测试                                                                     *
*  args datacfg：配置文件，以.data结尾，如："cfg/coco.data"                                           *
*  args cfgfile: 配置文件，以.cfg结尾，如："cfg/yolov3.cfg"                                           *
*  args weightfile: 权重文件，以.weights结尾，如："yolov3.weights"                                    *
*  args filename: 文件名，一张待测图片，如："data/dog.jpg"                                            *
*  args thresh：阈值设置                                                                              *
*  args hier_thresh：阈值（具体有待进一步理解）                                                       *
*  args outfile：保存检测结果的路径                                                                   *
*  args fullscreen：是否全屏显示                                                                      *
*  details：thresh为输入的-thresh值，hier_thresh默认为0.5，outfile不指定则为当前目录，fullscreen为0/1 *
******************************************************************************************************/
void test_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, char *outfile, int fullscreen)
{
	// 1、list类型定义：include/darknet.h list指向双向链表的头指针
	// 2、read_data_cfg函数：src/option_list.c 返回一个双向链表的头指针，每个节点为node类型，node节点
	//    的值域为kvp节点。三种节点的关系不详述
    list *options = read_data_cfg(datacfg);
	// option_find_str函数：src/option_list.c 从options所指链表中找key为"names"所对应节点的val值，找不
	// 到则返回“data/names.list”
    char *name_list = option_find_str(options, "names", "data/names.list");
	// get_labels函数：src/data.c  在其定义处会调用若干子函数将name_list为名的文件中的内容读取进来，再转化到二维数组中存储，
	// 此二维数组names的每一行对应着形如coco.names 文件中的一行文本 如person等 
    char **names = get_labels(name_list);

	// 1、image类型定义：include/darknet.h 结构体，包含：w,h,c和float *data
	// 2、load_alphabet函数：src/image.c  加载data/labels/文件夹中所有的字符标签图片
	// 现在alphabet中的每一个元素均为image的结构体对象，alphabet总共8行95列，每一行表示同大小规格的95个ascii图像
    image **alphabet = load_alphabet();
	
	// load_network函数：src/network.c 详细注释见该文档，十分复杂！
    network *net = load_network(cfgfile, weightfile, 0);
	// set_batch_network函数：src/network.c 设置网络的batchsize大小为1
    set_batch_network(net, 1);
	
	// 初始化随机种子
    srand(2222222);
    double time;
    char buff[256];
    char *input = buff;
    float nms=.45;
    while(1){
		// 如果命令行中包含了图片路径，则赋值给input，否则提示童虎输入
        if(filename){
            strncpy(input, filename, 256);
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
			// strtok函数原型：char *strtok(char s[], const char *delim);切割字符串，将str切分成一个个子串
			// 在第一次被调用的时间str是传入需要被切割字符串的首地址；在后面调用的时间传入NULL。 
			// 当s中的字符查找到末尾时，返回NULL; 如果查不到delim所标示的字符，则返回当前strtok的字符串的指针。
			/* 例子说明：
			#include<stdio.h>
			#include<string.h>
			int main(void)
			{
				char buf[]="hello@boy@this@is@heima";
				char*temp = strtok(buf,"@");
				while(temp)
				{
					printf("%s ",temp);
					temp = strtok(NULL,"@");
				}
				return0;
			}
            */
			// 预计输出结果："hello boy this is heima ",即每次输出一个单词，此处input保存的为第一个“\n”前面的内容
            strtok(input, "\n");
        }
		// load_image_color函数在src/image.c中有详细注解
		// 读入图片（本函数为预测，所以只读入一张图片）
        // 读入的im是3通道的，三通道分开存储，每通道二维数据按行存储（所有行并成一行），然后三通道再并成一行
        image im = load_image_color(input,0,0); 
		// letterbox_image函数：src/image.c 标准化输入图片的尺寸为神经网络能够处理的图片尺寸net.w、net.h（主要涉及缩放/嵌入操作）
        image sized = letterbox_image(im, net->w, net->h);
        // image sized = resize_image(im, net->w, net->h);
        // image sized2 = resize_max(im, net->w);
        // image sized = crop_image(sized2, -((net->w - sized2.w)/2), -((net->h - sized2.h)/2), net->w, net->h);
        // resize_network(net, sized.w, sized.h);
        layer l = net->layers[net->n-1];

		// 提取image的data准备送往网络进行检测
        float *X = sized.data;
        time=what_time_is_it_now();
		// network_predict函数：src/network.c 以X为输入，前向跑一遍net网络，返回网络的输出
        network_predict(net, X);
		// 输出一张图片跑一遍网络所需的时间
        printf("%s: Predicted in %f seconds.\n", input, what_time_is_it_now()-time);
        int nboxes = 0;
		// 1、detection数据结构定义在：include/darknet.h中
		// get_network_boxes函数：src/network.c 
        detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);
        // printf("%d\n", nboxes);
        // if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
        draw_detections(im, dets, nboxes, thresh, names, alphabet, l.classes);
        free_detections(dets, nboxes);
        if(outfile){
            save_image(im, outfile);
        }
        else{
            save_image(im, "predictions");
#ifdef OPENCV
            cvNamedWindow("predictions", CV_WINDOW_NORMAL); 
            if(fullscreen){
                cvSetWindowProperty("predictions", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
            }
            show_image(im, "predictions", 0);
#endif
        }

        free_image(im);
        free_image(sized);
        if (filename) break;
    }
}

/*****************************************************************************
*  func: 解析argv的参数，根据第三个参数的不同意义调用不同的函数完成对应功能  *
*  args argc：argv中参数的个数                                               *
*  args argv：执行程序时输出的参数列表，此时已不是完整的，在上层函数已经删   *
*             除了若干参数，具体参考本函数的调用处                           *
*****************************************************************************/
void run_detector(int argc, char **argv)
{
	// 将argv参数数组中 -prefix的值赋值给prefix，没有该参数则赋值0，操作完后从
	// argv数组中删除该参数及其对应值（注意有时需要数据类型转化，函数内完成，不赘述）
	// -prefix这个参数具体意义不详，但只要设置了-prefix后，都不会显示demo窗口
    char *prefix = find_char_arg(argc, argv, "-prefix", 0);
	// 将argv参数数组中 -thresh的值赋值给thresh，没有该参数则赋值0.5，操作完后从
	// argv数组中删除该参数及其对应值，-thresh表示显示框的阈值，也可以是有效框的阈值
    float thresh = find_float_arg(argc, argv, "-thresh", .5);
	// 将argv参数数组中 -hier的值赋值给hier_thresh，没有该参数则赋值0.5，操作完后从
	// argv数组中删除该参数及其对应值，-hier在yolov3中没有实际用处
    float hier_thresh = find_float_arg(argc, argv, "-hier", .5);
	// 将argv参数数组中 -c的值赋值给cam_index，没有该参数则赋值0，操作完后从
	// argv数组中删除该参数及其对应值，cam_index表示所用摄像头的索引
    int cam_index = find_int_arg(argc, argv, "-c", 0);
	// 将argv参数数组中 -s的值赋值给frame_skip，没有该参数则赋值0，操作完后从
	// argv数组中删除该参数及其对应值，frame_skip表示跳帧数
    int frame_skip = find_int_arg(argc, argv, "-s", 0);
	// 将argv参数数组中 -avg的值赋值给avg，没有该参数则赋值3，操作完后从
	// argv数组中删除该参数及其对应值
    int avg = find_int_arg(argc, argv, "-avg", 3);
	// 参数总个数小于4时，报错提示使用规则并退出
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }
	// 解析输入参数，获取GPU使用情况，如果使用单个GPU，那么调用时不需要指明GPU卡号，默认使用卡号0上的GPU;
    // 如果使用多块GPU，那么在调用时，其中有两个参数必须为：-gpus 0,1,2...（以逗号隔开）
    // 前者指明是GPU卡号参数，后者为多块GPU的卡号，find_char_arg就是将0,1,2...读入gpu_list
    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
	// 将argv参数数组中 -out的值赋值给outfile，没有该参数则赋值0，操作完后从
	// argv数组中删除该参数及其对应值
    char *outfile = find_char_arg(argc, argv, "-out", 0);
	// 整型数组，为所有使用GPU的卡号集合（如果使用多个GPU，那么会有0,1,2..多个值，
    // 如果只使用一块GPU，那么只有一个元素值0,这是默认卡号；如果不使用GPU，那么以下3个参数最终其实没有用到，
    // 只是为了统一接口，这三个参数不能删）
    int *gpus = 0;
    int gpu = 0;
    int ngpus = 0;
	// 如果不用GPU进行训练，会执行else语句，此时ngpus=1，这不是说还是会使用GPU，因为在真正调用使用GPU的函数之前，
    // 还是会判断#ifdef GPU，如果没有定义，即使ngpus=1也不会使用GPU进行训练；此外，ngpus是最终网络划分的个数，
    // 这个参数不管使不使用GPU都会用到：使用多个GPU时，需要将网络划分成多个，分到各个GPU上进行训练；
	// 而使用一块GPU或者不使用GPU时，显然ngpus都得等于1
    if(gpu_list){
        printf("%s\n", gpu_list);
        int len = strlen(gpu_list);
        ngpus = 1;
        int i;
		// 统计使用几块GPU，每遇到一个‘，’说明又增加了一块GPU
        for(i = 0; i < len; ++i){
            if (gpu_list[i] == ',') ++ngpus;
        }
		// 为保存GPU编号的数组开辟存储空间，开辟空间的块数等于GPU的块数
        gpus = calloc(ngpus, sizeof(int));
		// strchr是查找','第一次出现的位置，指针加1相当于找下一个‘，’后面的内容
        for(i = 0; i < ngpus; ++i){
			// 获得具体使用的GPU编号，转化为整型保存
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',')+1;
        }
    } else {
        gpu = gpu_index;
        gpus = &gpu;
		// 即使不使用GPU，ngpus也为1，表示一个网络
        ngpus = 1;
    }


	// 注意clear参数，这个参数不要轻视，我们都知道在训练模型的时候有一个统计当前迭代了多少个step的参数global_step,
	// 在darknet的训练过程中，global_step这个参数是靠推算出来的：net->seen表示当前已经训练了多少张图片，net->seen/batch
	// 则表示已经训练了多少个batch,而调整学习率的策略是STEPS，即根据训练次数来调整的，所以这个global_step就很关键，clear
	// 参数在这的意义就是决定是否置net->seen等于0，其意图很显然了，为非零值表示置0，0则表示不置0.
    int clear = find_arg(argc, argv, "-clear");
	
	// 这些参数实际使用并不多见，实际意义也不难猜测，此处不细标，具体可以尝试设置对应参数值看效果
	// 一般实在测试的时候使用这些参数，尤其是测试视频的时候
    int fullscreen = find_arg(argc, argv, "-fullscreen");
    int width = find_int_arg(argc, argv, "-w", 0);
    int height = find_int_arg(argc, argv, "-h", 0);
    int fps = find_int_arg(argc, argv, "-fps", 0);
    // int class = find_int_arg(argc, argv, "-class", 0);

	// datacfg形如cfg/coco.data
    char *datacfg = argv[3];
	// cfg形如cfg/yolov3.cfg
    char *cfg = argv[4];
	// 当参数个数满足一定要求时才能取值，此时取argv[5]已经是删除了若干参数的值
    char *weights = (argc > 5) ? argv[5] : 0;
    char *filename = (argc > 6) ? argv[6]: 0;
	// 根据第三个参数的不同，完成不同的功能
    if(0==strcmp(argv[2], "test")) test_detector(datacfg, cfg, weights, filename, thresh, hier_thresh, outfile, fullscreen);
	
	// “train”时进行网络训练，例如当命令行格式为：
	// ./darknet detector train cfg/coco.data cfg/yolov3.cfg darknet53.conv.74 -gpus 0,1,2,3
	// 此时函数train_detector的实际调用为：train_detector("cfg/coco.data","cfg/yolov3.cfg","darknet53.conv.74",gpus,4,0)
	// train_detector函数：本文件。继续完成网络的训练
    else if(0==strcmp(argv[2], "train")) train_detector(datacfg, cfg, weights, gpus, ngpus, clear);
	
    else if(0==strcmp(argv[2], "valid")) validate_detector(datacfg, cfg, weights, outfile);
    else if(0==strcmp(argv[2], "valid2")) validate_detector_flip(datacfg, cfg, weights, outfile);
    else if(0==strcmp(argv[2], "recall")) validate_detector_recall(cfg, weights);
	
	// 此处假设第三个参数为demo时，处理视频或摄像头输入
    else if(0==strcmp(argv[2], "demo")) {
		// read_data_cfg函数：src/option_list 读取形如coco.data中得配置信息到链表options
		// 链表的每一个结点node存储了一条配置信息
        list *options = read_data_cfg(datacfg);
		// option_find_XX函数：从链表中找node的val域等于第二个参数，如“classes”的结点，找到后根据需要决定是否进行
		// 类型转化，此处需转化为整型，找不到则使用默认值20
        int classes = option_find_int(options, "classes", 20);
		// 作用同上，只是此处查找到的值为字符串类型 
        char *name_list = option_find_str(options, "names", "data/names.list");
		// get_labels函数：src/data.c  在其定义处会调用若干子函数将name_list为名的文件中的内容读取进来，再转化到二维数组中存储，
		// 此二维数组names的每一行对应着形如coco.names 文件中的一行文本 如person等
        char **names = get_labels(name_list);
		// demo函数：src/demo.c 详细注释见函数定义处
        demo(cfg, weights, thresh, cam_index, filename, names, classes, frame_skip, prefix, avg, hier_thresh, width, height, fps, fullscreen);
    }
    //else if(0==strcmp(argv[2], "extract")) extract_detector(datacfg, cfg, weights, cam_index, filename, class, thresh, frame_skip);
    //else if(0==strcmp(argv[2], "censor")) censor_detector(datacfg, cfg, weights, cam_index, filename, class, thresh, frame_skip);
}
