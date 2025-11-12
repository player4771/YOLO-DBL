import torch
import joblib
from PIL import Image
from math import sqrt
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision import transforms

import ultralytics

def extract_data(model, model_input:torch.Tensor, layer_indexes:list|tuple[int,...]):

    results = {}
    def get_hook(layer_index):
        def hook(module, input, output):
            results[layer_index] = {
                'module': module, #(module structure)
                'input': input, #input: tuple[Tensor(N,C,H,W), ...]
                'output': output, #output: tuple[Tensor(C,H,W), ...]
            }
        return hook

    # YOLO核心层在model.model.model(nn.ModuleList)
    modules = model.model.model

    handles = []
    for layer_index in layer_indexes:
        if layer_index < 0:  # 负数索引
            layer_index = len(modules) + layer_index

        if layer_index < len(modules):
            module = modules[layer_index] # 获取目标层
            handle = module.register_forward_hook(get_hook(layer_index)) # 为该层注册 hook
            handles.append(handle) # 保存 handle，以便在推理后移除
        else:
            raise IndexError(f" 索引 {layer_index} 超出了层数范围 ({len(modules)})")

    with torch.no_grad():
        model.eval()(model_input) # forward会自动触发所有已注册的hook函数

    for handle in handles:
        handle.remove() # 防止内存泄漏和不必要的计算

    for k,v in results.items():
        print(k,':')
        print('module: ', type(v['module']))
        print('input:   ', end='')
        for item in v['input']:
            print(item.shape if hasattr(item,'shape') else type(item), sep='', end=', ')
        print('\noutput:  ', end='')
        for item in v['output']:
            print(item.shape if hasattr(item,'shape') else type(item), sep='', end=', ')
        print()

    return results

def find_plot_size(tensor_shape:tuple[int,...]) -> tuple[int,int]:
    if len(tensor_shape) == 2:
        return 1,1
    elif len(tensor_shape) in (3,4):
        C = tensor_shape[-3]
        if sqrt(C) % 1 == 0:  #可以直接分解为n*n，输出为正方形
            return int(sqrt(C)), int(sqrt(C))
        elif sqrt(C*2) % 1 == 0: #可分解为2^n*2^(n-1)
            return int(sqrt(C*2)), int(sqrt(C/2))
        elif sqrt(C/15) % 1 == 0: #可分解为5n*3n(黄金分割比)，看起来舒服些
            #注:判断面积C是否可分解为p:q的两整数边, 需验证:sqrt(C/pq)?=0
            #令k=sqrt(C/pq), 分解出的两边分别为pk和qk
            return int(sqrt(C/15))*5, int(sqrt(C/15))*3
        else:
            for width in range(int(sqrt(C)), 0, -1):
                if C % width == 0:
                    return width, C//width
        raise Exception("How the fuck could this be???")
    else:
        raise NotImplementedError(f"Unsupported shape: {tensor_shape}, required 2D(HW), 3D(CHW), or 4D(NCHW).")

def plot_feature_map(feature_map:torch.Tensor, plot_size:tuple[int,int]=None): #feature_map:CHW
    plot_size = find_plot_size(feature_map.shape) if plot_size is None else plot_size #WH,且W必定大于H

    fm_numpy = feature_map.detach().cpu().numpy()[0] #NCHW, N=1
    # 下面三行是把所有通道铺平，变为一整张图
    #fm_numpy = fm_numpy.reshape(plot_size+fm_numpy.shape[1:]) #C,H,W -> row,col,H,W
    #fm_numpy = fm_numpy.transpose(0, 2, 1, 3) #这是为了下面的reshape. shape: row,H,col,W
    #fm_numpy = fm_numpy.reshape(plot_size[1]*fm_numpy.shape[3], plot_size[0]*fm_numpy.shape[1]) #col*W, row*H

    fig, axes = plt.subplots(*plot_size[::-1], figsize=plot_size, dpi=100) #nrows,ncols = H,W，所以要翻转一下

    for i,axi in enumerate(axes.flat):
        axi.imshow(fm_numpy[i,:,:], cmap='viridis')
        axi.axis('off')

    adjust_l = 0.005 #以左侧边距为基准
    adjust_b = adjust_l * plot_size[0] / plot_size[1]
    adjust_r = 1 - adjust_l #左右边距相等
    adjust_t = 1 - adjust_b #上下边距相等
    # 参数前四个是图片整体四周的边距, 后两个是每个子图间的间距, 以左下角为基准, 值为比例.
    fig.subplots_adjust(adjust_l, adjust_b, adjust_r, adjust_t, wspace=0.05, hspace=0.05)
    fig.show()

    return fig, axes

def plot_feature_maps(*feature_maps) -> None:
    for fm in feature_maps:
        plot_feature_map(fm)

if __name__ == "__main__":
    model = ultralytics.YOLO(r'E:\Projects\PyCharm\Paper2\models\YOLO\runs\train16\weights\best.pt')
    sample = r"E:\Projects\Datasets\example\algal+gray.jpg"
    model_input = transforms.ToTensor()(Image.open(sample)).unsqueeze(0) #ToTensor会自动HWC->CHW和标准化

    layer_indexes = (26, 31, 36)

    cache_file = f'./cache/{hash(layer_indexes)}.cache' #用哈希当文件名就是一坨屎，但是能用
    if Path(cache_file).exists():
        results = joblib.load(cache_file)
    else:
        results = extract_data(model, model_input, layer_indexes)
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(results, cache_file)

    plot_feature_maps(*tuple([results[i]['output'] for i in layer_indexes]))