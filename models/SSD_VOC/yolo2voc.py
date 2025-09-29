import os
import shutil
import yaml
import argparse
from PIL import Image
from tqdm import tqdm
import xml.etree.ElementTree as ET


def create_voc_dirs(output_dir):
    """
    在输出目录中创建 PASCAL VOC 所需的文件夹结构。
    """
    voc_root = os.path.join(output_dir, 'VOCdevkit', 'VOC2012')
    dirs = {
        'root': voc_root,
        'annotations': os.path.join(voc_root, 'Annotations'),
        'imagesets_main': os.path.join(voc_root, 'ImageSets', 'Main'),
        'jpeg_images': os.path.join(voc_root, 'JPEGImages')
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def yolo_to_voc_bbox(box, img_size):
    """
    将 YOLO 格式的归一化边界框转换为 PASCAL VOC 格式的绝对坐标。
    (x_center, y_center, width, height) -> (xmin, ymin, xmax, ymax)
    """
    img_w, img_h = img_size
    x_center, y_center, w, h = box

    abs_w = w * img_w
    abs_h = h * img_h
    x_center_abs = x_center * img_w
    y_center_abs = y_center * img_h

    xmin = int(x_center_abs - abs_w / 2)
    ymin = int(y_center_abs - abs_h / 2)
    xmax = int(x_center_abs + abs_w / 2)
    ymax = int(y_center_abs + abs_h / 2)

    # 边界检查
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(img_w - 1, xmax)
    ymax = min(img_h - 1, ymax)

    return xmin, ymin, xmax, ymax


def create_xml_annotation(image_filename, img_size, objects, class_names):
    """
    为单个图像创建 XML 标注内容。
    """
    img_w, img_h = img_size

    annotation = ET.Element('annotation')

    folder = ET.SubElement(annotation, 'folder')
    folder.text = 'VOC2012'

    filename_xml = ET.SubElement(annotation, 'filename')
    filename_xml.text = image_filename

    source = ET.SubElement(annotation, 'source')
    database = ET.SubElement(source, 'database')
    database.text = 'Unknown'

    size = ET.SubElement(annotation, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(img_w)
    height = ET.SubElement(size, 'height')
    height.text = str(img_h)
    depth = ET.SubElement(size, 'depth')
    depth.text = '3'  # 假设是3通道彩色图

    segmented = ET.SubElement(annotation, 'segmented')
    segmented.text = '0'

    for obj in objects:
        class_id, box = obj
        class_name = class_names[class_id]
        voc_box = yolo_to_voc_bbox(box, img_size)

        obj_xml = ET.SubElement(annotation, 'object')

        name = ET.SubElement(obj_xml, 'name')
        name.text = class_name

        pose = ET.SubElement(obj_xml, 'pose')
        pose.text = 'Unspecified'

        truncated = ET.SubElement(obj_xml, 'truncated')
        truncated.text = '0'

        difficult = ET.SubElement(obj_xml, 'difficult')
        difficult.text = '0'

        bndbox = ET.SubElement(obj_xml, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(voc_box[0])
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(voc_box[1])
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(voc_box[2])
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(voc_box[3])

    # Python 3.9+ 支持 ET.indent 来美化输出
    try:
        ET.indent(annotation, space="\t", level=0)
    except AttributeError:
        # 兼容旧版本 Python
        pass

    return ET.ElementTree(annotation)


def convert(yolo_root, output_dir):
    """
    执行转换的主函数。
    """
    # 1. 解析 YAML 文件
    yaml_path = os.path.join(yolo_root, 'data_abs.yaml')
    if not os.path.exists(yaml_path):
        print(f"错误: 在 '{yolo_root}' 中找不到 data_abs.yaml 文件。")
        return

    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    class_names = data['names']

    # 2. 创建 VOC 目录结构
    voc_dirs = create_voc_dirs(output_dir)

    # 3. 遍历数据集 (train, val)
    sets_to_convert = {'train': data.get('train'), 'valid': data.get('val')}

    for set_name, image_dir in sets_to_convert.items():
        if not image_dir or not os.path.isdir(image_dir):
            print(f"警告: '{set_name}' 的图像目录 '{image_dir}' 不存在或未在 YAML 中定义，将跳过。")
            continue

        label_dir = image_dir.replace('images', 'labels')
        if not os.path.isdir(label_dir):
            print(f"警告: 找不到对应的标签目录 '{label_dir}'，将跳过 '{set_name}' 集。")
            continue

        print(f"\n正在处理 '{set_name}' 数据集...")

        # 获取图像列表
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # 打开 ImageSets/Main 下的 txt 文件
        imageset_file = open(os.path.join(voc_dirs['imagesets_main'], f'{set_name}.txt'), 'w')

        for image_filename in tqdm(image_files, desc=f"Converting {set_name}"):
            base_filename = os.path.splitext(image_filename)[0]

            # 写入 ImageSets
            imageset_file.write(f"{base_filename}\n")

            # 复制图片
            src_img_path = os.path.join(image_dir, image_filename)
            dst_img_path = os.path.join(voc_dirs['jpeg_images'], image_filename)
            shutil.copyfile(src_img_path, dst_img_path)

            # 读取图像尺寸
            with Image.open(src_img_path) as img:
                img_size = img.size  # (width, height)

            # 读取并转换标注
            yolo_label_path = os.path.join(label_dir, f'{base_filename}.txt')
            objects = []
            if os.path.exists(yolo_label_path):
                with open(yolo_label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            box = tuple(map(float, parts[1:]))
                            objects.append((class_id, box))

            # 创建并保存 XML 文件
            xml_tree = create_xml_annotation(image_filename, img_size, objects, class_names)
            xml_path = os.path.join(voc_dirs['annotations'], f'{base_filename}.xml')
            xml_tree.write(xml_path, encoding='utf-8', xml_declaration=True)

        imageset_file.close()

    print("\n转换完成！")
    print(f"PASCAL VOC 格式的数据集已保存在: {voc_dirs['root']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert YOLO dataset to PASCAL VOC format.')
    parser.add_argument('--yolo-root', default='E:/Projects/Datasets/tea_leaf_diseases', type=str, required=False,
                        help='YOLO 数据集的主文件夹路径 (包含 data_abs.yaml 的目录)。')
    parser.add_argument('--output-dir', default='E:/Projects/Datasets/tea_leaf_diseases_voc', type=str, required=False,
                        help='转换后 PASCAL VOC 数据集的输出目录。')

    args = parser.parse_args()

    convert(args.yolo_root, args.output_dir)