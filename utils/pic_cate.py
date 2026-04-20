import os
import shutil
import toml
from pathlib import Path
from PIL import Image


# 变量区
#target_directory = 'H:/images/train/glass-shoes/to-trim/reg/'
#source_directory = 'D:/images/gamecg/eden-ritter/train/'
#target_directory = source_directory + '/to-trim/'

from PIL import Image
import cv2
import numpy as np
import math
from concurrent.futures import ThreadPoolExecutor

def crop_and_resize_image(input_path, output_path, target_width, target_height):
    """
    将输入图片裁剪并缩放到指定的目标分辨率，尽量保留人脸或人物主体。

    参数:
        input_path: 输入图片路径
        output_path: 输出图片路径
        target_width: 目标分辨率宽度
        target_height: 目标分辨率高度
    """
    print("[crop_and_resize_image]Process:" + input_path + ", output:" + output_path)
    # 打开输入图片
    img = Image.open(input_path)
    original_width, original_height = img.size

    # 计算裁剪比例因子
    target_ratio = target_width / target_height
    original_ratio = original_width / original_height

    if original_ratio >= target_ratio:
        # 如果原图宽高比大于目标宽高比，裁剪宽度
        crop_height = original_height
        crop_width = int(original_height * target_ratio)
    else:
        # 如果原图宽高比小于目标宽高比，裁剪高度
        crop_width = original_width
        crop_height = int(original_width / target_ratio)

    # 计算裁剪区域的起始坐标
    if original_ratio >= target_ratio:
        crop_x = (original_width - crop_width) // 2
        crop_y = 0
    else:
        crop_x = 0
        crop_y = (original_height - crop_height) // 2

    # 使用Pillow裁剪图片
    cropped_img = img.crop((crop_x, crop_y, crop_x + crop_width, crop_y + crop_height))

    # 转换为OpenCV格式以进行人脸识别
    cv_image = cv2.cvtColor(np.array(cropped_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    # 加载人脸检测器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 如果检测到人脸，调整裁剪位置以保留人脸
    if len(faces) > 0:
        # 获取人脸的中心点位置
        (x, y, w, h) = faces[0]
        face_center_x = x + w // 2
        face_center_y = y + h // 2

        # 根据人脸位置调整裁剪区域
        new_crop_x = max(0, face_center_x - crop_width // 2)
        new_crop_y = max(0, face_center_y - crop_height // 2)

        # 确保裁剪区域不会超出图片边界
        new_crop_x = min(new_crop_x, original_width - crop_width)
        new_crop_y = min(new_crop_y, original_height - crop_height)

        # 使用新的裁剪区域
        cropped_img = img.crop((new_crop_x, new_crop_y, new_crop_x + crop_width, new_crop_y + crop_height))

    # 缩放到目标分辨率
    scaled_width = math.ceil(target_width)
    scaled_height = math.ceil(target_height)
    resized_img = cropped_img.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)

    # 如果最终结果因为小数原因多1像素，进行裁剪
    final_width = target_width if scaled_width == target_width else scaled_width
    final_height = target_height if scaled_height == target_height else scaled_height
    if final_width != target_width or final_height != target_height:
        resized_img = resized_img.crop((0, 0, target_width, target_height))

    # 保存为无损压缩的PNG文件
    resized_img.save(output_path, "PNG", optimize=True, compress_level=9)

    print(f"图片已处理并保存到: {output_path}")

# 定义一个函数来计算长宽比
def calculate_aspect_ratio(width, height):
    return float(width) / float(height)

# 定义一个函数来确定最接近的长宽比
def find_closest_aspect_ratio(aspect_ratio, ratios):
    differences = [abs(aspect_ratio - ratio) for ratio in ratios]
    return ratios[differences.index(min(differences))]

def process_single_image(image_file, source_directory, target_directory, trimmed_directory, save_dir_mapper, aspect_ratios):
    print("Process:" + image_file)
    img_path = os.path.join(source_directory, image_file)
    with Image.open(img_path) as img:
        width, height = img.size
    aspect_ratio = calculate_aspect_ratio(width, height)
    closest_ratio = find_closest_aspect_ratio(aspect_ratio, aspect_ratios)
    print("closest_ratio:" + str(closest_ratio))
    find_folder_name = save_dir_mapper[str(closest_ratio)]["folder_name"]
    print("[destination_folder] find: " + find_folder_name)
    destination_folder = os.path.join(target_directory, find_folder_name)
    os.makedirs(destination_folder, exist_ok=True)
    destination_path = os.path.join(destination_folder, image_file)
    shutil.copyfile(img_path, destination_path)
    if len(trimmed_directory) > 0:
        init_img_folder_path = os.path.join(trimmed_directory, find_folder_name)
        os.makedirs(init_img_folder_path, exist_ok=True)
        img_trimmed_output_path = os.path.join(init_img_folder_path, image_file)
        crop_and_resize_image(
            img_path,
            img_trimmed_output_path,
            save_dir_mapper[str(closest_ratio)]["width"],
            save_dir_mapper[str(closest_ratio)]["height"]
        )
    img_file_path = Path(img_path).resolve()
    dir_path = img_file_path.parent
    txt_filename = img_file_path.stem + ".txt"
    txt_full_path = str(dir_path.joinpath(txt_filename))
    dest_img_file_path = Path(destination_path).resolve()
    dest_dir_path = dest_img_file_path.parent
    dest_txt_full_path = str(dest_dir_path.joinpath(dest_img_file_path.stem + ".txt"))
    if os.path.isfile(txt_full_path):
        print("Copy " + txt_full_path + " to " + dest_txt_full_path)
        shutil.copyfile(txt_full_path, dest_txt_full_path)
        if len(trimmed_directory) > 0:
            init_txt_folder_path = os.path.join(trimmed_directory, find_folder_name)
            os.makedirs(init_txt_folder_path, exist_ok=True)
            init_txt_file_path = os.path.join(init_txt_folder_path, txt_filename)
            shutil.copyfile(txt_full_path, init_txt_file_path)
    return {
        "folder_name": find_folder_name,
        "closest_ratio": closest_ratio
    }

def do_main(source_directory, target_directory, trimmed_directory="", train_name = "", worker_count=1):

    # 定义要分类的长宽比列表

    ## 定义起始值和结束条件
    start_width = 64
    start_height = 1984
    end_width = 1984
    end_height = 64

    # 初始化保存文件夹目录名
    save_dir_mapper = {}
    aspect_ratios = []

    # 当前值
    current_width = start_width
    current_height = start_height

    # 生成序列
    while current_width <= end_width and current_height >= end_height:
        save_dir_mapper[str(float(current_width) / float(current_height))] = {"width": current_width, "height": current_height, "folder_name": str(current_width) + "x" + str(current_height)}
        aspect_ratios.append(float(current_width) / float(current_height))
        current_width += 64
        current_height -= 64


    # aspect_ratios = [ 10./23, 3./8, 8./25, 23./10, 8./3, 25./8, 2./3, 1./1, 3./2, 4./7, 7./4, 2./1, 1./2]

    # 20 / 35, 4 / 7 
    # 512 x 1600 8: 25  0.32
    # 576 x 1536 9: 24, 3:8 0.375  2.6667
    # 640 x 1472 10: 23  0.43478
    # 768 x 1344 0.5714

    #如果目标不存在则创建
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    '''
    ### 改为随用创建，不再一开始就初始化目录避免产生过多新目录
    # 初始化一个字典来存储每个比例的文件夹路径
    folders = {str(ratio): os.path.join(target_directory, str(ratio)) for ratio in aspect_ratios}

    # 确保目标文件夹存在
    for folder in folders.values():
        if not os.path.exists( save_dir_mapper.get(folder, folder)): #如果取不到默认就用输入的str
            os.makedirs(save_dir_mapper.get(folder, folder))
    
    '''

    # 获取目标目录下所有的图片文件
    image_files = [f for f in os.listdir(source_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    print ("check save_dir_mapper:")
    print (save_dir_mapper)

    datasets = [] #准备生成数据集
    datasets_batch_size = 1
    datasets_enable_bucket = False
    trimmed_dir_name = os.path.basename(os.path.normpath(trimmed_directory)) if str(trimmed_directory).strip() else ""
    datasets_image_dir_base = "./train/" + train_name + "/"
    if trimmed_dir_name:
        datasets_image_dir_base += trimmed_dir_name + "/"
    datasets_num_repeats = 6

    '''
        {
    "datasets": [
        {
        "batch_size": 1,
        "enable_bucket": false,
        "resolution": [
            768,
            1344
        ],
        "subsets": [
            {
            "image_dir": "./train/3db515/10_3db515-girl/long_port/",
            "num_repeats": 6
            }
        ]
        },
        {
        "batch_size": 1,
        "enable_bucket": false,
        "resolution": [
            832,
            1248
        ],
        "subsets": [
            {
            "image_dir": "./train/3db515/10_3db515-girl/port/",
            "num_repeats": 6
            }
        ]
        }
    ]
    }
    '''
    worker_count = max(1, int(worker_count))
    print("worker_count:" + str(worker_count))
    process_results = []
    if worker_count == 1:
        for image_file in image_files:
            process_results.append(
                process_single_image(
                    image_file,
                    source_directory,
                    target_directory,
                    trimmed_directory,
                    save_dir_mapper,
                    aspect_ratios
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    process_single_image,
                    image_file,
                    source_directory,
                    target_directory,
                    trimmed_directory,
                    save_dir_mapper,
                    aspect_ratios
                )
                for image_file in image_files
            ]
            for future in futures:
                process_results.append(future.result())
    datasets_folder_names = set()
    for result in process_results:
        find_folder_name = result["folder_name"]
        if find_folder_name in datasets_folder_names:
            continue
        closest_ratio = result["closest_ratio"]
        datasets.append(
            {
                "batch_size": datasets_batch_size,
                "enable_bucket": datasets_enable_bucket,
                "resolution": [
                    save_dir_mapper[str(closest_ratio)]["width"],
                    save_dir_mapper[str(closest_ratio)]["height"]
                ],
                "subsets": [{
                    "image_dir": datasets_image_dir_base + find_folder_name,
                    "num_repeats": datasets_num_repeats
                }]
            }
        )
        datasets_folder_names.add(find_folder_name)

    #输出 datasets_xxx.toml
    output_json = {"datasets": datasets}
    toml_string = toml.dumps(output_json)
    with open( os.path.join( trimmed_directory, "datasets.toml"), "w+") as wfp:
        wfp.write(toml_string)

    ss = [
    'H:/images/train/lo_v2_train/train/img/lo_alice/',
    'H:/images/train/lo_v2_train/train/img/lo_chinese/',
    'H:/images/train/lo_v2_train/train/img/lo_classical/',
    'H:/images/train/lo_v2_train/train/img/lo_daily/',
    'H:/images/train/lo_v2_train/train/img/lo_fantasy/',
    'H:/images/train/lo_v2_train/train/img/lo_gothic/',
    'H:/images/train/lo_v2_train/train/img/lo_japanese/',
    'H:/images/train/lo_v2_train/train/img/lo_starry/',
    'H:/images/train/lo_v2_train/train/img/lo_sweet/'

]
'''

ss = ['D:/workspace/collect20231111211935/fetch/']
for row in ss:
    source_directory = row
    target_directory = source_directory + '/to-trim/'
    do_main(source_directory, target_directory)
'''

def do_test():
    crop_and_resize_image('data/test_crop/2025_02_05_20_49_18_2349233-2.png', 'data/test_crop/output1.png', 1280, 720)

def do_base():
    source_directory = "D:/images/artistcg/TAYA - pixiv/train"
    target_directory = source_directory
    trimmed_directory = target_directory + "/trimmed/" #初始化切分后的文件夹
    train_name = "taya"
    do_main(source_directory, target_directory, trimmed_directory, train_name)
    print("图片已按照长宽比分类完毕。")

if __name__ == "__main__":
    do_base()
