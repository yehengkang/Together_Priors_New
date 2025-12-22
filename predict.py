from pathlib import Path
import shutil
import time
import re
import cv2
import numpy as np
from PIL import Image

from yolo import YOLO


def extract_datetime_from_filenames(filenames):
    pattern = r"loss_\d+_(\d+)_(\d+)_(\d+)_"

    m = re.search(pattern, filenames)
    if m:
        month, day, hour = m.groups()
        return month + day + '-' + hour


if __name__ == "__main__":
    
    # dir_origin_path = "/media/omnisky/Disk8.0T/datasets/voc_fog_9578+2129/test/VOCtest-FOG"
    dir_origin_path  = "datasets/city/cityscapes_foggy_val_0.01.txt"
    # model_path       = "logs/loss_2025_12_21_23_47_36_city_prior/ep100-loss1.330-val_loss2.549.pth"
    model_path       = "logs/loss_2025_12_21_23_50_33_ciyt_base/ep100-loss0.611-val_loss2.590.pth"
    # model_path       = "logs/loss_2025_11_19_22_27_46_base_16batch/ep100-loss0.495-val_loss1.467.pth"
    data_time        = extract_datetime_from_filenames(model_path.split('/')[1])
    dir_save_path    = 'img_out/test_' + data_time + "_city_base" + '/'
    
    print("Loading model...")
    yolo = YOLO(model_path = model_path)
    print("Model loaded.")
    # ----------------------------------------------------------------------------------------------------------#
    #   mode：
    #   'predict'
    #   'video'
    #   'fps'
    #   'dir_predict'
    # ----------------------------------------------------------------------------------------------------------#
    
    mode            = "dir_predict"
    mode            = "txt_predict" if Path(dir_origin_path).suffix == '.txt' else mode
    crop            = False
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0
    test_interval   = 100
    save_img        = True
    cal_psnr        = True

    if mode == "predict":

        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image, crop = crop)
                r_image.show()

    elif mode == "txt_predict":
        import os
        from tqdm import tqdm
        from utils.psnr import compute_psnr, compute_ssim
        
        detection_path = os.path.join(dir_save_path, "detect_result")
        clear_out_path = os.path.join(dir_save_path, "clear_image")
        foggy_path     = os.path.join(dir_save_path, "foggy_image")
        for path in [detection_path, clear_out_path, foggy_path]:
            if not os.path.exists(path):
                os.makedirs(path)
        
        PSNR_all = []
        SSIM_all = []
        
        # 创建用于保存PSNR结果的文件
        psnr_file_path = os.path.join(dir_save_path, "psnr_results.txt")
        psnr_file      = open(psnr_file_path, "w")
        psnr_file.write("Image Name\tPSNR\n")
        
        txt_path = Path(dir_origin_path)
        with txt_path.open('r', encoding='utf-8') as f:
            image_path_all = f.read()
        for idx, txt_line in enumerate(tqdm(image_path_all.split('\n'))):
            if idx >= 200: break
            image_path = txt_line.split(' ')[0]
            img_name   = os.path.basename(image_path)
            img_suff   = Path(img_name).suffix
            if img_suff in {'.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff'}:
                clear_name      = re.sub(r'_foggy_beta_.*$', '', img_name)
                clear_in_path   = os.path.dirname(image_path).replace("_foggy", "") + "/" + clear_name + img_suff
                       
                image           = Image.open(image_path)
                r_image_detect, r_image     = yolo.detect_image(image, show_on_clear = True)
                # r_image     = yolo.detect_image(image)
        
                if save_img:
                    # r, g, b = r_image_detect.split()  # 分离通道
                    # img_swapped = Image.merge("RGB", (b, g, r))  # R ↔ B
                    # img_swapped.save(os.path.join(detection_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)
                    
                    r_image_detect.save(os.path.join(detection_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)
                    shutil.copy(clear_in_path, clear_out_path)
                    shutil.copy(image_path, foggy_path)
                
                if cal_psnr:
                    pred = np.array(r_image)
                    gt   = np.array(Image.open(clear_in_path))
                    psnr = compute_psnr(pred, gt)
                    ssim = compute_ssim(pred, gt)
                    
                    PSNR_all.append(psnr)
                    SSIM_all.append(ssim)
                    psnr_file.write(f"{img_name}:\t{psnr:.4f}\t{ssim:.4f}\n")
        
        if cal_psnr: 
            print("PSNR_all:", np.mean(PSNR_all))
            psnr_file.write(f"Average_PSNR\t{np.mean(PSNR_all):.4f}\n")
            print("SSIM_all:", np.mean(SSIM_all))
            psnr_file.write(f"Average_SSIM\t{np.mean(SSIM_all):.4f}\n")
            
    elif mode == "dir_predict":
        import os
        from tqdm import tqdm
        from utils.psnr import compute_psnr, compute_ssim
        
        detection_path = os.path.join(dir_save_path, "detect_result")
        clear_out_path = os.path.join(dir_save_path, "clear_image")
        foggy_path     = os.path.join(dir_save_path, "foggy_image")
        for path in [detection_path, clear_out_path, foggy_path]:
            if not os.path.exists(path):
                os.makedirs(path)
        
        PSNR_all = []
        SSIM_all = []
        
        # 创建用于保存PSNR结果的文件
        psnr_file_path = os.path.join(dir_save_path, "psnr_results.txt")
        psnr_file = open(psnr_file_path, "w")
        psnr_file.write("Image Name\tPSNR\n")
        
        img_names = os.listdir(dir_origin_path)
        for idx, img_name in enumerate(tqdm(img_names)):
            if idx >= 200: break
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path      = os.path.join(dir_origin_path, img_name)
                clear_in_path   = image_path.replace("VOCtest-FOG", "JPEGImages")
                
                image           = Image.open(image_path)
                r_image_detect, r_image     = yolo.detect_image(image, show_on_clear = True)
                # r_image     = yolo.detect_image(image)
        
                if save_img:
                    r_image_detect.save(os.path.join(detection_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)
                    shutil.copy(clear_in_path, clear_out_path)
                    shutil.copy(image_path, foggy_path)
                
                if cal_psnr:
                    pred = np.array(r_image)
                    gt   = np.array(Image.open(clear_in_path))
                    psnr = compute_psnr(pred, gt)
                    ssim = compute_ssim(pred, gt)
                    
                    PSNR_all.append(psnr)
                    SSIM_all.append(ssim)
                    psnr_file.write(f"{img_name}:\t{psnr:.4f}\t{ssim:.4f}\n")
        
        if cal_psnr: 
            print("PSNR_all:", np.mean(PSNR_all))
            psnr_file.write(f"Average_PSNR\t{np.mean(PSNR_all):.4f}\n")
            print("SSIM_all:", np.mean(SSIM_all))
            psnr_file.write(f"Average_SSIM\t{np.mean(SSIM_all):.4f}\n")
    
    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc  = cv2.VideoWriter_fourcc(*'XVID')
            size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out     = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("The camera (video) cannot be read correctly!")

        fps = 0.0
        while(True):
            t1 = time.time()
            ref, frame = capture.read()
            if not ref:
                break
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(np.uint8(frame))
            frame = np.array(yolo.detect_image(frame))
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()
        if video_save_path!="":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()
        
    elif mode == "fps":
        img = Image.open('img/1.jpg')
        tact_time = yolo.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
