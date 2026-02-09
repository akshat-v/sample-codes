import cv2
import numpy as np
import math
import os
import time
from multiprocessing import Pool, cpu_count, freeze_support
import cv2
import os
import numpy as np
import pytesseract
from pathlib import Path
import numpy as np
import math

import matplotlib.pyplot as plt 
from skimage.filters import threshold_multiotsu

INPUT_ROOT = r"E:\H2assignment\Diameter_OCR\images"          
OUTPUT_ROOT = r"E:\H2assignment\Diameter_OCR\paperalgo_fast" 

class TextGraphicsSeparator:
    def __init__(self, image_input, H_av: int):
        self.original = image_input
        if self.original is None:
            raise ValueError("Image not found or is None.")
        
        # Binarize
        if len(self.original.shape) == 3:
            gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.original

        _, self.binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        self.H, self.W = self.binary.shape
        self.H_av = H_av
        
        # tuning parameters
        self.T1 = 4.0 * H_av          
        self.T2 = 0.15                
        self.T3 = 6.0                 
        self.T4 = 0.2 * H_av          
        self.T5 = 0.6 * H_av          
        self.T6 = 0.2                 
        self.T7 = 0.7 * H_av          
        self.T8 = 0.25 * H_av         
        self.T9 = 5.0                 

    def _erase_horizontal_lines(self, img: np.ndarray, threshold: float) -> np.ndarray:
        result = img.copy()
        kernel_width = int(threshold)
        if kernel_width < 1: return result
        
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
        detected_lines = cv2.morphologyEx(result, cv2.MORPH_OPEN, horizontal_kernel)
        result = cv2.bitwise_and(result, cv2.bitwise_not(detected_lines))
        return result

    def _shear_image(self, img: np.ndarray, angle_deg: float) -> np.ndarray:
        if angle_deg == 0: return img.copy()
        result = np.zeros_like(img)
        tan_alpha = math.tan(math.radians(angle_deg))
        for j in range(self.W):
            shift = int((tan_alpha * j) % self.H)
            result[:, j] = np.roll(img[:, j], shift)
        return result

    def _unshear_image(self, img: np.ndarray, angle_deg: float) -> np.ndarray:
        if angle_deg == 0: return img.copy()
        result = np.zeros_like(img)
        tan_alpha = math.tan(math.radians(angle_deg))
        for j in range(self.W):
            shift = int((tan_alpha * j) % self.H)
            result[:, j] = np.roll(img[:, j], -shift)
        return result

    def step_3_1_erase_linear_components(self, img: np.ndarray) -> np.ndarray:
        current_img = img.copy()
        current_img = self._erase_horizontal_lines(current_img, self.T1)
        current_img = cv2.rotate(current_img, cv2.ROTATE_90_CLOCKWISE)
        current_img = self._erase_horizontal_lines(current_img, self.T1)
        current_img = cv2.rotate(current_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        slant_T1 = self.T1 * 1.2 
        angles = [22.5, -22.5, 45, -45, 67.5, -67.5]
        for alpha in angles:
            sheared = self._shear_image(current_img, alpha)
            adjusted_T1 = slant_T1 * math.cos(math.radians(alpha))
            sheared_clean = self._erase_horizontal_lines(sheared, adjusted_T1)
            current_img = self._unshear_image(sheared_clean, alpha)
            
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        current_img = cv2.morphologyEx(current_img, cv2.MORPH_OPEN, kernel)
        return current_img

    def step_3_2_filter_strokes(self, img: np.ndarray) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
        output_img = np.zeros_like(img)
        
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            wb_ratio = area / (w * h)
            hw_ratio = h / w if w > 0 else 0
            if hw_ratio < 1 and hw_ratio > 0: hw_ratio = 1.0 / hw_ratio
            longer_edge = max(w, h)
            
            if (wb_ratio < self.T2) or (hw_ratio > self.T3) or (longer_edge < self.T4):
                continue 
            else:
                output_img[labels == i] = 255
        return output_img

    def step_3_3_brushing(self, img: np.ndarray) -> np.ndarray:
        kernel_size = int(self.T5)
        if kernel_size < 1: kernel_size = 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    def step_3_4_morphology(self, img: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        eroded = cv2.erode(img, kernel, iterations=1)
        dilated = cv2.dilate(eroded, kernel, iterations=1)
        return dilated

    def step_3_5_filter_NCCs(self, img: np.ndarray) -> (np.ndarray, list):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
        text_mask = np.zeros_like(img)
        text_boxes = []
        
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            wb_ratio2 = area / (w * h)
            longer_edge = max(w, h)
            shorter_edge = min(w, h)
            hw_ratio2 = h / w
            if hw_ratio2 < 1: hw_ratio2 = 1.0 / hw_ratio2
            
            is_graphics = (
                (wb_ratio2 < self.T6) or
                (longer_edge <= self.T7) or
                (shorter_edge <= self.T8) or
                ((shorter_edge <= self.T7) and (hw_ratio2 >= self.T9))
            )
            
            if not is_graphics:
                text_mask[labels == i] = 255
                text_boxes.append((x, y, w, h))
        return text_mask, text_boxes

    def process_pipeline(self, use_clean_source=False):
    
        img_no_lines = self.step_3_1_erase_linear_components(self.binary)
        
        img_strokes = self.step_3_2_filter_strokes(img_no_lines)
        
        img_brushed = self.step_3_3_brushing(img_strokes)
    
        img_morph = self.step_3_4_morphology(img_brushed)
        
        _, boxes = self.step_3_5_filter_NCCs(img_morph)
       
        final_text_only = np.zeros_like(self.binary)
        
        source_image = img_no_lines if use_clean_source else self.binary
        pad = int(0.25 * self.H_av)
        if pad < 2: pad = 2
        
        for (x, y, w, h) in boxes:
            x_new = max(0, x - pad)
            y_new = max(0, y - pad)
            w_new = min(self.W - x_new, w + 2*pad)
            h_new = min(self.H - y_new, h + 2*pad)
            
            roi = source_image[y_new : y_new+h_new, x_new : x_new+w_new]
            current_area = final_text_only[y_new : y_new+h_new, x_new : x_new+w_new]
            merged_area = cv2.bitwise_or(current_area, roi)
            final_text_only[y_new : y_new+h_new, x_new : x_new+w_new] = merged_area
            
        return final_text_only

#helper functions
def imread_unicode(path):
    try:
        stream = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(stream, cv2.IMREAD_COLOR)
    except Exception:
        return None

def imwrite_unicode(path, img):
    try:
        ext = os.path.splitext(path)[1] or ".png"
        result, n = cv2.imencode(ext, img)
        if result:
            with open(path, mode='wb') as f:
                f.write(n.tobytes())
            return True
        return False
    except Exception:
        return False

def custom_opencv_logic(image):

    processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processor = TextGraphicsSeparator(processed, H_av=20)
    text_layer = processor.process_pipeline()
    return text_layer


def process_single_file(args):
    """
    This function runs in a separate process.
    args: tuple (src_path, dst_path, file_name)
    """
    src_path, dst_path, file_name = args
    
    if os.path.exists(dst_path):
        return f"[Skip] Exists: {file_name}"

    #Load
    original_image = imread_unicode(src_path)
    if original_image is None:
        return f"[Error] Load Failed: {file_name}"

    try:
        #Process
        processed_image = custom_opencv_logic(original_image.copy())

        #Save
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        
        success = imwrite_unicode(dst_path, processed_image)
        if not success:
            return f"[Error] Save Failed: {file_name}"
            
        return None 
        
    except Exception as e:
        return f"[Error] Logic crashed on {file_name}: {e}"

if __name__ == "__main__":
    freeze_support() 
    
    print(f"Source: {INPUT_ROOT}")
    print(f"Target: {OUTPUT_ROOT}")
    
    tasks = []
    print("Scanning files...")
    for root, dirs, files in os.walk(INPUT_ROOT):
        for file in files:
            if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                continue
            
            #paths
            rel_path = os.path.relpath(root, INPUT_ROOT)
            current_output_dir = os.path.join(OUTPUT_ROOT, rel_path)
            src_path = os.path.join(root, file)
            dst_path = os.path.join(current_output_dir, file)
            
         
            tasks.append((src_path, dst_path, file))

    total_files = len(tasks)
    print(f"Found {total_files} images to process.")
    print("-" * 30)

    #multiprocessing
    cores = cpu_count()
    print(f"Starting processing pool with {cores} cores...")
    
    t0 = time.time()
    
    with Pool(processes=cores) as pool:
        results = pool.imap_unordered(process_single_file, tasks)
    
        count = 0
        errors = []
        
        for res in results:
            count += 1
            if res is not None:
                errors.append(res)
                print(res)
            
            if count % 5 == 0 or count == total_files:
                print(f"Progress: {count}/{total_files} ({(count/total_files)*100:.1f}%)", end='\r')

    t1 = time.time()
    print(f"\n\nDone! Processed {count} images in {t1-t0:.2f} seconds.")
    
    if errors:
        print(f"Encountered {len(errors)} errors:")
        for e in errors:
            print(e)