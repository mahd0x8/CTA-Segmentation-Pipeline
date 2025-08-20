import os, shutil
import pydicom  # type: ignore
import numpy as np
import cv2, glob

def normalize_image_intensity(img, hu_min=-500, hu_max=1300):
    """
    Normalizes the image intensity values to a range of [0, 1].
    """
    img = np.clip(img, hu_min, hu_max)
    return (img - hu_min) / (hu_max - hu_min)

def get_dicom_files(path_dicom):
    """
    Recursively finds all DICOM files in the given directory.
    """
    dicom_files = []
    for dir_name, _, file_list in os.walk(path_dicom):
        for filename in file_list:
            dicom_files.append(os.path.join(dir_name, filename))
    return dicom_files

def extract_instance_numbers(dicom_files):
    """
    Extracts InstanceNumber metadata from each DICOM file for sorting.
    """
    instance_dict = {}
    for file in dicom_files:
        try:
            dcm = pydicom.dcmread(file)
            instance_dict[file] = dcm.InstanceNumber
        except Exception:
            continue
    return sorted(instance_dict.items(), key=lambda x: x[1])

def convert_dicom_to_png(dicom_files, output_folder):
    """
    Converts DICOM images to PNG format and saves them to the output folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  

    for file_path, _ in dicom_files:
        try:
            ds = pydicom.dcmread(file_path)
            filename = str(int(ds.InstanceNumber))+".png"
            image = ds.pixel_array.astype(np.float32)
            intercept = getattr(ds, "RescaleIntercept", 0)
            
            if intercept != 0:
                image = image * ds.RescaleSlope + intercept
            
            norm_img = normalize_image_intensity(image)
            img_uint8 = (norm_img * 255).astype(np.uint8)

            cv2.imwrite(os.path.join(output_folder, filename), img_uint8)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

def main():
    for path_dicom in glob.glob(r'/home/xys-05/CT project/Dataset/Upper Limb Studies/*/DICOMS/'):
        print(path_dicom)
        output_folder = os.path.join(os.path.dirname(os.path.dirname(path_dicom)), 'PNG_IMAGES')
        
        try: shutil.rmtree(output_folder) 
        except : pass 

        dicom_files = get_dicom_files(path_dicom)
        sorted_dicom_files = extract_instance_numbers(dicom_files)
        convert_dicom_to_png(sorted_dicom_files, output_folder)

if __name__ == "__main__":
    main()
