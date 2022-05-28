import DicomPipeLine 
import cv2


# Create a save object which is used to save npy arrays to .pickle.gzip files.
gzip_save_obj = DicomPipeLine.Pickle_Gzip()

# Specify the main root DICOM data folder. for example the path 'F:\DICOM data' is the root folder.
dcm_main_folder_path = r'F:\DICOM data' 

# Specify the names of subfolders to scan for the .dcm files.
# for example subfolders '2', '3', '4' and '5' contains the .dcm files.
sub_folders_to_scan = ['2', '3', '4', '5']

# Create a "sort dcm files" object which scans for all the dcm files in a root folder and prepares them for conversion to npy.
dcm_folders_obj = DicomPipeLine.Sort_DCM_Files(dcm_main_folder_path, sub_folders_to_scan)
print(dcm_folders_obj.get_all_folders())

# specify a empty folder to collect all the compressed data to.
database_path = 'F:\Final Year Project\Data Pipeline\TestDataBase_2'

# specify all the attributes to extract from the root DICOM folder.

# these are the attributes to extract from each patient
dcm_attributes_level_0 = [
    'PatientBirthDate',
    'PatientSex',
    'PatientSize',
    'PatientWeight',
    'StudyDate', 
    'StudyTime',
    'InstitutionAddress', 
    'InstitutionName',
    'Manufacturer', 
    'ManufacturerModelName'
]

# these are the attributes to extract from each scan per patient
dcm_attributes_level_1 = [
    'AcquisitionDate', 
    'AcquisitionTime', 
    'PatientPosition',
    'SeriesDescription',
    'SeriesDate',
    'SeriesTime'
]

# these are the attributes to extract from each frame per scan 
dcm_attributes_level_2 = [
    'Rows',
    'Columns',
    'InstanceNumber',
    'PixelSpacing',
    'ImagePositionPatient',
    'ImageOrientationPatient',
    'SliceLocation',
    'SliceThickness'
]

# specify all 3 levels of attributes in a array
dcm_attributes = [dcm_attributes_level_0, dcm_attributes_level_1, dcm_attributes_level_2]

# Create a "dicom input to numpy output" object, this object will be used to convert the sorted dcm files ( from sort_DCM_Files obj ) to compressed numpy files (.pickle.gzip)
to_numpy_obj = DicomPipeLine.DICOM_Input_To_Numpy_Output(
    gzip_save_obj, 'F:\Final Year Project\Data Pipeline\TestDataBase_2',
    dcm_folders_obj.get_all_folders(), dcm_attributes
)

# call the "Iterate_To_Numpy" function of the to_numpy_obj which starts the conversion process.
to_numpy_obj.Iterate_To_Numpy()

print(to_numpy_obj.error_stack)

# Create a data streaming obj , which will be used to stream data from the newly created database.
streamer = DicomPipeLine.Stream_Data(gzip_save_obj, 'F:\Final Year Project\Data Pipeline\DataBase_2')

# Used the streamer obj to stream all the data.
prev_folder = None
prev_scan = None
for img_array, folder_name, scan_name in streamer.iterate_image_data():
    
    img_array = cv2.normalize(img_array, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
    cv2.imshow('View Window',img_array)
    
    if folder_name != prev_folder or prev_scan != scan_name:
        print(f'folder : {folder_name} | Scan : {scan_name}')
        prev_folder = folder_name
        prev_scan = scan_name

    cv2.waitKey(2)