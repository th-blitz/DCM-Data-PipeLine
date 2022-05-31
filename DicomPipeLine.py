import numpy as np  
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import gzip
import pickle
import gc


def sync(f):
    f.flush()
    os.fsync(f.fileno())

class Pickle_Gzip:
    # import pickle
    # import gzip
    def __init__(self, path = '', over_write = True):
        self.path = path 
        self.over_write = over_write
        self.extension = '.pickle.gzip'

    def save(self, array, name):
        path = os.path.join(self.path, name + self.extension)
        if os.path.exists(path):
            if self.over_write == False:
                return 
        with gzip.open(path, 'wb+') as f:
            pickle.dump(array, f)
            sync(f)

    def load(self, path):
        with gzip.open(path, 'rb') as f:
            return pickle.load(f)

class Sort_DCM_Files:

    def __init__(self, main_folder_path, scan_folders):
        self.main_folder_path = main_folder_path
        self.scan_folders = scan_folders 
        self.all_folders = self.__get_dcm_files(main_folder_path, scan_folders)

    def __get_dcm_files(self, main_folder_path, scan_folders):

        all_folders = []
        total_count = 0

        sub_folders = [f for f in os.scandir(main_folder_path) if f.is_dir()]
        for folder in tqdm(sub_folders):
            subs = [f for f in os.scandir(folder.path) if f.is_dir()]
            sub_name = folder.name 
            temp = []
            for sub in subs:
                if sub.name in scan_folders:
                    temp.append(sub.name)
            if len(temp) > 0:
                all_folders.append([folder, temp])
                total_count += len(temp)

        print(f'- Found {len(all_folders)} folders with a total of {total_count} scans')
        return all_folders 

    def get_all_folders(self):
        return self.all_folders 

class DICOM_Input_To_Numpy_Output:

    def __init__(self, save_obj, save_path, all_folders, dcm_attributes):
        self.save_obj = save_obj
        self.all_folders = all_folders 
        self.save_path = save_path 
        self.dcm_attributes = dcm_attributes
        self.error_stack = []

    def get_Attribute(self, obj, attribute):

        try: 
            attribute_value = str(obj.data_element(attribute).value)
        except Exception as e:
            attribute_value = 'NaN'
        
        return attribute_value

    def To_Numpy(self, folder_path, dcm_folder_structure, dcm_attributes, source_folder_name):

        patient_description = {
            'Attributes': ['sourceFolder'],
            'Values': [source_folder_name]
        }

        slice_descriptions = []

        series_arrays = []

        for i, key in enumerate(dcm_folder_structure):
            dcm_folder_structure[i] = glob.glob(
                os.path.join( folder_path, dcm_folder_structure[i], r'*.dcm')
            )
        
        obj = pydicom.read_file(dcm_folder_structure[0][0])
        
        for key in dcm_attributes[0]:
            patient_description['Attributes'].append(key) 
            patient_description['Values'].append(self.get_Attribute(obj, key))
        patient_description = pd.DataFrame(patient_description)

        scans_descriptions = [[x] for x in dcm_attributes[1]]
        scans_descriptions_cols = ['Attributes']

        for i, folder in enumerate(dcm_folder_structure):
            obj_one = pydicom.read_file(dcm_folder_structure[i][0])

            scans_descriptions_cols.append(self.get_Attribute(obj_one, 'SeriesDescription'))
            for i, key in enumerate(dcm_attributes[1]):
                scans_descriptions[i].append(self.get_Attribute(obj_one, key))

            
            objs = [ pydicom.read_file(files) for j, files in enumerate(folder)]
            objs.sort(key = lambda x : -1 * int(x.ImagePositionPatient[2]))

            temp_arr = []
            slice_descriptions_one = []
            for obj in tqdm(objs):
                temp_arr.append(obj.pixel_array)
                slice_descriptions_one.append([self.get_Attribute(obj, key) for key in dcm_attributes[2]])

            slice_descriptions.append(pd.DataFrame(slice_descriptions_one, columns = dcm_attributes[2]))

            series_arrays.append(np.array(temp_arr))

        scans_descriptions = pd.DataFrame(scans_descriptions, columns = scans_descriptions_cols)

        return patient_description, scans_descriptions, slice_descriptions, series_arrays, scans_descriptions_cols[1:]

    def Iterate_To_Numpy(self):

        serialize = 0
        for folder in self.all_folders:

            try:
                print(f'- processing folder {folder[0].path}')
                patient_desc, scans_desc, slice_desc, arrays, names = self.To_Numpy(folder[0], folder[1], self.dcm_attributes, folder[0].name) 
            except Exception as e:
                error_template = {}
                error_template['folder name'] = folder[0].name
                error_template['folder path'] = folder[0].path 
                error_template['error message'] = str(e)
                error_template['error arguments'] = str(e.args)
                self.error_stack.append(error_template)
                print(error_template)
                continue

            save_path = os.path.join(self.save_path, str(serialize))

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            descriptions_path = os.path.join(save_path, f'patient_description.csv')
            scans_descriptions_path = os.path.join(save_path, f'scans_descriptions.csv')
            patient_desc.to_csv(descriptions_path)
            scans_desc.to_csv(scans_descriptions_path)

            print(f'- Saving Processed scans {names} ....')
            i = 0
            for array_desc, array in zip(slice_desc, arrays):
                array_desc.to_csv(os.path.join(save_path, names[i] + '.csv'))
                self.save_obj.path = save_path
                self.save_obj.save(array, names[i])
                i += 1

            gc.collect()
            serialize += 1

class Stream_Data:

    def __init__(self, save_obj, save_path):    
        self.save_obj = save_obj 
        self.save_path = save_path
        self.patient_folders = sorted(self.get_folders(), key = lambda x: int(x.name))
        

    def get_folders(self):
        return [f for f in os.scandir(self.save_path) if f.is_dir()]

    def get_patient_details(self, folder_name):
        
        if folder_name not in [f.name for f in self.patient_folders]:
            print(f'folder : {folder_name} not found in database : {self.save_path}')
            return None, None, None
        
        patient_description = pd.read_csv(os.path.join(self.save_path, folder_name, 'patient_description.csv'))
        scans_descriptions = pd.read_csv(os.path.join(self.save_path, folder_name, 'scans_descriptions.csv'))

        scan_details = []
        for col in scans_descriptions.columns[2:]:
            scan_details.append(pd.read_csv(os.path.join(self.save_path, folder_name, f'{col}.csv')))

        return patient_description, scans_descriptions, scan_details

    def iterate_image_data(self, From, To):

        iterable_paths = []
        names = []
        for folder in self.patient_folders[From : To]:
            for x in glob.glob(os.path.join(folder, f'*{self.save_obj.extension}')):
                iterable_paths.append(x)
                names.append([folder.name, x.split('\\')[-1].split('.')[0]])

        queue = []

        i = 0
        i_max = 0
        j = 0
        np_arrays = self.save_obj.load(iterable_paths[j])
        i_max = np_arrays.shape[0]
        while j < len(iterable_paths) - 1:

            if i == i_max:
                j += 1
                np_arrays = self.save_obj.load(iterable_paths[j])
                i_max = np_arrays.shape[0]
                i = 0
                

            yield np_arrays[i], names[j][0], names[j][1]
            i += 1

        return 

    def iterate(self, folder_name, scan_name):
        
        scan_file = scan_name + f'{self.save_obj.extension}'
        scan_path = os.path.join(self.save_path, folder_name, scan_file)
        print(scan_path)
        npy_image_arrays = self.save_obj.load(scan_path)

        for frame in npy_image_arrays:

            yield frame 

        return 
