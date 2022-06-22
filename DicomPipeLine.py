import numpy as np  
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import glob
import os
import pandas as pd
from tqdm import tqdm
import gzip
import pickle
import gc
import registration as reg

Main_Attributes = {
    'Source_Folder':'sourceFolder', 
    'Series_Description':'SeriesDescription'
}

Main_File_Names = {

}

# save_path and source_path

class Path_Settings:
    # import os
    def __init__(self, source_path, save_path, scan_for_folders):

        self.blacklist = []

        # String objs
        self.source_path = source_path
        self.save_path = save_path
        # list obj
        self.scan_for_folders = scan_for_folders
        # save obj
        self.save_obj = Pickle_Gzip()

        # list of os.path objs
        self.save_folders = []
        # list of source folder names
        self.processed_source_folders = []
        # list of os.path objs for eg: [<os_obj>'1.3.24.34342', [<os_obj>'2', <os_obj>'3', <os_obj>'4']]
        self.source_folders = []
        
        self.Refresh( print_progress = False )

        self.saved_serialization = [int(f.name) for f in self.save_folders]
        if len(self.saved_serialization) != 0:
            self.expected_serialization = sorted([x for x in range(1, max(self.saved_serialization)) if x not in self.saved_serialization])
            self.serialize = max(self.saved_serialization)
        else:
            self.expected_serialization = []
            self.saved_serialization = []
            self.serialize = 0

    def Check_Path(self):
        return

    def pop(self):

        source_folder = self.source_folders.pop(0)
        self.processed_source_folders.append(source_folder[0].name)
        if len(self.expected_serialization) != 0:
            serialize = self.expected_serialization.pop(0)
        else:
            self.serialize += 1
            serialize = self.serialize
        return source_folder, str(serialize) 
        
    def Refresh(self, print_progress = True):
        self.save_folders = [f for f in os.scandir(self.save_path) if f.is_dir()]
        self.processed_source_folders = []
        for f in self.save_folders:
            patient_desc = pd.read_csv(os.path.join(f.path, 'patient_description.csv'))
            self.processed_source_folders.append(patient_desc['Values'][0])
        
        self.source_folders = self.get_all_source_folders(print_progress = print_progress)

        for f_name in self.processed_source_folders:
            for f in self.source_folders:
                if f_name == f[0].name or f[0].name in self.blacklist:
                    self.source_folders.remove(f)

        self.serialize = len(self.processed_source_folders)
        return 

    def Reset(self):
        return 

    def get_all_source_folders(self, print_progress = True):

        all_folders = []
        total_count = 0

        if print_progress == True:
            for folder in tqdm([f for f in os.scandir(self.source_path) if f.is_dir()]):
                subs = [f for f in os.scandir(folder.path) if f.is_dir()]
                sub_name = folder.name 
                temp = []
                for sub in subs:
                    if sub.name in self.scan_for_folders:
                        temp.append(sub)
                if len(temp) > 0:
                    all_folders.append([folder, temp])
                    total_count += len(temp)
            print(f'- Found {len(all_folders)} folders with a total of {total_count} scans')
        else:
            for folder in [f for f in os.scandir(self.source_path) if f.is_dir()]:
                subs = [f for f in os.scandir(folder.path) if f.is_dir()]
                sub_name = folder.name 
                temp = []
                for sub in subs:
                    if sub.name in self.scan_for_folders:
                        temp.append(sub)
                if len(temp) > 0:
                    all_folders.append([folder, temp])
                    total_count += len(temp)
        return all_folders 


class DCM_Input_To_NPY_Output:

    def __init__(self, path_settings_obj, dcm_attributes):

        self.path_settings_obj = path_settings_obj
        self.dcm_attributes = dcm_attributes 
        self.error_stack = []

    def get_Attribute(self, obj, attribute):

        try: 
            attribute_value = str(obj.data_element(attribute).value)
        except Exception as e:
            attribute_value = 'NaN'
        
        return attribute_value

    def To_Numpy(self, patient_folder):

        # patient_folder eg: [<dir obj> '1.3.240.032903', [<dir obj> '1', <dir obj> '2']]
        patient_description = {
            'Attributes': [Main_Attributes['Source_Folder']],
            'Values': [patient_folder[0].name]
        }
        slice_descriptions = []
        series_arrays = []

        for i, sub_folder in enumerate(patient_folder[1]):
            patient_folder[1][i] = glob.glob(
                os.path.join(sub_folder.path, '*.dcm')
            )

        obj = pydicom.read_file(patient_folder[1][0][0])

        for key in self.dcm_attributes[0]:
            patient_description['Attributes'].append(key)
            patient_description['Values'].append(self.get_Attribute(obj, key))
        patient_description = pd.DataFrame(patient_description)

        scans_descriptions = [[x] for x in self.dcm_attributes[1]]
        scans_descriptions_cols = ['Attributes']

        for i, folder in enumerate(patient_folder[1]):
            obj_one = pydicom.read_file(patient_folder[1][i][0])

            scans_descriptions_cols.append(self.get_Attribute(obj_one, Main_Attributes['Series_Description']))
            for i, key in enumerate(self.dcm_attributes[1]):
                scans_descriptions[i].append(self.get_Attribute(obj_one, key))

            
            objs = [ pydicom.read_file(files) for j, files in enumerate(folder)]
            objs.sort(key = lambda x : -1 * int(x.ImagePositionPatient[2]))

            temp_arr = []
            slice_descriptions_one = []
            for obj in tqdm(objs):
                temp_arr.append(obj.pixel_array)
                slice_descriptions_one.append([self.get_Attribute(obj, key) for key in self.dcm_attributes[2]])

            slice_descriptions.append(pd.DataFrame(slice_descriptions_one, columns = self.dcm_attributes[2]))

            series_arrays.append(np.array(temp_arr))

        scans_descriptions = pd.DataFrame(scans_descriptions, columns = scans_descriptions_cols)

        return patient_description, scans_descriptions, slice_descriptions, series_arrays, scans_descriptions_cols[1:] 

    def iterate(self, number_of_source_folders = 0):

        if number_of_source_folders > len(self.path_settings_obj.source_folders):
            print(f'- specified number of folders {number_of_source_folders} exceeds total folders {len(self.path_settings_obj.source_folders)}')
            number_of_source_folders = 1
            print('- Hence processing only one folder')

        iteration_count = number_of_source_folders

        while iteration_count > 0:

            source_folder, target_folder = self.path_settings_obj.pop()

            try:
                print(f'- processing folder {source_folder[0].path}')
                patient_desc, scans_desc, slice_desc, arrays, names = self.To_Numpy(source_folder)
            except:
                error_template = {}
                error_template['folder name'] = source_folder[0].name
                error_template['folder path'] = source_folder[0].path 
                error_template['error message'] = str(e)
                error_template['error arguments'] = str(e.args)
                self.error_stack.append(error_template)
                print(error_template)
                continue

            save_path = os.path.join(self.path_settings_obj.save_path, target_folder)

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
                self.path_settings_obj.save_obj.path = save_path
                self.path_settings_obj.save_obj.save(array, names[i])
                i += 1

            gc.collect()
            iteration_count -= 1


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

    def save(self, data, name):
        path = os.path.join(self.path, name + self.extension)
        if os.path.exists(path):
            if self.over_write == False:
                return 
        with gzip.open(path, 'wb+') as f:
            pickle.dump(data, f)
            sync(f)

    def load(self, path):
        with gzip.open(path, 'rb') as f:
            return pickle.load(f)


class Save:

    def __init__(self, save_folder_path):
        self.save_folder_path = save_folder_path
        self.save_obj = Pickle_Gzip(save_folder_path)

    def save(self, data, file_name):
        self.save_obj.save(data, file_name)

    def load(self, file_name):
        load_path = os.path.join(self.save_folder_path, file_name + self.extension)
        return self.save_obj.load(load_path)


class Stream_Data:

    def __init__(self, path_settings_obj):    
        self.save_obj = path_settings_obj.save_obj
        self.save_path = path_settings_obj.save_path
        self.patient_folders = sorted(self.get_folders(), key = lambda x: int(x.name))
        
    def save_transform_points(self,folder, img_a, img_b, points_a, points_b):
        save_path = os.path.join(self.save_path, folder)
        self.save_obj.path = save_path
        self.save_obj.save([img_a, img_b, points_a, points_b], 'transformation_points')
        return

    def get_scans(self, folder_name, scan_a, scan_b, transform = False):
        scan_a_array = self.get(folder_name, scan_a)
        scan_b_array = self.get(folder_name, scan_b)

        if transform == False:
            return scan_a_array, scan_b_array
        
        points_path = os.path.join(self.save_path, folder_name, f'transformation_points{self.save_obj.extension}')
        img_a, img_b, pts_a, pts_b = self.save_obj.load(points_path)
        trf_obj = reg.TransFormation(img_a, img_b, pts_a, pts_b)

        transformed_frames = []
        for y in scan_b_array:
            transformed_frames.append(trf_obj.transform(y))

        transformed_frames = np.array(transformed_frames)
        return scan_a_array, transformed_frames


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


    def iter(self, folder_name, scan_name):
            
        scan_file = scan_name + f'{self.save_obj.extension}'
        scan_path = os.path.join(self.save_path, folder_name, scan_file)
        print(scan_path)
        npy_image_arrays = self.save_obj.load(scan_path)

        for array in npy_image_arrays:
            yield array 

        return 

    def get(self, folder_name, scan_name):
        scan_file = scan_name + f'{self.save_obj.extension}'
        scan_path = os.path.join(self.save_path, folder_name, scan_file)
        print(scan_path)
        return self.save_obj.load(scan_path)
        

    



