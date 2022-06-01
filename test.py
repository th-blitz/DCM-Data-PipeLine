import DicomPipeLine


path_obj = DicomPipeLine.Path_Settings(
    'F:\DICOM data', 'F:\Final Year Project\Data Pipeline\TestDataBase_2', ['2','3','4','5']
)

print(len(path_obj.source_folders))
print(len(path_obj.processed_source_folders))
print(len(path_obj.save_folders))

# these are the attributes to extract from each patient
dcm_attributes_level_0 = [
    'PatientName',
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

to_numpy_obj = DicomPipeLine.DCM_Input_To_NPY_Output(path_obj, dcm_attributes)

# to_numpy_obj.iterate(2)

streamer = DicomPipeLine.Stream_Data(path_obj)

# for x in streamer.iter('2', 'WB CECT 1.25mm'):
#     print(x.shape)
