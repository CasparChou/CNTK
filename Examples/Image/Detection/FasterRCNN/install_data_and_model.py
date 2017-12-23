# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import zipfile
import os, sys

def create_grocery_mappings(grocery_folder):
    sys.path.append(os.path.join(grocery_folder, "..", "..", "Detection", "utils", "annotations"))
    from annotations_helper import create_class_dict, create_map_files
    abs_path = os.path.dirname(os.path.abspath(__file__))
    data_set_path = os.path.join(abs_path, "..", "..", "DataSets", "Grocery")
    class_dict = create_class_dict(data_set_path)
    create_map_files(data_set_path, class_dict, training_set=True)
    create_map_files(data_set_path, class_dict, training_set=False)

if __name__ == '__main__':
    base_folder = os.path.dirname(os.path.abspath(__file__))

    sys.path.append(os.path.join(base_folder, "..", "..", "DataSets", "Pascal"))
    from install_pascalvoc import download_pascal_data
    download_pascal_data()

    sys.path.append(os.path.join(base_folder, "..", "..", "..", "..", "PretrainedModels"))
    from download_model import download_model_by_name
    download_model_by_name("VGG16_ImageNet_Caffe")

    sys.path.append(os.path.join(base_folder, "..", "..", "DataSets", "Pascal", "mappings"))
    from mapping import create_pascal_mappings
    print("Creating mapping files for Pascal data set..")
    create_pascal_mappings(os.path.join(base_folder, "..", "..", "DataSets", "Pascal", "mappings"))
