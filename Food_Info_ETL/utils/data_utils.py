import os
from typing import Literal

class DataUtils:
    """
    """
    # dataset_path=os.path.join("..","..","data","chronolab")
    dataset_path=os.path.join("..","dataset")

    @staticmethod
    def get_food_ids():
        food_ids=[
            name for name in os.listdir(DataUtils.dataset_path)
            if os.path.isdir(os.path.join(DataUtils.dataset_path,name))
        ]
        return food_ids

    @staticmethod
    def get_food_image_paths(food_id):
        dir_path=os.path.join(DataUtils.dataset_path,f"{food_id}")
        image_paths=[
            os.path.join(dir_path,name) 
            for name in os.listdir(dir_path)
            if os.path.isfile(os.path.join(dir_path,name))
        ]
        return image_paths
