import os

from dpl_common.acquisition import Acquisition
from dpl_common.helpers import analysed_folder_name

# NOTE: hidden behaviour = ignore `analysed` folder, and folders that start with `__`

class Mission:
    def __init__(self, folder: str):
        self.folder = folder
        self.analysed_folder = os.path.join(folder, analysed_folder_name)
        self.__read_folder(folder)

    def __read_folder(self, root_folder: str):
        self.acquisitions = []
        acq_names = set()
        for folder in os.listdir(root_folder):
            if folder == analysed_folder_name or folder.startswith("__") or not os.path.isdir(os.path.join(root_folder, folder)):
                continue
            acq_names.add(folder)
        if os.path.exists(self.analysed_folder):
            for file in os.listdir(self.analysed_folder):
                if file.endswith(".tif"):
                    acq_names.add(file[:-4])
        for name in acq_names:
            self.acquisitions.append(Acquisition(name, root_folder))
        self.acquisitions.sort(key=lambda x:x.get_name())

    def get_acquisitions(self) -> list[Acquisition]:
        return self.acquisitions

    def get_folder(self) -> str:
        return self.folder

    def get_analysed_folder(self) -> str:
        return self.analysed_folder