# -*- coding: utf-8 -*-
# @Author: C. Marcus Chuang
# @Date:   2017-09-27 17:37:31
# @Last Modified by:   C. Marcus Chuang
# @Last Modified time: 2017-09-28 23:24:16

from __future__ import division
import pandas as pd


class CodeMapper(object):
    """
    mapping codes and names of land cover types.
    All names are in lower cases.
    Explanation of the code names can be found here:
    https://www.nass.usda.gov/Research_and_Science/Cropland/metadata/metadata_mt15.htm

    code name file downloaded from:
    https://www.nass.usda.gov/Research_and_Science/Cropland/docs/cdl_codes_names.xlsx
    """

    def __init__(self, code_name):
        """
        code_name: a pandas dataframe (read from the file downloaded from usda)
                   with 2 columns:
                   "MasterCat": code
                   "Crop"     : type of the land cover
        attributes:
        c_to_name: a dictionary mapping code to name. key: code, value: names
        name_to_c: a dictionary mapping name to code.
        """
        self.df = code_name
        self._process()

    def _process(self):
        """
        build the two dictionaries: c_to_name and name_to_c
        if name is NaN, the entries would be ignored.
        """
        self.c_to_name, self.name_to_c = {}, {}
        for code, name in zip(self.df.MasterCat, self.df.Crop):
            try:
                name = name.lower()
                self.c_to_name[code] = name
                self.name_to_c[name] = code
            except AttributeError:  # name == NaN
                pass
        return

    def get_code(self, name):
        """
        Get code by crop name; KeyError if name is not in name_to_c

        Args:
            name (str): name of the crop. will be converted to lowercase
        Returns:
            str, the corresponding crop code
        """
        return self.name_to_c[name.lower()]

    def get_name(self, code):
        """
        Get name by crop code; KeyError if code is not in c_to_name

        Args:
            code (int): code of the landtype.
        Returns:
            str, the corresponding crop code
        """
        return self.c_to_name[code]  # error if code not in c_to_name
        # return self.c_to_name.get(code, "---")

    def get_code_list(self, *names):
        """
        Get a list of codes by crop names; KeyError if name is not in name_to_c

        Args:
            *names: names of the crop
        Returns:
            list of (str), the corresponding crop codes
        """
        return [self.get_code(name) for name in names]

    def get_name_list(self, *codes):
        """
        Get a list of names by crop codes; KeyError if code is not in c_to_name

        Args:
            *codes: code of the landtype.
        Returns:
            list of (int), the corresponding crop names
        """
        return [self.get_name(code) for code in codes]

    def get_other_codes(self, *names):
        """
        Get a list of codes that is not in *names

        Args:
            *names: names of the landtype.
        Returns:
            list of (int), the corresponding crop codes
        """
        target = set(self.get_code_list(*names))
        return [code for name, code in self.name_to_c.items()
                if code not in target]

    def get_other_names(self, *codes):
        """
        Get a list of names that is not in *codes

        Args:
            *codes: codes of the landtype.
        Returns:
            list of (str), the corresponding crop names
        """
        return [name for code, name in self.c_to_name.items()
                if code not in set(codes)]

    def is_crop(self, code):
        """
        check if the given code is crop. KeyError if code is not in code_to_c

        Args:
            code (str): code of the landtype
        Returns:
            bool
        """
        return self.get_name(code) in crop_names


non_crop = ['Grassland/Pasture', 'Evergreen_Forest', 'Shrubland',
            'Fallow/Idle_Cropland', 'Open_Water', 'Woody_Wetlands', 'Barren',
            'NoData', 'Herbaceous_Wetlands', 'Developed/Low_Intensity',
            'Developed/Open_Space', 'Sod/Grass_Seed',
            'Developed/Med_Intensity', 'Deciduous_Forest']

crops = ['Winter_Wheat', 'Spring_Wheat', 'Durum_Wheat', 'Alfalfa', 'Barley',
         'Peas', 'Corn', 'Other_Hay/Non_Alfalfa', 'Lentils', 'Canola',
         'Dry_Beans', 'Sugarbeets', 'Mustard', "Triticale", 'Potatoes',
         'Safflower', 'Other_Small_Grains', 'Flaxseed', 'Oats', 'Millet',
         'Soybeans']

non_crop_names = set(s.lower() for s in non_crop)
crop_names = set(s.lower() for s in crops)

code_name = pd.read_csv("./data/cdl_codes_names.csv")
codemapper = CodeMapper(code_name)
