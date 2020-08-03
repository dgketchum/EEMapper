# ===============================================================================
# Copyright 2018 dgketchum
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================

SHP_TO_YEAR_AND_COUNT = {
    'irrigated_test': {2003: 245, 2008: 2069, 2009: 2284, 2010: 2915, 2011: 2676,
                       2012: 2608, 2013: 3020, 2015: 833},
    'fallow_test': {2009: 24, 2010: 69, 2011: 69, 2012: 100, 2013: 81},
    'irrigated_train': {2003: 741, 2008: 6574, 2009: 6849, 2010: 8342, 2011: 7297,
                        2012: 7169, 2013: 8640, 2015: 2580},
    'fallow_train': {2009: 134, 2010: 301, 2011: 301, 2012: 425, 2013: 430},
    'uncultivated_test': {2003: 2372, 2008: 2372, 2009: 2372, 2010: 2372, 2011: 2372, 2012: 2372,
                          2013: 2372, 2015: 2372},
    'uncultivated_train': {2003: 9537, 2008: 9537, 2009: 9537, 2010: 9537, 2011: 9537, 2012: 9537,
                           2013: 9537, 2015: 9537},
    'unirrigated_test': {2003: 3584, 2008: 3584, 2009: 3584, 2010: 3584, 2011: 3584, 2012: 3584,
                         2013: 3584, 2015: 3584},
    'unirrigated_train': {2003: 12238, 2008: 12238, 2009: 12238, 2010: 12238, 2011: 12238, 2012: 12238,
                          2013: 12238, 2015: 12238},
    'wetlands_test': {2003: 1252, 2008: 1252, 2009: 1252, 2010: 1252, 2011: 1252, 2012: 1252,
                      2013: 1252, 2015: 1252},
    'wetlands_train': {2003: 6245, 2008: 6245, 2009: 6245, 2010: 6245, 2011: 6245, 2012: 6245,
                       2013: 6245, 2015: 6245}
}


def shapefile_counts():
    return SHP_TO_YEAR_AND_COUNT


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
