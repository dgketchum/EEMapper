# ===============================================================================
# Copyright 2020 dgketchum
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
from pprint import pprint

def variable_importance():
    return [('CDL', 0.0634524752804307),
            ('NLCD', 0.05690229985097357),
            ('Near Infrared Late Spring', 0.04846799523587754),
            ('NDVI Max, Current Year', 0.045404237850643424),
            ('Near Infrared Summer', 0.04015844163951064),
            ('NDVI Max, Previous Year', 0.03563385257746881),
            ('Slope', 0.02745370484079356),
            ('NDVI Max, Two Years Previous', 0.027120492992345962),
            ('Latitude', 0.02708653896853951),
            ('Red Late Spring', 0.02537686479728658),
            ('Shortwave Infrared 2 Late Spring', 0.021513328748268033),
            ('Shortwave Infrared 2 Summer', 0.01904087261906528),
            ('Blue Late Spring', 0.016482011071812634),
            ('Red Summer', 0.014488388146875208),
            ('Near Infrared Early Spring', 0.014109373888127854),
            ('Blue Summer', 0.014008990183727296),
            ('Band 3 Late Spring', 0.012889125815902564),
            ('Shortwave Infrared 1 Summer', 0.011456235384148413),
            ('Longitude', 0.011414648884118032),
            ('Average Maximum Temperature, May', 0.011356055510761928),

            ('Average Maximum Temperature, April', 0.011262788934088353),
            ('Shortwave Infrared 1 Late Spring', 0.010766815521641755),
            ('tmax_6_1', 0.010381679009479833),
            ('prec_5', 0.010237884870189656),
            ('tpi_250', 0.01010375436780672),
            ('tpi_150', 0.009646955491424625),
            ('tmax_8', 0.00962357858336669),
            ('elevation', 0.009548001216463549),
            ('tmax_9', 0.009384967919768793),
            ('tpi_1250', 0.009177353266752366),
            ('prec_4', 0.008615873598120697),
            ('tmax_3_1', 0.008562484434796583),
            ('Band 3 Summer', 0.008005155368602353),
            ('prec_6', 0.0078851822152077),
            ('tmax_8_1', 0.007682154983091757),
            ('tmax_11', 0.007561127363055792),
            ('tavg_6', 0.007337233094385815),
            ('prec_3', 0.007332025556682602),
            ('tavg_5', 0.007312472482971807),
            ('prec_7', 0.007311856536881584),
            ('tmax_7_1', 0.0068430822338193),
            ('tmax_10', 0.006765866971534662),
            ('prec_10', 0.006633957215546568),
            ('prec_11', 0.006474117707037606),
            ('prec_1', 0.006313820949409482),
            ('tmin_5_1', 0.0062825593501239785),
            ('tmin_7_1', 0.0060392553253056066),
            ('prec_9', 0.005993855454584021),
            ('prec_8', 0.005908564364526284),
            ('tavg_4', 0.005860604537902074),
            ('prec_2', 0.0054741202130253915),
            ('tmax_1_1', 0.0054726663853758035),
            ('tmin_6_1', 0.005343130873879674),
            ('prec', 0.00524861040132165),
            ('B6', 0.005166791290553566),
            ('tavg', 0.004819368648763654),
            ('tavg_7', 0.004787644548246965),
            ('tmin_4_1', 0.004677801517941165),
            ('tmax_2_1', 0.004628333655043371),
            ('B7', 0.004076416939486642),
            ('tmin_8_1', 0.004027227707450473),
            ('tmin_9', 0.003892805334454492),
            ('tmin_8', 0.0037766228345785017),
            ('tavg_11', 0.003663487324016229),
            ('tavg_8', 0.0035220368663057304),
            ('B6_1', 0.003480281433773459),
            ('pet_total_wy_espr', 0.0034753767102169916),
            ('B7_1', 0.003414854431499345),
            ('tavg_1', 0.003393793463175746),
            ('tavg_3', 0.003204547438746576),
            ('tmin_3_1', 0.0030731040262944737),
            ('tavg_10', 0.0030714853239895224),
            ('tavg_2', 0.0030557217163336655),
            ('tmin_11', 0.00304178194524441),
            ('tmin_1_1', 0.0029906827746985316),
            ('B5', 0.002980391543357021),
            ('tmin_2_1', 0.0029191089040493247),
            ('tmax_1', 0.0027054223591845014),
            ('pet_total_wy', 0.0027034958389201775),
            ('tmax', 0.002700484587488438),
            ('tmin_10', 0.002600364234622706),
            ('pet_total_fl', 0.0025421688027858548),
            ('pet_total_wy_espr_1', 0.002539382646194657),
            ('tavg_9', 0.002500463007314728),
            ('tmmn_p90_cy', 0.0024873590100329304),
            ('tmmn_p50_cy', 0.0024635513246822967),
            ('tmin_4', 0.00246038297740898),
            ('tmin', 0.0023933942310409148),
            ('tmmx_p50_cy', 0.0023838148331634425),
            ('tmax_4', 0.0023286382092591014),
            ('tmax_5', 0.0023173336524520494),
            ('B4_1', 0.0023171209316247767),
            ('tmmx_p10_cy', 0.0023154604791868517),
            ('B2_1', 0.0023102795316000827),
            ('tmax_7', 0.0022711951709503166),
            ('precip_total_smr', 0.0022257296564084),
            ('tmax_6', 0.0021187256151930984),
            ('B3_1', 0.0021161789238096722),
            ('tmmn_p10_cy', 0.0021085689204314905),
            ('tmax_3', 0.0021075258554974744),
            ('pet_total_wy_smr', 0.0020804911841324674),
            ('tmax_2', 0.0020795145172560646),
            ('pet_total_espr', 0.0020071672368384757),
            ('wd_est_smr', 0.0019384548490692314),
            ('tmin_5', 0.0019219678140427767),
            ('wd_est_lspr', 0.0018610307229229388),
            ('precip_total_fl', 0.0017979859656002905),
            ('B4', 0.0017831679254514993),
            ('B3', 0.0017800721928572319),
            ('precip_total_lspr', 0.0017617688409138851),
            ('precip_total_wy_espr', 0.0017593943097505243),
            ('wd_est_fl', 0.0017578045585903152),
            ('pet_total_smr', 0.0017567031169599286),
            ('wd_est_espr', 0.001756147745065555),
            ('tmin_7', 0.0017554845563464935),
            ('wd_est_wy', 0.0017524453356127106),
            ('tmmx_p90_cy', 0.0016868057083063495),
            ('pet_total_lspr', 0.0016763868785166242),
            ('wd_est_wy_espr', 0.0016725601856504662),
            ('wd_est_wy_smr', 0.001647812393423027),
            ('tmin_6', 0.0016388750099385503),
            ('wd_est_cy', 0.0016123594375595275),
            ('tmin_3', 0.0015816629631309202),
            ('tmin_2', 0.0015699658039169718),
            ('wd_est_wy_espr_1', 0.0015266629501),
            ('tmin_1', 0.001507045548027179),
            ('B2', 0.0014528605524679072),
            ('precip_total_espr', 0.0014415072432761317),
            ('precip_total_wy', 0.001408804378669008),
            ('precip_total_wy_smr', 0.0013952117301119387),
            ('precip_total_wy_espr_1', 0.0012903256152915522),
            ('aspect', 0.0009753544678820362)]


def dec4_names():
    return ['nlcd',
            'cdl',
            'LAT_GCS',
            'nd_max_cy',
            'B5_3',
            'nd_4',
            'slope',
            'B7_4',
            'nd_3',
            'nd_max_m1',
            'B5_4',
            'B2_3',
            'prec_5',
            'Lon_GCS',
            'nd_max_m2',
            'prec_4',
            'B6_4',
            'B4_4',
            'prec_6',
            'elevation',
            'B4_3',
            'tpi_150',
            'B7_3',
            'tmax_10',
            'tpi_250',
            'B3_3',
            'tmin_7_1',
            'prec_7',
            'B6_1',
            'tpi_1250',
            'tmax_8',
            'prec_1',
            'tmax_11',
            'B3_4',
            'B6_3',
            'B5_2',
            'prec_8',
            'prec_10',
            'prec_11',
            'tmin_4_1',
            'prec',
            'prec_3',
            'tmax_4_1',
            'tmin_5_1',
            'tavg_4',
            'B2_4',
            'prec_2',
            'prec_9',
            'nd_2',
            'B7_1',
            'tmax_6_1',
            'pet_total_espr',
            'tmax_3_1',
            'tmax_5_1',
            'tmax',
            'tavg',
            'tmax_2_1',
            'tmin_9',
            'tmax_1_1',
            'tmax_9',
            'tmin_6_1',
            'B6_2',
            'tmax_8_1',
            'tavg_5',
            'tmax_7_1',
            'tmin_8_1',
            'pet_total_wy',
            'wd_est_cy',
            'B7_2',
            'wd_est_lspr',
            'pet_total_lspr',
            'tmin_3_1',
            'tavg_1',
            'pet_total_wy_espr_1',
            'tmmn_p50_cy',
            'tmin_8',
            'pet_total_wy_espr',
            'tavg_11',
            'tavg_6',
            'tavg_3',
            'precip_total_wy_espr',
            'tavg_7',
            'pet_total_wy_smr',
            'tmin',
            'nd_1',
            'tmmx_p10_cy',
            'precip_total_lspr',
            'tmmn_p10_cy',
            'wd_est_wy_espr_1',
            'tmin_1_1',
            'tmax_1',
            'precip_total_smr',
            'tmax_4',
            'B6',
            'tmin_2',
            'wd_est_fl',
            'tmax_3',
            'tavg_2',
            'tmin_10',
            'nd',
            'tmin_11',
            'tmax_5',
            'B2_2',
            'precip_total_fl',
            'tavg_8',
            'B7',
            'tmax_6',
            'wd_est_wy_smr',
            'pet_total_fl',
            'wd_est_wy_espr',
            'tavg_9',
            'wd_est_wy',
            'tmmn_p90_cy',
            'wd_est_espr',
            'tmax_7',
            'tmmx_p50_cy',
            'tmmx_p90_cy',
            'tmin_2_1',
            'B4_2',
            'precip_total_espr',
            'tmax_2',
            'wd_est_smr',
            'pet_total_smr',
            'tavg_10',
            'tmin_1',
            'B4_1',
            'precip_total_wy',
            'tmin_4',
            'precip_total_wy_smr',
            'B5_1',
            'precip_total_wy_espr_1',
            'B3_2',
            'tmin_3',
            'tmin_7',
            'B3',
            'B2',
            'B4',
            'tmin_5',
            'B5',
            'B3_1',
            'tmin_6',
            'B2_1',
            'aspect']


def original_names():
    return ['cdlcrp',
            'cdlclt',
            'nlcd',
            'slope',
            'B7_3',
            'gi_4',
            'B4_3',
            'gi_3',
            'B7_cy',
            'LAT_GCS',
            'evi_3',
            'gi_cy',
            'gsw',
            'B6_3',
            'nw_3',
            'Lon_GCS',
            'evi_4',
            'prec_5',
            'elevation',
            'evi_cy',
            'B2_3',
            'B3_3',
            'B6_cy',
            'B5_3',
            'nd_4',
            'nd_cy',
            'B7_4',
            'nd_3',
            'B7_2',
            'B6_2',
            'tpi_1250',
            'prec_4',
            'prec_10',
            'B5_2',
            'prec_8',
            'B5_cy',
            'B7_1',
            'prec_3',
            'prec_9',
            'tpi_250',
            'prec_1',
            'prec_6',
            'prec_7',
            'evi_2',
            'evi_1',
            'tpi_150',
            'tmin_5_1',
            'B6_4',
            'B5_4',
            'gi_2',
            'nw_2',
            'B6_1',
            'tmax_11',
            'pet_total_fl',
            'tmax_10',
            'tmin_6_1',
            'tmax_9',
            'prec_2',
            'tmax_4_1',
            'tmax_8',
            'tavg_5',
            'tmin_4_1',
            'tmax_1_1',
            'tmax_8_1',
            'nd_2',
            'tmax_2_1',
            'prec_11',
            'tavg_4',
            'tmmn_p10_cy',
            'pet_total_wy_espr',
            'pet_total_wy',
            'pet_total_wy_espr_1',
            'tmin_7_1',
            'tavg_6',
            'tmax_1',
            'prec',
            'tmmx_p10_cy',
            'tavg',
            'tmax_3_1',
            'tmax_5_1',
            'tavg_2',
            'tavg_1',
            'nd_1',
            'tavg_7',
            'tmin_8',
            'pet_total_wy_smr',
            'tmin_3_1',
            'tmax_6_1',
            'B3_4',
            'tmin_9',
            'tmin_1_1',
            'tmax_7_1',
            'tavg_11',
            'aspect',
            'tavg_3',
            'B4_cy',
            'tmin_8_1',
            'tmin_11',
            'B3_cy',
            'gi_1',
            'tavg_10',
            'tmmn_p90_cy',
            'tmin_10',
            'tmin_4',
            'B4_4',
            'B2_4',
            'tmin_2_1',
            'tmmx_p50_cy',
            'B2_cy',
            'B4_2',
            'tmax_4',
            'tmin_5',
            'tavg_8',
            'precip_total_wy',
            'tmax_3',
            'wd_est_lspr',
            'B5_1',
            'tmax_2',
            'precip_total_lspr',
            'tmin',
            'tmin_3',
            'precip_total_wy_smr',
            'tavg_9',
            'tmax',
            'B3_2',
            'B2_2',
            'tmax_7',
            'precip_total_smr',
            'tmax_5',
            'nw_cy',
            'tmin_1',
            'precip_total_wy_espr',
            'pet_total_espr',
            'tmmn_p50_cy',
            'tmax_6',
            'tmin_7',
            'nw_1',
            'precip_total_fl',
            'nw_4',
            'tmin_2',
            'wd_est_smr',
            'tmmx_p90_cy',
            'pet_total_smr',
            'precip_total_wy_espr_1',
            'B3_1',
            'B4_1',
            'B2_1',
            'pet_total_lspr',
            'wd_est_wy',
            'wd_est_fl',
            'wd_est_cy',
            'precip_total_espr',
            'wd_est_wy_smr',
            'tmin_6',
            'wd_est_wy_espr_1',
            'wd_est_wy_espr',
            'wd_est_espr']


def klamath_params():
    return ['nlcd', 'cdlcrp', 'cdl', 'evi_cy', 'slope', 'evi_3',
            'evi_4', 'LAT_GCS', 'B5_3', 'tpi_1250', 'nd_4', 'nd_3',
            'nw_3', 'gi_4', 'tpi_250', 'gi_3', 'elevation', 'tpi_150',
            'gi_cy', 'nd_max_cy', 'B5_cy', 'prec_3', 'prec_8', 'evi_2',
            'B5_4', 'prec_9']


def dec_2020_variables():
    l = [('nlcd', 0.09892328398384927),
         ('cdl', 0.15356820234869015),
         ('LAT_GCS', 0.7150022799817601),
         ('nd_max_cy', 0.8860559566787004),
         ('B5_3', 0.8949152542372881),
         ('nd_4', 0.9392898052691867),
         ('slope', 0.9354618015963512),
         ('B7_4', 0.9433447098976109),
         ('nd_3', 0.9425517702070808),
         ('nd_max_m1', 0.9440993788819876),
         ('B5_4', 0.9413761883205071),
         ('B2_3', 0.9434676162920098),
         ('prec_5', 0.956492027334852),
         ('Lon_GCS', 0.9552572706935123),
         ('nd_max_m2', 0.9585666293393057),
         ('prec_4', 0.9600550964187328),
         ('B6_4', 0.9613537617196433),
         ('B4_4', 0.9639256105758458),
         ('prec_6', 0.9644232948107863),
         ('elevation', 0.9672021419009371),
         ('B4_3', 0.9695067264573991),
         ('tpi_150', 0.9673543414191793),
         ('B7_3', 0.9662746838251609),
         ('tmax_10', 0.9676411515286766),
         ('tpi_250', 0.9600544711756696),
         ('B3_3', 0.9660351201478743),
         ('tmin_7_1', 0.9698256593652212),
         ('prec_7', 0.9678229119123688),
         ('B6_1', 0.9664894846313843),
         ('tpi_1250', 0.9685675797036372),
         ('tmax_8', 0.9711256485450034),
         ('prec_1', 0.9713368493464801),
         ('tmax_11', 0.9636650868878357),
         ('B3_4', 0.9722416608350828),
         ('B6_3', 0.9665914221218962),
         ('B5_2', 0.9668741662961317),
         ('prec_8', 0.9731268503757686),
         ('prec_10', 0.9694085656016316),
         ('prec_11', 0.9715382877795347),
         ('tmin_4_1', 0.967896174863388),
         ('prec', 0.96639231824417),
         ('prec_3', 0.9734030461468516),
         ('tmax_4_1', 0.9683299156983367),
         ('tmin_5_1', 0.969170143718127),
         ('tavg_4', 0.9750736460457738),
         ('B2_4', 0.9738503155996393),
         ('prec_2', 0.9711841512832058),
         ('prec_9', 0.9747147012754531),
         ('nd_2', 0.9709487063095779),
         ('B7_1', 0.9666666666666667)]
    return [x[0] for x in l]


def precision_curve():
    """
    1 ['nlcd'] 0.09892328398384927
2 ['cdl'] 0.15356820234869015
3 ['LAT_GCS'] 0.7150022799817601
4 ['nd_max_cy'] 0.8860559566787004
5 ['B5_3'] 0.8949152542372881
6 ['nd_4'] 0.9392898052691867
7 ['slope'] 0.9354618015963512
8 ['B7_4'] 0.9433447098976109
9 ['nd_3'] 0.9425517702070808
10 ['nd_max_m1'] 0.9440993788819876
11 ['B5_4'] 0.9413761883205071
12 ['B2_3'] 0.9434676162920098
13 ['prec_5'] 0.956492027334852
14 ['Lon_GCS'] 0.9552572706935123
15 ['nd_max_m2'] 0.9585666293393057
16 ['prec_4'] 0.9600550964187328
17 ['B6_4'] 0.9613537617196433
18 ['B4_4'] 0.9639256105758458
19 ['prec_6'] 0.9644232948107863
20 ['elevation'] 0.9672021419009371
21 ['B4_3'] 0.9695067264573991
22 ['tpi_150'] 0.9673543414191793
23 ['B7_3'] 0.9662746838251609
24 ['tmax_10'] 0.9676411515286766
25 ['tpi_250'] 0.9600544711756696
26 ['B3_3'] 0.9660351201478743
27 ['tmin_7_1'] 0.9698256593652212
28 ['prec_7'] 0.9678229119123688
29 ['B6_1'] 0.9664894846313843
30 ['tpi_1250'] 0.9685675797036372
31 ['tmax_8'] 0.9711256485450034
32 ['prec_1'] 0.9713368493464801
33 ['tmax_11'] 0.9636650868878357
34 ['B3_4'] 0.9722416608350828
35 ['B6_3'] 0.9665914221218962
36 ['B5_2'] 0.9668741662961317
37 ['prec_8'] 0.9731268503757686
38 ['prec_10'] 0.9694085656016316
39 ['prec_11'] 0.9715382877795347
40 ['tmin_4_1'] 0.967896174863388
41 ['prec'] 0.96639231824417
42 ['prec_3'] 0.9734030461468516
43 ['tmax_4_1'] 0.9683299156983367
44 ['tmin_5_1'] 0.969170143718127
45 ['tavg_4'] 0.9750736460457738
46 ['B2_4'] 0.9738503155996393
47 ['prec_2'] 0.9711841512832058
48 ['prec_9'] 0.9747147012754531
49 ['nd_2'] 0.9709487063095779
50 ['B7_1'] 0.9666666666666667
51 ['tmax_6_1'] 0.9723035352398108
52 ['pet_total_espr'] 0.974585635359116
53 ['tmax_3_1'] 0.9753616636528029
54 ['tmax_5_1'] 0.9793072424651372
55 ['tmax'] 0.9773857982813207
56 ['tavg'] 0.9763392857142857
57 ['tmax_2_1'] 0.9763832658569501
58 ['tmin_9'] 0.9790020320614135
59 ['tmax_1_1'] 0.9750281214848144
60 ['tmax_9'] 0.9764325323475046
61 ['tmin_6_1'] 0.9784075573549258
62 ['B6_2'] 0.9775255391600454
63 ['tmax_8_1'] 0.978067169294037
64 ['tavg_5'] 0.9746922024623803
65 ['tmax_7_1'] 0.9757521329142343
66 ['tmin_8_1'] 0.9785191956124314
67 ['pet_total_wy'] 0.9794826048171276
68 ['wd_est_cy'] 0.9778079710144928
69 ['B7_2'] 0.9804367606915377
70 ['wd_est_lspr'] 0.9806349921188922
71 ['pet_total_lspr'] 0.9781121751025992
72 ['tmin_3_1'] 0.9808083088733348
73 ['tavg_1'] 0.9777677969907927
74 ['pet_total_wy_espr_1'] 0.9815152898219991
75 ['tmmn_p50_cy'] 0.9796149490373726
76 ['tmin_8'] 0.9774436090225563
77 ['pet_total_wy_espr'] 0.9796057104010877
78 ['tavg_11'] 0.9767285746251958
79 ['tavg_6'] 0.9787622744918931
80 ['tavg_3'] 0.9778581111613195
81 ['precip_total_wy_espr'] 0.9832222477591358
82 ['tavg_7'] 0.9818018422826331
83 ['pet_total_wy_smr'] 0.9809502465262214
84 ['tmin'] 0.9801846430984013
85 ['nd_1'] 0.9845577912962097
86 ['tmmx_p10_cy'] 0.9808644754615038
87 ['precip_total_lspr'] 0.9787762474599232
88 ['tmmn_p10_cy'] 0.9829390354868062
89 ['wd_est_wy_espr_1'] 0.9843643779741672
90 ['tmin_1_1'] 0.9817933545744196
91 ['tmax_1'] 0.9825367647058824
92 ['precip_total_smr'] 0.9823266219239374
93 ['tmax_4'] 0.9801156069364162
94 ['B6'] 0.9836367826688177
95 ['tmin_2'] 0.9812785388127854
96 ['wd_est_fl'] 0.98014440433213
97 ['tmax_3'] 0.9797721460125552
98 ['tavg_2'] 0.9860698789678009
99 ['tmin_10'] 0.980655439235321
100 ['nd'] 0.9818423383525243
101 ['tmin_11'] 0.9796664381996801
102 ['tmax_5'] 0.9836660617059891
103 ['B2_2'] 0.9850279329608939
104 ['precip_total_fl'] 0.9839475469138593
105 ['tavg_8'] 0.9822646657571623
106 ['B7'] 0.9846639603067208
107 ['tmax_6'] 0.9777332442663104
108 ['wd_est_wy_smr'] 0.9805077062556664
109 ['pet_total_fl'] 0.9833447410449464
110 ['wd_est_wy_espr'] 0.9845835250805338
111 ['tavg_9'] 0.9852612773559625
112 ['wd_est_wy'] 0.9849505840071878
113 ['tmmn_p90_cy'] 0.9830009066183137
114 ['wd_est_espr'] 0.9842413327330032
115 ['tmax_7'] 0.9827272727272728
116 ['tmmx_p50_cy'] 0.9848278985507246
117 ['tmmx_p90_cy'] 0.9856164383561644
118 ['tmin_2_1'] 0.9860234445446348
119 ['B4_2'] 0.9803227946053504
120 ['precip_total_espr'] 0.9816326530612245
121 ['tmax_2'] 0.9832525341560159
122 ['wd_est_smr'] 0.9850543478260869
123 ['pet_total_smr'] 0.9839743589743589
124 ['tavg_10'] 0.9836725564750615
125 ['tmin_1'] 0.9798026164792288
126 ['B4_1'] 0.9847645429362881
127 ['precip_total_wy'] 0.9838958746966688
128 ['tmin_4'] 0.9814353633688023
129 ['precip_total_wy_smr'] 0.9859625668449198
130 ['B5_1'] 0.9846674182638105
131 ['precip_total_wy_espr_1'] 0.983739837398374
132 ['B3_2'] 0.9826263537906137
133 ['tmin_3'] 0.9829467939972715
134 ['tmin_7'] 0.981455725544738
135 ['B3'] 0.9815699658703072
136 ['B2'] 0.9838195077484048
137 ['B4'] 0.9834467120181406
138 ['tmin_5'] 0.9822363926212708
139 ['B5'] 0.9832875457875457
140 ['B3_1'] 0.9837324898328061
141 ['tmin_6'] 0.9850145381346455
142 ['B2_1'] 0.9836771707095897
143 ['aspect'] 0.9840054066231133
144 ['gsw'] 0.9812374357828904
    """


def select_variables(n=50):
    """12 August 2021 Variables
    excluded: ('cropland', 0.8299824473544954),
    """
    l = [('nlcd', 0.7202826917050348),
         ('cultivated', 0.6893999449547821),
         ('slope', 0.36318373989279606),
         ('evi_3', 0.21911480248421414),
         ('nw_3', 0.21308324029642675),
         ('B7_3', 0.2035816753073449),
         ('LAT_GCS', 0.19435914276199195),
         ('nd_4', 0.17944280195217932),
         ('evi_4', 0.17927947929664798),
         ('gsw', 0.16927422353380175),
         ('nd_3', 0.1676559927039752),
         ('evi_cy', 0.15565004741797836),
         ('gi_4', 0.1544958435701865),
         ('nd_max_cy', 0.1518437797772695),
         ('gi_cy', 0.1512782542385925),
         ('gi_3', 0.1491951111385143),
         ('B7_cy', 0.1286446900905019),
         ('Lon_GCS', 0.12386826188496355),
         ('B4_3', 0.11697343420709685),
         ('prec_5', 0.1046216613749336),
         ('B7_m1', 0.10151790361854812),
         ('elevation', 0.10042255811065297),
         ('evi_m1', 0.09477629963523237),
         ('B6_3', 0.09289626002373824),
         ('B6_cy', 0.08979382453046862),
         ('B5_3', 0.08979305382035027),
         ('B6_2', 0.0816088195317891),
         ('B7_m2', 0.080219612792706),
         ('evi_m2', 0.08018253794310944),
         ('nd_m1', 0.07549811357259366),
         ('B2_3', 0.07506729518989907),
         ('gi_m1', 0.07382628820397084),
         ('B6_m2', 0.07296382872778079),
         ('prec_8', 0.07195219609623243),
         ('gi_m2', 0.07182825235885029),
         ('tpi_1250', 0.07134573259189245),
         ('prec_3', 0.0705840043257861),
         ('B7_4', 0.06365244702413674),
         ('nd_m2', 0.06295277464429969),
         ('B6_m1', 0.06271964261554216),
         ('B7_2', 0.06258395061634436),
         ('tpi_250', 0.06166084639935801),
         ('prec_4', 0.060072936936905824),
         ('B5_2', 0.059921436259806865),
         ('B3_3', 0.05948872789832256),
         ('prec_9', 0.05765353764987377),
         ('prec_7', 0.0555446284685805),
         ('prec_6', 0.05533366563295573),
         ('tpi_150', 0.05482485779537995),
         ('tmin_5_1', 0.05404145108902089),
         ('tmin_4_1', 0.05200184841549458),
         ('nw_2', 0.05061843715245985),
         ('evi_2', 0.04937099422686948),
         ('tmin_6_1', 0.044879311508737835),
         ('B5_cy', 0.04308042028483345),
         ('evi_1', 0.04286567402640121),
         ('B5_4', 0.04125166415136468),
         ('prec_10', 0.03898577562644369),
         ('B7_1', 0.03897593051334667),
         ('prec_2', 0.03749920022643107),
         ('nd_1', 0.037488995899980615),
         ('tmin_7_1', 0.03626124886843454),
         ('tmax_8', 0.03588917351104601),
         ('gi_2', 0.03578038464072689),
         ('B6_1', 0.03577684428366244),
         ('B6_4', 0.03409339333975213),
         ('nd_2', 0.033243516059239545),
         ('prec_1', 0.032549286438175354),
         ('tmax_11', 0.030429767464632905),
         ('tavg_4', 0.03005755123922534),
         ('tmin_3_1', 0.029988651580328617),
         ('tmax_10', 0.029882561641006423),
         ('prec_11', 0.02960206427033293),
         ('B5_m1', 0.02923798406657787),
         ('tavg_6', 0.028585208017979546),
         ('tavg_5', 0.028456640364560783),
         ('tmax_4_1', 0.02832009672451222),
         ('tmax_2_1', 0.028294657605390475),
         ('prec', 0.027674782321739415),
         ('gi_1', 0.027661045623766582),
         ('tmin_8_1', 0.027358571365328353),
         ('tmax_1_1', 0.027219493321667468),
         ('B5_m2', 0.02685266835032197),
         ('tavg', 0.026434443196659708),
         ('tmax_9', 0.025566679040078186),
         ('tmax_5_1', 0.02447004072009074),
         ('tavg_1', 0.02441104377821587),
         ('tavg_7', 0.02433238732581497),
         ('tavg_11', 0.024173841692659938),
         ('tmax_6_1', 0.023475865756397638),
         ('B4_cy', 0.022756511676715867),
         ('tmin_8', 0.02269260036104464),
         ('aspect', 0.022679061661464824),
         ('tmax_3_1', 0.022485504417196045),
         ('tavg_2', 0.022266084763541073),
         ('tmax_8_1', 0.022158346281348627),
         ('tmax_1', 0.020948196590081784),
         ('tmin_1_1', 0.02064545828507967),
         ('tmmn_p90_cy', 0.020627326105627612),
         ('tmax_7_1', 0.020291623718158097),
         ('B3_m2', 0.020139030465178093),
         ('tmin_9', 0.020138741725147395),
         ('tavg_3', 0.020134384911494198),
         ('tavg_10', 0.019893389843111484),
         ('B4_m1', 0.019842879755960764),
         ('pet_total_fl', 0.01977567595659639),
         ('tmin_11', 0.01976602042204564),
         ('B3_4', 0.018781924360556104),
         ('B4_m2', 0.01838891759474544),
         ('B4_2', 0.018212686439853874),
         ('tmin_10', 0.0173813494861045),
         ('pet_total_wy_espr', 0.01715833992333593),
         ('tmin_2_1', 0.017051494361142405),
         ('precip_total_smr', 0.016807428323893096),
         ('tavg_8', 0.016564728173104972),
         ('pet_total_wy_espr_1', 0.015859946400860202),
         ('tmax_2', 0.015267261716254631),
         ('B3_cy', 0.014884555942322665),
         ('pet_total_wy', 0.014393682735104652),
         ('B3_2', 0.014356728645853233),
         ('B2_2', 0.014354457498254711),
         ('B3_m1', 0.014264923431349249),
         ('tmmn_p10_cy', 0.013912169877768861),
         ('tavg_9', 0.013822614230882322),
         ('tmmx_p10_cy', 0.01382094963983099),
         ('precip_total_lspr', 0.01380873675017625),
         ('B4_4', 0.013111716904240098),
         ('nw_4', 0.013060326441239622),
         ('B5_1', 0.013022729205583825),
         ('precip_total_wy', 0.012609420480815514),
         ('wd_est_lspr', 0.012385450712296076),
         ('tmmn_p50_cy', 0.012015605934614036),
         ('precip_total_fl', 0.011413578512318783),
         ('tmax', 0.01121278638150384),
         ('wd_est_smr', 0.01106873945970865),
         ('tmax_4', 0.010941790450555471),
         ('tmax_7', 0.010880001189676953),
         ('tmax_3', 0.01085398243494991),
         ('B2_m2', 0.010780343652498105),
         ('nw_cy', 0.010701267082033282),
         ('tmax_6', 0.010579694803234231),
         ('precip_total_wy_smr', 0.010480375558579872),
         ('tmmx_p50_cy', 0.010423435635926539),
         ('nw_m2', 0.010385876818348194),
         ('tmax_5', 0.010320225348462608),
         ('wd_est_fl', 0.010315719528076719),
         ('B2_cy', 0.010269908172963043),
         ('pet_total_wy_smr', 0.010245479990594821),
         ('B2_m1', 0.009847566168883775),
         ('wd_est_cy', 0.009814029926218359),
         ('precip_total_wy_espr', 0.009725932326381265),
         ('pet_total_espr', 0.009388840867514316),
         ('tmin_3', 0.009325570509741404),
         ('wd_est_wy', 0.009213460969565344),
         ('pet_total_lspr', 0.009156684341641068),
         ('tmin_1', 0.009135632218147107),
         ('nw_m1', 0.009049692083386868),
         ('nw_1', 0.009032372280978525),
         ('wd_est_wy_espr_1', 0.00896353053075991),
         ('pet_total_smr', 0.008893678446696697),
         ('B2_4', 0.008811152717094466),
         ('precip_total_wy_espr_1', 0.008783434976664237),
         ('B3_1', 0.008558847707920442),
         ('tmin', 0.008521376809115713),
         ('precip_total_espr', 0.008485638608748253),
         ('wd_est_wy_smr', 0.008475930812412089),
         ('tmin_7', 0.008400563288041802),
         ('wd_est_wy_espr', 0.00839607886431794),
         ('B2_1', 0.008350330493048806),
         ('B4_1', 0.008339386192070648),
         ('tmmx_p90_cy', 0.008267930971746653),
         ('wd_est_espr', 0.00808264524170495),
         ('tmin_4', 0.007939899856968937),
         ('tmin_2', 0.007828549155314146),
         ('tmin_5', 0.00751407089821131),
         ('tmin_6', 0.007248090878755033)]
    selected = [x[0] for x in l[:n]]
    return selected


if __name__ == '__main__':
    dec = dec_2020_variables()
    dec = sorted(dec, key=lambda tup: tup[0])
    pprint(dec)
# ========================= EOF ====================================================================
