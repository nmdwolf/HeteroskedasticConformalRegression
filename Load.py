import math
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from INIT import MyScaler

import torch

FOLDER = "./DATA/"

def calculateThreshold(data = None):
    if data is None:
        abs_width = np.abs(max(data) - min(data))
        log_estim = np.log((abs_width * 3) ** 2)
        return -log_estim
    else:
        return -25

def init(choice, seed = 2, test_frac = 0.2, val_frac = 0.5, verbose = False, \
         scaler_y = StandardScaler(), cp_mode = True, overlap = 0, to_torch = False):

    X, y = None, None
    X_train, X_val, X_test = None, None, None
    y_train, y_val, y_test = None, None, None
    mean = None
    scaler = None
    folder = FOLDER

    val_frac = min(1, val_frac + overlap)
    additional = {}

    if choice == "naval":
        frame = pd.read_csv(folder+"naval/data.txt")

        y = frame["GT Compressor decay state coefficient"].to_numpy()
        X = pd.concat([frame.iloc[:, :8], frame.iloc[:, 9:11], frame.iloc[:, 12:-2]], axis = 1).to_numpy()

    elif choice == "appliance":
        frame = pd.read_csv(folder+"appliance/energydata_complete.csv")
        frame = frame.drop(columns=["date"])

        y = frame["Appliances"].to_numpy()
        X = frame.iloc[:, 1:].to_numpy()

    elif choice == "turbine":

        frame = pd.read_excel(folder+"turbine/Folds5x2_pp.xlsx")

        y = frame["PE"].to_numpy()
        X = frame.iloc[:, :4].to_numpy()

    elif choice == "concrete":

        frame = pd.read_excel(folder+"concrete/Concrete_Data.xls")

        y = frame.iloc[:, -1].to_numpy()
        X = frame.iloc[:, :-1].to_numpy()


    elif choice == "puma32H":

        frame_train = pd.read_csv(folder+"puma32H/puma32H.data", header = None)
        frame_test = pd.read_csv(folder+"puma32H/puma32H.test", header = None)
        frame = pd.concat([frame_train, frame_test])

        y = frame.iloc[:, -1].to_numpy()
        X = frame.iloc[:, :-1].to_numpy()

    elif choice == "crime2":

        frame = pd.read_csv(folder+"crime2/communities.data", na_values = '?', header = None)
        frame = frame.iloc[:, 5:]

        if verbose:
            print("Raw input:", frame.shape)
            frame.drop(frame.columns[frame.isnull().any()].tolist(), axis = 1, inplace = True)
            print("Dropped NA:", frame.shape)
            frame = frame.select_dtypes(include = "number")
            print("Dropped Strings:", frame.shape)
        else:
            frame.drop(frame.columns[frame.isnull().any()].tolist(), axis = 1, inplace = True)
            frame = frame.select_dtypes(include = "number")

        y = frame.iloc[:, -1].to_numpy()
        X = frame.iloc[:, :-1].to_numpy()

#        attrib = pd.read_csv(folder + 'crime2/communities_attributes.csv', delim_whitespace = True)
#        data = pd.read_csv(folder + 'crime2/communities.data', names = attrib['attributes'])
#        data = data.drop(columns=['state','county',
#                          'community','communityname',
#                          'fold'], axis=1)

#        data = data.replace('?', np.nan)

        # Impute mean values for samples with missing values
#        from sklearn.impute import SimpleImputer

#        imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')

#        imputer = imputer.fit(data[['OtherPerCap']])
#        data[['OtherPerCap']] = imputer.transform(data[['OtherPerCap']])
#        data = data.dropna(axis=1)
#        X = data.iloc[:, :100].to_numpy()
#        y = data.iloc[:, 100].to_numpy()


    elif (choice == "blog") or (choice == "blogL"):

        frame = pd.read_csv(folder+"blog/blogData_train.csv", na_values = '?', header = None)
        removables = frame.columns[frame.isnull().any()].tolist()
        if verbose:
            print("Raw input:", frame.shape)
            frame.drop(removables, axis = 1, inplace = True)
            print("Dropped NA:", frame.shape)
        else:
            frame.drop(removables, axis = 1, inplace = True)

#         selection = frame.columns[(frame==0).mean() <= .7].to_list()
#         frame = frame.iloc[:, selection]

        y = frame.iloc[:, -1].to_numpy()
        if choice == "blogL":
            y = np.log(y + 1e-10)

        X = frame.iloc[:, :-1].to_numpy()

#         frame_test = pd.read_csv("blog/blogData_test-2012.02.01.00_00.csv", na_values = '?', header = None)
#         frame_test.drop(removables, axis = 1, inplace = True)
#         frame_test = frame_test.iloc[:, selection]

#         y_test = np.ravel(frame_test.iloc[:, -1].to_numpy())
#         X_test = frame_test.iloc[:, :-1]


    elif (choice == "fb1") or (choice == "fb1L"):

        frame_train = pd.read_csv(folder+"fb1/Features_Variant_1.csv", na_values = '?', header = None)
        removables = frame_train.columns[frame_train.isnull().any()].tolist()
        if verbose:
            print("Raw input:", frame_train.shape)
            frame_train.drop(removables, axis = 1, inplace = True)
            print("Dropped NA:", frame_train.shape)
        else:
            frame_train.drop(removables, axis = 1, inplace = True)

        y = frame_train.iloc[:, -1].to_numpy()
        if choice == "fb1L":
            y = np.log(y + 1e-10)

        X = pd.concat([frame_train.iloc[:, :37], frame_train.iloc[:, 38:-1]], axis = 1).to_numpy()

#         frame_test = pd.read_csv(folder+"fb1/Test_Case_1.csv", na_values = '?', header = None)
#         frame_test.drop(removables, axis = 1, inplace = True)
#         for i in range(2,11):
#             frame = pd.read_csv(folder+"fb1/Test_Case_"+str(i)+".csv", na_values = '?', header = None)
#             frame.drop(removables, axis = 1, inplace = True)
#             frame_test = pd.concat([frame_test, frame])

#         y_test = np.ravel(frame_test.iloc[:, -1].to_numpy())
#         X_test = pd.concat([frame_test.iloc[:, :37], frame_test.iloc[:, 38:-1]], axis = 1)

    elif choice == "residential":

        frame = pd.read_excel(folder+"residential/Residential-Building-Data-Set.xlsx", header = 1)
        frame = frame.drop(columns = frame.columns[:5]) # First 5 columns contain temporal and regional information
        X = frame.iloc[:, :-2].to_numpy()
#         y = frame.iloc[:, -2].to_numpy() # price
        y = frame.iloc[:, -1].to_numpy() # construction cost

    elif choice == "traffic":

#        file = arff.loadarff(folder+"traffic/data.arff")
#        frame = pd.DataFrame(file[0])
#        frame["Hour"] = frame["Hour"].astype(np.str)
#        time = frame["Hour"].to_numpy()
#        time = np.array([[time[i][2:-4], time[i][-3:-1]] for i in range(len(time))], dtype = "float")
#        X = frame.iloc[:, 1:-1].to_numpy()
#        X = np.concatenate([time, X], axis = 1)
#        y = frame.iloc[:, -1].to_numpy()

        frame = pd.read_csv(folder+"traffic/data.csv", sep = ";").replace(',','.', regex=True).astype("float32")

        sin_vals = frame["Hour (Coded)"].apply(lambda x: np.sin(2*math.pi*(x + 12)/48))
        cos_vals = frame["Hour (Coded)"].apply(lambda x: np.cos(2*math.pi*(x + 12)/48))
        X = pd.concat([sin_vals, cos_vals, frame.iloc[:, 1:-1]], axis = 1).to_numpy()

#        X = frame.iloc[:, :-1].to_numpy()
        y = frame.iloc[:, -1].to_numpy()
        
    elif choice == "star":

        df = pd.read_csv(folder + "star/STAR.csv")
        df.loc[df['gender'] == 'female', 'gender'] = 0
        df.loc[df['gender'] == 'male', 'gender'] = 1

        df.loc[df['ethnicity'] == 'cauc', 'ethnicity'] = 0
        df.loc[df['ethnicity'] == 'afam', 'ethnicity'] = 1
        df.loc[df['ethnicity'] == 'asian', 'ethnicity'] = 2
        df.loc[df['ethnicity'] == 'hispanic', 'ethnicity'] = 3
        df.loc[df['ethnicity'] == 'amindian', 'ethnicity'] = 4
        df.loc[df['ethnicity'] == 'other', 'ethnicity'] = 5

        df.loc[df['stark'] == 'regular', 'stark'] = 0
        df.loc[df['stark'] == 'small', 'stark'] = 1
        df.loc[df['stark'] == 'regular+aide', 'stark'] = 2

        df.loc[df['star1'] == 'regular', 'star1'] = 0
        df.loc[df['star1'] == 'small', 'star1'] = 1
        df.loc[df['star1'] == 'regular+aide', 'star1'] = 2

        df.loc[df['star2'] == 'regular', 'star2'] = 0
        df.loc[df['star2'] == 'small', 'star2'] = 1
        df.loc[df['star2'] == 'regular+aide', 'star2'] = 2

        df.loc[df['star3'] == 'regular', 'star3'] = 0
        df.loc[df['star3'] == 'small', 'star3'] = 1
        df.loc[df['star3'] == 'regular+aide', 'star3'] = 2

        df.loc[df['lunchk'] == 'free', 'lunchk'] = 0
        df.loc[df['lunchk'] == 'non-free', 'lunchk'] = 1

        df.loc[df['lunch1'] == 'free', 'lunch1'] = 0
        df.loc[df['lunch1'] == 'non-free', 'lunch1'] = 1

        df.loc[df['lunch2'] == 'free', 'lunch2'] = 0
        df.loc[df['lunch2'] == 'non-free', 'lunch2'] = 1

        df.loc[df['lunch3'] == 'free', 'lunch3'] = 0
        df.loc[df['lunch3'] == 'non-free', 'lunch3'] = 1

        df.loc[df['schoolk'] == 'inner-city', 'schoolk'] = 0
        df.loc[df['schoolk'] == 'suburban', 'schoolk'] = 1
        df.loc[df['schoolk'] == 'rural', 'schoolk'] = 2
        df.loc[df['schoolk'] == 'urban', 'schoolk'] = 3

        df.loc[df['school1'] == 'inner-city', 'school1'] = 0
        df.loc[df['school1'] == 'suburban', 'school1'] = 1
        df.loc[df['school1'] == 'rural', 'school1'] = 2
        df.loc[df['school1'] == 'urban', 'school1'] = 3

        df.loc[df['school2'] == 'inner-city', 'school2'] = 0
        df.loc[df['school2'] == 'suburban', 'school2'] = 1
        df.loc[df['school2'] == 'rural', 'school2'] = 2
        df.loc[df['school2'] == 'urban', 'school2'] = 3

        df.loc[df['school3'] == 'inner-city', 'school3'] = 0
        df.loc[df['school3'] == 'suburban', 'school3'] = 1
        df.loc[df['school3'] == 'rural', 'school3'] = 2
        df.loc[df['school3'] == 'urban', 'school3'] = 3

        df.loc[df['degreek'] == 'bachelor', 'degreek'] = 0
        df.loc[df['degreek'] == 'master', 'degreek'] = 1
        df.loc[df['degreek'] == 'specialist', 'degreek'] = 2
        df.loc[df['degreek'] == 'master+', 'degreek'] = 3

        df.loc[df['degree1'] == 'bachelor', 'degree1'] = 0
        df.loc[df['degree1'] == 'master', 'degree1'] = 1
        df.loc[df['degree1'] == 'specialist', 'degree1'] = 2
        df.loc[df['degree1'] == 'phd', 'degree1'] = 3

        df.loc[df['degree2'] == 'bachelor', 'degree2'] = 0
        df.loc[df['degree2'] == 'master', 'degree2'] = 1
        df.loc[df['degree2'] == 'specialist', 'degree2'] = 2
        df.loc[df['degree2'] == 'phd', 'degree2'] = 3

        df.loc[df['degree3'] == 'bachelor', 'degree3'] = 0
        df.loc[df['degree3'] == 'master', 'degree3'] = 1
        df.loc[df['degree3'] == 'specialist', 'degree3'] = 2
        df.loc[df['degree3'] == 'phd', 'degree3'] = 3

        df.loc[df['ladderk'] == 'level1', 'ladderk'] = 0
        df.loc[df['ladderk'] == 'level2', 'ladderk'] = 1
        df.loc[df['ladderk'] == 'level3', 'ladderk'] = 2
        df.loc[df['ladderk'] == 'apprentice', 'ladderk'] = 3
        df.loc[df['ladderk'] == 'probation', 'ladderk'] = 4
        df.loc[df['ladderk'] == 'pending', 'ladderk'] = 5
        df.loc[df['ladderk'] == 'notladder', 'ladderk'] = 6


        df.loc[df['ladder1'] == 'level1', 'ladder1'] = 0
        df.loc[df['ladder1'] == 'level2', 'ladder1'] = 1
        df.loc[df['ladder1'] == 'level3', 'ladder1'] = 2
        df.loc[df['ladder1'] == 'apprentice', 'ladder1'] = 3
        df.loc[df['ladder1'] == 'probation', 'ladder1'] = 4
        df.loc[df['ladder1'] == 'noladder', 'ladder1'] = 5
        df.loc[df['ladder1'] == 'notladder', 'ladder1'] = 6

        df.loc[df['ladder2'] == 'level1', 'ladder2'] = 0
        df.loc[df['ladder2'] == 'level2', 'ladder2'] = 1
        df.loc[df['ladder2'] == 'level3', 'ladder2'] = 2
        df.loc[df['ladder2'] == 'apprentice', 'ladder2'] = 3
        df.loc[df['ladder2'] == 'probation', 'ladder2'] = 4
        df.loc[df['ladder2'] == 'noladder', 'ladder2'] = 5
        df.loc[df['ladder2'] == 'notladder', 'ladder2'] = 6

        df.loc[df['ladder3'] == 'level1', 'ladder3'] = 0
        df.loc[df['ladder3'] == 'level2', 'ladder3'] = 1
        df.loc[df['ladder3'] == 'level3', 'ladder3'] = 2
        df.loc[df['ladder3'] == 'apprentice', 'ladder3'] = 3
        df.loc[df['ladder3'] == 'probation', 'ladder3'] = 4
        df.loc[df['ladder3'] == 'noladder', 'ladder3'] = 5
        df.loc[df['ladder3'] == 'notladder', 'ladder3'] = 6

        df.loc[df['tethnicityk'] == 'cauc', 'tethnicityk'] = 0
        df.loc[df['tethnicityk'] == 'afam', 'tethnicityk'] = 1

        df.loc[df['tethnicity1'] == 'cauc', 'tethnicity1'] = 0
        df.loc[df['tethnicity1'] == 'afam', 'tethnicity1'] = 1

        df.loc[df['tethnicity2'] == 'cauc', 'tethnicity2'] = 0
        df.loc[df['tethnicity2'] == 'afam', 'tethnicity2'] = 1

        df.loc[df['tethnicity3'] == 'cauc', 'tethnicity3'] = 0
        df.loc[df['tethnicity3'] == 'afam', 'tethnicity3'] = 1
        df.loc[df['tethnicity3'] == 'asian', 'tethnicity3'] = 2

        df = df.dropna()

        grade = df["readk"] + df["read1"] + df["read2"] + df["read3"]
        grade += df["mathk"] + df["math1"] + df["math2"] + df["math3"]
        grade /= 8

        names = df.columns
        target_names = names[8:16]
        data_names = np.concatenate((names[1:8],names[17:]))
        X = df.loc[:, data_names].astype(float).to_numpy()
        y = grade.to_numpy()

    elif choice[:5] == "synth":

        X = np.load(folder + "SYNTHETIC/" + choice[5:] + "_" + str(seed) + "_X.npy")
        y = np.load(folder + "SYNTHETIC/" + choice[5:] + "_" + str(seed) + "_y.npy")
        mean = np.load(folder + "SYNTHETIC/" + choice[5:] + "_" + str(seed) + "_mean.npy")
        var = np.load(folder + "SYNTHETIC/" + choice[5:] + "_" + str(seed) + "_var.npy")
    
    else:
        raise Exception("Unknown data set:" + str(choice))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_frac, random_state = seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = val_frac, random_state = seed)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    scaler_y = MyScaler(scaler_y.fit(y_train.reshape(-1, 1)))
    y_train = scaler_y.transform(y_train)
    y_val = scaler_y.transform(y_val)
    y_test = scaler_y.transform(y_test)

    if not cp_mode:
        X_train = np.concatenate([X_train, X_val], axis = 0)
        y_train = np.concatenate([y_train, y_val], axis = 0)
        X_val = X_train
        y_val = y_train

    if to_torch:
        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).float()
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).float()
        X_val = torch.from_numpy(X_val).float()
        y_val = torch.from_numpy(y_val).float()
        X_test = torch.from_numpy(X_test).float()
        y_test = torch.from_numpy(y_test).float()

    additional.update({"X": X, "y": y, "X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val, "X_test": X_test, "y_test": y_test, "features": X_train.shape[1], "scaler": scaler, "scaler_y": scaler_y, "folder": folder, "choice": choice, "val_frac": val_frac, "test_frac": test_frac, "identifier": choice, "seed": seed})
    
    if mean is not None:
        
        mean_train, mean_test, var_train, var_test = train_test_split(mean, var, test_size = test_frac, random_state = seed)
        mean_train, mean_val, var_train, var_val = train_test_split(mean_train, var_train, test_size = val_frac, random_state = seed)

        mean_train = scaler_y.transform(mean_train)
        mean_val = scaler_y.transform(mean_val)
        mean_test = scaler_y.transform(mean_test)
        var_train = var_train / scaler_y.var_
        var_val = var_val / scaler_y.var_
        var_test = var_test / scaler_y.var_

        if to_torch:
            mean = torch.from_numpy(mean).float()
            var = torch.from_numpy(var).float()
            mean_train = torch.from_numpy(mean_train).float()
            var_train = torch.from_numpy(var_train).float()
            mean_val = torch.from_numpy(mean_val).float()
            var_val = torch.from_numpy(var_val).float()
            mean_test = torch.from_numpy(mean_test).float()
            var_test = torch.from_numpy(var_test).float()

        additional.update({"mean": mean, "mean_train": mean_train, "mean_val": mean_val, "mean_test": mean_test, \
                           "var": var, "var_train": var_train, "var_val": var_val, "var_test": var_test})

    return additional
