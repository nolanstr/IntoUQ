import numpy as np
import matplotlib.pyplot as plt
import pickle


def convert_to_numpy(array):
    a = np.zeros(array.shape)
    for i in range(array.shape[0]):
        import pdb;pdb.set_trace()
        a[i] = float(array[i,0].item())
    return a

def get_data():

    runs = np.arange(1, 101)
    X, y = [], []
    for run in runs:
        tag = str(run).zfill(3)
        FILE = open(f"./model_outputs/output_{tag}.pkl", "rb")
        data = pickle.load(FILE)
        FILE.close()
        displacements = data["displacements"]
        coords = data["coords"]
        X_ = np.unique(coords[:,0].round(decimals=0))
        max_disp = [np.min(displacements[np.argwhere(coords[:,0]==_x).flatten(),-1]).tolist() for _x in X_]
        E = data["E"]
        nu = data["nu"]
        E = np.full((X_.shape[0], 1), E)
        nu = np.full((X_.shape[0], 1), nu)

        _X = np.hstack((X_.reshape((-1,1)), E, nu))
        _y = max_disp#.reshape((-1,1))
        X.append(_X)
        y.append(_y)
    X = np.vstack(X)
    y = np.concatenate(y).reshape((-1,1))
    plt.scatter(X[:,0].flatten(), y.flatten())
    plt.show()
    return X, y

if __name__ == "__main__":
    X, y = get_data() 
    np.save("data_X.npy", X)
    np.save("data_y.npy", y)
