import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter





def FrankeFunction(x,y,noisefactor):
        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
        term5 = noisefactor*np.random.normal(0,1,(len(x),len(y)))
        return term1 + term2 + term3 + term4 + term5
        

def create_X(x, y, n ):
        if len(x.shape) > 1:
                x = np.ravel(x)
                y = np.ravel(y)

        N = len(x)
        l = int((n+1)*(n+2)/2)          # Number of elements in beta                                                               
        X = np.ones((N,l))

        for i in range(1,n+1):
                q = int((i)*(i+1)/2)
                for k in range(i+1):
                        X[:,q+k] = (x**(i-k))*(y**k)

        return X

def init_data(N,noisefactor):
    """
    Input: 
        N = datapoints, 
        noisefactor = scalar for gaussian distributed noise
    
    Output: 
        x_ = meshgrid of datapoints to compute the Franke Function in X directions
        y_ = meshgrid of datapoints to compute the Franke Function in Y directions
        z  = 2D array of the frankefunction

    """
    x, y = np.linspace(0,1,N), np.linspace(0,1,N)
    x_,y_ = np.meshgrid(x,y)
    z = FrankeFunction(x_, y_, noisefactor)
    return x_, y_, z    

def Scaling(X_train, X_test):
        (N, p) = X_train.shape
        if p > 1:
            scaler = StandardScaler()
            scaler.fit(X_train[:,1:])
            X_train = scaler.transform(X_train[:,1:])
            X_test = scaler.transform(X_test[:,1:])
            
            # Adding the intercept after the scaling, as the StandardScaler removes the 1s in the first column.
            intercept_train = np.ones((len(X_train),1))
            intercept_test = np.ones((len(X_test),1))
            X_train = np.concatenate((intercept_train,X_train),axis=1)
            X_test = np.concatenate((intercept_test,X_test),axis=1)
        else:
            X_train = X_train
            X_test = X_test
        return X_train, X_test

def X_scaling(X):
        (N, p) = X.shape
        if p > 1:
            scaler = StandardScaler()
            scaler.fit(X[:,1:])
            X = scaler.transform(X[:,1:])            
            # Adding the intercept after the scaling, as the StandardScaler removes the 1s in the first column.
            intercept = np.ones((len(X),1))
            X = np.concatenate((intercept,X),axis=1)

        else:
            X = X
        return X
    
def PreProcess(x, y,z, test_size, n):
    """
    Input:
        x         : The meshgrid datapoints for x
        y         : The meshgrid datapoints for y
        z         : The FrankeFunction datapoints
        test_size : The test_size for testing training datasplit
        n         : The maximum number of polynomial degree, (x+y)^n, that the functions can fit.
    
    Ravels the x, y and z
    
    Output:
        X_train   : Training Design matrix
        X_test    : Testing Design Matrix
        z_train   : Train Data
        z_test    : Test Data
    """
    z = np.ravel(z)
    # Creating design matrix for the maximum polynomial degree. 
    X = create_X(x,y,n)

    # Test Train splitting of data
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

    # Scaling the train and test set
    X_train, X_test = Scaling(X_train, X_test)
    return X_train, X_test, z_train, z_test 
    

def MSE(y,ypred):
    MSE = np.mean((y-ypred)**2)
    return MSE

def R2(y,ypred):
    return 1-np.sum((y-ypred)**2)/np.sum((y-np.mean(y))**2)


def ErrBiasVar(z_test,z_pred):
    """
    Takes in corresponding test and predicted data and computes the 
    error, bias and variance
    """
    error = np.mean( np.mean((z_test - z_pred)**2, axis=1, keepdims=True) )
    bias  = np.mean( (z_test - np.mean(z_pred, axis=1, keepdims=True))**2 )
    variance = np.mean( np.var(z_pred, axis=1, keepdims=True) )
    return error, bias, variance

def SVDinv(A):
    ''' 
    Takes as input a numpy matrix A and returns inv(A) based on singular value decomposition (SVD).
    # SVD is numerically more stable than the inversion algorithms provided by
    # numpy and scipy.linalg at the cost of being slower. (Hjorth-Jensen, 2020) 
    '''
    
    U, s, VT = np.linalg.svd(A, full_matrices = False)
    invD = np.diag(1/s)
    UT = np.transpose(U); V = np.transpose(VT);
    return np.matmul(V,np.matmul(invD,UT))

def Shuffle_Data(x,z, replacement=True):
    if replacement == True:
        size = len(x)
        index = np.random.randint(1, size, size=size)
        x = x[index,:]
        z = z[index]
    else:
        shuffle_index = np.arange(len(x))
        np.random.shuffle(shuffle_index)
        x, z = x[shuffle_index], z[shuffle_index]
    return x, z

def k_foldsplit(N, folds):
    fold_size = np.int(N/folds)
    
    foldmask = np.ones((N,folds), dtype=bool)
    
    for j in range(folds):
        mask = np.ones(N, dtype=bool)
        index = np.arange(fold_size)+j*fold_size
        mask[index] = False
        
        foldmask[:,j] = mask
    return foldmask

def surfplotter(x,y,BETA,n, title):

   
    z_total = create_X(x,y,n) @ BETA
    z_total = z_total.reshape(x.shape)
    
    # Plotting the ith polynomial prediction on [0,1]x[0,1]
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(x, y, z_total, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # Customize the z axis.
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_title(title, fontsize = 16)
    ax.set_zlim(-0.10, 1.40)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()



def betaplot(beta, title):
    fig, ax = plt.subplots(figsize=(15,6))
    x_pos = np.arange(len(beta))
    ax.bar(x_pos, beta, alpha=0.5,capsize=10)
    ax.set_ylabel('Beta coeffient', fontsize=14)
    ax.set_xticks(x_pos)
    #ax.set_xticklabels(labels, fontsize=14)
    ax.set_title(title, fontsize=20)

    plt.show()

def terrainInit(filename, x_start, x_end, y_start, y_end):
    from imageio import imread
    terrain1 = imread(filename)
    # Initializing the data
    z = np.array(terrain1[x_start:x_end, y_start:y_end])
    return z

def terrain_sampling(x,y,z):
    """
    Samples down the terrain data to fit the length of x and y,
    effectively reducing the resolution of the data.
    """
    z_sample = np.zeros((len(x),len(y)))
    zx, zy = z.shape
    
    for i in range(len(x)):
        for j in range(len(y)):
            z_sample[i,j] = z[np.int(zx/len(x))*i, np.int(zy/len(y))*j]
    return z_sample