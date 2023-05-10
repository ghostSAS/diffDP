import matplotlib.pyplot as plt

def plot_time_series(t,data,split=0,nrows=1,ncols=1):
    if not split:
        plt.plot(t,data)
    else:
        for i in range(nrows*ncols):
            ax = plt.subplot(nrows,ncols,i)
            ax.plot(t,data[:,i])
    
    plt.show()