from globals import *


for i in range(10):

    # make data
    # Z = np.random.randint(0, 255, (100, 100, 3))
    Z = np.zeros((100, 100, 3))
    Z[random.randint(1, 99), random.randint(1, 99)] = np.array([10, 10, 10])

    # plot
    fig, ax = plt.subplots()
    ax.imshow(Z, origin='lower')
    plt.show()



