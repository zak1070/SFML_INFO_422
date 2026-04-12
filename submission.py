import numpy as np

def get_narx1_data(n_samples, noise_var, u_in=None):
    """
    Generates synthetic data for the NARX1 pilot.
    Takes the number of samples and the noise variance. If no input signal is provided, it feeds the system with random uniform noise just to excite it.
    """
    # Setting up empty arrays. 
    # Using zeros is handy here because it automatically takes care of the y(0)=0 initial conditions
    out_y1 = np.zeros(n_samples + 1)
    out_y2 = np.zeros(n_samples + 1)
    
    # Create a uniform input if we didn't pass one
    if u_in is None:
        u_in = np.random.uniform(-1, 1, n_samples + 1)
        
    # Get standard deviation from variance for the normal distribution
    noise_std = np.sqrt(noise_var)
    w1 = np.random.normal(0, noise_std, n_samples + 1)
    w2 = np.random.normal(0, noise_std, n_samples + 1)

    # Main simulation loop
    # Starting at k=1 since the formula looks one step back
    for k in range(1, n_samples):
        out_y1[k+1] = 0.5 * out_y2[k-1] + np.sin(out_y2[k]) + 0.3 * u_in[k-1] + w1[k+1]
        out_y2[k+1] = 0.5 * out_y1[k-1] + np.sin(out_y1[k]) + 0.2 * u_in[k] + w2[k+1]
        
    # Reshaping u_in so it's a proper column vector
    return u_in.reshape(-1, 1), np.column_stack((out_y1, out_y2))


def get_narx2_data(n_samples, noise_var, u_in=None):
    """
    Generates data for the NARX2 pilot.
    The formula needs at least 2 past steps to compute the next one, so the loop has to start at k=2.
    """
    out_y1 = np.zeros(n_samples + 1)
    out_y2 = np.zeros(n_samples + 1)
    
    # Handling the 2D input signal
    if u_in is None:
        u1 = np.random.uniform(-1, 1, n_samples + 1)
        u2 = np.random.uniform(-1, 1, n_samples + 1)
        u_in = np.column_stack((u1, u2))
    else:
        # Split the provided matrix into two separate signals for the formula
        u1 = u_in[:, 0]
        u2 = u_in[:, 1]
        
    noise_std = np.sqrt(noise_var)
    w1 = np.random.normal(0, noise_std, n_samples + 1)
    w2 = np.random.normal(0, noise_std, n_samples + 1)

    # Loop starts at 2 to avoid negative indexing
    for k in range(2, n_samples):
        
        # Breaking down the y1 equation to keep it readable
        num_y1 = out_y1[k] * out_y1[k-1] * out_y1[k-2] * (out_y1[k-2] - 1) * u2[k-1] + u2[k]
        den_y1 = 1 + out_y2[k-1]**2 + out_y2[k-2]**2
        out_y1[k+1] = (num_y1 / den_y1) + w1[k+1]
        
        # Breaking down the y2 equation
        num_y2 = out_y2[k] * out_y2[k-1] * out_y2[k-2] * (out_y2[k-2] - 1) * u1[k-1] + u1[k]
        den_y2 = 1 + out_y1[k-1]**2 + out_y1[k-2]**2
        out_y2[k+1] = (num_y2 / den_y2) + w2[k+1]
        
    return u_in, np.column_stack((out_y1, out_y2))