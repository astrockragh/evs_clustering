import numpy as np
import pandas as pd
from astropy.cosmology import Planck18 as cosmo
import cosmic_variance as cv
from tqdm import tqdm
# Generate the values of z for bins of fixed comoving distance

def scale_cv( side1, side2, z_OD, z_r, low_z = 3, max_z = 15, n_z = 10):
    # highest z to evaluate at

    # lowest z

    # Number of points to evaluate at
    # The time to run goes approximately like n_z*4 minutes
    # The points will be placed at uniform comoving distance from each other

    # Define the parameters of the overdensity we're evaluating
    high_z = np.copy(low_z)
    # define the comoving distance in the z-direction
    dist0 = cosmo.comoving_distance(z_OD+z_r)-cosmo.comoving_distance(z_OD-z_r)

    dist = 0
    bins = []
    while high_z < max_z:
        high_z += 0.0001
        dist = cosmo.comoving_distance(high_z)-cosmo.comoving_distance(low_z)
        if dist.value > dist0.value:
            bins.append(np.round(low_z, 4))
            low_z = np.copy(high_z)

    bins = np.array(bins)
    low_zs, high_zs = bins[::len(bins)//n_z], bins[1::len(bins)//n_z]
    print(f'The redshifts to evaluate at are {low_zs}')
    #### these arguments are optional ####
    acc = 'low' # accuracy of the calculation, 'low' or 'high, low is default, faster and sufficient for almost all applications
    verbose = True # if True, will print out the progress of the calculation, default is False

    #If you want to use a different cosmology, you can specify it by the following in the get_cv call
    # OmegaM = 0.308, OmegaL = 0.692, OmegaBaryon = 0.022/(0.678)**2 sigma8 = 0.82, ns = 0.96, h = 0.678

    print(f'Calculating the cosmic variance, estimated (very uncertain) time is {(len(low_zs)*240)//60:.2f} minutes')
    ## array to save cv values to scale the base_cv
    cv_scales = []
    for  l, h in tqdm(zip(low_zs, high_zs), total = len(low_zs)):
        cv_scale = cv.get_cv(side1, side2, np.array([l, h]), name = None, acc=acc, verbose = verbose)
        cv_scales.append(cv_scale)

    ## stack the new sigma_dm (dm = dark matter)
    cvs_new = pd.concat(cv_scales).reset_index(drop =True)
    zc = cvs_new['zmid']
    cv_dm_new = cvs_new['cv_dm']

    ## read the dataframe containing the bias values
    cvs_df = pd.read_csv('dfs/base_cv.csv')

    ## get the mass columns
    cv_cols = list( cvs_df.columns[2:] )
    scaled_df = []
    ## loop over the redshift points
    for i in range(len(zc)):
        # get the redshift row
        cv_row = cvs_df.iloc[np.argmin(np.abs(cvs_df['z'].to_numpy()-zc.to_numpy()[i]))]
        # scale the cosmic variance at that redshift by the dark matter sigmas
        cv_row[cv_cols] = cv_row[cv_cols].to_numpy()*(cv_dm_new.iloc[i]/cv_row['sigdm'])
        cv_row['dz'] = cvs_new.iloc[i]['dz']
        cv_row['z'] = cvs_new.iloc[i]['zmid']

        scaled_df.append(cv_row.to_numpy())

    # save the dataframe
    new_scaled_df = pd.DataFrame(scaled_df, columns = cvs_df.columns)
    new_scaled_df.to_csv(f'dfs/scaled_{z_OD}_{z_r}.csv')
    return new_scaled_df
