import numpy as np
from scipy import integrate
from scipy.stats import norm, uniform
from scipy.special import gammaincinv
import hmf, h5py, astropy, sys, tqdm, time, os, warnings
import pandas as pd
from astropy.cosmology import Planck18

pd.options.mode.chained_assignment = None  # default='warn'
warnings.filterwarnings("ignore", category=RuntimeWarning)

add_sig = 0 #do you think that this field is additionally overdense compared to the rest of the Universe? Set 0 for assuming that the field is average
od = 'test_new' # part of the next name
out_str = f'{od}_OD_fast_{add_sig}' # where to save this, as data/<out_str>.h5
recalculate_cv = True # should we recalculate the clustering strength? This takes a long time to do compared to the rest of the code, so if you've already done the calculation of the cosmic variance, set this to False to speed up the calculation
OD_geometry = 'ellipsoid' ##options are 'ellipsoid', 'cylinder', and 'square'. Note that all distances supplied are supposed to be radii, NOT diameters 

if od == 'test_new':
    z_r = 0.01
    z_OD = 3.21
    dDEC_OD, dRA_OD = 100/3600, 100/3600 # in degrees
    Ntrials = int(1e5)

## The parameters of the survey, for the calculation of the survey
## We need \DeltA RA, \Delta DEC and the minimum/maximum redshift
dRA_survey, dDEC_survey = 1402/3600, 844.6/3600 # in degrees
z_min_survey, z_max_survey = 3, 5
z_min_survey = float(z_min_survey)
z_max_survey = float(z_max_survey)

## Parameters for computing the cosmic variance ##
## Calculations will be centered in a uniform range from low_z to max_z in n_z steps (uniform in comoving distance) ##
n_z = 20 # Number of redshifts to do the calculation at (the code scales linearly with this parameter)
low_z = z_min_survey/1.05 # The minimum redshift to consider calculating the cosmic variance for
max_z = 18 # The minimum redshift to consider calculating the cosmic variance for

## If you do not need to precalculate the cosmic variance, the filename for the dataframe with cosmic variance values should be here
if not recalculate_cv:
    # file with pre-computed cosmic variance
    df_loc_precomputed = f'dfs/scaled_{z_OD}_{z_r}.csv' #do it automatically
    # df_loc_precomputed = f'dfs/scaled_3.21_0.073.csv' #or put it in by hand

# define basic parameters
cosmo = Planck18 # which cosmology to use
all_sky = 41252.96125 # number of square degrees in the sky
f = np.pi/180 # change between degrees and radians
baryon_frac = 0.16 # from Planck 2018 ( 0.0224/(0.12+0.0224) )

## Parameters for defining the halo mass calculation ##
mmin = 3 # Minimum halo mass for HMF
mmax = 15 # Maximum halo mass for HMF

## Parameters for defining the stellar mass calculation ##
## I would advise against changing these unless absolute necessary, it is likely to mess up things in the integrations ##

maxMass = 12.5 # what's the maximum galaxy mass to consider, this should be solidly (~1dex) above the actual masses you're interested
minMass = 7.0 # minimum mass to consider
dM = 0.5 #mass scale for computing clustering, do not change unless you're changing the cosmic variance calculator
dlog10m = 0.01 #numerical HMF resolution, should be <<mass resolution
Ndm = int(dM//dlog10m) #number of steps in a dM dex mass interval
Nm = 100 #mass evaluation grid resolution, set so that we get dM/Nm = 0.005
mbins = np.arange(minMass, maxMass+dM/Nm-1e-10, dM/Nm)
mbins = np.round(mbins, 3) # necessary to match with dataframe 

# Method to take trials from the inverse CDF
# of a gamma distribution with a given variance and mean
def trial_OD(x, sig_cv, mean):
    """
    Generate points from the inverse CDF from a gamma function that combines cosmic variance 
    and Poisson noise. This corresponds to the number count distribution for A GIVEN MASS BIN in the SMF.

    Parameters
    ----------
    x : array-like
        The input probabilities (values between 0 and 1) at which to 
        compute the values from the inverse CDF of the gamma distribution.
    sig_cv : float
        The coefficient of variation due to cosmic variance for galaxies in that mass bin. This quantifies the 
        fractional root variance (not uncertainty, it's a real, physical variance) coming from variations in large-scale structure.    
    mean : float
        The expected (mean) value of number count of galaxies in the mass bin. Can be computed from the HMF.

    Returns
    -------
    inv : array-like 
        The gamma distribution values at the different points on the CDF, corresponding to the probabilities given in the x - array.

    Notes
    -----
    The total variance is:
        var = (σ_CV * mean)^2 + mean

    The distribution is approximated as a gamma distribution with:
        Shape parameter: k = mean^2 / var
        Scale parameter: t = var / mean

    Sampling is performed using the inverse incomplete gamma function:
        rand = t * gammaincinv(k, x)

    This method is particularly important when the contribution from cosmic variance 
    is non-negligible which is usually the case for the overdensities that we are interested in.
    If Poisson variance is much higher than the cosmic variance, the results here may not make perfect sense.
    """
    var = sig_cv**2 * mean**2 + mean  # Total variance: cosmic variance + Poisson
    k = mean**2 / var                 # Shape parameter of the gamma distribution
    t = var / mean                    # Scale parameter of the gamma distribution
    inv = t * gammaincinv(k, x, out=None)  # Sampling via inverse incomplete gamma
    return inv

def sigma_delta_evs(z_r, z_OD, dDEC_OD, dRA_OD, z_min_survey, z_max_survey, dRA_survey, dDEC_survey, 
                    geometry='ellipsoid', n_samp=None, add_sig=0.0):
    """
    Calculate the expected extreme value statistics (EVS) dsitribution for overdensity percentile within a survey volume.

    This function estimates the expected distribution of the density percentile fo the most extreme overdensities 
    in a survey given the survey geometry and the geometry of the overdensity regions.

    Parameters
    ----------
    z_r : float
        The redshift depth of the overdensity region.
    z_OD : float
        The central redshift of the overdensity region.
    dDEC_OD : float
        The angular size of the overdensity region in Declination (degrees).
    dRA_OD : float
        The angular size of the overdensity region in Right Ascension (degrees).
    z_min_survey : float
        The minimum redshift covered by the survey.
    z_max_survey : float
        The maximum redshift covered by the survey.
    dRA_survey : float
        The total angular width of the survey in Right Ascension (degrees).
    dDEC_survey : float
        The total angular width of the survey in Declination (degrees).
    geometry : {'ellipsoid', 'cylinder', 'rectangle'}, optional
        The assumed geometry of the overdensity volume. Default is 'ellipsoid'.
    
    n_samp : int, optional
        The number of sampling points for the extreme value distribution. The number should be significantly higher than (V_{survey}/V_{OD}).
        If None, a fallback value is used.

    add_sig : float, optional
        Additional sigma to add/subtract to the sigma_OD, corresponds to whether or not the field is over/under-dense. Keep as 0 to assume an average field.

    Returns
    -------
    evs : array-like
        The probability density function of the extreme values over the sampling grid `x`.
    x : array-like
        The sampling grid for the percentiles.
    V_OD_comoving : float
        The volume of the overdensity region in (comoving) Mpc^3.
    V_survey_comoving : float
        The total survey volume in (comoving) Mpc^3.

    Notes
    -----
    - All Right Ascension (RA) and Declination (DEC) values should be provided in degrees.
    - The function accounts for cosmic expansion by converting between proper volumes and comoving volumes for the overdensity.
    - The calculation is not intended for cases where the number of potential overdensity subvolumes (N) in the total volme is small (N < 30 is probably a suitable limit).
    - The volume calculation uses the specified geometry:
        - 'ellipsoid': volume of an ellipsoid
        - 'cylinder': volume of a cylinder
        - 'rectangle': rectangular prism approximation (includes factor of 8 to go from radii to diameters)
    - The expected extreme value distribution is derived from the uniform order statistics.
    """
    # Convert angular sizes to physical distances
    d1 = dDEC_OD * np.pi / 180 * cosmo.comoving_distance(z_OD)
    d2 = dRA_OD * np.pi / 180 * cosmo.comoving_transverse_distance(z_OD)
    dz = cosmo.comoving_distance(z_OD + z_r / 2) - cosmo.comoving_distance(z_OD - z_r / 2)

    # Calculate overdensity volume based on selected geometry
    if geometry == 'ellipsoid':
        V_OD = 4 / 3 * np.pi * d1 * d2 * dz / (1 + z_OD) ** 3
    elif geometry == 'cylinder':
        base = np.pi * d1 * d2
        V_OD = base * dz / (1 + z_OD) ** 3
    elif geometry == 'rectangle':
        V_OD = dz * d1 * d2 * 8 / (1 + z_OD) ** 3
    else:
        raise ValueError(f"Unknown overdensity geometry: {geometry}. Possible geometries are: 'ellipsoid', 'cylinder', or 'rectangle'.")

    # Calculate survey volume, this is well known, z_av may not be exactly the right way to do it though
    z_av = (z_min_survey + z_max_survey) / 2 
    d1 = dDEC_survey * np.pi / 180 * cosmo.comoving_distance(z_av)
    d2 = dRA_survey * np.pi / 180 * cosmo.comoving_transverse_distance(z_av)
    dz = cosmo.comoving_distance(z_max_survey) - cosmo.comoving_distance(z_min_survey)
    V_survey = d1 * d2 * dz / (1 + z_min_survey) ** 3

    # Estimate number of independent overdensity regions in the survey
    N = V_survey / V_OD
    if N < 30:
        print("ATTENTION: The volume you are investigating is not significantly smaller than the survey volume. "
              "This code is not intended for this regime, and WILL give results that are not fully correct.")

    sig = norm.isf(1 / N)
    if add_sig:
        N = 1 / (1 - norm.cdf(sig + add_sig))
        sig = norm.isf(1 / N)

    print(f'The physical volume of the overdensity is {V_OD.value:.2f} pMpc³, in comoving units it is {V_OD.value * (1 + z_OD) ** 3:.2f} Mpc³, '
          f'and the total physical survey volume is {V_survey.value:.2f} pMpc³, in comoving units it is {V_survey.value * (1 + z_min_survey) ** 3:.2f} Mpc³.')
    print(f'The total number of possible subvolumes is: {N:.2f}, corresponding to a typical outlier degree of {sig:.2f} sigma in gaussian units.')

    # Sampling grid for percentiles
    if n_samp:
        x = np.linspace(1 / (10 * n_samp), 1 - 1 / (10 * n_samp), n_samp)
    else:
        x = np.linspace(100 / N, 1, int(100 / N))

    # Extreme value distribution from uniform order statistics, which is the appropriate distribution for percentiles
    evs = N * uniform.pdf(x) * (uniform.cdf(x) ** (N - 1))

    # Return comoving volumes for downstream calculations with the HMF
    return evs, x, V_OD * (1 + z_OD) ** 3, V_survey * (1 + z_min_survey) ** 3

def evs_clustering(cv_df, x, mf=hmf.MassFunction(hmf_model="Behroozi"), V=1, z=4):
    """
    Calculate the extreme value statistics (EVS) for the stellar mass function (SMF) 
    incorporating clustering effects. This code is specifically designed for the case
    where one can reasonably assume that the galaxy we are interested in analyzing is
    sitting 

    This function computes the distribution of the most extreme stellar masses 
    expected in a survey volume at a given redshift, accounting for cosmic variance 
    derived from a provided cosmic variance table.

    Parameters
    ----------
    cv_df : pandas.DataFrame
        A cosmic variance table containing redshift-dependent CV values for 
        different stellar mass bins.
    x : array-like
        Sampling grid over cumulative probabilities (percentiles), typically from 
        sigma_delta_evs.
    mf : hmf.MassFunction object, optional
        A halo mass function object. Default is Behroozi (2013) HMF model since it 
        reproduces N-body sims a little better and also produces slightly more massive halos.
        The choice was also motivated to be able to compare directly to Carnall et al. (2024).
    V : float
        Volume of the overdensity region (in comoving Mpc³).
    z : float
        Redshift at which the calculation is performed.

    Returns
    -------
    phi_maxs : array-like
        The probability distribution of the most extreme stellar masses (phi_max).
    smfs : array-like
        Realizations of the stellar mass function (SMF) including cosmic variance.
    mbins : array-like
        Stellar mass bin edges in log10(M_star).
    N_trapz : array-like
        Number counts in each stellar mass bin derived via trapezoidal integration.
    f : float
        Bin correction factor used to rescale the SMF due to discrete binning.

    Notes
    -----
    - The function updates the mass function to the target redshift and uses a 
      redshift-dependent stellar baryon fraction (SBF).
    - Stellar masses are calculated by converting halo masses using the SBF and 
      the universal baryon fraction.
    - The function computes the extreme value distribution for each stellar mass 
      bin based on the provided cosmic variance table.
    - Care is taken to ensure the cosmic variance increases monotonically with 
      stellar mass to prevent nonphysical results.
    - The result `phi_maxs` represents the distribution of the most extreme 
      stellar masses across the survey volume.

    Example
    -------
    >>> phi_maxs, smfs, mbins, N_trapz, f = evs_clustering(
    ...     cv_df, x, V=1e5, z=4)

    """
    mf.update(z=z, Mmin=mmin, Mmax=mmax, dlog10m=dlog10m)
    mf.cosmo_model = Planck18  # Ensure cosmology is consistent

    dndm = mf.dndlog10m * V  # Absolute dN/dlog10m
    mass = mf.m              # Halo masses
    sbf = 0.051 + 0.024 * (z - 4)  # Stellar baryon fraction at redshift z

    stellar_mass = mass * sbf * baryon_frac  # Stellar masses

    # Integrate dN/dlog10m over moving bins
    N_trapz = []
    for i in range(len(mass) - Ndm):
        inte = np.trapz(dndm[i:i + Ndm], np.log10(mass[i:i + Ndm]))
        N_trapz.append(inte)
    N_trapz = np.array(N_trapz)

    # Bin correction factor to normalize counts, this makes sure that the
    # total number of galaxies stays the same regardless of bin size 
    n_tot = integrate.trapz(dndm, np.log10(mass))
    f = n_tot / np.sum(N_trapz)

    # Align stellar mass array to integration range. This also adjusts for the effects of binning
    stellar_mass = stellar_mass[int(Ndm) // 2: -int(Ndm) // 2]

    smfs = []
    cv_df_z = cv_df.iloc[np.argmin(np.abs(cv_df['z'] - z))] # get the redshift for the 
    cols = [str(m) for m in mbins[:-1]] # 

    # Ensure cosmic variance increases monotonically with mass, just in case something went wrong with the bias calibration
    cv_df_z[cols] = np.maximum.accumulate(cv_df_z[cols])

    for m in mbins[:-1]:
        arg = np.argmin(np.abs(np.log10(stellar_mass) - m)) # find closest stellar mass in the integral table
        N = N_trapz[arg - 1] # get the number count for the mass bin
        cv = cv_df_z[str(m)] # get the cosmic variance for that bin
        smfs.append(trial_OD(x, float(cv), N))

    smfs = np.vstack(smfs) * f # Apply bin correction, otherwise you artificially boost the number counts
    Ns = np.sum(smfs, axis=0) # the total integrated number count for each stellar mass function, i.e., the total number of galaxies in a subvolume across background density percentiles
    fs = smfs / Ns # making the SMFs into PDFs for probability calculation
    fs = fs.T
    Fs = np.cumsum(smfs, axis=0) # create the unnormalized CDF from the SMF
    Fs = Fs.T / Ns.reshape(-1, 1) # normalize the CDF
    phi_maxs = Ns.reshape(-1, 1) * fs * np.power(Fs, Ns.reshape(-1, 1) - 1) # do the EVS calculation

    return phi_maxs.T, smfs, mbins, N_trapz, f

# Optional: Add parent directory to sys.path (might be needed if relative imports fail)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Try importing the scale_cv function from two possible module paths, 
try:
    from make_scaled_cv import scale_cv
except:
    from evstats.make_scaled_cv import scale_cv

def run_evs_clustering(survey_params, od_params, run_params):
    """
    Run the extreme value statistics (EVS) clustering pipeline for a survey and an overdensity.

    Parameters
    ----------
    survey_params : dict
        Dictionary with survey parameters:
            - 'z_min' : float
                Minimum redshift of the survey.
            - 'z_max' : float
                Maximum redshift of the survey.
            - 'dRA_survey' : float
                Survey width in Right Ascension (degrees).
            - 'dDEC_survey' : float
                Survey height in Declination (degrees).

    od_params : dict
        Dictionary with overdensity parameters:
            - 'z_OD' : float
                Central redshift of the overdensity.
            - 'z_r' : float
                Redshift depth (radius) of the overdensity.
            - 'dDEC_OD' : float
                Radius of the overdensity in Declination (degrees).
            - 'dRA_OD' : float
                Radius of the overdensity in Right Ascension (degrees).
            - 'Ntrials' : int, optional
                Number of subdivisions of the ]0;1[ interval of percentiles to perform the calculation at. This should be significantly higher than V_survey/V_OD.
            - 'geometry' : str, optional
                Geometry of the overdensity region ('ellipsoid', 'cylinder', 'rectangle').

    run_params : dict
        Dictionary with runtime parameters:
            - 'name' : str
                Name for the output file.
            - 'recalculate_cv' : bool
                Whether to recalculate the cosmic variance grid (True) or load precomputed values (False). 
            - 'n_z' : int
                Number of redshift slices at which to evaluate cosmic variance and to do the calculation of the EVS distribution.
            - 'add_sig' : float
                Additional offset in Gaussian units to IF one is willing to assume that the field is either over or underdense
                0 is default and is the same as assuming that the field is average, which is **not** exactly the correct assumption (see Jespersen et al. 2025a).

    Returns
    -------
    results : dict
        Dictionary containing all calculated results:
            - 'mbins' : array
                Stellar mass bin centers (log10 scale).
            - 'zs' : array
                Array of redshift slices used in the calculation.
            - 'smfs_z' : array
                Stellar mass functions at each redshift.
            - 'phi_max_convolved_z' : array
                Density-probability-convolved maximum probability distributions for each redshift as a function of mass.
            - 'phi_maxs_z' : array
                Maximum PDFs (unconvolved) as a function of mass for each redshift, and overdensity percentile.
            - 'evs_OD' : array
                Extreme value sampling grid (percentile space).
            - 'f' : array
                Bin correction factors. Only returned for debugging purposes
            - 'N_trapz_z' : array
                Number counts in each stellar mass bin so that we can visualize them later

    Notes
    -----
    - This function is a direct adaptation of the original script for interactive use.
    - The cosmic variance can either be recalculated or loaded from disk.
    - Results are automatically saved to an HDF5 file in the 'data/' folder.

    """

    import numpy as np
    from scipy import integrate
    from scipy.stats import norm
    import hmf, h5py, pandas as pd, tqdm, os, time, warnings
    from astropy.cosmology import Planck18
    from make_scaled_cv import scale_cv

    pd.options.mode.chained_assignment = None
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Unpack survey parameters
    z_min_survey = float(survey_params['z_min']) # minimum redshift of survey where galaxy was found
    z_max_survey = float(survey_params['z_max']) # maximum redshift of survey where galaxy was found
    dRA_survey = survey_params['dRA_survey'] # RA width of survey where galaxy was found
    dDEC_survey = survey_params['dDEC_survey'] # DEC width of survey where galaxy was found

    # Unpack overdensity parameters
    z_OD = od_params['z_OD'] # redshift center of the overdenity
    z_r = od_params['z_r'] # the overdensity radius in redshift space
    dDEC_OD = od_params['dDEC_OD'] # the overdensity radius in declination
    dRA_OD = od_params['dRA_OD'] # the overdensity radius in right ascension
    Ntrials = od_params.get('Ntrials', None) # Number of points to sample the probability grid over on the ]0;1[ interval. If unspecified, it will be estimated later.
    geometry = od_params.get('geometry', 'ellipsoid') # geometry of the overdensity.

    # Unpack runtime parameters
    add_sig = run_params.get('add_sig', 0) # should we assume that the field is additionally under/overdense
    recalculate_cv = run_params.get('recalculate_cv', True) # should the cosmic variance be recalculated
    n_z = run_params.get('n_z', 20) # how many points in redshift space to do the calculation at
    od_name = run_params.get('name', 'evs_clustering') # name of the overdensity

    out_str = f'{od_name}_OD_{z_OD}_addsig_{add_sig}'
    cosmo = Planck18

    if not recalculate_cv:
        df_loc_precomputed = f'dfs/scaled_{z_OD}_{z_r}.csv'

    # Run this block only if the script is executed directly
    if recalculate_cv:
        print('Calculating the cosmic variance')
        # If we want to recalculate the cosmic variance grid from scratch
        start = time.time()

        # Calculate the cosmic variance grid based on the survey and overdensity parameters
        cv_df = scale_cv(dDEC_OD, dRA_OD, z_OD, z_r, low_z = z_min_survey/1.05, max_z=18, n_z=n_z)

        stop = time.time()
        print(f'Finished calculating cosmic variance in {(stop - start) / 60:.2f} minutes')

        # Load the newly computed cosmic variance file (this assumes it's saved in this location)
        df_loc = f'dfs/scaled_{z_OD}_{z_r}.csv'
        cv_df = pd.read_csv(df_loc)

    else:
        # If cosmic variance has already been precomputed, load it
        df_loc = df_loc_precomputed
        cv_df = pd.read_csv(df_loc)

        # Subsample the dataframe to have roughly n_z redshift slices
        cv_df = cv_df.iloc[np.arange(len(cv_df))[::len(cv_df) // n_z]]

    # Extract the redshifts from the cosmic variance table
    zs = np.array(cv_df['z'])

    # Calculate the expected extreme value statistics for the overdensity region
    evs_OD, x_OD, V_OD, V_survey = sigma_delta_evs(
        z_r, z_OD, dDEC_OD, dRA_OD, z_min_survey, z_max_survey, dRA_survey, dDEC_survey,
        geometry=geometry, n_samp=Ntrials, add_sig=add_sig)

    # Initialize lists to store results
    phi_max_convolved_z = []
    phi_maxs_z = []
    smfs_z = []
    mbins_z = []
    N_trapz_z = []
    f_z = []

    # Apply a mask to remove low-probability extreme value samples (speeds up the calculation drastically)
    mask0 = evs_OD > 1e-4
    evs_OD = evs_OD[mask0]
    x_OD = x_OD[mask0]

    print('Calculating at the following redshifts')
    print( np.round(zs, 3) )

    # Loop over all redshifts to calculate extreme value distributions
    for z in tqdm.tqdm(zs, total=len(zs)):
        # Calculate the extreme value statistics for the current redshift
        pm, smfs, mbins, N_trapz, f = evs_clustering(cv_df, x_OD, V = V_OD.value, z = z)
        # Mask out bins where the output is NaN (numerical stability)
        mask = ~np.any(np.isnan(pm), axis=0)
        # Select the valid extreme value probabilities
        evs_ODm = evs_OD[mask]
        # Calculate the convolved maximum PDF at this redshift
        pdf_norm = np.sum(pm[:, mask] * evs_ODm, axis=1) / np.sum(evs_ODm)
        # Store results for this redshift
        phi_max_convolved_z.append(pdf_norm)
        phi_maxs_z.append(pm)
        smfs_z.append(smfs)
        mbins_z.append(mbins)
        N_trapz_z.append(N_trapz)
        f_z.append(f)

    # Make a 'data' directory if it doesn't exist
    if not os.path.exists('data'):
        os.mkdir('data')

    # Save all results to an HDF5 file for later analysis
    with h5py.File('data/' + out_str + '.h5', 'w') as hf:
        hf.create_dataset('log10m', data=mbins)  # Mass bins
        hf.create_dataset('z', data=zs)  # Redshift array
        hf.create_dataset('smf', data=np.array(smfs_z, dtype=np.float32))  # Stellar mass functions
        hf.create_dataset('phi_max_conv', data=phi_max_convolved_z)  # Convolved maximum PDFs
        hf.create_dataset('phi_maxs', data=np.array(phi_maxs_z, dtype=np.float32))  # Maximum PDFs
        hf.create_dataset('evs_OD', data=evs_OD)  # Extreme value sampling grid
        hf.create_dataset('f', data=[f])  # Bin correction factor
        hf.create_dataset('N_trapz', data=N_trapz_z)  # Number counts in each mass bin

    print(f'Results saved to data/{out_str}.h5')

    return {
        'mbins': mbins,
        'zs': zs,
        'smfs_z': smfs_z,
        'phi_max_convolved_z': phi_max_convolved_z,
        'phi_maxs_z': phi_maxs_z,
        'evs_OD': evs_OD,
        'f': f_z,
        'N_trapz_z': N_trapz_z
    }


# Run this block only if the script is executed directly
if __name__ == '__main__':
    # General parameters for the run
    run_params = {
        'add_sig': 0,                     # No additional overdensity assumed
        'name': 'test_new',              # Output name suffix
        'recalculate_cv': True,          # Recompute cosmic variance
        'n_z': 20                        # Number of redshift slices for cosmic variance
    }

    # Overdensity parameters
    od_params = {
        'z_r': 0.01,                     # Redshift depth of overdensity
        'z_OD': 3.21,                    # Central redshift of overdensity
        'dDEC_OD': 100 / 3600,           # Declination size in degrees (converted from arcsec)
        'dRA_OD': 100 / 3600,            # RA size in degrees (converted from arcsec)
        'Ntrials': int(1e5),             # Number of percentiles to consider
        'geometry': 'ellipsoid'          # Geometry of overdensity region
    }

    # Survey geometry parameters
    survey_params = {
        'z_min': 3.0,                    # Minimum redshift of the survey
        'z_max': 5.0,                    # Maximum redshift of the survey
        'dRA_survey': 1402 / 3600,       # Survey RA width in degrees
        'dDEC_survey': 844.6 / 3600      # Survey DEC height in degrees
    }

    output_dict =  run_evs_clustering(survey_params, od_params, run_params)
#     if recalculate_cv:
#         # If we want to recalculate the cosmic variance grid from scratch
#         start = time.time()

#         # Calculate the cosmic variance grid based on the survey and overdensity parameters
#         cv_df = scale_cv(dDEC_OD, dRA_OD, z_OD, z_r, low_z=z_min_survey, max_z=max_z, n_z=n_z)

#         stop = time.time()
#         print(f'Finished calculating cosmic variance in {(stop-start)/60:.2f} minutes')

#         # Load the newly computed cosmic variance file (this assumes it's saved in this location)
#         df_loc = f'dfs/scaled_{z_OD}_{z_r}.csv'
#         cv_df = pd.read_csv(df_loc)

#     else:
#         # If cosmic variance has already been precomputed, load it
#         df_loc = df_loc_precomputed
#         cv_df = pd.read_csv(df_loc)

#         # Subsample the dataframe to have roughly n_z redshift slices
#         cv_df = cv_df.iloc[np.arange(len(cv_df))[::len(cv_df)//n_z]]

#     # Extract the redshifts from the cosmic variance table
#     zs = np.array(cv_df['z'])

#     # Calculate the expected extreme value statistics for the overdensity region
#     evs_OD, x_OD, V_OD, V_survey = sigma_delta_evs(
#         z_r, z_OD, dDEC_OD, dRA_OD, z_min_survey, z_max_survey, dRA_survey, dDEC_survey, 
#         n_samp=Ntrials, add_sig=add_sig)

#     # Initialize lists to store results
#     phi_max_convolved_z = []
#     phi_maxs_z = []
#     smfs_z = []
#     mbins_z = []
#     N_trapz_z = []
#     f_z = []

#     # Apply a mask to remove low-probability extreme value samples (speeds up the calculation)
#     mask0 = evs_OD > 1e-4
#     evs_OD = evs_OD[mask0]
#     x_OD = x_OD[mask0]

#     print('Calculating at the following redshifts')
#     print(zs)

#     # Loop over all redshifts to calculate extreme value distributions
#     for z in tqdm.tqdm(zs, total=len(zs)):
#         # Calculate the extreme value statistics for the current redshift
#         pm, smfs, mbins, N_trapz, f = evs_clustering(cv_df, x_OD, V=V_OD, z=z)

#         # Mask out bins where the output is NaN (numerical stability)
#         mask = ~np.any(np.isnan(pm), axis=0)
#         # Optional: Uncomment this to bypass the NaN mask entirely
#         # mask = np.ones_like(mask).astype(bool)

#         # Select the valid extreme value probabilities
#         evs_ODm = evs_OD[mask]

#         # Calculate the convolved maximum PDF at this redshift
#         pdf_norm = np.sum(pm[:, mask] * evs_ODm, axis=1) / np.sum(evs_ODm)

#         # Store results for this redshift
#         phi_max_convolved_z.append(pdf_norm)
#         phi_maxs_z.append(pm)
#         smfs_z.append(smfs)
#         mbins_z.append(mbins)
#         N_trapz_z.append(N_trapz)
#         f_z.append(f)

#     # Make a 'data' directory if it doesn't exist
#     if not os.path.exists('data'):
#         os.mkdir('data')

#     # Save all results to an HDF5 file for later analysis
#     with h5py.File('data/' + out_str + '.h5', 'w') as hf:
#         hf.create_dataset('log10m', data=mbins)  # Mass bins
#         hf.create_dataset('z', data=zs)  # Redshift array
#         hf.create_dataset('smf', data=np.array(smfs_z, dtype=np.float32))  # Stellar mass functions
#         hf.create_dataset('phi_max_conv', data=phi_max_convolved_z)  # Convolved maximum PDFs
#         hf.create_dataset('phi_maxs', data=np.array(phi_maxs_z, dtype=np.float32))  # Maximum PDFs
#         hf.create_dataset('evs_OD', data=evs_OD)  # Extreme value sampling grid
#         hf.create_dataset('f', data=[f])  # Bin correction factor
#         hf.create_dataset('N_trapz', data=N_trapz_z)  # Number counts in each mass bin
