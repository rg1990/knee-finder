import numpy as np
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import curve_fit
from scipy.signal import medfilt
import warnings
import matplotlib.pyplot as plt
import pickle


class KneeFinder():
    '''
    Write docstring
    '''
    
   
    def __init__(self, x, y, automatic_truncation=False, mode='knee', scale_y=False):
        self.x = x
        self.y = y
        self.y_orig = y
        self.scale_y = scale_y  # not currently used
                
        self.automatic_truncation = automatic_truncation # whether or not to truncate according to sigmoid fit
        self.mode = mode.lower() # valid options are "knee" or "elbow"
        if self.mode not in ['knee', 'elbow']:
            self.mode = 'knee'
        
        # Initialise end-of-life multipliers
        self.knee_eol_factor = 0.8
        self.elbow_eol_factor = 2.0
        
        # Initialise attributes to store results
        self.point = None
        self.onset = None
        self.point_y = None
        self.onset_y = None
        self.eol_reached = False
        self.eol_cycle = None
        
        # Suppress warnings that may occur during curve fitting
        self.ignore_fitting_warnings()
        
        # Load initial parameters and bounds for fitting operations
        with open("./kf_fitting_params.pkl", "rb") as a_file:
            self.fitting_params = pickle.load(a_file)
        del a_file
        
        ### Do the things that are only required once and will not change
        # Get a continuous array of integer cycle (x) values
        self.generate_x_int_array()
        
        # Generate the monotonic fit here, since this is required for everything
        # and we can avoid repeated calls within onset and point detection code
        self.y_mon = self.fit_monotonic()
        
        # Initialise the indices array to include all values initially.
        # Used for slicing to create x_trunc and y_trunc arrays
        self.indices = np.arange(len(self.x_cont))
        
        # Create the instance variable for the sigmoid fit. Set it to None initially
        self.sig_fit = None
        
        # Generate results for the first time
        # Compute the truncated arrays, get line_exp fit, onset, point and EOL
        self.update_results()
        
        # Check for y_scaling. 
        # if self.scale_y:
        #     self.set_y_scaling(True)
        

    # Methods  
    def ignore_fitting_warnings(self):
        # Ignore warnings that may occur during curve fitting.
        warnings.filterwarnings(
            action='ignore',
            message='overflow encountered',
            module=r'.*knee_finder')
        
    
    # Set up the models
    def _asym_sigmoidal(self, x, a, b, c, d, m):
        ''' Formula for asymmetric sigmoidal function '''
        return d + ((a - d) / (1 + (x / c) ** b) ** m)


    def _bw_func(self, x, b0, b1, b2, cp):
        ''' Formula for the single Bacon-Watts model'''
        return b0 + b1*(x-cp) + b2*(x-cp)*np.tanh((x-cp)/1e-8)


    def _bw2_func(self, x, b0, b1, b2, b3, cp, co):
        ''' Formula for the double Bacon-Watts model'''
        return b0 + b1*(x-cp) + b2*(x-cp)*np.tanh((x-cp)/1e-8) + b3*(x-co)*np.tanh((x-co)/1e-8)


    def _exponential(self, x, a, b, c, d, theta):
        ''' Formula for the line plus exponential model'''
        return d*np.exp(a*x - b) + c + theta*x


    # def set_y_scaling(self, new_setting):
    #     '''
    #     Toggle y scaling.
        
    #     TODO - This currently doesn't give onset_y and point_y
    #     values on the original scale of the input y data.
    #     '''          
    #     if new_setting == True:
    #         self.y = (self.y - np.min(self.y)) / (np.max(self.y) - np.min(self.y))
    #     else:
    #         self.y = self.y_orig
        
    #     self.y_is_scaled = new_setting
        
    #     # Get the results using the newly scaled y curve
    #     self.y_mon = self.fit_monotonic()
    #     self.update_results()


    def generate_x_int_array(self):
        '''
        Identify the first and last cycle numbers in the self.x array.
        In the case of non-integer cycle values, round up to the next
        integer for the first cycle and truncate the last cycle number.
        This is done in order to avoid issues with NaNs in the monotonic fit.
        
        Creates a numpy array of integers, self.x_cont, with step 1.
        '''
        
        first_cycle = int(np.ceil(self.x[0]))
        last_cycle = int(self.x[-1])
        
        # Create an array of integers from first to last cycle numbers.
        self.x_cont = np.arange(first_cycle, last_cycle+1)


    def update_truncated_arrays(self):
        '''
        Slice self.x_cont and self.y_mon to include data from the indices
        specified by a truncation operation (manual or automatic).       
        '''
        
        self.x_trunc = self.x_cont[self.indices]
        self.y_trunc = self.y_mon[self.indices]


    def update_manual_truncation(self, low_x_val=None, high_x_val=None):
        '''
        Manually apply truncation to exclude lower and upper sections of the 
        monotonic data which is used to obtain the line_exponential fit.
        
        Arguments should be specified in units of x, rather than index values.
        
        The indices closest to low_x_val and high_x_val are used as lower and
        upper limit indices for truncation.
        
        '''
        
        # Initialise low and high index values to use to create a numpy range
        low_idx = 0
        high_idx = len(self.x_cont)
        
        # Validate inputs
        if low_x_val is not None and low_x_val >= np.min(self.x_cont):
            low_idx = np.argmin(np.abs(self.x_cont - low_x_val))
        
        if high_x_val is not None and high_x_val >= np.min(self.x_cont):
            high_idx = np.argmin(np.abs(self.x_cont - high_x_val))
            
        # Define the updated range of indices that defines the values to be used
        self.indices = np.arange(low_idx, high_idx)
        
        # Generate the new x_trunc and y_trunc arrays based on the indices array
        self.update_truncated_arrays()
        self.update_results()


    def update_results(self):
       
        try: 
            # Check for automatic truncation
            if self.automatic_truncation == True:
                self.auto_truncate()
     
            # Get truncated arrays based on selected truncation method
            self.update_truncated_arrays()
            
            # if self.apply_filter:
            #     self.y_trunc = medfilt(self.y_trunc, 5)
            
            # Get the line-exponential curve by fitting to x_trunc and y_trunc
            self.exp_fit = self.fit_line_exp(self.x_trunc, self.y_trunc)
            
            # Get updated onset, point and EOL values
            self.find_onset()
            self.find_point()
            self.find_eol()
        
        except RuntimeError as err:
            error_msg = str(err)
            print(error_msg)
            # Check for the specific error message about fitting
            if 'Optimal parameters not found' in error_msg:
                print("Curve fitting failed.")
                self.onset = None
                self.point = None
            return
            

    def find_onset(self):
        # Get calculated onset and the BW2 fitted curve
        self.onset, self.bw2_fit = self.fit_double_bacon_watts(self.x_trunc, self.exp_fit)
        
        # Find the index of the value in x_cont that is closest to the
        # x value identified as the onset
        onset_idx = np.argmin(np.abs(self.x_cont - int(self.onset)))
        # Get the y value at detected onset and point, using y_mon
        self.onset_y = self.y_mon[onset_idx]
        
        
    def find_point(self):
        # Get the calculated point and the BW fitted curve
        self.point, self.bw_fit = self.fit_bacon_watts(self.x_trunc, self.exp_fit)
        
        # Find the index of the value in x_cont that is closest to the
        # x value identified as the point
        point_idx = np.argmin(np.abs(self.x_cont - int(self.point)))
        # Get the y value at detected onset and point, using y_mon
        self.point_y = self.y_mon[point_idx]
        
    
    def find_eol(self):
        '''
        Determine whether or not end of life (EOL) is reached,
        and if so, at which cycle.
        
        Use the monotonic fit to check for EOL. This is because:
            1. The values are within the range of experimental values. This
                is not the case if you were to extend the line_exp fit past 
                the truncation point. It decreases very rapidly, so you are
                almost sure to find an EOL, even though it doesn't occur in
                the actual experimental data.
            2. Putting (1) aside, if you find an EOL using the truncated or
                un-truncated line_exp fit, there is often a mismatch between
                the cycle number at which line_exp reaches 80% and the cycle
                number at which the monotonic fit reaches 80%. The monotonic
                fit, being essentially linear interpolation between points,
                much more closely fits the experimental data, so the EOL cycle
                number identified using monotonic fit is more reliable.
            3. If you use the truncated line_exp fit, you are potentially
                ignoring a large percentage of the data (past the truncation).
                You could then easily miss EOL occurrences from past the cycle
                at which the line_exp fit is truncated. Note, simply extending
                the line_exp fit past the truncation point is covered in (1).
        '''
        # Begin by assuming EOL is not reached
        self.eol_reached = False
        
        # Use the monotonic fit to check for EOL, because it is guaranteed
        # to be there for the whole curve and it's essentially linear interp.
        # It will be closer to the experimental data than line_exp or sig fits
        
        # Find the initial (experimental) value
        init_val = self.y_trunc[0]
        # Compute the EOL value based on mode, and find indices after EOL
        if self.mode == 'knee':
            self.eol_val = self.knee_eol_factor * init_val
            self.post_eol_indices = np.where(self.y_mon < self.eol_val)[0]
        
        elif self.mode == 'elbow':
            self.eol_val = self.elbow_eol_factor * init_val
            self.post_eol_indices = np.where(self.y_mon > self.eol_val)[0]
    
        # If it indices exist past the EOL index, set eol_reached to true and
        # find the first cycle number in x_cont after the EOL value is reached
        if len(self.post_eol_indices) > 0:
            self.eol_reached = True
            self.eol_idx = self.post_eol_indices[0]
            self.eol_cycle = self.x_cont[self.eol_idx]
    
    
    # Show the results of the analysis on a plot
    def plot_results(self, line_exp=False, mon=False, sig=False, data_style='-'):
        '''
        Plot the results of the KneeFinder analysis. Lines plotted by default
        are:
            - The original experimental data (self.x, self.y)
            - Horizontal lines corresponding to the y values of onset and point
            - Vertical lines corresponding to the x values of onset and point.
            
        Optionally, the arguments can be used to plot additional lines, produced
        by the curve-fitting operations.
        
        '''
        fig, ax = plt.subplots()
        ax.plot(self.x, self.y, data_style, label='Experimental data')
        
        # Add optional fitted lines, if speficied
        if mon:
            ax.plot(self.x_cont, self.y_mon, label='Monotonic fit', color='green')
        if line_exp:
            ax.plot(self.x_cont[self.indices], self.exp_fit, label='Line plus exponential fit', color='fuchsia')
        if sig and self.automatic_truncation and hasattr(self, 'sig_fit'):
            ax.plot(self.x_cont, self.sig_fit, label='Asymmetric sigmoidal fit', color='blue')
        
        ax.axvline(self.onset, linestyle='--', label=f'{self.mode.capitalize()} onset', color='orange')
        ax.axvline(self.point, linestyle='--', label=f'{self.mode.capitalize()} point', color='red')
        ax.axhline(self.onset_y, linestyle='--', color='orange')
        ax.axhline(self.point_y, linestyle='--', color='red')
        
        if self.eol_cycle != None:
            ax.axvline(self.eol_cycle, linestyle='--', label='End of life', color='black')
        
        ax.grid(alpha=0.4)
        ax.legend()
        plt.show()
        
    
    
    # Below are methods to obtain fitted curves using curve_fit
    # (monotonic, line_exp, asym_sigmoid, Bacon-Watts, double Bacon-Watts)
    def fit_monotonic(self):
        
        # Create an IsotonicRegression instance
        if self.mode == 'knee':
            ir = IsotonicRegression(increasing=False)
        elif self.mode == 'elbow':
            ir = IsotonicRegression(increasing=True)

        # Fit the monotonic curve to the experimental data
        ir.fit(self.x, self.y)
        
        # Get a y value for every integer cycle number, based on the fit
        y_monotonic = ir.predict(self.x_cont)
        
        return y_monotonic
    
    
    def fit_sigmoid(self, x_data, y_data):
        '''
        Write docstring
        '''
        
        # Get p0/bounds from self.fitting_params dict
        p0 = self.fitting_params[self.mode]['sig']['p0']
        bounds = self.fitting_params[self.mode]['sig']['bounds']
        
        sig_popt, _ = curve_fit(self._asym_sigmoidal,
                               x_data,
                               y_data,
                               p0=p0,
                               bounds=bounds)
        
        return self._asym_sigmoidal(x_data, *sig_popt)
    
    
    def fit_line_exp(self, x_data, y_data):
        '''
        Write docstring
        '''
                
        # Get p0/bounds from self.fitting_params dict
        p0 = self.fitting_params[self.mode]['line_exp']['p0']
        bounds = self.fitting_params[self.mode]['line_exp']['bounds']
        
        # Get optimal parameters for the _exponential model given the input data
        exp_popt, _ = curve_fit(self._exponential,
                                x_data,
                                y_data,
                                p0=p0,
                                bounds=bounds)
        
        # Apply these parameters to generate a line_exponential curve with the
        # same range of x-values as present in x_data
        return self._exponential(x_data, *exp_popt)

    
    def fit_bacon_watts(self, x_data, y_data):
        '''
        Write docstring.
        
        Pass the line_exponential fit to this method, to use the
        Bacon-Watts method to compute the cycle number at which
        the knee point occurs.
        
        Returns:
            popt_bw[3] (type: float)
                The x value at which the "point" has been identified.
            
            bw_fit (type: numpy array)
                An array which contains the Bacon-Watts fit, applied over
                the range of values present in x_data. This has the same
                shape as x_data.
                
        '''
        
        # Get p0/bounds from self.fitting_params dict
        p0 = self.fitting_params[self.mode]['bw']['p0']
        bounds = self.fitting_params[self.mode]['bw']['bounds']
        
        # Set the initial guess for "point", according to input data
        p0[3] = x_data[0] + ((x_data[-1] - x_data[0]) / 1.5)
        # Set the lower/upper bounds to be the first/last cycle in input data
        bounds[0][3] = x_data[0]
        bounds[1][3] = x_data[-1]    
   
        # Fit the Bacon Watts model to the input data
        popt_bw, _ = curve_fit(self._bw_func,
                               x_data,
                               y_data,
                               p0=p0,
                               bounds=bounds)
        
        # Apply the parameters to input x_data to generate the BW fit curve
        bw_fit = self._bw_func(x_data, *popt_bw)
        
        return popt_bw[3], bw_fit
        

    def fit_double_bacon_watts(self, x_data, y_data):
        '''
        Returns:
            popt_bw2[4] (type: float)
                The x value at which the "onset" has been identified.
            
            bw2_fit (type: numpy array)
                An array which contains the double Bacon-Watts fit, applied
                over the range of values present in x_data. This has the same
                shape as x_data.
        '''
        
        # Get p0/bounds from self.fitting_params dict
        p0 = self.fitting_params[self.mode]['bw2']['p0']
        bounds = self.fitting_params[self.mode]['bw2']['bounds']
        
        # Set the initial guesses for the 2 change points, based on input data
        p0[4] = x_data[0] + ((x_data[-1] - x_data[0]) / 2.0)
        p0[5] = x_data[0] + ((x_data[-1] - x_data[0]) / 1.5)
        
        # Set bounds so onset and point are within x_data's range of values
        bounds[1][4] = x_data[-1]
        bounds[1][5] = x_data[-1]
        bounds[0][4] = x_data[0]
        bounds[0][5] = x_data[0]

        # Fit the double Bacon-Watts model to the input data
        popt_bw2, _ = curve_fit(self._bw2_func,
                               x_data,
                               y_data,
                               p0=p0,
                               bounds=bounds)
        
        # Apply the parameters to input x_data to generate the BW2 fit curve
        bw2_fit = self._bw2_func(x_data, *popt_bw2)
        
        return popt_bw2[4], bw2_fit
    
    
    def compute_second_derivative(self, data):
        '''
        Write docstring
        '''
        
        # Using np.gradient gives us a result of the same shape
        dy_dx = np.gradient(data)
        d2y_dx2 = np.gradient(dy_dx)
        
        # Set a threshold, below which, the value is set to zero
        d2y_dx2[np.where(np.abs(d2y_dx2) < 1e-10)] = 0.0
        
        # Replace the last 2 values with the value at index [-3] to avoid
        # the sharp change that happens at the end of the d2y_dx2 array
        d2y_dx2[-2:] = d2y_dx2[-3]
        
        return d2y_dx2
       
    
    def get_auto_truncated_indices(self, data):
        '''
        The shape of the asymmetric sigmoid fit curve ensures that the second
        derivative should have at most one zero crossing.
        
        '''
        
        # Compute the second derivative of data
        self.d2 = self.compute_second_derivative(data)
        # Apply median filter to smooth out fluctuations
        self.d2 = medfilt(self.d2, 5)
        # Get an array of the sign of d2 to find changes
        self.d2_sign = np.sign(self.d2)
        
        # Use mode to determine which new sign value to look for to find changes
        if self.mode == 'knee':
            new_sign_val = 1
        elif self.mode == 'elbow':
            new_sign_val = -1
            
        # Find the indices where self.d2_sign has a value of new_sign_val
        new_sign_indices = np.where(self.d2_sign==new_sign_val)[0]
        
        if len(new_sign_indices)==0:
            indices = np.arange(0, len(data))
            return indices
        else:
            # Print and assert for testing
            #print(new_sign_indices[0])
            # Check there are no breaks in the "new sign" index values
            #assert(len(new_sign_indices) == new_sign_indices[-1]-new_sign_indices[0]+1)
            
            indices = np.arange(0, new_sign_indices[0])
            return indices
            
    
    def auto_truncate(self):
        '''
        Generate the asymmetric sigmoid fit and determine where the sign of
        the second derivative changes. The location of this change determines
        the indices which should be used for subsequent fitting
        (line_exp, BW, BW2).
        
        Try to fit the asymmetric sigmoid model to self.x_cont and self.y_mon,
        and generate self.sig_fit, which is the prediction, over self.x_cont.
        
        '''
        
        # Try to fit the sigmoid, but this may fail with a RuntimeError
        # if optimal parameters could not be found.
        try:
            self.sig_fit = self.fit_sigmoid(x_data=self.x_cont,
                                             y_data=self.y_mon)
            
            self.indices = self.get_auto_truncated_indices(data=self.sig_fit)
            
        except RuntimeError as err:
            error_msg = str(err)
            print(error_msg)
            # Check for the specific error message about fitting
            if 'Optimal parameters not found' in error_msg:
                print("Can't fit sigmoid for truncation. Skipping")
            
            # Since we can't fit the asym_sigmoid for truncation,
            # set self.indices as if we are not using auto truncation.
            self.indices = np.arange(len(self.y_mon))
           

    # NEW - Try a new, generic method that can set p0 for any curve fit
    def set_fit_p0(self, model, new_p0, data_mode=None):
        '''
        
        
        '''
                
        # By default, consider the mode of the current instance
        if data_mode is None:
            data_mode = self.mode
        
        if data_mode.lower() not in ['knee', 'elbow']:
            print(f"Error - Invalid value '{data_mode}' for data_mode")
            return
        
        # Get the current bounds, for validation of new parameter values
        current_bounds = self.fitting_params[data_mode][model]['bounds']
        
        # Input validation for new_p0 shape
        new_p0 = np.array(new_p0)
        required_shape = self.fitting_params['shapes'][model]['p0']
        
        if new_p0.shape != required_shape:
            print(f"Error - Invalid shape {new_p0.shape} for param_arr. Required shape is: {required_shape}")
            return
        
        # Check if the new parameters are within the current bounds
        elif np.all((new_p0 > current_bounds[0,:]) & (new_p0 < current_bounds[1,:])) == False:
            print(f"Error - p0 values must be within current bounds.\
                  \n{current_bounds}.")
            return
        
        # Update the p0 array in the fitting_params dictionary
        else:
            self.fitting_params[data_mode][model]['p0'] = new_p0
    
    
    def set_fit_bounds(self, model, new_bounds, data_mode=None):
        
        # Sense check the new_bounds input. All upper bound values must be
        # greater than lower bound values
        if np.any(np.diff(new_bounds, axis=0) < 0):
            print(f"Error - Invalid values in 'new_bounds' array:\n{new_bounds}.")
            print("Upper bounds must be greater than lower bounds.")
            return
        
        
        if data_mode is None:
            # By default, consider the mode of the current instance
            data_mode = self.mode
        
        elif data_mode.lower() not in ['knee', 'elbow']:
            print(f"Error - Invalid value '{data_mode}' for data_mode")
            return
        
        # Get the current p0 values, for validation of new bounds
        current_p0 = self.fitting_params[data_mode][model]['p0']
        
        # Input validation for param_array shape
        new_bounds = np.array(new_bounds)
        required_shape = self.fitting_params['shapes'][model]['bounds']
        
        if new_bounds.shape != required_shape:
            print(f"Error - Invalid shape {new_bounds.shape} for param_arr. Required shape is: {required_shape}")
            return
        
        # Check if the new bounds are compatible with current p0 values.
        elif np.any((new_bounds[0] > current_p0) | (new_bounds[1] < current_p0)):
            print(f"Error - New bounds must contain current p0 values.\
                  \n{current_p0}.")
            return
        
        # Update the bounds array in the fitting_params dictionary
        else:
            self.fitting_params[data_mode][model]['bounds'] = new_bounds
    
        
    # NEW
    def reset_fitting_p0_and_bounds(self, model, data_mode=None):
        
        if data_mode is None:
            # By default, consider the mode of the current instance
            data_mode = self.mode
        
        elif data_mode not in ['knee', 'elbow']:
            print(f"Error - Invalid value for 'data_mode' argument: {data_mode}.")
            return
        
        # Load the dictionary from file to refer to its contents
        with open("./kf_fitting_params.pkl", "rb") as a_file:
            temp_dict = pickle.load(a_file)
        del a_file
        
        # Replace parameters for selected model with values from temp_dict
        if model in temp_dict[data_mode].keys():
            self.fitting_params[data_mode][model] = temp_dict[data_mode][model]
        
        # Reset all fitting parameters and bounds by loading dictionary again
        elif model.lower() == 'all':
            self.fitting_params = temp_dict
            
        else:
            print(f"Error - Invalid value for 'model' argument: {model}.")
            return