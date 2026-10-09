# =============================================================================
# LEGACY / UNUSED in the grid-tied A/B study. The live heat-pump + PCM-TES model is
# physics/hp_tes/ (system.run_python_hp_tes, called from physics/blackbox.py). This
# file's FMU-based path (run_fmu) is not on the A/B code path. Kept for reference.
# =============================================================================
import Input_Parameters
from physics import Data_Conversion
from physics import RuleBasedControl
import pandas as pd
import numpy as np

def load_HP_Parameters(sheetname, file_path='data/HP_TES_Parameters.xlsx'):
    """
    Load the HP TES parameter workbook and return the 'Heating' or 'Cooling' worksheet as a DataFrame.

    Parameters
    ----------
    file_path : str, optional
        Path to the HP TES parameters workbook. Defaults to 'HP_TES_Parameters.xlsx'.

    Returns
    -------
    pd.DataFrame
        Data from the 'Heating' or 'Cooling' sheet containing COP and capacity mappings.
    """
    try:
        cooling_df = pd.read_excel(file_path, sheet_name=sheetname)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find HP TES parameters file at '{file_path}'.")
    except ValueError as exc:
        raise ValueError(f"Unable to read sheet from '{file_path}'.") from exc

    return cooling_df


def _fit_heating_polynomial_internal(heating_df, option, degree=2):
    """
    Internal helper function that fits the polynomial model and returns both the function and coefficients.
    
    Parameters
    ----------
    heating_df : pd.DataFrame
        DataFrame containing columns 'Ta', 'T_HP', and either 'COP' or 'Capacity'.
    option : str
        Either 'COP' or 'Capacity' to specify which column to fit.
    degree : int, optional
        Degree of the polynomial (default=2).
    
    Returns
    -------
    function
        A function that takes (ta, t_hp) as arguments and returns the predicted value.
    np.ndarray
        Array of fitted coefficients.
    int
        Degree of the polynomial.
    """
    # Validate option
    if option not in ['COP', 'Capacity']:
        raise ValueError(f"Option must be 'COP' or 'Capacity', got '{option}'.")
    
    # Set required columns and extract data
    required_columns = {'Ta', 'T_HP', option}
    missing = required_columns.difference(heating_df.columns)
    if missing:
        missing_str = "', '".join(sorted(missing))
        raise ValueError(f"Heating DataFrame is missing required columns: '{missing_str}'.")
    
    # Extract data
    ta = heating_df['Ta'].to_numpy(dtype=float)
    t_hp = heating_df['T_HP'].to_numpy(dtype=float)
    target = heating_df[option].to_numpy(dtype=float)

    # Create polynomial features matrix
    # Order: [1, ta, t_hp, ta*t_hp, ta^2, t_hp^2, ...]
    # For degree 2: [1, ta, t_hp, ta*t_hp, ta^2, t_hp^2]
    # For degree 1: [1, ta, t_hp]
    features_list = []
    
    # Constant term
    features_list.append(np.ones(len(ta)))
    
    # Linear terms (degree 1)
    if degree >= 1:
        features_list.append(ta)
        features_list.append(t_hp)
    
    # Higher order terms (degree 2 and above)
    # Order: interaction terms first, then pure terms
    for d in range(2, degree + 1):
        # Interaction terms (mixed powers)
        for i in range(1, d):  # Skip pure ta^d and t_hp^d, handle separately
            j = d - i
            if i > 0 and j > 0:
                feature = (ta ** i) * (t_hp ** j)
                features_list.append(feature)
        
        # Add pure ta^d term
        features_list.append(ta ** d)
        
        # Add pure t_hp^d term
        features_list.append(t_hp ** d)
    
    # Create design matrix
    X = np.column_stack(features_list)
    
    # Fit using least squares
    coefficients, residuals, rank, s = np.linalg.lstsq(X, target, rcond=None)
    
    # Create the prediction function
    def prediction_function(ta, t_hp):
        """
        Calculate predicted value based on ambient temperature and heat pump temperature.
        
        Parameters
        ----------
        ta : float or array-like
            Ambient temperature [°C]
        t_hp : float or array-like
            Heat pump water temperature [°C]
        
        Returns
        -------
        float or np.ndarray
            Predicted value(s)
        """
        ta = np.asarray(ta)
        t_hp = np.asarray(t_hp)
        
        # Create features for prediction (same order as training)
        pred_features = [np.ones_like(ta)]
        
        # Linear terms
        if degree >= 1:
            pred_features.append(ta)
            pred_features.append(t_hp)
        
        # Higher order terms (degree 2 and above)
        for d in range(2, degree + 1):
            for i in range(1, d):
                j = d - i
                if i > 0 and j > 0:
                    feature = (ta ** i) * (t_hp ** j)
                    pred_features.append(feature)
            
            # Add pure ta^d term
            pred_features.append(ta ** d)
            
            # Add pure t_hp^d term
            pred_features.append(t_hp ** d)
        
        X_pred = np.column_stack(pred_features)
        prediction = X_pred @ coefficients
        
        # Return scalar if inputs were scalars
        if np.isscalar(ta) and np.isscalar(t_hp):
            return float(prediction)
        return prediction
    
    return prediction_function, coefficients, degree


def fit_cop_heating_polynomial(heating_df, degree=2):
    """
    Fit a polynomial regression model to heating COP data and return a function to calculate COP.
    
    The polynomial form is: cop = a0 + a1*t_a + a2*t_hp + a3*t_a*t_hp + a4*t_a^2 + a5*t_hp^2 + ...
    
    Parameters
    ----------
    heating_df : pd.DataFrame
        DataFrame containing columns 'Ta', 'T_HP', and 'COP'.
    degree : int, optional
        Degree of the polynomial (default=2). For degree=2, includes terms up to t_a^2, t_hp^2, and t_a*t_hp.
    
    Returns
    -------
    function
        A function that takes (ta, t_hp) as arguments and returns the predicted COP.
    
    Raises
    ------
    ValueError
        If required columns are missing from heating_df.
    
    Examples
    --------
    >>> heating_df = load_hp_tes_parameters()
    >>> cop_func = fit_cop_heating_polynomial(heating_df, degree=2)
    >>> cop_value = cop_func(ta=20, t_hp=45)
    """
    cop_function, _, _ = _fit_heating_polynomial_internal(heating_df, 'COP', degree)
    return cop_function


def get_cop_heating_coefficients(heating_df, degree=2):
    """
    Fit a polynomial regression model to heating COP data and return the fitted coefficients.
    
    The polynomial form is: cop = a0 + a1*t_a + a2*t_hp + a3*t_a*t_hp + a4*t_a^2 + a5*t_hp^2 + ...
    
    Parameters
    ----------
    heating_df : pd.DataFrame
        DataFrame containing columns 'Ta', 'T_HP', and 'COP'.
    degree : int, optional
        Degree of the polynomial (default=2). For degree=2, includes terms up to t_a^2, t_hp^2, and t_a*t_hp.
    
    Returns
    -------
    dict
        Dictionary containing the fitted coefficients (a0, a1, a2, ...).
    
    Raises
    ------
    ValueError
        If required columns are missing from heating_df.
    
    Examples
    --------
    >>> heating_df = load_hp_tes_parameters()
    >>> coefficients = get_cop_heating_coefficients(heating_df, degree=2)
    >>> print(coefficients['a0'], coefficients['a1'], coefficients['a2'])
    """
    _, coefficients, _ = _fit_heating_polynomial_internal(heating_df, 'COP', degree)
    coeff_dict = {f'a{i}': float(coeff) for i, coeff in enumerate(coefficients)}
    return coeff_dict


def fit_capacity_heating_polynomial(heating_df, degree=2):
    """
    Fit a polynomial regression model to heating capacity data and return a function to calculate capacity.
    
    The polynomial form is: capacity = a0 + a1*t_a + a2*t_hp + a3*t_a*t_hp + a4*t_a^2 + a5*t_hp^2 + ...
    
    Parameters
    ----------
    heating_df : pd.DataFrame
        DataFrame containing columns 'Ta', 'T_HP', and 'Capacity'.
    degree : int, optional
        Degree of the polynomial (default=2). For degree=2, includes terms up to t_a^2, t_hp^2, and t_a*t_hp.
    
    Returns
    -------
    function
        A function that takes (ta, t_hp) as arguments and returns the predicted capacity.
    
    Raises
    ------
    ValueError
        If required columns are missing from heating_df.
    
    Examples
    --------
    >>> heating_df = load_hp_tes_parameters()
    >>> capacity_func = fit_capacity_heating_polynomial(heating_df, degree=2)
    >>> capacity_value = capacity_func(ta=20, t_hp=45)
    """
    capacity_function, _, _ = _fit_heating_polynomial_internal(heating_df, 'Capacity', degree)
    return capacity_function


def get_capacity_heating_coefficients(heating_df, degree=2):
    """
    Fit a polynomial regression model to heating capacity data and return the fitted coefficients.
    
    The polynomial form is: capacity = a0 + a1*t_a + a2*t_hp + a3*t_a*t_hp + a4*t_a^2 + a5*t_hp^2 + ...
    
    Parameters
    ----------
    heating_df : pd.DataFrame
        DataFrame containing columns 'Ta', 'T_HP', and 'Capacity'.
    degree : int, optional
        Degree of the polynomial (default=2). For degree=2, includes terms up to t_a^2, t_hp^2, and t_a*t_hp.
    
    Returns
    -------
    dict
        Dictionary containing the fitted coefficients (a0, a1, a2, ...).
    
    Raises
    ------
    ValueError
        If required columns are missing from heating_df.
    
    Examples
    --------
    >>> heating_df = load_hp_tes_parameters()
    >>> coefficients = get_capacity_heating_coefficients(heating_df, degree=2)
    >>> print(coefficients['a0'], coefficients['a1'], coefficients['a2'])
    """
    _, coefficients, _ = _fit_heating_polynomial_internal(heating_df, 'Capacity', degree)
    coeff_dict = {f'a{i}': float(coeff) for i, coeff in enumerate(coefficients)}
    return coeff_dict


def _fit_cooling_polynomial_internal(cooling_df, option, degree=2, heating_df=None):
    """
    Internal helper function that fits the polynomial model for cooling and returns both the function and coefficients.
    
    Note: Cooling data typically has only one T_HP value (e.g., 7°C). The function:
    1. First fits a polynomial in t_a only from the cooling data
    2. Uses heating polynomial coefficients as a reference to determine t_hp sensitivity
    3. Applies the inverse of heating's t_hp sensitivity to cooling (higher t_hp -> lower COP/capacity)
    4. Ensures the polynomial matches the original data at the given t_hp value
    5. Extrapolates correctly when t_hp deviates from the original value
    
    Parameters
    ----------
    cooling_df : pd.DataFrame
        DataFrame containing columns 'Ta', 'T_HP', and either 'COP' or 'Capacity'.
    option : str
        Either 'COP' or 'Capacity' to specify which column to fit.
    degree : int, optional
        Degree of the polynomial (default=2).
    heating_df : pd.DataFrame, optional
        Heating DataFrame to use as reference for t_hp sensitivity. If None, will try to load it.
    
    Returns
    -------
    function
        A function that takes (ta, t_hp) as arguments and returns the predicted value.
    np.ndarray
        Array of fitted coefficients.
    int
        Degree of the polynomial.
    """
    # Validate option
    if option not in ['COP', 'Capacity']:
        raise ValueError(f"Option must be 'COP' or 'Capacity', got '{option}'.")
    
    # Set required columns and extract data
    required_columns = {'Ta', 'T_HP', option}
    missing = required_columns.difference(cooling_df.columns)
    if missing:
        missing_str = "', '".join(sorted(missing))
        raise ValueError(f"Cooling DataFrame is missing required columns: '{missing_str}'.")
    
    # Extract data
    ta = cooling_df['Ta'].to_numpy(dtype=float)
    t_hp_data = cooling_df['T_HP'].to_numpy(dtype=float)
    target = cooling_df[option].to_numpy(dtype=float)
    
    # Get the unique t_hp value (should be only one, e.g., 7°C)
    unique_t_hp = np.unique(t_hp_data)
    if len(unique_t_hp) != 1:
        raise ValueError(f"Cooling data should have only one T_HP value, found {len(unique_t_hp)} unique values: {unique_t_hp}")
    
    t_hp_ref = float(unique_t_hp[0])  # Reference t_hp value (e.g., 7°C)
    
    # Step 1: Fit a polynomial in t_a only (since t_hp is constant)
    # This gives us: target = b0 + b1*ta + b2*ta^2 + ...
    ta_features_list = []
    ta_features_list.append(np.ones(len(ta)))
    
    for d in range(1, degree + 1):
        ta_features_list.append(ta ** d)
    
    X_ta = np.column_stack(ta_features_list)
    ta_coefficients, _, _, _ = np.linalg.lstsq(X_ta, target, rcond=None)
    
    # Step 2: Get heating coefficients as reference for t_hp sensitivity
    if heating_df is None:
        try:
            heating_df = load_HP_Parameters('Heating')
        except:
            heating_df = None
    
    # Get heating coefficients to understand t_hp sensitivity
    if heating_df is not None:
        try:
            _, heating_coeffs, _ = _fit_heating_polynomial_internal(heating_df, option, degree)
            # heating_coeffs: [a0, a1, a2, a3, a4, a5] for degree 2
            # a2 is the t_hp coefficient (positive for heating: higher t_hp -> higher COP)
            # a3 is the ta*t_hp interaction coefficient
            # a5 is the t_hp^2 coefficient
            
            # Use heating coefficients as reference, but invert for cooling
            # Calculate how heating COP/capacity changes with t_hp to understand sensitivity
            avg_heating = np.mean(heating_df[option].to_numpy(dtype=float))
            avg_cooling = np.mean(target)
            
            # Get heating coefficients
            a2_heating = heating_coeffs[2] if len(heating_coeffs) > 2 else 0.0
            a3_heating = heating_coeffs[3] if len(heating_coeffs) > 3 else 0.0
            a5_heating = heating_coeffs[5] if len(heating_coeffs) > 5 else 0.0
            
            # Calculate heating sensitivity: how much does heating value change per degree C of t_hp?
            # Use a representative ta value (middle of cooling data range)
            ta_representative = np.mean(ta)
            # For heating: d(cop)/d(t_hp) ≈ a2 + a3*ta + 2*a5*t_hp (at typical t_hp)
            # Use a typical t_hp for heating (e.g., 45°C)
            t_hp_heating_typical = 45.0
            heating_sensitivity = a2_heating + a3_heating*ta_representative + 2*a5_heating*t_hp_heating_typical
            
            # For cooling, we want similar magnitude of change but in opposite direction
            # Scale by the ratio of average values to account for different base magnitudes
            scale_factor = avg_cooling / avg_heating if avg_heating != 0 else 1.0
            
            # Cooling sensitivity should be similar magnitude but negative
            # Use the heating sensitivity as a guide, but scale appropriately
            cooling_sensitivity_magnitude = abs(heating_sensitivity * scale_factor)
            
            # For cooling: cop = base_value + a2*(t_hp - t_hp_ref) + ...
            # We want the change to be proportional, so distribute the sensitivity across coefficients
            # a2 contributes directly, a3 contributes through ta interaction, a5 through quadratic term
            
            # Set cooling coefficients to achieve milder sensitivity (less dramatic changes)
            # Use a fraction of the heating sensitivity to make changes more gradual
            sensitivity_reduction_factor = 0.4  # Reduce sensitivity to 40% of heating (milder scaling)
            
            # Distribute: a2 gets most of it, a3 and a5 get smaller portions
            a2_cooling = -cooling_sensitivity_magnitude * sensitivity_reduction_factor  # Main linear term (reduced)
            a3_cooling = -abs(a3_heating * scale_factor * sensitivity_reduction_factor * 0.3) if a3_heating != 0 else -0.002 * scale_factor
            a5_cooling = -abs(a5_heating * scale_factor * sensitivity_reduction_factor) if a5_heating != 0 else -0.0002 * scale_factor
        except:
            # Fallback if heating fit fails - use milder sensitivity
            avg_target = np.mean(target)
            a2_cooling = -abs(avg_target * 0.008)  # Reduced from 0.02 to 0.008 (milder)
            a3_cooling = -0.004  # Reduced from 0.01 to 0.004
            a5_cooling = -0.0004  # Reduced from 0.001 to 0.0004
    else:
        # Fallback if no heating data available - use milder sensitivity
        avg_target = np.mean(target)
        a2_cooling = -abs(avg_target * 0.008)  # Reduced from 0.02 to 0.008 (milder)
        a3_cooling = -0.004  # Reduced from 0.01 to 0.004
        a5_cooling = -0.0004  # Reduced from 0.001 to 0.0004
    
    # Step 3: Construct the full polynomial with t_hp terms
    # Initialize coefficients array
    # For degree 2: [a0, a1, a2, a3, a4, a5]
    # Order: [1, ta, t_hp, ta*t_hp, ta^2, t_hp^2]
    n_coeffs = 1 + 2*degree + (degree-1)  # 1 + 2*degree + interaction terms
    if degree == 2:
        n_coeffs = 6  # [a0, a1, a2, a3, a4, a5]
    elif degree == 1:
        n_coeffs = 3  # [a0, a1, a2]
    
    coefficients = np.zeros(n_coeffs)
    
    # Extract ta-only coefficients
    b0 = ta_coefficients[0]
    b1 = ta_coefficients[1] if len(ta_coefficients) > 1 else 0.0
    b2 = ta_coefficients[2] if len(ta_coefficients) > 2 else 0.0
    
    if degree == 2:
        # Set the ta terms to match the ta-only fit
        # a4 = b2 (ta^2 coefficient)
        coefficients[4] = b2
        
        # Set t_hp coefficients (inverse relationship)
        coefficients[2] = a2_cooling  # a2: t_hp coefficient
        coefficients[3] = a3_cooling  # a3: ta*t_hp interaction
        coefficients[5] = a5_cooling  # a5: t_hp^2 coefficient
        
        # At t_hp = t_hp_ref: a1 + a3*t_hp_ref = b1
        coefficients[1] = b1 - a3_cooling*t_hp_ref
        
        # Set a0 to ensure exact match at t_hp_ref
        # At t_hp = t_hp_ref: a0 + a2*t_hp_ref + a5*t_hp_ref^2 = b0
        coefficients[0] = b0 - a2_cooling*t_hp_ref - a5_cooling*t_hp_ref**2
        
    elif degree == 1:
        # For degree 1: cop = a0 + a1*ta + a2*t_hp
        # At t_hp = t_hp_ref: cop = (a0 + a2*t_hp_ref) + a1*ta = b0 + b1*ta
        coefficients[1] = b1  # a1 = b1
        
        # Use milder sensitivity for degree 1 as well
        if heating_df is not None:
            try:
                _, heating_coeffs, _ = _fit_heating_polynomial_internal(heating_df, option, degree)
                a2_heating = heating_coeffs[2] if len(heating_coeffs) > 2 else 0.0
                avg_heating = np.mean(heating_df[option].to_numpy(dtype=float))
                avg_cooling = np.mean(target)
                scale_factor = avg_cooling / avg_heating if avg_heating != 0 else 1.0
                sensitivity_reduction_factor = 0.4  # Milder scaling
                a2_cooling = -abs(a2_heating * scale_factor * sensitivity_reduction_factor)
            except:
                avg_target = np.mean(target)
                a2_cooling = -abs(avg_target * 0.008)
        else:
            avg_target = np.mean(target)
            a2_cooling = -abs(avg_target * 0.008)
        
        coefficients[2] = a2_cooling  # a2: t_hp coefficient (negative for inverse, milder)
        
        # a0 + a2*t_hp_ref = b0
        coefficients[0] = b0 - a2_cooling*t_hp_ref
    
    # Create the prediction function
    def prediction_function(ta, t_hp):
        """
        Calculate predicted value based on ambient temperature and heat pump temperature.
        
        For cooling, higher t_hp results in lower COP/capacity (inverse relationship).
        Values are clamped to be non-negative to prevent unrealistic predictions.
        
        Parameters
        ----------
        ta : float or array-like
            Ambient temperature [°C]
        t_hp : float or array-like
            Heat pump water temperature [°C]
        
        Returns
        -------
        float or np.ndarray
            Predicted value(s), clamped to be non-negative
        """
        ta = np.asarray(ta)
        t_hp = np.asarray(t_hp)
        
        # Create features for prediction (same order as training)
        pred_features = [np.ones_like(ta)]
        
        # Linear terms
        if degree >= 1:
            pred_features.append(ta)
            pred_features.append(t_hp)
        
        # Higher order terms (degree 2 and above)
        for d in range(2, degree + 1):
            for i in range(1, d):
                j = d - i
                if i > 0 and j > 0:
                    feature = (ta ** i) * (t_hp ** j)
                    pred_features.append(feature)
            
            # Add pure ta^d term
            pred_features.append(ta ** d)
            
            # Add pure t_hp^d term
            pred_features.append(t_hp ** d)
        
        X_pred = np.column_stack(pred_features)
        prediction = X_pred @ coefficients
        
        # Clamp to non-negative values to prevent unrealistic predictions
        prediction = np.maximum(prediction, 0.0)
        
        # Return scalar if inputs were scalars
        if np.isscalar(ta) and np.isscalar(t_hp):
            return float(prediction)
        return prediction
    
    return prediction_function, coefficients, degree


def fit_cop_cooling_polynomial(cooling_df, degree=2):
    """
    Fit a polynomial regression model to cooling COP data and return a function to calculate COP.
    
    The polynomial form is: cop = a0 + a1*t_a + a2*t_hp + a3*t_a*t_hp + a4*t_a^2 + a5*t_hp^2 + ...
    
    Note: For cooling, higher t_hp results in lower COP (inverse relationship compared to heating).
    The cooling data typically has only one T_HP value, but the function can predict for different t_hp values.
    
    Parameters
    ----------
    cooling_df : pd.DataFrame
        DataFrame containing columns 'Ta', 'T_HP', and 'COP'.
    degree : int, optional
        Degree of the polynomial (default=2). For degree=2, includes terms up to t_a^2, t_hp^2, and t_a*t_hp.
    
    Returns
    -------
    function
        A function that takes (ta, t_hp) as arguments and returns the predicted COP.
    
    Raises
    ------
    ValueError
        If required columns are missing from cooling_df.
    
    Examples
    --------
    >>> cooling_df = load_hp_tes_cooling()
    >>> cop_func = fit_cop_cooling_polynomial(cooling_df, degree=2)
    >>> cop_value = cop_func(ta=30, t_hp=10)
    """
    cop_function, _, _ = _fit_cooling_polynomial_internal(cooling_df, 'COP', degree)
    return cop_function


def get_cop_cooling_coefficients(cooling_df, degree=2):
    """
    Fit a polynomial regression model to cooling COP data and return the fitted coefficients.
    
    The polynomial form is: cop = a0 + a1*t_a + a2*t_hp + a3*t_a*t_hp + a4*t_a^2 + a5*t_hp^2 + ...
    
    Note: For cooling, higher t_hp results in lower COP (inverse relationship compared to heating).
    
    Parameters
    ----------
    cooling_df : pd.DataFrame
        DataFrame containing columns 'Ta', 'T_HP', and 'COP'.
    degree : int, optional
        Degree of the polynomial (default=2). For degree=2, includes terms up to t_a^2, t_hp^2, and t_a*t_hp.
    
    Returns
    -------
    dict
        Dictionary containing the fitted coefficients (a0, a1, a2, ...).
    
    Raises
    ------
    ValueError
        If required columns are missing from cooling_df.
    
    Examples
    --------
    >>> cooling_df = load_hp_tes_cooling()
    >>> coefficients = get_cop_cooling_coefficients(cooling_df, degree=2)
    >>> print(coefficients['a0'], coefficients['a1'], coefficients['a2'])
    """
    _, coefficients, _ = _fit_cooling_polynomial_internal(cooling_df, 'COP', degree)
    coeff_dict = {f'a{i}': float(coeff) for i, coeff in enumerate(coefficients)}
    return coeff_dict


def fit_capacity_cooling_polynomial(cooling_df, degree=2):
    """
    Fit a polynomial regression model to cooling capacity data and return a function to calculate capacity.
    
    The polynomial form is: capacity = a0 + a1*t_a + a2*t_hp + a3*t_a*t_hp + a4*t_a^2 + a5*t_hp^2 + ...
    
    Note: For cooling, higher t_hp results in lower capacity (inverse relationship compared to heating).
    The cooling data typically has only one T_HP value, but the function can predict for different t_hp values.
    
    Parameters
    ----------
    cooling_df : pd.DataFrame
        DataFrame containing columns 'Ta', 'T_HP', and 'Capacity'.
    degree : int, optional
        Degree of the polynomial (default=2). For degree=2, includes terms up to t_a^2, t_hp^2, and t_a*t_hp.
    
    Returns
    -------
    function
        A function that takes (ta, t_hp) as arguments and returns the predicted capacity.
    
    Raises
    ------
    ValueError
        If required columns are missing from cooling_df.
    
    Examples
    --------
    >>> cooling_df = load_hp_tes_cooling()
    >>> capacity_func = fit_capacity_cooling_polynomial(cooling_df, degree=2)
    >>> capacity_value = capacity_func(ta=30, t_hp=10)
    """
    capacity_function, _, _ = _fit_cooling_polynomial_internal(cooling_df, 'Capacity', degree)
    return capacity_function


def get_capacity_cooling_coefficients(cooling_df, degree=2):
    """
    Fit a polynomial regression model to cooling capacity data and return the fitted coefficients.
    
    The polynomial form is: capacity = a0 + a1*t_a + a2*t_hp + a3*t_a*t_hp + a4*t_a^2 + a5*t_hp^2 + ...
    
    Note: For cooling, higher t_hp results in lower capacity (inverse relationship compared to heating).
    
    Parameters
    ----------
    cooling_df : pd.DataFrame
        DataFrame containing columns 'Ta', 'T_HP', and 'Capacity'.
    degree : int, optional
        Degree of the polynomial (default=2). For degree=2, includes terms up to t_a^2, t_hp^2, and t_a*t_hp.
    
    Returns
    -------
    dict
        Dictionary containing the fitted coefficients (a0, a1, a2, ...).
    
    Raises
    ------
    ValueError
        If required columns are missing from cooling_df.
    
    Examples
    --------
    >>> cooling_df = load_hp_tes_cooling()
    >>> coefficients = get_capacity_cooling_coefficients(cooling_df, degree=2)
    >>> print(coefficients['a0'], coefficients['a1'], coefficients['a2'])
    """
    _, coefficients, _ = _fit_cooling_polynomial_internal(cooling_df, 'Capacity', degree)
    coeff_dict = {f'a{i}': float(coeff) for i, coeff in enumerate(coefficients)}
    return coeff_dict


        

def run_fmu(iteration, input_variables_dict, time_points, model):
    '''
    This function runs the Modelica FMU model for the HP-TES system. Certain parameters are initialized at the first iteration. At
    the start of each iteration,
    
    while in subsequent iterations, the model state is restored from the previous run. The function collects relevant outputs 
    into a pandas DataFrame.

    Inputs:
    iteration: current iteration number
    input_variables_dict: dictionary of input variable names and their corresponding values for the FMU model
    time_points: array of time points for the simulation
    model: the Modelica FMU model object

    Outputs:
    df: pandas DataFrame containing simulation results of the FMU model

    Queston: should we collect the exact system mode and determine the FCU mode and pass that into the passive model to update?
    '''
    input_names = list(input_variables_dict.keys())
    input_values = list(input_variables_dict.values())
    
    # Create input trajectory - time points and corresponding values
    # Assuming you want the same input values for all time points
    time_array = np.array([time_points[0], time_points[-1]])
    input_data = np.column_stack([time_array, 
                                 np.tile(input_values, (len(time_array), 1))])
    
    inputs = (input_names, input_data)
    
    opts = model.simulate_options()
    # Initialize Modelica model states at iteration = 1 or use last saved states from last iteration
    try:
        if iteration == 1:
            # opts = model.simulate_options()
            # Set up the experiment at iteration = 1
            model.set('hot_PCM.T_start', 273.15 + Input_Parameters.hot_PCM_T_start)
            model.set('cold_PCM.T_start', 273.15 + Input_Parameters.cold_PCM_T_start)
            model.set('zone_temp_cooling_setpoint', 273.15 + Input_Parameters.T_high)
            model.set('zone_temp_heating_setpoint', 273.15 + Input_Parameters.T_low)
            model.set('zone_temp_setpoint_delta', Input_Parameters.T_delta)
            model.set('AWHP.cycling_wait_time', Input_Parameters.cycling_wait_time) # in seconds, specifying how long the heat pumps cycles
            # model.set('dpTesHot_nominal', 9339) # in Pa
            # model.set('dpTesCold_nominal', 9339) # in Pa
            # model.set('dpValve_nominal', 1000)# in Pa, for all valves in the system
            # model.set('fanCoilUnit_hysteresis_air_control.hex.dp2_nominal', 200)
            #model.set('dpFcuWater_nominal', 4394) # in Pa

            model.set('hot_PCM.TSol', 273.15+Input_Parameters.PCM_H_lower_latent) #temperature of hot pcm when pcm is fully solid
            model.set('hot_PCM.TLiq', 273.15+Input_Parameters.PCM_H_upper_latent) #temperature of hot pcm when pcm is fully liquid
            model.set('hot_PCM.Tes_nominal', 9.5*3600000)  
            # thermal storage capacity of pcm within the latent range (TSol - TLiq) in jules
            model.set('hot_PCM.c', 2050) # sensible heat of pcm outside the latent range in J/kg/K
            # model.set('hot_PCM.LHea', 180800) # latent heat of pcm in J/kg
            model.set('convHot.hFixed', 10.51) 
            #W/K heat loss or gain of pcm per temperature differemce to ambient environment

            model.set('hybrid_2024_controller_new.modular_SOC_Manager1.hysteresis_discharged[1].uHigh', 0.05)
            # when hot pcm falls below 0 SOC, it is "fully discharged"
            # when hot pcm SOC value rises above the set value above, it is no longer "fully discharged"
            model.set('hybrid_2024_controller_new.modular_SOC_Manager1.hysteresis_charged[1].uLow', 0.95)
            # when hot pcm rises above 1 SOC, it is "fully charged"
            # when hot pcm SOC value falls below the set value above, it is no longer "fully charged"


            model.set('cold_PCM.TSol', 273.15+Input_Parameters.PCM_C_lower_latent) #temperature of hot pcm when pcm is fully solid
            model.set('cold_PCM.TLiq', 273.15+Input_Parameters.PCM_C_upper_latent) #temperature of hot pcm when pcm is fully liquid
            model.set('cold_PCM.Tes_nominal', 9.5*3600000)  
            # thermal storage capacity of pcm within the latent range (TSol - TLiq) in jules
            model.set('cold_PCM.c', 2050) # sensible heat of pcm outside the latent range in J/kg/K
            # model.set('cold_PCM.LHea', 126000) # latent heat of pcm in J/kg
            model.set('convCold.hFixed', 14.51) 
            #W/K heat loss or gain of pcm per temperature differemce to ambient environment

            model.set('hybrid_2024_controller_new.modular_SOC_Manager1.hysteresis_discharged[2].uHigh', 0.05)
            # when cold pcm falls below 0 SOC, it is "fully discharged"
            # when cold pcm SOC value rises above the set value above, it is no longer "fully discharged"
            model.set('hybrid_2024_controller_new.modular_SOC_Manager1.hysteresis_charged[2].uLow', 0.95)
            # when cold pcm rises above 1 SOC, it is "fully charged"
            # when cold pcm SOC value falls below the set value above, it is no longer "fully charged"

            model.set('heatExchangerHotScalingFactor', Input_Parameters.heatExchangerHotScalingFactor)
            model.set('heatExchangerColdScalingFactor', Input_Parameters.heatExchangerColdScalingFactor)
            # Unit is none. This is the variable that indicates the phase change thermal battery 
            # heat exchanger conductivity, 
            # the higher the value, the easier it is to transfer heat between the PCM and water
            model.set('UA_nominal', Input_Parameters.UA_nominal) 
            # in W/K. This is the variable that indicates the fan coil unit heat exchanger conductivity, 
            # the higher the value, the easier it is to transfer heat between air and water
            model.set('heat_pump_condenser_evaporator_volume', Input_Parameters.heat_pump_condenser_evaporator_volume) # in m3
            # This varible determines how soon the heat pump supply water temperature can reach the setpoint
            # when switching between cooling mode and heating mode
            
            # parameter for COP of heat pump
            # COP for heating
            model.set('heating_cop_a0', 7.202003525081306)
            model.set('heating_cop_a1', 0.18062703077853665)
            model.set('heating_cop_a2', -0.1477341556516753)
            model.set('heating_cop_a3', -0.002381072467636457)
            model.set('heating_cop_a4', 0.0008319667199337608)
            model.set('heating_cop_a5', 0.0008384541446034042)
            # COP for cooling
            model.set('cooling_cop_a0', 8.650914785834006)
            model.set('cooling_cop_a1', -0.20763083699950602)
            model.set('cooling_cop_a2', -0.06930220567653733)
            model.set('cooling_cop_a3', -0.0003181274443609665)
            model.set('cooling_cop_a4', 0.001455586245110767)
            model.set('cooling_cop_a5', -0.000373410550444541)
            # Capacity for heating
            model.set('heating_capacity_a0', 7662.690876075606)
            model.set('heating_capacity_a1', 278.0926084677243)
            model.set('heating_capacity_a2', 13.264680546280954)
            model.set('heating_capacity_a3', -2.744177232083285)
            model.set('heating_capacity_a4', -0.42115501152681524)
            model.set('heating_capacity_a5', -1.3852105061937135)
            # Capacity for cooling
            model.set('cooling_capacity_a0', 2176.431007927367)
            model.set('cooling_capacity_a1', 14.1847062996531)
            model.set('cooling_capacity_a2', -26.923757618843087)
            model.set('cooling_capacity_a3', -0.10684511342535456)
            model.set('cooling_capacity_a4', -0.628998968008253)
            model.set('cooling_capacity_a5', -0.179778200817228)
            
            opts['initialize'] = True
            opts["CVode_options"]["atol"] = 1e-4
            opts['solver'] = 'CVode'  # Python integrator
            opts["ncp"] = len(time_points)*2
            
        else:
            # Retrieve fmu model ending state from last iteration if iteration > 1
            state = model.get_fmu_state()
            opts['initialize'] = False
            opts['solver'] = 'CVode'  # Python integrator
            opts["ncp"] = len(time_points)*2
            model.set_fmu_state(state)

        # Run the simulation from start_time to final_time with specified inputs and options
        res = model.simulate(start_time=time_points[0], final_time=time_points[-1], input=inputs, options=opts)
    except Exception as e:
        print("FMU Initialization Error:")
        print(model.get_log())
        raise

    # Collect output
    time = res['time']
    data = {
        'Time': time, 
        'FMU mode from MPC': res['schedule_input'],
        'HP Electric Power': res['heat_pump_electric_power'],
        'HP Mode': res['heat_pump_mode'], 
        'HP ON OFF': res['heat_pump_on_off'], 
        'HP Useful Thermal Power': res['heat_pump_useful_thermal_power'],
        'Pump 1 Electric Power': res['pump_1_electric_power'], 
        'Pump 2 Electric Power': res['pump_2_electric_power'], 
        'PCM_Cold_SOC': res['soc_cold_pcm'], 
        'PCM_Hot_SOC': res['soc_hot_pcm'], 
        'PCM_Cold_Temp': res['temp_cold_pcm'], 
        'PCM_Hot_Temp': res['temp_hot_pcm'],
        'Heat Transfer Cold PCM': res['heat_transfer_cold_pcm'], 
        'Heat Transfer Hot PCM': res['heat_transfer_hot_pcm'], 
        'FCU Heat Delivered': res['fcu_heat_delivered'], 
        'Exact System Mode': res['exact_system_mode'], 
        'FCU Entering Water Temperature': res['fcu_entering_water_temperature'], 
        'FCU Leaving Water Temperature': res['fcu_leaving_water_temperature'],
        'Fan Coil Fan Mass Flow': res['fan_coil_fan_mass_flow'], 
        'Heat Pump Supply Water Temperature': res['heat_pump_supply_water_temperature'],
        'Heat Pump Return Water Temperature': res['heat_pump_return_water_temperature'], 
        'Heat Pump Compressor Speed': res['heat_pump_compressor_speed'],
        'COP Heating': res['AWHP.COPHea'],
        'COP Cooling': res['AWHP.COPCoo'],
        'FCU Mode': res['FCUmode'],
        'Pump 1 Mass Flow': res['pump_1_mass_flow'],
        'Pump 2 Mass Flow': res['pump_2_mass_flow'],
        'Fan Coil Fan Electric Power': res['fan_coil_fan_electric_power'],
        'hot pcm not fully discharged': res['hybrid_2024_controller_new.modular_SOC_Manager1.SOC_int[1]'],
        'cold pcm not fully discharged': res['hybrid_2024_controller_new.modular_SOC_Manager1.SOC_int[2]']
    }

    # This contains full information (at all timesteps) of the simulation
    df = pd.DataFrame(data)

    cleaned_output = {
        # 'Iteration': iteration,
        # End States
        'Time': df['Time'].iloc[-1],  
        'FMU mode from MPC': df['FMU mode from MPC'].iloc[-1],
        'Exact System Mode': df['Exact System Mode'].iloc[-1], 
        'FCU Mode': df['FCU Mode'].iloc[-1],
        'HP Mode': df['HP Mode'].iloc[-1], 
        'HP ON OFF': df['HP ON OFF'].iloc[-1], 
        'hot pcm not fully discharged': df['hot pcm not fully discharged'].iloc[-1],
        'cold pcm not fully discharged': df['cold pcm not fully discharged'].iloc[-1],
        'PCM_Cold_SOC': df['PCM_Cold_SOC'].iloc[-1], # [1]
        'PCM_Hot_SOC': df['PCM_Hot_SOC'].iloc[-1], # [1]
        'PCM_Cold_Temp': df['PCM_Cold_Temp'].iloc[-1]-273.15, # [°C]
        'PCM_Hot_Temp': df['PCM_Hot_Temp'].iloc[-1]-273.15, # [°C]
        'FCU Entering Water Temperature': df['FCU Entering Water Temperature'].iloc[-1]-273.15, # [°C]
        'FCU Leaving Water Temperature': df['FCU Leaving Water Temperature'].iloc[-1]-273.15, # [°C]
        'Heat Pump Supply Water Temperature': df['Heat Pump Supply Water Temperature'].iloc[-1]-273.15, # [°C]
        'Heat Pump Return Water Temperature': df['Heat Pump Return Water Temperature'].iloc[-1]-273.15, # [°C]
        'Heat Pump Compressor Speed': df['Heat Pump Compressor Speed'].iloc[-1], # [1]
        
        # Average
        'COP Heating': Data_Conversion.find_average(df, 'COP Heating'), # [1]
        'COP Cooling': Data_Conversion.find_average(df, 'COP Cooling'), # [1]
        'Fan Coil Fan Mass Flow': Data_Conversion.find_average(df, 'Fan Coil Fan Mass Flow'), # [kg/s]
        'Pump 1 Mass Flow': Data_Conversion.find_average(df, 'Pump 1 Mass Flow'), # [kg/s]
        'Pump 2 Mass Flow': Data_Conversion.find_average(df, 'Pump 2 Mass Flow'), # [kg/s]

        # Average (for Energy Aggregation)
        'HP Electric Power': Data_Conversion.find_average(df, 'HP Electric Power')/1000,  # [kW]
        'HP Useful Thermal Power': Data_Conversion.find_average(df, 'HP Useful Thermal Power')/1000,  # [kW]
        'Pump 1 Electric Power': Data_Conversion.find_average(df, 'Pump 1 Electric Power')/1000,  # [kW]
        'Pump 2 Electric Power': Data_Conversion.find_average(df, 'Pump 2 Electric Power')/1000,  # [kW]
        'Heat Transfer Cold PCM': Data_Conversion.find_average(df, 'Heat Transfer Cold PCM')/1000,  # [kW]
        'Heat Transfer Hot PCM': Data_Conversion.find_average(df, 'Heat Transfer Hot PCM')/1000,  # [kW]
        'FCU Heat Delivered': Data_Conversion.find_average(df, 'FCU Heat Delivered')/1000,  # [kW]
        'Fan Coil Fan Electric Power': Data_Conversion.find_average(df, 'Fan Coil Fan Electric Power')/1000  # [kW]
    }

    return df, cleaned_output


