from datetime import datetime

from zoneinfo import ZoneInfo


import numpy as np

import pandas as pd

import Input_Parameters


from pvlib.irradiance import get_total_irradiance

from pvlib.location import Location


# Weather DateTime from NSRDB / Open-Meteo is naive America/Los_Angeles wall-clock time.

SITE_TIMEZONE = "America/Los_Angeles"

_SITE_TZINFO = ZoneInfo(SITE_TIMEZONE)


# pvlib convention: 180° = south

_SOUTH_AZIMUTH = 180


def fuentes(poa_global, temp_air, wind_speed, noct,

            module_height=1.0, wind_height=10.0,

            emissivity=0.84, absorption=0.83,

            surface_tilt=30, module_width=1.0, module_length=1.6):

    """Fuentes cell temperature model implementation."""

    σ = 5.670373e-8

    h_conv = 5.7 + 3.8 * wind_speed


    temp_sky = (0.0552 * (temp_air + 273.15) ** 1.5) - 273.15

    temp_back = temp_air + (noct - 20) / 800 * poa_global


    numerator = poa_global * absorption

    numerator += h_conv * temp_air

    numerator += σ * emissivity * (temp_sky + 273.15) ** 4

    numerator -= σ * emissivity * (temp_back + 273.15) ** 4


    denominator = h_conv + 4 * σ * emissivity * (temp_back + 273.15) ** 3

    return numerator / denominator


def _localize_wall_clock_index(dt_index):

    """

    Localize naive local wall-clock times to America/Los_Angeles.


    Fall-DST repeats 01:00-01:59: first pass fold=0 (PDT), second fold=1 (PST).

    Spring-DST gaps are shifted forward one hour.

    """

    dt_index = pd.DatetimeIndex(pd.to_datetime(dt_index))

    if dt_index.tz is not None:

        return dt_index.tz_convert(SITE_TIMEZONE)


    seen = {}

    localized = []

    for ts in dt_index:

        key = ts.value

        fold = seen.get(key, 0)

        seen[key] = fold + 1

        naive = ts.to_pydatetime()

        try:

            aware = datetime(

                naive.year, naive.month, naive.day,

                naive.hour, naive.minute, naive.second, naive.microsecond,

                tzinfo=_SITE_TZINFO, fold=fold,

            )

        except Exception:

            shifted = ts + pd.Timedelta(hours=1)

            sn = shifted.to_pydatetime()

            aware = datetime(

                sn.year, sn.month, sn.day,

                sn.hour, sn.minute, sn.second, sn.microsecond,

                tzinfo=_SITE_TZINFO, fold=0,

            )

        localized.append(aware)

    return pd.DatetimeIndex(localized)


def _prepare_weather_index(weather):

    """Copy weather, index by DateTime, localize naive LA wall-clock timestamps."""

    weather = weather.copy()

    weather["DateTime"] = pd.to_datetime(weather["DateTime"])

    weather = weather.set_index("DateTime")

    if weather.index.tz is None:

        weather.index = _localize_wall_clock_index(weather.index)

    else:

        weather.index = weather.index.tz_convert(SITE_TIMEZONE)

    return weather


def _zenith_iam(solar_zenith):

    """Simple IAM from solar zenith (original model)."""

    zenith = np.asarray(solar_zenith, dtype=float)

    return np.where(zenith < 90, 1 - 0.05 * ((90 - zenith) / 90) ** 2, 0.0)


def _rated_kw_per_kwp():

    rated_kw = Input_Parameters.P_dc0 * Input_Parameters.η_PV

    if rated_kw <= 0:

        raise ValueError("P_dc0 * η_PV must be positive for capacity factor normalization")

    return rated_kw


def _solar_site_and_position(weather):

    """Shared site setup: weather must already be localized."""

    site = Location(

        latitude=Input_Parameters.latitude,

        longitude=Input_Parameters.longitude,

        tz=SITE_TIMEZONE,

    )

    solar_position = site.get_solarposition(weather.index)

    return site, solar_position


def generate_pv(weather, pv_capacity=None):

    """

    Single south-facing array at module_surface_tilt.


    Modes:

      pv_capacity=None -> capacity factor [0-1] (no installed kWp)

      pv_capacity set  -> AC power [kW] for that installed kWp


    Required columns: DateTime, DNI, GHI, DHI, Ta, WS

    """

    weather = _prepare_weather_index(weather)

    _, solar_position = _solar_site_and_position(weather)

    solar_zenith = solar_position["zenith"]


    poa = get_total_irradiance(

        surface_tilt=Input_Parameters.module_surface_tilt,

        surface_azimuth=_SOUTH_AZIMUTH,

        dni=weather["DNI"],

        ghi=weather["GHI"],

        dhi=weather["DHI"],

        solar_zenith=solar_zenith,

        solar_azimuth=solar_position["azimuth"],

        model="isotropic",

    )


    temp_cell = fuentes(

        poa["poa_global"],

        weather["Ta"],

        weather["WS"],

        Input_Parameters.noct_installed,

        module_height=Input_Parameters.module_height,

        wind_height=Input_Parameters.wind_height,

        emissivity=Input_Parameters.module_emissivity,

        absorption=Input_Parameters.module_absorption,

        surface_tilt=Input_Parameters.module_surface_tilt,

        module_width=Input_Parameters.module_width,

        module_length=Input_Parameters.module_length,

    )


    iam = _zenith_iam(solar_zenith)

    effective_poa = poa["poa_global"] * iam * Input_Parameters.optical_loss

    p_dc = (effective_poa / 1000) * Input_Parameters.P_dc0 * (

        1 + Input_Parameters.Γ_t * (temp_cell - Input_Parameters.T_ref)

    )

    power_per_kwp = np.maximum(p_dc * Input_Parameters.η_PV, 0.0)


    # Zero out when sun is below horizon

    ghi = np.asarray(weather["GHI"], dtype=float)
    sun_up = (np.asarray(solar_zenith) < 90) & (ghi > 1.0)
    power_per_kwp = np.where(sun_up, power_per_kwp, 0.0)


    if pv_capacity is not None:

        return power_per_kwp * pv_capacity


    return np.clip(power_per_kwp / _rated_kw_per_kwp(), 0.0, 1.0)


def pv_capacity_factor(weather):

    """South-facing capacity factor [0-1]; no installed kWp."""

    return generate_pv(weather)


def validate_pv_cf(weather, pv_cf, time_resolution_min=5.0, summer_date="2023-06-15"):

    """

    Sanity checks for south-facing CF output and timezone alignment.


    Returns dict of check results; raises ValueError if any critical check fails.

    """

    pv_cf = np.asarray(pv_cf, dtype=float)

    dt = pd.to_datetime(weather["DateTime"])


    checks = {}

    checks["cf_min"] = float(pv_cf.min())

    checks["cf_max"] = float(pv_cf.max())

    checks["cf_in_range"] = checks["cf_min"] >= 0 and checks["cf_max"] <= 1.0 + 1e-9


    night = np.asarray(weather["GHI"], dtype=float) <= 1.0

    if night.any():

        checks["cf_zero_at_night"] = float(pv_cf[night].max()) < 1e-6

    else:

        checks["cf_zero_at_night"] = True


    summer_mask = dt.dt.date == pd.Timestamp(summer_date).date()

    if summer_mask.any():

        day_cf = pv_cf[summer_mask.to_numpy()]

        active_h = (day_cf > 0.01).sum() * time_resolution_min / 60

        checks["summer_active_hours"] = float(active_h)


        day_w = weather.loc[summer_mask]

        ghi = np.asarray(day_w["GHI"], dtype=float)

        if ghi.max() > 50:

            ghi_peak_idx = int(np.argmax(ghi))

            cf_peak_idx = int(np.argmax(day_cf))

            # At 5-min resolution, peaks should be within ~2 hours after timezone fix

            checks["ghi_cf_peak_offset_steps"] = abs(ghi_peak_idx - cf_peak_idx)

            checks["timezone_aligned"] = checks["ghi_cf_peak_offset_steps"] <= 24

        else:

            checks["timezone_aligned"] = True

    else:

        checks["summer_active_hours"] = None

        checks["timezone_aligned"] = True


    failures = []

    if not checks["cf_in_range"]:

        failures.append(f"CF out of [0,1]: min={checks['cf_min']}, max={checks['cf_max']}")

    if not checks["cf_zero_at_night"]:

        failures.append("CF not zero when GHI<=1")

    if checks["summer_active_hours"] is not None and checks["summer_active_hours"] < 8:

        failures.append(f"Summer daylight too short: {checks['summer_active_hours']:.1f}h")

    if not checks["timezone_aligned"]:

        failures.append(

            f"GHI/CF peaks misaligned by {checks.get('ghi_cf_peak_offset_steps')} steps "

            "(possible timezone issue)"

        )


    checks["passed"] = len(failures) == 0

    if failures:

        raise ValueError("PV CF validation failed: " + "; ".join(failures))

    return checks
