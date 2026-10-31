"""Canonical observation factors shared by OELG exports and evaluation."""

STANDARD_FACTORS = (
    'Precipitation', 'Water Body Area', 'Water Level', 'Water Flow Rate',
    'Water Flow Velocity', 'Water Quality', 'Cloud Top Height',
    'Cloud Top Temperature', 'Cloud Fraction', 'Land Cover',
    'Vegetation Fraction', 'Soil Moisture', 'Surface Temperature',
    'Air Temperature', 'Air Humidity', 'Atmospheric Pressure',
    'Wind Speed', 'Wind Direction', 'Surface Deformation',
    'Infrastructure Deformation', 'Subsidence Depth', 'Crack Width',
    'Tilt Angle', 'Seepage Pressure', 'Structural Characteristics',
)


def factor_key(value):
    return ''.join(c for c in str(value) if c.isalnum()).casefold()


_CANONICAL = {factor_key(name):name for name in STANDARD_FACTORS}


def canonical_factors(names):
    """Normalize spacing/case only; do not infer standards from raw factor names."""
    result = set()
    for name in names:
        canonical = _CANONICAL.get(factor_key(name))
        if canonical is None:
            raise ValueError(f'Not one of the 25 StandardFactor names: {name!r}. Refresh standard-factor exports; raw factors are not guessed or translated.')
        result.add(canonical)
    return sorted(result)
