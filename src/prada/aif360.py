from .dataset import Binary
import os
import pandas as pd

# Medical Expenditure Panel Survey (MEPS)
# https://meps.ahrq.gov/
# Panel 21 (2016), includes all respondents.
# TODO: can be expanded to other years/states
# Protected attributes: "RACE" (numeric: 1.0=non-Hispanic White, 0.0=other), "SEX" (numeric: 1.0=Male, 2.0=Female).
# Target: high utilizer (>= 10 healthcare visits in the year).

_MEPS_URLS = {
    'h192': 'https://meps.ahrq.gov/mepsweb/data_files/pufs/h192ssp.zip',  # 2016
}

_MEPS_FEATURES = [
    'REGION', 'AGE', 'SEX', 'RACE', 'MARRY',
    'FTSTU', 'ACTDTY', 'HONRDC', 'RTHLTH', 'MNHLTH', 'HIBPDX', 'CHDDX', 'ANGIDX',
    'MIDX', 'OHRTDX', 'STRKDX', 'EMPHDX', 'CHBRON', 'CHOLDX', 'CANCERDX', 'DIABDX',
    'JTPAIN', 'ARTHDX', 'ARTHTYPE', 'ASTHDX', 'ADHDADDX', 'PREGNT', 'WLKLIM',
    'ACTLIM', 'SOCLIM', 'COGLIM', 'DFHEAR42', 'DFSEE42', 'ADSMOK42',
    'PCS42', 'MCS42', 'K6SUM42', 'PHQ242', 'EMPST', 'POVCAT', 'INSCOV', 'UTILIZATION',
]

# Categorical features to one-hot encode (SEX handled separately as binary 'male')
_MEPS_CATEGORICAL = [
    'REGION', 'MARRY', 'FTSTU', 'ACTDTY', 'HONRDC', 'RTHLTH', 'MNHLTH',
    'HIBPDX', 'CHDDX', 'ANGIDX', 'MIDX', 'OHRTDX', 'STRKDX', 'EMPHDX', 'CHBRON',
    'CHOLDX', 'CANCERDX', 'DIABDX', 'JTPAIN', 'ARTHDX', 'ARTHTYPE', 'ASTHDX',
    'ADHDADDX', 'PREGNT', 'WLKLIM', 'ACTLIM', 'SOCLIM', 'COGLIM', 'DFHEAR42',
    'DFSEE42', 'ADSMOK42', 'PHQ242', 'EMPST', 'POVCAT', 'INSCOV',
]

# Columns checked for data quality (values < -1 indicate missing/unknown)
_MEPS_QUALITY_COLS = [
    'FTSTU', 'ACTDTY', 'HONRDC', 'RTHLTH', 'MNHLTH', 'HIBPDX', 'CHDDX', 'ANGIDX',
    'EDUCYR', 'HIDEG', 'MIDX', 'OHRTDX', 'STRKDX', 'EMPHDX', 'CHBRON', 'CHOLDX',
    'CANCERDX', 'DIABDX', 'JTPAIN', 'ARTHDX', 'ARTHTYPE', 'ASTHDX', 'ADHDADDX',
    'PREGNT', 'WLKLIM', 'ACTLIM', 'SOCLIM', 'COGLIM', 'DFHEAR42', 'DFSEE42',
    'ADSMOK42', 'PHQ242', 'EMPST', 'POVCAT', 'INSCOV',
]


def _ensure_meps_csv(meps_dir, stem):
    """Download MEPS SSP zip from AHRQ and convert to CSV if not already present."""
    csv_path = os.path.join(meps_dir, f'{stem}.csv')
    if os.path.exists(csv_path):
        return csv_path

    try:
        import pyreadstat
    except ImportError:
        raise ImportError(
            "pyreadstat is required to convert MEPS SSP files. "
            "Install with: pip install pyreadstat"
        )

    import requests
    import zipfile
    import tempfile

    url = _MEPS_URLS[stem]
    print(f"Downloading {url} (~13-16 MB)...")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, f'{stem}ssp.zip')
        with open(zip_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)

        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmpdir)

        ssp_path = os.path.join(tmpdir, f'{stem}.ssp')
        print(f"Converting {stem}.ssp to CSV (this may take a minute)...")
        df, _ = pyreadstat.read_xport(ssp_path)
        os.makedirs(meps_dir, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Saved to {csv_path}")

    return csv_path


def _meps_preprocess(df, panel, year):
    """Preprocessing keeping all races and ethnicities."""
    yr = str(year)[-2:]  # '15' or '16'

    df = df[df['PANEL'] == panel].copy()

    # Categorize race: non-Hispanic White=1.0, all others=0.0
    def categorize_race(row):
        if (row['HISPANX'] == 2) and (row['RACEV2X'] == 1):  # non-Hispanic White
            return 1.0
        return 0.0
    
    df['RACE'] = df.apply(categorize_race, axis=1)

    # Panel-specific column renames (same structure across panels 19, 20, 21)
    df = df.rename(columns={
        'FTSTU53X': 'FTSTU', 'ACTDTY53': 'ACTDTY', 'HONRDC53': 'HONRDC',
        'RTHLTH53': 'RTHLTH', 'MNHLTH53': 'MNHLTH', 'CHBRON53': 'CHBRON',
        'JTPAIN53': 'JTPAIN', 'PREGNT53': 'PREGNT', 'WLKLIM53': 'WLKLIM',
        'ACTLIM53': 'ACTLIM', 'SOCLIM53': 'SOCLIM', 'COGLIM53': 'COGLIM',
        'EMPST53': 'EMPST', 'REGION53': 'REGION', 'MARRY53X': 'MARRY',
        'AGE53X': 'AGE', f'POVCAT{yr}': 'POVCAT', f'INSCOV{yr}': 'INSCOV',
    })

    # Data quality filters matching AIF360
    df = df[df['REGION'] >= 0]
    df = df[df['AGE'] >= 0]
    df = df[df['MARRY'] >= 0]
    df = df[df['ASTHDX'] >= 0]
    df = df[(df[_MEPS_QUALITY_COLS] >= -1).all(axis=1)]

    # Utilization: total healthcare visits >= 10
    visit_cols = [f'OBTOTV{yr}', f'OPTOTV{yr}', f'ERTOT{yr}', f'IPNGTD{yr}', f'HHTOTD{yr}']
    df['UTILIZATION'] = (df[visit_cols].sum(axis=1) >= 10).astype(float)

    # Keep only needed features and drop rows with NaNs in them
    available = [f for f in _MEPS_FEATURES if f in df.columns]
    df = df[available].dropna()

    return df


def _meps_transform_X_y(df, _):
    y = df['UTILIZATION'].astype(bool)
    X = df.drop(columns=['UTILIZATION']).copy()

    # One-hot encode categoricals (keep RACE and SEX as features)
    cat_cols = [c for c in _MEPS_CATEGORICAL if c in X.columns]
    X = pd.get_dummies(X, columns=cat_cols, prefix_sep='=')

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    return X, y


def _load_meps(self, panel, year, stem, force=False):
    if not self.hdf5_exists() or force:
        if not self.silent:
            print(f"loading {self.name()} (MEPS panel {panel}, {year})")
        meps_dir = os.path.join(self.data_dir, 'meps')
        csv_path = _ensure_meps_csv(meps_dir, stem)
        df = pd.read_csv(csv_path)
        df = _meps_preprocess(df, panel, year)
        X, y = self._transform_X_y(df, None)
        X, y = self._cast_X_y(X, y)
        self.store_hdf5(X, y)
    else:
        X, y = self.load_hdf5()
    return X, y


class MEPSPanel21(Binary):
    # Predict high healthcare utilization, MEPS 2016 panel 21 (all respondents, White vs Non-White). Protected: RACE, SEX.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.source = "meps"
        self.url = "https://meps.ahrq.gov/"

    def load_dataset(self, force=False):
        if self.X is None or self.y is None:
            self.X, self.y = _load_meps(self, panel=21, year=2016, stem='h192', force=force)
            super().load_dataset()

    def _transform_X_y(self, df, _):
        return _meps_transform_X_y(df, _)
