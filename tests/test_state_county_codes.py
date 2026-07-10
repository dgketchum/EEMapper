"""Phase 1 safety-net tests for irrmapper/data/state_county_codes.py.

These lock down the current behavior of the pure lookup-dictionary builders
before a refactor. They are hermetic (no I/O, no network, no Earth Engine).
"""

from irrmapper.data.state_county_codes import (
    state_name_abbreviation,
    state_fips_code,
    state_county_code,
    county_acres,
)


# --------------------------------------------------------------------------- #
# state_fips_code
# --------------------------------------------------------------------------- #
def test_state_fips_code_size_and_known_values():
    fips = state_fips_code()
    # 50 states + DC + 5 territories (AS, GU, MP, PR, VI) == 56 entries
    assert len(fips) == 56
    # spot-check well-known FIPS codes
    assert fips["MT"] == "30"
    assert fips["CA"] == "06"
    assert fips["CO"] == "08"
    assert fips["AK"] == "02"
    assert fips["WY"] == "56"


def test_state_fips_codes_are_two_char_strings():
    fips = state_fips_code()
    for abbr, code in fips.items():
        assert isinstance(code, str)
        # every FIPS code is a zero-padded two-character string
        assert len(code) == 2


# --------------------------------------------------------------------------- #
# state_name_abbreviation
# --------------------------------------------------------------------------- #
def test_state_name_abbreviation_size_and_values():
    abbr = state_name_abbreviation()
    assert len(abbr) == 56
    assert abbr["MT"] == "Montana"
    assert abbr["CA"] == "California"
    assert abbr["DC"] == "District of Columbia"


def test_abbreviation_and_fips_share_keys():
    # the abbreviation table and the FIPS table cover exactly the same states
    assert set(state_name_abbreviation().keys()) == set(state_fips_code().keys())


# --------------------------------------------------------------------------- #
# state_county_code
# --------------------------------------------------------------------------- #
def test_state_county_code_size_and_missing_territories():
    scc = state_county_code()
    # county-level data covers 52 units: the 56 FIPS states minus the four
    # territories that have no county breakdown here (AS, GU, MP, VI).
    assert len(scc) == 52
    assert set(state_fips_code()) - set(scc) == {"AS", "GU", "MP", "VI"}
    # no county-code state is missing from the FIPS table
    assert set(scc) - set(state_fips_code()) == set()


def test_montana_county_entries():
    scc = state_county_code()
    mt = scc["MT"]
    # Montana has 56 counties
    assert len(mt) == 56
    assert mt["093"] == {"GEOID": "30093", "NAME": "Silver Bow"}


def test_geoid_equals_statefips_plus_county_code():
    # cross-consistency: every county GEOID is exactly state FIPS + county code
    fips = state_fips_code()
    scc = state_county_code()
    for state, counties in scc.items():
        state_fips = fips[state]
        for county_code, info in counties.items():
            assert info["GEOID"] == state_fips + county_code
            assert set(info.keys()) == {"GEOID", "NAME"}


# --------------------------------------------------------------------------- #
# county_acres
# --------------------------------------------------------------------------- #
def test_county_acres_size_and_structure():
    ca = county_acres()
    assert len(ca) == 3220
    # keys are five-character FIPS+county GEOID strings
    for key in list(ca)[:25]:
        assert isinstance(key, str)
        assert len(key) == 5
    # each value carries a 'land' and 'water' float area
    val = ca["01001"]
    assert set(val.keys()) == {"land", "water"}
    assert isinstance(val["land"], float)
    assert isinstance(val["water"], float)
    # spot-check a known value (locked in as-is)
    assert abs(ca["01001"]["land"] - 380446.73969447915) < 1e-6


def test_county_acres_keys_match_state_county_geoids():
    # cross-consistency: county_acres is keyed by exactly the set of GEOIDs
    # produced by state_county_code()
    scc = state_county_code()
    all_geoids = {
        info["GEOID"] for counties in scc.values() for info in counties.values()
    }
    assert set(county_acres().keys()) == all_geoids
