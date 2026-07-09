import os

import pytest


def _has_ee_credentials():
    """Cheap check for Earth Engine credentials without initializing."""
    persistent = os.path.expanduser("~/.config/earthengine/credentials")
    return (
        os.path.exists(persistent)
        or "GOOGLE_APPLICATION_CREDENTIALS" in os.environ
        or "EARTHENGINE_TOKEN" in os.environ
    )


def pytest_collection_modifyitems(config, items):
    if _has_ee_credentials():
        return
    skip_ee = pytest.mark.skip(reason="requires authenticated Earth Engine credentials")
    for item in items:
        if "ee" in item.keywords:
            item.add_marker(skip_ee)


@pytest.fixture(scope="session")
def ee_initialized():
    """Initialize Earth Engine once per session for ee-marked tests."""
    import ee

    ee.Initialize(project="ee-dgketchum")
    return True
