"""Earth Engine asset operations via the native ``ee.data`` API.

Replaces the ``earthengine`` CLI subprocess calls in the retired
``map/assets.py`` (quarantined at ``legacy/assets_cli.py``) for the two
operations the pipeline actually uses. Return/behavior parity with the CLI:
``list_assets`` returns bare asset ids (``users/...`` for legacy assets,
``projects/.../assets/...`` otherwise) and ``copy_asset`` refuses to
overwrite an existing destination.
"""

import ee

_LEGACY_PREFIX = 'projects/earthengine-legacy/assets/'


def _parent(location):
    if location.startswith('projects/'):
        return location
    return _LEGACY_PREFIX + location


def copy_asset(ee_asset, dst):
    ee.data.copyAsset(ee_asset, dst)
    print('copied {} to {}'.format(ee_asset, dst))


def list_assets(location):
    response = ee.data.listAssets({'parent': _parent(location)})
    names = [a['name'] for a in response.get('assets', [])]
    return [n[len(_LEGACY_PREFIX):] if n.startswith(_LEGACY_PREFIX) else n
            for n in names]
