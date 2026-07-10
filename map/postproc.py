"""Config-driven post-processing.

export_special() here replaces call_ee.export_special(): the per-state
if/elif rule chains become data in the run config (configs/irrmapper_v1_2.toml).
tests/test_postproc_config_equivalence.py asserts the expressions built from
the canonical config are byte-identical to the legacy function's golden
fixture; the EE call sequence deliberately mirrors the original line for
line so the exported images are unchanged.
"""

import os

import ee

from map.assets import copy_asset
from map.cdl import get_cdl
from map.ee_utils import landsat_composites


def export_special(cfg, input_coll, out_coll, roi, state, years, start_tasks=True,
                   extra_props=None):
    """Post-process raw RF classifications for one state over the given years.

    Rules come from cfg.postproc[state]. With start_tasks=False the export
    tasks are built but not started (dry-run). extra_props are merged into
    the exported image's properties (run provenance); copy_only states copy
    the source asset verbatim and cannot be stamped.
    """
    rules = cfg.postproc[state]
    fc = ee.FeatureCollection(roi)
    ned = ee.Image("USGS/NED")
    slope = ee.Terrain.products(ned).select("slope")

    tasks = []
    for year in years:
        if rules.copy_only:
            src = os.path.join(input_coll, "{}_{}".format(state, year))
            dst = os.path.join(out_coll, "{}_{}".format(state, year))
            print("No rule written for this {}, copying".format(state))
            copy_asset(src, dst)
            continue

        start, end = "{}-03-01".format(year), "{}-12-30".format(year)
        ndvi = landsat_composites(
            year, start, end, fc, "gs", composites_only=True
        ).select("nd_max_gs")

        cropland = get_cdl(year)[1].select("cropland")

        target = ee.Image(os.path.join(input_coll, "{}_{}".format(state, year)))
        props = target.getInfo()["properties"]
        target = target.select("classification").clip(fc.geometry())

        sum_coll = ee.ImageCollection(input_coll)
        remap = ee.ImageCollection(sum_coll).map(
            lambda x: x.select("classification").remap([0, 1, 2, 3], [1, 0, 0, 0])
        )
        sum_img = remap.sum().rename("sum")

        pivot = ee.FeatureCollection(rules.pivot_asset_for_year(year)).filterBounds(fc)
        class_labels = ee.Image(0).byte()
        pivot = class_labels.paint(pivot, 1).rename("pivot")
        expr = target.addBands([sum_img, ndvi, slope, cropland, pivot])

        threshold = rules.threshold.for_year(year)

        sources = {
            "IRR": expr.select("classification"),
            "SUM": expr.select("sum"),
            "NDVI": expr.select("nd_max_gs"),
            "SLOPE": expr.select("slope"),
            "PIVOT": expr.select("pivot"),
        }

        expression_ = None
        first = True
        for step in rules.steps:
            if step.min_year is not None and year < step.min_year:
                continue
            expression_ = step.expression.format(t=threshold)
            if not first:
                sources["IRR"] = target.select("classification")
            target = (expr if first else target).expression(
                expression_, {v: sources[v] for v in step.variables}
            )
            first = False

        props.update({"post_process": expression_})
        if extra_props:
            props.update(extra_props)
        target = target.set(props)
        target = target.rename("classification")

        desc = "{}_{}".format(state, year)
        _id = os.path.join(out_coll, desc)
        task = ee.batch.Export.image.toAsset(
            target,
            description=desc,
            pyramidingPolicy={".default": "mode"},
            assetId=_id,
            scale=30,
            maxPixels=1e13,
        )

        if start_tasks:
            task.start()
            print(year, _id)
        tasks.append(task)

    return tasks
