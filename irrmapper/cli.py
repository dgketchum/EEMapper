"""Config-driven pipeline runner.

Replaces the edit-and-comment loop in statewise.py: every parameter a stage
needs comes from a TOML run config (configs/irrmapper_v1_2.toml), and every
executed export is stamped with the resolved-run provenance.

The default is a dry run: each stage builds its planned Earth Engine calls
and prints them without initializing EE or starting tasks. Pass
--execute (CLI) or dry_run=False to run them; a resolved manifest is then
written to [paths].run_manifests and its digest set on every exported asset.

    uv run irrmapper configs/irrmapper_v1_2.toml classify
    uv run irrmapper configs/irrmapper_v1_2.toml classify --states MT --execute
"""

import argparse
import json
import os

from irrmapper import auth
from irrmapper import postproc
from irrmapper.config import (
    asset_properties,
    load_config,
    resolve_path,
    resolved_manifest,
    write_manifest,
)
from irrmapper.models import rf_ee
from irrmapper.sampling import extracts


def _states(cfg, states):
    states = states or cfg.run.states
    unknown = set(states) - set(cfg.run.states)
    if unknown:
        raise ValueError("state(s) {} not in run.states".format(sorted(unknown)))
    return states


def feature_list(cfg, state):
    """The archived variable-importance feature list for a state's model."""
    path = os.path.join(
        resolve_path(cfg.paths.variable_importance),
        "variables_{}_{}.json".format(state, cfg.run.vintage_glob),
    )
    with open(path) as fp:
        d = json.load(fp)
    return [f[0] for f in d[state]]


def plan_extract(cfg, years, points_glob, out_glob, states=None):
    """Point-sample the feature stack: extracts.request_band_extract per state."""
    plans = []
    for state in _states(cfg, states):
        plans.append({
            "stage": "extract",
            "file_prefix": "bands_{}_{}".format(state, out_glob),
            "points_layer": os.path.join(
                cfg.paths.points, "points_{}_{}".format(state, points_glob)),
            "region": os.path.join(cfg.paths.boundaries, state),
            "years": list(years),
            "filter_bounds": True,
            "buffer": 1e5,
            "southern": state in cfg.run.southern_states,
            "diagnose": False,
        })
    return plans


def plan_classify(cfg, states=None, years=None):
    """Train and apply the per-state RF: rf_ee.export_classification."""
    plans = []
    for state in _states(cfg, states):
        plans.append({
            "stage": "classify",
            "out_name": state,
            "table": os.path.join(
                cfg.paths.training_tables,
                "{}_{}".format(state, cfg.run.vintage_glob)),
            "asset_root": cfg.paths.raw_collection,
            "region": os.path.join(cfg.paths.boundaries, state),
            "years": list(years or cfg.run.years),
            "input_props": feature_list(cfg, state),
            "bag_fraction": cfg.model.bag_fraction,
            "southern": state in cfg.run.southern_states,
        })
    return plans


def plan_postprocess(cfg, states=None, years=None):
    """Apply the per-state cleanup rules: postproc.export_special."""
    plans = []
    for state in _states(cfg, states):
        plans.append({
            "stage": "postprocess",
            "input_coll": cfg.paths.raw_collection,
            "out_coll": cfg.paths.comp_collection,
            "roi": os.path.join(cfg.paths.boundaries, state),
            "state": state,
            "years": list(years or cfg.run.years),
        })
    return plans


def plan_rasters(cfg, years, states=None, min_years=3, export_freq=True):
    """Export boolean/frequency GeoTIFFs to GCS: postproc.export_raster."""
    plans = []
    for state in _states(cfg, states):
        plans.append({
            "stage": "rasters",
            "irr_coll": cfg.paths.comp_collection,
            "roi": os.path.join(cfg.paths.boundaries, state),
            "years": list(years),
            "min_years": min_years,
            "state": state,
            "export_freq": export_freq,
        })
    return plans


def _show(plans):
    print(json.dumps(plans, indent=2))
    print("{} task group(s) planned (dry run; pass --execute to start)".format(
        len(plans)))


def _begin_run(cfg, config_path):
    """Authorize EE, write the resolved manifest, return the asset props."""
    if not auth.is_authorized(project=cfg.project.ee_project):
        raise RuntimeError("Earth Engine authorization failed")
    manifest = resolved_manifest(cfg, config_path)
    path = write_manifest(manifest, resolve_path(cfg.paths.run_manifests))
    print("run manifest: {}".format(path))
    return asset_properties(manifest)


def run_extract(cfg, years, points_glob, out_glob, states=None, dry_run=True,
                config_path=None):
    plans = plan_extract(cfg, years, points_glob, out_glob, states)
    if dry_run:
        _show(plans)
        return plans
    _begin_run(cfg, config_path)
    for p in plans:
        kwargs = {k: v for k, v in p.items() if k != "stage"}
        extracts.request_band_extract(**kwargs)
    return plans


def run_classify(cfg, states=None, years=None, dry_run=True, config_path=None):
    plans = plan_classify(cfg, states, years)
    if dry_run:
        _show(plans)
        return plans
    props = _begin_run(cfg, config_path)
    for p in plans:
        kwargs = {k: v for k, v in p.items() if k != "stage"}
        rf_ee.export_classification(extra_props=props, **kwargs)
    return plans


def run_postprocess(cfg, states=None, years=None, dry_run=True, config_path=None):
    plans = plan_postprocess(cfg, states, years)
    if dry_run:
        _show(plans)
        return plans
    props = _begin_run(cfg, config_path)
    for p in plans:
        kwargs = {k: v for k, v in p.items() if k != "stage"}
        postproc.export_special(cfg, extra_props=props, **kwargs)
    return plans


def run_rasters(cfg, years, states=None, min_years=3, export_freq=True,
                dry_run=True, config_path=None):
    plans = plan_rasters(cfg, years, states, min_years, export_freq)
    if dry_run:
        _show(plans)
        return plans
    _begin_run(cfg, config_path)
    for p in plans:
        kwargs = {k: v for k, v in p.items() if k != "stage"}
        postproc.export_raster(**kwargs)
    return plans


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", help="TOML run config")
    parser.add_argument("stage",
                        choices=["extract", "classify", "postprocess", "rasters"])
    parser.add_argument("--states", nargs="+", default=None,
                        help="subset of run.states (default: all)")
    parser.add_argument("--years", nargs="+", type=int, default=None,
                        help="override run.years (required for extract/rasters)")
    parser.add_argument("--points-glob", default=None,
                        help="extract only: vintage tag of the points assets")
    parser.add_argument("--out-glob", default=None,
                        help="extract only: vintage tag for the output extracts")
    parser.add_argument("--min-years", type=int, default=3,
                        help="rasters only: minimum irrigated years mask")
    parser.add_argument("--no-freq", action="store_true",
                        help="rasters only: skip the frequency export")
    parser.add_argument("--execute", action="store_true",
                        help="start the EE tasks (default is a dry run)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    dry = not args.execute

    if args.stage == "extract":
        if not (args.years and args.points_glob and args.out_glob):
            parser.error("extract requires --years, --points-glob and --out-glob")
        run_extract(cfg, args.years, args.points_glob, args.out_glob,
                    states=args.states, dry_run=dry, config_path=args.config)
    elif args.stage == "classify":
        run_classify(cfg, states=args.states, years=args.years, dry_run=dry,
                     config_path=args.config)
    elif args.stage == "postprocess":
        run_postprocess(cfg, states=args.states, years=args.years, dry_run=dry,
                        config_path=args.config)
    elif args.stage == "rasters":
        if not args.years:
            parser.error("rasters requires --years")
        run_rasters(cfg, args.years, states=args.states, min_years=args.min_years,
                    export_freq=not args.no_freq, dry_run=dry,
                    config_path=args.config)


if __name__ == "__main__":
    main()
