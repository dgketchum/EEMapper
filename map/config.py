"""Run configuration for IrrMapper.

Loads a TOML run-config into typed dataclasses, validates it, and produces
resolved run manifests for provenance. The canonical config reproducing the
current production flow is configs/irrmapper_v1_2.toml; a run should never
require editing module constants again.
"""

import dataclasses
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

EXPRESSION_VARIABLES = ("IRR", "SUM", "NDVI", "SLOPE", "PIVOT")


@dataclass
class ProductConfig:
    """Public product identity: the version stamped into run manifests.

    public_collection is the Earth Engine catalog ID
    (e.g. UMT/Climate/IrrMapper_RF/v1_2), distinct from the private
    working collections in [paths].
    """

    name: str
    version: str
    public_collection: str


@dataclass
class ProjectConfig:
    ee_project: str
    user_root: str
    gcs_bucket: str


@dataclass
class PathsConfig:
    data_root: str
    boundaries: str
    points: str
    training_tables: str
    raw_collection: str
    comp_collection: str
    boolean_collection: str


@dataclass
class RunSection:
    states: list
    years: list
    vintage_glob: str
    southern_states: list


@dataclass
class ModelConfig:
    kind: str = "smileRandomForest"
    number_of_trees: int = 150
    min_leaf_population: int = 1
    bag_fraction: float = 0.5


@dataclass
class ExportConfig:
    scale: int = 30
    max_pixels: float = 1e13


@dataclass
class ThresholdRule:
    """Post-processing SUM threshold.

    type 'fixed': t = value.
    type 'ramp': t = before if year < switch_year, else
    max(horizon - year - 1, 0) — the ramp that zeroes the threshold as the
    year approaches the end of the classified record.
    """

    type: str
    value: int = None
    before: int = None
    switch_year: int = None
    horizon: int = None

    def for_year(self, year):
        if self.type == "fixed":
            return self.value
        t = self.before if year < self.switch_year else self.horizon - year - 1
        return max(t, 0)


@dataclass
class ExpressionStep:
    """One sequential .expression() application in post-processing.

    expression is the verbatim template ({t} = threshold); variables lists
    the bindings the original code passed (not always the set the template
    uses — fidelity over tidiness). min_year skips the step for years
    before it.
    """

    expression: str
    variables: list
    min_year: int = None


@dataclass
class StateRules:
    copy_only: bool = False
    threshold: ThresholdRule = None
    pivot_assets: list = field(default_factory=list)
    steps: list = field(default_factory=list)

    def pivot_asset_for_year(self, year):
        for entry in self.pivot_assets:
            before = entry.get("before")
            if before is None or year < before:
                return entry["asset"]
        return None


@dataclass
class RunConfig:
    product: ProductConfig
    project: ProjectConfig
    paths: PathsConfig
    run: RunSection
    model: ModelConfig
    export: ExportConfig
    postproc: dict


def _build(cls, table, where):
    fields = {f.name for f in dataclasses.fields(cls)}
    unknown = set(table) - fields
    if unknown:
        raise ValueError("unknown key(s) {} in [{}]".format(sorted(unknown), where))
    try:
        return cls(**table)
    except TypeError as e:
        raise ValueError("invalid [{}] section: {}".format(where, e))


def _build_state_rules(state, table):
    where = "postproc.{}".format(state)
    if table.get("copy_only"):
        extra = set(table) - {"copy_only"}
        if extra:
            raise ValueError(
                "copy_only state [{}] has extra key(s) {}".format(where, sorted(extra))
            )
        return StateRules(copy_only=True)

    threshold = _build(ThresholdRule, table.get("threshold", {}), where + ".threshold")
    if threshold.type not in ("fixed", "ramp"):
        raise ValueError("[{}] threshold type must be fixed or ramp".format(where))
    if threshold.type == "fixed" and threshold.value is None:
        raise ValueError("[{}] fixed threshold requires value".format(where))
    if threshold.type == "ramp" and None in (
        threshold.before,
        threshold.switch_year,
        threshold.horizon,
    ):
        raise ValueError(
            "[{}] ramp threshold requires before, switch_year, horizon".format(where)
        )

    steps = [
        _build(ExpressionStep, s, where + ".steps") for s in table.get("steps", [])
    ]
    if not steps:
        raise ValueError("[{}] requires at least one expression step".format(where))
    for step in steps:
        bad = set(step.variables) - set(EXPRESSION_VARIABLES)
        if bad:
            raise ValueError(
                "[{}] unknown expression variable(s) {}".format(where, sorted(bad))
            )

    pivots = table.get("pivot_assets", [])
    for entry in pivots:
        if "asset" not in entry:
            raise ValueError("[{}] pivot_assets entries require asset".format(where))
    if not pivots:
        raise ValueError(
            "[{}] requires pivot_assets (the pivot band is always built)".format(where)
        )

    return StateRules(threshold=threshold, pivot_assets=pivots, steps=steps)


def load_config(path):
    with open(path, "rb") as fp:
        raw = tomllib.load(fp)

    sections = {"product", "project", "paths", "run", "model", "export", "postproc"}
    unknown = set(raw) - sections
    if unknown:
        raise ValueError("unknown top-level section(s): {}".format(sorted(unknown)))
    missing = sections - set(raw)
    if missing:
        raise ValueError("missing section(s): {}".format(sorted(missing)))

    cfg = RunConfig(
        product=_build(ProductConfig, raw["product"], "product"),
        project=_build(ProjectConfig, raw["project"], "project"),
        paths=_build(PathsConfig, raw["paths"], "paths"),
        run=_build(RunSection, raw["run"], "run"),
        model=_build(ModelConfig, raw["model"], "model"),
        export=_build(ExportConfig, raw["export"], "export"),
        postproc={s: _build_state_rules(s, t) for s, t in raw["postproc"].items()},
    )

    unruled = set(cfg.run.states) - set(cfg.postproc)
    if unruled:
        raise ValueError(
            "run.states without postproc rules: {}".format(sorted(unruled))
        )
    return cfg


def _git_sha():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or None
    except OSError:
        return None


def resolved_manifest(cfg, config_path=None):
    """Fully-resolved run provenance: config values + code + library versions."""
    try:
        import ee

        ee_version = ee.__version__
    except ImportError:
        ee_version = None
    return {
        "config": dataclasses.asdict(cfg),
        "config_file": os.path.abspath(config_path) if config_path else None,
        "git_sha": _git_sha(),
        "earthengine_api_version": ee_version,
        "created": datetime.now(timezone.utc).isoformat(),
    }
