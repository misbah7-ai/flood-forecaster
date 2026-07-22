"""
Out-of-band snapshot of every live input the app needs.

WHY THIS EXISTS
---------------
Before this module, a user opening the app WAS the trigger for the Open-Meteo
calls. Any bad luck in that instant (rate limit, timeout, shared egress IP)
became the user's problem, and app.py answered it with st.stop() -- a dead page.

Now the fetching is decoupled from the serving:

    .github/workflows/refresh-snapshot.yml   (hourly, + manual dispatch)
        -> python snapshot.py --refresh
        -> writes data/snapshot.json, committed back to the repo
        -> app.py reads that file and NEVER calls Open-Meteo on a user's behalf

If a scheduled refresh fails, nobody is watching: it retries next hour and the
app keeps serving the last good snapshot with an honest "as of <timestamp>"
label. The user's page load is no longer on the failure path.

WHAT IS STORED
--------------
The RAW inputs (weather / ensemble / GloFAS / river), not the scored result.
That is deliberate: app.py re-scores locally on every interaction (the what-if
storm builder, the rain-scale slider, the Model A/B toggle, the strictness
slider). Storing only the scored output would freeze all of that. Scoring is
local and fast; fetching is the fragile part, so only fetching is moved.

Per-source merge: each source is refreshed independently and a failure keeps the
PREVIOUS good value for that source alone. A GloFAS outage never costs you the
weather.

Run:
    venv\\Scripts\\python.exe snapshot.py --refresh    # fetch + write (CI does this)
    venv\\Scripts\\python.exe snapshot.py              # show what's in the snapshot
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import live_data as ld

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SNAPSHOT_FILE = DATA / "snapshot.json"
SCHEMA = 1

# Ensemble systems offered by the app's dropdown. All three are refreshed by the
# scheduled job so that switching systems in the UI costs zero API calls.
SNAPSHOT_MODELS = ("gfs_seamless", "icon_seamless", "ecmwf_ifs025")

# How old a snapshot may get before the app stops trusting the scheduler and
# opportunistically tries a live fetch itself (self-heal if CI is broken).
# GEFS/GFS publish every 6 h, so a healthy hourly job refreshes the *values*
# roughly 4x/day; 12 h without any change means something is actually wrong.
SELF_HEAL_AFTER_SECONDS = 12 * 3600

RIVER_HISTORY_DAYS = 35


# ---------------------------------------------------------------------------
# JSON helpers -- compact, deterministic, git-diff friendly
# ---------------------------------------------------------------------------
def _round(x, nd: int):
    """Round floats (incl. inside lists) for a compact, stable on-disk payload.
    Values are physical quantities read to 1-2 dp in the UI, so this loses
    nothing the app can display -- and it keeps the committed diffs small."""
    if isinstance(x, (list, tuple)):
        return [_round(v, nd) for v in x]
    if x is None:
        return None
    if isinstance(x, (float, np.floating)):
        return None if (np.isnan(x) or np.isinf(x)) else round(float(x), nd)
    if isinstance(x, (int, np.integer)):
        return int(x)
    return x


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _age_seconds(iso: str | None) -> float | None:
    if not iso:
        return None
    try:
        ts = datetime.fromisoformat(iso)
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - ts).total_seconds())


def _content_hash(sources: dict) -> str:
    """Hash of the DATA only, ignoring every timestamp. The refresh job uses this
    to avoid rewriting (and committing) a file whose weather values did not
    change -- Open-Meteo serves the same model run for hours at a time."""
    payload = {k: v.get("data") for k, v in sorted(sources.items())}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Per-source (de)serialisation
# ---------------------------------------------------------------------------
def _pack_forecast(df: pd.DataFrame) -> dict:
    d = df.sort_values("date").reset_index(drop=True)
    return {
        "date": [pd.Timestamp(x).date().isoformat() for x in d["date"]],
        "precip_mm": _round(d["precip_mm"].tolist(), 2),
        "tmax": _round(d["tmax"].tolist(), 1),
        "tmin": _round(d["tmin"].tolist(), 1),
        "is_observed": [bool(v) for v in d["is_observed"]],
    }


def _unpack_forecast(data: dict, fetched_at: str | None) -> pd.DataFrame:
    df = pd.DataFrame({
        "date": pd.to_datetime(data["date"]),
        "precip_mm": data["precip_mm"],
        "tmax": data["tmax"],
        "tmin": data["tmin"],
        "is_observed": data["is_observed"],
    })
    df[["precip_mm", "tmax", "tmin"]] = df[["precip_mm", "tmax", "tmin"]].ffill().bfill()
    df.attrs["fetched_at"] = fetched_at
    return df


def _pack_ensemble(ens: dict) -> dict:
    """Store the DAILY-aggregated member frame, not the raw hourly payload.

    _ensemble_frame() collapses ~192 hourly steps x ~31 members to 8 daily rows
    per member anyway, so the hourly detail is never used. Packing the aggregate
    is ~20x smaller on disk, which is what keeps an hourly commit cadence sane."""
    md = ens["member_daily"].copy()
    md["date"] = pd.to_datetime(md["date"])
    piv_p = md.pivot_table(index="date", columns="member", values="precip_mm")
    piv_tx = md.pivot_table(index="date", columns="member", values="tmax")
    piv_tn = md.pivot_table(index="date", columns="member", values="tmin")
    members = sorted(piv_p.columns.astype(str))
    return {
        "model": ens.get("model"),
        "n_members": int(ens.get("n_members") or len(members)),
        "date": [pd.Timestamp(x).date().isoformat() for x in piv_p.index],
        "members": members,
        "precip_mm": {m: _round(piv_p[m].tolist(), 2) for m in members},
        "tmax": {m: _round(piv_tx[m].tolist(), 1) for m in members},
        "tmin": {m: _round(piv_tn[m].tolist(), 1) for m in members},
    }


def _unpack_ensemble(data: dict, fetched_at: str | None) -> dict:
    dates = pd.to_datetime(data["date"])
    frames = []
    for m in data["members"]:
        frames.append(pd.DataFrame({
            "date": dates,
            "precip_mm": data["precip_mm"][m],
            "tmax": data["tmax"][m],
            "tmin": data["tmin"][m],
            "member": m,
        }))
    member_daily = pd.concat(frames, ignore_index=True)
    return {"member_daily": member_daily, "n_members": int(data["n_members"]),
            "model": data.get("model"), "fetched_at": fetched_at}


def _pack_glofas(g: dict) -> dict:
    return {
        "dates": [pd.Timestamp(d).date().isoformat() for d in g["dates"]],
        "median_cfs": _round(g["median_cfs"], 1),
        "p25_cfs": _round(g["p25_cfs"], 1),
        "p75_cfs": _round(g["p75_cfs"], 1),
        "offset": list(g["offset"]),
        "coords": list(g["coords"]),
        "indicative": bool(g["indicative"]),
        "peak_cfs": _round(g["peak_cfs"], 1),
    }


def _unpack_glofas(data: dict, fetched_at: str | None) -> dict:
    g = dict(data)
    g["dates"] = [pd.Timestamp(d).date() for d in data["dates"]]
    g["offset"] = tuple(data["offset"])
    g["coords"] = tuple(data["coords"])
    return g


def _pack_river_history(df: pd.DataFrame) -> dict:
    d = df.sort_values("date").reset_index(drop=True)
    return {"date": [pd.Timestamp(x).date().isoformat() for x in d["date"]],
            "q_cfs": _round(d["q_cfs"].tolist(), 1)}


def _unpack_river_history(data: dict, fetched_at: str | None) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime(data["date"]), "q_cfs": data["q_cfs"]})


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------
def load() -> dict | None:
    """The raw snapshot dict, or None if it has never been written / is corrupt."""
    try:
        snap = json.loads(SNAPSHOT_FILE.read_text())
    except (OSError, ValueError):
        return None
    return snap if snap.get("schema") == SCHEMA else None


def _get(snap: dict | None, key: str) -> tuple[dict | None, str | None]:
    """(data, fetched_at) for one source, or (None, None)."""
    if not snap:
        return None, None
    src = (snap.get("sources") or {}).get(key)
    if not src or src.get("data") is None:
        return None, None
    return src["data"], src.get("fetched_at")


def _meta(source: str, fetched_at: str | None, note: str = "") -> dict:
    """Provenance for one value the app is about to render. `source` is one of
    snapshot | live | stale-cache | unavailable -- app.py turns this into the
    freshness chip, so the user always knows how old the number is."""
    return {"source": source, "fetched_at": fetched_at,
            "age_seconds": _age_seconds(fetched_at), "note": note}


def _snapshot_is_serveable(fetched_at: str | None) -> bool:
    """Serve the snapshot without touching the network unless it is old enough
    that the scheduler itself looks broken."""
    age = _age_seconds(fetched_at)
    return age is not None and age < SELF_HEAL_AFTER_SECONDS


# --- the five accessors app.py uses -----------------------------------------
def weather(lat: float, lon: float, allow_network: bool = True):
    """(DataFrame, meta). Snapshot first; live only as self-heal; stale snapshot
    as the last resort. Raises LiveDataError only when there is genuinely
    nothing -- no snapshot, no cache, and no network."""
    snap = load()
    data, ts = _get(snap, "forecast")
    if data is not None and _snapshot_is_serveable(ts):
        return _unpack_forecast(data, ts), _meta("snapshot", ts)

    if allow_network:
        try:
            df = ld.fetch_forecast_weather(lat, lon)
            return df, _meta("stale-cache" if df.attrs.get("stale") else "live",
                             df.attrs.get("fetched_at"))
        except ld.LiveDataError as e:
            if data is None:
                raise
            return _unpack_forecast(data, ts), _meta("snapshot", ts, str(e))

    if data is None:
        raise ld.LiveDataError("no snapshot and network disabled")
    return _unpack_forecast(data, ts), _meta("snapshot", ts)


def ensemble(lat: float, lon: float, model: str, allow_network: bool = True):
    snap = load()
    data, ts = _get(snap, f"ensemble:{model}")
    if data is not None and _snapshot_is_serveable(ts):
        return _unpack_ensemble(data, ts), _meta("snapshot", ts)
    if allow_network:
        got = ld.fetch_ensemble_weather(lat, lon, model)
        if got is not None:
            return got, _meta("stale-cache" if got.get("stale") else "live",
                              got.get("fetched_at"))
    if data is not None:
        return _unpack_ensemble(data, ts), _meta("snapshot", ts)
    return None, _meta("unavailable", None)


def glofas(lat: float, lon: float, allow_network: bool = True):
    snap = load()
    data, ts = _get(snap, "glofas")
    if data is not None and _snapshot_is_serveable(ts):
        return _unpack_glofas(data, ts), _meta("snapshot", ts)
    if allow_network:
        got = ld.fetch_glofas(lat, lon)
        if got is not None:
            return got, _meta("stale-cache" if got.get("stale") else "live",
                              got.get("fetched_at"))
    if data is not None:
        return _unpack_glofas(data, ts), _meta("snapshot", ts)
    return None, _meta("unavailable", None)


def river(gauge_id: str, allow_network: bool = True):
    """USGS latest reading. Kept on a SHORTER leash than the weather: it is a
    separate host on a separate quota (it was never the thing rate-limiting us),
    and 'current river level' is the one number where staleness really shows."""
    snap = load()
    data, ts = _get(snap, "river")
    if allow_network:
        got = ld.fetch_current_river(gauge_id)
        if got is not None:
            return got, _meta("live", _now_iso())
    if data is not None:
        return data, _meta("snapshot", ts)
    return None, _meta("unavailable", None)


def river_history(gauge_id: str, days: int = RIVER_HISTORY_DAYS, allow_network: bool = True):
    snap = load()
    data, ts = _get(snap, "river_history")
    if data is not None and _snapshot_is_serveable(ts):
        return _unpack_river_history(data, ts), _meta("snapshot", ts)
    if allow_network:
        end = datetime.now(ld.BASIN_TZ).date()
        got = ld.fetch_usgs_daily(gauge_id, (end - timedelta(days=days)).isoformat(),
                                  end.isoformat())
        if got is not None and len(got):
            return got, _meta("live", _now_iso())
    if data is not None:
        return _unpack_river_history(data, ts), _meta("snapshot", ts)
    return None, _meta("unavailable", None)


# ---------------------------------------------------------------------------
# Write (the scheduled job)
# ---------------------------------------------------------------------------
def refresh(lat: float = 41.4133, lon: float = -78.1972,
            gauge_id: str | None = None,
            models: tuple[str, ...] = SNAPSHOT_MODELS,
            write: bool = True) -> dict:
    """Fetch every live source and merge into the snapshot.

    Each source is fetched and merged INDEPENDENTLY: whatever succeeds is
    updated, whatever fails leaves the previous good value untouched. The return
    value reports per-source status so CI can log (and alert on) partial
    failures without ever writing a half-empty snapshot."""
    if gauge_id is None:
        gauge_id = "01543000"
        try:  # prefer the id the trained bundle was actually built for
            gauge_id = ld.load_bundles()["A"]["basin"]["id"]
        except Exception:
            pass

    prev = load() or {}
    sources: dict = dict(prev.get("sources") or {})
    status: dict[str, str] = {}

    def _store(key: str, data, ok: bool, err: str = ""):
        if ok and data is not None:
            sources[key] = {"fetched_at": _now_iso(), "data": data}
            status[key] = "ok"
        else:
            kept = key in sources
            status[key] = f"failed ({err or 'no data'}); " + \
                          (f"kept previous from {sources[key]['fetched_at']}" if kept else "no previous value")

    # 1. deterministic forecast (-30 .. +7)  -- the one the model cannot run without
    try:
        _store("forecast", _pack_forecast(ld.fetch_forecast_weather(lat, lon, use_cache=False)), True)
    except Exception as e:
        _store("forecast", None, False, str(e)[:160])

    # 2. ensemble systems (the heaviest calls -- 3/hour from CI, vs 1 per visitor before)
    for m in models:
        try:
            ens = ld.fetch_ensemble_weather(lat, lon, m, use_cache=False)
            _store(f"ensemble:{m}", _pack_ensemble(ens) if ens else None, ens is not None)
        except Exception as e:
            _store(f"ensemble:{m}", None, False, str(e)[:160])

    # 3. GloFAS operational benchmark
    try:
        g = ld.fetch_glofas(lat, lon)
        _store("glofas", _pack_glofas(g) if g else None, g is not None)
    except Exception as e:
        _store("glofas", None, False, str(e)[:160])

    # 4/5. USGS (separate host, separate quota -- unaffected by Open-Meteo limits)
    try:
        r = ld.fetch_current_river(gauge_id)
        _store("river", r, r is not None)
    except Exception as e:
        _store("river", None, False, str(e)[:160])

    try:
        end = datetime.now(ld.BASIN_TZ).date()
        rh = ld.fetch_usgs_daily(gauge_id, (end - timedelta(days=RIVER_HISTORY_DAYS)).isoformat(),
                                 end.isoformat())
        _store("river_history", _pack_river_history(rh) if rh is not None and len(rh) else None,
               rh is not None and len(rh) > 0)
    except Exception as e:
        _store("river_history", None, False, str(e)[:160])

    if not sources:
        raise RuntimeError("refresh produced nothing and there was no previous snapshot")

    new_hash = _content_hash(sources)
    changed = new_hash != prev.get("content_hash")
    snap = {"schema": SCHEMA, "generated_at": _now_iso(), "content_hash": new_hash,
            "lat": lat, "lon": lon, "gauge_id": gauge_id, "sources": sources}

    # Only rewrite when the DATA moved. Open-Meteo serves the same model run for
    # hours, so an hourly job would otherwise commit an identical-but-for-the-
    # timestamp file 24x/day and bloat the repo for nothing.
    if write and changed:
        SNAPSHOT_FILE.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT_FILE.write_text(json.dumps(snap, separators=(",", ":"), sort_keys=True))

    return {"changed": changed, "status": status, "content_hash": new_hash,
            "n_sources": len(sources),
            "bytes": SNAPSHOT_FILE.stat().st_size if SNAPSHOT_FILE.exists() else 0}


# ---------------------------------------------------------------------------
def describe() -> str:
    snap = load()
    if not snap:
        return f"No snapshot at {SNAPSHOT_FILE} (run: python snapshot.py --refresh)"
    out = [f"snapshot {SNAPSHOT_FILE.name}  generated_at={snap['generated_at']}  "
           f"hash={snap.get('content_hash')}  ({SNAPSHOT_FILE.stat().st_size/1024:.0f} KB)"]
    for k, v in sorted((snap.get("sources") or {}).items()):
        age = _age_seconds(v.get("fetched_at"))
        out.append(f"  {k:<26} fetched {v.get('fetched_at')}  "
                   f"({age/3600:.1f} h ago)" if age is not None else f"  {k:<26} fetched ?")
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true", help="fetch every source and write the snapshot")
    ap.add_argument("--lat", type=float, default=41.4133)
    ap.add_argument("--lon", type=float, default=-78.1972)
    args = ap.parse_args()

    if args.refresh:
        rep = refresh(lat=args.lat, lon=args.lon)
        print(f"content_hash={rep['content_hash']}  changed={rep['changed']}  "
              f"sources={rep['n_sources']}  size={rep['bytes']/1024:.0f} KB")
        for k, v in sorted(rep["status"].items()):
            print(f"  {k:<26} {v}")
        # A refresh that saved NOTHING at all is the only CI-failing case; partial
        # failures are normal and are absorbed by the per-source merge above.
        if all(v != "ok" for v in rep["status"].values()) and not SNAPSHOT_FILE.exists():
            sys.exit(1)
    else:
        print(describe())
