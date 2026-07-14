import datetime
from zoneinfo import ZoneInfo
import requests
import json
import re
import os
import zipfile
import io
import glob
import shutil
import geopandas
from tempfile import TemporaryDirectory
import logging

logging.basicConfig(level=logging.INFO)


def store_version(key: str, version: str):
    logging.info(f"{key} version: {version}")
    # "0" is prepended in filename so that this file appears first in Github directory listing
    try:
        with open('waypoints/0versions.json', 'r') as f:
            version_dict = json.load(f)
    except BaseException:
        version_dict = {}
    version_dict[key] = version
    version_dict = dict(sorted(version_dict.items()))
    with open('waypoints/0versions.json', 'w', encoding='UTF-8') as f:
        json.dump(version_dict, f, indent=4)


os.makedirs("waypoints", exist_ok=True)

for csdi_dataset in [
    # 巴士路線
    # https://portal.csdi.gov.hk/geoportal/?lang=zh-hk&datasetId=td_rcd_1638844988873_41214
    {"name": "bus", "id": "td_rcd_1638844988873_41214"},
    # 專線小巴路線
    # https://portal.csdi.gov.hk/geoportal/?lang=zh-hk&datasetId=td_rcd_1697082463580_57453
    {"name": "gmb", "id": "td_rcd_1697082463580_57453"}
]:
    logging.info("csdi_dataset=" + json.dumps(csdi_dataset))
    logging.info("Fetching metadata")
    r = requests.get(
        "https://portal.csdi.gov.hk/geoportal/rest/metadata/item/" +
        csdi_dataset["id"])
    src_id = json.loads(r.content)['_source']['fileid'].replace('-', '')

    logging.info("Fetching FGDB")
    r = requests.get(
        "https://static.csdi.gov.hk/csdi-webpage/download/" + src_id + "/fgdb")
    z = zipfile.ZipFile(io.BytesIO(r.content))
    version = min([f.date_time for f in z.infolist()])
    version = datetime.datetime(
        *version, tzinfo=ZoneInfo("Asia/Hong_Kong"))
    store_version(csdi_dataset["name"], version.isoformat())
    gdb_name = next(s[0:s.index('/')]
                    for s in z.namelist() if s != "__MACOSX")

    with TemporaryDirectory() as tmpdir:
        logging.info("Extracting data")
        z.extractall(tmpdir)
        gdb_path = os.path.join(tmpdir, gdb_name)
        logging.info("Reading data (1)")
        gdf = geopandas.read_file(gdb_path, encoding='utf-8')
        logging.info("Transforming data")
        gdf.to_crs(epsg=4326, inplace=True)
        logging.info("Reading data (2)")
        data = gdf.to_geo_dict(drop_id=True)

    logging.info("Storing data")
    for feature in data["features"]:
        properties = feature["properties"]
        with open("waypoints/" + str(properties["ROUTE_ID"]) + "-" + ("O" if properties["ROUTE_SEQ"] == 1 else "I") + ".json", "w", encoding='utf-8') as f:
            f.write(
                re.sub(
                    r"([0-9]+\.[0-9]{5})[0-9]+",
                    r"\1",
                    json.dumps({
                        "features": [feature],
                        "type": "FeatureCollection"
                    },
                        ensure_ascii=False,
                        separators=(",", ":")
                    )
                )
            )


logging.info("Copying static data")
for file in glob.glob(r'./mtr/*.json'):
    shutil.copy(file, "waypoints")
for file in glob.glob(r'./lrt/*.json'):
    shutil.copy(file, "waypoints")
for file in glob.glob(r'./ferry/*.json'):
    shutil.copy(file, "waypoints")


# ---------------------------------------------------------------------------
# Refine franchised-bus lines by map-matching the official GTFS stop sequences
# to OpenStreetMap with pfaedle (https://github.com/ad-freiburg/pfaedle).
# CSDI above stays the base for EVERY route; a refined line only replaces a
# file when all of the route's stops lie within GATE_M metres of it, so a bad
# match can never ship (a stop stranded off its line was the failure mode of
# the previous pfaedle attempt). GMB is excluded: minibuses do not hold to a
# fixed line and their stop data is too coarse to gate against.
# pfaedle.cfg is the stock config plus a [bus, coach] block for Hong Kong
# tagging (bus=private @ franchised, highway=service) and a station-move
# penalty override — see the "HONG KONG" comments inside the file.

import csv
import shlex
import subprocess
import urllib.request
from collections import defaultdict

import numpy

GTFS_URL = "https://static.data.gov.hk/td/pt-headway-tc/gtfs.zip"
OSM_URL = "https://download.geofabrik.de/asia/china/hong-kong-latest.osm.pbf"
PFAEDLE_IMAGE = "ghcr.io/ad-freiburg/pfaedle:latest"
GATE_M = 100.0


def refine_franchised_bus_lines():
    with TemporaryDirectory(ignore_cleanup_errors=True) as work:
        for url, path in ((OSM_URL, f"{work}/hong-kong.osm.pbf"), (GTFS_URL, f"{work}/gtfs.zip")):
            logging.info(f"Fetching {url}")
            with urllib.request.urlopen(url, timeout=300) as r, open(path, "wb") as f:
                shutil.copyfileobj(r, f)
        with zipfile.ZipFile(f"{work}/gtfs.zip") as z:
            z.extractall(f"{work}/src")

        def rd(name):
            with open(f"{work}/src/{name}", encoding="utf-8-sig") as f:
                return list(csv.DictReader(f))

        routes = {r["route_id"]: r for r in rd("routes.txt")}
        kept = {}  # (route_id, seq) -> one representative trip_id
        for t in rd("trips.txt"):
            rid = t["route_id"]
            r = routes.get(rid, {})
            if r.get("route_type") != "3" or r.get("agency_id") == "GMB":
                continue
            # trip_id is "<route_id>-<seq>-..."; seq 1 = outbound, 2 = inbound
            seq = t["trip_id"][len(rid) + 1:].split("-")[0] if t["trip_id"].startswith(rid + "-") else ""
            if seq in ("1", "2"):
                kept.setdefault((rid, seq), t["trip_id"])
        rep = set(kept.values())
        stops = {s["stop_id"]: (float(s["stop_lon"]), float(s["stop_lat"])) for s in rd("stops.txt")}
        seqs = defaultdict(list)
        for s in rd("stop_times.txt"):
            if s["trip_id"] in rep:
                seqs[s["trip_id"]].append((int(s["stop_sequence"]), s["stop_id"]))

        os.makedirs(f"{work}/gtfs")

        def wr(name, header, rows):
            with open(f"{work}/gtfs/{name}", "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(header)
                w.writerows(rows)
        wr("agency.txt", ["agency_id", "agency_name", "agency_url", "agency_timezone"],
           [["HK", "HK", "https://hkbus.app", "Asia/Hong_Kong"]])
        wr("routes.txt", ["route_id", "agency_id", "route_short_name", "route_type"],
           [[rid, "HK", routes[rid].get("route_short_name", rid), "3"] for rid in {r for r, _ in kept}])
        wr("trips.txt", ["route_id", "service_id", "trip_id"],
           [[rid, "D", tid] for (rid, _), tid in kept.items()])
        wr("stop_times.txt", ["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"],
           [[tid, "", "", sid, i] for tid in rep for i, sid in sorted(seqs[tid])])
        wr("stops.txt", ["stop_id", "stop_name", "stop_lat", "stop_lon"],
           [[sid, sid, xy[1], xy[0]] for sid, xy in stops.items()])
        wr("calendar.txt", ["service_id", "monday", "tuesday", "wednesday", "thursday", "friday",
                            "saturday", "sunday", "start_date", "end_date"],
           [["D", 1, 1, 1, 1, 1, 1, 1, "20260101", "20261231"]])

        shutil.copy("pfaedle.cfg", f"{work}/pfaedle.cfg")
        cmd = os.environ.get("PFAEDLE_CMD")  # set to a native binary to skip Docker
        base = shlex.split(cmd) if cmd else ["docker", "run", "--rm",
                                        "--user", f"{os.getuid()}:{os.getgid()}",
                                        "-v", f"{work}:/data", PFAEDLE_IMAGE]
        prefix = "/data" if not cmd else work
        logging.info("Running pfaedle")
        subprocess.run(base + ["-c", f"{prefix}/pfaedle.cfg", "-x", f"{prefix}/hong-kong.osm.pbf",
                               "-o", f"{prefix}/out", f"{prefix}/gtfs"], check=True)

        shapes = defaultdict(list)
        with open(f"{work}/out/shapes.txt", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                shapes[row["shape_id"]].append(
                    (int(row["shape_pt_sequence"]), float(row["shape_pt_lon"]), float(row["shape_pt_lat"])))
        trip_shape = {}
        with open(f"{work}/out/trips.txt", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("shape_id"):
                    trip_shape[row["trip_id"]] = row["shape_id"]

    def max_stop_distance(pts, line):
        # metres from each stop to the polyline, equirectangular around 22.35°N
        m = numpy.column_stack([line[:, 0] * 102730.0, line[:, 1] * 110852.0])
        p = numpy.column_stack([pts[:, 0] * 102730.0, pts[:, 1] * 110852.0])
        a, b = m[:-1], m[1:]
        ab = b - a
        ab2 = (ab ** 2).sum(1)
        ab2[ab2 == 0] = 1e-9
        worst = 0.0
        for q in p:
            t = numpy.clip(((q - a) * ab).sum(1) / ab2, 0, 1)
            worst = max(worst, float(numpy.hypot(*(q - (a + t[:, None] * ab)).T).min()))
        return worst

    refined = gated = unmatched = 0
    for (rid, seq), tid in kept.items():
        sid = trip_shape.get(tid)
        if not sid or sid not in shapes:
            unmatched += 1
            continue
        coords = [[lon, lat] for _, lon, lat in sorted(shapes[sid])]
        pts = numpy.array([stops[x] for _, x in sorted(seqs[tid]) if x in stops], dtype=float)
        if len(pts) < 2 or len(coords) < 2 or max_stop_distance(pts, numpy.array(coords, dtype=float)) > GATE_M:
            gated += 1  # keep the CSDI line for this route
            continue
        feature = {"type": "Feature",
                   "properties": {"ROUTE_ID": rid, "ROUTE_SEQ": int(seq), "SOURCE": "pfaedle-osm"},
                   "geometry": {"type": "LineString", "coordinates": coords}}
        with open(f"waypoints/{rid}-{'O' if seq == '1' else 'I'}.json", "w", encoding="utf-8") as f:
            f.write(re.sub(r"([0-9]+\.[0-9]{5})[0-9]+", r"\1",
                           json.dumps({"features": [feature], "type": "FeatureCollection"},
                                      ensure_ascii=False, separators=(",", ":"))))
        refined += 1
    logging.info(f"pfaedle refinement: {refined} replaced, {gated} kept CSDI (gate), {unmatched} unmatched")
    store_version("bus-refined", datetime.datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat())


# A refinement failure fails the whole crawl: the deploy step is skipped and
# gh-pages keeps yesterday's complete data. Publishing CSDI-only would silently
# drop every refined line and all pfaedle-only route coverage.
refine_franchised_bus_lines()
