import datetime
from zoneinfo import ZoneInfo
import requests
import json
import re
import os
import csv
import math
import zipfile
import io
import glob
import shutil
import geopandas
from collections import defaultdict
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


# --- Outbound (O) / inbound (I) direction resolution --------------------
# CSDI writes ROUTE_SEQ (1/2), which does NOT track the operators' outbound/inbound
# convention, so labelling by ROUTE_SEQ swaps some routes' -O/-I files (issue #14).
# Resolve direction by IDs instead of terminal names (names vary too much between
# datasets to match reliably):
#   * CSDI ST_STOP_ID == GTFS stop_id (same registry) -> the feature's start coord
#   * hkbus routeFareList gtfsId == CSDI ROUTE_ID (same registry) -> that route's
#     directions, each with its bound (O/I) and first stop
# GTFS stop-ids and hkbus's operator stop-ids are different registries with no shared
# key, so the O-vs-I pick is by nearest start-stop *coordinate* (metres,
# not names).

GTFS_STOPS_URL = "https://static.data.gov.hk/td/pt-headway-en/gtfs.zip"
ROUTE_FARE_LIST_URL = "https://data.hkbus.app/routeFareList.json"
_CO_ALIAS = {
    "LWB": "kmb",
    "KMB": "kmb",
    "CTB": "ctb",
    "NLB": "nlb",
    "GMB": "gmb"}
_MAX_MATCH_M = 500  # a start stop farther than this is treated as unresolved


def _haversine(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dlmb = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + \
        math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(a))


def fetch_gtfs_stops():
    """GTFS stop_id -> (lat, lon). CSDI ST_STOP_ID values are GTFS stop_ids."""
    stops = {}
    try:
        z = zipfile.ZipFile(io.BytesIO(
            requests.get(GTFS_STOPS_URL, timeout=120).content))
        with z.open("stops.txt") as f:
            for row in csv.DictReader(io.TextIOWrapper(f, "utf-8-sig")):
                stops[row["stop_id"]] = (
                    float(row["stop_lat"]), float(row["stop_lon"]))
        logging.info(f"Fetched {len(stops)} GTFS stops")
    except Exception as e:
        logging.warning(
            f"GTFS stops fetch failed; O/I falls back to ROUTE_SEQ: {e}")
    return stops


def fetch_direction_index():
    """(ROUTE_ID, co) -> [(bound, lat, lng)] of each direction's first stop."""
    index = defaultdict(list)
    try:
        rfl = requests.get(ROUTE_FARE_LIST_URL, timeout=120).json()
        stop_list = rfl["stopList"]
        for v in rfl["routeList"].values():
            gtfs_id = v.get("gtfsId")
            if not gtfs_id:
                continue
            for co in v.get("co", []):
                bound = v.get("bound", {}).get(co)
                stops_co = v.get("stops", {}).get(co)
                loc = stop_list.get(stops_co[0], {}).get(
                    "location") if stops_co else None
                # only plain O/I; skip circular "OI"/"IO", tram "DT"/"UT",
                # cross-boundary "LMC-*"/"TKS-*" -> those fall back to
                # ROUTE_SEQ
                if bound in ("O", "I") and loc:
                    index[(gtfs_id, co)].append(
                        (bound, loc["lat"], loc["lng"]))
        logging.info(
            f"Fetched direction index for {
                len(index)}(route, co) pairs")
    except Exception as e:
        logging.warning(
            f"routeFareList fetch failed; O/I falls back to ROUTE_SEQ: {e}")
    return index


def _resolve_bound(properties, gtfs_stops, direction_index):
    """(bound, distance_m) for one feature by nearest start stop, or (None, None)."""
    coord = gtfs_stops.get(str(properties.get("ST_STOP_ID")))
    if not coord:
        return None, None
    route_id = str(properties.get("ROUTE_ID"))
    cos = [_CO_ALIAS.get(p, p.lower())
           for p in str(properties.get("COMPANY_CODE", "")).split("+")]
    best = None
    for co in cos:
        for bound, lat, lng in direction_index.get((route_id, co), []):
            d = _haversine(coord[0], coord[1], lat, lng)
            if best is None or d < best[1]:
                best = (bound, d)
    if best is None or best[1] > _MAX_MATCH_M:
        return None, None
    return best


def assign_directions(features, gtfs_stops, direction_index):
    """Assign O/I to every feature of one route, together.

    Uses the ID+coordinate resolution above; assigns a route's features jointly so
    a two-direction route always gets exactly one O and one I (never overwriting a
    direction's file). Falls back to ROUTE_SEQ for anything it can't resolve.
    """
    def seq_label(feature):
        return "O" if feature["properties"].get("ROUTE_SEQ", 1) == 1 else "I"

    def opposite(bound):
        return "I" if bound == "O" else "O"

    res = [_resolve_bound(f["properties"], gtfs_stops, direction_index)
           for f in features]

    if len(features) == 2:
        (ba, da), (bb, db) = res
        if ba and bb:
            if ba != bb:
                return [ba, bb]
            # both nearest the same terminal: trust the closer, complement the
            # other
            return [ba, opposite(ba)] if da <= db else [opposite(bb), bb]
        if ba:
            return [ba, opposite(ba)]
        if bb:
            return [opposite(bb), bb]
        first = seq_label(features[0])
        return [first, opposite(first)]

    if len(features) == 1:
        return [res[0][0] or seq_label(features[0])]

    return [(r[0] or seq_label(f)) for r, f in zip(res, features)]


os.makedirs("waypoints", exist_ok=True)
gtfs_stops = fetch_gtfs_stops()
direction_index = fetch_direction_index()

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
    features_by_route = {}
    for feature in data["features"]:
        features_by_route.setdefault(
            feature["properties"]["ROUTE_ID"], []).append(feature)

    for route_id, route_features in features_by_route.items():
        directions = assign_directions(
            route_features, gtfs_stops, direction_index)
        for feature, direction in zip(route_features, directions):
            with open("waypoints/" + str(route_id) + "-" + direction + ".json", "w", encoding='utf-8') as f:
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
