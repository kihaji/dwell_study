# Public mobility datasets for dwell (stay point) evaluation

Survey date: 2026-09-30. Every URL below was checked on that date with `curl -sIL` (HTTP status and
`Content-Length` shown where the server reports it). Small files (< 50 MB) marked "inspected" were
downloaded to a scratch directory to confirm their schema. What has been ingested is described in
[Ingestion](#ingestion) at the end.

This is PLAN step 9. Public data is for realism and scaling. Scored accuracy still comes from our
synthetic data. The few public sets that have real stop labels are called out, because they can
give a small real-world accuracy check.

Target event schema: `id, lat, lon, timestamp (UTC)`, plus optional `horizontalAccuracy` (CEP50 m)
or an error ellipse (`semiMajor, semiMinor, azimuth, probability`).

## Summary table

| Dataset | Style | Scale | Size (download) | Access / license | Stay labels? | Status |
|---|---|---|---|---|---|---|
| **GeoLife 1.3** (Microsoft) | Dense GPS, 1-5 s typical | 182 users, 17,621 trajectories, ~24.9M points, 2007-2012 | 313 MB zip | Direct download; MSR terms, cite papers | Transport-mode segments for 69 users (no stay labels) | 200 |
| **STAGA** (TU Berlin) | GPS, ~1 per 90 s effective | 1 person, 126 days, 122,808 points | 15 MB GPS CSV + 45 KB diary (+624 MB accelerometer, optional) | OSF, **CC BY 4.0**, no registration | **Yes**: per-point stop/trip + 689 diary stop intervals | 200, inspected |
| **ExtraSensory** (UCSD) | Phone location, ~1/min with gaps | 60 users, ~300k minutes | 2.3 MB locations + 1 MB labels | Free; cite paper | Self-reported context (`LOC_home`, `AT_WORK`, ...), weak | 200, inspected |
| **PoL staypoint benchmark** (arXiv 2607.19312, 2026) | *Synthetic* GPS, 1 min | 500 agents × 30 days, 21.6M rows, 65,873 staypoints | 18 MB (clean) to ~400 MB (each of 15 noisy/dropout variants) + 1 MB ground truth | OSF + GitHub; CC BY-NC-ND 4.0 per the paper | **Yes** (start/end/lat/lon) | 200, inspected |
| **Foursquare NYC/TKY (TSMC2014)** | Sparse check-ins (adtech-like) | NYC 227,428 / 1,083 users; TKY 573,703 / 2,293 users; Apr 2012-Feb 2013 | 25.5 MB zip | Direct download; research use, cite paper | Each check-in is a venue visit, with no duration | 200, inspected |
| Foursquare global (LBSN2Vec++) | Sparse check-ins | 22.8M check-ins, 114k users (raw: 90M / 2.7M) | 2.68 GB zip | figshare/4TU, **CC BY 4.0** | Venue visits | 200 |
| Massive-STEPS (2025) | Sparse check-ins, 2012-13 & 2017-18 | 15 cities, per-city CSVs | a few MB per city | Hugging Face, Apache-2.0 card (Foursquare-derived) | Venue visits | 200, sampled |
| Gowalla (SNAP) | Sparse check-ins | 6.44M check-ins, 196,591 users, 2009-2010 | 105 MB gz | Direct download; SNAP, cite | Venue visits | 200 |
| Brightkite (SNAP) | Sparse check-ins | 4.49M check-ins, 58,228 users, 2008-2010 | 60 MB gz | Direct download; SNAP, cite | Venue visits | 200 |
| **YJMob100K** (HuMob 2023) | 500 m grid × 30 min (CDR-like) | dataset1 100k users / 75 days; dataset2 25k users | 419 MB + 111 MB gz | Zenodo, **CC BY 4.0**, no registration | No | 200 |
| LYMob-4Cities (HuMob 2024 public release) | 500 m grid × 30 min | 4 anonymous Japanese cities, 6k-100k users each | cityA 431 MB, cityB 90 MB, cityC 68 MB, **cityD 31 MB** | Zenodo, **CC BY 4.0** | No | 200 |
| T-Drive sample (Microsoft) | Taxi GPS, ~177 s mean interval | 10,357 taxis, 1 week (Feb 2008), ~15M points | 9 zips, ~84 MB total | Direct download (needs a browser User-Agent); MSR terms | No (taxi stops, not human dwells) | 200 |
| Porto taxi (ECML/PKDD 2015) | Taxi GPS, 15 s, trips only | 442 taxis, 1.71M trips, 1 year | ~509 MB zip (UCI) | UCI, CC BY 4.0 | No; trip-only, no idle time between trips | 200 |
| Telecom Italia Milan grid | Aggregated CDR activity, 10 min × 235 m grid | Whole city, Nov-Dec 2013 | 62 daily files, 20.8 GB | Harvard Dataverse, ODbL | No; **no per-device data** | 202 (Dataverse) |
| NUMOSIM (2024) | *Synthetic* stay sequences only (no pings) | 200k agents, 8 weeks, Los Angeles | ~320 MB per stay-point file | OSF; license not set on OSF | Stays only | 200 |
| UCL England app data (2025) | Adtech-sourced, **already reduced to stays** (MSOA, 15 min) | 5,000 London devices, 30 days | 6.9 MB trajectories | Zenodo, CC BY 4.0 | Stays with activity label, no pings | 200 |
| MDC (Nokia/Idiap) | Phone GPS + GSM + WLAN | ~200 users, 2009-2011 | n/a | **Restricted**: signed agreement, academic only; Idiap MDC pages now 404 | Semantic place labels | Not accessible |
| D4D Côte d'Ivoire / Senegal (Orange) | Individual CDR (antenna) | millions of subscribers | n/a | **Challenge-only**, not redistributed | No | Not accessible |
| NetMob23 (Orange) | Aggregated app traffic, 100 m tiles | 20 French cities | 2.3 TB | Challenge application required; aggregate only | No | Restricted |
| SHL (Sussex-Huawei) | Phone GPS + IMU | 3 users, 7 months | preview ~7 GB (3 parts) | Free download | Mode labels incl. `Still` | 200 (landing page) |

"200" means the direct download URL (or the landing page where no direct file exists) returned HTTP
200 on 2026-09-30.

## Recommendation

Four core datasets, one per question, plus one optional extra.

1. **GeoLife 1.3: dense real GPS (realism and scaling).** It is the standard real GPS trace set,
   and trackintel and scikit-mobility have readers for it (`ti.io.read_geolife`). There are about
   25M points across 182 ids, including some very long single-id tracks, which gives a real-data
   scale ladder. It has no stay labels. The 69 labeled users have transport-mode segments, and the
   gaps between consecutive mode segments can serve as weak "stay" intervals, as DeepStay did
   (arXiv 2306.06068). Download is 313 MB.
2. **STAGA: real GPS with real stop ground truth.** It is small, CC BY 4.0, needs no registration,
   and every point carries `diary_label ∈ {stop, trip}`. `STAGA diary.csv` gives 689 stop intervals
   taken from a live mobility diary. It is the only freely downloadable real dataset I found with
   accurate stop start/end times. It is a single id, so it serves as a real-world accuracy check,
   not a scaling test. Download is 15 MB. **ExtraSensory** (3.3 MB, 60 ids, `LOC_home`/`AT_WORK`
   style labels) is a cheap second real labeled set if we want more than one id, but its labels are
   self-reported and noisy.
3. **Foursquare NYC/TKY (TSMC2014): sparse, bursty, user-driven (adtech-like).** Its event timing
   comes from user actions, with long gaps, much like adtech pings. It has 3,376 ids and 800k
   events and is 25 MB. It is small enough for dev. For scaling, use the same loader on
   **Gowalla** (6.4M) or **LBSN2Vec++** (22.8M, 2.7 GB).
4. **YJMob100K: grid/CDR-like at city scale.** It is a 500 m × 30 min grid for 100k ids over
   75 days. That is a quantized, cell-level observation process, the closest open analogue to CDR
   (per-device CDR datasets like D4D and MDC are not publicly obtainable). It is CC BY 4.0 on
   Zenodo, with no registration. For dev, use **LYMob-4Cities cityD** (31 MB), which has the same
   format.

Optional 5th: **PoL staypoint benchmark (2026).** It is synthetic, but it comes from an external
simulator, has 65,873 labeled staypoints, and has published baselines for trackintel,
T-DBSCAN, SSPE, HMM and others at 4 noise × 4 dropout levels. Running our harness on it lets us
compare directly with published F1/temporal-IoU numbers, and it checks that our generator is not
unfairly easy. The clean file is 18 MB. Each noisy variant is 250-400 MB. The license is
non-commercial and no-derivatives according to the paper, so check that before redistributing
anything derived from it.

Not recommended: T-Drive and Porto (taxi behaviour, and Porto has no stays at all); the Milan grid
(no per-device ids); NUMOSIM and the UCL England data (stays only, no pings, although NUMOSIM's
LA stay schedules could drive our synthetic ping generator); MDC, D4D and NetMob (not obtainable).

## How each recommended dataset maps to our schema

| Dataset | id | lat/lon | timestamp → UTC | Accuracy | Shape we can emit | Synthetic error needed? |
|---|---|---|---|---|---|---|
| GeoLife | user folder (`000`-`181`) | exact WGS84 | `.plt` date + time are GMT per the user guide, so use as-is | none | 1 natively; 2/3 by adding error | Optional. Consumer GPS is roughly 5-20 m. To emit shape 2/3, set a nominal CEP (e.g. 10 m) and perturb, or wrap in CDR/adtech profiles by subsampling and adding error |
| STAGA | constant (single subject) | real-looking but **appears displaced** (points sit near 44.38 N, 10.78 E, while data was recorded in Berlin). Treat as anonymized, geometry preserved | `ts` has a +01/+02 offset, so convert to UTC. Dates are shifted to start 2000-01-01 | none | 1; 2/3 via synthetic error | Optional |
| ExtraSensory | UUID file name | exact | unix epoch seconds | none in the absolute-location file | 1 | Optional |
| Foursquare TSMC2014 | `userId` | **venue** coordinates, not device fix | `utcTimestamp` (UTC string) plus `timezoneOffset` minutes | none | 1; 2/3 via synthetic error | **Yes** for adtech realism: jitter around the venue with heterogeneous per-id CEP (10 m-5 km), and optionally add burst pings around each check-in |
| YJMob100K / LYMob | `uid` | **grid cell `x,y ∈ 1..200` only**; city and origin are hidden | **`d` (day 0-74), `t` (30-min slot 0-47)**; real dates are masked | none | 2 or 3 after synthesis | **Yes.** Choose an arbitrary origin and projection, map cell centers to lat/lon, and draw a random offset within the cell. A uniform position in a 500 m square has CEP50 = 500·√(0.5/π) ≈ 200 m, so `horizontalAccuracy ≈ 200` (or an equivalent ellipse). Synthesize timestamps as base date + d days + t·30 min + uniform jitter. `x,y = 999` marks masked test-period records in dataset2; drop them |
| PoL benchmark | `agent_id` | exact (Atlanta) | `time` (naive datetime, 1-min ticks, 2024-01) | noise level is in the file name (0/10/25/50 m Gaussian) | 2 (CEP50 = σ·1.1774 from the stated σ, if σ is per-axis) | No; noise is already applied |

Stay labels available for scoring:
- STAGA: `STAGA diary.csv` (`start_time,end_time`), stop intervals.
- PoL: `ground_truth.parquet` (`agent_id, startTime, endTime, latitude, longitude, loc_id_per_user, num_visits`).
- ExtraSensory: per-minute binary context labels (e.g. `original_label:LOC_home`, `AT_WORK`), to be merged into intervals.
- GeoLife: `labels.txt` per labeled user (`Start Time, End Time, Transportation Mode`). Gaps between segments are weak stay candidates only.

## Per-dataset notes

### GeoLife GPS Trajectories 1.3 (Microsoft Research Asia)
- Download: `https://download.microsoft.com/download/F/4/8/F4894AA5-FDBC-481E-9285-D5F8C4C4F039/Geolife%20Trajectories%201.3.zip`
  (200, 313,164,406 bytes, Last-Modified 2025-02-17). The landing page
  `https://www.microsoft.com/en-us/download/details.aspx?id=52367` returns 403 to curl but works in a browser.
- Format: `Data/<user>/Trajectory/*.plt`. Skip 6 header lines. Each row is
  `lat, lon, 0, altitude_ft, days_since_1899-12-30, date, time`. 69 users have `labels.txt`
  (`Start Time\tEnd Time\tTransportation Mode`). About 5.5M points fall in labeled segments.
- Sampling: mostly dense (1-5 s, or 5-10 m), with gaps between trajectories. Mostly Beijing.
- Terms: free for research, and you must cite Zheng et al. (WWW 2008, WWW 2009, IEEE Data Eng. Bull. 2010). User guide:
  <https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/User20Guide-1.2.pdf>.
- Small derived samples that also resolve:
  - scikit-mobility tutorial sample, users 001 and 005, 217,653 rows, CSV `lat,lng,datetime,uid`:
    `https://raw.githubusercontent.com/scikit-mobility/tutorials/master/mda_masterbd2020/data/geolife_sample.txt.gz` (1.8 MB, inspected).
  - MovingPandas examples:
    `https://raw.githubusercontent.com/movingpandas/movingpandas-examples/main/data/geolife_small.gpkg` (1.0 MB) and `.../geolife_sample.gpkg` (24.7 MB).
    The old path in the main `movingpandas` repo (`tutorials/data/`) is now 404.
  - trackintel test fixtures: `https://github.com/mie-lab/trackintel/tree/master/tests/data` contains
    `geolife/`, `geolife_long/` and `geolife_long_10_MB/` (.plt subsets of users 000-006), plus
    `geolife_staypoints.csv` and `sp_tpls.csv`. These are **trackintel's own outputs, not human
    ground truth**. They are useful as regression fixtures for our trackintel wrapper.

### STAGA (Stop and Trip Annotated GPS and Accelerometer), Spang et al., ISPRS Archives 2022
- Paper: <https://doi.org/10.5194/isprs-archives-XLVIII-4-W1-2022-443-2022>. Data: <https://osf.io/34sft/> (CC BY 4.0).
- Files:
  - `STAGA gps aggregated.csv`: `https://osf.io/download/y84dv/` (15,134,180 B). Columns `ts,x,y,longitude,latitude,motion_score,diary_label`, with 78,865 stop and 43,943 trip points.
  - `STAGA diary.csv`: `https://osf.io/download/z5txn/` (45,004 B), 689 `start_time,end_time` stop intervals.
  - `STAGA accelerometer.csv`: `https://osf.io/download/4nzep/` (624 MB, not needed).
- Sampling: the logger was set to 0.1 Hz, but the effective rate is about 0.011 Hz (one fix every ~90 s) with gaps. That is useful because it sits between dense GPS and adtech.
- Annotation tool: <https://github.com/rgreinacher/gps-diary> (BSD-3).

### ExtraSensory (Vaizman et al., UCSD, 2017)
- Site: <http://extrasensory.ucsd.edu/>. "Publicly available ... conditioned on citing our original paper."
- Locations: `http://extrasensory.ucsd.edu/data/additional_data_files/ExtraSensory.per_uuid_absolute_location.zip` (2,303,359 B). Contains 60 `*.absolute_locations.csv.gz` files with `timestamp,latitude,longitude`. Inspected.
- Labels: `http://extrasensory.ucsd.edu/data/additional_data_files/ExtraSensory.per_uuid_original_labels.zip` (992,570 B). Per-minute binary context labels including location contexts. Inspected.
- Sampling: 20 s recording every minute while the app ran, with frequent gaps. Mostly the San Diego/UCSD area.

### PoL staypoint detection benchmark ("Staypoint Detection from Noisy Trajectory Data [Experiment Paper]", arXiv 2607.19312, 2026)
- Code and results: <https://github.com/amirih/staypoint>. Data: <https://osf.io/v8e4g/> (folder `v1`).
- Clean trajectories: `https://osf.io/download/4jr3g/` (`trajectories_noiselevel0_dropoutlevel0.parquet`, 18,262,950 B). Columns `tick,time,agent_id,latitude,longitude`, with 500 agents and 21.6M rows at 1-min ticks for Jan 2024. Inspected.
- Example noisy variant: `https://osf.io/download/es9da/` (`noiselevel25_dropoutlevel2`, 349,502,500 B). There are 15 noisy/dropout variants, each 60-400 MB.
- Ground truth: `https://raw.githubusercontent.com/amirih/staypoint/main/data/v1/ground_truth.parquet` (1,066,085 B, 65,873 staypoints). Inspected.
- Published baselines at noise 25 m / dropout 2 (r=0.001°, t=5): SSPE F1 0.79, HMM-GEM 0.76,
  trackintel hyperband-tuned 0.75, T-DBSCAN (swept) 0.73. Default trackintel scores about 0.02,
  which reflects parameter mismatch, not the algorithm itself.
- License: the paper states CC BY-NC-ND 4.0. The OSF project has no license field set.

### Foursquare NYC / Tokyo (Yang et al., IEEE TSMC 2015)
- Download: `http://www-public.imtbs-tsp.eu/~zhang_da/pub/dataset_tsmc2014.zip` (25,546,284 B, inspected).
  **The old `www-public.it-sudparis.eu` host no longer resolves.** scikit-mobility's built-in
  `foursquare_nyc` loader still points at the old host, so it will fail. Load the file ourselves.
- Landing page: <https://sites.google.com/site/yangdingqi/home/foursquare-dataset>.
- Format: TSV (latin-1) with `userId, venueId, venueCategoryId, venueCategory, lat, lon, timezoneOffset, utcTimestamp`. NYC has 227,428 rows and 1,083 users. TKY has 573,703 rows and 2,293 users.
- Terms: research use, cite the TSMC paper.

### Foursquare global-scale (LBSN2Vec++, Yang et al. WWW 2019 / TKDE)
- `https://ndownloader.figshare.com/files/29050374` (LSBN2vec++_dataset_WWW2019.zip, 2,684,003,043 B). DOI 10.4121/15112308, **CC BY 4.0**.
- The same landing page offers 22.8M check-ins (114k users, 3.8M venues, Apr 2012-Jan 2014) plus a raw set of 90M check-ins from 2.7M users.

### Massive-STEPS (Wongso et al., 2025)
- <https://github.com/cruiseresearchgroup/Massive-STEPS>. Per-city HF datasets, e.g.
  `https://huggingface.co/datasets/CRUISEResearchGroup/Massive-STEPS-Tokyo/resolve/main/tokyo_checkins.csv` (2.66 MB).
  Columns `trail_id,user_id,venue_id,latitude,longitude,...,timestamp`. It uses more recent (2017-18) check-ins in 15 cities. The card says Apache-2.0, but the underlying data is Foursquare/Semantic Trails.

### Gowalla / Brightkite (SNAP, Cho et al. KDD 2011)
- `https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz` (105,470,044 B)
- `https://snap.stanford.edu/data/loc-brightkite_totalCheckins.txt.gz` (59,986,698 B)
- Format: TSV `user, check-in time (ISO, UTC 'Z'), lat, lon, location_id`. Pages:
  <https://snap.stanford.edu/data/loc-Gowalla.html>, <https://snap.stanford.edu/data/loc-Brightkite.html>.

### YJMob100K (HuMob Challenge 2023; Yabe et al., Scientific Data 2024)
- Zenodo record 10142719 (latest; 10836269 is the same files). **CC BY 4.0**, open.
  - `https://zenodo.org/api/records/10142719/files/yjmob100k-dataset1.csv.gz/content` (419,026,822 B)
  - `https://zenodo.org/api/records/10142719/files/yjmob100k-dataset2.csv.gz/content` (111,175,824 B)
  - `https://zenodo.org/api/records/10142719/files/cell_POIcat.csv.gz/content` (626,648 B)
- Format: `uid,d,t,x,y`. There is one row per (user, 30-min slot) in which the user produced any ping. Cells are 500 m on a 200 × 200 grid, and the city and dates are masked.
- Paper: arXiv 2307.03401.

### LYMob-4Cities (HuMob Challenge 2024 public release)
- Zenodo record 14219563, **CC BY 4.0**, open. Example dev-sized file:
  `https://zenodo.org/api/records/14219563/files/cityD-dataset.zip/content` (30,735,837 B). Others:
  cityA 431 MB, cityB 90 MB, cityC 68 MB, plus POI files.
- The challenge-time record 13237029 is **restricted**. Use 14219563.

### T-Drive trajectory sample (Yuan et al., KDD 2011 / GIS 2010)
- Parts `https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/{06,07,08,09,010,011,012,013,014}.zip`
  (7.8, 11.7, 13.5, 11.8, 10.7, 10.3, 8.8, 7.5, 1.8 MB, total ≈ 84 MB). The server returns 200 only with a browser-like
  `User-Agent` such as `Mozilla/5.0 (X11; Linux x86_64)` (a bare `Mozilla/5.0` or the curl default gets 403 or no response). There is also a OneDrive mirror linked from
  <https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/>.
- Format: per-taxi `taxi_id,datetime,lon,lat`.

### Porto taxi (ECML/PKDD 2015 challenge)
- UCI: `https://archive.ics.uci.edu/static/public/339/taxi+service+trajectory+prediction+challenge+ecml+pkdd+2015.zip`
  (200, streamed without Content-Length; about 509 MB per UCI). DOI 10.24432/C55W25, CC BY 4.0.
- figshare derivative: `https://ndownloader.figshare.com/files/22677902` (`porto_trajectories_all.csv`, 173,370,107 B, CC BY 4.0). I did not inspect its schema.
- Each row is one trip with a `POLYLINE` JSON of 15-s fixes and a start `TIMESTAMP`. There are no points between trips, so it contains no dwells. Use it only for scale and throughput tests.

### Telecom Italia Big Data Challenge: Milan telecom activity
- Harvard Dataverse `doi:10.7910/DVN/EGZHFV` ("Telecommunications - SMS, Call, Internet - MI"). ODbL 1.0. 62 daily files, 20.8 GB.
- Values are aggregated per 235 m grid square per 10 min, with **no device ids**. It cannot test dwell detection. Its possible use is fitting the diurnal CDR event-rate profile for our synthetic CDR generator.

### Restricted or unavailable
- **MDC (Nokia Mobile Data Challenge)**: required a signed Idiap agreement (academic, research-only). The former project pages (`idiap.ch/project/mdc`, `idiap.ch/dataset/mdc`) now return 404, and MDC is not listed on <https://www.idiap.ch/en/research/data>. Treat it as unobtainable without contacting Idiap.
- **D4D Côte d'Ivoire (2013) / Senegal (2014)**: individual antenna-level CDRs released only to challenge participants under contract (arXiv 1407.4885). Not publicly downloadable.
- **NetMob23**: 2.3 TB of aggregated service traffic, access by challenge application only (<https://netmob2023challenge.networks.imdea.org/>).
- **Breadcrumbs (UNIL, 2019)**: GPS + POI ground truth for 81 users. Access is by email request (<https://doplab.unil.ch/breadcrumbs/>). It would be a good labeled set if they grant access.
- **SHL**: labeled `Still` periods are available, but the preview alone is about 7 GB and covers 3 users. Low value per byte for our purpose.
- **Commercial adtech (Veraset, Quadrant, Cuebiq/Spectus)**: free "samples" exist only under vendor terms, so they are not a reproducible public dataset. No open raw adtech ping dataset was found. The closest open items are the UCL England release (already reduced to MSOA-level stays; Zenodo 13327082) and check-in data made adtech-like with synthetic error.

### Newer (2023-2026) stop/stay-detection benchmarks found
- PoL staypoint benchmark (2026), described above. It is the only one with raw pings + stay labels at scale.
- NUMOSIM (2024, arXiv 2409.03024, <https://osf.io/sjyfr/>). Synthetic LA stay sequences for 200k agents with POIs and injected anomalies. It has **no pings**. Example: `https://osf.io/download/aqm9x/` (stay_points_train.parquet, 317 MB).
- DeepStay (2023, arXiv 2306.06068, <https://github.com/christianll9/deepstay>). It derives stay labels from ExtraSensory and weak labels for GeoLife from transport-mode gaps and OSM. The code is reusable for building GeoLife weak labels.
- STAGA (2022), described above.

## URL check commands (re-run to confirm availability)

```bash
H='-sIL -o /dev/null -w %{http_code}\n --max-time 30'
curl $H "https://download.microsoft.com/download/F/4/8/F4894AA5-FDBC-481E-9285-D5F8C4C4F039/Geolife%20Trajectories%201.3.zip"
curl $H "https://osf.io/download/y84dv/"            # STAGA gps
curl $H "https://osf.io/download/z5txn/"            # STAGA diary
curl $H "http://extrasensory.ucsd.edu/data/additional_data_files/ExtraSensory.per_uuid_absolute_location.zip"
curl $H "https://osf.io/download/4jr3g/"            # PoL clean trajectories
curl $H "https://raw.githubusercontent.com/amirih/staypoint/main/data/v1/ground_truth.parquet"
curl $H "http://www-public.imtbs-tsp.eu/~zhang_da/pub/dataset_tsmc2014.zip"
curl $H "https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz"
curl $H "https://zenodo.org/api/records/10142719/files/yjmob100k-dataset2.csv.gz/content"
curl $H "https://zenodo.org/api/records/14219563/files/cityD-dataset.zip/content"
curl $H -A "Mozilla/5.0 (X11; Linux x86_64)" "https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/06.zip"  # bare "Mozilla/5.0" gets 403
# sizes: curl -sIL <url> | grep -i content-length
# Zenodo metadata: curl -s https://zenodo.org/api/records/10142719 | jq '.files[]|{key,size}'
```

All of the above returned 200 on 2026-09-30.

## Ingestion

Code: `src/dwells/public/` (one module per dataset), tests in `tests/test_public.py`. Run:

```bash
uv run python -m dwells.public.build [--only staga|pol|foursquare|geolife]
```

Raw files go to `data/public/raw/<name>/` and are skipped if already there (urllib with a
browser User-Agent). Each output `data/public/<dataset>/` is laid out like a synthetic dataset:
master `events.parquet` (`id, timestamp, lat, lon, true_lat, true_lon, source, sig_a, sig_b, az,
p_mixed, truth_dwell_id`), optional `truth_*.parquet`, and `meta.json` (source URLs, license,
transformations, `sampling.describe` stats). Every output loads with `io.load_events(dir, shape)`
for all five shapes. `p_mixed` is a seeded random 0.68/0.95 per event, as in the generator.

**Error columns.** GeoLife, STAGA and PoL report no accuracy. They get a nominal circular GPS
1-sigma of 8 m (`sig_a = sig_b = 8`, `az = 0`) and positions are not perturbed, so
`true_lat/true_lon` equal `lat/lon`. So shapes 2/3 carry a constant, uninformative error on these
sets, and "true" position means "as recorded", not ground truth. Foursquare is different, see below.

| Dataset dir | Source | Events | Ids | Truth | Gap p50 / p95 | Time span (UTC) | Disk |
|---|---|---|---|---|---|---|---|
| `staga` | STAGA GPS + diary | 122,808 | 1 | 689 dwells | 10 s / 110 s | 1999-12-31 to 2000-05-06 (shifted dates) | 4 MB |
| `pol` | PoL clean (`noiselevel0_dropoutlevel0`) + ground truth | 21,600,997 | 500 | 65,873 dwells | 60 s / 60 s | 2024-01-01 05:00 to 2024-01-31 05:01 | 48 MB |
| `geolife_top20` | GeoLife 1.3, 20 users with most points | 14,768,085 | 20 | none | 2 s / 5 s | 2000 to 2012-07 | 285 MB |
| `geolife_all` | GeoLife 1.3, all users | 24,876,977 | 182 | none | 2 s / 5 s | 2000 to 2012-07 | 465 MB |
| `foursquare_nyc` | TSMC2014 NYC check-ins | 227,428 | 1,083 | 227,428 visits | 4.9 h / 5.0 d | 2012-04-03 to 2013-02-16 | 15 MB |

### STAGA (`staga`, CC BY 4.0)
- `ts` has +01:00/+02:00 offsets and is converted to UTC. The authors shifted dates to start 2000-01-01.
- `truth_dwells` holds the 689 diary stop intervals (`dwell_id` 0..688 by start time). An event gets
  `truth_dwell_id` when it falls in the closed interval `[start_ts, end_ts]`. This matches the raw
  per-point `diary_label` exactly (78,865 stop and 43,943 trip points, zero disagreements).
- Dwell `lat/lon` is the median of its points. Every interval has at least one point. `place_id`
  groups dwell positions within 50 m by leader clustering, which gives 321 places. `category` and
  `radius_m` are null.
- Label semantics: these are diary stops, i.e. whatever the subject logged as a stop. There is **no
  minimum duration**: the shortest is 59 s, 22% are under 5 min and 39% under 15 min (median
  27 min). Short stops (traffic, transfers) will count as misses for algorithms tuned to 15+ min,
  so filter the truth by duration to match the swept `min_duration`.
- Surprises: the median gap is 10 s (p95 110 s). The survey's "~90 s effective" figure is a
  long-run average that includes outages. The anonymized coordinates are not one rigid shift:
  there are 23 jumps over 20 km between consecutive fixes, several of them about 2,000 km in
  seconds (for example 2000-01-28 04:29). The displacement seems to change between segments, so
  distances within a segment are fine but places cannot be compared across a jump.

### PoL staypoint benchmark (`pol`, CC BY-NC-ND 4.0 per the paper)
- **Local research use only. Do not redistribute the raw or derived data.**
- Only the clean variant was ingested. All variants with noise above 0 are 245-400 MB, and the
  repo README links only the OSF project, not individual files. The OSF v1 folder lists
  dropout-only variants under 150 MB (`noiselevel0_dropoutlevel1` 94 MB,
  `noiselevel0_dropoutlevel2` 83 MB at `https://osf.io/download/69ddb33792c88facca118339/`,
  `noiselevel0_dropoutlevel3` 60 MB). They were not downloaded.
- The naive `time` is treated as America/New_York local time (agents are home overnight at local
  clock hours) and converted to UTC (+5 h in January). This only shifts times; labels and
  intervals keep their alignment.
- `truth_dwells` comes from `ground_truth.parquet`. The intervals are whole minutes and closed:
  the tick at `endTime` counts as inside. `place_id = loc_id_per_user`, which is **per agent**
  (0-12), not global. Dwell lat/lon comes from the ground truth. 97.7% of events fall inside a
  staypoint.
- Label semantics: simulator staypoints, with a minimum of 60 s. Only 0.3% are under 5 min and
  1.1% under 15 min (median 3.1 h). During a stay the clean positions are exactly constant, so
  clean-data detection is easy. The published numbers are for noise 25 m / dropout 2.

### GeoLife 1.3 (`geolife_top20`, `geolife_all`, MSR research terms, cite Zheng et al.)
- All 18,670 `.plt` files are parsed. `date + time` (GMT per the user guide) is used as UTC. `id`
  is the user folder (`000`-`181`). One row with out-of-range coordinates is dropped. No truth:
  `labels.txt` transport modes are not used.
- `geolife_top20` is the 20 users with the most points (listed in `meta.json`). They hold 59% of
  all points.
- Surprises: timestamps have 1 s resolution and the files contain sub-second bursts and
  overlapping trajectories. So 911,309 rows (3.7%) share an `(id, timestamp)` with another row,
  often at different coordinates. These are kept as-is: dedupe or aggregate in the algorithm if
  zero time deltas are a problem. Three points carry bogus year-2000 dates.
- Build is about 35 s. Peak RSS is about 11 GB (validation loads every shape of the 25M-row frame).

### Foursquare NYC (`foursquare_nyc`, research use, cite Yang et al. TSMC 2015)
- `utcTimestamp` is already UTC (`+0000`). `timezoneOffset` is dropped. The file is latin-1.
- Made adtech-like. `true_lat/true_lon` is the venue position, and `lat/lon` is the venue plus
  synthetic error, seeded and reproducible. Each ping draws a source from `gps/wifi/cell` with the
  generator's adtech weights 0.45/0.30/0.17 (renormalized, without ip/stale). The 1-sigma is
  lognormal with medians 8 / 35 / 900 m. Axis ratios are U(1, 1.5), or U(1, 3) for cell, using
  the generator's `_rand_axes`. The offset is drawn from the reported ellipse (honest, no
  calibration noise). The resulting sig_a has p50 21 m and p95 1.7 km, and the realized offset
  has p50 20 m and p95 1.7 km.
- One event per check-in, with no burst pings added. 614 rows share an `(id, timestamp)`.
- `truth_visits.parquet` (`id, venue_id, timestamp, lat, lon, category`) holds the raw check-ins at
  venue coordinates. It is for qualitative checks (whether detected places line up with venues)
  and is not dwell truth, since check-ins have no duration.
