# Datasets

This page describes the synthetic data generator, the dataset catalog under `data/synthetic/`, and
the public datasets that were ingested. Use it to judge how realistic and correct the evaluation data
is. The numbers come from the code in `src/dwells/generator/` and from the `meta.json`,
`truth_*.parquet` and `events.parquet` files on disk as of 2026-09-30. Figures were not produced
because matplotlib is not installed, so everything is in tables.

Code map:

| File | Role |
|---|---|
| `src/dwells/generator/area.py` | OSM study areas: road graph and categorized places |
| `src/dwells/generator/simulate.py` | Agents, households, routines, travel. Produces the true trajectory and the true stays |
| `src/dwells/generator/sampling.py` | Sensing profiles `gps` / `cdr` / `adtech`: when events happen and how much error they carry |
| `src/dwells/generator/build.py` | Catalog specs; writes `events.parquet`, `truth_*.parquet`, `meta.json` |
| `src/dwells/io.py` | `shape_view`: turns the master events frame into the 5 evaluated shapes |
| `src/dwells/uncertainty.py` | CEP50 / ellipse / covariance conversions |
| `src/dwells/metrics.py` | `colocation_truth`, `dwell_metrics` (uses `n_events` for the observable-recall metric) |

Rebuild: `uv run python -m dwells.generator.build [--only NAME] [--group dev|eval|scale] [--list]`.

---

## 1. Study areas (`area.py`)

| Area | Center (lat, lon) | Radius | Road network | Travel speed | UTC offset used |
|---|---|---|---|---|---|
| `urban`: Chicago Loop / River North / South Loop | 41.8830, -87.6310 | 2,500 m | OSM `drive`, largest strongly connected component: 1,648 nodes, 3,908 directed edges | 18 km/h | -6 h (CST) |
| `suburban`: Macon, GA | 32.8407, -83.6324 | 7,000 m | OSM `drive`, largest SCC: 4,768 nodes, 12,758 edges | 45 km/h | -5 h (EST) |

The areas are downloaded once with osmnx (`graph_from_point`, `features_from_point`) and cached as
parquet:

- `data/osm/<area>/nodes.parquet`: `osmid, lat, lon`
- `data/osm/<area>/edges.parquet`: `u, v, length_m`
- `data/osm/<area>/features.parquet`: `category, lat, lon, area_m2` (centroid and footprint area of each feature)
- `data/osm/_cache/`: raw osmnx HTTP cache

`load_area` never touches the network if `features.parquet` exists. Edge weight is
`length_m / speed`. Parallel edges keep the minimum weight.

### Place categorization (`_categorize`)

Features are fetched with tags `building`, `amenity`, `shop`, `office`, and
`leisure ∈ {park, fitness_centre, sports_centre}`. The rules below are applied in order and the
first match wins:

| Rule | Category |
|---|---|
| amenity ∈ restaurant, cafe, fast_food, bar, pub, food_court | food |
| amenity = fuel | fuel |
| amenity = place_of_worship | worship |
| amenity ∈ school, kindergarten, university, college | school |
| shop ∈ supermarket, convenience, greengrocer, grocery | grocery |
| any other shop | shop |
| leisure ∈ fitness_centre, sports_centre | gym |
| leisure = park | park |
| any office tag, or amenity ∈ hospital, clinic, bank, townhall, courthouse, police | work |
| building ∈ house, residential, apartments, detached, semidetached_house, terrace, bungalow, dormitory | home |
| building ∈ commercial, office, industrial, retail, warehouse, hospital, government, civic, hotel | work |
| building = yes | `building_yes` (resolved at load time) |
| anything else | dropped |

At load time `building_yes` becomes **home** if its footprint is under 400 m² and **work**
otherwise. Place radius is `sqrt(area_m2 / π)` clipped to [8, 150] m. Point features (area 0) get
15 m. Every place is snapped to its nearest road node, which is where trips start and end.

Resulting place inventory:

| Category | Urban n | Urban median radius (m) | Suburban n | Suburban median radius (m) |
|---|---:|---:|---:|---:|
| home | 6,036 | 8.0 | 35,240 | 8.0 |
| work | 2,055 | 15.3 | 2,505 | 14.6 |
| food | 1,489 | 15.0 | 141 | 11.6 |
| shop | 844 | 15.0 | 87 | 15.0 |
| park | 128 | 31.6 | 81 | 28.4 |
| grocery | 94 | 15.0 | 23 | 13.2 |
| gym | 71 | 15.0 | 3 | 15.0 |
| school | 68 | 32.6 | 52 | 15.0 |
| worship | 42 | 16.6 | 85 | 15.0 |
| fuel | 13 | 10.8 | 46 | 14.5 |

Places are not deduplicated. An amenity point inside a building polygon is a second place a few
meters from the building centroid, which adds dense near-duplicates in the urban core. 4,758 of the
urban places (35,277 suburban) were untagged `building=yes`, so most homes come from the
footprint-size rule rather than from real residential tags.

### Travel (`simulate._Router`)

- Each agent gets one Dijkstra shortest-travel-time tree from every road node attached to one of its
  places.
- Expected trip time = road time + (access distance from place centroid to its node, at both ends) /
  1.4 m/s + 90 s for parking. If there is no road path (not expected within the SCC), the fallback
  is straight-line distance at 10 m/s.
- Realized trip time = expected × `exp(N(0, 0.15))`: multiplicative lognormal noise, σ = 0.15, drawn
  independently per trip.
- Geometry is place centroid → node polyline → place centroid. The agent moves at **constant speed
  along the polyline**, with knots spaced by distance fraction. There is no acceleration, no stop at
  intersections and no congestion. Positions between knots are linear interpolations in local ENU
  meters (equirectangular around the area center).

---

## 2. Agent model (`simulate.py`)

Defaults (`SimConfig`): `start_date = 2024-03-04` (a Monday), `workplace_pool_frac = 0.2`,
`neighbor_frac = 0.25`, `friend_pairs_frac = 0.3`, `meet_rate_per_week = 1.0`,
`travel_noise = 0.15`. `n_agents`, `n_days`, `seed` and `utc_offset_h` come from the catalog spec.
Days are planned in local time: `day0 = local midnight` = UTC midnight − `utc_offset_h`.

### Households and shared places

- Household size is drawn from {1, 2, 3, 4} with p = {0.30, 0.35, 0.20, 0.15}. Members share the
  home place.
- **Home choice.** Households after the first have a 25 % chance (`neighbor_frac`) to be a
  **next-door neighbor**: pick a random existing household and take the nearest unused home to it.
  `truth_agents.neighbor_of` records which household. Otherwise the home is uniform over all unused
  home places. In the eval sets the neighbor homes are very close: median 8.6 m apart in urban (p10
  5.5, p90 11.2, 11 households) and 21.4 m in suburban (p10 14.4, p90 33.3, 12 households). They
  are different `place_id`s, so they are not true colocations. They are near-miss decoys.
- **Shared per household:** one grocery (distance decay, 1,500 m) and, with probability 0.4, one
  worship place (2,000 m).
- **Workplace pool.** `max(3, int(0.2 × n_agents))` work places are drawn uniformly from all work
  places. Each worker or shift worker picks one of them uniformly, so a 100-agent set has 20 pooled
  workplaces (22 distinct used in eval, including schools) and about 3 people per workplace.
  Coworkers and shift workers share a workplace, sometimes at different times.
- **School pool.** `max(2, n_agents // 20)` schools, capped by the number available. Students pick
  one uniformly.

### Personas

Each member draws a persona independently: worker 0.55, shift 0.08, student 0.10, home 0.15,
remote 0.12. Shift workers are split 50/50 into `night` and `swing`. Eval realizations: urban
46 / 9 / 13 / 16 / 16 and suburban 42 / 12 / 12 / 21 / 13 (worker / shift / student / home / remote).

### Favorite places (distance decay)

`_PlacePicker.near` samples without replacement with weight `exp(-d / scale)`, where d is the
Euclidean distance from an anchor. The anchor is home, or the work/school place for "lunch" and
"gym":

| Favorite | Anchor | Scale | k |
|---|---|---|---|
| grocery | home | 1,500 m (household-shared) | 1 |
| food | home | 1,500 m | 2 |
| lunch (food) | work/school, else home | 400 m | 2 |
| shop | home | 2,000 m | 2 |
| park | home | 1,500 m | 1 |
| fuel | home | 2,000 m | 1 |
| gym | work/school, else home | 2,000 m, only with probability 0.35 | 1 |
| worship | household-shared (40 % of households) | 2,000 m | 1 |

Discretionary stay durations are uniform in minutes: food 25–90, grocery 12–45, shop 10–60,
gym 40–100, park 20–120, worship 60–120, fuel 3–9, friend meeting 60–180. Every stay lasts at least
60 s (`MIN_STAY_S`). If a planned departure is earlier than arrival + 60 s, the agent leaves after
exactly 60 s and the rest of the day shifts later.

### Daily routines (`_plan_day`)

`N(μ, σ)` below means local clock time μ h with Gaussian σ h. "Errand" means: pick a category at
random from the allowed list among the agent's favorites, pick a favorite place in it, and stay the
category duration. After each errand there is a 50 % chance to chain straight to the next one.
Otherwise the agent goes home and waits U(0.5, 2) h before the next errand. After the last errand
the agent goes home.

| Persona / day | Routine |
|---|---|
| **Night shift** (every day) | If at work, go home at N(6.0, 0.2). With probability 0.4, one errand (grocery / shop / food / gym) at N(14, 1). Friend meetings only if they start before 20:30. **Mon–Fri**: leave for work at N(21.5, 0.2). The day ends there, so there is no weekend routine and no evening errand. Works Mon–Fri nights. |
| **Swing shift, Mon–Fri** | With probability 0.4, one errand (grocery / gym / shop) at N(10, 1). Leave for work at N(13.6, 0.15), leave work at N(22.5, 0.2). No meetings or evening errand. Weekends use the generic routine below. |
| **Worker / student, Mon–Fri** | Morning: with probability 0.12 (if a fuel favorite exists), fuel at N(7.5, 0.4) then work. Otherwise leave home at N(7.75, 0.4) (worker) or N(7.4, 0.4) (student). Worker lunch with probability 0.3: a lunch favorite at N(12, 0.25) for U(25, 60) min, then back to work. Leave at N(17.0, 0.6) (worker) or N(15.0, 0.6) (student). With probability 0.35, one errand (grocery / gym / shop / fuel), otherwise straight home. |
| **Generic** (weekends for everyone except night shift; home and remote personas on weekdays) | Number of errands: 0–2 on weekdays, 1–3 on weekends (uniform integers). **Sunday**: worship at N(9.3, 0.3) if the household has one, then home. **Remote, weekday**: with probability 0.5, one errand (gym / food / park) at N(12, 1.5). Then the errands (grocery / shop / park / food / gym / fuel), starting at N(10.5, 1.5). |
| **Evening** (all except the shift branches that return early) | If the agent has a friend meeting today, attend it. Otherwise, with probability 0.15, one errand (food / park / shop) at N(19.5, 0.8). |

### Friend meetings

`int(0.3 × n_agents)` random pairs are drawn, skipping pairs from the same household. Each pair
has 1–3 candidate places: 2 food places and 1 park near the midpoint of the two homes (1,500 m
scale). The number of meetings is Poisson(1.0 × n_days / 7), on distinct days. Start time is
U(12, 16) h on weekends and U(18.5, 20) h on weekdays. Duration is U(60, 180) min.

Each agent decides independently. It skips a meeting if the meeting starts less than 15 min from
"now" (the agent is still busy), or, for night shift, if it starts after 20:30. The agent leaves at
`start − expected travel time`, so the actual arrival is off by the travel noise, and goes home at
`end`. **`truth_meetings` holds the *scheduled* meetings, not the realized ones.** In
`eval_urban`, 52 of 83 meetings were attended by both people, 21 by one and 10 by neither. In
`eval_suburban` it was 51 of 79 by both and 6 by neither. Use `colocation_truth` (below), not
`truth_meetings`, as the colocation label.

### Stay wander

The stays themselves are knots at the place centroid. At sampling time, each event that falls inside
a stay is displaced **uniformly within a disk of the place radius** (`r = R·sqrt(U)`), drawn
independently per event. The displaced point becomes `true_lat/true_lon`, and the sensing error is
added on top of it. With home radius 8 m and most POIs at 15 m, wander is small next to GPS error.
It is larger for parks and big workplaces (up to 150 m).

---

## 3. Sensing profiles (`sampling.py`)

The RNG is seeded per dataset as `default_rng([seed, profile_index])`. All three profiles share the
same pipeline: event times → true position (trajectory interpolation plus stay wander) → reported
position and reported error ellipse. The reported error is `sig_a ≥ sig_b` (1-σ meters) plus `az`
(the bearing of the major axis, in [0, 180)).

Diurnal activity factor by local hour, used by cdr and adtech for thinning: 0–6 h 0.08,
6–8 h 0.6, 8–22 h 1.0, 22–24 h 0.5.

### Timing

| | gps | cdr | adtech |
|---|---|---|---|
| Base process | Per-agent period `base ~ U(10, 60)` s. Regular while traveling and for the first 10 min after arrival. After that the **stationary throttle** switches to a period of U(240, 360) s (drawn per stay) | Inhomogeneous Poisson: rate `~ lognormal(ln 1.2, 0.6)` per hour per agent, thinned by the diurnal factor | Sessions: rate `~ lognormal(ln 0.4, 1.0)` per hour per agent (a wide light-to-heavy spread), thinned by the diurnal factor |
| Bursts | none | Each event spawns Geometric follow-ups (mean 0.8) within U(0, 300) s | Each session spawns Geometric follow-ups (mean 2.0) within U(0, 480) s |
| Outages | Poisson 0.15 per day, each lasting U(1, 8) h, all events inside dropped | Poisson 0.05 per day, U(2, 12) h | none |

For gps, each regular block (travel leg, warm-up, throttled stay) gets a single random phase
offset U(0, period). A block's last samples can spill up to one period into the next segment.

The throttle uses the true stay intervals. It is an oracle, with no motion-detection delay on
departure. `scale_gps_single_1m` uses `base_range = (4, 6)` and `throttle = False`, which gives
about 5 s sampling the whole time.

### Error models

**gps.** 1-σ `~ lognormal(ln 6 m, 0.5)`, multiplied by **2.5 during any stay** (the indoor
inflation, also applied in parks). The ellipse axis ratio is U(1, 1.6) and the azimuth is
U(0, 180). The offset is drawn from the true covariance (see calibration below). **2 % outliers**
(`source = gps_outlier`) get an extra 50–300 m displacement added on top, while the reported
ellipse stays unchanged.

**cdr.** Towers sit on a square grid with 600 m spacing in urban and 2,500 m in suburban (500 m on
the test grid), each jittered by N(0, 0.2·spacing). Each tower has three 120° sectors with a random
rotation U(0, 120)°. Sector radius R = 0.9·spacing. For each event:
- serving tower = the nearest tower, or the second nearest with probability 0.2 (handover), drawn
  independently per event;
- sector = the 120° wedge of that tower containing the true point;
- **reported position = the sector centroid** of a uniform 120° wedge of radius R. This is a
  deterministic quantization, not a noisy draw;
- **reported ellipse = the spatial covariance of that uniform wedge**, computed once by Monte Carlo
  (20k points). The major axis runs across the sector bearing.

As a result every CDR event in an area has the **same** `sig_a, sig_b`: urban 206.8 / 120.0 m,
suburban 861.5 / 499.9 m. Only the azimuth varies. There are only 176 (urban) and 78 (suburban)
distinct reported positions in the eval sets. Calibration noise is not applied to CDR.

**adtech.** Each ping draws a location source:

| Source | p | 1-σ (lognormal median, shape) | Axis ratio |
|---|---|---|---|
| gps | 0.45 | 8 m, 0.4 | U(1, 1.5) |
| wifi | 0.30 | 35 m, 0.4 | U(1, 1.5) |
| cell | 0.17 | 900 m, 0.5 | U(1, 3) |
| ip | 0.08 | 6,000 m, 0.4 | U(1, 1.5) |

- gps, wifi and cell offsets are drawn from the (calibrated) ellipse. There are no outliers and no
  indoor inflation for adtech gps.
- **IP**: the reported position is replaced by one of three fixed **IP hubs** chosen uniformly, at
  ENU offsets (0, 0), (+3,000, −2,000) and (−2,500, +1,500) m from the area center. The same offsets
  are used in both areas, which puts two of the hubs outside the 2.5 km urban area. The reported σ
  is the 6 km lognormal value and has nothing to do with the real distance to the hub.
- **Stale fixes**: each ping after the first has a 4 % chance to copy the previous ping's reported
  position and ellipse (a cached fix re-reported with a new timestamp). Chains are possible.
  `true_lat/true_lon` and `truth_dwell_id` still describe the current time.

### Calibration: reported vs true error ("honest")

For gps, wifi and cell draws, the offset comes from `N(0, Σ_true)` with
`Σ_true = Σ_reported / c²` and `c ~ lognormal(0, 0.25)` drawn **independently per event**. The
reported ellipse is "honest" in the sense that it is right on average (median c = 1). Any single
event can be over- or under-stated by about ±25 % in σ. The lognormal mixture makes the error
tails heavier than the stated Gaussian. Measured coverage of the reported 1-σ ellipse, scaled to
p (eval_suburban):

| Source | Median reported σ_a | Median true error | Inside 68 % ellipse | Inside 95 % ellipse |
|---|---:|---:|---:|---:|
| gps (gps profile) | 15.3 m | 14.9 m | 0.672 | 0.919 |
| gps_outlier | 15.4 m | 176 m | 0.006 | 0.021 |
| cdr | 861 m | 828 m | 0.617 | 0.999 |
| adtech gps | 8.9 m | 9.1 m | 0.674 | 0.921 |
| adtech wifi | 39 m | 40 m | 0.671 | 0.918 |
| adtech cell | 1,240 m | 1,065 m | 0.673 | 0.920 |
| adtech ip | 6,724 m | 4,719 m | 0.859 | 0.973 |
| adtech stale | 27 m | 34 m (p95 4.7 km) | 0.534 | 0.789 |

What this shows: nominal 68 % coverage is right for Gaussian sources. 95 % coverage is about 0.92
because of the calibration noise. CDR is bounded rather than Gaussian: too tight at 68 %, far too
loose at 95 %. IP is conservative. Outliers and stale fixes are the dishonest events.

### Event counts by profile (eval sets)

| | gps | cdr | adtech |
|---|---|---|---|
| Events per id per day, p10 / p50 / p90 (min – max) | 338 / 381 / 484 (317–744) | 18 / 35 / 70 (10–152) | 6.4 / 26 / 80 (0.8–285) |
| Fraction of events inside a true stay | 0.84 | 0.97 | 0.97 |
| Source mix | gps 98 %, gps_outlier 2 % | cdr 100 % | gps 43.1 %, wifi 28.9 %, cell 16.4 %, ip 7.7 %, stale 3.9 % |

---

## 4. Master events and the five shapes

`events.parquet` has one row per event:

| Column | Meaning |
|---|---|
| `id` | agent id (`u00000` …) |
| `timestamp` | UTC, `Datetime(us, UTC)` |
| `lat, lon` | **reported** position (WGS84) |
| `true_lat, true_lon` | true position, including stay wander (truth only) |
| `source` | gps, gps_outlier, cdr, wifi, cell, ip, stale (truth only) |
| `sig_a, sig_b, az` | reported 1-σ major/minor axis (m) and major-axis bearing in [0, 180) |
| `p_mixed` | per event 0.68 or 0.95 (50/50), used by shape `3_mixed` |
| `truth_dwell_id` | `dwell_id` of the true stay containing the timestamp, null while traveling (truth only) |

`io.shape_view(master, shape)` projects the frame. **All shapes share the same `lat/lon`**, so
differences between shapes come only from the error information.

| Shape | Columns added to `id, lat, lon, timestamp` | Derivation |
|---|---|---|
| `1` | none | |
| `2` | `horizontalAccuracy` (CEP50, m) | `sqrt((sig_a² + sig_b²)/2) · 1.1774`: the CEP50 of the circle with the same RMS radius. Orientation and elongation are lost |
| `3_p68` | `semiMajor, semiMinor, azimuth, probability=0.68` | axes = σ · k(p), with k = sqrt(−2 ln(1−p)) = 1.5096 |
| `3_p95` | same, `probability=0.95` | k = 2.4477 |
| `3_mixed` | same, `probability=p_mixed` | per-event k |

`uncertainty.with_covariance` inverts these exactly. Shape 3 at any p recovers the same covariance
as the master `sig_a, sig_b, az`. Shape 2 recovers a circle with the same trace. Truth columns only
appear with `keep_truth=True`.

---

## 5. Ground-truth tables

| File | Columns | Notes |
|---|---|---|
| `truth_dwells.parquet` | `id, dwell_id, place_id, category, start_ts, end_ts, duration_s, lat, lon, radius_m, n_events` | Every true stay, including overnight home stays, the stay open at t0 and the stay still open at the end (extended to t_end). Travel is not a dwell. `lat/lon` is the place centroid. `category` is the place category, so friend meetings appear as food or park. **`n_events`** = number of events whose timestamp falls in the stay, i.e. that have `truth_dwell_id` = this dwell. It counts every event, including km-scale IP/cell and stale fixes. |
| `truth_places.parquet` | `place_id, category, lat, lon, radius_m` | Only places that someone visited. Ids are global per area |
| `truth_agents.parquet` | `id, household, persona, home, work, neighbor_of, shift` | `work` is the school place for students and null for home/remote. `neighbor_of` is a household index, −1 if none |
| `truth_meetings.parquet` | `id_a, id_b, day, place_id, start, end` | **Scheduled** meetings. See the attendance numbers in section 2 |

Observability in the evaluation: `metrics.dwell_metrics` keeps true dwells with
`duration_s ≥ min_duration_s` and reports plain recall plus `recall_observable`, where observable
means `n_events ≥ 2`.

### Colocation truth (`metrics.colocation_truth`)

This is a self-join of `truth_dwells` on **`place_id`** over pairs with `id_a < id_b`:

- `same_time`: the two stays overlap by at least `min_overlap_s = 900` s. The label carries the
  overlap interval.
- `diff_time`: the stays do not overlap (`overlap_s ≤ 0`) and the gap is at most 7 days. Here
  `overlap_s` is set to 0 and `overlap_start > overlap_end`, i.e. the columns hold the gap bounds.

Because the join is on `place_id`, next-door neighbors never produce a true colocation, so decoys
are pure negatives. Same-time truth is mostly household members at home and coworkers at work:

| | eval_urban | eval_suburban |
|---|---|---|
| same_time episodes / distinct pairs | 4,854 / 231 | 4,337 / 288 |
| diff_time episodes / distinct pairs | 62,188 / 721 | 60,113 / 1,204 |
| same_time by category | home 2,847, work 1,551, school 210, worship 171, food 39, park 26, other 10 | home 2,557, work 1,342, school 180, worship 79, food 63, park 47, gym 33, grocery 28, shop 8 |

The truth comes from true stays whatever their observability, so a pair counts as colocated even if
neither person has an event during the overlap.

---

## 6. Catalog

Specs come from `build.catalog()`. `dev_*` uses seed 11, `eval_*` seed 1, `scale_gps_*` seed 5,
`scale_adtech_*` seed 6 and the single-id set seed 7. `sim` is seeded by the spec seed only, so
**within one area and seed the gps, cdr and adtech datasets share the same agents and true stays**.
Cross-profile comparisons are therefore paired. Scale sets with different `n_agents` get different
workplace pools, so they are not nested subsets of each other.

Stats come from `meta.json` (`sampling.describe`). Gaps are consecutive-event gaps within an id,
over all ids. σ is `sig_a`.

| Dataset | Area | Profile | Agents | Days | Events | Events/id/day | Gap p50 / p95 (s) | σ p50 / p95 (m) | Truth dwells | Build (s) |
|---|---|---|---:|---:|---:|---:|---|---|---:|---:|
| dev_urban_gps | urban | gps | 20 | 7 | 53,724 | 383.8 | 272 / 351 | 15.3 / 37.3 | 492 | 0.2 |
| dev_urban_cdr | urban | cdr | 20 | 7 | 5,331 | 38.3 | 248 / 10,010 | 206.8 / 206.8 | 492 | 0.1 |
| dev_urban_adtech | urban | adtech | 20 | 7 | 3,936 | 28.2 | 128 / 14,371 | 27.2 / 5,456.8 | 492 | 0.1 |
| dev_suburban_gps | suburban | gps | 20 | 7 | 51,274 | 366.2 | 272 / 355 | 15.8 / 37.7 | 516 | 0.2 |
| dev_suburban_cdr | suburban | cdr | 20 | 7 | 5,331 | 38.3 | 248 / 10,010 | 861.5 / 861.5 | 516 | 0.2 |
| dev_suburban_adtech | suburban | adtech | 20 | 7 | 3,936 | 28.2 | 128 / 14,371 | 27.2 / 5,456.8 | 516 | 0.2 |
| eval_urban_gps | urban | gps | 100 | 21 | 890,207 | 423.9 | 259 / 348 | 14.8 / 36.9 | 7,162 | 1.2 |
| eval_urban_cdr | urban | cdr | 100 | 21 | 86,559 | 41.2 | 246 / 8,954 | 206.8 / 206.8 | 7,162 | 0.6 |
| eval_urban_adtech | urban | adtech | 100 | 21 | 83,626 | 39.9 | 112 / 9,792 | 26.8 / 5,866.0 | 7,162 | 0.6 |
| eval_suburban_gps | suburban | gps | 100 | 21 | 839,210 | 399.6 | 266 / 349 | 15.3 / 37.4 | 7,022 | 1.3 |
| eval_suburban_cdr | suburban | cdr | 100 | 21 | 86,559 | 41.2 | 246 / 8,954 | 861.5 / 861.5 | 7,022 | 0.8 |
| eval_suburban_adtech | suburban | adtech | 100 | 21 | 83,626 | 39.9 | 112 / 9,792 | 26.8 / 5,866.0 | 7,022 | 0.9 |
| scale_gps_1 | suburban | gps | 1 | 28 | 9,849 | 351.8 | 270 / 338 | 15.8 / 37.8 | 106 | 0.1 |
| scale_gps_8 | suburban | gps | 8 | 28 | 97,283 | 434.3 | 257 / 347 | 14.9 / 36.9 | 816 | 0.2 |
| scale_gps_80 | suburban | gps | 80 | 28 | 907,807 | 405.3 | 264 / 349 | 15.2 / 37.2 | 7,843 | 1.3 |
| scale_gps_800 | suburban | gps | 800 | 28 | 9,367,454 | 418.2 | 262 / 348 | 15.0 / 37.1 | 77,891 | 12.9 |
| scale_gps_single_1m | suburban | gps (5 s, no throttle) | 1 | 60 | 957,479 | 15,958.0 | 5 / 5 | 16.9 / 39.0 | 167 | 0.6 |
| scale_adtech_20 | suburban | adtech | 20 | 28 | 19,719 | 35.2 | 125 / 12,267 | 27.0 / 6,016.6 | 1,974 | 0.2 |
| scale_adtech_200 | suburban | adtech | 200 | 28 | 197,517 | 35.3 | 115 / 11,451 | 26.6 / 5,911.8 | 19,362 | 1.9 |
| scale_adtech_2000 | suburban | adtech | 2,000 | 28 | 1,969,661 | 35.2 | 113 / 11,500 | 26.7 / 5,889.6 | 193,666 | 18.4 |
| scale_adtech_18000 | suburban | adtech | 18,000 | 28 | 16,494,494 | 32.7 | 118 / 12,500 | 26.6 / 5,894.7 | 1,747,991 | 216.0 |

Things to note in this table:
- The gps median gap (about 4.5 min) is set by the stationary throttle, since 84 % of events fall
  inside stays.
- For cdr and adtech, urban and suburban have **identical event counts and gap statistics**. The
  times do not depend on the trajectory, and the RNG seed is the same, so each agent's timestamps in
  urban are exactly its suburban timestamps shifted by 1 h (the UTC offset difference). Only
  positions and truth differ.
- CDR σ is one constant per area (see section 3), so shapes 2 and 3 add only the azimuth for cdr.
- `scale_gps_single_1m` has 957,479 events, slightly under the 1M its name suggests, because
  outages remove events.

### True dwell durations by category

The truth is identical across profiles within an area. Eval sets:

| Category | Urban n | Urban p10 / p50 / p90 (min) | Suburban n | Suburban p10 / p50 / p90 (min) | Share < 5 min (U / S) | Share < 15 min (U / S) |
|---|---:|---|---:|---|---|---|
| home | 3,068 | 57.5 / 808.9 / 1,327.2 | 2,998 | 57.8 / 815.8 / 1,367.2 | 0.053 / 0.044 | 0.055 / 0.046 |
| work | 1,039 | 217.4 / 493.9 / 572.6 | 1,004 | 223.7 / 502.4 / 574.5 | 0 / 0 | 0 / 0 |
| food | 766 | 31.2 / 54.1 / 86.6 | 724 | 31.2 / 53.9 / 86.8 | 0 / 0 | 0 / 0 |
| shop | 538 | 13.9 / 34.0 / 55.2 | 482 | 15.0 / 35.4 / 56.2 | 0 / 0 | 0.123 / 0.100 |
| fuel | 472 | 3.7 / 5.9 / 8.1 | 504 | 3.7 / 6.1 / 8.3 | 0.356 / 0.327 | 1.0 / 1.0 |
| park | 459 | 30.4 / 72.0 / 113.9 | 504 | 30.3 / 73.0 / 114.8 | 0 / 0 | 0 / 0 |
| grocery | 381 | 14.8 / 28.8 / 41.6 | 425 | 15.3 / 29.8 / 42.2 | 0 / 0 | 0.110 / 0.087 |
| school | 195 | 391.6 / 451.1 / 498.9 | 180 | 392.2 / 436.6 / 498.7 | 0 / 0 | 0 / 0 |
| worship | 126 | 67.8 / 90.5 / 115.2 | 87 | 62.8 / 82.7 / 109.6 | 0 / 0 | 0 / 0 |
| gym | 118 | 46.1 / 68.4 / 94.6 | 114 | 46.6 / 70.2 / 95.8 | 0 / 0 | 0 / 0 |
| **all** | 7,162 | median 175.6 | 7,022 | median 158.3 | 0.046 / 0.042 | 0.104 / 0.103 |

- The short home stays are **exactly 60 s** (164 urban, 131 suburban). They come from the
  minimum-stay rule: the agent gets home, and the next planned departure (evening errand, meeting,
  or the Sunday errand block after worship) is already in the past.
- The longest stays are multi-day home stays of home-persona agents: 78 h urban, 95 h suburban.
- Fuel is the only category that disappears completely at `min_duration ≥ 15 min`.

### Observability: the recall ceiling

An algorithm can only find a dwell it has events for. The table gives the share of true dwells
with `n_events ≥ 2` for all dwells and at each swept `min_duration`. It also gives the stricter
share whose in-dwell events span at least the threshold (`span ≥ D`), and at least half of it
(`span ≥ D/2`, roughly what an IoU ≥ 0.5 match needs if the algorithm does not extend dwells
across gaps).

| Set | Profile | All dwells: n ≥ 1 / n ≥ 2 | ≥ 5 min: n≥2 / span≥D | ≥ 15 min: n≥2 / span≥D / span≥D/2 | ≥ 30 min: n≥2 / span≥D | ≥ 60 min: n≥2 / span≥D |
|---|---|---|---|---|---|---|
| eval_urban | gps | 0.986 / **0.981** | 0.987 / 0.981 | 0.987 / 0.977 / 0.987 | 0.988 / 0.968 | 0.993 / 0.973 |
| eval_urban | cdr | 0.758 / **0.676** | 0.709 / 0.639 | 0.750 / 0.649 / 0.671 | 0.786 / 0.650 | 0.884 / 0.740 |
| eval_urban | adtech | 0.604 / **0.544** | 0.569 / 0.516 | 0.601 / 0.457 / 0.489 | 0.629 / 0.467 | 0.723 / 0.540 |
| eval_suburban | gps | 0.986 / **0.982** | 0.987 / 0.979 | 0.989 / 0.975 / 0.988 | 0.989 / 0.967 | 0.992 / 0.978 |
| eval_suburban | cdr | 0.753 / **0.670** | 0.699 / 0.629 | 0.741 / 0.641 / 0.664 | 0.780 / 0.646 | 0.883 / 0.742 |
| eval_suburban | adtech | 0.609 / **0.545** | 0.568 / 0.519 | 0.602 / 0.456 / 0.491 | 0.635 / 0.466 | 0.731 / 0.539 |

Dev sets (7 days, 20 agents), all dwells with n ≥ 2: urban gps 0.988, cdr 0.681, adtech 0.504;
suburban gps 0.979, cdr 0.676, adtech 0.510.

Per id (eval_suburban), the share of an agent's dwells with n ≥ 2 at p10 / p50 / p90: gps 0.955 /
0.989 / 1.0, cdr 0.51 / 0.67 / 0.82, adtech 0.30 / 0.56 / 0.78.

By category (eval_suburban, n ≥ 2):

| Profile | home | work | school | worship | park | gym | food | shop | grocery | fuel |
|---|---|---|---|---|---|---|---|---|---|---|
| gps | 0.984 | 0.994 | 1.0 | 0.943 | 0.98 | 0.982 | 0.981 | 0.975 | 0.979 | 0.958 |
| cdr | 0.852 | 0.944 | 0.989 | 0.644 | 0.562 | 0.544 | 0.474 | 0.299 | 0.273 | 0.046 |
| adtech | 0.722 | 0.792 | 0.900 | 0.391 | 0.351 | 0.342 | 0.311 | 0.263 | 0.193 | 0.034 |

How to read these:
- **Adtech recall is capped at about 0.60** at the default 15-min threshold (n ≥ 2), and at
  about 0.46–0.49 when the events must also span the dwell. Short errands (shop, grocery, food) are
  mostly invisible. Home, work and school carry nearly all the recoverable signal.
- For adtech, n ≥ 2 is itself optimistic. It counts IP (6 km) and stale fixes. Requiring 2 events
  with σ ≤ 100 m that are not stale drops observability to 0.511 (all) and 0.565 (≥ 15 min) in
  eval_suburban (urban 0.508 / 0.563).
- CDR's temporal ceiling is about 0.74–0.75 at 15 min. Its spatial resolution is another,
  independent limit: every place in the same sector (600 m or 2.5 km grid) reports the same point.
  An algorithm can detect the stay but cannot tell places in the same sector apart.
- The gps ceiling of about 0.98 is set by outages.

---

## 7. Known simplifications and threats to validity

**Time**
- **Fixed UTC offset, no DST.** The simulation starts on 2024-03-04. US DST began on 2024-03-10, so
  in the 21-day eval sets (and the 28/60-day scale sets) every day after March 10 is planned 1 h
  off from real local time. Within the data this is consistent: downstream `patterns` and `nextloc`
  take the same fixed `utc_offset_h`. A tz-aware conversion would show a 1 h shift from March 10.
- Routines are clock-driven Gaussians around fixed times, with no day-to-day persistence (no habits
  that drift, no holidays, no travel out of the area, no sick days). A night-shift worker's day has
  no weekend-specific routine.
- There is no visitor inflow. Every agent lives in the area and never leaves the bounding radius.

**Space and movement**
- **Urban travel uses the drive network** at a constant 18 km/h plus 90 s parking and 1.4 m/s
  access walking. There is no pedestrian network, no transit and no multimodal trips. For the
  Chicago Loop this gets route geometry and door-to-door times wrong. Movement is constant speed
  along straight node-to-node segments, with no stops at signals.
- **No indoor or multi-floor model.** A building is a disk (radius from footprint, clipped to
  8–150 m). People in different units of the same tower share one centroid, and homes are small
  footprints. Urban "homes" are mostly small untagged buildings (< 400 m²), whereas real Loop
  residents live in large apartment towers, which this rule classifies as work.
- **Uniform wander**: iid uniform in the disk, per event, with no temporal correlation, no
  preference for rooms or doors, and no leaving the footprint for a smoke break.
- Near-duplicate places (POI point plus building polygon) are not merged.
- Home choice is uniform over all homes and workplace choice is uniform over the pool, so commute
  lengths do not follow a realistic distribution. Only favorites use distance decay. In suburban,
  only 3 gyms exist, so gym-goers bunch up.
- Friend meetings are the only social activity. Scheduled meetings are dropped silently when an
  agent is busy.

**Sensing**
- **Iid error per fix.** GPS and adtech errors are independent between consecutive fixes. Real GPS
  error is strongly autocorrelated (multipath bias persists for minutes indoors), which makes
  averaging within a dwell look better here than it would on real data. Calibration noise is also
  drawn per event, whereas real mis-calibration tends to be per device, OS or source.
- The GPS indoor ×2.5 inflation applies to every stay, including parks. Outliers are a flat 2 %,
  50–300 m, with no bias toward stays or urban canyons.
- The GPS stationary throttle is an oracle that switches exactly at true arrival + 10 min and back
  exactly at departure. Real OS geofencing reacts with a delay, and departure detection lags too.
- **CDR sector model**: square jittered tower grid, three equal 120° sectors, a fixed radius
  0.9·spacing, and a centroid and covariance of a uniform wedge. There is no terrain or propagation
  model, and no timing-advance or distance ring. Handover is an independent 20 % coin flip per event
  (in reality it depends on load and position, and ping-pong is correlated in time). The
  second-nearest tower's sector can lie beyond R from the true point. The reported ellipse is
  identical for every event in an area.
- **Adtech model**: the source mix is iid per ping, not per app or session. The stale-fix model is
  a 4 % Bernoulli copy of the previous ping, i.e. within-id and short-range in time. Real SDK caches
  can re-report fixes hours old from a different place. The 3 IP hubs are fixed ENU offsets shared
  by both areas, with a reported σ unrelated to the hub distance. There are no outages, and no
  device or user change within an id.
- No duplicated or out-of-order events, no timestamp jitter, no clock skew and no coordinate
  rounding.
- cdr and adtech timestamps in urban and suburban come from the same realization (shifted 1 h).
  Differences between the two areas under these profiles reflect geography only, which helps the
  comparison but halves the effective sampling diversity.

**Truth**
- `truth_dwells` includes every stay ≥ 60 s, including 60 s artefact stops at home, and does not
  split overnight stays at midnight. Evaluation filters by `min_duration_s`.
- `n_events` counts all events in the interval, whatever their accuracy, so the "observable" recall
  is an upper bound for cdr and adtech.
- The wander offset is folded into `true_lat/true_lon`. Centroid error against the truth-dwell
  `lat/lon` (the place centroid) therefore includes up to one place radius of real spread.

---

## 8. Public datasets (summary)

Full details, licenses and caveats are in [`public_datasets.md`](public_datasets.md) (section
"Ingestion"). Each `data/public/<name>/` has the same layout as a synthetic set: master
`events.parquet`, optional `truth_*.parquet`, and `meta.json`. GeoLife, STAGA and PoL have no
accuracy fields. They get a nominal circular σ = 8 m and unperturbed positions (true = observed),
so shapes 2 and 3 carry no information on them.

| Dir | Source | Events | Ids | Truth | Gap p50 / p95 | Notes |
|---|---|---:|---:|---|---|---|
| `staga` | STAGA GPS + travel diary (CC BY 4.0) | 122,808 | 1 | 689 diary stops (min 59 s) | 10 s / 110 s | Dates shifted by the authors to 2000; anonymized coordinates with jumps |
| `pol` | PoL staypoint benchmark, clean variant (CC BY-NC-ND, local use only) | 21,600,997 | 500 | 65,873 staypoints (min 60 s) | 60 s / 60 s | `place_id` is per agent |
| `pol_50` | first 50 PoL ids by string sort (`public.build --subset pol pol_50 50`) | 2,160,099 | 50 | 7,088 | as pol | Used for evaluating slow algorithms. **Its `meta.json` stats are inherited from `pol`** (see note) |
| `geolife_top20` | GeoLife 1.3, the 20 users with the most points | 14,768,085 | 20 | none | 2 s / 5 s | Realism and scaling only |
| `geolife_all` | GeoLife 1.3, all users | 24,876,977 | 182 | none | 2 s / 5 s | |
| `foursquare_nyc` | TSMC2014 NYC check-ins | 227,428 | 1,083 | 227,428 visits (`truth_visits`) | 4.9 h / 5.0 d | Venue position plus synthetic adtech-style gps/wifi/cell error, honest, no calibration noise |

Note on `pol_50`: `subset()` copies the source `meta.json` and only adds `subset_of` and `n_ids`,
so its `stats` block (events 21,600,997, ids 500) and `truth_dwells` (65,873) describe the full
`pol`. The real counts, taken from the parquet files, are the ones in the table above.
