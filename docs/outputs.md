# Output data structures

Produced by `dwells.pol.pipeline.analyze()` (and the batch CLI `python -m dwells.pol.build`,
which writes one parquet file per table to `results/pol/<dataset>/` plus `meta.json`
with the parameters used). Schemas are defined in `src/dwells/pol/schema.py`.

Conventions:
* All times are UTC.
* Rings are closed lists of `[lon, lat]` pairs (GeoJSON order), directly usable by
  deck.gl `PolygonLayer`.
* Hour arrays are indexed by UTC hour 0–23; weekday 0 = Monday; `weekday_hour` index =
  weekday × 24 + hour.
* Ids are global within one run: `visit_id` and `place_id` are unique across identifiers.

Join keys:

```
events.visit_id   -> visits.visit_id
events.place_id   -> places.place_id
visits.place_id   -> places.place_id
transitions.from_place / to_place, next_places.from_place / to_place -> places.place_id
presence.place_id -> places.place_id
anomalies.visit_id -> visits.visit_id
```

## events: one row per input event

| column | meaning |
|---|---|
| id, timestamp, lat, lon | the input report |
| event_idx | position of the report within its identifier (0-based, time order) |
| location_source | the optional provenance label as supplied (`gps`, `network`, `fused`, `ip`, `user_provided`, `unknown`); null when the input had no such column |
| sigma_m | its 1σ error in meters after normalizing the accuracy fields (or the assumed value for its location_source / the default) |
| coarse | true when too vague to locate a stay (see the data contract) |
| visit_id, place_id | the visit/place it belongs to (null = traveling, an outlier, or an unassigned coarse fix) |

## visits: one row per dwell

| column | meaning |
|---|---|
| visit_id, place_id, id | keys |
| start_ts, end_ts, duration_s | estimated stay (midpoint of the arrival/departure windows) |
| first_event_ts, last_event_ts | first and last report assigned to the visit |
| arrive_after, arrive_by | arrival happened in this window (last precise report elsewhere, first report here); arrive_after is null at the start of the data |
| depart_after, depart_by | departure window; depart_by is null at the end of the data |
| n_events, n_informative | reports in the visit / of those, precise (non-coarse) ones |
| lat, lon, sigma_major, sigma_minor, azimuth | visit center (inverse-variance fused) and its 1σ uncertainty ellipse (m, bearing of the major axis) |
| extent_m | RMS distance of the visit's precise reports from its center (activity footprint) |
| weekday, start_hour | of start_ts (UTC) |
| prev_place_id, next_place_id | place of the previous / next visit of the identifier |
| confidence | detection confidence 0–100 = 100 × (0.5 × (1 − e^(−n_informative/5)) + 0.25 / (1 + sigma_major/50) + 0.25 × observed share of the estimated span) |
| anomalies | list of anomaly kinds flagged for this visit |
| ellipse_ring | 95% ellipse of the visit center |
| hull_ring | convex hull of the visit's precise reports (a small circle if fewer than 3 distinct positions) |

## places: one row per recurring place of an identifier

| column | meaning |
|---|---|
| place_id, id, lat, lon | center: inverse-variance fused over visits |
| sigma_major, sigma_minor, azimuth | 1σ uncertainty of the center (fused visit uncertainty plus the scatter of visit centers) |
| extent_m | RMS distance of all precise member reports from the center |
| n_visits, n_days, n_events, total_dwell_s, median_dwell_s, first_seen, last_seen | visit statistics |
| time_share | share of the identifier's total dwell time spent here |
| hour_presence_s | 24 values: seconds present per UTC hour of day (over all visits) |
| hour_events | 24 values: member reports per UTC hour |
| weekday_visits | 7 values: visits starting on each weekday |
| weekday_hour_presence_s | 168 values: seconds present per weekday × hour |
| regularity | mean of day coverage, hour focus and repeat (0–1; see stats.py and the analyst guide) |
| score, score_evidence, score_precision, score_regularity | pattern confidence 0–100 = 100 × (0.35 evidence + 0.15 precision + 0.50 regularity), components 0–1 |
| rank_duration, rank_score, rank_visits | 1 = top place of the identifier by total time / score / visits |
| label | `primary` (most time), `secondary` (second most, on ≥ 3 days), else null. Usually home and work, but no local-time inference is made |
| address, place_name, place_category, geocode_distance_m, geocode_source | reverse geocode of the center (nearest OSM address and named feature within 150 m / 60 m) |
| ellipse_ring, hull_ring | 95% center ellipse; convex hull of all precise member reports |

## transitions and next_places

`transitions`: one row per (identifier, from_place, to_place) with consecutive visits at
different places.
* `n` = moves observed.
* `prob` = n / departures from from_place.
* `prob_low` = Wilson 95% lower bound.
* `median_travel_s`.
* `typical_depart_hour` (UTC mode).
* `last_ts`.

`next_places`: the top 5 transitions per from_place.
* `rank`, `prob`, `prob_low`.
* `n_from` = departures observed from that place (the evidence behind the probability).

## presence

`(id, date, hour, place_id) -> seconds`: seconds at each place in each UTC clock hour.
It feeds the presence calendar and the timeline.

## anomalies

One row per flagged visit:

| kind | rule |
|---|---|
| unusual_hour | place has ≥ 5 visits and < 10% of its other visits arrived within ±1 h of this arrival |
| unusually_long / unusually_short | place has ≥ 6 visits and log-duration is > 3.5 robust standard deviations (MAD) from the place median |
| unusual_weekday | place has ≥ 8 visits and this is its only visit on that weekday |
| rare_place | the only visit to a place, for an identifier with ≥ 7 active days |

`severity` (0–1) orders the flags; `detail` is a readable explanation.

## quality: one row per identifier

| column | meaning |
|---|---|
| shape | input shape (1 centerpoint, 2 accuracy radius, 3 ellipse) |
| n_events, first_ts, last_ts, span_days, active_days, events_per_active_day | volume |
| hour_coverage | share of clock hours in the observed span with at least one report |
| gap_p50_s, gap_p95_s, gap_max_s | time between consecutive reports |
| sigma_p50_m, sigma_p95_m, coarse_frac | accuracy |
| duplicate_frac | same time and same position as the previous report |
| stale_frac | same position as the previous report at a new time |
| speed_outlier_frac | implausible jumps (> 60 m/s and farther than 3× the combined error) |
| visits, places, events_in_visits_frac | outcome summary |
| source_coverage | share of reports with a location_source other than unknown (null without the column) |
| source_mismatch_frac | share of reports whose label contradicts the stated accuracy: gps with 1σ > 100 m, ip with 1σ < 1 km (null for centerpoint-only data or without the column) |
