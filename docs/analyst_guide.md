# Pattern-of-life analyst guide

This guide explains every view, number and term in the Pattern of Life app. It is
written for analysts, not mathematicians. When a number comes from a formula, the guide
says what it means in pattern-of-life terms and gives a plain-language rule of thumb.
The ⓘ icons in the app show short versions of the same text.

**The question the app answers.** For one identifier (one device): *where does it stay,
when is it there, where does it go next, and how sure can we be?*

## Core ideas

### Location report
One row of input data: a device was somewhere at a time. Reports are imperfect:

- **Accuracy (error).** Every report has some error. A phone's GPS might be off by 5–20 m.
  A position estimated from cell towers can be off by hundreds of meters to kilometers.
  When the data says how accurate each report is, the app uses that; when it doesn't, it
  assumes a typical value for the source.
- **Coarse report.** A report so vague (worse than about 1 km, and much worse than the
  device's usual reports) that it cannot tell one building from another. Coarse reports
  never start, end or move a visit. They can only confirm that the device was "somewhere
  around here".
- **Gaps.** Devices report irregularly. Some report every few seconds, others a few times
  a day. Silence does not mean the device moved.

### Visit (dwell)
<!-- help:visit -->
One continuous stay in one spot: from arrival to departure. A visit is detected when
consecutive reports agree that the device stayed put. "Agree" means they are close to
each other *given their stated accuracy*, so imprecise reports are allowed to scatter
more. A single stray report does not end a visit. A visit ends after three reports
elsewhere in a row, or when the device was clearly away for at least 15 minutes. The
minimum visit length (default 15 minutes) is the analyst's definition of a dwell.

### Arrival and departure windows
<!-- help:arrival_window -->
Data rarely captures the exact moment of arrival. We know the device was elsewhere at the
last report before the visit, and here at the first report of the visit. The arrival
happened somewhere in between. The app shows that window ("arrived between 07:42 and
08:15") and uses its middle as the estimated start. Departure works the same way. Wide
windows mean the timing is uncertain, which is common with sparse adtech or cell data.

### Place
<!-- help:place -->
A recurring location, such as a home, a workplace or a favorite café, made of one or more
visits. Visits are grouped into the same place when their positions agree within their
uncertainty, so a precise GPS visit and a vague cell-tower visit can belong to the same
place if they are compatible. Places are numbered by total time spent: **P1** is where the
identifier spends the most time.

**Primary / secondary anchor.** P1 is labelled the *primary anchor*, and the second-most-
visited place (if seen on 3 or more days) the *secondary anchor*. These are usually home
and work. The app does not claim that, because all times are UTC and it does not infer
local day or night.

### All times are UTC
<!-- help:utc -->
Every time, hour-of-day and day-of-week view uses UTC (Zulu), not local time. Convert
mentally when reading routines; for example, 13:00 UTC is 08:00 in US Central daylight
time.

## Map
<!-- help:map -->
The map shows the selected identifier:

- **Grey dots:** reports not assigned to any visit (traveling, stray reports).
- **Colored dots:** reports belonging to a visit. One color per place, consistent across
  every view (P1 red, P2 green, P3 blue, ...).
- **Large circles:** place centers. Bigger = more time spent there. Labels show rank,
  anchor tag and the nearest address or named place when one was found.
- Click any place, visit outline or report to focus the whole app on that place.

Use the overlay checkboxes to choose how a place's spatial extent is drawn.

### Uncertainty ellipse
<!-- help:uncertainty_ellipse -->
The outline that contains the place's true center with 95% confidence. Think of it as
"we are 95% sure the center of activity is inside this shape." It shrinks as more precise
reports accumulate. For GPS it is often only a few meters across; for cell data it can
span several blocks. It is not the size of the building, only how well its center is
known.

### Convex hull
<!-- help:convex_hull -->
The tightest outline around every precise report made during the visits (like stretching
a rubber band around the dots). It shows where the device was actually seen while staying
there: the footprint of activity, for example the whole office campus or the house and its
yard. A single stray report can enlarge it, so read it together with the ellipse.

### Member events and links
<!-- help:member_events -->
Shows exactly which reports belong to the selected visit (or place): each member report
is linked to the visit center by a thin line, and drawn with a circle the size of its own
stated accuracy. Use it to check *why* a visit was detected and whether it rests on a few
reports or many.

### Next-place arcs
<!-- help:next_place_arcs -->
When a place is selected, arcs fan out to where the identifier usually went next. Thicker
arcs are more likely moves. The label shows the observed share and, in brackets, a cautious
lower bound (see *Probability "at least"*). Arcs only reflect departures actually observed:
a move that happened while the device was silent is invisible.

## Selected place card

### Pattern confidence (score)
<!-- help:score -->
A 0–100 summary of how strongly the data supports that this place is a real, recurring
part of the routine. It combines three ingredients, each shown separately:

- **Evidence (35%):** how much data there is.
- **Precision (15%):** how exactly we know where it is.
- **Regularity (50%):** how routinely it is visited.

Rules of thumb: 70+ = a strong routine anchor; 40–70 = a real but less regular place;
below 40 = a one-off or weakly supported place.

### Evidence
<!-- help:score_evidence -->
Grows with the number of precise reports and the total hours spent there, and levels off
(diminishing returns): about 30 reports or about a day of total presence already gives most
of the credit. Low evidence means: be careful, few observations.

### Precision
<!-- help:score_precision -->
How tightly the place center is pinned down: 1.0 means within a few meters, 0.5 means
about ±50 m, lower values mean hundreds of meters. Cell-only data will always score low
here even for real places.

### Regularity
<!-- help:score_regularity -->
How routine the visits are. It is the average of three simple checks:

- **Day coverage:** on what share of the days we observed the device was it seen here?
  Home is often near 100%.
- **Hour focus:** does it tend to be here at the same hours? A place visited only
  09:00–17:00 scores higher than one visited at random times. Technically this is one
  minus the normalized "entropy" of the hour profile, a standard measure of how spread
  out a distribution is.
- **Repeat visits:** 1 visit = 0, 3 visits ≈ 0.5, 10 visits ≈ 0.95.

### Visit confidence
<!-- help:visit_confidence -->
The same idea for a single visit (0–100), about detection rather than routine: did enough
precise reports support it, is its center precise, and did reports cover most of the
estimated stay (rather than just its start)? Low confidence visits are best treated as
"possible stops".

## Top 10 places (rankings)
<!-- help:rankings -->
Ordered lists of the identifier's places by four criteria:

- **Total time:** where it spends the most time overall.
- **Pattern confidence:** the most reliable routine places (see score).
- **Repeat visits:** places it returns to most often, even briefly (a coffee shop visited
  daily can outrank a long weekly visit).
- **Distinct days:** places seen on the most different days.

Click a row to select the place.

## Movement graph
<!-- help:graph -->
A network diagram of the routine. Each circle is a place (bigger = more time spent), and
each arrow a move from one place to the next that was observed at least once. Arrow
thickness and labels show how often the identifier took that route out of the place (e.g.
"60%" = 6 of 10 departures went there). Labels appear on routes seen at least twice. Hubs
with many arrows are central to the routine; isolated circles are rare or one-off places.
Click a circle to select that place.

## Tables

### Places table
<!-- help:places_table -->
Every place with its key numbers: visits, distinct days, total and median visit length,
share of all dwell time, number of reports, score and its components, how precisely the
center is known (±m), the activity extent, and when it was first and last seen. Sort by
any column (click the header) or filter (type in the cell under the header, e.g. `>5` in
"visits").

### Visits table
<!-- help:visits_table -->
Every visit to the selected place (tick "all places" for the whole history):

- **start / end / duration:** the estimated stay.
- **arrived between / left between:** the arrival and departure windows.
- **reports / precise:** how many reports support it, and how many of those were precise.
- **confidence:** visit confidence (0–100).
- **center ±m:** how precisely the visit's center is known (1σ, about a 68% radius).
- **from / to:** the previous and next place.
- **flags:** any anomalies.

Click a row to highlight that visit on the map and the radar.

### Next places
<!-- help:next_places -->
Where the identifier went after leaving the selected place, best first:

- **probability:** share of observed departures.
- **at least:** cautious lower bound.
- **departures seen:** the evidence behind it.

"70% (at least 35%), 4 departures" means a likely but thinly supported pattern.
"70% (at least 62%), 120 departures" is a solid one.

### Probability "at least"
<!-- help:prob_low -->
Raw percentages can mislead when based on few observations: "100%" from a single departure
is not a reliable pattern. The "at least" value is the Wilson 95% lower bound, a standard
statistical adjustment that answers: *what is the lowest the true probability could
plausibly be, given how little we have seen?* It rises toward the raw percentage as
evidence accumulates, so rank by it when you want confident predictions.

## Charts

### When there (radar chart)
<!-- help:radar -->
A 24-hour clock for the selected place. Each spoke is an hour (UTC). The filled shape
shows how many hours in total the identifier spent there at each hour of day, over all
visits. A home typically fills the night hours; a workplace the daytime block. The thin
line shows when reports were received there (scaled), which helps tell genuine absence from
silent periods. When a visit is selected, the dashed line overlays that visit's hours, so
you can see whether it fits the usual pattern.

### Visits per day of week (heatmap)
<!-- help:weekday_heatmap -->
Rows are the top places, columns the days of the week (UTC), and darker cells mean more
visits started on that day. It reveals weekly routines, for example a gym on Mon/Wed/Fri,
a workplace only on weekdays, or a place of worship on Sundays.

### Weekday × hour heatmap (selected place)
<!-- help:weekday_hour_heatmap -->
For the selected place: rows are days of week, columns hours (UTC), and color shows total
hours present. It is the most detailed answer to "when is this identifier likely to be
here?"; hot cells are the best times to expect it there.

### Daily timeline
<!-- help:timeline -->
One row per day; colored bars show when the identifier was at which place (colors match
the map). White space means traveling or no data. Repeating vertical patterns are the
routine; days that look different stand out. Click a bar to select that visit.

### Presence calendar
<!-- help:presence_calendar -->
A compact grid of days (rows) by hours (columns, UTC), each cell colored by the place where
the identifier spent most of that hour. Long-term routines, holidays, trips and changes of
routine are visible at a glance. Grey cells are lesser places; empty cells have no
visit (traveling or no data).

## Anomalies
<!-- help:anomalies -->
Visits that break the identifier's own routine, ordered by severity (0–1; the slider hides
weak flags):

- **unusual hour:** arrived at an hour when it almost never arrives at that place (fewer
  than 1 in 10 of its other visits arrived within an hour of that time).
- **unusually long / short:** much longer or shorter than its typical visit there. Visit
  lengths are compared on a logarithmic scale using a "robust" spread measure (median
  absolute deviation), so a few extreme visits do not distort what "typical" means.
- **unusual weekday:** the only visit on that weekday to a regularly visited place.
- **rare place:** a place visited once, by an identifier with an established routine.

Anomalies are leads for review, not conclusions. Short returns home between errands, for
example, are often flagged as "unusually short".

## Data quality
<!-- help:data_quality -->
How much to trust everything else. Key indicators:

- **reports / active day:** fewer than about 10 makes most short stops invisible.
- **hours with data:** share of hours with at least one report.
- **typical gap / 95th percentile gap:** how long the device usually stays silent.
- **typical accuracy and coarse reports:** how precise the data is. Mostly-coarse data
  gives few, vague places.
- **duplicates / repeated positions:** may indicate cached or replayed locations, which
  can stretch visits.
- **implausible jumps:** reports that would require impossible speeds. A high share means
  bad positions that can create false visits.
- **reports in visits:** the share of reports that ended up in a visit.

The charts show reports per day (all vs. inside visits) and the distribution of stated
accuracy (log scale: 10 m, 100 m, 1 km...).

## Tuning panel
<!-- help:tuning -->
Re-runs detection for the current identifier with different settings so you can see how
sensitive the conclusions are. Nothing saved is changed; a red banner shows when you are
looking at a live re-run.

- **place radius:** how much positions may wander within one place. Larger merges nearby
  stops.
- **max gap between reports:** the longest silence a stay can bridge.
- **min dwell:** your definition of a dwell.
- **gate probability:** how strict "close enough given the accuracy" is. Higher means more
  tolerant.
- **misses to end a stay / away time to end a stay:** how quickly a departure is accepted.
- **coarse fix threshold:** reports vaguer than this cannot locate a stay.

If a conclusion only appears with unusual settings, it is fragile.

## Glossary of shorthand

| term | meaning |
|---|---|
| 1σ ("one sigma") | a typical error size: about 68% of reports fall within this distance of the truth |
| CEP50 | circular error probable: radius containing 50% of true positions |
| 95% ellipse | outline containing the truth with 95% confidence |
| UTC | Coordinated Universal Time (Zulu); no daylight saving |
| F1 | detection accuracy score used during tuning, 0–1 (combines missed and false visits) |
| entropy | a standard measure of how spread out a distribution is (low = concentrated) |
| MAD | median absolute deviation, a spread measure that ignores outliers |
| Wilson lower bound | cautious estimate of a probability that accounts for small sample sizes |
