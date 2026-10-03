"""Short in-app help texts (info icons). docs/analyst_guide.md explains each in depth;
tests/test_help.py checks every key here has a matching section there."""

HELP = {
    "map": "Map of the selected identifier. Grey dots are raw location reports; colored dots belong to a "
           "visit (same color = same place). Toggle the extent overlays to see where the place is "
           "(uncertainty ellipse) and where the device was actually seen there (convex hull).",
    "uncertainty_ellipse": "The area that contains the true center of the place with 95% confidence. Small "
                           "for precise GPS, large for cell-tower data.",
    "convex_hull": "The smallest outline enclosing every precise location report made during the visits: "
                   "the footprint of activity, not a confidence area.",
    "member_events": "The individual location reports assigned to a visit, with lines to the visit center. "
                     "Error circles show each report's own stated accuracy.",
    "next_place_arcs": "Arcs from the selected place to where the identifier usually went next. Thicker = "
                       "more likely. Based only on observed departures.",
    "place": "A recurring location: one or more visits that sit at the same spot, merged when their "
             "positions agree within their uncertainty.",
    "visit": "One continuous stay (dwell) at a place: from estimated arrival to estimated departure.",
    "arrival_window": "The arrival happened between the last report elsewhere and the first report here; "
                      "the estimated start is the middle of that window.",
    "score": "Pattern confidence, 0-100: how strongly the data supports that this place is a real, "
             "recurring part of the routine. Built from evidence (35%), precision (15%) and regularity (50%).",
    "score_evidence": "How much data supports the place: number of precise reports and total hours spent.",
    "score_precision": "How tightly the place center is pinned down (100% = within a few meters).",
    "score_regularity": "How routine the visits are: seen on many of the observed days, at consistent hours, "
                        "and visited repeatedly.",
    "visit_confidence": "Detection confidence of one visit, 0-100: enough precise reports, a precise center, "
                        "and reports covering most of the stay.",
    "rankings": "Top places by total time, by pattern confidence, by number of visits, or by number of "
                "distinct days. Click a row to select the place.",
    "graph": "Directed graph of movement between places. Node size = time spent; arrow width = how often "
             "that move happened; labels show the probability of that next step.",
    "places_table": "All places of the identifier with their key statistics. Sort or filter any column; "
                    "select a row to focus everything on that place.",
    "visits_table": "Every visit to the selected place (or all places) with arrival/departure times, "
                    "reports and confidence.",
    "radar": "When the identifier is usually at the selected place: time present per hour of day (UTC), "
             "over all visits. The dashed trace overlays the selected visit.",
    "weekday_heatmap": "Visits per day of week for the top places; darker = more visits on that weekday.",
    "weekday_hour_heatmap": "Time present at the selected place for each weekday and hour (UTC).",
    "timeline": "Each day as a row, colored bars where the identifier was at a place; blank means "
                "traveling or no data.",
    "presence_calendar": "Grid of days by hours colored by the place occupied, to spot routines and breaks "
                         "in routine at a glance.",
    "next_places": "Where the identifier went after leaving the selected place, with probability and a "
                   "cautious lower bound that is low when based on few departures.",
    "prob_low": "Cautious (95% lower-bound) estimate of a probability. With few observations it is much "
                "lower than the raw share, signalling weak evidence.",
    "anomalies": "Visits that break the usual routine: unusual hour, unusual length, an unusual weekday, or a "
                 "rarely visited place.",
    "data_quality": "How much and how good the input data is for this identifier. Low reporting rates, large "
                    "errors, many implausible jumps or location-source labels that conflict with the stated "
                    "accuracy reduce the reliability of everything else.",
    "tuning": "Re-run detection for this identifier with different settings to see how sensitive the "
              "results are. Does not change the saved results.",
    "same_place_tolerance": "How far (in combined error widths) a report may sit from the place center and "
                            "still count as being there. 99% = up to about 3 error widths; higher = more tolerant. "
                            "A report is treated as elsewhere only when there is less than (100% - tolerance) "
                            "chance it came from the place.",
    "vague_report_cutoff": "Reports less precise than this (1-sigma error, m) AND more than 3x less precise than the "
                           "device's usual reports cannot start, end, extend or move a stay. They are kept and can "
                           "still be attached to a visit. 1 km at 1-sigma ~ 1.2 km CEP50 ~ 2.4 km at 95%.",
    "utc": "All times and hour/day breakdowns are in UTC.",
}
