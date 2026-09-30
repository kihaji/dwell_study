Overview
--------

This project is going to be research into dwell, sometimes called stays or staypoints, pattern of life analysis. We will be creating and evaluating implementations of multiple existing algorithms and libraries, with the goal to find the positives and negatives of each that will ultimately be included in another application for pattern of life analysis.

Input Data
----------

Input data will come from a variety of sources; gps records, call data records, or adtech based data, but we have a cleaning pipeline already in place that normalizes the structure of the incoming datasets. There are 3 shapes of data that will be encountered. We call each row of data, or measurement, an "event". Across all 3 shapes of data there are common fields:
* id - This is a unique identifier associated with a device. A data set may have N identifier, with each identier having multiple events
* lat - The latitude centerpoint of the measurement. This will be in Decimal Degrees. All spatial data is in WGS84 
* lon - The longitude of the centerpoint of the measurement. This will be in Decimal Degrees. All Spatial data is in WGS84
* timestamp - This is the UTC/Zulu datetime the event occured. 

The 3 dataset shapes are then
* Dataset 1 - This is centerpoint only measurements with no error values.
* Dataset 2 - This adds the field horizontalAccuracy, which is the accuracy Circular Error of Probability in meters. It is considered to be 50% 
* Dataset 3 - This adds the fields:
        * semiMajor and semiMinor which are the semi-major and semi-minor of the axes for an error ellipse in meters. 
        * azimuth - this is a degrees based azimuth of the ellipse rotation. This will be between 0-180 
        * probability - this is the probability factor for the error ellipse. This will be a float between 0.0 - .99

The data measurements are not always uniform in time. There can be large gaps as some of the datasets are driven by user activity, not regular samples. 


Existing Algorithms and libraries
---------------------------------

I have been working on this for awhile, and have a number of project doing work on this.
* /home/kihaji/projects/ellipses
* /home/kihaji/projects/new_ellipses
* /home/kihaji/projects/analytic_app   All 3 of these have some version of dwell analytic in them in various forms through multiple git branches
* https://github.com/gefgu/fastmob  a python library that specializes in mobility analysis
* https://github.com/mie-lab/trackintel  another python library that specializes in mobility analysis
* https://github.com/scikit-mobility/scikit-mobility  as scikit python library specializing in mobility analysis

We do not have to limit ourselves to these, but they are a place to start, and we can bring in additional libraries to test, or implementations to create if there are other ideas out there with no implementation


Analytic Questions
------------------

A non-exhaustive list of analytic questions we will want to answer with our analysis are:
* Where does an identifer dwell
* Is there a pattern to where they dwell
* Is there a pattern to their movement after a dwell
* Data on the dwells; when they start, when they end, how often they return, how many events happen there.
* Are there other identifiers that are also dwelling there with them (colocation) or dwell at different times
* What are the probable next locations an identifier goes after a dwell, IE top-N next locations

Technical Environment
---------------------

Right now, the only limitation we will have is using Python 3.12 or later, and using UV to manage the virtual environment. We will eventually want to be able to feed both Plotly Dash based tables and graphs, as well as a deckgl-dash (https://github.com/kihaji/deckgl-dash) map for visualization. We would also like to potentially use this on a spark cluster with access to millions of events per identifier, so scaling should be a factor of the evaluation and work.

You are in a git repo, so make judicious use of branches and git trees if you want to research something that may not work out. Dont delete things, commit to a branch and return to main without merging and add an explanation in the final output report what the branch did and why it failed.


Current Limitations
-------------------

We have no existing data, so we will need to generate multiple sets of data with various volumes and patterns if none can be found. All work will be done local, so while spark is consideration, we do not have a method of testing.

Outputs
-------

For this initial version, individual scripts that can be run against the same generated or curated datasets, each one implementing a libraries usage, or a new impementation. There should be a harness to test each one and evaluate them based on scalability, accuracy, extensibility, tradeoffs, best application, and ease of use. As many markdown files with summaries of the algorithms and implementations, including references to external sources if needed, should be produced as well so I can read through and evaluate the correctness of our work and outputs.
