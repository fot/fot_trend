import numpy as np
import matplotlib.pyplot as plt

from Chandra.Time import DateTime
from Ska.engarchive import fetch_eng as fetch


def find_span_indices(dval):
    dval = list(dval)
    dval.insert(0, False) # Prepend to make indices line up
    idata = np.array(dval, dtype=type(1))
    d = np.diff(idata)

    starts = d == 1
    stops = d == -1

    starts = list(starts)
    stops = list(stops)

    if idata[-1] == 1:
        stops.insert(-1, True)

    starts = np.where(starts)[0]
    stops = np.where(stops)[0]
    
    return zip(starts, stops)


def find_time_spans(times, inds):
    return [(DateTime(times[ind[0]]).date, DateTime(times[ind[1]]).date) for ind in inds]


def filter_stats(msid, t1, t2, dt=0.256, statemsid=None, eqstate=None, rangemsid=None, inrange=None):
    msids = [msid,]
    if statemsid:
        msids.append(statemsid)
    if rangemsid:
        msids.append(rangemsid)

    if len(msids) == 1:
        data = fetch.Msid(msid, t1, t2, stat=None)
        data.interpolate(dt=dt)
        datavals = data.vals
        datatimes = data.times
        keep = np.ones(len(data.times)) == 1
    
    if len(msids) > 1:
        data = fetch.Msidset(msids, t1, t2, stat=None)
        data.interpolate(dt=dt)
        keep = np.ones(len(data.times)) == 1

        if statemsid:
            data[statemsid].vals = np.array([s.strip().lower() for s in data[statemsid].vals])
            statecheck = data[statemsid].vals == eqstate.lower()
            keep = keep & statecheck

        if rangemsid:
            rangecheck1 = data[rangemsid].vals >= inrange[0]
            rangecheck2 = data[rangemsid].vals < inrange[1]
            keep = keep & rangecheck1 & rangecheck2

        datavals = data[msid].vals
        datavals[~keep] = np.nan
        datatimes = data[msid].times

    return mstats(datavals, datatimes, stat='daily')


def mstats(datavals, datatimes, stat='daily'):

    dtypes = [('t1', '|S21'), ('t2', '|S21'), ('min', np.float64), ('mean', np.float64), 
              ('max', np.float64), ('mintime', '|S21'), ('maxtime', '|S21'),
              ('stddev', np.float64), ('counts', np.int32), ('active', object)]

    datavals = np.array(datavals, dtype=type(np.float()))

    if 'daily' in stat.lower():
        timesec = 24 * 3600
    elif 'monthly' in stat.lower():
        timesec = 24 * 3600 * 30

    stats = []
    startday = DateTime(datatimes[0]).date[:8]
    endsec = DateTime(DateTime(datatimes[-1]).date[:8] + ':00:00:00.000').secs + 24 * 3600

    t1 = datatimes[0]
    t2 = DateTime(startday + ':00:00:00.000').secs + timesec
    while t2 <= endsec:
        ind1 = datatimes >= t1
        ind2 = datatimes < t2
        ind = ind1 & ind2

        if all(np.isnan(datavals[ind])):
            minind = 0
            maxind = 0
            meanval = np.nan
            stddev = np.nan
            selectedtimespans = []
        else:
            minind = np.nanargmin(datavals[ind])
            maxind = np.nanargmax(datavals[ind])
            meanval = np.nanmean(datavals[ind])
            stddev = np.nanstd(datavals[ind])
            selectedtimespans = find_time_spans(datatimes[ind], find_span_indices(~np.isnan(datavals[ind])))
            if len(selectedtimespans) == 0:
                selectedtimespans = []

        minval = datavals[ind][minind]
        maxval = datavals[ind][maxind]
        mintime = DateTime(datatimes[ind][minind]).date
        maxtime = DateTime(datatimes[ind][maxind]).date
        counts = sum(~np.isnan(datavals[ind]))

        t1str = DateTime(t1).date
        t2str = DateTime(t2).date
        stats.append((t1str, t2str, minval, meanval, maxval, mintime, maxtime, stddev, counts, 
                      selectedtimespans))

        t1 = t2
        t2 = DateTime(t2).secs + 24 * 3600

    return np.array(stats, dtype=dtypes)


def write_csv(msid, stats):
    with open('{}_stats.csv'.format(msid.upper()), 'w') as fid:
        fid.write('t1sec, t2sec, t1, t2, min, mean, max, mintime, maxtime, stddev, counts, active start and stop times ----> \n')
        for row in stats:
            fid.write('{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {},'.format(
                    DateTime(row['t1']).secs, DateTime(row['t2']).secs, row['t1'], row['t2'], 
                    row['min'], row['mean'], row['max'], row['mintime'], row['maxtime'], 
                    row['stddev'], row['counts']))
            for pair in row['active']:
                fid.write('{}, {},,'.format(pair[0], pair[1]))

            fid.write('\n')


