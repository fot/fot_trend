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


def filter_outliers(datavals, maxoutlierstddev=5):
    return np.abs(datavals - np.mean(datavals)) <= (np.std(datavals) * maxoutlierstddev)


def lowpass(vals, samp=32.8, cutoff=0.1):
    B, A = butter(1, cutoff / (samp / 2.), btype='low') # 1st order Butterworth low-pass
    return lfilter(B, A, vals, axis=0)


def get_report_boundaries(startyear=1999, endyear=None, trim_future=True):
    
    startyear = int(startyear)
    
    if not endyear:
        endyear = int(DateTime().date[:4])
    else:
        endyear = int(endyear)
    
    pairs = [[str(y) + 'Aug01 at 00:00:00.000', str(y+1) + 'Feb01 at 00:00:00.000'] 
             for y in range(startyear, endyear + 1, 1)] 
             
    reportboundaries = [pairs[i][j] for i in range(len(pairs)) for j in range(2)]

    if trim_future:
        while DateTime(reportboundaries[-1]).secs > DateTime().secs:
            reportboundaries.pop(-1)

    return reportboundaries


def get_month_boundaries(startyear=1999, endyear=None, trim_start_mission=True, trim_future=True):
    
    startyear = int(startyear)
    
    if not endyear:
        endyear = int(DateTime().date[:4])
    else:
        endyear = int(endyear)
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan']

    boundaries = [str(y) + months[m] + '01 at 00:00:00.000' 
                  for y in range(startyear, endyear + 1, 1) for m in range(12)] 

    if trim_future:
        while DateTime(boundaries[-1]).secs > DateTime().secs:
            boundaries.pop(-1)
    
    if trim_start_mission:
        while DateTime(boundaries[0]).secs < DateTime('1999:204:00:00:00').secs:
            boundaries.pop(0)
        
    return boundaries



def calc_monthly_stats(t, dmin, dmean, dmax):
    """ Calculate monthly stats

    :param t: Time numpy array
    :param dmin: Minimum statistical data in a numpy array
    :param dmean: Mean statistical data in a numpy array
    :param dmax: Maximum statistical data in a numpy array

    Note: dmin, dmean, and dmax are meant to include the statistical data from the ska archive,
    such as the 5min or daily stats. 

    This method can use raw values instead of statistical values if the same raw values are passed
    for each the dmin, dmean, and dmax positional arguments.

    """
        
    numindspermonth = np.int(30*3600*24 / np.nanmedian(np.diff(t)))
    nummonths = np.int(len(t) / numindspermonth)
    mtimes = [np.nanmean(t[n:n + numindspermonth]) for n in 
               range((nummonths - 1) * numindspermonth, 0, -numindspermonth)]
    mtimes.reverse()
    mtimes = np.array(mtimes)
    
    mmaxes = [np.nanmax(dmax[n:n + numindspermonth]) for n in 
               range((nummonths - 1) * numindspermonth, 0, -numindspermonth)]
    mmaxes.reverse()
    mmaxes = np.array(mmaxes)
    
    mmeans = [np.nanmean(dmean[n:n + numindspermonth]) for n in 
               range((nummonths - 1) * numindspermonth, 0, -numindspermonth)]
    mmeans.reverse()
    mmeans = np.array(mmeans)
    
    mmins = [np.nanmin(dmin[n:n + numindspermonth]) for n in 
               range((nummonths - 1) * numindspermonth, 0, -numindspermonth)] 
    mmins.reverse()    
    mmins = np.array(mmins)

    return (mtimes, mmaxes, mmeans, mmins)


def calc_daily_stats(t, dmin, dmean, dmax):
    """ Calculate daily stats

    :param t: Time numpy array
    :param dmin: Minimum statistical data in a numpy array
    :param dmean: Mean statistical data in a numpy array
    :param dmax: Maximum statistical data in a numpy array

    Note: dmin, dmean, and dmax are meant to include the 5min statistical data from the ska
    archive. 

    This method can use raw values instead of statistical values if the same raw values are passed
    for each the dmin, dmean, and dmax positional arguments.

    """

    daystart = DateTime(DateTime(t[0]).date[:8] + ':00:00:00.000').secs
    daystop = DateTime(DateTime(t[-1]).date[:8] + ':00:00:00.000').secs

    daysecs = 3600.* 24.
    days = np.arange(daystart, daystop + daysecs, daysecs)
    # daybins = np.digitize(t, bins=days)
    # b = np.bincount(daybins -1)
    # c = np.hstack((0, np.cumsum(b)))
    # ind = [(k1, k2-1) for k1, k2 in zip(c[:-1], c[1:])]
    ind = digitizebins(t, days)

    daymins = np.array([np.nanmin(dmin[i[0]:(i[-1]+1)]) if i[-1] - i[0] > 0 else np.nan for i in ind])
    daymeans = np.array([np.nanmean(dmean[i[0]:(i[-1]+1)]) if i[-1] - i[0] > 0 else np.nan for i in ind])
    daymaxes = np.array([np.nanmax(dmax[i[0]:(i[-1]+1)]) if i[-1] - i[0] > 0 else np.nan for i in ind])

    
    # If there is a partial day of data at the end of the input data array(s), then there may be one less
    # day of data than the number of days in the `days` array.
    if len(days) > len(daymins):
        days = days[:-1]

    return days, daymins, daymeans, daymaxes


def calc_5min_stats(t, dmin, dmean, dmax):
    """ Calculate 5min stats

    :param t: Time numpy array
    :param dmin: Minimum statistical data in a numpy array
    :param dmean: Mean statistical data in a numpy array
    :param dmax: Maximum statistical data in a numpy array

    Note: dmin, dmean, and dmax are meant to include the 5min statistical data from the ska
    archive. 

    This method can use raw values instead of statistical values if the same raw values are passed
    for each the dmin, dmean, and dmax positional arguments.

    """

    bounds = np.arange(DateTime(t[0]).secs, DateTime(t[-1]).secs + 328, 328)
    ind = digitizebins(t, bounds)

    daymins = np.array([np.nanmin(dmin[i[0]:(i[-1]+1)]) if i[-1] - i[0] > 0 else np.nan for i in ind])
    daymeans = np.array([np.nanmean(dmean[i[0]:(i[-1]+1)]) if i[-1] - i[0] > 0 else np.nan for i in ind])
    daymaxes = np.array([np.nanmax(dmax[i[0]:(i[-1]+1)]) if i[-1] - i[0] > 0 else np.nan for i in ind])

    meantimes = np.diff(bounds)/2 + bounds[:-1]
    return meantimes, daymins, daymeans, daymaxes


def digitizebins(data, bins):
    """ Calculate indices to binned data

    :param data: 1d data array to be divided up into bins
    :param bins: 1d array of bin boundaries

    :returns: array of start and stop indices

    Note: at this point, data needs to be sequential. I intend on generalizing this function for
    non sequential data, however that will come as an enhancement in the future.

    """
    databins = np.digitize(data, bins=bins)
    b = np.bincount(databins - 1)
    c = np.hstack((0, np.cumsum(b)))
    return np.array([(k1, k2-1) for k1, k2 in zip(c[:-1], c[1:])])



