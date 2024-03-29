import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import sys
from os.path import expanduser

# from Chandra.Time import DateTime
from cxotime import CxoTime as DateTime
from cheta import fetch_eng

home = expanduser("~")
sys.path.append(home + '/AXAFLIB/pylimmon/')
import pylimmon

addthispath = home + '/AXAFLIB/fot_bad_intervals/'
sys.path.append(addthispath)
import fot_bad_intervals

# np.random.seed(42)


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
    
    return list(zip(starts, stops))


def find_time_spans(times, inds):
    return [(DateTime(times[ind[0]]).date, DateTime(times[ind[1]]).date) for ind in inds]


def filter_stats(msid, t1, t2, dt=0.256, statemsid=None, eqstate=None, rangemsid=None, inrange=None, time_pad=3600):
    """ Filter MSIDs based on other state or other numeric MSIDs

    """
    msids = [msid,]
    if statemsid:
        msids.append(statemsid)
    if rangemsid:
        msids.append(rangemsid)

    if len(msids) == 1:
        data = fetch_eng.Msid(msid, t1, t2, stat=None)
        data.interpolate(dt=dt)
        datavals = data.vals
        datatimes = data.times
        keep = np.ones(len(data.times)) == 1
    
    if len(msids) > 1:
        data = fetch_eng.Msidset(msids, t1, t2, stat=None)
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

    # Back up a half day to provide buffer to account for leap seconds, then find the end of each day
    # daystart = DateTime(DateTime(t[0]).date[:8] + ':00:00:00.000').secs - 3600 * 12
    # daystop = DateTime(DateTime(t[-1]).date[:8] + ':00:00:00.000').secs - 3600 * 12

    # daysecs = 3600.* 24.
    # middays = np.arange(daystart, daystop + daysecs, daysecs)

    # try:
    #     dates = [DateTime(d).date.split(':') for d in middays]
    #     days = [DateTime('{:s}:{:03d}:00:00:00'.format(date[0], int(date[1])+1)).secs for date in dates]
    #     days = np.array(days)
    # except:
    #     import code
    #     vars = globals().copy()
    #     vars.update(locals())
    #     shell = code.InteractiveConsole(vars)
    #     shell.interact()


    day_grid = np.arange(t[0], t[-1], 3600 * 12)
    day_set = list(set([date[:8] for date in DateTime(day_grid).date]))
    days = np.array(DateTime(day_set).secs)
    days.sort()
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


def isnumeric(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def filter_out_more_bad_data(datatimes):
    ''' Filter out additional data
    '''

    def filterout(datatimes, keep, t1, t2):
        bad1 = datatimes > DateTime(t1).secs
        bad2 = datatimes < DateTime(t2).secs
        bad = bad1 & bad2
        allbad = ~keep | bad
        keep = ~allbad
        return keep

    t1 = '2015:336:00:00:00'
    t2 = '2015:341:00:00:00'
    keep = np.array([True] * len(datatimes))
    return filterout(datatimes, keep, t1, t2)


class MSIDTrend(object):
    """ Create an object to make linear predictions for telemetry.

    ---------------------------------------------------------------------------
    This function is written to provide some tools to make basic predictions
    for telemetry using a linear curve fit through monthly data.


    Author: Matthew Dahmer


    ---------------------------------------------------------------------------
    Requires one input argument:

    msid: This must be a valid msid name in the engineering telemetry archive


    ---------------------------------------------------------------------------
    Includes these optional keyword arguments:

    tstart: This must be a date that is a valid input argument for the
            Chandra.Time.DateTime object. If this is None, then the default
            action for the Chandra.Time.DateTime object is to use the current
            time, which would not make sense.

    tstop: Similar to tstart in that this must be a valid input argument for
           the Chandra.Time.DateTime object, however a value of None would
           make sense here as this is just the current time. If None is used
           then the telemetry queried from the engineering archive will include
           the most recent values available.

    trendmonths: This is the number of months counting backwards to use in
                 creating a trending prediction. Each month is assumed to be
                 30 days, so 'monthly' data are grabbed in 30 day chunks.

    numstddev: This is the number of standard deviations to use as a factor of
               safety when making a prediction of when the data will reach a
               limit. The distance of the monthly data used to generate a
               linear curve fit from that curve fit is used to generate the
               standard deviation. For more details on the definition of this
               standard deviation, please see the documentation for the Numpy
               std() function.


    ---------------------------------------------------------------------------
    Creates an object with these attributes:

    msid: This is the msid used to create the object.

    numstddev: See documentation above.

    trendmonths: See documentation above.

    tstart: See documentation above.

    tstop: See documentation above.

    safetylimits: The database limits are replaced with the G_LIMMON limits if
                  the G_LIMMON limits are more permissive (which would only be
                  the case if the database limits are outdated). Only those
                  individual limits that are more permissive are replaced, not
                  the entire set. This object contains these updated limits.

    telem: Fetch data from the engineering telemetry archive, using the daily
           stats. Monthly telemetry is calculated and added to this object.

    get_polyfit_line: Returns the coefficients for the linear curve fit through
                    the last N months of data. The data is expected to be
                    either the monthly maximum, monthly minimum, or monthly
                    mean. Also returns the standard deviation of the data about
                    this curve fit.

    get_prediction: Return the prediction at one or more times for the requested
                   data.

    get_limit_intercept: Return the date when the data is expected to reach a
                       limit. See the documentation in the code below for more
                       details.
    """

    def __init__(self, msid, tstart='2000:001:00:00:00', tstop=None,
                 trendmonths=36, numstddev=2, removeoutliers=True,
                 maxoutlierstddev=5, filter_monthly_data=False,
                 remove_safing_actions=True):

        self.msid = msid
        self.tstart = DateTime(tstart).date

        if tstop == None:
            self.tstop = DateTime().date
        else:
            self.tstop = DateTime(tstop).date

        self.trendmonths = trendmonths
        self.numstddev = numstddev
        self.removeoutliers = removeoutliers
        self.maxoutlierstddev = maxoutlierstddev
        self.filter_monthly_data = filter_monthly_data
        self.remove_safing_actions = remove_safing_actions

        self.telem = self._get_monthly_telemetry()
        self.safetylimits = pylimmon.get_safety_limits(msid)

        db = pylimmon.open_sqlite_file()
        cursor = db.cursor()
        cursor.execute('''SELECT a.msid, a.setkey, a.default_set, a.warning_low,
                          a.caution_low, a.caution_high, a.warning_high FROM limits AS a
                          WHERE a.mlmenable=1 AND a.setkey = a.default_set AND a.msid = ?
                          AND a.modversion = (SELECT MAX(b.modversion) FROM limits AS b
                          WHERE a.msid = b.msid and a.setkey = b.setkey)''', [msid.lower(), ])
        lims = cursor.fetchone()
        self.trendinglimits = {'warning_low': lims[3], 'caution_low': lims[4], 'caution_high': lims[5],
                               'warning_high': lims[6]}

    def filter_outliers(self, datavals):
        keep = np.abs(datavals - np.mean(datavals)) <= (np.std(datavals) *
                                                        self.maxoutlierstddev)
        return keep

    def _get_monthly_telemetry(self):
        """ Retrieve the telemetry and calculate the 30 day stats

        30 day stats start at the most recent time point, so there are likely
        to be a remainder of daily datapoints not used at the begining of the
        dataset
        """

        if '_wide' in self.msid.lower():
            msid = self.msid[:8]
        else:
            msid = self.msid
        telem = fetch_eng.Msid(msid, self.tstart, self.tstop, stat='daily')

        if (msid.lower()[0] == '4') or ('ohr' in msid.lower()) or (msid.lower()[:2] == 'oo'):
            keep = fot_bad_intervals.get_keep_ind(telem.times, 'tel', msid.lower(), 'daily')
        else:
            keep = fot_bad_intervals.get_keep_ind(telem.times, None, msid.lower(), 'daily')
            
        if isnumeric(msid[0]):
            if int(msid[0]) < 4:
                keep2 = filter_out_more_bad_data(telem.times)
                keep = keep & keep2

        before_after_pad = (24*3600, 24*3600)
        keep_not_safing = fot_bad_intervals.filter_safing_actions(telem.times, 'daily', 
            safe_modes=True, nsm_modes=True, pad=before_after_pad, transitions_only=False, tpad=None)
        keep = keep & keep_not_safing

        telem.times = telem.times[keep]
        telem.vals = telem.vals[keep]
        telem.maxes = telem.maxes[keep]
        telem.mins = telem.mins[keep]
        telem.means = telem.means[keep]


        # if '4oavobat' in msid.lower():
        #     ind = telem.times > DateTime('2014:342:16:29:14.500').secs
        #     telem.vals[ind] = 50 + 2 * (telem.vals[ind] - 50)
        #     telem.maxes[ind] = 50 + 2 * (telem.maxes[ind] - 50)
        #     telem.mins[ind] = 50 + 2 * (telem.mins[ind] - 50)
        #     telem.means[ind] = 50 + 2 * (telem.means[ind] - 50)
        #     print('Fixed 4oavobat calibration data in ska engineering archive!!!\n')

        if self.removeoutliers:
            keepmean = self.filter_outliers(telem.means)
            keepmax = self.filter_outliers(telem.maxes)
            keepmin = self.filter_outliers(telem.mins)

        else:
            keepmean = np.array([True] * len(telem.means))
            keepmax = np.array([True] * len(telem.maxes))
            keepmin = np.array([True] * len(telem.mins))

        # Save this for future access outside of this function, you need to
        # merge the keep arrays since they all share a common time array.
        keep = keepmean & keepmax & keepmin
        telem.keep = keep


        # Calculate the mean time value for each 30 day period going backwards.
        #
        # Monthly values are calculated going backwards because it is more
        # important to have a full month at the end than at the beginning since
        # recent data is more likely to reflect future trends than older data.
        #
        # Note that the hours used for each daily time value are 12pm.
        #
        # Determine monthly min, max, and mean values.
        #
        # Data is reported in chronological order
        #
        days = len(telem.times[keep])
        telem.monthlytimes = [np.mean(telem.times[keep][n:n + 30]) for n in
                              range(days - 30, 0, -30)]
        telem.monthlytimes.reverse()
        telem.monthlytimes = np.array(telem.monthlytimes)

        telem.monthlymaxes = [np.max(telem.maxes[keep][n:n + 30]) for n in
                              range(days - 30, 0, -30)]
        telem.monthlymaxes.reverse()
        telem.monthlymaxes = np.array(telem.monthlymaxes)

        telem.monthlymins = [np.min(telem.mins[keep][n:n + 30]) for n in
                             range(days - 30, 0, -30)]
        telem.monthlymins.reverse()
        telem.monthlymins = np.array(telem.monthlymins)

        telem.monthlymeans = [np.mean(np.double(telem.means[keep][n:n + 30]))
                              for n in range(days - 30, 0, -30)]
        telem.monthlymeans.reverse()
        telem.monthlymeans = np.array(telem.monthlymeans)

        if self.filter_monthly_data:
            keep_max = fot_bad_intervals.filter_outliers(telem.monthlymaxes)
            keep_min = fot_bad_intervals.filter_outliers(telem.monthlymins)
            keep_mean = fot_bad_intervals.filter_outliers(telem.monthlymeans)

            keep = keep_max & keep_min & keep_mean
            telem.montlytimes = telem.monthlytimes[keep]
            telem.monthlymaxes = telem.monthlymaxes[keep]
            telem.monthlymins = telem.monthlymins[keep]
            telem.monthlymeans = telem.monthlymeans[keep]

        return telem

    def get_polyfit_line(self, data):
        """ Return the linear curve fit and standard deviation for the data.

        Return the coefficients for the linear curve fit through
        the last N months of data. The data is expected to be either the
        monthly maximum, monthly minimum, or monthly mean.Also return the
        standard deviation of the data about this curve fit.

        The number of months used is specified using the trendmonths input
        argument.
        """

        # Select the last N months of data
        datarange = data[-self.trendmonths:]
        timerange = self.telem.monthlytimes[-self.trendmonths:]

        # Calculate the coefficients
        p = np.polyfit(timerange, datarange, 1)

        # Calculate the standard deviation of the fit.
        #
        # This is the standard deviation of the distance of the datapoints
        # from the curve fit (line).This standard deviation is intended to be
        # used as a simplistic measure of the variation of the data about the
        # fit line. Some multiple of this number may be used as a
        # "factor of safetly" when making predictions (such as when the
        # maximum value may reach a warning high limit).
        stddev = np.std(datarange - np.polyval(p, timerange))

        return (p, stddev)


    def get_prediction(self, date, maxminmean='max'):
        """ Return the prediction at one or more times for the requested data.

        The date passed into this function can be either a
        Chandra.Time.DateTime object, or a compatible input argument for a
        Chandra.time.DateTime class.

        The maxminmean input argument is case insensitive but must be one of
        three valid strings:
            min
            max
            mean
        """

        # Ensure the date is a date object (either a single or multiple
        # element object) and convert it to seconds
        date = DateTime(date).secs

        # Return the coeficients and associated standard deviation for the
        # requested linear curve fit.
        if maxminmean.lower() == 'max':

            p, stddev = self.get_polyfit_line(self.telem.monthlymaxes)

        elif maxminmean.lower() == 'min':

            p, stddev = self.get_polyfit_line(self.telem.monthlymins)

        elif maxminmean.lower() == 'mean':

            p, stddev = self.get_polyfit_line(self.telem.monthlymeans)

        return np.polyval(p, date)


    def get_limit_intercept(self, thresholdtype, limittype='safety'):
        """ Return the date when the data is expected to reach a limit.

        Valid values for thresholdtype are:
            warning_low
            caution_low
            caution_high
            warning_high
        """

        # Ensure the thresholdtype is lower case, since all such keys to the
        # safetylimits dict are lower case.
        thresholdtype = thresholdtype.lower()

        def getdate(self, p, stddev, thresholdtype, limittype):
            """ Return the date when the specified threshold is reached.

            p and stddev are the linear curve fit parameters and standard
            deviation output from get_polyfit_line.

            thresholdtype has three valid input strings:
                warning_low
                caution_low
                caution_high
                warning_high

            limittype has two valid input strings:
                safety
                trending

            There are two ways one can incorporate a safety factor, one is to
            modify the threshold, the other is to modify the linear curve fit.
            In this case the threshold is modified by either subtracting or
            adding a multiple of the standard deviation depending on the type of
            threshold the user is interested in.

            If there is a time when the threshold is reached, then a date
            string is returned, otherwise None is returned. Keep in mind that
            if None is input into a Chandra.Time.DateTime object, it returns
            the current time.
            """

            # Get the threshold value and include the appropriate safety factor
            if limittype.lower() == 'trending':
                if thresholdtype == 'warning_high' or \
                                thresholdtype == 'caution_high':

                    threshold = (self.trendinglimits[thresholdtype] - stddev *
                                 self.numstddev)
                else:
                    threshold = (self.trendinglimits[thresholdtype] + stddev *
                                 self.numstddev)
            else:
                if thresholdtype == 'warning_high' or \
                                thresholdtype == 'caution_high':

                    threshold = (self.safetylimits[thresholdtype] - stddev *
                                 self.numstddev)
                else:
                    threshold = (self.safetylimits[thresholdtype] + stddev *
                                 self.numstddev)

            # Calculate the date at which the modified threshold is reached
            seconds = (threshold - p[1]) / p[0]
            if seconds < DateTime('3000:001:00:00:00').secs:
                crossdate = DateTime(seconds).date

            else:
                crossdate = '3000:001:00:00:00'

            return crossdate

        if thresholdtype == 'warning_high' or thresholdtype == 'caution_high':
            # If an upper limit threshold is used, then fit the line to the
            # monthly maximum data.

            p, stddev = self.get_polyfit_line(self.telem.monthlymaxes)

            # If an upper limit threshold is used, then there is no cross date
            # if the slope is negative.
            if p[0] > 0:

                crossdate = getdate(self, p, stddev, thresholdtype, limittype)

            else:

                crossdate = None
                print(('Slope for %s is %e, so no %s limit cross' %
                      (self.msid, p[0], thresholdtype)))

        if thresholdtype == 'warning_low' or thresholdtype == 'caution_low':
            # If a lower limit threshold is used, then fit the line to the
            # monthly minimum data.

            p, stddev = self.get_polyfit_line(self.telem.monthlymins)

            # If a lower limit threshold is used, then there is no cross date
            # if the slope is positive.
            if p[0] < 0:

                crossdate = getdate(self, p, stddev, thresholdtype, limittype)

            else:

                crossdate = None
                print(('Slope for %s is %e, so no %s limit cross' %
                      (self.msid, p[0], thresholdtype)))

        return crossdate


    def get_resampled_polyfit_line(self, data, drop_fraction=0.5, num_passes=50):
        """ Return the median linear curve fit and standard deviation for the data
            using resampled monthly data.

        Return the coefficients for the linear curve fit through
        the last N months of data. The data is expected to be either the
        monthly maximum, monthly minimum, or monthly mean.Also return the
        standard deviation of the data about this curve fit.

        The number of months used is specified using the trendmonths input
        argument.
        """

        # Select the last N months of data
        datarange = data[-self.trendmonths:]
        timerange = self.telem.monthlytimes[-self.trendmonths:]

        coef = []
        stddev = []
        # Calculate the coefficients
        for n in range(num_passes):
            random_ind = np.random.randint(0, len(timerange), int((1 - drop_fraction) * len(timerange)))
            p = np.polyfit(timerange[random_ind], datarange[random_ind], 1)
            coef.append(tuple(p))
            std = np.std(datarange[random_ind] - np.polyval(p, timerange[random_ind]))

            stddev.append(std)

        # coef = np.array(tuple(coef), dtype=[('intercept', float), ('slope', float)])
        coef = np.array(coef)
        stddev = np.array(stddev)

        # self.resampled_polyfit_coefficients = coef
        # self.resampled_standard_deviations = stddev

        median_p = (np.median(coef[:,0]), np.median(coef[:,1]))
        median_stddev = np.median(stddev)

        return coef, stddev, median_p, median_stddev


    def get_resampled_prediction(self, date, maxminmean='max', drop_fraction=0.5, num_passes=50):
        """ Return the prediction at one or more times for the requested data.

        The date passed into this function can be either a
        Chandra.Time.DateTime object, or a compatible input argument for a
        Chandra.time.DateTime class.

        The maxminmean input argument is case insensitive but must be one of
        three valid strings:
            min
            max
            mean
        """

        # Ensure the date is a date object (either a single or multiple
        # element object) and convert it to seconds
        date = DateTime(date).secs

        # Return the coeficients and associated standard deviation for the
        # requested linear curve fit.
        if maxminmean.lower() == 'max':

            p, stddev, median_p, median_stddev = self.get_resampled_polyfit_line(self.telem.monthlymaxes, 
                drop_fraction=drop_fraction, num_passes=num_passes)

        elif maxminmean.lower() == 'min':

            p, stddev, median_p, median_stddev = self.get_resampled_polyfit_line(self.telem.monthlymins, 
                drop_fraction=drop_fraction, num_passes=num_passes)

        elif maxminmean.lower() == 'mean':

            p, stddev, median_p, median_stddev = self.get_resampled_polyfit_line(self.telem.monthlymeans, 
                drop_fraction=drop_fraction, num_passes=num_passes)


        return np.polyval(p, date)


    def get_resampled_limit_intercept(self, thresholdtype, limittype='safety', drop_fraction=0.5, num_passes=50):
        """ Return the date when the data is expected to reach a limit.

        Valid values for thresholdtype are:
            warning_low
            caution_low
            caution_high
            warning_high
        """

        # Ensure the thresholdtype is lower case, since all such keys to the
        # safetylimits dict are lower case.
        thresholdtype = thresholdtype.lower()

        def getdate(self, p, stddev, thresholdtype, limittype):
            """ Return the date when the specified threshold is reached.

            p and stddev are the linear curve fit parameters and standard
            deviation output from get_polyfit_line.

            thresholdtype has three valid input strings:
                warning_low
                caution_low
                caution_high
                warning_high

            limittype has two valid input strings:
                safety
                trending

            There are two ways one can incorporate a safety factor, one is to
            modify the threshold, the other is to modify the linear curve fit.
            In this case the threshold is modified by either subtracting or
            adding a multiple of the standard deviation depending on the type of
            threshold the user is interested in.

            If there is a time when the threshold is reached, then a date
            string is returned, otherwise None is returned. Keep in mind that
            if None is input into a Chandra.Time.DateTime object, it returns
            the current time.
            """

            # Get the threshold value and include the appropriate safety factor
            if limittype.lower() == 'trending':
                if thresholdtype == 'warning_high' or \
                                thresholdtype == 'caution_high':

                    threshold = (self.trendinglimits[thresholdtype] - stddev *
                                 self.numstddev)
                else:
                    threshold = (self.trendinglimits[thresholdtype] + stddev *
                                 self.numstddev)
            else:
                if thresholdtype == 'warning_high' or \
                                thresholdtype == 'caution_high':

                    threshold = (self.safetylimits[thresholdtype] - stddev *
                                 self.numstddev)
                else:
                    threshold = (self.safetylimits[thresholdtype] + stddev *
                                 self.numstddev)

            # Calculate the date at which the modified threshold is reached
            seconds = (threshold - p[1]) / p[0]
            if (seconds < DateTime('3000:001:00:00:00').secs) & (seconds > DateTime('1000:001:00:00:00').secs):
                crossdate = DateTime(seconds).date

            else:
                crossdate = '3000:001:00:00:00'

            return crossdate


        if thresholdtype == 'warning_high' or thresholdtype == 'caution_high':
            # If an upper limit threshold is used, then fit the line to the
            # monthly maximum data.

            ps, stddevs, median_p, median_stddev = self.get_resampled_polyfit_line(self.telem.monthlymaxes, 
                drop_fraction=drop_fraction, num_passes=num_passes)


            # import code
            # code.interact(local=dict(globals(), **locals())) 

            # If an upper limit threshold is used, then there is no cross date
            # if the slope is negative.
            if median_p[0] > 0:

                crossdates = [getdate(self, p, stddev, thresholdtype, limittype) for p, stddev in zip(ps, stddevs)]
                crossdate = DateTime(np.median([DateTime(d).secs for d in crossdates if d is not None])).date

            else:

                crossdate = None
                print(('Slope for %s is %e, so no %s limit cross' %
                      (self.msid, median_p[0], thresholdtype)))

        if thresholdtype == 'warning_low' or thresholdtype == 'caution_low':
            # If a lower limit threshold is used, then fit the line to the
            # monthly minimum data.

            ps, stddevs, median_p, median_stddev = self.get_resampled_polyfit_line(self.telem.monthlymins, 
                drop_fraction=drop_fraction, num_passes=num_passes)

            # If a lower limit threshold is used, then there is no cross date
            # if the slope is positive.
            if median_p[0] < 0:

                crossdates = [getdate(self, p, stddev, thresholdtype, limittype) for p, stddev in zip(ps, stddevs)]
                crossdate = DateTime(np.median([DateTime(d).secs for d in crossdates if d is not None])).date

            else:

                crossdate = None
                print(('Slope for %s is %e, so no %s limit cross' %
                      (self.msid, median_p[0], thresholdtype)))

        return crossdate


def DoubleMADsFromMedian(x, zeromadaction="warn"):
    """ Median Absolute Deviation Calculation    

    :param x: 1 dimenional data array
    :param zeromadaction: determines the action in the event of an MAD of zero, anything other
        than 'warn' will throw an exception

    Code is derived from examples shown here:
    http://eurekastatistics.com/using-the-median-absolute-deviation-to-find-outliers/
    """
    
    def DoubleMAD(x, zeromadaction="warn"):
        """ Core Median Absolute Deviation Calculation    
        """

        m = np.median(x)
        absdev = np.abs(x - m)
        leftmad = np.median(np.abs(x[x<=m]))
        rightmad = np.median(np.abs(x[x>=m]))
        if (leftmad == 0) or (rightmad == 0):
            if zeromadaction.lower() == 'warn':
                print('Median absolute deviation is zero, this may cause problems.')
            else:
                raise ValueError('Median absolute deviation is zero, this may cause problems.')
        return leftmad, rightmad
    
    twosidedmad = DoubleMAD(x, zeromadaction)
    m = np.median(x)
    xmad = np.ones(len(x)) * twosidedmad[0]
    xmad[x > m] = twosidedmad[1]
    maddistance = np.abs(x - m) / xmad
    maddistance[x==m] = 0
    return maddistance

