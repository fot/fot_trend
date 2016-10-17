
import sys
from os.path import expanduser
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.font_manager as font_manager
from ipywidgets import interact, interactive, Select, Text, Checkbox, Tab, Box, HBox, Button
from IPython.display import clear_output, display, HTML
import warnings

from Chandra.Time import DateTime
from Ska.engarchive import fetch_eng as fetch

from plot_cxctime_custom import *
home = expanduser("~")
sys.path.append(home + '/AXAFLIB/pylimmon/')
import pylimmon
sys.path.append(home + '/AXAFLIB/fot_bad_intervals/')
from fot_bad_intervals import get_keep_ind as keepind
lightfont = font_manager.FontProperties(weight='light')


def thin_dataset(data, num=2000, kind='both'):
    ''' Reduce dataset size.
    
    :param kind: Classification of returned indices per block, 'max', 'min', or 'both'
    
    num is the number of blocks
    n is the block length
    '''
    if num > len(data) * 0.5:
        warnings.warn('Number of requestd blocks, {}, is more than half the input data length, {}'.format(num, len(data)))
        return np.arange(len(data)), np.arange(len(data)), 1

    kind = kind.lower()
    n = len(data) / num
    blocks = int(len(data) / float(n)) # number of whole blocks
    padlen = len(data) - blocks * n # number of numeric elements in partial block
    if padlen > 0:
        blocks = blocks + 1
        b = np.hstack((data, np.array([np.nan,] * (n - padlen)))) # fill in with nans
    else:
        b = np.copy(data)
    chunks = np.reshape(b,(blocks, n))
    maxinds = np.nanargmax(chunks, axis=1)
    mininds = np.nanargmin(chunks, axis=1)

    # nanargmax and nanargmin not ignoring nans for some reason, so use the builtin to force it
    maxinds[-1] = np.where(chunks[-1] == max(chunks[-1]))[0][0]
    mininds[-1] = np.where(chunks[-1] == min(chunks[-1]))[0][0]

    indmax = list(maxinds + np.arange(0, len(b), n))
    indmin = list(mininds + np.arange(0, len(b), n))
    
    if 'both' in kind:
        ind = list(set(indmin + indmax))
    elif 'max' in kind:
        ind = indmax
    elif 'min' in kind:
        ind = indmin
    #       extrema indices, block start indices,     block length
    return  np.sort(ind),    np.arange(0, len(b), n), n


def plot_msid_interactive(msid='aacccdpt', group='sc', tstart='2001:001', tstop=None, 
              stat='daily', plot_warning_low=False, plot_caution_low=False, 
              plot_caution_high=True, plot_warning_high=True, remove_bads=True):

    def add_limit_lines():
        if limitquery:
            plotdate_dates = cxctime2plotdate(dates)
            if plot_warning_low:
                _ = ax.step(plotdate_dates, wL, where='post', color='r', zorder=4)
            if plot_caution_low:
                _ = ax.step(plotdate_dates, cL, where='post', color='#FFA500', zorder=6)
            if plot_caution_high:
                _ = ax.step(plotdate_dates, ch, where='post', color='#FFA500', zorder=5)
            if plot_warning_high:
                _ = ax.step(plotdate_dates, wh, where='post', color='r', zorder=3)
            
    def update_plot_data(fig, ax):
            
        ticklocs, _, _ = plot_cxctime(data.times[good], data.vals[good], fmt='-', 
                                         fig=fig, ax=ax, color='#555555', zorder=2)

        if 'none' not in statstr:

            if len(data.vals) > 4000:
                indmax, indmaxstart, blocklen = thin_dataset(data.maxes[good], kind='max')
                indmin, indminstart, blocklen = thin_dataset(data.mins[good], kind='min')
                times_max = data.cxctimes[good][indmaxstart]
                maxvals = data.maxes[good][indmax]
                times_min = data.cxctimes[good][indminstart]
                minvals = data.mins[good][indmin]

                times = np.sort(np.concatenate((times_max, times_min)))
                maxvals = np.interp(times, times_max, maxvals)
                minvals = np.interp(times, times_min, minvals)

            else:
                times = data.times[good]
                maxvals = data.maxes[good]
                minvals = data.mins[good]
            
            times = np.repeat(times, 2)[1:]
            maxes =  np.repeat(maxvals, 2)[:-1]
            mins = np.repeat(minvals, 2)[:-1]

            _ = ax.fill_between(times, mins, maxes, color='#aaaaaa', zorder=1)
#             ax.plot(times_max, maxvals, 'r')
#             ax.plot(times_min, minvals, 'r')
#             ax.plot(data.cxctimes[good], data.maxes[good], 'b')
#             ax.plot(data.cxctimes[good], data.mins[good], 'b')

        add_limit_lines()
        
    def get_msid_limits(msid):
        limdict = pylimmon.get_limits(msid)
        enabled = np.array(limdict['limsets'][0]['mlmenable']) == 1
        last_enabled = np.where(enabled)[0]
        if len(last_enabled) > 0:
            if last_enabled[-1] < (len(enabled)-1):
                enabled[last_enabled[-1] + 1] = True

        wL = np.array(limdict['limsets'][0]['warning_low'])[enabled]
        cL = np.array(limdict['limsets'][0]['caution_low'])[enabled]
        ch = np.array(limdict['limsets'][0]['caution_high'])[enabled]
        wh = np.array(limdict['limsets'][0]['warning_high'])[enabled]
        dates =  np.array(limdict['limsets'][0]['times'])[enabled]
        return wL, cL, ch, wh, dates

        
    if 'none' in str(stat).lower():
        stat = None
        
    statstr = str(stat).lower()
    msid = msid.lower()

    tstart = DateTime(tstart).secs
    if 'none' in unicode(tstop).lower():
        tstop = DateTime().secs
    else:
        tstop = DateTime(tstop).secs
    data = fetch.Msid(msid, tstart, tstop, stat=stat)
    data.cxctimes = cxctime2plotdate(data.times)
    
    if remove_bads:
        good = keepind(data.times, group, msid, stat)
    else:
        good = np.array([True] * len(data.times))
    good = np.where(good)[0]

    if hasattr(data, 'tdb'):
        title = '{}: {}'.format(msid.upper(), data.tdb.technical_name)
    else:
        title = msid.upper()
    
    if data.unit:
        units = data.unit
    else:
        units = ''

    wide = ['OOBTHR08', 'OOBTHR09', 'OOBTHR10', 'OOBTHR11', 'OOBTHR12', 'OOBTHR13', 'OOBTHR14',
            'OOBTHR15', 'OOBTHR17', 'OOBTHR18', 'OOBTHR19', 'OOBTHR20', 'OOBTHR21', 'OOBTHR22',
            'OOBTHR23', 'OOBTHR24', 'OOBTHR25', 'OOBTHR26', 'OOBTHR27', 'OOBTHR28', 'OOBTHR29',
            'OOBTHR30', 'OOBTHR31', 'OOBTHR33', 'OOBTHR34', 'OOBTHR35', 'OOBTHR36', 'OOBTHR37',
            'OOBTHR38', 'OOBTHR39', 'OOBTHR40', 'OOBTHR41', 'OOBTHR44', 'OOBTHR45', 'OOBTHR46',
            'OOBTHR49', 'OOBTHR50', 'OOBTHR51', 'OOBTHR52', 'OOBTHR53', 'OOBTHR54', '4OAVHRMT',
            '4OAVOBAT']
    limitquery = False
    if plot_warning_low or plot_caution_low or plot_caution_high or plot_warning_high:
        try:

            wL, cL, ch, wh, dates = get_msid_limits(msid)

            if msid.upper() in wide:
                 wL2, cL2, ch2, wh2, dates2 = get_msid_limits(msid.upper() + '_WIDE')
                 wL = np.concatenate((wL, wL2))
                 cL = np.concatenate((cL, cL2))
                 ch = np.concatenate((ch, ch2))
                 wh = np.concatenate((wh, wh2))
                 dates = np.concatenate((dates, dates2))

            limitquery = True
        except IndexError:
            pass
    
    if statstr == 'daily':
        xtickfontsize = 20
        fig_h_start = 0.23
        fig_w_start = 0.15
    else:
        xtickfontsize = 14
        fig_h_start = 0.3
        fig_w_start = 0.15
    fig_height = 1. - fig_h_start - 0.1
    fig_width = 1. - fig_w_start - 0.1

    plt.close(plt.gcf())
    
        
    fig = plt.figure(facecolor=[1,1,1],figsize=(14,8))
    fig.set_label(msid.upper())
    ax = fig.add_axes([fig_w_start, fig_h_start, fig_width, fig_height])
    ax.hold(True)
    _ = ax.set_xlim(cxctime2plotdate(DateTime([tstart, tstop]).secs))

    _ = ax.grid(True)
    _ = ax.set_ylabel(units, fontsize=22, fontproperties=lightfont)
    _ = ax.set_xlabel('Time (Seconds)', fontsize=22, fontproperties=lightfont)
    _ = ax.set_title(title, fontsize=30, y=1.03)
    _ = ax.tick_params(axis='both', which='major', labelsize=20)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation_mode='anchor', ha='right')

    
#     ax.callbacks.connect('xlim_changed', update_plot_data)
    

    update_plot_data(fig, ax)




def gen_figure(msids, group_name):
    def select_next_msid(junk):
        current_msid = msid_select.value
        options = msid_select.options
        i = options.index(current_msid)
        if i < len(options) - 1:
            msid_select.value=options[i + 1]

    dummyfig = plt.figure(facecolor=[1,1,1],figsize=(14,8))
    msid_select = Select(description='MSID:',options=msids, visible=True, padding=4)
    button_next_msid = Button(description='Next', padding=4)
    button_next_msid.on_click(select_next_msid)
    msid_select_group = HBox(children=[msid_select, button_next_msid])

    latest = DateTime().date
    t1 = Text(description='Start Date:', value='2000:001:00:00:00.000', visible=True, padding=4)
    t2 = Text(description='Stop Date:', value=latest, visible=True, padding=4)
    time_select = Box(children=[t1, t2])
    page1 = HBox(children=[msid_select_group, time_select])

    wL = Checkbox(description='Plot Warning Low', value=False, visible=True, padding=4)
    cL = Checkbox(description='Plot Caution Low', value=False, visible=True, padding=4)
    cH = Checkbox(description='Plot Caution High', value=True, visible=True, padding=4)
    wH = Checkbox(description='Plot Warning High', value=True, visible=True, padding=4)
    low_select = Box(children=[wL, cL])
    high_select = Box(children=[wH, cH])
    page2 = HBox(children=[high_select, low_select])

    stat_select = Select(description='Stat:',options=('daily', '5min', 'None'), visible=True, padding=4)
    filter_bads = Checkbox(description='Filter Bad Times:', value=True, visible=True, padding=4)
    group_select = Select(description='Group Name:',options=['sc', 'tel', 'isim'], visible=True, value=group_name, padding=4)
    left_select = Box(children=[stat_select, filter_bads])
    page3 = HBox(children=[left_select, group_select], description='Misc.')

    q = interactive(plot_msid_interactive, msid=msid_select, group=group_select, tstart=t1, tstop=t2, stat=stat_select,
                 plot_warning_low=wL, plot_caution_low=cL, plot_caution_high=cH, 
                 plot_warning_high=wH, remove_bads=filter_bads)

    tabs = Tab(children=[page1, page2, page3])

    tabs.set_title(0, 'MSID, Dates')
    tabs.set_title(1, 'Limits')
    tabs.set_title(2, 'Misc.')

    display(tabs)





