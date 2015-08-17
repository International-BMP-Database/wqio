from __future__ import print_function, division

import types
import pdb
import time
import sys
import os
import warnings
from six import StringIO

import numpy as np
import matplotlib.pyplot as plt
import pandas
from scipy import stats
import statsmodels.api as sm


def santizeTimestamp(timestamp):
    if not isinstance(timestamp, pandas.Timestamp):
        try:
            timestamp = pandas.Timestamp(timestamp)
        except:
            raise ValueError('{} could not be coerced into a pandas.Timestamp')

    return timestamp


def getSeason(date):
    '''Defines the season from a given date.

    Parameters
    ----------
    date : datetime.datetime object or similar
        Any object that represents a date and has `.month` and `.day`
        attributes

    Returns
    -------
    season : str

    Notes
    -----
    Assumes that all seasons changed on the 22nd (e.g., all winters
    start on Decemeber 22). This isn't strictly true, but it's good
    enough for now.

    '''
    date = santizeTimestamp(date)
    if (date.month == 12 and date.day >= 22) or \
            (date.month in [1, 2]) or \
            (date.month == 3 and date.day < 22):
        return 'winter'
    elif (date.month == 3 and date.day >= 22) or \
            (date.month in [4, 5]) or \
            (date.month == 6 and date.day < 22):
        return 'spring'
    elif (date.month == 6 and date.day >= 22) or \
            (date.month in [7, 8]) or \
            (date.month == 9 and date.day < 22):
        return 'summer'
    elif (date.month == 9 and date.day >= 22) or \
            (date.month in [10, 11]) or \
            (date.month == 12 and date.day < 22):
        return 'autumn'
    else: # pragma: no cover
        raise ValueError('could not assign season to  {}'.format(date))


def addSecondColumnLevel(levelval, levelname, olddf):
    '''
    Takes a simple index on a dataframe's columns and adds a new level
    with a single value.
    E.g., df.columns = ['res', 'qual'] -> [('Infl' ,'res'), ('Infl', 'qual')]
    '''
    if isinstance(olddf.columns, pandas.MultiIndex):
        raise ValueError('Dataframe already has MultiIndex on columns')
    colarray = [[levelval]*len(olddf.columns), olddf.columns]
    colindex = pandas.MultiIndex.from_arrays(colarray)
    newdf = olddf.copy()
    newdf.columns = colindex
    newdf.columns.names = [levelname, 'quantity']
    return newdf


def getUniqueDataframeIndexVal(df, indexlevel):
    '''
    Confirms that a given level of a dataframe's index only has
    one unique value. Useful for confirming consistent units.

    Raises error if level is not a single value. Returns value
    Otherwise
    '''
    index = np.unique(df.index.get_level_values(indexlevel).tolist())
    if index.shape != (1,):
        raise ValueError('index level "%s" is not unique!' % indexlevel)

    return index[0]


def sigFigs(x, n, expthresh=5, tex=False, pval=False, forceint=False):
    '''
    Formats a number into a string with the correct number of sig figs.

    Input:
        x (numeric) : the number you want to round
        n (int) : the number of sig figs it should have
        tex (bool) : toggles the scientific formatting of the number
        pval (bool) : if True and x < 0.001, will return "<0.001"
        forceint : if true, simply returns int(x)

    Typical Usage:
        >>> print(sigFigs(1247.15, 3))
               1250
        >>> print(sigFigs(1247.15, 7))
               1247.150
    '''
    # check on the number provided
    if x is not None and not np.isinf(x) and not np.isnan(x):

        # check on the sigFigs
        if n < 1:
            raise ValueError("number of sig figs must be greater than zero!")

        # return a string value unaltered
        #if type(x) == types.StringType:
        if isinstance(x, str):
            out = x

        elif pval and x < 0.001:
            out = "<0.001"
            if tex:
                out = '${}$'.format(out)

        elif forceint:
            out = '{:,.0f}'.format(x)

        # logic to do all of the rounding
        elif x != 0.0:
            order = np.floor(np.log10(np.abs(x)))

            if -1.0 * expthresh <= order <= expthresh:
                decimal_places = int(n - 1 - order)

                if decimal_places <= 0:
                    out = '{0:,.0f}'.format(round(x, decimal_places))

                else:
                    fmt = '{0:,.%df}' % decimal_places
                    out = fmt.format(x)

            else:
                decimal_places = n - 1
                if tex:
                    #raise NotImplementedError('no exponential tex formatting yet')
                    fmt = r'$%%0.%df \times 10 ^ {%d}$' % (decimal_places, order)
                    out = fmt % round(x / 10 ** order, decimal_places)
                else:
                    fmt = '{0:.%de}' % decimal_places
                    out = fmt.format(x)

        else:
            out = str(round(x, n))

    # with NAs and INFs, just return 'NA'
    else:
        out = 'NA'

    return out


def _sig_figs(x):
    '''
    Wrapper around `utils.sigFig` so it only requires 1 argument for the
        purpose of "apply"-ing it to a pandas dataframe

    Input:
        x : any number you want

    Writes:
        None

    Returns
        A string representation of `x` with 3 significant figures
    '''
    return sigFigs(x, n=3, tex=True)


def formatResult(result, qualifier, sigfigs=3):
    """ Formats a results with its qualifier

    Parameters
    ----------
    results : float
        The concentration or particulate strength
    qualifier : string
        The result's qualifier
    sigfigs : int
        The number of significant digits to which `result` should be
        formatted

    Returns
    -------
    formatted : string

    Example
    -------
    >>> wqio.formatResult(1.23, '<', sigfigs=4)
    "<1.230"

    """

    return '{}{}'.format(qualifier, sigFigs(result, sigfigs))


def _boxplot_legend(ax, notch=False, shrink=2.5, fontsize=10, showmean=False):
    '''
    Help function to draw a boxplot legend.
    Input:
        ax (matplotlib axes object) : axes on which the legend
            will be plotted
        notch (bool, default False) : whether or not to include
            notches (confidence intervals) around the median
        shrink (float, default 2.5) : some measure of how far away
            the annotation arrows should be from the elements to
            which they point.
    '''
    # load static, randomly generated data
    x = np.array([[
        1.7117, 2.5470, 3.2817, 2.3303, 2.7066, 4.2024, 2.7184, 2.9790,
        2.7782, 1.9440, 3.9939, 4.3938, 6.1780, 3.2937, 3.6596, 2.3589,
        1.5408, 3.7236, 2.9327, 4.2844, 3.5441, 3.9499, 2.0023, 3.7872,
        3.4989, 2.2898, 2.7913, 3.2796, 2.3650, 3.5436, 3.3459, 3.8699,
        3.7448, 2.0149, 2.1290, 4.2193, 4.3932, 1.6687, 5.1053, 2.3849,
        1.6996, 3.1484, 3.4078, 2.0051, 0.88211, 2.038, 3.3291, 2.3526,
        1.4030, 2.7147
    ]])

    # plot the boxplot
    bpLeg = ax.boxplot(x, notch=notch, positions=[1], widths=0.5, bootstrap=10000)

    # plot the mean
    if showmean:
        ax.plot(1, x.mean(), marker='o', mec='k', mfc='r', zorder=100)

    # format stuff (colors and linewidths and whatnot)
    plt.setp(bpLeg['whiskers'], color='k', linestyle='-', zorder=10)
    plt.setp(bpLeg['boxes'], color='k', zorder=10)
    plt.setp(bpLeg['medians'], color='r', zorder=5, linewidth=1.25)
    plt.setp(bpLeg['fliers'], marker='o', mfc='none', mec='r', ms=4, zorder=10)
    plt.setp(bpLeg['caps'], linewidth=0)

    # positions of the boxplot elements
    if notch:
        x05, y05 = 1.00, bpLeg['caps'][0].get_ydata()[0]
        x25, y25 = bpLeg['boxes'][0].get_xdata()[1], bpLeg['boxes'][0].get_ydata()[0]
        x50, y50 = bpLeg['medians'][0].get_xdata()[1], bpLeg['medians'][0].get_ydata()[0]
        x75, y75 = bpLeg['boxes'][0].get_xdata()[1], bpLeg['boxes'][0].get_ydata()[5]
        x95, y95 = 1.00, bpLeg['caps'][1].get_ydata()[0]
        xEx, yEx = 1.00, bpLeg['fliers'][0].get_ydata()[0]
        xCIL, yCIL = bpLeg['boxes'][0].get_xdata()[0], bpLeg['boxes'][0].get_ydata()[2]
        xCIU, yCIU = bpLeg['boxes'][0].get_xdata()[0], bpLeg['boxes'][0].get_ydata()[4]

    else:
        x05, y05 = bpLeg['caps'][0].get_xdata()[1], bpLeg['caps'][1].get_ydata()[0]
        x25, y25 = bpLeg['boxes'][0].get_xdata()[0], bpLeg['boxes'][0].get_ydata()[0]
        x50, y50 = bpLeg['medians'][0].get_xdata()[1], bpLeg['medians'][0].get_ydata()[0]
        x75, y75 = bpLeg['boxes'][0].get_xdata()[0], bpLeg['boxes'][0].get_ydata()[2]
        x95, y95 = bpLeg['caps'][1].get_xdata()[1], bpLeg['caps'][0].get_ydata()[0]
        xEx, yEx = 0.95, bpLeg['fliers'][0].get_ydata()[0]

    # axes formatting
    ax.set_xlim([0, 2])
    ax.set_yticks(range(9))
    ax.set_yticklabels([])
    ax.set_xticklabels([], fontsize=5)
    ax.xaxis.set_ticks_position("none")
    ax.yaxis.set_ticks_position("none")

    # annotation formats
    ap = dict(arrowstyle="->", shrinkB=shrink)

    # text for the labels
    note1 = r'{1) Interquartile range: $\mathrm{IQR} = \mathrm{Q3} - \mathrm{Q1}$}'
    note2 = '{2) Geometric means are plotted only for bacteria data.\nOtherwise, arithmetic means are shown.'
    legText = {
        '5th': r'{Min. data $\ge \mathrm{Q1} - 1.5 \times \mathrm{IQR}$}',
        '25th': r'{$25^{\mathrm{th}}\:\mathrm{percentile},\:\mathrm{Q}1$}',
        '50th': r'{$50^{\mathrm{th}} \mathrm{percentile}$, median}',
        '75th': r'{$75^{\mathrm{th}}\:\mathrm{percentile},\:\mathrm{Q3}$}',
        '95th': r'{Max. data $\le \mathrm{Q3} + 1.5 \times \mathrm{IQR}$}',
        'Out': r'{Outlier $>\mathrm{Q3} + 1.5 \times \mathrm{IQR}$} (note 1)',
        'CIL': r'{Lower 95\% CI about the median}',
        'CIU': r'{Upper 95\% CI about the median}',
        'notes': 'Notes:\n%s\n%s' % (note1, note2),
        'mean': r'Mean (note 2)'
    }

    # add notes
    ax.annotate(legText['notes'], (0.05, 0.01), xycoords='data', fontsize=fontsize)

    # label the mean
    if showmean:
        ax.annotate(legText['mean'], (1, x.mean()), xycoords='data',
                    xytext=(1.15, 5.75), textcoords='data', va='center',
                    arrowprops=ap, fontsize=fontsize)

    # label the lower whisker
    ax.annotate(legText['5th'], (x05, y05), xycoords='data',
                xytext=(1.25, y05), textcoords='data', va='center',
                arrowprops=ap, fontsize=fontsize)

    # label 1st quartile
    ax.annotate(legText['25th'], (x25, y25), xycoords='data',
                xytext=(1.55, y25*0.75), textcoords='data', va='center',
                arrowprops=ap, fontsize=fontsize)

    # label the median
    ax.annotate(legText['50th'], (x50, y50), xycoords='data',
                xytext=(1.45, y50), textcoords='data', va='center',
                arrowprops=ap, fontsize=fontsize)

    # label the 3rd quartile
    ax.annotate(legText['75th'], (x75, y75), xycoords='data',
                xytext=(1.55, y75*1.25), textcoords='data', va='center',
                arrowprops=ap, fontsize=fontsize)

    # label the upper whisker
    ax.annotate(legText['95th'], (x95, y95), xycoords='data',
                xytext=(0.05, 7.00), textcoords='data', va='center',
                arrowprops=ap, fontsize=fontsize)

    # label an outlier
    ax.annotate(legText['Out'], (xEx, yEx), xycoords='data',
                xytext=(1.00, 7.50), textcoords='data', va='center',
                arrowprops=ap, fontsize=fontsize)

    # label confidence intervals around the mean
    if notch:
        ax.annotate(legText['CIL'], (xCIL, yCIL), xycoords='data',
                    xytext=(0.05, 0.75*yCIL), textcoords='data', va='center',
                    arrowprops=ap, fontsize=fontsize)

        ax.annotate(legText['CIU'], (xCIU, yCIU), xycoords='data',
                    xytext=(0.05, 1.25*yCIU), textcoords='data', va='center',
                    arrowprops=ap, fontsize=fontsize)

    # legend and grid formats
    ax.set_frame_on(False)
    ax.yaxis.grid(False, which='major')
    ax.xaxis.grid(False, which='major')


def makeBoxplotLegend(filename='bmp/tex/boxplotlegend', figsize=4, **kwargs):
    '''
    Creates an explanatory diagram for boxplots. **kwargs are fed to _boxplot_legend
    '''
    # setup the figure
    fig, ax = plt.subplots(figsize=(figsize, figsize))

    # call the helper function that does the heavy lifting
    _boxplot_legend(ax, **kwargs)

    # optimize the figure's layout
    fig.tight_layout()

    # save and close
    fig.savefig(filename + '.pdf', transparent=True, dpi=300)
    fig.savefig(filename + '.png', transparent=True, dpi=300)
    plt.close(fig)


def processFilename(filename):
    '''
    Sanitizes a filename. DON'T feed it a full path
    Typical Usage
        >>> processFilename('FigureBenzon/Inzo_1')
        FigureBenzonInzo1
    '''
    badchars = [' ', ',', '+', '$', '_', '{', '}', '/', '&']
    fn = filename
    for bc in badchars:
        fn = fn.replace(bc, '')
    return fn


def constructPath(name, ext, *args):
    '''
    Builds a path from a filename, extension, and directories.
    Infers the last directory from the extension
    >>> print(constructPath('test1 2', 'tex', 'cvc', 'output'))
    cvc/output/tex/test12.tex
    '''
    destinations = {
        'png': 'img',
        'pdf': 'img',
        'csv': 'csv',
        'tex': 'tex',
    }
    dest_dir = destinations[ext]
    filename = processFilename(name + '.' + ext)
    basepath = os.path.join(*args)
    filepath = os.path.join(basepath, dest_dir, filename)
    return filepath


def makeTablesFromCSVStrings(tablestring, texpath=None, csvpath=None):
    '''
    Takes a string already in CSV format and writes it to a CSV file and a
        LaTeX table

    Input:
        tablestring (string) : CSV-formatted string of data
        filename (string) : filename of the output files

    Writes:
         CSV and LaTeX files of the data

    Returns:
        None
    '''
    # make a dataframe of the csvstring
    df = pandas.read_csv(StringIO(tablestring))

    # write the CSV file
    if csvpath is not None:
        with open(csvpath, 'w') as csv:
            csv.write(tablestring)

    # write the LaTeX file
    if texpath is not None:
        with open(texpath, 'w') as tex:
            tex.write(df.to_latex(index=False).replace("Unnamed: 1", ""))


def addStatsToOutputSummary(csvpath, index_cols=['Date'], na_values='--'):
    '''
    Reads a CSV file in to a pandas dataframe and appends stats to the bottom

    Input:
        filepath (string) : full path to a file *without* the extension

    Writes:
        1) a CSV file of the data in `filepath + '.csv'` with stats at the
            bottom of the file
        2) a LaTeX table version of the file above

    Returns:
        None

    TODO: figure out what to do with -counts-
    '''

    # read in the data, pretending that the dates are strings
    summary = pandas.read_csv(csvpath, parse_dates=False, index_col=index_cols,
                              na_values=na_values)

    orig_cols = summary.columns.tolist()

    # compute stats
    stat_df = summary.describe()

    # append stats to the bottom of the table
    summary = summary.append(stat_df)

    # set the index's name
    summary.index.names = index_cols

    # dump the CSV
    summary[orig_cols].to_csv(csvpath, na_rep='--')
    return summary


def addExternalValueToOutputSummary(csvpath, comparedict, comparecol,
                                    index_cols=['Date'], na_values='--'):
    # read in the data, pretending that the dates are strings
    summary = pandas.read_csv(csvpath, parse_dates=False, index_col=index_cols,
                              na_values=na_values)
    original_columns = summary.columns

    # make the comparison values a dataframe
    compare_df = pandas.DataFrame(sorted(list(comparedict.values())),
                                  index=sorted(list(comparedict.keys())),
                                  columns=[comparecol])

    # append that df
    summary = summary.append(compare_df)

    # set the index's name
    summary.index.names = index_cols

    # dump the CSV
    summary[original_columns].to_csv(csvpath, na_rep='--')
    return summary


def sanitizeTex(texstring):
    newstring = (
        texstring.replace(r'\\%', r'\%')
                 .replace(r'\\', r'\tabularnewline')
                 .replace('\$', '$')
                 .replace('\_', '_')
                 .replace('ug/L', '\si[per-mode=symbol]{\micro\gram\per\liter}')
                 .replace(r'\textbackslashtimes', r'\times')
                 .replace(r'\textbackslash', '')
                 .replace(r'\textasciicircum', r'^')
                 .replace('\{', '{')
                 .replace('\}', '}')
    )
    return newstring


def csvToTex(csvpath, texpath, na_rep='--', float_format=_sig_figs, pcols=15,
             addmidrules=None, replaceTBrules=True, replacestats=True):
    '''
    Convert data in CSV format to a LaTeX table

    Input:
        csvpath (string) : full name and file path of the input data file
        texpath (string) : full name and file path of the output LaTeX file
        na_rep (string, default "--") : how NA values should be written
        float_format (fxn, default `_sig_figs`) : single input function that
            will return the correct representation of floating point numbers

    Writes:
        A LaTeX table representation of the data found in `csvpath`

    Returns:
        None

    '''
    # read in the data pandas
    data = pandas.read_csv(csvpath, parse_dates=False, na_values=[na_rep])

    # open a new file and use pandas to dump the latex and close out
    with open(texpath, 'w') as texfile:
        data.to_latex(texfile, float_format=float_format, na_rep=na_rep,
                      index=False)

    if pcols > 0:
        lines = []
        texfile = open(texpath, 'r')
        header = texfile.readline()
        header_sections = header.split('{')
        old_col_def = header_sections[-1][:-2]
        new_col_def = ''
        for n in range(len(old_col_def)):
            if n == 0:
                new_col_def = new_col_def + 'l'
            new_col_def = new_col_def + 'x{%smm}' % pcols

        lines.append(header.replace(old_col_def, new_col_def))
        rest_of_file = sanitizeTex(texfile.read())

        if replaceTBrules:
            rest_of_file = rest_of_file.replace("\\toprule", "\\midrule")
            rest_of_file = rest_of_file.replace("\\bottomrule", "\\midrule")

        if replacestats:
            rest_of_file = rest_of_file.replace("std", "Std. Dev.")
            rest_of_file = rest_of_file.replace("50\\%", "Median")
            rest_of_file = rest_of_file.replace("25\\%", "25th Percentile")
            rest_of_file = rest_of_file.replace("75\\%", "75th Percentile")
            rest_of_file = rest_of_file.replace("count", "Count")
            rest_of_file = rest_of_file.replace("mean", "Mean")
            rest_of_file = rest_of_file.replace("min ", "Min. ")
            rest_of_file = rest_of_file.replace("max", "Max.")

            # XXX: omg hack
            rest_of_file = rest_of_file.replace("AluMin.um", "Aluminum")

        if addmidrules is not None:
            if hasattr(addmidrules, 'append'):
                for amr in addmidrules:
                    rest_of_file = rest_of_file.replace(amr, '\\midrule\n%s' % amr)
            else:
                rest_of_file = rest_of_file.replace(amr, '\\midrule\n%s' % addmidrules)

        lines.extend(rest_of_file)
        texfile.close()

        texfile = open(texpath, 'w')
        texfile.writelines(lines)
        texfile.close()


def csvToXlsx(csvpath, xlsxpath, na_rep='--', float_format=None):
    '''
    Convert data in CSV format to a Excel workbook

    Input:
        csvpath (string) : full name and file path of the input data file
        xlsxpath (string) : full name and file path of the output .xlsx file
        na_rep (string, default "--") : how NA values should be written
        float_format (fxn, default `_sig_figs`) : single input function that
            will return the correct representation of floating point numbers

    Writes:
        A LaTeX table representation of the data found in `csvpath`

    Returns:
        None

    '''
    # read in the data pandas
    data = pandas.read_csv(csvpath, parse_dates=False, na_values=[na_rep])

    # use pandas to dump the excel file and close out
    data.to_excel(xlsxpath, float_format=float_format, na_rep=na_rep, index=False)


def nested_getattr(baseobject, attribute):
    '''
    Returns the value of an attribute of an object that
    is nested several layers deep.

    Input:
        baseobject : this seriously can be anything
        attribute (string) : and string representation of what you want

    Writes:
        None

    Output:
        No telling. It depends on what you ask for.

    Example:
        >>> nested_getattr(dataset, 'influent.stats.mean')
    '''
    for attr in attribute.split('.'):
        baseobject = getattr(baseobject, attr)
    return baseobject


def normalize_units(dataframe, units_map, targetunit, paramcol='parameter',
                    rescol='Outflow_res', unitcol='Outflow_unit'):

    try:
        units_map[targetunit]
    except KeyError:
        raise ValueError('{0} is not contained in `units_map`'.format(targetunit))

    # standardize units in the wqdata
    dataframe['normalize'] = dataframe[unitcol].map(units_map.get)
    if isinstance(targetunit, dict):
        dataframe['targetunit'] = dataframe[paramcol].map(targetunit.get)
    else:
        dataframe['targetunit'] = targetunit

    dataframe['convert'] = dataframe['targetunit'].map(units_map.get)
    dataframe[rescol] = dataframe[rescol] * dataframe['normalize'] / dataframe['convert']

    # reassign unites
    dataframe[unitcol] = dataframe.targetunit
    return dataframe


def normalize_units2(data, normFxn, convFxn, unitFxn, paramcol='parameter',
                     rescol='res', unitcol='unit', dlcol=None):
    d = data.copy()
    normalization = d[unitcol].apply(normFxn)
    conversion = d[paramcol].apply(convFxn)

    factor = normalization / conversion

    d[rescol] *= factor
    if dlcol is not None:
        d[dlcol] *= factor

    d.loc[:, unitcol] = d[paramcol].apply(unitFxn)
    return d


def makeTexTable(tablefile, caption, sideways=False, footnotetext=None,
                 clearpage=False, pos='h!'):
    if sideways:
        tabletype = 'sidewaystable'
        clearpage = True
    else:
        tabletype = 'table'

    if clearpage:
        clearpagetext = r'\clearpage'
    else:
        clearpagetext = ''

    if footnotetext is None:
        notes = ''
    else:
        notes = footnotetext

    tablestring = r"""
    \begin{%s}[%s]
        \rowcolors{1}{CVCWhite}{CVCLightGrey}
        \caption{%s}
        \centering
        \input{%s}
    \end{%s}
    %s
    %s
    """ % (tabletype, pos, caption, tablefile, tabletype, notes, clearpagetext)
    return tablestring


def makeLongLandscapeTexTable(df, caption, label, footnotetext=None, index=False):
    if footnotetext is None:
        notes = ''
    else:
        notes = footnotetext

    tabletexstring = df.to_latex(index=index, float_format=_sig_figs, na_rep='--')
    valuelines = tabletexstring.split('\n')[4:-3]
    valuestring = '\n'.join(valuelines)

    def _multicol_format(args):
        n, col = args
        if n == 0:
            align = 'l'
        else:
            align = 'p{16mm}'

        return r"\multicolumn{1}{%s}{%s}" % (align, col.replace('%', r'\%'))

    dfcols = df.columns.tolist()

    colalignlist = ['c'] * len(dfcols)
    colalignlist[0] = 'l'
    colalignment = ''.join(colalignlist)

    col_enum = list(enumerate(dfcols))
    columns = ' &\n\t\t'.join(list(map(_multicol_format, col_enum)))

    tablestring = r"""
    \begin{landscape}
        \centering
        \rowcolors{1}{CVCWhite}{CVCLightGrey}
        \begin{longtable}{%s}
            \caption{%s} \label{%s} \\
            \toprule
                %s \\
            \toprule
            \endfirsthead

            \multicolumn{%d}{c}
            {{\bfseries \tablename\ \thetable{} -- continued from previous page}} \\
            \toprule
                %s \\
            \toprule
            \endhead

            \toprule
                \rowcolor{CVCWhite}
                \multicolumn{%d}{r}{{Continued on next page...}} \\
            \bottomrule
            \endfoot

            \bottomrule
            \endlastfoot

%s

        \end{longtable}
    \end{landscape}
    %s
    \clearpage
    """ % (colalignment, caption, label, columns, len(dfcols),
           columns, len(dfcols), valuestring, notes)
    return tablestring


def makeTexFigure(figFile, caption, pos='hb', clearpage=True):
    '''
    Create the LaTeX for include a figure in a document

    Input:
        figFile (string) : path to the image you want to include
        caption (string) : what it should say in the figure's caption
        pos (string, default 'hb') : placement preferences
            (h='here' or b='below')
        clearpage (bool, default True) : whether or not the LaTeX
            command "\clearpage" should be called after the figure

    Returns:
        figurestring (string) : the LaTeX string to include a figure
            in the appendix reports
    '''
    if clearpage:
        clearpagetext = r'\clearpage'
    else:
        clearpagetext = ''

    figurestring = r"""
    \begin{figure}[%s]   %% FIGURE
        \centering
        \includegraphics[scale=1.00]{%s}
        \caption{%s}
    \end{figure}         %% FIGURE
    %s
    """ % (pos, figFile, caption, clearpagetext)
    return figurestring


def stringify(value, fmt, attribute=None):
    if attribute is not None and value is not None:
        quantity = nested_getattr(value, attribute)
    else:
        quantity = value

    if quantity is None:
        return '--'
    else:
        return fmt % quantity


def pH2concentration(pH, *args):
    '''
    Takes a pH value and converts it to proton concentration
    in mg/L
    '''
    # check that we recieved a valid input:
    if pH < 0 or pH > 14:
        raise ValueError('pH = %f but must be between 0 and 14' % pH)

    # avogadro's number (items/mole)
    avogadro = 6.0221413e+23

    # mass of a proton (kg)
    proton_mass = 1.672621777e-27

    # grams per kilogram
    kg2g = 1000

    # milligrams per gram
    g2mg = 1000

    return 10**(-1*pH) * avogadro * proton_mass * kg2g * g2mg


def estimateFromLineParams(xdata, slope, intercept, xlog=False, ylog=False):
    '''
    Estimate the dependent of a linear fit given x-data and linear parameters

    Parameters
    ----------
    xdata : numpy array or pandas Series/DataFrame
        The input independent variable of the fit

    slope : float
        Slope of the best-fit line

    intercept : float
        y-intercept of the best-fit line

    xlog : bool (default = False)
        Toggles whether or not the x-data are lognormally distributed

    ylog : bool (default = False)
        Toggles whether or not the y-data are lognormally distributed

    Returns
    -------
    yhat : same type as xdata
        Estimate of the dependent variable.

    '''

    x = np.array(xdata)
    if ylog:
        if xlog:
            yhat = np.exp(intercept) * x  ** slope
        else:
            yhat = np.exp(intercept) * np.exp(slope) ** x

    else:
        if xlog:
            yhat = slope * np.log(x) + intercept

        else:
            yhat = slope * x + intercept

    return yhat


def redefineIndexLevel(dataframe, levelname, value, criteria=None, dropold=True):
    '''
    Redefine a selection of BMPs into another or new category
    Input:
        dataframe : pandas DataFrame.

        levelname : string
            The name of the index level that needs to be modified. The catch
            here is that this value needs to be valid after calling
            `dataframe.reset_index()`. In otherwords, if you have a 3-level
            column index and you want to modify the "Units" level of the index,
            you should actually pass `("Units", "", "")`. Annoying, but that's
            life right now.

        value : string or int
            The replacement value for the index level.

        critera : function/lambda expression or None
            This should return True/False in a manner consitent with the
            `.select()` method of a pandas dataframe. See that docstring
            for more info. If None, the redifinition will apply to the whole
            dataframe.

        dropold : optional bool (defaul is True)
            Toggles the replacement (True) or addition (False) of the data
            of the redefined BMPs into the the `data` dataframe.

    Returns:
        None

    '''

    if criteria is not None:
        selection = dataframe.select(criteria)
    else:
        selection = dataframe.copy()

    if dropold:
        dataframe = dataframe.drop(selection.index)

    selection.reset_index(inplace=True)
    selection[levelname] = value
    selection = selection.set_index(dataframe.index.names)

    return dataframe.append(selection).sort_index()


def checkIntervalOverlap(interval1, interval2, oneway=False):
    '''Checks if two numeric intervals overlaps

    Parameters
    ----------
    interval1, interval2 : array like
        len = 2 sequences to compare

    oneway : bool, default = False
        if true, only checks that interval1 falls at least partially
        inside interval2, but not the other way around

    Returns
    -------
    bool

    '''
    test1 = np.min(interval2) <= np.max(interval1) <= np.max(interval2)
    test2 =  np.min(interval2) <= np.min(interval1) <= np.max(interval2)

    if oneway:
        return test1 or test2
    else:
        test3 = checkIntervalOverlap(interval2, interval1, oneway=True)
        return test1 or test2 or test3


def makeTimestamp(row, datecol='sampledate', timecol='sampletime',
                  issuewarnings=False):
    '''Makes a pandas.Timestamp from separate date/time columns

    Parameters
    ----------

    row : dict-like (ideallty a row in a dataframe)
    datecol : optional string (default = 'sampledate')
        Name of the column containing the dates
    timecol : optional string (default = 'sampletime')
        Name of the column containing the times

    Returns
    -------

    tstamp : pandas.Timestamp

    '''

    fallback_datetime = pandas.Timestamp('1901-01-01 00:00')

    if row[datecol] is None or pandas.isnull(row[datecol]):
        fb_date = True
    else:
        fb_date = False
        try:
            date = pandas.Timestamp(row[datecol]).date()
        except ValueError:
            fb_date = True

    if fb_date:
        date = fallback_datetime.date()
        if issuewarnings:
            warnings.warn("Using fallback date from {}".format(row[datecol]))

    if row[timecol] is None or pandas.isnull(row[timecol]):
        fb_time = True
    else:
        fb_time = False
        try:
            time = pandas.Timestamp(row[timecol]).time()
        except ValueError:
            fb_time = True

    if fb_time:
        time = fallback_datetime.time()
        if issuewarnings:
            warnings.warn("Using fallback time from {}".format(row[timecol]))

    dtstring = '{} {}'.format(date, time)
    tstamp = pandas.Timestamp(dtstring)

    return tstamp


def whiskers_and_fliers(x, q1, q3, transformout=None):
    wnf = {}
    if transformout is None:
        transformout = lambda x: x

    iqr = q3 - q1
    # get low extreme
    loval = q1 - (1.5 * iqr)
    whislo = np.compress(x >= loval, x)
    if len(whislo) == 0 or np.min(whislo) > q1:
        whislo = q1
    else:
        whislo = np.min(whislo)

    # get high extreme
    hival = q3 + (1.5 * iqr)
    whishi = np.compress(x <= hival, x)
    if len(whishi) == 0 or np.max(whishi) < q3:
        whishi = q3
    else:
        whishi = np.max(whishi)

    wnf['fliers'] = np.hstack([
        transformout(np.compress(x < whislo, x)),
        transformout(np.compress(x > whishi, x))
    ])
    wnf['whishi'] = transformout(whishi)
    wnf['whislo'] = transformout(whislo)

    return wnf


def getWaterYear(date):
    """ Returns the water year of a given date

    Parameters
    ----------
    date : datetime-like
        A datetime or Timestamp object

    Returns
    -------
    wateryear : string
        The water year of `date`

    Example
    -------
    >>> import datetime
    >>> import wqio
    >>> x = datetime.datetime(2005, 11, 2)
    >>> print(wqio.utils.getWaterYear(x))
        '2005/2006'

    """

    year = date.year
    yearstring = '{}/{}'
    if date.month >= 10:
        return yearstring.format(year, year + 1)
    else:
        return yearstring.format(year - 1, year)


def fit_line(x, y, xhat=None, fitprobs=None, fitlogs=None, dist=None):
    """ Fits a line to x-y data in various forms (raw, log, prob scales)

    Parameters
    ----------
    x, y : array-like
        Independent and dependent data, respectively.
    xhat : array-like or None, optional
        The values at which yhat should should be estimated. If
        not provided, falls back to the sorted values of ``x``.
    fitprobs, fitlogs : str, options.
        Defines how data should be transformed. Valid values are
        'x', 'y', or 'both'. If using ``fitprobs``, variables should
        be expressed as a percentage, i.e.,
        Probablility transform = lambda x: ``dist``.ppf(x / 100.).
        Log transform = lambda x: np.log(x).
        Take care to not pass the same value to both ``fitlogs`` and
        ``figprobs`` as both transforms will be applied.
    dist : scipy.stats distribution or None, optional
        A fully-spec'd scipy.stats distribution such that ``dist.ppf``
        can be called. If not provided, defaults to scipt.stats.norm.

    Returns
    -------
    xhat, yhat : numpy arrays
        Linear model estimates of ``x`` and ``y``.
    results : a statmodels result object
        The object returned by statsmodels.OLS.fit()

    """

    def _check_fit_arg(arg, argname):
        valid_args = ['x', 'y', 'both', None]
        if arg not in valid_args:
            msg = 'Valid value for {} ({}). Must be on of {}'
            raise ValueError(msg.format(argname, arg, valid_args))

    _check_fit_arg(fitprobs, "fitprobs")
    _check_fit_arg(fitlogs, "fitlogs")

    if xhat is None:
        xhat = np.array([np.min(x), np.max(x)])

    if dist is None:
        dist = stats.norm

    if fitprobs in ['x', 'both']:
        x = dist.ppf(x/100.)
        xhat = dist.ppf(np.array(xhat)/100.)

    if fitprobs in ['y', 'both']:
        y  = dist.ppf(y/100.)

    if fitlogs in ['x', 'both']:
        x = np.log(x)
    if fitlogs in ['y', 'both']:
        y = np.log(y)

    x = sm.add_constant(x)
    model = model = sm.OLS(y, x)
    results = model.fit()


    yhat = estimateFromLineParams(xhat, results.params[1],
                                        results.params[0],
                                        xlog=fitlogs in ['x', 'both'],
                                        ylog=fitlogs in ['y', 'both'])

    if fitprobs in ['y', 'both']:
        yhat = 100.* dist.cdf(yhat)
    if fitprobs in ['x', 'both']:
        xhat = 100.* dist.cdf(xhat)

    return xhat, yhat, results


def processAndersonDarlingResults(ad_results):
    """ Return a nice string of Anderson-Darling test results

    Parameters
    ----------
    ad_result : tuple or namedtuple
        The packed output from scipt.stats.anderson

    Returns
    -------
    result : str
        A string representation of the confidence in the result.
    """
    a2, crit, sig = ad_results
    try:
        ci = 100 - sig[a2 < crit][-1]
        return '%0.1f%%' % (ci,)
    except IndexError:
        ci = 100 - sig[0]
        return '<%0.1f%%' % (ci,)


class ProgressBar:
    def __init__(self, sequence, width=50, labels=None, labelfxn=None):
        '''Progress bar for notebookes:

        Basic Usage:
        >>> X = range(1000)
        >>> pbar = utils.ProgressBar(X)
        >>> for n, x in enumerate(X, 1):
        >>>     # do stuff with x
        >>>     pbar.animate(n)


        '''
        self.sequence = sequence
        self.iterations = len(sequence)
        self.labels = labels
        self.labelfxn = labelfxn
        self.prog_bar = '[]'
        self.fill_char = '*'
        self.width = width
        self.__update_amount(0)

    def animate(self, iter):
        print('\r', self, end='')
        sys.stdout.flush()
        self.update_iteration(iter + 1)

    def update_iteration(self, elapsed_iter):
        self.__update_amount((elapsed_iter / float(self.iterations)) * 100.0)
        if self.labels is None and self.labelfxn is None:
            self.prog_bar += '  %d of %s complete' % (elapsed_iter, self.iterations)
        elif elapsed_iter <= self.iterations:
            if self.labels is None:
                label = self.labelfxn(self.sequence[elapsed_iter-1])
            else:
                label = self.labels[elapsed_iter-1]

            self.prog_bar += '  %d of %s (%s)' % (elapsed_iter, self.iterations, label)

    def __update_amount(self, new_amount):
        percent_done = int(round((new_amount / 100.0) * 100.0))
        all_full = self.width - 2
        num_hashes = int(round((percent_done / 100.0) * all_full))
        self.prog_bar = '[' + self.fill_char * num_hashes + ' ' * (all_full - num_hashes) + ']'
        pct_place = (len(self.prog_bar) // 2) - len(str(percent_done))
        pct_string = '%d%%' % percent_done
        self.prog_bar = self.prog_bar[0:pct_place] + \
            (pct_string + self.prog_bar[pct_place + len(pct_string):])

    def __str__(self):
        return str(self.prog_bar)
