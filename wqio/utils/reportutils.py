from textwrap import dedent
import os
import subprocess

from six import StringIO
import numpy as np
import pandas

from wqio.tests import helpers
from . import numutils
from . import figutils


def _sig_figs(x):
    """ Wrapper around `utils.sigFig` (n=3, tex=True) requiring only
    argument for the purpose of easily "apply"-ing it to a pandas
    dataframe.
    """

    return numutils.sigFigs(x, n=3, tex=True)


def makeBoxplotLegend(filename='bmp/tex/boxplotlegend', figsize=4, **kwargs):
    """ Creates an explanatory diagram for boxplots.

    Parameters
    ----------
    filename : string
        File name and path (without extension) where the figure will be
        saved.
    figsize : float or int
        Size of the figure in inches.
    kwargs : keyword arguments
        Keyword arguments passed directly to `_boxplot_legend`

    Returns
    -------
    None

    """

    # setup the figure
    fig, ax = plt.subplots(figsize=(figsize, figsize))

    # call the helper function that does the heavy lifting
    figutils.boxplot_legend(ax, **kwargs)

    # optimize the figure's layout
    fig.tight_layout()

    # save and close
    fig.savefig(filename + '.pdf', transparent=True, dpi=300)
    fig.savefig(filename + '.png', transparent=True, dpi=300)
    plt.close(fig)


def sanitizeTex(texstring):
    """ Cleans up overly eager LaTeX renderings from pandas.

    Parameters
    ----------
    texstring : string
        The string of LaTeX code to be cleaned up

    Returns
    -------
    sanitized : string
        Cleaned up LaTeX string.

    """

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
    """ Convert data in CSV format to a LaTeX table

    Parameters
    ----------
    csvpath : string
        Full name and file path of the input data file.
    texpath : string
        Full name and file path of the output LaTeX file.
    na_rep : string, default "--"
        How NA values should be written.
    float_format : callable (default = `_sig_figs`)
        Single input function that will return the correct
        representation of floating point numbers.
    pcols : int (default = 15)
        Width of the columns for the LaTeX table.
    addmidrules : string or list of strings, optional
        (List of) string(s) to be replaced with "\midrule".
    replaceTBrules : bool, default = True
        When True, replaces "\toprule" and "\bottomrule" with
        "\midrule".
    replacestats : bool, default = True
        When True, the labels of statistics are cleaned up a bit (e.g.,
        "75%" -> "75th Percentile")

    Returns
    -------
    None

    """

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
    """ Convert data in CSV format to an Excel workbook

    Parameters
    ----------
    csvpath : string
        Full name and file path of the input data file.
    xlsxpath : string
        Full name and file path of the output .xlsx file.
    na_rep : string (default = "--")
        How NA values should be represented.
    float_format : callable, optional
        Single input function that will return the correct
        representation of floating point numbers.

    Returns
    -------
    None

    """
    # read in the data pandas
    data = pandas.read_csv(csvpath, parse_dates=False, na_values=[na_rep])

    # use pandas to dump the excel file and close out
    data.to_excel(xlsxpath, float_format=float_format, na_rep=na_rep, index=False)


def makeTexTable(tablefile, caption, sideways=False, footnotetext=None,
                 clearpage=False, pos='h!'):
    """ Creates a table block for a LaTeX document. Does not add it any
    file.

    Parameters
    ----------
    tablefile : string
        Name of the .tex file that actually contains the table.
    caption : string
        Caption/title that should be given to the table.
    sideways : bool (default = False)
        When True, a landscape table block is produced. Otherwise, the
        table is in portrait mode.
    footnotetext : string, optional
        Any text that should be added as a footnote.
    clearpage : bool (default = False)
        When True, a "\clearpage" command is appended to the end of the
        table block.
    pos : string (default = "h!")
        LaTeX float position specification. Default values tries its
        best to place the table where the block appears in the LaTeX
        document.

    Returns
    -------
    tablestring : string
        The table block text that can be -- but has not been -- added
        to a LaTeX document.

    """
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

    tablestring = dedent(r"""
    \begin{%s}[%s]
        \rowcolors{1}{CVCWhite}{CVCLightGrey}
        \caption{%s}
        \centering
        \input{%s}
    \end{%s}
    %s
    %s
    """) % (tabletype, pos, caption, tablefile, tabletype, notes, clearpagetext)
    return tablestring


def makeLongLandscapeTexTable(df, caption, label, footnotetext=None, index=False):
    """ Create a multi-page landscape label for a LaTeX document.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to be turned into the table.
    caption : string
        Caption/title to be given to the table.
    label : string
        Unique identifier for references to table within LaTeX.
    footnotetext : string, optional
        Any text that should be added as a footnote.
    index : bool (default = False)
        Toggles the inclusion of the dataframe's index in to the table.
        Default behavior omits it.

    Returns
    -------
    tablestring : string
        The table block text that can be -- but has not been -- added
        to a LaTeX document.

    """

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
    columns = ' &\n        '.join(list(map(_multicol_format, col_enum)))

    tablestring = dedent(r"""
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
    """) % (colalignment, caption, label, columns, len(dfcols),
           columns, len(dfcols), valuestring, notes)
    return tablestring


def makeTexFigure(figFile, caption, pos='hb', clearpage=True):
    """ Create the LaTeX for include a figure in a document. Does not
    actually add it to any document.

    Parameters
    ----------
    figfile : string
        Name of the image (.pdf) file that actually contains the figure.
    caption : string
        Caption/title that should be given to the table.
    sideways : bool (default = False)
        When True, a landscape table block is produced. Otherwise, the
        table is in portrait mode.
    footnotetext : string, optional
        Any text that should be added as a footnote.
    clearpage : bool (default = False)
        When True, a "\clearpage" command is appended to the end of the
        table block.
    pos : string (default = "h!")
        LaTeX float position specification. Default values tries its
        best to place the table where the block appears in the LaTeX
        document.

    Returns
    -------
    tablestring : string
        The table block text that can be -- but has not been -- added
        to a LaTeX document.

    """
    if clearpage:
        clearpagetext = r'\clearpage'
    else:
        clearpagetext = ''

    figurestring = dedent(r'''
    \begin{figure}[%s]   %% FIGURE
        \centering
        \includegraphics[scale=1.00]{%s}
        \caption{%s}
    \end{figure}         %% FIGURE
    %s
    ''') % (pos, figFile, caption, clearpagetext)
    return figurestring


def processFilename(filename):
    """ Sanitizes a filename for LaTeX. DON'T feed it a full path.

    Parameters
    ----------
    filename : string
        The name of the file to be sanitized.

    Returns
    -------
    sanitized : string
        Mutated filename without characters that might cause errors in
        LaTeX.

    Example
    -------
    >>> processFilename('FigureBenzon/Inzo_1')
    FigureBenzonInzo1

    """

    badchars = [' ', ',', '+', '$', '_', '{', '}', '/', '&']
    fn = filename
    for bc in badchars:
        fn = fn.replace(bc, '')
    return fn


class LaTeXDirectory(object):
    """ Context manager to help compile latex docs from python.

    Switches to the latex document's folder and remains there while
    inside the manager. The present working directory is restored once
    the context manager exits.

    Parameters
    ----------
    texpath : string
        The LaTeX source file or the directory in which it is found.

    """

    def __init__(self, texpath):
        self.home = os.getcwd()
        if os.path.isfile(texpath):
            self.texpath = os.path.dirname(texpath)
        else:
            self.texpath = texpath

    def __enter__(self):
        os.chdir(self.texpath)
        return self

    def __exit__(self, *args):
        os.chdir(self.home)

    def compile(self, texdoc, clean=False):
        """ Compile a LaTeX document inside the context manager

        Parameters
        ----------
        texdoc : string
            File name of a .tex file in the LaTeX directory
        clean : bool (default = False)
            When True, all of non-PDF files resulting from compilation
            are removed. By default, they are left on the file system.

        Returns
        -------
        tex : int or None
            The status (1 or 0) of the compilation. If LaTeX is not
            available, None is returned.

        """

        if helpers.checkdep_tex() is not None:
            # use ``pdflatex`` to compile the document
            tex = subprocess.call(['pdflatex', texdoc, '--quiet'],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  shell=False)

            if clean:
                extensions = ['aux', 'log', 'nav', 'out', 'snm', 'toc']
                for ext in extensions:
                    junkfiles = glob.glob('*.{}'.format(ext))
                    for junk in junkfiles:
                        os.remove(junk)

        else:
            tex = None

        return tex


@np.deprecate
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


@np.deprecate
def makeTablesFromCSVStrings(tablestring, texpath=None, csvpath=None):
    """ Takes a string already in CSV format and writes it to a CSV
    and LaTeX files.

    Parameters
    ----------
    tablestring : string
        CSV-formatted string of the data.
    texpath, csvpath : string or None, optional
        Filenames of the output files.

    Returns
    -------
    None

    """

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

@np.deprecate
def addStatsToOutputSummary(csvpath, index_cols=['Date'], na_values='--'):
    """ Reads a CSV file and appends its stats to the bottom.

    Parameters
    ----------
    csvpath : string
        Full path to the CSV file.
    index_cols : list
        List of columns in the CSV to treat as the dataframe's index.
    na_values : string
        How NA/NaN values are represented in the file.

    Returns
    -------
    summary : pandas.DataFrame
        The statistics that were appended to the file.

    """

    # TODO: figure out what to do with -counts-
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

@np.deprecate
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
