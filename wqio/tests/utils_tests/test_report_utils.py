import os
import sys
import glob
from textwrap import dedent

import pandas

import pytest
import numpy.testing as nptest
from wqio import testing

from wqio.utils import reportutils, numutils


def test__sig_figs_helper():
    x = 1.2
    assert reportutils._sig_figs(x) == numutils.sigFigs(x, 3, tex=True)


def test_sanitizeTex():
    inputstring = r""" \
    \$x\_\{4\}\textasciicircum\{2\} \textbackslashtimes \% ug/L$ \textbackslash \\
    """
    desiredstring = r""" \
    $x_{4}^{2} \times \% \si[per-mode=symbol]{\micro\gram\per\liter}$  \tabularnewline
    """

    assert reportutils.sanitizeTex(inputstring) == desiredstring


class Tests_with_paths(object):
    def setup(self):
        if os.path.split(os.getcwd())[-1] == 'src':
            self.prefix = os.path.join('.', 'utils', 'tests')
        else:
            self.prefix = os.path.join('..', 'utils', 'tests')

        self.tablestring = "Date,A,B,C,D\nX,1,2,3,4\nY,5,6,7,8\nZ,9,0,1,2"
        self.testcsvpath = testing.test_data_path('testtable.csv')
        with open(self.testcsvpath, 'w') as testfile:
            testfile.write(self.tablestring)

    @pytest.mark.skipif(True, reason="WIP")
    def test_makeBoxplotLegend(self):
        reportutils.makeBoxplotLegend(testing.test_data_path('bplegendtest'))

    def test_constructPath(self):
        testpath = reportutils.constructPath('test1 2', 'pdf', 'cvc', 'output')
        expectedpath = os.path.join('cvc', 'output', 'img', 'test12.pdf')
        assert testpath == expectedpath

    def test_makeTablesFromCSVStrings(self):
        reportutils.makeTablesFromCSVStrings(self.tablestring,
                    texpath=testing.test_data_path('testtable.tex'),
                    csvpath=self.testcsvpath)

    def test_addStatsToOutputSummary(self):
        inputcols = pandas.read_csv(self.testcsvpath).columns.tolist()
        summary = reportutils.addStatsToOutputSummary(self.testcsvpath)
        known_index = ['X', 'Y', 'Z', 'count', 'mean', 'std',
                       'min', '25%', '50%', '75%', 'max']
        assert summary.index.tolist() == known_index
        assert inputcols[1:] == summary.columns.tolist()

    def test_csvToTex(self):
        testfile = 'testtable_toTex.tex'
        reportutils.csvToTex(self.testcsvpath, testing.test_data_path(testfile))

        if sys.platform == 'win32':
            knownfile = 'testtable_toTex_KnownWin.tex'
        else:
            knownfile = 'testtable_toTex_KnownUnix.tex'

        with open(testing.test_data_path(testfile), 'r') as test, \
             open(testing.test_data_path(knownfile), 'r') as known:
            assert test.read() == known.read()

    def test_csvToXlsx(self):
        reportutils.csvToXlsx(self.testcsvpath, testing.test_data_path('testtable_toXL.xlsx'))

    def test_addExternalValueToOutputSummary(self):
        compardict = {'test1': 10, 'test2': 11}
        comparecol = 'A'
        reportutils.addExternalValueToOutputSummary(self.testcsvpath, compardict,
                                              comparecol, index_cols=['Date'])

        known = 'Date,A,B,C,D\nX,1,2.0,3.0,4.0\nY,5,6.0,7.0,8.0\n' \
                'Z,9,0.0,1.0,2.0\ntest1,10,--,--,--\ntest2,11,--,--,--\n'

        with open(self.testcsvpath, 'r') as outfile:
            test = outfile.read()

        assert test == known


def test_makeTexTable_normal():
    known = dedent(r"""
        \begin{table}[h!]
            \rowcolors{1}{CVCWhite}{CVCLightGrey}
            \caption{test caption}
            \centering
            \input{fake.tex}
        \end{table}


    """)

    test = reportutils.makeTexTable('fake.tex', 'test caption')
    assert known == test


def test_makeTexTable_allOptions():
    known = dedent(r"""
    \begin{sidewaystable}[bt]
        \rowcolors{1}{CVCWhite}{CVCLightGrey}
        \caption{test caption}
        \centering
        \input{fake.tex}
    \end{sidewaystable}
    test footnote
    \clearpage
    """)
    test = reportutils.makeTexTable('fake.tex', 'test caption', sideways=True,
           footnotetext='test footnote', clearpage=True, pos='bt')
    assert known == test


class Test_makeLongLandscapeTexTable(object):
    def setup(self):
        self.maxDiff = None
        dfdict = {
            'W': {
                'a': 0.84386963791251501,
                'b': -0.22109837444207142,
            },
            'X': {
                'a': 0.70049867715201963,
                'b': 1.4764939161054218,
            },
            'Y': {
                'a': -1.3477794473987552,
                'b': -1.1939220296611821,
            },
        }
        self.df = pandas.DataFrame.from_dict(dfdict)
        self.known_nofootnote = dedent(r"""
            \begin{landscape}
                \centering
                \rowcolors{1}{CVCWhite}{CVCLightGrey}
                \begin{longtable}{lcc}
                    \caption{test caption} \label{label} \\
                    \toprule
                    \multicolumn{1}{l}{W} &
                    \multicolumn{1}{p{16mm}}{X} &
                    \multicolumn{1}{p{16mm}}{Y} \\
                    \toprule
                    \endfirsthead

                    \multicolumn{3}{c}
                    {{\bfseries \tablename\ \thetable{} -- continued from previous page}} \\
                    \toprule
                    \multicolumn{1}{l}{W} &
                    \multicolumn{1}{p{16mm}}{X} &
                    \multicolumn{1}{p{16mm}}{Y} \\
                    \toprule
                    \endhead

                    \toprule
                    \rowcolor{CVCWhite}
                    \multicolumn{3}{r}{{Continued on next page...}} \\
                    \bottomrule
                    \endfoot

                    \bottomrule
                    \endlastfoot

             0.844 & 0.700 & -1.35 \\
            -0.221 &  1.48 & -1.19 \\

                \end{longtable}
            \end{landscape}

            \clearpage
        """)

        self.known_withfootnote = dedent(r"""
            \begin{landscape}
                \centering
                \rowcolors{1}{CVCWhite}{CVCLightGrey}
                \begin{longtable}{lcc}
                    \caption{test caption} \label{label} \\
                    \toprule
                    \multicolumn{1}{l}{W} &
                    \multicolumn{1}{p{16mm}}{X} &
                    \multicolumn{1}{p{16mm}}{Y} \\
                    \toprule
                    \endfirsthead

                    \multicolumn{3}{c}
                    {{\bfseries \tablename\ \thetable{} -- continued from previous page}} \\
                    \toprule
                    \multicolumn{1}{l}{W} &
                    \multicolumn{1}{p{16mm}}{X} &
                    \multicolumn{1}{p{16mm}}{Y} \\
                    \toprule
                    \endhead

                    \toprule
                    \rowcolor{CVCWhite}
                    \multicolumn{3}{r}{{Continued on next page...}} \\
                    \bottomrule
                    \endfoot

                    \bottomrule
                    \endlastfoot

             0.844 & 0.700 & -1.35 \\
            -0.221 &  1.48 & -1.19 \\

                \end{longtable}
            \end{landscape}
            test note
            \clearpage
        """)

    def test_makeLongLandscapeTexTable_noFootnote(self):
        test = reportutils.makeLongLandscapeTexTable(self.df, 'test caption', 'label')
        assert self.known_nofootnote == test

    def test_makeLongLandscapeTexTable_Footnote(self):
        test = reportutils.makeLongLandscapeTexTable(self.df, 'test caption', 'label', 'test note')
        assert self.known_withfootnote == test


def test_makeTexFigure():
    known = dedent(r"""
    \begin{figure}[hb]   % FIGURE
        \centering
        \includegraphics[scale=1.00]{fake.pdf}
        \caption{test caption}
    \end{figure}         % FIGURE
    \clearpage
    """)
    test = reportutils.makeTexFigure('fake.pdf', 'test caption')
    assert known == test


def test_processFilename():
    startname = 'This-is a, very+ &dumb$_{test/name}'
    endname = 'This-isaverydumbtestname'
    assert endname == reportutils.processFilename(startname)


class Test_LaTeXDirectory(object):
    def setup(self):
        self.origdir = os.getcwd()
        self.deepdir = os.path.join(os.getcwd(), 'test1', 'test2', 'test3')
        self.deepfile = os.path.join(self.deepdir, 'f.tex')
        tex_content = dedent(r"""
            \documentclass[12pt]{article}
            \usepackage{lingmacros}
            \usepackage{tree-dvips}
            \begin{document}

            \section*{Notes for My Paper}

            Don't forget to include examples of topicalization.
            They look like this:

            {\small
            \enumsentence{Topicalization from sentential subject:\\
            \shortex{7}{a John$_i$ [a & kltukl & [el &
              {\bf l-}oltoir & er & ngii$_i$ & a Mary]]}
            { & {\bf R-}clear & {\sc comp} &
              {\bf IR}.{\sc 3s}-love   & P & him & }
            {John, (it's) clear that Mary loves (him).}}
            }

            \subsection*{How to handle topicalization}

            I'll just assume a tree structure like (\ex{1}).

            {\small
            \enumsentence{Structure of A$'$ Projections:\\ [2ex]
            \begin{tabular}[t]{cccc}
                & \node{i}{CP}\\ [2ex]
                \node{ii}{Spec} &   &\node{iii}{C$'$}\\ [2ex]
                    &\node{iv}{C} & & \node{v}{SAgrP}
            \end{tabular}
            \nodeconnect{i}{ii}
            \nodeconnect{i}{iii}
            \nodeconnect{iii}{iv}
            \nodeconnect{iii}{v}
            }
            }

            \subsection*{Mood}

            Mood changes when there is a topic, as well as when
            there is WH-movement.  \emph{Irrealis} is the mood when
            there is a non-subject topic or WH-phrase in Comp.
            \emph{Realis} is the mood when there is a subject topic
            or WH-phrase.

            \end{document}
        """)

        if os.path.exists(self.deepdir):
            self.teardown()

        os.makedirs(self.deepdir)
        with open(self.deepfile, 'w') as dfile:
            dfile.write(tex_content)

    def teardown(self):
        allfiles = glob.glob(os.path.join(self.deepdir, "f.*"))
        for af in allfiles:
            os.remove(af)
        os.removedirs(self.deepdir)

    def test_dir(self):
        assert os.getcwd() == self.origdir
        with reportutils.LaTeXDirectory(self.deepdir):
            assert os.getcwd() == self.deepdir

        assert os.getcwd() == self.origdir

    def test_file(self):
        assert os.getcwd() == self.origdir
        with reportutils.LaTeXDirectory(self.deepfile):
            assert os.getcwd() == self.deepdir

        assert os.getcwd() == self.origdir

    @pytest.mark.skipif(testing.checkdep_tex() is None, reason='No LaTeX')
    def test_compile_smoke(self):
        with reportutils.LaTeXDirectory(self.deepfile) as latex:
            latex.compile(self.deepfile)
