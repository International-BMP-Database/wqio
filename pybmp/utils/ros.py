from __future__ import division

import pdb
import os
import sys

if sys.version_info.major == 3:
    from io import StringIO
else:
    from StringIO import StringIO

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas

from .exceptions import DataError


__all__ = ['getTestData', 'rosSort', 'MR']


def getTestData():
    '''
    Generates test data for an ROS estimate.
    Input:
        None

    Output:
        Structured array with the values (results or DLs) and qualifers
        (blank or "ND" for non-detects)
    '''
    raw_csv = StringIO("""res,qual
     2.00,=
     4.20,=
     4.62,=
     5.00,ND
     5.00,ND
     5.50,ND
     5.57,=
     5.66,=
     5.75,ND
     5.86,=
     6.65,=
     6.78,=
     6.79,=
     7.50,=
     7.50,=
     7.50,=
     8.63,=
     8.71,=
     8.99,=
     9.50,ND
     9.50,ND
     9.85,=
     10.82,=
     11.00,ND
     11.25,=
     11.25,=
     12.20,=
     14.92,=
     16.77,=
     17.81,=
     19.16,=
     19.19,=
     19.64,=
     20.18,=
     22.97,=""")

    return pandas.read_csv(raw_csv)


def rosSort(dataframe, rescol='res', qualcol='qual', ndsymbol='ND'):
    '''
    This function prepares a dataframe for ROS. It sorts ascending with
    non-detects on top. something like this:
        [2, 4, 4, 10, 3, 5, 6, 10, 12, 40, 78, 120]
    where [2, 4, 4, 10] being the ND reults (masked the output).

    Input:
        dataframe : a pandas dataframe with results and qualifiers.
            The qualifiers of the dataframe must have two states:
            detect and non-detect.
        rescol (default = 'res') : name of the column in the dataframe
            that contains result values.
        qualcol (default = 'qual') : name of the column in the dataframe
            that containes qualifiers. There must be a single, unique
            qualifer that indicates that a result is non-detect.
        ndsymbol (default = 'ND' : the value in `qualcol` that indicates
            that a value in nondetect. *Important*: any other value will
            be treated as a detection.

    Output:
        Sorted dataframe with a dropped index.
    '''
    # separate detects from non-detects
    nondetects = dataframe[dataframe[qualcol] == ndsymbol].sort(columns=rescol)
    detects = dataframe[dataframe[qualcol] != ndsymbol].sort(columns=rescol)

    # remerge the separated values
    ros_data = nondetects.append(detects)

    return ros_data #.reset_index(drop=True)


class MR(object):
    def __init__(self, data, rescol='res', qualcol='qual', ndsymbol='ND',
                 fitlogs=True, dist='norm'):
        '''
        ROS = ranked-order statistics
        This class implements the MR method outlined Hirsch and Stedinger (1987)
        to estimate the censored (non-detect) values of a dataset. An example
        dataset is available via the `utils.ros.getTestData` function.

        Input:
            data : pandas DataFrame
                The censored dataset for which the non-detect values need to be
                estimated.

            rescol : optional string (default='res')
                The name of the column containing the numerical valuse of the
                dataset. Non-detect values should be set to the detection limit.

            qualcol : optional string (default='qual')
                The name of the column containing the qualifiers marking the
                results as censored.

            ndsymbol : optional string (default='ND')
                The value of the `qualcol` column of `data` that marks as result
                as being censored. In processing, all qualifiers that are equal
                to `ndsymbol` well be set to 'ND'. All other values will be set
                to '='.

        Attributes:
            N_tot : int
                Total number of results in the dataset

            N_nd : int
                Total number of non-detect results in the dataset.

            DLs : pandas DataFrame
                A DataFrame of the unique detection limits found in `data` along
                with the `A`, `B`, `C`, and `PE` quantities computed by the
                estimation.

            data : pandas DataFrame
                An expanded version of the original dataset `data` passed the
                constructor. New columns include the plotting positions,
                Z-score, and estimated data. Additionally `qualcol` and `rescol`
                columns will have been renamed to `qual` and `res`,
                respectively. Also the qualifier values will have been
                standardized per the `ndsymbol` section above.

            debug : pandas DataFrame
                A full version of the `data` DataFrame that inlucdes other
                quantities computed during the estimation such as the "normal"
                and "averaged" ranks and the preliminary Z-score


        Usage:
            >>> from utils import ros
            >>> myData = ros.MR(dataframe, rescol='Result',
                                qualcol='Qualifiers', testing=False)

        Use the `getTestData` function in `utils.ros` to get a the example
        dataset used to test this function.

        Creating an MR object will automatically estimate the final data values.
        Continuing the example above, that data would be accessed via the
        'final_data' column of the `.data` attribute (a pandas DataFrame) of the
        ros.MR object:
            >>> from utils import ros
            >>> myData = ros.MR(...)
            >>> print(myData.data['final_data'])

        Calling ros.MR.plot() will produce a graph comparing the estimated data
        with raw data containing detection limit substituions.
        '''

        def _ros_DL_index(row):
            '''
            Helper function to create an array of indices for the detection
            limits (self.DLs) corresponding to each data point
            '''
            DLIndex = np.zeros(len(self.data.res))
            if self.DLs.shape[0] > 0:
                index, = np.where(self.DLs['DL'] <= row['res'])
                DLIndex = index[-1]
            else:
                DLIndex = 0

            return DLIndex

        if not isinstance(data, pandas.DataFrame):
            raise ValueError("Input `data` must be a pandas.DataFrame")

        if not data.index.is_unique:
            raise ValueError("Index of input DataFrame `data` must be unique")

        if data[rescol].min() <= 0:
            raise ValueError('All result values of `data` must be positive')

        # rename the dataframe columns to the standard names
        # these will be used throughout ros.py when convenient
        newdata = data.rename(columns={rescol: 'res', qualcol: 'qual'})

        # confirm a datatype real quick
        try:
            newdata.res = np.float64(newdata.res)
        except ValueError:
            raise ValueError('Result data is not uniformly numeric')

        # and get the basic info
        self.N_tot = newdata.shape[0]
        self.N_nd = newdata[newdata.qual == ndsymbol].shape[0]

        # clear out all of the non-ND quals
        newdata.qual[newdata.qual != ndsymbol] = '='
        newdata.qual[newdata.qual == ndsymbol] = 'ND'

        # sort the data
        self.data = rosSort(newdata, rescol='res', qualcol='qual',
                            ndsymbol=ndsymbol)

        self.fitlogs = fitlogs
        if isinstance(dist, str):
            self.dist = getattr(stats, dist)
        else:
            self.dist = dist

        # create a dataframe of detection limits and their parameters
        # used in the ROS estimation
        self.DLs = self.cohn()

        # create a DLIndex column that references self.DLs
        self.data['DLIndex'] = self.data.apply(_ros_DL_index, axis=1)

        # compute the ranks of the data
        self._ros_ranks()

        # comput the plotting positions, z-scores, and final values
        self.data = self.estimator()

        # create the debug attribute as a copy of the self.data attribute
        self.debug = self.data.copy(deep=True)

        # select out only the necessary columns for data
        self.data = self.data[['final_data', 'res', 'qual']]

    def cohn(self):
        '''
        Creates an array of unique detection limits in the dataset
        '''

        def _A(row):
            '''
            Helper function to compute the `A` quantity.
            '''
            # index of results above the lower DL
            above = self.data.res >= row['lower']

            # index of results below the upper DL
            below = self.data.res < row['upper']

            # index of non-detect results
            detect = self.data.qual != 'ND'

            # return the number of results where all condictions are True
            return self.data[above & below & detect].shape[0]

        def _B(row):
            '''
            Helper function to compute the `B` quantity
            '''
            # index of data less than the lower DL
            less_than = self.data.res < row['lower']

            # index of data less than or equal to the lower DL
            less_thanequal = self.data.res <= row['lower']

            # index of detects, non-detects
            detect = self.data.qual != 'ND'
            nondet = self.data.qual == 'ND'

            # number results less than or equal to lower DL and non-detect
            LTE_nondets = self.data[less_thanequal & nondet].shape[0]

            # number of results less than lower DL and detected
            LT_detects = self.data[less_than & detect].shape[0]

            # return the sum
            return LTE_nondets + LT_detects

        def _C(row):
            '''
            Helper function to compute the `C` quantity
            '''
            censored_below = self.data.res[self.data.qual == 'ND'] == row['lower']
            return censored_below.sum()

        # unique values
        DLs = pandas.unique(self.data.res[self.data.qual == 'ND'])

        # if there is a results smaller than the minimum detection limit,
        # add that value to the array
        if DLs.shape[0] > 0:
            if self.data.res.min() < DLs.min():
                DLs = np.hstack([self.data.res.min(), DLs])

            # create a dataframe
            DLs = pandas.DataFrame(DLs, columns=['DL'])

            # copy the DLs in two columns. offset the 2nd (upper) column
            DLs['lower'] = DLs['DL']
            if DLs.shape[0] > 1:
                DLs['upper'] = DLs['DL'].shift(-1)

                # fill in the missing values with infinity
                DLs.fillna(value=np.inf, inplace=True)
            else:
                DLs['upper'] = np.inf

            # compute A, B, and C
            DLs['A'] = DLs.apply(_A, axis=1)
            DLs['B'] = DLs.apply(_B, axis=1)
            DLs['C'] = DLs.apply(_C, axis=1)

            # add an extra row
            DLs = DLs.reindex(range(DLs.shape[0]+1))

            # add the 'PE' column, initialize with zeros
            DLs['PE'] = 0.0

        else:
            dl_cols = ['DL', 'lower', 'upper', 'A', 'B', 'C', 'PE']
            DLs = pandas.DataFrame(np.empty((0,7)), columns=dl_cols)

        return DLs

    def _ros_ranks(self):
        '''
        Determine the ranks of the data according to the following logic
        rank[n] = rank[n-1] + 1 when:
            n is 0 OR
            n > 0 and d[n].masked is True and j[n] <> d[n-1] OR
            n > 0 and d[n].masked is False and d[n-1].masked is True OR
            n > 0 and d[n].masked is False and d[n-1].masked is False and j[n] <> j[n-1]

        rank[n] = 1
            n > 0 and d[n].masked is True and j[n] == d[n-1] OR
            n > 0 and d[n].masked is False and d[n-1].masked is False and j[n] == j[n-1]

        where j[n] is the index of the highest DL that is less than the current data value

        Then the ranks of non-censored equivalent data values are averaged.
        '''
        # get the length of the dataset and initialize the normal (raw) ranks
        self.data['Norm Ranks'] = float(self.N_tot)

        #norm_ranks = np.ones(self.N_tot, dtype='f2')

        # loop through each value and compare to the previous value
        # see docstring for more info on the logic behind all this
        for n, index in enumerate(self.data.index):
            if n == 0 \
            or self.data['DLIndex'].iloc[n] != self.data['DLIndex'].iloc[n-1] \
            or self.data.qual.iloc[n] != self.data.qual.iloc[n-1]:
                self.data.loc[index, 'Norm Ranks'] = 1
            else:
                self.data.loc[index, 'Norm Ranks'] = self.data['Norm Ranks'].iloc[n-1] + 1

        # go through each index and see if the value is a detection
        # and average the ranks of all equivalent values,
        def avgrank(r):
            if r['qual'] != 'ND':
                index = (self.data.DLIndex == r['DLIndex']) & \
                        (self.data.res == r['res']) & \
                        (self.data.qual != 'ND')
                return self.data['Norm Ranks'][index].mean()
            else:
                return r['Norm Ranks']

        self.data['Avg Ranks'] = self.data.apply(avgrank, axis=1)

    def estimator(self):
        '''
        Estimates the values of the censored data
        '''

        def _ros_plotting_pos(row):
            '''
            Helper function to compute the ROS'd plotting position
            '''
            dl_1 = self.DLs.iloc[row['DLIndex']]
            dl_2 = self.DLs.iloc[row['DLIndex']+1]
            if row['qual'] == 'ND':
                return (1 - dl_1['PE']) * row['Norm Ranks']/(dl_1['C']+1)
            else:
                return (1 - dl_1['PE']) + (dl_1['PE'] - dl_2['PE']) * \
                        row['Norm Ranks'] / (dl_1['A']+1)

        def _select_final_data(row):
            '''
            Helper fucntion to select "final" data from original detects
            and estimated non-detects
            '''
            if row['qual'] == 'ND':
                return row['modeled_data']
            else:
                return row['res']

        def _select_half_DLs(row):
            '''
            Helper function to select half DLs when there are too few detects
            '''
            if row['qual'] == 'ND':
                return 0.5 * row['res']
            else:
                return row['res']

        # detect/non-detect selectors
        detect_selector = self.data.qual != 'ND'
        nondet_selector = self.data.qual == 'ND'

        # if there are no non-detects, just spit everything back out
        if self.N_nd == 0:
            self.data['final_data'] = self.data['res']

        # if there are too few detects, use half DL
        elif self.N_tot - self.N_nd < 2 or self.N_nd/self.N_tot > 0.8:
            self.data['final_data'] = self.data.apply(_select_half_DLs, axis=1)

        # in most cases, actually use the MR method to estimate NDs
        else:
            # compute the PE values
            for j in self.DLs.index[:-1][::-1]:
                self.DLs.loc[j, 'PE'] = self.DLs.loc[j+1, 'PE'] + \
                   self.DLs.loc[j, 'A'] / \
                   (self.DLs.loc[j, 'A'] + self.DLs.loc[j, 'B']) * \
                   (1 - self.DLs.loc[j+1, 'PE'])

            # compute the plotting position of the data (uses the PE stuff)
            self.data['plot_pos'] = self.data.apply(_ros_plotting_pos, axis=1)

            # correctly sort the plotting positions of the ND data:
            ND_plotpos = self.data['plot_pos'][self.data['qual'] == 'ND']
            ND_plotpos.values.sort()
            self.data['plot_pos'][self.data['qual'] == 'ND'] = ND_plotpos

            # estimate a preliminary value of the Z-scores
            self.data['Zprelim'] = self.dist.ppf(self.data['plot_pos'])

            # fit a line to the logs of the detected data
            if self.fitlogs:
                detect_vals = np.log(self.data['res'][detect_selector])
            else:
                detect_vals = self.data['res'][detect_selector]
            fit = stats.linregress(self.data['Zprelim'][detect_selector],
                                   detect_vals)

            # save the fit params to an attribute
            self.fit = fit

            # pull out the slope and intercept for use later
            slope, intercept = fit[:2]

            # model the data based on the best-fit curve
            self.data['modeled_data'] = np.exp(
                slope*self.data['Zprelim'][nondet_selector] + intercept
            )

            # select out the final data
            self.data['final_data'] = self.data.apply(
                _select_final_data,
                axis=1
            )

        return self.data

    def plot(self, filename):
        '''
        makes a simple plot showing the original and modeled data
        '''
        fig, ax1 = plt.subplots()
        ax1.plot(self.data.Z[self.data.qual != 'ND'],
                 self.data.res[self.data.qual != 'ND'],
                 'ko', mfc='Maroon', ms=6, label='original detects', zorder=8)

        ax1.plot(self.data.Z[self.data.qual == 'ND'],
                 self.data.res[self.data.qual == 'ND'],
                 'ko', ms=6, label='original non-detects', zorder=8, mfc='none')

        ax1.plot(self.data.Z, self.data.final_data, 'ks', ms=4, zorder=10,
                 label='modeled data', mfc='DodgerBlue')

        ax1.set_xlabel(r'$Z$-score')
        ax1.set_ylabel('concentration')
        ax1.set_yscale('log')
        ax1.legend(loc='upper left', numpoints=1)
        ax1.xaxis.grid(True, which='major', ls='-', lw=0.5, alpha=0.35)
        ax1.yaxis.grid(True, which='major', ls='-', lw=0.5, alpha=0.35)
        ax1.yaxis.grid(True, which='minor', ls='-', lw=0.5, alpha=0.17)
        plt.tight_layout()
        fig.savefig(filename)
        return fig


class arrayMR:
    def __init__(self, data, rescol='res', qualcol='qual', ndsymbol='ND'):
        '''
        ROS = ranked-order statistics
        This class implements the MR method outlined Hirsch and Stedinger (1987) to estimate
        the censored (non-detect) values of a dataset.
        Usage:
            >>> from utils import ros
            >>> myData = ros.MR(dataframe, rescol='Result', qualcol='Qualifiers', testing=False)

            Setting the 'testing' parameter to True will ignore your data and instead
            provide you with a hard-coded dataset stored in the ros.getData function.
            Otherwise, the MR class will use the provided dataframe whose columns containing
            the results and qualifier values are contained in `rescol` and `qualcol`,
            respectively.

        Creating an MR object will automatically estimate the final data values. Continuing
        the example above, that data would be accessed via the 'final_data' attribute of the
        ros.MR object:
            >>> from utils import ros
            >>> myData = ros.MR(...)
            >>> print(myData.final_data)

        The other public attributes of the ros.MR object are as follows:
            ros.MR.raw_vals - The raw data pulled from the databse. Detection limits are used
                              with non-detect ('U' or 'UJ' flagged) data.
            ros.MR.mask - Array of boolean values indicating if a ros.MR.raw_val is censored
            ros.MR.data - A numpy MaskedArray object with the censored data masked.
            ros.MR.DLs - Array of the uniqie detection limits of the dataset. If a
                         non-censored value exists that is lower than the minimum detection
                         limit, it is appended on to this array.
            ros.MR.DLIndex - Array of indices for each ros.MR.raw_val that points to the
                             corresponding detection limit.
            ros.MR.nrakks - Array of unaveraged ranks for the data
            ros.MR.aranks - Array of averaged (final) ranks for the data
            ros.MR.final_data - Array of data where the censored values have been replaced
                                with estimation.

        Called ros.MR.plot() will produce a graph comparing the estimated data with raw data
        containing detection limit substituions.
        '''
        # rename the dataframe columns to the standard names used throughout ros.py
        newdata = data.rename(columns={rescol: 'res', qualcol: 'qual'})
        try:
            newdata.res = np.float64(newdata.res)
        except ValueError:
            raise
        newdata.qual[newdata.qual != ndsymbol] = ''

        # sort the data and get the basic info
        self.ndsymbol = ndsymbol
        self.data = rosSort(newdata, rescol='res', qualcol='qual',
                            ndsymbol='ND')
        self.N = data.shape[0]
        self.DLs = self.cohn()
        self.DLIndex = self._ros_DL_index()

        self.nondetect_index = self.data['qual'] == 'ND'
        self.detect_index = self.data['qual'] != 'ND'

        # compute the ranks of the data
        self.nranks, self.aranks = self._ros_ranks()

        # comput the plotting positions, z-scores, and final values
        self.plot_pos, self.Z, self.final_data = self.estimator()

    def cohn(self):
        '''
        Creates an array of unique detection limits in the dataset
        '''
        # unique values
        DLs = pandas.unique(self.data.res[self.data.qual == 'ND'])

        # if there is a results smaller than the minimum detection limit,
        # add that value to the array
        if DLs.shape[0] > 0 and self.data.res.min() < DLs.min():
            DLs = np.hstack([self.data.res.min(), DLs])

        return DLs

    def _ros_ranks(self):
        '''
        Determine the ranks of the data according to the following logic
        rank[n] = rank[n-1] + 1 when:
            n is 0 OR
            n > 0 and d[n].masked is True and j[n] <> d[n-1] OR
            n > 0 and d[n].masked is False and d[n-1].masked is True OR
            n > 0 and d[n].masked is False and d[n-1].masked is False and j[n] <> j[n-1]

        rank[n] = 1
            n > 0 and d[n].masked is True and j[n] == d[n-1] OR
            n > 0 and d[n].masked is False and d[n-1].masked is False and j[n] == j[n-1]

        where j[n] is the index of the highest DL that is less than the current data value

        Then the ranks of non-censored equivalent data values are averaged.
        '''
        # get the length of the dataset and initialize the normal (raw) ranks
        N = self.data.res.shape[0]
        norm_ranks = np.ones(N, dtype='f2')

        # loop through each value and compare to the previous value
        # see docstring for more info on the logic behind all this
        for n in range(N):
            if n == 0 \
                or self.DLIndex[n] != self.DLIndex[n-1] \
                    or self.data.qual.irow(n) != self.data.qual.irow(n-1):
                norm_ranks[n] = 1
            else:
                norm_ranks[n] = norm_ranks[n-1] + 1

        # make a fresh copy of the rank array for modification
        # go through each index and see if the value is a detection
        # and average the ranks of all equivalent values,
        avgd_ranks = norm_ranks.copy()
        for n in range(N):
            if self.data.qual.irow(n) != 'ND':
                thisres = self.data.res.irow(n)

                index = (self.DLIndex==self.DLIndex[n]) & \
                        (self.data.res==thisres) & \
                        (self.data.qual != 'ND')

                avgd_ranks[index] = norm_ranks[index].mean()

        self.data['nranks'] = norm_ranks
        self.data['aranks'] = avgd_ranks
        return norm_ranks, avgd_ranks

    def _ros_DL_index(self):
        '''
        creates an array of indices for the detection limits (self.DLs)
        corresponding to each data point
        '''
        # pdb.set_trace()
        DLIndex = np.zeros(len(self.data.res))
        if self.DLs.shape[0] > 0:
            for n in range(self.N):
                value = self.data.res.irow(n)
                censored = self.data.qual.irow(n) == 'ND'
                index, = np.where(self.DLs <= value)
                DLIndex[n] = index[-1]
        self.data['DLIndex'] = DLIndex

        return DLIndex

    def _compute_ABC(self, det_lims):
        '''
        Intermediate values needed to compute plotting positions.
        TODO: get better terminolgy for these
        '''
        lowerDL = np.min(det_lims)
        upperDL = np.max(det_lims)

        A_above = self.data[self.data.res >= lowerDL]
        A_above_and_below = A_above[A_above.res < upperDL]
        A_uncensored = A_above_and_below[A_above_and_below.qual != 'ND']
        A = A_uncensored.shape[0]

        B_LT = self.data[self.data.res < lowerDL]
        B_LTE = self.data[self.data.res <= lowerDL]
        B = B_LTE[B_LTE.qual == 'ND'].shape[0] + B_LT[B_LT.qual != 'ND'].shape[0]

        censored_below = self.data.res[self.data.qual == 'ND'] == lowerDL
        # pdb.set_trace()
        C = censored_below.sum()

        return float(A), float(B), float(C)

    def estimator(self):
        '''
        Estimates the values of the censored data
        '''
        N_tot = self.data.shape[0]
        N_nd = self.data[self.data.qual == 'ND'].shape[0]
        plot_pos = np.zeros(N_tot)

        if N_nd == 0:
            modeled_data = np.array([])
            #Z = np.array([])
            fit = []
        elif N_tot - N_nd < 2 or N_nd/N_tot > 0.8:
            modeled_data = self.data.res[self.data.qual == 'ND'] * 0.5
            Z = np.array([])
            fit = []

        else:
            num_DLs = self.DLs.shape[0]

            PE = np.zeros(num_DLs+1)
            A = np.zeros(num_DLs)
            B = np.zeros(num_DLs)
            C = np.zeros(num_DLs)
            for j in range(num_DLs-1, -1, -1):
                #define A and B
                try:
                    det_lims = (self.DLs[j], self.DLs[j+1])
                except IndexError:
                    det_lims = (self.DLs[j], np.inf)

                A[j], B[j], C[j] = self._compute_ABC(det_lims)
                PE[j] = PE[j+1] + A[j]/(A[j]+B[j]) * (1-PE[j+1])

            self.A, self.B, self.C, self.PE = A, B, C, PE
            for n in range(N_tot):
                j = self.DLIndex[n]
                if self.data.qual[n] == 'ND':
                    plot_pos[n] = (1-PE[j]) * self.aranks[n]/(C[j]+1)

                else:
                    plot_pos[n] = (1-PE[j]) + (PE[j] - PE[j+1]) * self.aranks[n] / (A[j]+1)

            Zprelim = np.ma.MaskedArray(data=stats.distributions.norm.ppf(plot_pos), mask=self.data.qual == 'ND')
            fit = stats.mstats.linregress(Zprelim, np.array(np.log(self.data.res)))
            self.slope, self.intercept = fit[:2]
            modeled_data = np.exp(self.slope*Zprelim.data[Zprelim.mask] + self.intercept)
            self.Zprelim = Zprelim

        # pdb.set_trace()
        full_data = np.hstack([modeled_data, self.data.res[self.data.qual != 'ND']])
        Zfinal, final_data = stats.probplot(full_data, fit=0)
        return plot_pos, Zfinal, final_data


def compfig(mr):
    fig, ax1 = plt.subplots()

    ax1.plot(mr.data.Z, mr.data.final_data, linestyle='none',
             marker='o', markeredgecolor='CornflowerBlue', markerfacecolor='none',
             markersize=8, markeredgewidth=2, label='Estimated Data')
    ax1.plot(mr.data.Z, mr.data.res, linestyle='none',
             marker='.', markeredgecolor='DarkGreen', markerfacecolor='DarkGreen',
             markersize=8, label='Original Data')
    ax1.legend(loc='upper left')
    ax1.set_yscale('log')
    ax1.set_xlabel('Z-score')
    ax1.set_ylabel('Concentration (units)')
    fig.tight_layout()
    fig.savefig('rostest.png', dpi=300)
