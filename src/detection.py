import numpy as np
from sklearn.neighbors import NearestNeighbors, BallTree, KDTree
from sklearn.cluster import KMeans
from scipy.stats import mannwhitneyu
from sklearn.random_projection import GaussianRandomProjection


class DataCopyingDetector:

    def __init__(self, lmbda=20, gamma=1 / 4000, k=1, d_proj=2):
        """
        Data Copying Detector class.

        Parameters
        ----------
        lmbda : float
            Characterizes the rate at which q must overrepresent points close to a point x.
        gamma : float
            Characterizes the maximum size (by probability mass) of a region that is considered
            to be data copying
        k : int
            Number of random projections
        d_proj : int
            Dimensionality of the target space of the projection, if None no projection is applied
        """
        if lmbda < 1:
            raise ValueError("lmbda must be greater than 1.")
        else:
            self.lmbda = lmbda
        if gamma < 0 or gamma > 1:
            raise ValueError("gamma must be between 0 and 1.")
        else:
            self.gamma = gamma
        self.k = k
        self.d_proj = d_proj

    def estimate_cr(self, S, q, m=20000, type="original"):
        """
        Estimates the copying rate of a generative model q from a training sample S.

        Parameters
        ----------
        S : np.array
            The training sample
        q : GenerativeModel
            The generative model to estimate the copying rate of. Samples can be generated using q.sample()
        m : int
            The number of samples to generate from q
        type : str | float
            The type of algorithm to use.
            Options are:
                - "original": runs the original algorithm by Bhattacharjee et al. (2023)
                - "set":
                - "mean":
                - "median":
                - "max":
                - "min":
                - "majority":

        Returns
        -------
        float
            The estimated copying rate
        """
        T = q.sample(m, S="test")
        U = q.sample(m, S="val")
        if type == "original":
            return len(self.get_copying_indices(S, T, U)) / m
        elif type == "set":
            return self.cr_set(S, T, U)
        else:
            raise ValueError("Type not recognized.")

    def cr_majority(self, S, T, U):
        """
        Parameters
        ----------
        S : np.array
            The training sample
        T : np.array
            A synthetic sample from the generative model to estimate r_star
        U : np.array
            A second synthetic sample from the generative model to estimate the copying rate

        Returns
        -------
        float
            The estimated copying rate
        """
        idx = np.array(np.zeros(len(U), self.k)).astype(bool)

        for i in range(self.k):
            proj = GaussianRandomProjection(self.d_proj)
            proj.fit(S)
            S_proj = proj.transform(S)
            T_proj = proj.transform(T)
            U_proj = proj.transform(U)
            idx[:, i] = self.get_copying_indices(S_proj, T_proj, U_proj)

        return (idx.sum(axis=1) > self.k / 2).mean()

    def cr_set(self, S, T, U):
        """
        Estimates the copying rate of a generative model q from a training sample S.

        Parameters
        ----------
        S : np.array
            The training sample
        q : GenerativeModel
            The generative model to estimate the copying rate of. Samples can be generated using q.sample()
        m : int
            The number of samples to generate from q

        Returns
        -------
        float
            The estimated copying rate
        """
        indices = set()
        for _ in range(self.k):
            proj = GaussianRandomProjection(self.d_proj)
            proj.fit(S)
            S_proj = proj.transform(S)
            T_proj = proj.transform(T)
            U_proj = proj.transform(U)
            indices.update(self.get_copying_indices(S_proj, T_proj, U_proj))

        return len(indices) / len(U)

    def get_copying_indices(self, S, T, U):
        """
        Extracts the indices of the copied points in U.

        Parameters
        ----------
        S : np.array
            The training sample
        T : np.array
            A synthetic sample from the generative model to estimate r_star
        U : np.array
            A second synthetic sample from the generative model to estimate the copying rate

        Returns
        -------
        np.array
            The indices of the copied points
        """
        # sanity checks for the dimensions
        if not S.shape[1] == T.shape[1] == U.shape[1]:
            raise ValueError("The dimensions of the samples do not match.")
        if self.d_proj == 1:
            return self._get_copying_indices_1D(S, T, U)
        else:
            return self._get_copying_indices(S, T, U)

    def _get_copying_indices_1D(self, S, T, U):
        """
        Faster implementation for 1D data.

        Parameters
        ----------
        S : np.array
            The training sample
        T : np.array
            A synthetic sample from the generative model to estimate r_star
        U : np.array
            A second synthetic sample from the generative model to estimate the copying rate

        Returns
        -------
        np.array
            The indices of the copied points
        """
        assert S.shape[1] == 1
        assert T.shape[1] == 1
        assert U.shape[1] == 1

        # sort the samples
        S_sorted = np.sort(S, axis=0)
        T_sorted = np.sort(T, axis=0)
        idx_sorted = np.argsort(U, axis=0)
        U_sorted = U[idx_sorted]

        b = 400  # following the paper
        lower_idx_S, upper_idx_S = 0, b

        copied_indices = set()
        for i, x in enumerate(S_sorted):
            ### Find r_max
            r_star = max(x - S_sorted[lower_idx_S], S_sorted[upper_idx_S] - x)
            r = self.gamma * r_star * (len(S) / b)
            p_r = self.gamma

            ## TODO: check if really all points are included
            lower_idx_T, upper_idx_T = np.searchsorted(T_sorted, [x - r, x + r + 1e-10])
            count_T = upper_idx_T - lower_idx_T
            q_r = count_T / len(T)

            ## radii candidates and sort them in descending order
            radii = np.abs(T_sorted[lower_idx_T:upper_idx_T] - x)
            radii = np.append(-np.sort(-radii), 0)

            ## iterate over the radii
            for r_candidate in radii:
                if self.lmbda * p_r <= q_r:
                    break
                else:
                    r = r_candidate
                    p_r = b * r / (len(S) * r_star)
                    lower_idx_T, upper_idx_T = np.searchsorted(
                        T_sorted, [x - r, x + r + 1e-10]
                    )
                    count_T = upper_idx_T - lower_idx_T
                    q_r = count_T / len(T)

            ### add the indices of the copied points in U to the set
            lower_idx_U, upper_idx_U = np.searchsorted(U_sorted, [x - r, x + r + 1e-10])
            copied_indices.update(idx_sorted[lower_idx_U:upper_idx_U])

            ### update the indices of S
            if (
                upper_idx_S < len(S) - 1
                and S_sorted[upper_idx_S + 1] - S_sorted[i + 1]
                < S_sorted[i + 1] - S_sorted[lower_idx_S]
            ):
                upper_idx_S += 1
                lower_idx_S += 1
            else:
                continue

        return np.array(list(copied_indices))

    def _get_copying_indices(self, S, T, U):
        """
        General implementation for higher dimensions.

        Parameters
        ----------
        S : np.array
            The training sample
        T : np.array
            A synthetic sample from the generative model to estimate r_star
        U : np.array
            A second synthetic sample from the generative model to estimate the copying rate

        Returns
        -------
        np.array
            The indices of the copied points
        """
        n, d = S.shape
        # d = d - 1
        b = 400  # following the paper
        S_tree = KDTree(S)
        T_tree = KDTree(T)

        if self.gamma * n < b:
            radii_max = S_tree.query(S, k=b)[0][:, -1]
            radii_p_gamma = (self.gamma * n * radii_max**d / b) ** (1 / d)
        else:
            radii_p_gamma = S_tree.query(S, k=int(self.gamma * n))[0][:, -1]

        _, radii_candidates = T_tree.query_radius(
            S, radii_p_gamma, return_distance=True, sort_results=True
        )
        radii = np.zeros(n)

        for i in range(n):
            if len(radii_candidates[i]) == 0:
                continue
            else:
                for j in range(len(radii_candidates[i]), 0, -1):
                    q_i_s = j / len(T)
                    p_i_s = (b * radii_candidates[i][j - 1] ** d) / (
                        n * radii_max[i] ** d
                    )
                    # TODO: p_i has to be calculated then by the number of points in the ball i.e. querying the tree which is slower
                    if self.lmbda * p_i_s <= q_i_s:
                        radii[i] = radii_candidates[i][j - 1]
                        break

        U_tree = KDTree(U)
        return np.unique(np.concatenate(U_tree.query_radius(S, radii)).ravel())

    def cr_slow(self, S: np.ndarray, q: object, m: int) -> float:
        """
        DEPRICATED: But clean code without any loops. This is the slowest implementation.
        Estimates the data copying rate. Algorithm by Bhattacharjee et al. (2023)

        Parameters
        ----------
        S : BallTree or numpy.ndarray
            The training data to estimate the data copying rate on.
        q : Object
            The fitted generative model. Samples can be generated by calling q.sample()
        m : int
            The number of samples to generate from q to create sets T and U.
        verbose : bool
            Whether to print the progress.

        Returns
        -------
        float
            The estimated data copying rate.
        """
        n, d = S.shape
        S_tree = KDTree(S)
        T = q.sample(m, S="test")
        T_tree = KDTree(T)
        b = 400  # following the paper
        n_radii = int(m / 20)  # number of radii candidates
        # TODO: this should depend gamma and n

        radii, _ = T_tree.query(S, k=n_radii)
        radii = np.flip(np.c_[np.zeros((n, 1)), radii], axis=1)

        radii_S, _ = S_tree.query(S, k=b)
        radii_max = radii_S[:, -1].reshape(-1, 1)
        pi_s = (b * radii**d) / (n * radii_max**d)  # regularity assumption
        qi_s = np.tile(np.array([i for i in range(n_radii, -1, -1)]), (n, 1)) / m

        bools = np.logical_and(pi_s <= self.gamma, self.lmbda * pi_s <= qi_s)
        radii = radii[np.arange(n), np.argmax(bools, axis=1)]

        T_tree = KDTree(q.sample(m, S="val"))
        return len(np.unique(np.concatenate(T_tree.query_radius(S, radii)).ravel())) / m

    # def cr_1D(self, S, q, m=20000):
    #     """
    #     Estimates the copying rate of a generative model q from a training sample S using a single random projection
    #     onto one dimension. This is a simplified version of the algorithm by Bhattacharjee et al. (2023) for 1D data.

    #     Parameters
    #     ----------
    #     S : np.array
    #         The training sample
    #     q : GenerativeModel
    #         The generative model to estimate the copying rate of. Samples can be generated using q.sample()
    #     m : int
    #         The number of samples to generate from q to create sets T and U

    #     Returns
    #     -------
    #     float
    #         The estimated copying rate
    #     """
    #     n, _ = S.shape
    #     b = 400  ### TODO: discuss with robi
    #     count = 0  # count of points considered as copied
    #     rp = GaussianRandomProjection(1)

    #     ### sample from generative model
    #     ### sort every dataset in ascending order (O(max(n, m)log(max(n, m)))
    #     S = np.sort(rp.fit_transform(S), axis=None)
    #     T = np.sort(rp.transform(q.sample(n_samples=m)), axis=None)
    #     U = np.sort(rp.transform(q.sample(n_samples=m)), axis=None)

    #     ### iterate over the points in S
    #     lower_idx_S, upper_idx_S = 0, b

    #     for i, x in enumerate(S):

    #         ### Find r_max
    #         # TODO: Only do this if b/n > gamma
    #         r_star = max(x - S[lower_idx_S], S[upper_idx_S] - x)
    #         r = self.gamma * r_star * (n / b)
    #         p_r = self.gamma

    #         ## TODO: check if really all points are included
    #         lower_idx_T, upper_idx_T = np.searchsorted(T, [x - r, x + r + 1e-10])
    #         count_T = upper_idx_T - lower_idx_T
    #         q_r = count_T / m

    #         ## radii candidates and sort them in descending order
    #         radii = np.abs(T[lower_idx_T:upper_idx_T] - x)
    #         radii = np.append(-np.sort(-radii), 0)

    #         ## iterate over the radii
    #         for r_candidate in radii:
    #             if self.lmbda * p_r <= q_r:
    #                 break
    #             else:
    #                 r = r_candidate
    #                 p_r = b * r / (n * r_star)
    #                 lower_idx_T, upper_idx_T = np.searchsorted(
    #                     T, [x - r, x + r + 1e-10]
    #                 )
    #                 count_T = upper_idx_T - lower_idx_T
    #                 q_r = count_T / m

    #         ### Count points in U within the radius
    #         lower_idx_U, upper_idx_U = np.searchsorted(U, [x - r, x + r + 1e-10])
    #         count += upper_idx_U - lower_idx_U
    #         ### delete points from U that are within the radius
    #         U = np.delete(U, np.arange(lower_idx_U, upper_idx_U))

    #         ### update the indices of S
    #         if (
    #             upper_idx_S < n - 1
    #             and S[upper_idx_S + 1] - S[i + 1] < S[i + 1] - S[lower_idx_S]
    #         ):
    #             upper_idx_S += 1
    #             lower_idx_S += 1
    #         else:
    #             continue

    #     return count / m

    # def cr_multiple_proj(self, S, q, m=20000):
    #     """
    #     Estimates the copying rate of a generative model q from a training sample S using multiple (k) random projections
    #     onto one dimension. This is a variation of the algorithm by Bhattacharjee et al. (2023) for 1D data optimized for speed.

    #     Parameters
    #     ----------
    #     S : np.array
    #         The training sample
    #     q : GenerativeModel
    #         The generative model to estimate the copying rate of. Samples can be generated using q.sample()
    #     m : int
    #         The number of samples to generate from q
    #     k : int
    #         The number of random projections to use

    #     Returns
    #     -------
    #     float
    #         The estimated copying rate
    #     """
    #     n, d = S.shape
    #     T = q.sample(m, S="test")
    #     U = q.sample(m, S="val")

    #     b = 400
    #     idx_c = np.array([])  # array to store the indices of copied samples
    #     for _ in range(self.k):
    #         rp = GaussianRandomProjection(1)
    #         S_proj = np.sort(rp.fit_transform(S), axis=None)
    #         T_proj = np.sort(rp.transform(T), axis=None)
    #         U_proj = rp.transform(U)
    #         sorted_idx = np.argsort(U_proj, axis=0)
    #         U_proj = U_proj[sorted_idx].flatten()

    #         lower_idx_S, upper_idx_S = 0, b

    #         idx_c_temp = np.array([])

    #         for i, x in enumerate(S_proj):

    #             ### Find r_max
    #             r_star = max(x - S_proj[lower_idx_S], S_proj[upper_idx_S] - x)
    #             r = self.gamma * r_star * (n / b)
    #             p_r = self.gamma

    #             ## TODO: check if really all points are included
    #             lower_idx_T, upper_idx_T = np.searchsorted(
    #                 T_proj, [x - r, x + r + 1e-10]
    #             )
    #             count_T = upper_idx_T - lower_idx_T
    #             q_r = count_T / m

    #             ## radii candidates and sort them in descending order
    #             radii = np.abs(T_proj[lower_idx_T:upper_idx_T] - x)
    #             radii = np.append(-np.sort(-radii), 0)

    #             ## iterate over the radii
    #             for r_candidate in radii:
    #                 if self.lmbda * p_r <= q_r:
    #                     break
    #                 else:
    #                     r = r_candidate
    #                     p_r = b * r / (n * r_star)
    #                     lower_idx_T, upper_idx_T = np.searchsorted(
    #                         T_proj, [x - r, x + r + 1e-10]
    #                     )
    #                     count_T = upper_idx_T - lower_idx_T
    #                     q_r = count_T / m

    #             ### Count points in U within the radius
    #             lower_idx_U, upper_idx_U = np.searchsorted(
    #                 U_proj, [x - r, x + r + 1e-10]
    #             )
    #             idx_c_temp = np.unique(
    #                 np.append(idx_c_temp, sorted_idx[lower_idx_U:upper_idx_U])
    #             )

    #             ### update the indices of S
    #             if (
    #                 upper_idx_S < n - 1
    #                 and S_proj[upper_idx_S + 1] - S_proj[i + 1]
    #                 < S_proj[i + 1] - S_proj[lower_idx_S]
    #             ):
    #                 upper_idx_S += 1
    #                 lower_idx_S += 1
    #             else:
    #                 continue

    #         idx_c = np.unique(np.append(idx_c, idx_c_temp))

    #     return len(idx_c) / m


class ThreeSampleDetector:
    """
    This class implements the three-sample test for data copying detection by Meehan et al. (2020)
    Code is partly taken from https://github.com/casey-meehan/data-copying
    Paper: http://proceedings.mlr.press/v108/meehan20a/meehan20a-supp.pdf
    Copyright (c) 2020 Casey Meehan
    """

    def __init__(self, num_regions: int):
        """
        Three-sample test for data copying detection.

        Parameters
        ----------
        k : int
            The number of cells to partition the data into.
        """
        self.num_regions = num_regions

    @staticmethod
    def _Zu(Pn, Qm, T):
        """
        Source: https://github.com/casey-meehan/data-copying
        Paper: http://proceedings.mlr.press/v108/meehan20a/meehan20a-supp.pdf

        Extracts distances to training nearest neighbor
        L(P_n), L(Q_m), and runs Z-scored Mann Whitney U-test.
        For the global test, this is used on the samples within each cell.

        Inputs:
            Pn: (n X d) np array representing test sample of
                length n (with dimension d)

            Qm: (m X d) np array representing generated sample of
                length n (with dimension d)

            T: (l X d) np array representing training sample of
                length l (with dimension d)

        Ouptuts:
            Zu: Z-scored U value. A large value >>0 indicates
                underfitting by Qm. A small value <<0 indicates.
        """
        m = Qm.shape[0]
        n = Pn.shape[0]

        # fit NN model to training sample to get distances to test and generated samples
        T_NN = NearestNeighbors(n_neighbors=1).fit(T)
        LQm, _ = T_NN.kneighbors(X=Qm, n_neighbors=1)
        LPn, _ = T_NN.kneighbors(X=Pn, n_neighbors=1)

        # Get Mann-Whitney U score and manually Z-score it using the conditions of null hypothesis H_0
        u, _ = mannwhitneyu(LQm, LPn, alternative="less")
        mean = (n * m / 2) - 0.5  # 0.5 is continuity correction
        std = np.sqrt(n * m * (n + m + 1) / 12)
        Z_u = (u - mean) / std
        return Z_u

    @classmethod
    def _Zu_cells(cls, Pn, Pn_cells, Qm, Qm_cells, T, T_cells):
        """
        Source: https://github.com/casey-meehan/data-copying
        Paper: http://proceedings.mlr.press/v108/meehan20a/meehan20a-supp.pdf

        Collects the Zu statistic in each of k cells.
        There should be >0 test (Pn) and train (T) samples in each of the cells.

        Inputs:
            Pn: (n X d) np array representing test sample of length
                n (with dimension d)

            Pn_cells: (1 X n) np array of integers indicating which
                of the k cells each sample belongs to

            Qm: (m X d) np array representing generated sample of
                length n (with dimension d)

            Qm_cells: (1 X m) np array of integers indicating which of the
                k cells each sample belongs to

            T: (l X d) np array representing training sample of
                length l (with dimension d)

            T_cells: (1 X l) np array of integers indicating which of the
                k cells each sample belongs to

        Outputs:
            Zus: length k np array, where entry i indicates the Zu score for cell i
        """
        # assume cells are labeled 0 to k-1
        k = len(np.unique(Pn_cells))
        Zu_cells = np.zeros(k)

        # get samples in each cell and collect Zu
        for i in range(k):
            Pn_cell_i = Pn[Pn_cells == i]
            Qm_cell_i = Qm[Qm_cells == i]
            T_cell_i = T[T_cells == i]
            # check that the cell has test and training samples present
            if len(Pn_cell_i) * len(T_cell_i) == 0:
                raise ValueError(
                    "Cell {:n} lacks test samples and/or training samples. Consider reducing the number of cells in partition.".format(
                        i
                    )
                )

            # if there are no generated samples present, add a 0 for Zu. This cell will be excluded in \Pi_\tau
            if len(Qm_cell_i) > 0:
                Zu_cells[i] = cls._Zu(Pn_cell_i, Qm_cell_i, T_cell_i)
            else:
                Zu_cells[i] = 0
                # print("cell {:n} unrepresented by Qm".format(i))

        return Zu_cells

    @classmethod
    def _C_T(cls, Pn, Pn_cells, Qm, Qm_cells, T, T_cells, tau):
        """
        Source: https://github.com/casey-meehan/data-copying
        Paper: http://proceedings.mlr.press/v108/meehan20a/meehan20a-supp.pdf

        Runs C_T test given samples and their respective cell labels.
        The C_T statistic is a weighted average of the in-cell Zu statistics, weighted
        by the share of test samples (Pn) in each cell. Cells with an insufficient number
        of generated samples (Qm) are not included in the statistic.

        Inputs:
            Pn: (n X d) np array representing test sample of length
                n (with dimension d)

            Pn_cells: (1 X n) np array of integers indicating which
                of the k cells each sample belongs to

            Qm: (m X d) np array representing generated sample of
                length n (with dimension d)

            Qm_cells: (1 X m) np array of integers indicating which of the
                k cells each sample belongs to

            T: (l X d) np array representing training sample of
                length l (with dimension d)

            T_cells: (1 X l) np array of integers indicating which of the
                k cells each sample belongs to

            tau: (scalar between 0 and 1) fraction of Qm samples that a
                cell needs to be included in C_T statistic.

        Outputs:
            C_T: The C_T statistic for the three samples Pn, Qm, T
        """

        m = Qm.shape[0]
        n = Pn.shape[0]
        k = np.max(np.unique(T_cells)) + 1  # number of cells

        # First, determine which of the cells have sufficient generated samples (Qm(pi) > tau)
        labels, cts = np.unique(Qm_cells, return_counts=True)
        Qm_cts = np.zeros(k)
        Qm_cts[labels.astype(int)] = cts  # put in order of cell label
        Qm_of_pi = Qm_cts / m
        Pi_tau = (
            Qm_of_pi > tau
        )  # binary array selecting which cells have sufficient samples

        # Get the fraction of test samples in each cell Pn(pi)
        labels, cts = np.unique(Pn_cells, return_counts=True)
        Pn_cts = np.zeros(k)
        Pn_cts[labels.astype(int)] = cts  # put in order of cell label
        Pn_of_pi = Pn_cts / n

        # Now get the in-cell Zu scores
        Zu_scores = cls._Zu_cells(Pn, Pn_cells, Qm, Qm_cells, T, T_cells)

        # compute C_T:
        C_T = Pn_of_pi[Pi_tau].dot(Zu_scores[Pi_tau]) / np.sum(Pn_of_pi[Pi_tau])

        return C_T, np.min(Zu_scores)

    def C_T(self, Q, X_train, X_test):
        """
        Run the three-sample test for a generative model Q.

        Parameters
        ----------
        Q : object | np.array
            The generative model to test or the samples generated by the model.
        X_train : np.array
            The training data.
        X_test : np.array
            The test data.
        num_trials : int
            The number of trials to run the test.

        Returns
        -------
        float
            The test result.
        """
        m = len(X_test)
        # partition the data into k regions
        kmeans = KMeans(n_clusters=self.num_regions).fit(X_train)
        T_cells = kmeans.labels_
        Pn_cells = kmeans.predict(X_test)
        if isinstance(Q, np.ndarray):
            Qm = Q
        elif X_train.shape[1] >= 64:
            Qm = Q.sample(m, S="val").astype("float32")  ## TODO: hacky way
        else:
            Qm = Q.sample(m, S="val")
        Qm_cells = kmeans.predict(Qm)
        return self._C_T(
            X_test,
            Pn_cells=Pn_cells,
            Qm=Qm,
            Qm_cells=Qm_cells,
            T=X_train,
            T_cells=T_cells,
            tau=20 / m,
        )
