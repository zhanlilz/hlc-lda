#!/usr/bin/env python

# Harmonize multiple input classification rasters into one classification given
# number of classes in the output raster using the Latent Dirichlet Allocation
# (LDA) model. 
# 
# Zhan Li, zhanli1986@gmail.com
# Created: Wed Oct  3 13:53:01 PDT 2018

import sys
import cProfile

import argparse
import itertools
import logging, logging.config

import numpy as np
import pandas as pd
import scipy.sparse as spspa

from osgeo import gdal, gdal_array

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.externals import joblib

# from mpi4py import MPI

gdal.AllRegister()

LOGGING = {
    "version" : 1, 
    "formatters" : {
        "default" : {
            "format" : "%(asctime)s %(levelname)s %(message)s", 
        }, 
    }, 
    "handlers" : {
        "console" : {
            "class" : "logging.StreamHandler", 
            "level" : "DEBUG", 
            "formatter" : "default", 
            "stream" : "ext://sys.stdout", 
        }, 
    }, 
    "root" : {
        "handlers" : ["console"], 
        "level" : "DEBUG", 
    }, 
}
logging.config.dictConfig(LOGGING)
logger = logging.getLogger('harmonize-class-rasters')


class HarmonizeClassRasters:
    def __init__(self, 
            class_code2vocab, class_errmat, vocab_creation="union",
            **kwargs):
        # class_code2vocab: list of dict, class_code_within_each_raster:common_class_code_or_name_across_rasters
        # class_errmat: list of pandas DataFrame.
        # vocab_creation: "union" or "combination", options to create vocabulary from the input class labels of two or more rasters.
        # 
        # kwargs: keyword arguments for the LDA model,
        # LatentDirichletAllocation[http://scikit-learn.org/dev/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html#sklearn.decomposition.LatentDirichletAllocation]
        # if scikit-learn package; LDA[https://lda.readthedocs.io/en/stable/]
        # if lda package
        self._vocab_union = 1
        self._vocab_combination = 2

        self.class_code2vocab = [pd.Series(cv) for cv in class_code2vocab]
        errmat_in_vocab = []
        if class_errmat is None:
            errmat_in_vocab = [pd.DataFrame(np.eye(len(set(cv.values()))), set(cv.values()), set(cv.values())) for cv in class_code2vocab]
        else:
            for cv, old_em in zip(class_code2vocab, class_errmat):
                em = old_em.copy()
                em.index = pd.Series(cv).reindex(em.index).values
                em.columns = pd.Series(cv).reindex(em.columns).values
                errmat_in_vocab.append(em)
        self.class_errmat = errmat_in_vocab
        self.kwargs = kwargs

        # generate all the unique classes (words) as the vocabulary
        if vocab_creation=="union":
            self.vocab_creation = self._vocab_union
            self.vocab = list(set(list(itertools.chain(*[c2v.values() for c2v in class_code2vocab]))))
            self._dw = pd.Series(np.zeros(len(self.vocab)), index=pd.Index(self.vocab))
            self._m2t_prob = None
        elif vocab_creation=="combination":
            self.vocab_creation = self._vocab_combination
            self.vocab = list(itertools.product(*[set(c2v.values()) for c2v in class_code2vocab]))
            index = pd.MultiIndex.from_tuples(self.vocab)
            self._dw = pd.Series(np.zeros(len(self.vocab)), index=index)
            if class_errmat is None:
                m2t_prob = None # pd.DataFrame(np.eye(len(self.vocab)), index=index, columns=index)
            else:
                m2t_prob = pd.DataFrame(np.zeros((len(self.vocab), len(self.vocab))), index=index, columns=index)
                em_col_comb = list(itertools.product(*[em.columns.values for em in errmat_in_vocab]))
                for idx in m2t_prob.index.values:
                    m2t_list = [errmat_in_vocab[i].loc[val, :]/errmat_in_vocab[i].loc[val, :].sum() for i, val in enumerate(idx)]
                    m2t_list = list(zip(*itertools.product(*m2t_list)))
                    m2t = m2t_list[0]
                    for val in m2t_list[1:]:
                        m2t = np.multiply(m2t, val)
                    m2t_prob.loc[idx, em_col_comb] = m2t
            self._m2t_prob = m2t_prob
        else:
            raise RuntimeError("Unknown option for vocabulary creation")

        self.lda = LatentDirichletAllocation(**kwargs)


    def _translateArray(self, img, code2vocab):
        # img: 2D array
        # code2vocab: pandas series to translate class codes (indexes of the
        # series) to vocabulary codes (values of the series).
        out = img.copy()
        for idx, v in code2vocab.items():
            if idx != v:
                out[img==idx] = v
        return out

    def genDocWordFromArray(self, multiband_img, use_errmat=True, N_factor=1):
        self._dw[:] = 0
        img_list = []
        for ib in range(multiband_img.shape[2]):
            img = multiband_img[:, :, ib]
            img_list.append( self._translateArray(img, self.class_code2vocab[ib]) )

        if self.vocab_creation == self._vocab_union:
            for ib, words in enumerate(img_list):
                uw, uc = np.unique(words, return_counts=True)
                uw_mask = np.ones_like(uw, dtype=np.bool)
                for v in set(uw) - set(self.vocab):
                    uw_mask = np.logical_and(uw_mask, uw!=v)

                uw = uw[uw_mask]
                uc = uc[uw_mask]

                n_words = np.sum(uc)
                if use_errmat:
                    # Do adjustment of word counts according to error matrix
                    em = self.class_errmat[ib]
                    em_row = em.loc[uw, :].values
                    tmp = em_row / np.tile(np.sum(em_row, axis=1)[:, np.newaxis], (1, em_row.shape[1]))
                    # Calculate the proportion of vocabulary words in this image
                    # Later N_factor multiplication gives word counts that 
                    # create a document of this designated number of words. 
                    # This can be used to have all the documents of the same
                    # lengths/word counts in the LDA training. 
                    uc = np.matmul(uc, tmp)
                    uw = em.columns
                # Calculate the proportion of vocabulary words in this image
                # Later N_factor multiplication gives word counts that create a
                # document of this designated number of words.  This can be
                # used to have all the documents of the same lengths/word
                # counts in the LDA training. 
                self._dw.loc[uw] += uc / n_words
        elif self.vocab_creation == self._vocab_combination:
            uw, uc = np.unique(np.asarray(list(zip(*[img.flatten() for img in img_list]))), axis=0, return_counts=True)
            uw_mask = np.ones(uw.shape[0], dtype=np.bool)
            for v in set([tuple(val) for val in uw.tolist()]) - set(self.vocab):
                uw_mask = np.logical_and(uw_mask, np.all(uw!=np.tile(v, (uw.shape[0], 1)), axis=1))
            uw = uw[uw_mask, :]
            uc = uc[uw_mask]
            n_words = np.sum(uc)
            uw = [tuple(val) for val in uw]
            if use_errmat and (self._m2t_prob is not None):
                uc = np.matmul(uc, self._m2t_prob.loc[uw, :])
                uw = self._m2t_prob.columns.values
            # Calculate the proportion of vocabulary words in this image
            # Later N_factor multiplication gives word counts that 
            # create a document of this designated number of words. 
            # This can be used to have all the documents of the same
            # lengths/word counts in the LDA training.
            self._dw.loc[uw] = uc / n_words
        else:
            raise RuntimeError("Unknown option for vocabulary creation.")
        return self._dw.values.copy() * N_factor

    def fitTopicModel(self, X, partial=True):
        if partial:
            self.lda.partial_fit(X)
        else:
            self.lda.fit(X)

    def getTopicWordDist(self):
        return self.lda.components_

    def estDocTopicDist(self, X):
        return self.lda.transform(X)

    def estHarmonized(self, mb_img, img_mask, N_factor=1, class_nodata=0, prob_nodata=0):
        # img_mask: valid being 1 and invalid (not to be processed) being 0.
        win_ysize, win_xsize, nbands = mb_img.shape
                    
        pixel_mask = img_mask.ravel()

        pixel_word = np.array([mb_img[:, :, ib].ravel() for ib in range(nbands)]).T

        pixel_prob = np.zeros((len(pixel_mask), self.lda.n_components)) + prob_nodata
        pixel_class = np.zeros(len(pixel_mask)) + class_nodata

        if np.sum(pixel_mask) > 0:
            pixel_prob[pixel_mask, :] = self.estDocTopicDist(np.array([ self.genDocWordFromArray(pw[np.newaxis, np.newaxis, :], use_errmat=True, N_factor=N_factor) for pw in pixel_word[pixel_mask, :] ]))
            pixel_class[pixel_mask] = np.argmax(pixel_prob[pixel_mask, :], axis=1)+1

        class_img = pixel_class.reshape(win_ysize, win_xsize)
        prob_img = np.dstack([pixel_prob[:, ib].reshape(win_ysize, win_xsize) for ib in range(nbands)])

        return class_img, prob_img


def getCmdArgs():
    p = argparse.ArgumentParser(description="Harmonize input classification maps using Latent Dirichlet Allocation")

    p.add_argument("input_rasters", metavar="CLASSIFICATION_RASTER", nargs="+", help="List of classification rasters to be harmonized.")

    p.add_argument("--bands", dest="bands", metavar="BAND_INDEX", required=False, type=int, nargs="+", help="Band index to the classsification in each input raster. Default: the first band of each raster, i.e. all 1.")
    p.add_argument("--vocab_creation", dest="vocab_creation", metavar="METHOD_TO_CREATE_VOCABULARY", required=False, choices=["union", "combination"], default="union", help="How to create a vocabulary from the lists of class labels of input rasters. 'union': the union of input lists, e.g. [20, 30] and [20, 81, 200] creates [20, 30, 81, 200], ; 'combination': the combination of input lists, e.g. [20, 30] and [20, 81, 200] creates [(20, 20), (20, 81), (20, 200), (30, 20), (30, 81), (30, 200)].")
    p.add_argument("--class2vocab", dest="class2vocab", metavar="CLASS_CODES_TO_VOCAB_CSV", required=True, nargs="+", help="List of CSV files. Each row of a CSV file gives a class code of the input raster in the first column and a common class name or code across all the input rasters in the second column. NO header.")
    p.add_argument("--error_matrix", dest="error_matrix", metavar="ERROR_MATRIX_CSV", required=False, nargs="+", default=None, help="List of CSV files, each of which gives the error matrix of the input classification raster. Error matrices are in terms of area proportions. Rows are class label codes and columns are reference label codes. If not given, assuming the accuracies of all the classes in each raster are 100%%.")
    p.add_argument("--cv", dest="cv_split", metavar="FOLDS_FOR_CROSS_VALIDATION", required=False, type=int, default=None, help="Number of folds to select tiles for cross validation. Default: None, no cross validation.")
    p.add_argument("--test", dest="test", metavar="CROSS_VALIDATION_TEST_MASK", required=False, default=None, help="A raster mask to select test pixels for cross validation, with test pixels being 1 in the mask raster.")

    p.add_argument("--n_topics", dest="n_topics", required=True, type=int, help="Number of topics or classes in the output harmonized classification.")
    p.add_argument("--doc_tile_size", dest="doc_tile_size", required=False, type=int, default=1000, help="Size of a tile in the raster to take as a document in the LDA model. Default: 1000")
    p.add_argument("--N_factor", dest="N_factor", required=False, type=int, default=None, help="Sample size for bootstrapping class labels per pixel according to error matrix. Default: None, sample size will be determined according to the decimal precisions of error matrix entries.")
    p.add_argument("--n_jobs", dest="n_jobs", required=False, type=int, default=1, help="The number of jobs to run in parallel by the LDA model fitting. Default: 1.")
    p.add_argument("--n_batch_docs", dest="n_batch_docs", required=False, type=int, default=1000, help="Number of documents being stored and fed to LDA model fitting one time in a batch to avoid running out of memory. Default: 1000.")
    p.add_argument("--doc_topic_prior", dest="doc_topic_prior", required=False, type=float, default=None, help="Prior of document topic distribution, i.e. the Dirichlet distribution parameter alpha in Blei et al., 2003. Default: 1/n_topics")
    p.add_argument("--topic_word_prior", dest="topic_word_prior", required=False, type=float, default=None, help="Prior of topic word distribution, i.e., the Dirichlet distribution parameter eta in Blei et al., 2003. Default: 1/n_vocab_words")

    p.add_argument("--out_model", dest="out_model", required=True, default=None, help="Output binary file in joblib format that stores the trained LDA model.")
    p.add_argument("--out_topic_word", dest="out_topic_word", required=False, default=None, help="Output CSV file of the estimated topic-word distribution by the LDA.")
    p.add_argument("--out_lut", dest="out_lut", required=False, default=None, help="Output CSV file of the look-up table of the LDA model.")
    p.add_argument("--out_class", dest="out_class", required=False, default=None, help="Output raster for the harmonized classification.")
    p.add_argument("--out_prob", dest="out_prob", required=False, default=None, help="Output raster for the class probabilities after the harmonization.")
    p.add_argument("--out_format", dest="out_format", required=False, default="ENVI", help="Output raster format. Default: ENVI.")

    cmdargs = p.parse_args()

    if len(cmdargs.input_rasters) != len(cmdargs.bands):
        raise RuntimeError("Number of given band indices must be the same with the number of given input classification rasters.")
    if len(cmdargs.input_rasters) != len(cmdargs.class2vocab):
        raise RuntimeError("Number of given CSV files of class codes and common class codes/names must be the same with the number of given input classification rasters.")
    if ( cmdargs.error_matrix is not None ) and ( len(cmdargs.input_rasters) != len(cmdargs.error_matrix) ):
        raise RuntimeError("Number of given CSV files of error matrix must be the same with the number of given input classification rasters.")
    if ( cmdargs.cv_split is not None) and ( cmdargs.cv_split < 2 ):
        raise RuntimeError("To do cross valiation, the folds must be at least 2!")
    return cmdargs

def main(cmdargs):
    rank = 0 # residual from MPI version, here is single-process version, rank of the process is simply 0.
    hcr_rasters = cmdargs.input_rasters
    # check the input rasters and calcuate the number of tiles (docs)
    hcr_datasets = [ gdal.Open(rfile, gdal.GA_ReadOnly) for rfile in hcr_rasters ]
    hcr_raster_xsize, hcr_raster_ysize = hcr_datasets[0].RasterXSize, hcr_datasets[0].RasterYSize

    if not np.all([ds.RasterXSize==hcr_raster_xsize for ds in hcr_datasets]):
        raise RuntimeError("Input rasters have different X dimensions!")
    if not np.all([ds.RasterYSize==hcr_raster_ysize for ds in hcr_datasets]):
        raise RuntimeError("Input rasters have different Y dimensions!")

    hcr_bands = [ ds.GetRasterBand(i) for ds, i in zip(hcr_datasets, cmdargs.bands) ]

    cv_split = cmdargs.cv_split
    hcr_test_ds = None
    hcr_test_bd = None
    if cmdargs.test is not None:
        hcr_test_ds = gdal.Open(cmdargs.test, gdal.GA_ReadOnly)
        hcr_test_bd = hcr_test_ds.GetRasterBand(1)

    tile_xsize = cmdargs.doc_tile_size
    tile_ysize = cmdargs.doc_tile_size

    ndocs_batch = cmdargs.n_batch_docs

    # Set up dict of class codes within raster to common class codes/names across raster
    class_code2vocab = []
    for csvfname in cmdargs.class2vocab:
        arr = np.loadtxt(csvfname, dtype=int, delimiter=',')
        class_code2vocab.append( { row[0]:row[1] for row in arr } )
    class_errmat = None
    use_errmat = False
    if cmdargs.error_matrix is not None:
        class_errmat = []
        for csvfname in cmdargs.error_matrix:
            tmp = pd.read_csv(csvfname, index_col=0, header=0)
            tmp.columns = [int(val) for val in tmp.columns]
            class_errmat.append(tmp)
        use_errmat = True

    if cmdargs.N_factor is None:
        if class_errmat is not None:
            tmpval = np.min([np.min(em.values[em.values!=0]) for em in class_errmat])
            tmpval = 10**(len(str(int(1./tmpval))) + 1)
            N_factor = 1000 if 1000>tmpval else tmpval
        else:
            N_factor = 1000
    else:
        N_factor = cmdargs.N_factor

    if cmdargs.doc_topic_prior is None:
        doc_topic_prior = 1./cmdargs.n_topics
    else:
        doc_topic_prior = cmdargs.doc_topic_prior
    vocab = set(list(itertools.chain(*[c2v.values() for c2v in class_code2vocab])))
    if cmdargs.topic_word_prior is None:
        topic_word_prior = 1./len(vocab)
    else:
        topic_word_prior = cmdargs.topic_word_prior

    vocab_creation = cmdargs.vocab_creation

    hcr = HarmonizeClassRasters(
            class_code2vocab, class_errmat, vocab_creation=vocab_creation, 
            n_components=cmdargs.n_topics, max_iter=1000, evaluate_every=1, perp_tol=1e-1, 
            n_jobs=cmdargs.n_jobs, batch_size=ndocs_batch,  
            doc_topic_prior=doc_topic_prior, topic_word_prior=topic_word_prior)

    # Set up a look-up table (LUT) to store the occurrence counts, LDA scores,
    # topic prob. estimates of all the combinations of input class labels from
    # different rasters.
    # 
    # MultiIndex: class legends of input rasters
    # Columns: occurrence counts, LDA scores, topic probs., primary topic ID
    # (the topic with the largest prob.)
    index_list = [set(c2v.values()) for c2v in class_code2vocab]
    hcr_lut_index = pd.MultiIndex.from_product(index_list)
    nrows = len(hcr_lut_index)
    ncols = 1 + 1 + 1 + 1 + cmdargs.n_topics
    prob_topic_colnames = ["prob_topic_{0:d}".format(i+1) for i in range(cmdargs.n_topics)]
    hcr_lut = pd.DataFrame(np.zeros((nrows, ncols)), index=hcr_lut_index, 
            columns=["total_npix", "test_npix", "lda_score", "primary_topic_id"] + prob_topic_colnames)
    hcr_lut_values = hcr_lut.values.copy()

    ntiles_x = np.ceil(hcr_raster_xsize / tile_xsize).astype(np.int)
    ntiles_y = np.ceil(hcr_raster_ysize / tile_ysize).astype(np.int)
    dw_mat = np.zeros((ndocs_batch, len(hcr.vocab)))
    word_count = np.zeros(len(hcr.vocab))
    doc_idx = 0
    tmp = np.argmax([np.dtype(gdal_array.GDALTypeCodeToNumericTypeCode(bd.DataType)).itemsize for bd in hcr_bands])
    img_dtype = gdal_array.GDALTypeCodeToNumericTypeCode(hcr_bands[tmp].DataType)
    nodata = np.iinfo(img_dtype).max

#    prf = cProfile.Profile()
#    prf.enable()
    ndigits = max(len(str(ntiles_x)), len(str(ntiles_y)))
    progress_tot = ntiles_x * ntiles_y
    progress_pct = 10
    progress_frc = int(progress_pct/100.*progress_tot)
    if progress_frc == 0:
        progress_frc = 1
        progress_pct = int(progress_frc/float(progress_tot)*100)
    
    progress_cnt = 0
    progress_npct = 0
    if cv_split is not None:
        logger.info("Process {0:d}: Search non-empty tiles for cross-validation sampling ...".format(rank))
        valid_tiles = []
        for iby in range(ntiles_y):
            for ibx in range(ntiles_x): 
                xoff, yoff = tile_xsize*ibx, tile_ysize*iby
                win_xsize = tile_xsize if ibx<ntiles_x-1 else hcr_raster_xsize-xoff
                win_ysize = tile_ysize if iby<ntiles_y-1 else hcr_raster_ysize-yoff

                mb_img = [bd.ReadAsArray(xoff, yoff, win_xsize, win_ysize).astype(img_dtype) for bd in hcr_bands]
                valid_flag = False
                for ib, img in enumerate(mb_img):
                    tmp = set(hcr.class_code2vocab[ib].index.values)
                    tmp_len = len(tmp)
                    if len(tmp - set(np.unique(img))) < tmp_len:
                        valid_flag = True
                        break
                if valid_flag:
                    valid_tiles.append(np.ravel_multi_index((iby, ibx), (ntiles_y, ntiles_x)))
                progress_cnt += 1
                if progress_cnt % progress_frc == 0:
                    progress_npct += progress_pct
                    if progress_npct <= 100:
                        logger.info("Process {1:d}: Finish searching non-empty tiles {0:d}%".format(progress_npct, rank))
        logger.info("Process {1:d}: Finish searching non-empty tiles, {0:d} non-empty tiles found".format(len(valid_tiles), rank))

        # Random sampling for each CV test
        cv_test_tiles = []
        cv_hcr_list = []
        cv_dw_mat_list = [np.zeros((ndocs_batch, len(hcr.vocab)))]*cv_split
        cv_doc_idx_list = [0]*cv_split
        cv_score_sum_list = [0]*cv_split
        cv_perplexity_list = [0]*cv_split
        cv_size = int(len(valid_tiles) / cv_split)
        for i in range(cv_split):
            tmp = np.random.choice(valid_tiles, size=cv_size, replace=False)
            tmprow, tmpcol = np.unravel_index(tmp, (ntiles_y, ntiles_x))
            tmp = spspa.coo_matrix((np.ones_like(tmprow, dtype=np.bool), (tmprow, tmpcol)), shape=(ntiles_y, ntiles_x), dtype=np.bool)
            cv_test_tiles.append(tmp.tocsr())
            cv_hcr_list.append(HarmonizeClassRasters(
                class_code2vocab, class_errmat, vocab_creation=vocab_creation, 
                n_components=cmdargs.n_topics, max_iter=1000, evaluate_every=1, perp_tol=1e-1, 
                n_jobs=cmdargs.n_jobs, batch_size=ndocs_batch,  
                doc_topic_prior=doc_topic_prior, topic_word_prior=topic_word_prior))

    progress_cnt = 0
    progress_npct = 0
    logger.info("Process {0:d}: Start building document-word matrix from input classification rasters ...".format(rank))
    for iby in range(ntiles_y):
        for ibx in range(ntiles_x):                   
            xoff, yoff = tile_xsize*ibx, tile_ysize*iby
            win_xsize = tile_xsize if ibx<ntiles_x-1 else hcr_raster_xsize-xoff
            win_ysize = tile_ysize if iby<ntiles_y-1 else hcr_raster_ysize-yoff

            mb_img = [bd.ReadAsArray(xoff, yoff, win_xsize, win_ysize).astype(img_dtype) for bd in hcr_bands]
            # mask out invalid pixels
            img_mask = np.zeros_like(mb_img[0], dtype=np.bool)
            for ib, img in enumerate(mb_img):
                tmp_mask = np.ones_like(img, dtype=np.bool)
                for v in hcr.class_code2vocab[ib].index:
                    tmp_mask = np.logical_and(tmp_mask, img!=v)
                img_mask = np.logical_or(img_mask, tmp_mask)

            tmp_mask = np.logical_not(img_mask)
            to_do_flag = tmp_mask.sum() > 0
            if to_do_flag:
                uq_idx, uq_cnt = np.unique(np.array([hcr._translateArray(img[tmp_mask], hcr.class_code2vocab[i]) for i, img in enumerate(mb_img)]), axis=1, return_counts=True)
                hcr_lut.loc[[*zip(*uq_idx.tolist())], "total_npix"] += uq_cnt

            if to_do_flag and hcr_test_bd is not None:
                test_mask = hcr_test_bd.ReadAsArray(xoff, yoff, win_xsize, win_ysize)
                # img_mask = np.logical_or(img_mask, test_mask==1)
                tmp_mask = np.logical_not(np.logical_or(img_mask, test_mask==1))
                # to_do_flag = tmp_mask.sum() > 0
                if tmp_mask.sum() > 0: # to_do_flag:
                    uq_idx, uq_cnt = np.unique(np.array([hcr._translateArray(img[tmp_mask], hcr.class_code2vocab[i]) for i, img in enumerate(mb_img)]), axis=1, return_counts=True)
                    hcr_lut.loc[[*zip(*uq_idx.tolist())], "test_npix"] += uq_cnt
                    
            if to_do_flag:
                for img in mb_img:
                    img[img_mask] = nodata
                dw_mat[doc_idx, :] = hcr.genDocWordFromArray(np.dstack(mb_img), use_errmat=use_errmat, N_factor=N_factor)
                if cv_split is not None:
                    for i in range(cv_split):
                        if not cv_test_tiles[i][iby, ibx]:
                            cv_dw_mat_list[i][cv_doc_idx_list[i], :] = dw_mat[doc_idx, :]
                            cv_doc_idx_list[i] += 1
                doc_idx += 1

            if doc_idx == ndocs_batch:
                word_count += np.sum(dw_mat[0:doc_idx, :], axis=0)
                hcr.fitTopicModel(dw_mat[0:doc_idx, :])
                doc_idx = 0
            if cv_split is not None:
                for i in range(cv_split):
                    if cv_doc_idx_list[i] == ndocs_batch:
                        cv_hcr_list[i].fitTopicModel(cv_dw_mat_list[i][0:cv_doc_idx_list[i], :])
                        cv_doc_idx_list[i] = 0

            progress_cnt += 1
            if progress_cnt % progress_frc == 0:
                progress_npct += progress_pct
                if progress_npct <= 100:
                    logger.info("Process {1:d}: Finish reading input rasters and building document-word matrix {0:d}%".format(progress_npct, rank))

#        if doc_idx > ndocs_batch-2:
#            break

    if doc_idx > 0:
        word_count += np.sum(dw_mat[0:doc_idx, :], axis=0)
        hcr.fitTopicModel(dw_mat[0:doc_idx, :])
        doc_idx = 0
    tmp = np.where(word_count == 0)[0]
    if len(tmp) > 0:
        logger.warning("Process {0:d}: Some classes never appeared in the input rasters: \n".format(rank) + str(np.array(list(hcr.vocab))[tmp]))
    if cv_split is not None:
        for i in range(cv_split):
            if cv_doc_idx_list[i] > 0:
                cv_hcr_list[i].fitTopicModel(cv_dw_mat_list[i][0:cv_doc_idx_list[i], :])
                cv_doc_idx_list[i] = 0

#    prf.disable()
#    prf.print_stats(sort="time")

    # The following only on master node, no parallelization.
    logger.info("Process {0:d}: Finish fitting LDA model ...".format(rank))

    # Estimate the topic probs. for all the combinations of the input class
    # labels and their LDA scores.
    logger.info("Process {0:d}: Start building LUT of the topics and LDA score per combination of input class legends ...".format(rank))
#         prf = cProfile.Profile()
#         prf.enable()
    dw_mat = np.array([ hcr.genDocWordFromArray(np.array(idx)[np.newaxis, np.newaxis, :], use_errmat=use_errmat, N_factor=N_factor) for idx in hcr_lut.index ])
    dt_dist = hcr.estDocTopicDist(dw_mat)
    hcr_lut.loc[:, prob_topic_colnames] = dt_dist
    hcr_lut.loc[:, "primary_topic_id"] = np.argmax(dt_dist, axis=1)+1
    for i, idx in enumerate(hcr_lut.index):
        hcr_lut.loc[idx, "lda_score"] = hcr.lda.score(dw_mat[[i], :])
#         prf.disable()
#         prf.print_stats(sort="time")
    logger.info("Process {0:d}: Finish building LUT of the topics and LDA score per combination of input class legends ...".format(rank))

    if hcr_test_bd is not None:
        # Calculate the perplexity and score over the test pixels
        logger.info("Process {0:d}: Start estimating perplexity and score over test pixels ...".format(rank))
#         prf = cProfile.Profile()
#         prf.enable()
        score_sum = np.sum(hcr_lut["lda_score"] * hcr_lut["test_npix"])
        perplexity = np.exp(-1 * np.sum(hcr_lut["lda_score"] * hcr_lut["test_npix"]) / (N_factor*np.sum(hcr_lut["test_npix"])))
    else:
        score_sum = np.nan
        perplexity = np.nan

    if cv_split is not None:
        logger.info("Process {0:d}: Start estimating perplexity and score over test tiles in the cross validation ...".format(rank))
        progress_npct=0
        progress_cnt = 0
        for iby in range(ntiles_y):
            for ibx in range(ntiles_x):
                if np.any([cv_test_tiles[i][iby, ibx] for i in range(cv_split)]):
                    xoff, yoff = tile_xsize*ibx, tile_ysize*iby
                    win_xsize = tile_xsize if ibx<ntiles_x-1 else hcr_raster_xsize-xoff
                    win_ysize = tile_ysize if iby<ntiles_y-1 else hcr_raster_ysize-yoff

                    mb_img = [bd.ReadAsArray(xoff, yoff, win_xsize, win_ysize).astype(img_dtype) for bd in hcr_bands]
                    # mask out invalid pixels
                    img_mask = np.zeros_like(mb_img[0], dtype=np.bool)
                    for ib, img in enumerate(mb_img):
                        tmp_mask = np.ones_like(img, dtype=np.bool)
                        for v in hcr.class_code2vocab[ib].index:
                            tmp_mask = np.logical_and(tmp_mask, img!=v)
                        img_mask = np.logical_or(img_mask, tmp_mask)

                    tmp_mask = np.logical_not(img_mask)
                    to_do_flag = tmp_mask.sum() > 0
                    if to_do_flag:
                        for img in mb_img:
                            img[img_mask] = nodata
                        dw_mat = hcr.genDocWordFromArray(np.dstack(mb_img), use_errmat=use_errmat, N_factor=N_factor)
                        for i in range(cv_split):
                            if cv_test_tiles[i][iby, ibx]:
                                cv_score_sum_list[i] += cv_hcr_list[i].lda.score(dw_mat[np.newaxis, :])
                                cv_perplexity_list[i] += np.log(cv_hcr_list[i].lda.perplexity(dw_mat[np.newaxis, :]))
                progress_cnt += 1
                if progress_cnt % progress_frc == 0:
                    progress_npct += progress_pct
                    if progress_npct <= 100:
                        logger.info("Process {1:d}: Finish inference of test tiles in cross validation {0:d}%".format(progress_npct, rank))
        logger.info("Process {4:d}: doc_size = {0:d}, n_topics = {1:d}, n_factor = {2:d}, cv_size = {3:d}, cross validation report: ".format(cmdargs.doc_tile_size, cmdargs.n_topics, N_factor, cv_size, rank))
        logger.info("Process {0:d}: cv_seq, perplexity_cv, score_cv".format(rank))
        for i in range(cv_split):
            logger.info("Process {3:d}: {0:d}, {1:e}, {2:e}".format(i, np.exp(cv_perplexity_list[i]/cv_size), cv_score_sum_list[i]/cv_size, rank))
        logger.info("Process {3:d}: {0:s}, {1:e}, {2:e}".format("mean", np.exp(np.mean(cv_perplexity_list)/cv_size), np.mean(cv_score_sum_list)/cv_size, rank))

#         prf.disable()
#         prf.print_stats(sort="time")

    logger.info("Process {5:d}: doc_size = {2:d}, n_topics = {3:d}, n_factor = {4:d}, perplexity_test_pixels = {0:e}, score_test_pixels = {1:e}".format(perplexity, score_sum, cmdargs.doc_tile_size, cmdargs.n_topics, N_factor, rank))

    joblib.dump(hcr.lda, cmdargs.out_model)
    logger.info("Process {1:d}: trained LDA model saved to {0:s}".format(cmdargs.out_model, rank))
    vocab_joblib = ".".join(cmdargs.out_model.split('.')[0:-1])+"_vocab.joblib"
    joblib.dump(hcr.vocab, vocab_joblib)
    logger.info("Process {1:d}: vocabulary list of trained LDA model saved to {0:s}".format(vocab_joblib, rank))

    if cmdargs.out_topic_word is not None:
        pd.DataFrame(hcr.getTopicWordDist(), columns=hcr._dw.index).to_csv(cmdargs.out_topic_word)
        logger.info("Process {1:d}: Topic-word distribution written to {0:s}".format(cmdargs.out_topic_word, rank))
    if cmdargs.out_lut is not None:
        hcr_lut.to_csv(cmdargs.out_lut)
        logger.info("Process {1:d}: Look-up table of LDA model written to {0:s}".format(cmdargs.out_lut, rank))

    if cmdargs.out_class is not None:
        class_raster = cmdargs.out_class
        class_format = cmdargs.out_format
        prob_raster = cmdargs.out_prob
        prob_format = cmdargs.out_format

        class_nodata = 0
        prob_nodata = 0
        
        if hcr.lda.n_components < np.iinfo(np.int8).max:
            out_type = np.int8
        elif hcr.lda.n_components < np.iinfo(np.int16).max:
            out_type = np.int16
        elif hcr.lda.n_components < np.iinfo(np.int32).max:
            out_type = np.int32
        elif hcr.lda.n_components < np.iinfo(np.int64).max:
            out_type = np.int64
        else:
            out_type = np.float
        prob_type = np.float32

        block_xsize, block_ysize = hcr_bands[0].GetBlockSize()
        hcr_type_size = np.max([np.dtype(gdal_array.GDALTypeCodeToNumericTypeCode(bd.DataType)).itemsize for bd in hcr_bands])
        tmpn = int( gdal.GetCacheMax() / (block_xsize*block_ysize*hcr_type_size) )
        if tmpn > int(hcr_raster_ysize / block_ysize):
            tmpn = int(hcr_raster_ysize / block_ysize)
        if tmpn > 1:
            block_ysize = tmpn * block_ysize

        nblocks_x, nblocks_y = np.ceil(hcr_raster_xsize/block_xsize).astype(int), np.ceil(hcr_raster_ysize/block_ysize).astype(int)

        block_meta_data = np.zeros(4, dtype=np.int)
        logger.info("Process {0:d}: Start estimating and writing pixel topics (harmonized class) ...".format(rank))
        logger.info("Process {4:d}: n_blocks_x = {0:d}, n_blocks_y = {1:d}, block_xsize = {2:d}, block_ysize = {3:d}".format(nblocks_x, nblocks_y, block_xsize, block_ysize, rank))

        # Use N-dimensional array to speed up LUT search
        hcr_lut_class_arr = np.zeros([len(ilevel) for ilevel in hcr_lut.index.levels], dtype=out_type)
        hcr_lut_prob_arr = np.zeros([len(ilevel) for ilevel in hcr_lut.index.levels]+[len(prob_topic_colnames)], dtype=prob_type)
        # Put the LUT into the array
        hcr_lut_class_arr[tuple(hcr_lut.index.labels)] = hcr_lut.loc[:, "primary_topic_id"]
        for i, coln in enumerate(prob_topic_colnames):
            hcr_lut_prob_arr[tuple(hcr_lut.index.labels+[[i]*len(hcr_lut.index)])] = hcr_lut.loc[:, coln]

        class_driver = gdal.GetDriverByName(class_format)

        class_ds = class_driver.Create(class_raster, hcr_raster_xsize, hcr_raster_ysize, 1, 
                gdal_array.NumericTypeCodeToGDALTypeCode(out_type))
        class_bd = class_ds.GetRasterBand(1)

        if prob_format is not None:
            prob_driver = gdal.GetDriverByName(prob_format)
        else:
            prob_driver = class_driver

        if prob_raster is not None:
            prob_ds = prob_driver.Create(prob_raster, hcr_raster_xsize, hcr_raster_ysize, hcr.lda.n_components, 
                    gdal_array.NumericTypeCodeToGDALTypeCode(prob_type))
        else:
            prob_ds = None

        progress_cnt = 0
        progress_tot = nblocks_x * nblocks_y
        progress_pct = 10
        progress_frc = int(progress_pct/100.*progress_tot)
        if progress_frc == 0:
            progress_frc = 1
        progress_npct = 0
        for iby in range(nblocks_y):
            for ibx in range(nblocks_x):
                xoff, yoff = ibx*block_xsize, iby*block_ysize
                # win_xsize, win_ysize = hcr_bands[0].GetActualBlockSize(ibx, iby)
                win_xsize = block_xsize if ibx<nblocks_x-1 else hcr_raster_xsize-xoff
                win_ysize = block_ysize if iby<nblocks_y-1 else hcr_raster_ysize-yoff
               
                block_meta_data[:] = [xoff, yoff, win_xsize, win_ysize]
                # On slave processes, 
                mb_img = [bd.ReadAsArray(xoff, yoff, win_xsize, win_ysize) for bd in hcr_bands]
                # mask out invalid pixels
                img_mask = np.zeros_like(mb_img[0], dtype=np.bool)
                for ib, img in enumerate(mb_img):
                    tmp_mask = np.ones_like(img, dtype=np.bool)
                    for v in hcr.class_code2vocab[ib].index:
                        tmp_mask = np.logical_and(tmp_mask, img!=v)
                    img_mask = np.logical_or(img_mask, tmp_mask)
                img_mask = np.logical_not(img_mask)

                to_do_flag = np.sum(img_mask) > 0
                class_img = np.zeros_like(img_mask, dtype=out_type) + class_nodata
                if to_do_flag:
                    # Convert image of vocabulary class labels to image of indexes to these classes in each input raster
                    idx_img_list = []
                    for i, img in enumerate(mb_img):
                        tmp_idx = hcr._translateArray(img[img_mask], hcr.class_code2vocab[i])
                        for idx, ilevel in enumerate(hcr_lut.index.levels[i]):
                            tmp_idx[tmp_idx==ilevel] = idx
                        idx_img_list.append(tmp_idx)
                    class_img[img_mask] = hcr_lut_class_arr[tuple(idx_img_list)]

                if prob_raster is not None:
                    prob_img_list = [np.zeros_like(img_mask, dtype=prob_type) + prob_nodata for coln in prob_topic_colnames]
                    if to_do_flag:
                        for i, coln in enumerate(prob_topic_colnames):
                            prob_img_list[i][img_mask] = hcr_lut_prob_arr[tuple(idx_img_list+[[i]*len(idx_img_list[0])])]
                    prob_img = np.dstack(prob_img_list)

                class_bd.WriteArray(class_img, int(block_meta_data[0]), int(block_meta_data[1]))
                if prob_raster is not None:
                    for i in range(prob_img.shape[2]):
                        prob_ds.GetRasterBand(i+1).WriteArray(prob_img[:,:,i], int(block_meta_data[0]), int(block_meta_data[1]))

                progress_cnt += 1
                if progress_cnt % progress_frc == 0:
                    if progress_frc == 1:
                        progress_npct = int(100*progress_cnt/progress_tot)
                    else:
                        progress_npct += progress_pct
                    if progress_npct <= 100:
                        logger.info("Process {1:d}: Finish pixel inference and writing {0:d}%".format(progress_npct, rank))

        class_ds.FlushCache()
        prob_ds.FlushCache()

        class_ds.SetGeoTransform(hcr_datasets[0].GetGeoTransform())
        class_ds.SetProjection(hcr_datasets[0].GetProjectionRef())
        prob_ds.SetGeoTransform(hcr_datasets[0].GetGeoTransform())
        prob_ds.SetProjection(hcr_datasets[0].GetProjectionRef())

        class_ds = None
        prob_ds = None

    # On both master and slave processes, close the raster files that GDAL has
    # opened.
    for i in range(len(hcr_datasets)):
        hcr_datasets[i] = None
    if cmdargs.test is not None:
        hcr_test_ds = None

if __name__ == "__main__":
    cmdargs = getCmdArgs()
    main(cmdargs)
