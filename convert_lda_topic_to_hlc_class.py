#/usr/bin/env python

# Convert the probabilities of latent topics from a trained LDA model to
# probabilities of targeted harmonized land cover classes. 
# 
# Zhan Li, zhan.li@canada.ca
# Created: Fri May  3 22:48:40 PDT 2019

import argparse
import logging, logging.config

import numpy as np
import pandas as pd

from sklearn.externals import joblib

import scipy.stats as spsta
import scipy.optimize as spopt
import scipy.sparse as spspa

from numba import jit, float64

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
logger = logging.getLogger('lda-topic2hlc-class')

def getCmdArgs():
    p = argparse.ArgumentParser(description="Estimate probabilities of harmonized land cover class from the probabilities of latent topics from a trained LDA model")

    p.add_argument("--lda_model", "-m", dest="lda_model_file", metavar="LDA_MODEL_JOBLIB", required=True, help="A trained LDA model that has been pickled by joblib.")
    p.add_argument("--lda_vocab", "-w", dest="lda_vocab_file", metavar="LDA_VOCAB_WORDS_JOBLIB", required=True, help="List of LDA model's vocabulary words that has been pickled by joblib.")
    p.add_argument("--lda_topics", "-t", dest="lda_topics_csv", metavar="LDA_TOPIC_PROB_CSV", required=False, help="An optional CSV file that lists topic probabilities of documents inferred by the given LDA model, with rows being documents and columns being topics, and the first row being header. The first few columns can be extra values to index documents but all the remaining columns must be topic probabilities in the same numbers of topics given by the LDA model file. If provided, the topic probabilities will be converted to HLC class probabilities and saved in a CSV file. Otherwise, only topic-to-HLC conversion matrix will be estimated and saved.")
    p.add_argument("--agreed_word", "-A", dest="agreed_words_csv", metavar="LUT_WORD_COMB2HLC", required=True, help="A look-up table of agreed vocabulary words (source classes) to targeted HLC classes, with each row being agreed words followed by its corresponding HLC class in the last column, and first row being the header.")
    p.add_argument("--t2h", dest="t2h_csv", metavar="TOPIC2HLC_CSV", required=True, help="Output CSV file that stores the estimated topic-to-HLC matrix from the LUT of agreed words and HLC classes.")
    p.add_argument("--hlc_prob", dest="hlc_prob_csv", metavar="HLC_PROB_CSV", required=False, help="Output CSV file that stores the estimated HLC class probabilities from topic probabilities if a CSV file of topic probabilities is given.")
    p.add_argument("--N_factor", "-N", dest="N_factor", metavar="WORD_COUNT_PER_DOC", required=False, type=int, default=1000, help="Word count per pseudo document. Default: 1000.")
    p.add_argument("--n_resample", "-n", dest="n_resample", metavar="RESAMPLE_COUNT_PER_HLC", required=False, type=int, default=100, help="Resample count per HLC class. Default: 100.")
    p.add_argument("--replicate", "-r", dest="n_replicates", metavar="TIMES_TO_REPEAT_ESTIMATION", required=False, type=int, default=1, help="Number of times to repeat the estimation. The estimation of topic-to-HLC matrix and HLC class probabilities inherits randomness from the pseudo-document generation and initial values for optimization. running multiple times to have a stable concensus. Default: 1.")

    p.add_argument("--seed", dest="random_seed", metavar="SEED_FOR_RANDOM_GENERATOR", required=False, type=int, default=None, help="A seed for random generator to generate the same pseudo random numbers for processing. Default: None, do not seed the generator.")

    cmdargs = p.parse_args()
    if (cmdargs.lda_topics_csv is not None and cmdargs.hlc_prob_csv is None):
        raise RuntimeError("CSV of topic probabilities is given but no output CSV for estimated HLC class probabilities!")
    if (cmdargs.lda_topics_csv is None and cmdargs.hlc_prob_csv is not None):
        raise RuntimeError("CSV of topic probabilities is missing but output CSV for estimated HLC class probabilities is given!")

    return cmdargs

@jit(float64(float64[:], float64[:,:], float64[:], float64[:]), nopython=True, parallel=True)
def objectiveFunc(x, A, b, wt):
    return np.sum((A.dot(x) - b)**2 * wt)

@jit(float64[:](float64[:], float64[:, :]), nopython=True, parallel=True)
def eqConsFunc(x, B):
    return B.dot(x)-np.ones(B.shape[0])

@jit(float64[:, :](float64[:], float64[:, :]), nopython=True, parallel=True)
def eqConsJac(x, B):
    return B

@jit(float64[:](float64[:]), nopython=True, parallel=True)
def ineqCons1Func(x):
    return x

@jit(float64[:, :](float64[:]), nopython=True, parallel=True)
def ineqCons1Jac(x):
    return np.identity(x.size)

@jit(float64[:](float64[:]), nopython=True, parallel=True)
def ineqCons2Func(x):
    return 1 - x

@jit(float64[:, :](float64[:]), nopython=True, parallel=True)
def ineqCons2Jac(x):
    return -1*np.identity(x.size)

def main(cmdargs):
    model_file = cmdargs.lda_model_file
    vocab_file = cmdargs.lda_vocab_file
    lda_lut_csv = cmdargs.lda_topics_csv
    agreed_words_csv = cmdargs.agreed_words_csv

    t2h_csv = cmdargs.t2h_csv
    hlc_prob_csv = cmdargs.hlc_prob_csv

    N_factor = cmdargs.N_factor
    n_per_hlc = cmdargs.n_resample
    n_replicates = cmdargs.n_replicates

    random_seed = cmdargs.random_seed

    if random_seed is not None:
        np.random.seed(random_seed)

    # Read agreed word combinations and their HLC class
    hlc_conc_lut = pd.read_csv(agreed_words_csv, header=0)
    # load model and vocab
    lda_model = joblib.load(model_file)
    lda_vocab = joblib.load(vocab_file)

    # an array of HLC classes
    hlc_code_list = np.unique(hlc_conc_lut.iloc[:, -1])
    if np.issubsctype(hlc_code_list.dtype, np.number):
        hc_dtype = np.int
    else:
        hc_dtype = hlc_code_dtype.dtype
    hlc_code_list = hlc_code_list.astype(hc_dtype)
    # number of sources of words and an array of sources
    n_maps = hlc_conc_lut.shape[1]-1
    map_names = list(hlc_conc_lut.columns[0:-1])
    hlc_conc_lut = hlc_conc_lut.set_index(map_names)

    # Check if all the given agreed words are in LDA's vocabulary
    vocab_index = pd.Index(lda_vocab)
    for val in map_names:
        indexer = vocab_index.get_indexer(hlc_conc_lut.index.get_level_values(val))
        if np.sum(indexer==-1) > 0:
            raise RuntimeError("Some agreed words in the column of {0:s} are not in the LDA vocabulary!".format(val))

    # Read topic probabilities of some documents if given
    lda_lut_df = pd.read_csv(lda_lut_csv, header=0)

    # Create pseudo documents from agreed word combinations
    train_y = np.zeros((hlc_conc_lut.shape[0], len(hlc_code_list)))
    index = pd.Index(hlc_code_list).get_indexer(hlc_conc_lut.iloc[:, -1].astype(hc_dtype))
    train_y[np.array(range(0, train_y.shape[0])), index] = 1
    train_y = pd.DataFrame(np.hstack([train_y, 
        hlc_conc_lut.iloc[:, -1].values[:, np.newaxis], 
        np.ones((train_y.shape[0], 1))]), 
        index=hlc_conc_lut.index, columns=list(hlc_code_list)+["hlc", "weight"])
    df = train_y.reset_index()

    csv_suffix_fmtstr = "_{{0:0{0:d}d}}.csv".format(len(str(n_replicates)))
    for i in range(n_replicates):
        # Bootstrapping resampling agreed word combinations to create more pseudo docs.
        train_y = []
        for val in hlc_code_list:
            sflag = df["hlc"].astype(hc_dtype) == val
            balls = df.loc[sflag, :].values.tolist()
            x = np.array([tuple(val) for val in balls], dtype=[(str(i), type(val)) for i, val in enumerate(balls[0])])
            train_y.append(np.asarray(np.random.choice(x, size=n_per_hlc, replace=True).tolist()))
        train_y = np.vstack(train_y)
        train_y = np.hstack([train_y, np.zeros((train_y.shape[0], n_maps))])
        prop_names = [str(val)+"_prop" for val in map_names]
        train_y = pd.DataFrame(train_y, columns = list(df.columns)+prop_names)
        train_y.loc[:, prop_names] = np.random.dirichlet(np.ones(n_maps)*1./n_maps, size=train_y.shape[0])

        dw_mat = np.zeros((train_y.shape[0], len(lda_vocab)))
        for mn, pn in zip(map_names, prop_names):
            indexer = vocab_index.get_indexer(train_y[mn])
            dw_mat[np.array(range(dw_mat.shape[0])), indexer] = (train_y[pn] * N_factor).astype(int)
        train_X = pd.DataFrame(lda_model.transform(dw_mat), 
                               index = train_y.index, 
                               columns=["prob_topic_{0:d}".format(i+1) for i in range(lda_model.n_components)])

        train_data = pd.concat([train_y, train_X], axis=1, join="inner")

        # Set up the optimization problem to estimate T2H matrix
        n_topics = lda_model.n_components

        indptr = np.array(list(range(len(hlc_code_list)+1)), dtype=int)
        indices = np.array(list(range(len(hlc_code_list))), dtype=int)
        # for bsr sparse matrix, data is in the dimension of [nblocks, nrows_of_a_block, ncols_of_a_block]
        bsrmat_list = []
        tpcols = ["prob_topic_{0:d}".format(i+1) for i in range(n_topics)]
        for idx in train_data.index:
            data = np.tile(train_data.loc[idx, tpcols].values, (len(hlc_code_list), 1, 1))
            bsrmat_list.append( spspa.bsr_matrix((data, indices, indptr), shape=(len(hlc_code_list), len(hlc_code_list)*n_topics)) )
        A = spspa.vstack(bsrmat_list, format="bsr").toarray()

        b = train_data.loc[:, hlc_code_list].stack().values
        wt = train_data.loc[:, "weight"].values
        wt = np.repeat(wt, len(hlc_code_list))

        B = spspa.hstack([spspa.identity(n_topics)]*len(hlc_code_list)).toarray()

        # use random samples from Dirichlet distribution to initialize x0. 
        x0 = np.random.dirichlet(np.ones(len(hlc_code_list))*1./len(hlc_code_list), size = n_topics).T
        x0 = pd.DataFrame(x0, index=hlc_code_list, columns=list(range(n_topics)))
        x0.index.names = ["hlc"]
        x0.columns.names = ["topic"]
        x0 = x0.stack().reorder_levels(["hlc", "topic"]).values.squeeze()

        # Start optimization
        bounds = spopt.Bounds(0, 1)

        eq_cons = [{'type': 'eq', 
                    'fun': eqConsFunc, 
                    'jac': eqConsJac, 
                    'args': (B,)}]
        ineq_cons = [{'type': 'ineq', 
                      'fun': ineqCons1Func, 
                      'jac': ineqCons1Jac}, 
                     {'type': 'ineq', 
                      'fun': ineqCons2Func, 
                      'jac': ineqCons2Jac}]
        opt_out = spopt.minimize(objectiveFunc, x0, args=(A, b, wt), method="SLSQP", 
                                 bounds=bounds, constraints=eq_cons+ineq_cons, 
                                 options=dict(maxiter=200))

        t2h_mat = opt_out.x.reshape((-1, n_topics)).T
        t2h_mat = pd.DataFrame(t2h_mat, columns=hlc_code_list)

        # Write output
        if n_replicates == 1:
            cur_t2h_csv = t2h_csv
        else:
            cur_t2h_csv = t2h_csv.rstrip(".csv") + csv_suffix_fmtstr.format(i+1)
        with open(cur_t2h_csv, "w") as outfobj:
            outfobj.write(("# opt.success = {0:s}, " \
                    + "opt.status = {1:d}, " \
                    + "opt.message = {2:s}, " \
                    + "opt.nit = {3:d}\n").format(str(opt_out.success), 
                        opt_out.status, opt_out.message, opt_out.nit))
            t2h_mat.to_csv(outfobj, mode="a")

        if hlc_prob_csv is not None and lda_lut_csv is not None:
            # Now estimate the HLC class probabilities if requested
            tmp = np.matmul(lda_lut_df.iloc[:, -n_topics:].values, t2h_mat.values)
            hlc_prob_lut = pd.concat([lda_lut_df.iloc[:, 0:-n_topics], 
                pd.DataFrame(tmp, index=lda_lut_df.index, columns=hlc_code_list)], 
                axis=1)
            if n_replicates == 1:
                cur_hlc_prob_csv = hlc_prob_csv
            else:
                cur_hlc_prob_csv = hlc_prob_csv.rstrip(".csv") + csv_suffix_fmtstr.format(i+1)
            with open(cur_hlc_prob_csv, "w") as outfobj:
                outfobj.write("# t2h_matrix = {0:s}\n".format(cur_t2h_csv))
                hlc_prob_lut.to_csv(outfobj, mode="a")


if __name__ == "__main__":
    cmdargs = getCmdArgs()
    main(cmdargs)
