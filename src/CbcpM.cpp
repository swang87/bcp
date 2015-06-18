/*
 bcp: an R package for performing a Bayesian analysis
 of change point problems.

 Copyright (C) 2013 Susan Wang and John W. Emerson

 This program is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 2 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, a copy is available at
 http://www.r-project.org/Licenses/

 -------------------
 FILE: Cbcp.cpp  */

/*  LIBRARIES  */
#include <Rcpp.h>                               // Are all these needed???
#include <stdio.h>
#include <math.h>
#include <Rmath.h>
#include <stdlib.h>
#include <R_ext/Random.h>
#include <R.h>
#include <Rdefines.h>
#include <vector>

using namespace std;
using namespace Rcpp;

/* DYNAMIC CREATION OF LOCAL VECTORS AND MATRICES */
typedef vector<double> DoubleVec;
typedef vector<DoubleVec> DoubleMatrix;
typedef vector<int> IntVec;

/* BEGIN BCP-M stuff */
class HelperVariablesM
{
public:
  DoubleMatrix cumy;
  double ybar;
  double ysqall;
  IntVec cumksize;

  HelperVariablesM(NumericMatrix &, SEXP &);
  void print();
};
HelperVariablesM::HelperVariablesM(NumericMatrix &data, 
                        SEXP &pid)
{
  NumericVector id(pid);
  int mm = data.ncol();
  int nn2 = data.nrow();
  int N = id[id.size()-1]+1; // number of locations
  int curr_id = 0;
  cumksize.push_back(1);

  DoubleVec cumyc(N);
  cumy.assign(mm, cumyc);
  ysqall = 0.0;
  ybar = 0.0;
  
  for (int i = 0; i < mm; i++) {
    cumy[i][0] = data(0, i);
    ysqall += pow(data(0, i), 2);
  }
  for (int j = 1; j < nn2; j++) {
    if (id[j] > curr_id) {
      for (int i = 0; i < mm; i++) {
        cumy[i][id[j]] = cumy[i][curr_id] + data(j,i);
        ysqall += pow(data(j, i), 2);
      }
      cumksize.push_back(cumksize[curr_id]+1);
      curr_id++;
    } else {
      for (int i = 0; i < mm; i++) {
        cumy[i][curr_id] += data(j, i);
        ysqall += pow(data(j, i), 2);
      }
      cumksize[curr_id]++;
    }
  }
  for (int i = 0; i < mm; i++)
    ybar += cumy[i][N-1];
  ybar /= (nn2 * mm);
}
void HelperVariablesM::print() {
  Rprintf("Helper Variables Print ----\n");
  // int n = cumksize.size();
  Rprintf("ybar:%0.2f, ysqall:%0.2f\n", ybar, ysqall);
  // for (int i = 0; i < n; i++) {
  //   Rprintf("i:%d, k:%0.2d, Y:%0.2f\n",
  //     i, cumksize[i], cumy[3][i]);
  // }
}
class ParamsM  
{
public:
  double w1;
  int mm;
  int nn; // number of locs
  int nn2; // number of obs
  double p0;
  DoubleVec priors;

  ParamsM(double, int, int, int, double);
  void print();
};

ParamsM::ParamsM(double pw1, int pmm, int pnn, int pnn2,
                 double pp0)
{
  w1 = pw1;
  mm = pmm;
  nn = pnn;
  nn2 = pnn2;
  p0 = pp0;

  double tmp;
  for (int j = 1; j < nn - 2; j++) { // maybe it can be bigger?
    tmp = Rf_pbeta(p0, (double) j, (double) nn - j + 1, 1, 1)
          + Rf_lbeta((double) j, (double) nn - j + 1);
    priors.push_back(tmp);
  }

}
void ParamsM::print() {
  Rprintf("Params-- nodes: %d locs: %d dim: %d p0:%0.2f w:%0.2f\n", 
           nn, nn2, mm, p0, w1);
}

class MCMCStepM
{
public:
  double W;
  double B;
  int b;
  double lik;

  // blocks variables
  IntVec rho;

  IntVec bend;
  IntVec bsize;
  DoubleMatrix bmean;
  DoubleVec bZ;

  MCMCStepM(const MCMCStepM &);
  MCMCStepM(HelperVariablesM &, ParamsM &);
  void print();
};
MCMCStepM::MCMCStepM(const MCMCStepM &step)
{
  W = step.W;
  B = step.B;
  b = step.b;
  lik = step.lik;
}
double likelihoodM(double B, double W, int b, ParamsM &params)
{
  double lik;
  if (B == 0) {
    lik = params.priors[b - 1] + (params.mm + 1) * log(params.w1) / 2 -
          (params.nn2 * params.mm - 1) * log(W) / 2;
  } else if (b >= params.nn - 4 / params.mm) {
    lik = -DBL_MAX;
  } else {
    lik = params.priors[b - 1] - (params.mm * b + 1) * log(B) / 2 -
          ((params.nn2 - b) * params.mm - 2) * log(W) / 2
          + Rf_pbeta(B * params.w1 / W / (1 + B * params.w1 / W), 
                     (double) (params.mm * b + 1) / 2,
                     (double) ((params.nn2 - b) * params.mm - 2) / 2, 1, 1)
          + Rf_lbeta((double) (params.mm * b + 1) / 2,
                     (double) ((params.nn2 - b) * params.mm - 2) / 2);
  }
  return lik;
}
MCMCStepM::MCMCStepM(HelperVariablesM &helpers, ParamsM &params)
{
  int nn = params.nn; // for conciseness
  int nn2 = params.nn2;
  double bZtmp = 0;
  DoubleVec bmean1(params.mm);
  b = 1; // start with 1 block
  for (int i = 0; i < nn - 1; i++) {
    rho.push_back(0);
  }
  rho.push_back(1);

  bend.push_back(nn - 1);
  bsize.push_back(nn2);
  for (int i = 0; i < params.mm; i++) {
    bmean1[i] = helpers.cumy[i][nn - 1] / nn2;
    bZtmp += pow(bmean1[i], 2) * nn2;
  }
  bmean.assign(1, bmean1);
  bZ.push_back(bZtmp);

  B = bZtmp - params.nn2 * params.mm * pow(helpers.ybar, 2);
  W = helpers.ysqall - bZtmp;
  lik = likelihoodM(B, W, b, params);
}
void MCMCStepM::print()
{
  Rprintf("MCMCStep Info\n");
  Rprintf("B: %0.4f  W:%0.4f  b:%d   lik:%0.2f\n", B, W,
          b, lik);
}
int sampleFromLikelihoodsM(DoubleVec &likvals)
{
  double p = exp(likvals[0] - likvals[1]);
  p = p / (1 + p);
  double u = Rf_runif(0.0, 1.0);
  if (u < p) return 0;
  else return 1;
}

MCMCStepM pass(MCMCStepM &step, HelperVariablesM &helpers, ParamsM &params)
{

  int i, j, cp;

  int bsize1, bsize2, bsize3;
  double tmp;
  DoubleVec bmean1(params.mm, 0);
  DoubleVec bmean2(params.mm, 0);
  DoubleVec bmean3(params.mm, 0);
  double bZ1, bZ2, bZ3;

  DoubleVec likvals(2);
  IntVec bvals(2);
  DoubleVec Wvals(2);
  DoubleVec Bvals(2);

  DoubleVec bmeanlast(params.mm);
  double bZlast = 0;

  // this is used to reference the current block in the MCMCStep we came in with
  int prevblock = 0;

  // this is used to reference the current MCMCStep we build from scratch
  MCMCStepM stepnew(step);
  int currblock = 0;

  // some other variables to denote stuff in current block and previous block
  // Note that "last" refers to the immediately left-adjacent block
  // whereas "prev" refers to the same block in the variable step
  double thisblockZ = step.bZ[0];
  int thisbend = step.bend[0];

  double lastblockZ = 0;
  int lastbend = -1; // this is simply a notational convenience

  // start the loop
  for (i = 0; i < params.nn - 1; i++) {
    if (i == step.bend[prevblock]) {
      // we're at an old change point, so we need to refresh "this" to be the
      // immediately following block
      lastblockZ = thisblockZ;
      prevblock++;

      thisbend = step.bend[prevblock];
      thisblockZ = step.bZ[prevblock];
    }
    /****
     * consider merging blocks if currently a change point
     */
    bvals[0] = stepnew.b;
    if (step.rho[i] == 0) {
      Bvals[0] = stepnew.B;
      Wvals[0] = stepnew.W;
      likvals[0] = stepnew.lik;
    } else {
      bvals[0]--;
      tmp = thisblockZ + lastblockZ;
      bZ3 = 0;
      if (lastbend > -1) {
        bsize3 = helpers.cumksize[thisbend] - helpers.cumksize[lastbend];
        for (j = 0; j < params.mm; j++) {
          bmean3[j] = (helpers.cumy[j][thisbend] - helpers.cumy[j][lastbend]) / bsize3;
          bZ3 += pow(bmean3[j], 2) * bsize3;
        }
      } else {
        bsize3 = helpers.cumksize[thisbend];
        for (j = 0; j < params.mm; j++) {
          bmean3[j] = helpers.cumy[j][thisbend] / bsize3;
          bZ3 += pow(bmean3[j], 2) * bsize3;
        }
      }
      if (params.mm == 1 && bvals[0] == 1) Bvals[0] = 0; // force this to avoid rounding errs
      else 
        Bvals[0] = stepnew.B - tmp + bZ3;
      Wvals[0] = stepnew.W + tmp - bZ3;
      likvals[0] = likelihoodM(Bvals[0], Wvals[0], bvals[0], params);
    }


    /****
     * consider breaking blocks if not a change point
     */
    bvals[1] = stepnew.b;
    if (step.rho[i] == 1) {
      Bvals[1] = stepnew.B;
      Wvals[1] = stepnew.W;
      likvals[1] = stepnew.lik;
    } else {
      bZ1 = 0;
      bZ2 = 0;
      bvals[1]++;
      bsize2 = helpers.cumksize[thisbend] - helpers.cumksize[i];

      if (lastbend > -1)
        bsize1 = helpers.cumksize[i] - helpers.cumksize[lastbend];
      else
        bsize1 = helpers.cumksize[i];
      tmp = thisblockZ;
      for (j = 0; j < params.mm; j++) {
        bmean2[j] = (helpers.cumy[j][thisbend] - helpers.cumy[j][i]) / bsize2;
        if (lastbend > -1) {
          bmean1[j] = (helpers.cumy[j][i] - helpers.cumy[j][lastbend]) / bsize1;
        } else {
          bmean1[j] = helpers.cumy[j][i] / bsize1;
        }
        bZ1 += pow(bmean1[j], 2) * bsize1;
        bZ2 += pow(bmean2[j], 2) * bsize2;
      }
      Bvals[1] = stepnew.B - tmp + bZ1 + bZ2;
      Wvals[1] = stepnew.W + tmp - bZ1 - bZ2;
      likvals[1] = likelihoodM(Bvals[1], Wvals[1], bvals[1], params);
    }

    // do the sampling and then updates
    cp = sampleFromLikelihoodsM(likvals);

    stepnew.lik = likvals[cp];
    stepnew.B = Bvals[cp];
    stepnew.W = Wvals[cp];
    stepnew.b = bvals[cp];

    if (cp != step.rho[i]) { // we modified the change point status
      if (cp == 0) {
        // removed a change point
        // update last block's stuff since the last block is now further back
        thisblockZ = bZ3;
        if (currblock > 0) {
          lastbend = stepnew.bend[currblock - 1];
          lastblockZ = stepnew.bZ[currblock - 1];
        } else {
          lastblockZ = 0;
          lastbend = -1; // this is simply a notational convenience
        }
      } else { // added a change point
        thisblockZ = bZ2;
        lastblockZ = bZ1;
      }
    }
    stepnew.rho.push_back(cp);

    if (stepnew.rho[i] == 1) {
      if (step.rho[i] == 1) { // never calculated these quantities yet; do it now
        lastblockZ = 0;

        if (lastbend > -1)
          bsize1 = helpers.cumksize[i] - helpers.cumksize[lastbend];
        else
          bsize1 = helpers.cumksize[i];
        for (j = 0; j < params.mm; j++) {
          if (lastbend > -1) {
            bmean1[j] = (helpers.cumy[j][i] - helpers.cumy[j][lastbend]) / bsize1;
          } else {
            bmean1[j] = helpers.cumy[j][i] / bsize1;
          }
          lastblockZ += pow(bmean1[j], 2) * bsize1;
        }
      }
      // we've added a change point, so we want to record some stuff
      stepnew.bsize.push_back(bsize1);
      stepnew.bend.push_back(i);
      stepnew.bmean.push_back(bmean1);
      stepnew.bZ.push_back(lastblockZ);
      currblock++;
      lastbend = i;

    }
  }
  // done with a full pass, now let's add info on the final block
  if (lastbend > -1)
    stepnew.bsize.push_back(params.nn2 - helpers.cumksize[lastbend]);
  else 
    stepnew.bsize.push_back(params.nn2);
  for (j = 0; j < params.mm; j++) {
    if (lastbend > -1) {
      bmeanlast[j] = (helpers.cumy[j][params.nn - 1] - helpers.cumy[j][lastbend]) / stepnew.bsize[currblock];
    } else {
      bmeanlast[j] = helpers.cumy[j][params.nn - 1] / params.nn2;
    }
    bZlast += pow(bmeanlast[j], 2) * stepnew.bsize[currblock];
  }
  stepnew.bmean.push_back(bmeanlast);
  stepnew.bZ.push_back(bZlast);
  stepnew.bend.push_back(params.nn - 1);

  return stepnew;
}

// [[Rcpp::export]]
SEXP rcpp_bcpM(SEXP pdata, SEXP pid, SEXP pmcmcreturn, SEXP pburnin, SEXP pmcmc,
                         SEXP pa, SEXP pw)
{

  NumericMatrix data(pdata);
  int mcmcreturn = INTEGER_DATA(pmcmcreturn)[0];
  int burnin = INTEGER_DATA(pburnin)[0];
  int mcmc = INTEGER_DATA(pmcmc)[0];

  double w0 = NUMERIC_DATA(pw)[0]; // w1
  double p0 = NUMERIC_DATA(pa)[0];
  if (p0 == 0)
    p0 = 0.001;

  // INITIALIZATION OF LOCAL VARIABLES
  int i, j, m, k;
  double wstar, xmax;

  int MM = burnin + mcmc;
  // INITIALIZATION OF OTHER OBJECTS
  // Rprintf("numrows:%d\n", data.nrow());
  HelperVariablesM helpers(data, pid);
  ParamsM params(w0, data.ncol(), helpers.cumksize.size(), data.nrow(), p0);
  MCMCStepM step(helpers, params);
  // helpers.print();
  // params.print();
  int nn = params.nn;
  int mm = data.ncol();
  int nn2 = nn;
  int MM2 = burnin + mcmc;
  if (mcmcreturn == 0) {
    MM2 = 1;
    nn2 = 1;
  }
  // Things to be returned to R:
  NumericMatrix pmean(nn, mm);
  NumericMatrix ss(nn, mm);
  NumericMatrix pvar(nn, mm);
  NumericVector pchange(nn);
  NumericVector blocks(burnin + mcmc);
  NumericMatrix rhos(nn2, MM2);
  // NumericVector liks(MM2);
  NumericMatrix results(nn2*MM2,mm);

  double tmpMean;
  
  // Rprintf("starting\n");
  GetRNGstate(); // Consider Dirk's comment on this.
  // step.print();
  for (i = 0; i < nn; i++) {
    pchange[i] = 0;
    for (j = 0; j < mm; j++) {
      pmean(i, j) = 0;
    }
  }
  for (m = 0; m < MM; m++) {

    step = pass(step, helpers, params);
    // Rprintf(" m %d\n", m);
    // step.print();
    blocks[m] = step.b;
    if (m >= burnin || mcmcreturn == 1) {
      // compute posteriors
      if (step.B == 0) {
        wstar = params.w1 * (step.b*params.mm + 1) / (step.b * params.mm +3);
      } else {
        xmax = step.B * params.w1 / step.W / (1 + step.B * params.w1 / step.W);
        wstar = log(step.W) - log(step.B)
          + Rf_lbeta((double) (step.b* params.mm + 3) / 2, (double) ((params.nn2 - step.b)*params.mm - 4) / 2)
          + Rf_pbeta(xmax, (double) (step.b*params.mm + 3) / 2, (double) ((params.nn2  - step.b)*params.mm - 4) / 2, 1, 1)
          - Rf_lbeta((double) (step.b*params.mm + 1) / 2, (double) ((params.nn2  - step.b)*params.mm - 2) / 2)
          - Rf_pbeta(xmax, (double) (step.b * params.mm+ 1) / 2, (double) ((params.nn2  - step.b)*params.mm - 2) / 2, 1, 1);
        wstar = exp(wstar);
      }
      // for posterior estimate of overall noise variance      
      // if (m >= burnin)
        // pvar += (step.W + wstar*step.B)/(params.nn2 * params.mm-3); 
      k = 0;
      for (j = 0; j < nn; j++) {
        if (m >= burnin)
          pchange[j] += (double) step.rho[j];
        for (i = 0; i < mm; i++) {
          tmpMean = step.bmean[k][i] * (1 - wstar) + helpers.ybar * wstar;
          if (m >= burnin) {            
            pmean(j, i) += tmpMean;
            ss(j, i) += tmpMean * tmpMean;
          }
          if (mcmcreturn == 1) 
            results(m*nn+j, i) = tmpMean;
        }
                    
        if (mcmcreturn == 1)
          rhos(j, m) = step.rho[j];
        if (step.rho[j] == 1) k++;

      }
    }
  }
  // Rprintf("post processing\n");
  // step.print();
  // post processing
  for (j = 0; j < nn; j++) {
    pchange[j] /= mcmc;
    for (i = 0; i < mm; i++) {
      pmean(j, i) /= mcmc;
      pvar(j, i) = (ss(j, i) / mcmc - pmean(j,i)*pmean(j,i))*(mcmc/(mcmc-1));
    }
  }
  // Rprintf("ending\n");

  PutRNGstate();

  List z;
  z["posterior.mean"] = pmean;
  z["posterior.var"] = pvar;
  z["posterior.prob"] = pchange;
  z["blocks"] = blocks;
  z["mcmc.rhos"] = rhos;
  z["mcmc.means"] = results;
  // z["lik"] = liks;
  return z;

} /* END MAIN  */

/* END BCP-M STUFF*/
