/*
 PPMGraphR: an R package for performing a Bayesian analysis
 of change point problems on a general graph.

 Copyright (C) 2012 Xiaofei Wang and Jay Emerson

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
#include <RcppArmadillo.h>  
#include <stdio.h>
#include <time.h>
#include <sys/timeb.h>
#include <math.h>
#include <Rmath.h>
#include <stdlib.h>
#include <R_ext/Random.h>
#include <R.h>
#include <Rdefines.h>
#include <vector>
//#include "rinterface.h"

using namespace std;
using namespace Rcpp;
using namespace arma;

/* DYNAMIC CREATION OF LOCAL VECTORS AND MATRICES */
typedef vector<double> DoubleVec;
typedef vector<int> IntVec;
typedef vector<IntVec> IntMatrix;
typedef vector<DoubleVec> DoubleMatrix;

/* Define some structures/objects*/



class ParamsGR  // maybe include beta someday?
{
public:
  NumericVector w;
  double alpha;
  int N;
  int nn2;
  int kk;
  int burnin;
  int mcmc;
  bool doneBurnin;
  double p1; // probability of correct APP
  double ba; 
  int freqAPP;
  colvec Y;
  mat X;
  uvec pred_cols;
  mat sigma_jitter;

  int boundaryType; // type = 1 is node, type = 2 is edge
  ParamsGR(SEXP, SEXP, SEXP, SEXP, int, int, 
         int, int, SEXP, SEXP, double);
  void print();
};
ParamsGR::ParamsGR(SEXP py, SEXP px, SEXP pw, SEXP a, int numLocs, int type, 
                int pburnin, int pmcmc, SEXP pp1, SEXP pfreqAPP,
                double pba)
{
  Y = as<colvec>(py);
  X = as<mat>(px);
  w = as<NumericVector>(pw);

  alpha = NUMERIC_DATA(a)[0];
  N = numLocs;
  nn2 = Y.n_rows;
  kk = w.size()-1;
  sigma_jitter = ones(kk,kk)*0.01;
  if (alpha == 0.0) {
    alpha = 0.1;  // this validation should be done in R
  }
  boundaryType = type;
  burnin = pburnin;
  mcmc = pmcmc;
  p1 = NUMERIC_DATA(pp1)[0];
  freqAPP = INTEGER_DATA(pfreqAPP)[0];
  ba = pba;
  pred_cols = zeros<uvec>(kk);
  for (int i = 0; i < kk; i++) {
    pred_cols[i] = i+1;
  }
  doneBurnin = false;
}
void ParamsGR::print()
{
  Rprintf("alpha=%0.2f, ba=%0.2f w.size:%d kk:%d\n", alpha, ba, w.size(), kk);
}

class NodeGR
{

public:
  int id;
  double value; // this is the actual value if only 1 obs, mean of all values at loc o/w
  int component;
  int active; // 0 = no, 1 = match some neighbor, 2 = disagree w all neighbors
  int boundlen;
  int size;
  IntegerVector neighbors;
  NodeGR(double, int, int, int, List &);
  void calcActiveAndBound(vector<NodeGR> &);
  void printNeighbors(vector<NodeGR> &);
};

NodeGR::NodeGR(double v, int c, int s, int idnum, List &adj)
{
  id = idnum;
  value = v;
  component = c;
  active = 0;
  boundlen = 0;
  SEXP adjvec = adj[idnum];
  IntegerVector adj1(adjvec);
  neighbors = adj1;
  size = s;
}
void NodeGR::printNeighbors(vector<NodeGR> &nodes)
{
  for (int i = 0; i < neighbors.size(); i++) {
    Rprintf("nb: %d, boundlen: %d, nb-comp:%d\n", neighbors[i], boundlen,
            nodes[neighbors[i]].component);
  }
}
void NodeGR::calcActiveAndBound(vector<NodeGR> &nodes)
{
  boundlen = 0;
  for (int i = 0; i < neighbors.size(); i++) {
    boundlen += (component != nodes[neighbors[i]].component);
  }
  if (boundlen == neighbors.size()) {
    active = 2;
  } else if (boundlen > 0) {
    active = 1;
  } else {
    active = 0;
  }
}

class GraphR
{
public:
  vector<NodeGR> nodes;
  double mean;
  IntMatrix boundarymat;
  uvec ids; 
  double sumysq; // for W calculation

  GraphR(SEXP &, SEXP &, NumericVector &, List &);
  void print(bool);
  void updateNodeGR(int, int);
  void updateBoundaryMatrix(int, int, int);
  void recomputeBoundary(ParamsGR &, int, int);
  void checkBound(int);
};
void GraphR::checkBound(int M)
{
  int i, j, nbBlock;
  int totBound = 0;
  int totBound2 = 0;
  for (i = 0; i < nodes.size(); i++) {
    IntVec blen(M, 0);
    for (j = 0; j < nodes[i].neighbors.size(); j++) {
      nbBlock = nodes[nodes[i].neighbors[j]].component;
      if (blen[nbBlock] == 0 & nbBlock != nodes[i].component) {
        blen[nbBlock] = 1;
        totBound++;
      }
    }
  }

  for (i = 0; i < nodes.size(); i++) {
    for (j = 0; j < M; j++) {
      totBound2 += boundarymat[j][i];
    }
  }
  Rprintf("totBound: %d | totBound2: %d\n", totBound, totBound2);
}
class ComponentGR
{

public:
  int size;
  double Z;
  double mean;
  double Q;
  double logC;
  double K;
  int tau;
  uvec nodeIds; // 1 indicates node is in block (length = N)
  uvec obsIds; // 1 indicates observation is in block (length = nn2)

  ComponentGR(ParamsGR &, NodeGR &, GraphR &);
  void addNodeGR(ParamsGR &, DoubleVec &, NodeGR &, GraphR &, int);
  void removeNodeGR(ParamsGR &, DoubleVec &, NodeGR &, GraphR &);
  void print();
  void changeTau(ParamsGR &, DoubleVec &, int);
  void initMemb(NodeGR &, GraphR &);
};


double logKcalc(int bsize, int tau, ParamsGR& params) {
  double kratio = params.ba/(bsize+params.ba);

  double tmp = (kratio*(bsize >= 2*params.kk) + 
          (bsize < 2*params.kk))*(tau==0) + 
    (1-kratio)*(bsize >= 2*params.kk)*(tau==1);

 // Rprintf("K:%0.2f %0.2f %0.2f\n", kratio, (kratio*(bsize >= 2*params.kk) + 
 //          (bsize < 2*params.kk))*(tau==0), tmp);  
  return log(tmp);
}
//simulate n samples from MVN(0, sigma=1e-5)
mat mvrnormArma(int n, ParamsGR& params) {
  mat Y = randn(n, params.kk);
  return Y * params.sigma_jitter;
}
vec betaPostPredCalcs(ParamsGR& params, mat &XtildeTrue,
                      DoubleVec &w, 
                      uvec &these, uvec &these2) {

  int i;
  int n = these.size();
  mat Winv = zeros(params.kk, params.kk);
  mat Xtilde = XtildeTrue;
  mat XXW = Xtilde.t()*Xtilde;
  bool ok = FALSE;
  while(!ok) {
    for (i = 0; i < params.kk; i++) {
      if (XXW(i,i) < 1e-12) {
        Xtilde = XtildeTrue+ mvrnormArma(n, params);
        XXW = Xtilde.t()*Xtilde;
        break;
      }
      Winv(i,i) = XXW(i,i)*w[i+1]/(1-w[i+1]);
      if (i == params.kk-1) ok = TRUE;
    }
  }
  XXW = XXW + Winv;
  mat sumxy = Xtilde.t()*params.Y.elem(these); 
  vec betapost = XXW.i()*sumxy;
  return betapost;
}
DoubleVec matrixCalcs(ParamsGR& params, 
                      DoubleVec &w, 
                      uvec &nodes) {
  int i;
  DoubleVec ret(2); // (Z, logC)
  uvec these = find(nodes==1);
  int n = these.size();
  mat Winv = zeros(params.kk, params.kk);
  mat Pmat = eye(n, n) - ones(n,n)/n; 
  mat Xtilde = Pmat*params.X.submat(these, params.pred_cols);
  mat XXW = Xtilde.t()*Xtilde;
  bool ok = FALSE;
  while(!ok) {
    for (i = 0; i < params.kk; i++) {
      if (XXW(i,i) < 1e-12) {
        Xtilde = Pmat*(params.X.submat(these, params.pred_cols)+ mvrnormArma(n, params));
        XXW = Xtilde.t()*Xtilde;
                // Rprintf("XXW:%0.15f\n", XXW(i,i));
        break;
      }
      Winv(i,i) = XXW(i,i)*w[i+1]/(1-w[i+1]);
      if (i == params.kk-1) ok = TRUE;
    }
  }
  XXW = XXW + Winv;
  mat sumxy = Xtilde.t()*params.Y.elem(these);
  ret[0] = as_scalar(sumxy.t()*XXW.i()*sumxy);
  
  double detval, detsign;
  mat tmp = XXW*Winv.i();
  // Rprintf("XXW:%0.2f sumxy:%0.2f\n", XXW(0,0), sumxy(0,0));
  log_det(detval, detsign, tmp);
  // Rprintf("Winv: %0.2f x^2: %0.2f x-bar:%0.2f\n", Winv(0,0), helpers.cumxsq[0][end],
  //   helpers.cumx[0][end]);
  ret[1] = -0.5*detval;
  return ret; 
}
ComponentGR::ComponentGR(ParamsGR& params, NodeGR &node, GraphR &graph)
{
  size = node.size;
  mean = node.value/size;
  Z = pow(mean, 2);
  obsIds = zeros<uvec>(params.nn2);
  uvec these = find(graph.ids==node.id);
  for (int i = 0; i < these.n_rows; i++) {
    obsIds[these[i]] = 1;
  }
  nodeIds = zeros<uvec>(params.N);
  nodeIds[node.id] = 1;
  tau = 0;
  Q = 0;
  K = logKcalc(size, tau, params);
  logC = 0;

}
void ComponentGR::addNodeGR(ParamsGR &params, DoubleVec &w, 
                        NodeGR &node, GraphR &graph, int ptau)
{
  size += node.size;
  mean = ((size - node.size) * mean + node.value) / size;
  Z = size * pow(mean, 2);
  uvec these = find(graph.ids==node.id);
  for (int i = 0; i < these.n_rows; i++) {
    obsIds[these[i]] = 1;
  }
  nodeIds[node.id] = 1;
  changeTau(params, w, ptau);
}

void ComponentGR::initMemb(NodeGR &node, GraphR &graph)
{
  size += node.size;
  mean = ((size - node.size) * mean + node.value) / size;
  Z = size * pow(mean, 2);
  uvec these = find(graph.ids==node.id);
  for (int i = 0; i < these.n_rows; i++) {
    obsIds[these[i]] = 1;
  }
  nodeIds[node.id] = 1;
}

void ComponentGR::changeTau(ParamsGR &params, DoubleVec &w, 
                          int ptau)
{
  tau = ptau;
  K = logKcalc(size, tau, params);
  if (tau == 1) {
    DoubleVec out = matrixCalcs(params, w, obsIds);
    Q = out[0];
    logC = out[1]; 
  } else {
    Q = 0;
    logC = 0;
  }
}
void ComponentGR::removeNodeGR(ParamsGR &params, DoubleVec &w, 
                          NodeGR &node, GraphR &graph)
{
  // recompute the values for the current component (assuming we move this node to another component)
  if (size == node.size) {
    mean = 0.0;
    Z = 0.0;
    size = 0;
  } else {
    mean = (mean * size - node.value) / (size - node.size);
    size -= node.size;
    Z = size * pow(mean, 2);
  }
  uvec these = find(graph.ids==node.id);
  for (int i = 0; i < these.n_rows; i++) {
    obsIds[these[i]] = 0;
  }
  nodeIds[node.id] = 0;
  if (size < params.kk*2) changeTau(params, w, 0);
  else changeTau(params, w, tau);
}
void ComponentGR::print()
{
  Rprintf("Z: %0.2f, size:%d, mean: %0.2f Q:%0.2f logC:%0.2f K:%0.2f tau:%0d\n", 
          Z, size, mean, Q, logC, K, tau);
}
typedef vector<ComponentGR> PartitionGR;
void recomputeVals(GraphR &graph, PartitionGR &components)
{
  // DoubleVec W(components.size(), 0.0);
  DoubleVec B(components.size(), 0.0);
  DoubleVec means(components.size(), 0.0);
  int currblock, i;
  for (i = 0; i < graph.nodes.size(); i++) {
    currblock = graph.nodes[i].component;
    means[currblock] += graph.nodes[i].value;
    //   Rprintf("memb:%d, val:%0.2f means:%0.2f\n", currblock, graph.nodes[i].value, means[currblock]);
    // W[currblock] += graph.nodes[i].value * graph.nodes[i].value;
  }
  for (i = 0; i < components.size(); i++) {
    means[i] /= components[i].size;
    B[i] = components[i].size * pow(means[i], 2);
    // W[i] -= B[i];
    Rprintf("Recomputed: i:%d, B: %0.2f, size: %d, mean: %0.2f\n", i, B[i],
            components[i].size, means[i]);
  }
}

GraphR::GraphR(SEXP &pdata, SEXP &pids, NumericVector &membs, List &adj)
{
  NumericVector data(pdata);
  ids = as<uvec>(pids);
  mean = 0.0;
  sumysq = 0.0;
  int M = 0;
  int nbComponentGR, i, j;

  // for multiple observations per location
  double nodesum = 0;
  int nodesize = 0;
  int curr_id = 0; 

  for (i = 0; i < data.size(); i++) {
    mean += data[i];    
    sumysq += pow(data[i], 2);
    if (ids[i] > curr_id) {
      NodeGR node(nodesum, (int) membs[curr_id], nodesize, curr_id, adj);
      nodes.push_back(node);

      curr_id++;
      nodesum = data[i];
      nodesize = 1;
      if (M < (int) membs[curr_id] + 1) {
        M = (int) membs[curr_id] + 1;
      }
    } else {
      nodesum += data[i];
      nodesize++;
    } 
    
  }
  // the last node will need to be added
  NodeGR node(nodesum, (int) membs[curr_id], nodesize, curr_id, adj);
  nodes.push_back(node);

  mean /= data.size();
  IntVec vboundary(nodes.size(), 0);
  boundarymat.assign(M, vboundary);
  for (i = 0; i < nodes.size(); i++) {
    for (j = 0; j < nodes[i].neighbors.size(); j++) {
      nbComponentGR = nodes[nodes[i].neighbors[j]].component;
      if (nbComponentGR != nodes[i].component) {
        boundarymat[nbComponentGR][i] = 1;
      }
    }
  }
}
void GraphR::print(bool all = false)
{
  Rprintf("overall mean:%0.2f, num pixels: %d\n", mean, nodes.size());
  if (all) {
    for (int i = 0; i < nodes.size(); i++) {
      Rprintf("Node i:%d in block: %d, size:%d, sum(obs):%0.2f, boundlen: %d\n", i, nodes[i].component, 
              nodes[i].size, nodes[i].value, nodes[i].boundlen);
    }
    Rprintf("Boundary matrix\n");
    for (int i = 0; i < nodes.size(); i++) {
      for (int j = 0; j < 3; j++) {
        Rprintf("%d", boundarymat[j][i]);
      }
      Rprintf("\n");
    }
  }
}
void GraphR::updateNodeGR(int nodeId, int componentId)
{
  nodes[nodeId].component = componentId;
  nodes[nodeId].calcActiveAndBound(nodes);
  for (int i = 0; i < nodes[nodeId].neighbors.size(); i++) {
    nodes[nodes[nodeId].neighbors[i]].calcActiveAndBound(nodes);
  }
}
void GraphR::updateBoundaryMatrix(int nodeId, int newblock, int currblock)
{
  boundarymat[newblock][nodeId] = 0;
  int boundaryOld = 0;
  int i, j, neighborNodeGRId, nbBoundaryOld;
  for (i = 0; i < nodes[nodeId].neighbors.size(); i++) {
    neighborNodeGRId = nodes[nodeId].neighbors[i];
    if (nodes[neighborNodeGRId].component == currblock) {
      boundaryOld = 1;
    }
    if (nodes[neighborNodeGRId].component != newblock) {
      boundarymat[newblock][neighborNodeGRId] = 1;
    }

    nbBoundaryOld = 0;
    for (j = 0; j < nodes[neighborNodeGRId].neighbors.size(); j++) {
      if (nodes[nodes[neighborNodeGRId].neighbors[j]].component == currblock
          && nodes[neighborNodeGRId].component != currblock) {
        nbBoundaryOld = 1;
        break;
      }
    }
    boundarymat[currblock][neighborNodeGRId] = nbBoundaryOld;
  }
  boundarymat[currblock][nodeId] = boundaryOld;
}
// type =1 : node counting
// type = 2: edge counting
void GraphR::recomputeBoundary(ParamsGR &params, int M, int len)
{
  int blen = 0;
  int nbblock, i, j;

  if (params.boundaryType == 1) {
    IntVec vboundary(params.N, 0);
    IntMatrix boundarymat2(M, vboundary);
    for (i = 0; i < nodes.size(); i++) {
      for (j = 0; j < nodes[i].neighbors.size(); j++) {
        nbblock = nodes[nodes[i].neighbors[j]].component;
        if (nodes[i].component != nbblock & boundarymat2[nbblock][i] == 0) {
          boundarymat2[nbblock][i] = 1;
          blen += 1;
        }
      }
    }
    for (i = 0; i < nodes.size(); i++) {
      for (j = 0; j < M; j++) {
        if (boundarymat2[j][i] != boundarymat[j][i]) {
          Rprintf("ERROR:\n");
        }
      }
    }
    if (blen != len) {
      Rprintf("ERROR len\n");
    }

  } else if (params.boundaryType == 2) {
    for (i = 0; i < nodes.size(); i++) {
      for (j = 0; j < nodes[i].neighbors.size(); j++) {
        blen += nodes[nodes[i].neighbors[j]].component != nodes[i].component;
      }
    }
  }

  // Rprintf("boundlen:%d\n", blen);
}
void printPartitionGR(PartitionGR &components)
{
  for (int i = 0; i < components.size(); i++) {
    Rprintf("i:%d ", i);
    components[i].print();
  }
}
class MCMCStepGR
{
public:
  double ll;
  double W;
  double B;
  int len;
  int M;
  double Q;
  double logC;
  double K;
  DoubleVec w;

  MCMCStepGR();
  MCMCStepGR(PartitionGR &, GraphR &, ParamsGR &, DoubleVec &);
  void calcLogLik(ParamsGR &);
  void updateLogLik(ParamsGR &, GraphR &, PartitionGR &, ComponentGR &, ComponentGR &, NodeGR &, int);
  void updateLogLikForMerge(ParamsGR &, GraphR &, PartitionGR &, ComponentGR &, int, int);
  void print();
};
void MCMCStepGR::print()
{
  Rprintf("ll:%0.2f, W:%0.2f, B:%0.2f, logC:%0.2f, K:%0.2f, Q:%0.2f, len =%d, M=%d\n", 
           ll, W, B, logC, K, Q, len, M);
  for (int i = 0; i < w.size(); i++) Rprintf("w: %0.6f", w[i]);
  Rprintf("\n");
} 
MCMCStepGR::MCMCStepGR()
{
  ll = 0.0;
  W = 0.0;
  B = 0.0;
  len = 0;
  M = 0;
  logC = 0;
  Q = 0;
  K = 0;

}
MCMCStepGR::MCMCStepGR(PartitionGR &components, GraphR &graph, 
                   ParamsGR &params, DoubleVec &w0)
{
  int i, j;
  W = graph.sumysq;
  B = -params.nn2 * pow(graph.mean, 2);
  M = components.size();
  len = 0;
  Q = 0;
  logC = 0;
  K = 0;
  for (i = 0; i < components.size(); i++) {
    W -= components[i].Z;
    B += components[i].Z;
    Q += components[i].Q;
    logC += components[i].logC;
    K += components[i].K;
  }
  w = w0;
  for (i = 0; i < params.N; i++) {   
    if (params.boundaryType == 1) {
      for (j = 0; j < M; j++) {
        len += graph.boundarymat[j][i];
      }
    }
    else if (params.boundaryType == 2) 
      len += graph.nodes[i].boundlen;
  }
  calcLogLik(params);
}
class MCMCGR
{
public:
  DoubleVec ll;
  IntVec Mvals;
  DoubleVec wstarvals;
  IntVec boundlens;
  DoubleVec simErr;
  IntVec type2pix;

  int k; // keeping track of which position we're at

  // posterior stuff
  vec pmeans;
  vec pvar;
  vec ss;
  DoubleVec pboundary;
  DoubleVec movedBlock; // to keep track # times a node moves

  // current values from most recent MCMC step
  MCMCStepGR step;
  MCMCGR(PartitionGR &, GraphR &, ParamsGR &, DoubleVec &);
  void addStep(ParamsGR &);
  void postProcessing(ParamsGR &, int, mat &);
  // void simulateValues(GraphR &, PartitionGR &, ParamsGR &);
};
MCMCGR::MCMCGR(PartitionGR &components, GraphR &graph, ParamsGR &params, DoubleVec &w0)
{
  MCMCStepGR step1(components, graph, params, w0);
  step = step1;

  pvar = zeros<vec>(params.N);
  pmeans = zeros<vec>(params.N);
  ss = zeros<vec>(params.N);
  pboundary.assign(params.N, 0);
  movedBlock.assign(params.N, 0);
  simErr.assign(params.N, 0);

  int MM = params.burnin + params.mcmc + 101;
  ll.assign(MM, 0);
  Mvals.assign(MM, 0);
  wstarvals.assign(MM, 0);
  type2pix.assign(params.mcmc + params.burnin, 0);
  boundlens.assign(MM, 0);

  k = 0;
  addStep(params);
}
void MCMCGR::postProcessing(ParamsGR &params, int mcmc, mat & betaPosts)
{
  for (int i = 0; i < params.N; i++) {
    pmeans[i] /= mcmc;
    pboundary[i] /= mcmc;
    simErr[i] /= mcmc;
    movedBlock[i] /= mcmc*(params.freqAPP+1);
    pvar[i] = (ss[i] /mcmc - pmeans[i]*pmeans[i])*(mcmc/(mcmc-1));
  }
  betaPosts /= mcmc;
  betaPosts.cols(params.kk+1, betaPosts.n_cols-1) -= betaPosts.cols(0, params.kk)%betaPosts.cols(0, params.kk);
  //  Rprintf("mcmcits: %d\n", params.itsMCMC);
}
void MCMCGR::addStep(ParamsGR &params)
{
  ll[k] = step.ll;
  Mvals[k] = step.M;
  boundlens[k] = step.len;
  
  double wstar = 0.0;
  // step.print();
  if (step.M > 1) {
    double Wtilde = step.W - step.Q;
    double xmax = step.B * params.w[0] / Wtilde / 
                    (1 + (step.B * params.w[0] / Wtilde));
    // Rprintf("W:%0.2f Q:%0.2f\n", step.W, step.Q);

    wstar = log(Wtilde) - log(step.B)
            + Rf_pbeta(xmax, (double) (step.M + 3) / 2, (double) (params.nn2 - step.M - 4) / 2, 1, 1)
            + Rf_lbeta((double) (step.M + 3) / 2, (double) (params.nn2 - step.M - 4) / 2)
            - Rf_pbeta(xmax, (double) (step.M + 1) / 2, (double) (params.nn2 - step.M - 2) / 2, 1, 1)
            - Rf_lbeta((double) (step.M + 1) / 2, (double) (params.nn2 - step.M - 2) / 2);
    wstar = exp(wstar);
  } else {
    wstar = params.w[0] / 2;
  }
  wstarvals[k] = wstar;
  
  k++;
}
void MCMCStepGR::calcLogLik(ParamsGR &params)
{
  double Wtilde = W - Q;

  if (M == 1) {
    ll = logC + K + log(params.w[0]) 
          - (params.nn2 - 1) * log(Wtilde) / 2;
  } else if (M >= params.N - 5) {
    ll = -DBL_MAX;
  } else {
    double xmax = (B * params.w[0] / Wtilde) / (1 + (B * params.w[0] / Wtilde));
    //  Rprintf("xmax:%0.2f M:%d B:%0.2f W:%0.2f len:%d\n", xmax, M, B, W, len);
    ll = logC + K + len * log(params.alpha)
         + Rf_pbeta(xmax, (double) (M + 1) / 2, (double) (params.nn2 - M - 2) / 2, 1, 1)
         + Rf_lbeta((double) (M + 1) / 2, (double) (params.nn2 - M - 2) / 2) 
         - (M + 1) * log(B) / 2
         - (params.nn2 - M - 2) * log(Wtilde) / 2;
  }
}

void MCMCStepGR::updateLogLik(ParamsGR &params, GraphR &graph, 
                            PartitionGR &partition, ComponentGR &newcomp,
                            ComponentGR &oldcomp, NodeGR &node, int newCompId)
{
  int oldtau = partition[node.component].tau;
  if (newCompId == node.component && oldtau == newcomp.tau) {
    return;
  }
  if (newCompId != node.component) {
    int neighborsOldBlock, neighborOfOldComp, i, j;
    double Zdiff;
    M += 1 * (newCompId == M) - 1 * (partition[node.component].size == node.size);
    // update bound length
    if (params.boundaryType == 1) {
      if (newCompId >= graph.boundarymat.size()) {
        IntVec vboundary(params.N, 0);
        graph.boundarymat.push_back(vboundary);
      }
      neighborsOldBlock = 0;
      for (i = 0; i < node.neighbors.size(); i++) {
        // subtract neighboring nodes that are no longer on the boundary of oldComp
        // this disqualifies any neighboring nodes that are in oldComp
        if (graph.nodes[node.neighbors[i]].component != node.component) {
          // these now used to be boundary of oldComp;
          // check if any of their other neighbors are in oldComp
          neighborOfOldComp = 0;
          NodeGR neighborNodeGR = graph.nodes[node.neighbors[i]];
          for (j = 0; j < neighborNodeGR.neighbors.size(); j++) {
            if (neighborNodeGR.neighbors[j] == node.id) {
              continue;
            }
            if (graph.nodes[neighborNodeGR.neighbors[j]].component == node.component) {
              neighborOfOldComp = 1;
              break;
            }
          }
          len -= (1 - neighborOfOldComp);
        } else {
          neighborsOldBlock = 1;
        }


        // add nodes that were not previously on the boundary of newComp but now are
        len -= graph.boundarymat[newCompId][node.neighbors[i]];
        len += (graph.nodes[node.neighbors[i]].component != newCompId);
      }
      // node is no longer boundary of newComp
      // add 1 if node is boundary of old block
      len -= graph.boundarymat[newCompId][node.id];
      len += neighborsOldBlock;

    } else if (params.boundaryType == 2) {
      len -= 2 * node.boundlen;
      for (i = 0; i < node.neighbors.size(); i++) {
        len += 2 * (graph.nodes[node.neighbors[i]].component != newCompId);
      }
    }

    // update globals
    Zdiff = partition[node.component].Z - newcomp.Z - oldcomp.Z;
    if (newCompId < partition.size()) {
      Zdiff += partition[newCompId].Z;
    }
    B -= Zdiff;
    W += Zdiff;
  }

  Q += newcomp.Q - partition[node.component].Q;
  K += newcomp.K - partition[node.component].K;
  logC += newcomp.logC - partition[node.component].logC;
  if (newCompId != node.component) {
    Q += oldcomp.Q;
    K += oldcomp.K;
    logC += oldcomp.logC;
    if (newCompId < partition.size()) {
      Q -= partition[newCompId].Q;
      K -= partition[newCompId].K;
      logC -= partition[newCompId].logC;
    }
  }  
  

  if (abs(W) < 1.0e-12) {
    W = 1.0e-12;
  }
  calcLogLik(params);
}
void MCMCStepGR::updateLogLikForMerge(ParamsGR &params, GraphR &graph, 
                            PartitionGR &partition, ComponentGR &newcomp, 
                            int currblock, int newblock)
{
  int i;
  double Zdiff;
  M--;
  // update bound length
  if (params.boundaryType == 1) {      
    // subtract boundaries between the old and new comp (since merged now)
    for (i = 0; i < params.N; i++) {
      if (newcomp.nodeIds[i] == 1) {
        len = len - graph.boundarymat[newblock][i] - graph.boundarymat[currblock][i];
      }
      // subtract 1 if a node is boundary to both blocks;
      if (graph.boundarymat[currblock][i] == 1 && graph.boundarymat[newblock][i] == 1) {
        len--;
      }
        
    }
  } 

  // update globals
  Zdiff = partition[newblock].Z + partition[currblock].Z - newcomp.Z;
  B -= Zdiff;
  W += Zdiff;


  Q += newcomp.Q - partition[newblock].Q - partition[currblock].Q;
  K += newcomp.K - partition[newblock].K - partition[currblock].K;
  logC += newcomp.logC - partition[newblock].logC - partition[currblock].logC;
  if (abs(W) < 1.0e-12) {
    W = 1.0e-12;
  }
  calcLogLik(params);
}
int sampleLogLikGR(vector<MCMCStepGR> possibleSteps, double maxll)
{
  int j;
  double myrand = Rf_runif(0.0, 1.0);
  DoubleVec llcum(possibleSteps.size());

  llcum[0] = exp(possibleSteps[0].ll - maxll);
  for (j = 1; j < possibleSteps.size(); j++) {
    llcum[j] = llcum[j - 1] + exp(possibleSteps[j].ll - maxll);
  }

  // begin binary search
  int startkk = 0;
  int endkk = llcum.size() - 1;
  int midkk;

  while (startkk != endkk) {
    midkk = floor((startkk + endkk) / 2);
    if (myrand <= llcum[midkk] / llcum[llcum.size() - 1]) {
      // search lower half
      endkk = midkk;
    } else {
      // search upper half
      startkk = midkk + 1;
    }
  }

  return (endkk);
}

void updateComponentGRs(ParamsGR &params, MCMCGR &mcmc, PartitionGR &components, 
                      GraphR &graph, vector<MCMCStepGR> &possibleSteps,
                      vector<ComponentGR> &possibleBlocks, 
                      int currblock, int newblock, int index,
                      int nodeId)
{
  int i;
  if (newblock == currblock && 
    components[currblock].tau == possibleBlocks[index+1].tau) {
    return;
  }
  mcmc.step = possibleSteps[index];

  // if (mcmc.step.ll != mcmc.step.ll) {
  //   Rprintf("ll is nan!\n");
  // }
  if (newblock == currblock) {
    // tau changed
    components[currblock] = possibleBlocks[index+1];
    return;
  } else { // newly added 11.01 to check movedBlocks
    if (params.doneBurnin) mcmc.movedBlock[nodeId]++;
  }

  components[currblock] = possibleBlocks[0];
  if (newblock == components.size()) {
    components.push_back(possibleBlocks[index+1]);
  } else {
    components[newblock] = possibleBlocks[index+1];
  }

  graph.updateNodeGR(nodeId, newblock);

  // update the boundaryMatrix
  if (params.boundaryType == 1) {
    graph.updateBoundaryMatrix(nodeId, newblock, currblock);
  }

  if (components[currblock].size == 0) {
    if (currblock == components.size() - 1) {
      components.pop_back();
    } else {
      components[currblock] = components.back();
      components.pop_back();

      for (i = 0; i < graph.nodes.size(); i++) {
        if (graph.nodes[i].component == components.size()) {
          graph.nodes[i].component = currblock;
        }
        if (params.boundaryType == 1) {
          if (graph.boundarymat[components.size()][i] == 1) {
            graph.boundarymat[currblock][i] = 1;
            graph.boundarymat[components.size()][i] = 0;
          }
        }
      }
    }
  }  
}

void updateComponentGRsForMerge(ParamsGR &params, MCMCGR &mcmc, PartitionGR &components, 
                      GraphR &graph, MCMCStepGR &possibleStep,
                      ComponentGR &possibleBlock, 
                      int currblock, int newblock)
{
  int i;
  if (newblock == currblock) {
    return;
  }
  mcmc.step = possibleStep;
  components[newblock] = possibleBlock;

  // update the boundaryMatrix
  if (params.boundaryType == 1) {
    for (i = 0; i < params.N; i++) {
      if (components[newblock].nodeIds[i]==1) {
        graph.updateNodeGR(i, newblock);
        graph.boundarymat[newblock][i] = 0;            
      } else if (graph.boundarymat[currblock][i] == 1) graph.boundarymat[newblock][i] = 1;
      graph.boundarymat[currblock][i] = 0;   // currblock effectively nonexistent now
    }
  }
  if (currblock == components.size() - 1) {
    components.pop_back();
  } else {
    components[currblock] = components.back();
    components.pop_back();

    for (i = 0; i < params.N; i++) {
      if (graph.nodes[i].component == components.size()) {
        graph.nodes[i].component = currblock;
      }
      if (params.boundaryType == 1) {
        if (graph.boundarymat[components.size()][i] == 1) {
          graph.boundarymat[currblock][i] = 1;
          graph.boundarymat[components.size()][i] = 0;
        }
      }
    }
  }
  graph.recomputeBoundary(params, mcmc.step.M, mcmc.step.len);
}
void fullPixelPass(GraphR &graph, PartitionGR &components, ParamsGR &params, MCMCGR &mcmc, 
                   bool silent)
{
  int i, j, maxComp, l, tau;
  int currblock, s;
  double maxll;
  for (i = 0; i< params.N; i++) {  
    currblock = graph.nodes[i].component;
    maxComp = components.size() + (components[currblock].size != graph.nodes[i].size);
    IntVec blockNums;
    // this vector is not a partition, but rather stores the modified
    // candidate component information
    // [0] will always be the original component minus node i
    vector<ComponentGR> possibleBlocks; 
    vector<MCMCStepGR> possibleSteps;
    maxll = mcmc.step.ll;

    possibleBlocks.push_back(components[currblock]);
    possibleBlocks[0].removeNodeGR(params, mcmc.step.w, graph.nodes[i], graph); // remove the node from current block
    l = 1;
    // loop through all possible components
    for (j = 0; j < maxComp; j++) {
      for (tau = 0; tau < 2; tau++) {
        if (j == components.size()){
          if (tau == 0) {
            ComponentGR newestBlock(params, graph.nodes[i], graph);
            possibleBlocks.push_back(newestBlock);
            possibleSteps.push_back(mcmc.step);
          } else break;
        } else if (j == currblock) {
          if (tau == 1 && components[currblock].size < 2* params.kk) {
            break;              
          }
          possibleBlocks.push_back(components[j]);
          possibleSteps.push_back(mcmc.step);
          if (tau != components[currblock].tau) {                
            possibleBlocks[l].changeTau(params, mcmc.step.w, tau);
          }      
        } else {
          if (tau == 0) {
            possibleSteps.push_back(mcmc.step);
            possibleBlocks.push_back(components[j]);
            possibleBlocks[l].addNodeGR(params, mcmc.step.w, graph.nodes[i], graph, 0);
          } else if (possibleBlocks[l-1].size >= 2* params.kk) {
            possibleSteps.push_back(mcmc.step);
            possibleBlocks.push_back(possibleBlocks[l-1]);
            possibleBlocks[l].changeTau(params, mcmc.step.w, tau);
          } else break;
        }
        possibleSteps[l-1].updateLogLik(params, graph, components, 
                          possibleBlocks[l],
                          possibleBlocks[0], 
                          graph.nodes[i], j);
        blockNums.push_back(j);
        if (possibleSteps[l-1].ll > maxll) {
          maxll = possibleSteps[l-1].ll;
        }
        l++;
      
      }
    }

    s = sampleLogLikGR(possibleSteps, maxll);  
    updateComponentGRs(params, mcmc, components, graph, possibleSteps, 
                         possibleBlocks, currblock, blockNums[s],
                         s, i);

  }
}
void blockMergePass(GraphR &graph, PartitionGR &components, ParamsGR &params, MCMCGR &mcmc, bool silent)
{
  int i, j, l;
  int s;
  double maxll;
  // Rprintf("len:%d\n", mcmc.step.len);
  for (i = 0; i< mcmc.step.M; i++) {

    vector<ComponentGR> possibleBlocks; 
    vector<MCMCStepGR> possibleSteps;
    IntVec blockNums;
    maxll = mcmc.step.ll;
    l = 0;
    for (j = 0; j < mcmc.step.M; j++) {

      if (components[i].tau != components[j].tau) continue;
      // recalculate our block quantities
      possibleBlocks.push_back(components[i]);
      blockNums.push_back(j); // store the destination block
      possibleSteps.push_back(mcmc.step);
      if (i != j) {
        possibleBlocks[l].size = components[i].size + components[j].size;
        possibleBlocks[l].mean = (components[i].mean*components[i].size + components[j].mean*components[j].size)/
                                  possibleBlocks[l].size;
        possibleBlocks[l].Z = possibleBlocks[l].size*pow(possibleBlocks[l].mean,2);
        possibleBlocks[l].tau = components[i].tau;
        possibleBlocks[l].nodeIds = components[i].nodeIds + components[j].nodeIds;
        possibleBlocks[l].obsIds = components[i].obsIds + components[j].obsIds;
        possibleBlocks[l].K = logKcalc(possibleBlocks[l].size, possibleBlocks[l].tau, params);
        if (components[j].tau==1) {
          DoubleVec out = matrixCalcs(params, mcmc.step.w, possibleBlocks[l].obsIds);
          possibleBlocks[l].Q = out[0];
          possibleBlocks[l].logC = out[1]; 
        } 
        possibleSteps[l].updateLogLikForMerge(params, graph, components, possibleBlocks[l], i, j);
        if (possibleSteps[l].ll > maxll) {
          maxll = possibleSteps[l].ll;
        }
      }    
      // Rprintf("j:%d  len:%d\n", j, possibleSteps[l].len);
      l++;

    }
    s = sampleLogLikGR(possibleSteps, maxll); 
    updateComponentGRsForMerge(params, mcmc, components, 
                      graph, possibleSteps[s],
                      possibleBlocks[s], 
                      i, blockNums[s]);
  }
}
void wPass(PartitionGR &components, ParamsGR &params, MCMCGR &mcmc)
{
  int i,j;
  double probs;
  for (i = 1; i< params.w.size(); i++) {
    vector<ComponentGR> possibleBlocks = components; 
    MCMCStepGR candidateStep = mcmc.step;
    candidateStep.w = mcmc.step.w;
    candidateStep.w[i] += Rf_runif(-0.05*params.w[i], 0.05*params.w[i]);
    if (candidateStep.w[i] > params.w[i] || candidateStep.w[i] < 0) continue;
    candidateStep.Q = 0;
    candidateStep.logC = 0;

    for (j = 0; j < mcmc.step.M; j++) {
      possibleBlocks[j].changeTau(params, candidateStep.w, 
                                  possibleBlocks[j].tau); 
      candidateStep.Q += possibleBlocks[j].Q;
      candidateStep.logC += possibleBlocks[j].logC;
    }

    candidateStep.calcLogLik(params);
    probs = exp(candidateStep.ll-mcmc.step.ll);
    probs = probs/(1+probs);
    if (Rf_runif(0.0,1.0) < probs) {
      mcmc.step = candidateStep;
      components = possibleBlocks;
    } 
  }
}
void activePixelPass(GraphR &graph, PartitionGR &components, ParamsGR &params, MCMCGR &mcmc, bool silent)
{
  int i, j, k, l, s, tau;
  int currblock, passtype;
  double maxll, maxComp;

  if (params.p1 == 1) {
    passtype = 1;  // correct APP
  } else if (params.p1 == 0) {
    passtype = 2;  // modified APP
  } else {
    double u = Rf_runif(0.0, 1.0);
    if (u < params.p1) {
      passtype = 1;
    } else {
      passtype = 2;
    }
  }

  for (i = 0; i < params.N; i++) {
    if (graph.nodes[i].active == 0) {
      continue;
    }
    // Rprintf("i:%d  ",i);

    currblock = graph.nodes[i].component;
    IntegerVector neighbors = graph.nodes[i].neighbors;
    maxll = mcmc.step.ll;

    // if (graph.nodes[i].active == 2) {
    //   mcmc.type2pix[mcmc.k - 101]++;
    // }
    vector<ComponentGR> possibleBlocks; 
    vector<MCMCStepGR> possibleSteps; 
    IntVec blockNums;   
    possibleBlocks.push_back(components[currblock]);
    possibleBlocks[0].removeNodeGR(params, mcmc.step.w, graph.nodes[i], graph); // remove the node from current block
    l = 1;

    if (graph.nodes[i].active == 1 || passtype == 2) {
      IntVec blockTried(mcmc.step.M, 0);
      // loop through all possible components
      for (k = 0; k < neighbors.size(); k++) {
        j = graph.nodes[neighbors[k]].component;
        if (blockTried[j] == 1) continue;
        blockTried[j] = 1;
        for (tau = 0; tau < 2; tau++) {
          if (j == currblock) {
            if (tau == 1 && components[currblock].size < 2* params.kk) {
              break;              
            }
            possibleBlocks.push_back(components[j]);
            possibleSteps.push_back(mcmc.step);
            if (tau != components[currblock].tau) {                
              possibleBlocks[l].changeTau(params, mcmc.step.w, tau);
            }      
          } else {
            if (tau == 0) {
              possibleSteps.push_back(mcmc.step);
              possibleBlocks.push_back(components[j]);
              possibleBlocks[l].addNodeGR(params, mcmc.step.w, graph.nodes[i], graph, 0);
            } else if (possibleBlocks[l-1].size >= 2* params.kk) {
              possibleSteps.push_back(mcmc.step);
              possibleBlocks.push_back(possibleBlocks[l-1]);
              possibleBlocks[l].changeTau(params, mcmc.step.w, tau);
            } else break;
          }
          possibleSteps[l-1].updateLogLik(params, graph, components, 
                            possibleBlocks[l],
                            possibleBlocks[0], 
                            graph.nodes[i], j);
          blockNums.push_back(j);
          if (possibleSteps[l-1].ll > maxll) {
            maxll = possibleSteps[l-1].ll;
          }
          l++;
          // Rprintf("i:%d j:%d l:%d currblock:%d\n", i, j, l, currblock);
        } 
      }  
      
    } else { // active = 2; disagrees with all neighbors
      maxComp = (double) (mcmc.step.M+(components[currblock].size > graph.nodes[i].size));
      IntVec blocksToTry(maxComp, 1); // indicators of whether or not we try these blocks
      for (k = 0; k < neighbors.size(); k++) {
        blocksToTry[graph.nodes[neighbors[k]].component] = 0;
      }
      // loop through all possible components
      for (j = 0; j < maxComp; j++) {
        if (blocksToTry[j] == 0) {
          continue;
        }
        for (tau = 0; tau < 2; tau++) {
          if (j == components.size()){
            if (tau == 0) {
                ComponentGR newestBlock(params, graph.nodes[i], graph);
                possibleBlocks.push_back(newestBlock);
                possibleSteps.push_back(mcmc.step);
            } else break;
          } else if (j == currblock) {
            if (tau == 1 && components[currblock].size < 2* params.kk) {
              break;              
            }
            possibleBlocks.push_back(components[j]);
            possibleSteps.push_back(mcmc.step);
            if (tau != components[currblock].tau) {                
              possibleBlocks[l].changeTau(params, mcmc.step.w, tau);
            }      
          } else {
            
            if (tau == 0) {
              possibleSteps.push_back(mcmc.step);
              possibleBlocks.push_back(components[j]);
              possibleBlocks[l].addNodeGR(params, mcmc.step.w, graph.nodes[i], graph, 0);
            } else if (possibleBlocks[l-1].size >= 2* params.kk) {
              possibleSteps.push_back(mcmc.step);
              possibleBlocks.push_back(possibleBlocks[l-1]);
              possibleBlocks[l].changeTau(params, mcmc.step.w, tau);
            } else break;
          }
          possibleSteps[l-1].updateLogLik(params, graph, components, 
                            possibleBlocks[l],
                            possibleBlocks[0], 
                            graph.nodes[i], j);
          blockNums.push_back(j);
          if (possibleSteps[l-1].ll > maxll) {
            maxll = possibleSteps[l-1].ll;
          }
          l++;
        // Rprintf("i:%d j:%d l:%d currblock:%d\n", i, j, l, currblock);
        }
      }
    }
    s = sampleLogLikGR(possibleSteps, maxll);
    updateComponentGRs(params, mcmc, components, graph, possibleSteps, 
                     possibleBlocks, currblock, blockNums[s],
                     s, i);
  }

}

// [[Rcpp::export]]
SEXP rcpp_ppmR(SEXP py, SEXP px, SEXP pgrpinds, SEXP pid, SEXP padj, SEXP pmcmcreturn, 
              SEXP pburnin, SEXP pmcmc, SEXP pa, SEXP pc,
              SEXP pmembs, 
              SEXP pboundaryType, SEXP pba, 
              SEXP p1, SEXP pfreqAPP)
{
  double ba = NUMERIC_DATA(pba)[0];

  List adj(padj);
  NumericVector membInit(pmembs);

  int mcmcreturn = INTEGER_DATA(pmcmcreturn)[0];
  int burnin = INTEGER_DATA(pburnin)[0];
  int itsmcmc = INTEGER_DATA(pmcmc)[0];
  int bType = INTEGER_DATA(pboundaryType)[0];
  int MM = burnin + itsmcmc;

  // some helper variables
  mat grpInds = as<mat>(pgrpinds);
  int i, m, bsize;
  double wstar;

  int freqAPP = INTEGER_DATA(pfreqAPP)[0];
  // initialize my graph
  // Rprintf("Initializing objects\n");
  GraphR graph(py, pid, membInit, adj);
  ParamsGR params(py, px, pc, pa, graph.nodes.size(), 
                bType, burnin,
                itsmcmc, p1, pfreqAPP, ba);
  // params.print();
  PartitionGR components;
  // Rprintf("Initializing graph\n");
  for (i = 0; i < params.N; i++) {
    graph.nodes[i].calcActiveAndBound(graph.nodes);
    if ((int) membInit[i] >= components.size()) {
      ComponentGR newComp(params, graph.nodes[i], graph);
      components.push_back(newComp);
    } else {
      components[(int) membInit[i]].initMemb(graph.nodes[i], graph);
    }
  }
  DoubleVec w0(params.kk+1);
  for (i = 0; i <= params.kk; i++) {
    if (i > 0 & i <= params.kk)
      w0[i] = params.w[i]/2;
    else if (i == 0) 
      w0[i] = params.w[0];
  }
  // Rprintf("ybar:%0.4f\n", graph.mean);
  for (i = 0; i < components.size(); i++) {
    components[i].changeTau(params, w0, 0);
    // components[i].print();
  }
  MCMCGR mcmc(components, graph, params, w0);
  /* --- MAKE THINGS TO BE RETURNED TO R --- */

  // Rprintf("Initializing matrix\n");
  int MM2, nn2;
  if (mcmcreturn == 1) {
    MM2 = MM;
    nn2 = params.N;
  } else {
    MM2 = 1;
    nn2 = 1;
  }
  NumericMatrix membsa(nn2, MM2);
  NumericMatrix taus(nn2, MM2); 
  // store (fitted, intercept, slopes)
  mat results = zeros(MM2*nn2, 2+params.kk); // store conditional means

  mat betaposts = zeros(params.N, (1+params.kk)*2);
  uvec interceptcols1(1);
  uvec interceptcols2(1);
  interceptcols1[0] = 0;
  interceptcols2[0] = params.kk+1;
  /* --- END THINGS TO BE RETURNED TO R --- */
  GetRNGstate(); // Consider Dirk's comment on this.

  // Rprintf("Before 100 pixel passes loop\n");
  // mcmc.step.print();

  /* ----------------START THE BIG LOOP--------------------------- */
  // first 100 iterations, do complete pixel passes
  bool silent = true;
  for (m = 0; m < 100; m++) {
    fullPixelPass(graph, components, params, mcmc, silent);
    blockMergePass(graph, components, params, mcmc, silent);
    wPass(components, params, mcmc);   
    mcmc.addStep(params);
  }
  // Rprintf("After 100 full pixel passes\n");

  // now do the rest:
  // 
  for (m = 0; m < MM; m++) {
    // Rprintf("m:%d\n", m);
    fullPixelPass(graph, components, params, mcmc, true);
    if (mcmc.step.len > 0) {
      for (i = 0; i < freqAPP; i++) {
        activePixelPass(graph, components, params, mcmc, true);
      }
      blockMergePass(graph, components, params, mcmc, silent);
    }
    wPass(components, params, mcmc);

    // passes are done, now store important info
    mcmc.addStep(params);
    // mcmc.step.print();
    if (m == burnin) params.doneBurnin = true; // silly but this is used in another func
    wstar = mcmc.wstarvals[mcmc.k-1];

    if (params.doneBurnin || mcmcreturn == 1) {
      // for posterior estimate of error variance
      // if (params.doneBurnin) 
      //   mcmc.pvar += (mcmc.step.W + mcmc.step.B * wstar) / (params.nn2 - 3);
      for (i = 0; i < params.N; i++) {
        if (mcmcreturn == 1){
          membsa(i, m) = graph.nodes[i].component;        
          if (i < mcmc.step.M) {
            taus(i, m) = components[i].tau;
          }   
        }  
        if (params.doneBurnin) 
          mcmc.pboundary[i] += (graph.nodes[i].active > 0);
        if(i < mcmc.step.M) { 
          bsize = components[i].size;
          uvec these = find(components[i].obsIds == 1);
          uvec these2 = find(components[i].nodeIds == 1);
          uvec resultRows = these2 + params.N*m;          
          mat grpIndMat = grpInds(these2, these);  
          vec intercepts = grpIndMat*(ones(bsize, 1)*(components[i].mean * (1 - wstar) 
                           + graph.mean * wstar));
          vec fitted = intercepts;
          
          if (components[i].tau == 1) {
            mat Pmat = eye(bsize, bsize) - ones(bsize, bsize)/bsize; 
            mat Xtilde = Pmat*params.X.submat(these, params.pred_cols);
            vec betapost = betaPostPredCalcs(params, Xtilde, mcmc.step.w, these, 
                                        these2);
            mat betapostMat = repmat(betapost.t(), these2.n_rows, 1); 
            intercepts -= betapostMat*params.X.submat(these, params.pred_cols).t()*ones(bsize, 1)/bsize;
            fitted += grpIndMat*Xtilde*betapost;

            if (params.doneBurnin)  {              
              betaposts(these2, params.pred_cols) +=  betapostMat;
              betaposts(these2, params.pred_cols+params.kk+1) += betapostMat%betapostMat;
            }
            if (mcmcreturn == 1)
              results(resultRows, params.pred_cols+1) = betapostMat;
          } 
          if (mcmcreturn == 1) {
            results(resultRows, interceptcols1) = fitted;
            results(resultRows, interceptcols1+1) = intercepts;
          }
          if (params.doneBurnin) {
            mcmc.pmeans.elem(these2) += fitted;
            mcmc.ss.elem(these2) += fitted % fitted;
            betaposts(these2, interceptcols1) += intercepts;
            betaposts(these2, interceptcols2) += intercepts % intercepts;
          }
            
        } 
      }
      
    }

    

  }
  // mcmc.step.print();
  // printPartitionGR(components);

  // Rprintf("Begin post-processing\n");

  mcmc.postProcessing(params, itsmcmc, betaposts);
  PutRNGstate();
  //
  List z;
  z["posterior.mean"] = wrap(mcmc.pmeans);
  z["posterior.var"] = wrap(mcmc.pvar);
 
  // z["ll"] = wrap(mcmc.ll);
  z["posterior.prob"] = wrap(mcmc.pboundary);
  z["mcmc.rhos"] = membsa;
  z["blocks"] = wrap(mcmc.Mvals);
  z["len"] = wrap(mcmc.step.len);
  // z["type2pix"] = wrap(mcmc.type2pix);
  // z["simErr"] = wrap(mcmc.simErr);
  // z["tau"] = wrap(taus);
  z["mcmc.means"] = wrap(results);
  z["betaposts"] = wrap(betaposts);
  z["movedBlock"] = wrap(mcmc.movedBlock);
  // Rprintf("End printing\n");

  //  closeLogFile();

  return z;
}
/* END MAIN  */

