/*
 PPMGraph: an R package for performing a Bayesian analysis
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
#include <Rcpp.h>
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

/* DYNAMIC CREATION OF LOCAL VECTORS AND MATRICES */
typedef vector<double> DoubleVec;
typedef vector<int> IntVec;
typedef vector<IntVec> IntMatrix;
typedef vector<DoubleVec> DoubleMatrix;

/* Define some structures/objects*/
class Node
{

public:
  int id;
  DoubleVec value;
  int component;
  int active; // 0 = no, 1 = match some neighbor, 2 = disagree w all neighbors
  int boundlen;
  int size;
  IntegerVector neighbors;
  Node(DoubleVec &, int, int, int, List &);
  void calcActiveAndBound(vector<Node> &);
  void printNeighbors(vector<Node> &);
};

Node::Node(DoubleVec &v, int c, int s, int idnum, List &adj)
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
void Node::printNeighbors(vector<Node> &nodes)
{
  for (int i = 0; i < neighbors.size(); i++) {
    Rprintf("nb: %d, boundlen: %d, nb-comp:%d\n", neighbors[i], boundlen,
            nodes[neighbors[i]].component);
  }
}
void Node::calcActiveAndBound(vector<Node> &nodes)
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

class ParamsG   // maybe include beta someday?
{
public:
  double w0;
  double alpha;
  int N; // number of locs
  int nn2; // number of obs
  int burnin;
  int mcmc;
  int mm;
  double p1; // probability of correct APP

  int boundaryType; // type = 1 is node, type = 2 is edge
  ParamsG(double, double, int, int, int, int, int, double, int);
  void print();
};
class Graph
{
public:
  vector<Node> nodes;
  double mean;
  IntMatrix boundarymat;
  // uvec ids; 
  double sumysq; // for W calculation

  Graph(NumericMatrix &, SEXP&, NumericVector &, List &);
  void print(bool);
  void updateNode(int, int);
  void updateBoundaryMatrix(int, int, int);
  void recomputeBoundary(ParamsG &, int, int);
  void checkBound(int);
};
void Graph::checkBound(int M)
{
  int totBound = 0;
  int totBound2 = 0;
  int i, j, nbBlock;

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
class Component
{

public:
  int size;
  double Z;
  DoubleVec mean;

  Component(ParamsG &);
  Component(Node &);
  void addNode(Node &);
  void removeNode(Node &);
  void print();
};

Component::Component(ParamsG &params)
{
  size = 0;
  Z = 0.0;
  mean.assign(params.mm, 0);
}
Component::Component(Node &node)
{
  size = node.size;
  Z = 0;

  for (int i = 0; i < node.value.size(); i++) {
    mean.push_back(node.value[i]/node.size);
    Z += pow(mean[i], 2);
  }
  Z *= size;
}
void Component::addNode(Node &node)
{
  size += node.size;
  Z = 0;
  for (int i = 0; i < node.value.size(); i++) {
    mean[i] = ((size - node.size) * mean[i] + node.value[i]) / size;
    Z += pow(mean[i], 2);
  }
  Z *= size;
}
void Component::removeNode(Node &node)
{
  int i;

  // recompute the values for the current component (assuming we move this node to another component)
  Z = 0;
  if (size == node.size) {
    for (i = 0; i < node.value.size(); i++) {
      mean[i] = 0.0;
    }
    size = 0;
  } else {
    for (i = 0; i < node.value.size(); i++) {
      mean[i] = (mean[i] * size - node.value[i]) / (size - node.size);
      Z += pow(mean[i], 2);
    }
    size -= node.size;
    Z *= size;
  }
}
void Component::print()
{
  Rprintf("Z: %0.2f, size:%d, mean:%0.2f\n", Z, size, mean[0]);
}
typedef vector<Component> Partition;
void recomputeVals(Graph &graph, Partition &components, ParamsG &params)
{
  DoubleVec W(components.size(), 0.0);
  DoubleVec B(components.size(), 0.0);
  DoubleVec mean(params.mm, 0.0);
  DoubleMatrix means(components.size(), mean);
  int currblock, i, j;

  for (i = 0; i < graph.nodes.size(); i++) {
    currblock = graph.nodes[i].component;
    for (j = 0; j < params.mm; j++) {
      means[currblock][j] += graph.nodes[i].value[j];
      //   Rprintf("memb:%d, val:%0.2f means:%0.2f\n", currblock, graph.nodes[i].value, means[currblock]);
      W[currblock] += pow(graph.nodes[i].value[j], 2);
    }
  }

  for (i = 0; i < components.size(); i++) {
    B[i] = 0;

    for (j = 0; j < params.mm; j++) {
      means[i][j] /= components[i].size;
      B[i] += components[i].size * pow(means[i][j], 2);
    }

    W[i] -= B[i];
    Rprintf("Recomputed: i:%d, W: %0.2f, B: %0.2f, size: %d, %0.2f\n", i, W[i], B[i],
            components[i].size);
  }
}

ParamsG::ParamsG(double w, double a, int numLocs, int numNodes, int type, int pburnin,
               int pmcmc, double pp1, int pmm)
{
  w0 = w;
  alpha = a;
  N = numLocs;
  nn2 = numNodes;
  if (alpha == 0.0) {
    alpha = 0.1;  // this validation should be done in R
  }

  boundaryType = type;
  burnin = pburnin;
  mcmc = pmcmc;
  p1 = pp1;
  mm = pmm;
}
void ParamsG::print()
{
  Rprintf("alpha=%0.2f, w0=%0.2f, locs:%d, obs:%d \n", alpha, w0, N, nn2);
}

Graph::Graph(NumericMatrix &data, SEXP& pid, NumericVector &membs, List &adj)
{
  mean = 0.0;
  sumysq = 0.0;
  int M = 0;
  int i, j, nbComponent;
  DoubleVec values2;
  IntegerVector ids(pid);

  // for multiple observations per location
  DoubleVec nodesum(data.ncol());
  int nodesize = 0;
  int curr_id = 0; 

  for (i = 0; i < data.nrow(); i++) {
    if (ids[i] > curr_id) {
      Node node(nodesum, (int) membs[curr_id], nodesize, curr_id, adj);
      nodes.push_back(node);
      
      curr_id++;
      for (j = 0; j < data.ncol(); j++) {
        mean += data(i, j);
        sumysq += pow(data(i, j), 2);
        nodesum[j] = data(i,j);
      }
      nodesize = 1;
      if (M < (int) membs[i] + 1) {
        M = (int) membs[i] + 1;
      }
    } else {
      for (j = 0; j < data.ncol(); j++) {
        mean += data(i, j);
        sumysq += pow(data(i, j), 2);
        nodesum[j] += data(i,j);
      }
      nodesize++;
    }
  }
  // the last node will need to be added
  Node node(nodesum, (int) membs[curr_id], nodesize, curr_id, adj);
  nodes.push_back(node);

  mean /= (data.nrow() * data.ncol());

  IntVec vboundary(data.nrow(), 0);
  boundarymat.assign(M, vboundary);

  for (i = 0; i < nodes.size(); i++) {
    for (j = 0; j < nodes[i].neighbors.size(); j++) {
      nbComponent = nodes[nodes[i].neighbors[j]].component;

      if (nbComponent != nodes[i].component) {
        boundarymat[nbComponent][i] = 1;
      }
    }
  }
}
void Graph::print(bool all = false)
{
  Rprintf("overall mean:%0.2f, overall ysq:%0.2f, num pixels: %d\n", 
          mean, sumysq, nodes.size());

  if (all) {
    for (int i = 0; i < nodes.size(); i++) {
      Rprintf("Node i:%d in block: %d, size:%d, sum(obs):%0.2f, boundlen: %d\n", i, nodes[i].component, 
              nodes[i].size, nodes[i].value[1], nodes[i].boundlen);
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
void Graph::updateNode(int nodeId, int componentId)
{
  nodes[nodeId].component = componentId;
  nodes[nodeId].calcActiveAndBound(nodes);

  for (int i = 0; i < nodes[nodeId].neighbors.size(); i++) {
    nodes[nodes[nodeId].neighbors[i]].calcActiveAndBound(nodes);
  }
}
void Graph::updateBoundaryMatrix(int nodeId, int newblock, int currblock)
{
  boundarymat[newblock][nodeId] = 0;
  int boundaryOld = 0;
  int neighborNodeId, nbBoundaryOld, i, j;

  for (i = 0; i < nodes[nodeId].neighbors.size(); i++) {
    neighborNodeId = nodes[nodeId].neighbors[i];

    if (nodes[neighborNodeId].component == currblock) {
      boundaryOld = 1;
    }

    if (nodes[neighborNodeId].component != newblock) {
      boundarymat[newblock][neighborNodeId] = 1;
    }

    nbBoundaryOld = 0;

    for (j = 0; j < nodes[neighborNodeId].neighbors.size(); j++) {
      if (nodes[nodes[neighborNodeId].neighbors[j]].component == currblock
          && nodes[neighborNodeId].component != currblock) {
        nbBoundaryOld = 1;
        break;
      }
    }

    boundarymat[currblock][neighborNodeId] = nbBoundaryOld;
  }

  boundarymat[currblock][nodeId] = boundaryOld;
}
// type =1 : node counting
// type = 2: edge counting
void Graph::recomputeBoundary(ParamsG &params, int M, int len)
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
void printPartition(Partition &components)
{
  for (int i = 0; i < components.size(); i++) {
    Rprintf("i:%d ", i);
    components[i].print();
  }
}
class MCMCStepG
{
public:
  double ll;
  double W;
  double B;
  int len;
  int M;

  MCMCStepG();
  MCMCStepG(Partition &, Graph &, ParamsG &);
  void calcLogLik(ParamsG &);
  void updateLogLik(ParamsG &, Graph &, Partition &, Component &, Component &, Node &, int);
  void print();
};
void MCMCStepG::print()
{
  Rprintf("ll:%0.2f, W:%0.2f, B:%0.2f, len =%d, M=%d\n", ll, W, B, len, M);
}
MCMCStepG::MCMCStepG()
{
  ll = 0.0;
  W = 0.0;
  B = 0.0;
  len = 0;
  M = 0;
}
MCMCStepG::MCMCStepG(Partition &components, Graph &graph, ParamsG &params)
{
  int i, j;
  W = graph.sumysq;
  B = -params.nn2 * params.mm * pow(graph.mean, 2);
  M = components.size();
  len = 0;

  for (i = 0; i < components.size(); i++) {
    W -= components[i].Z;
    B += components[i].Z;
  }

  if (params.boundaryType == 1) {
    for (i = 0; i < params.N; i++) {
      for (j = 0; j < M; j++) {
        len += graph.boundarymat[j][i];
      }
    }
  } else if (params.boundaryType == 2) {
    for (i = 0; i < params.N; i++) {
      len += graph.nodes[i].boundlen;
    }
  }

  calcLogLik(params);
}
class MCMC
{
public:
  DoubleVec ll;
  IntVec Mvals;
  DoubleVec wstarvals;
  IntVec boundlens;
  DoubleVec simErr;
  IntVec type2pix;

  int k; // keeping track of which position we're at

  // current values from most recent MCMC step
  MCMCStepG step;
  MCMC(Partition &, Graph &, ParamsG &);
  void addStep(ParamsG &);
  // void postProcessing(Params &, int);
  void simulateValues(Graph &, Partition &, ParamsG &);
};
MCMC::MCMC(Partition &components, Graph &graph, ParamsG &params)
{
  MCMCStepG step1(components, graph, params);
  step = step1;

  // pvar = 0.0;
  // pmeans(params.N, params.mm);
  // pboundary.assign(params.N, 0);
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

void MCMC::addStep(ParamsG &params)
{
  ll[k] = step.ll;
  Mvals[k] = step.M;
  boundlens[k] = step.len;
  
  double xmax = step.B * params.w0 / step.W / (1 + (step.B * params.w0 / step.W));
  double wstar = 0.0;

  if (step.B > 0) {
    //    Rprintf("xmax:%0.2f, log(W/B):%0.2f\n", xmax, log(step.W) - log(step.B));
    wstar = log(step.W) - log(step.B)
            + Rf_pbeta(xmax, (double) (step.M * params.mm + 3) / 2,
                       (double) ((params.nn2 - step.M) * params.mm - 4) / 2, 1, 1)
            + Rf_lbeta((double) (step.M * params.mm + 3) / 2,
                       (double) ((params.nn2 - step.M) * params.mm - 4) / 2)
            - Rf_pbeta(xmax, (double) (step.M * params.mm + 1) / 2,
                       (double) ((params.nn2 - step.M) * params.mm - 2) / 2, 1, 1)
            - Rf_lbeta((double) (step.M * params.mm + 1) / 2,
                       (double) ((params.nn2 - step.M) * params.mm - 2) / 2);
    wstar = exp(wstar);
  } else {
    wstar = params.w0 * (step.M * params.mm + 1) / (step.M * params.mm + 3);
  }

  wstarvals[k] = wstar;
  
  k++;
}
void MCMCStepG::calcLogLik(ParamsG &params)
{
  double xmax = B * params.w0 / W / (1 + B * params.w0 / W);

  if (B == 0) {
    ll = len * log(params.alpha) + (params.mm + 1) * log(params.w0) / 2
         - (params.mm * params.nn2 - 1) * log(W) / 2;
  } else if (M >= params.N - 4 / params.mm) {
    ll = -DBL_MAX;
  } else {
    ll = len * log(params.alpha)
         + Rf_pbeta(xmax, (double) (params.mm * M + 1) / 2,
                    (double) ((params.nn2 - M) * params.mm - 2) / 2, 1, 1)
         + Rf_lbeta((double) (params.mm * M + 1) / 2,
                    (double) ((params.nn2 - M) * params.mm - 2) / 2)
         - (params.mm * M + 1) * log(B) / 2
         - ((params.nn2 - M) * params.mm - 2) * log(W) / 2;
  }
}

void MCMCStepG::updateLogLik(ParamsG &params, Graph &graph, Partition &partition, Component &newcomp,
                            Component &oldcomp, Node &node, int newCompId)
{
  if (newCompId == node.component) {
    return;
  }

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
        Node neighborNode = graph.nodes[node.neighbors[i]];

        for (j = 0; j < neighborNode.neighbors.size(); j++) {
          if (neighborNode.neighbors[j] == node.id) {
            continue;
          }

          if (graph.nodes[neighborNode.neighbors[j]].component == node.component) {
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

  Zdiff = partition[node.component].Z - newcomp.Z - oldcomp.Z;

  if (newCompId < partition.size()) {
    Zdiff += partition[newCompId].Z;
  }

  B -= Zdiff;
  W += Zdiff;

  if (params.mm == 1 && M == 1) B = 0;

  if (abs(W) < 1.0e-12) {
    W = 1.0e-12;
  }

  calcLogLik(params);
}

int sampleLogLik(vector<MCMCStepG> possibleSteps, double maxll)
{
  int j;
  double myrand = Rf_runif(0.0, 1.0);
  DoubleVec llcum(possibleSteps.size());

  llcum[0] = exp(possibleSteps[0].ll - maxll);

  //  Rprintf("j:0, ll:%0.2f, myrand:%0.2f\n", possibleSteps[0].ll, myrand);
  for (j = 1; j < possibleSteps.size(); j++) {
    llcum[j] = llcum[j - 1] + exp(possibleSteps[j].ll - maxll);
    //    Rprintf("j:%d, ll:%0.2f\n", j, possibleSteps[j].ll);

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

  //  Rprintf("myrand:%0.2f newblock:%d\n",  myrand, endkk);

  return (endkk);
}

void updateComponents(ParamsG &params, MCMC &mcmc, Partition &components, Graph &graph,
                      vector<MCMCStepG> &possibleSteps, vector<Component> &possibleBlocks, int currblock, int newblock,
                      int nodeId, int indexSteps = -1)
{
  int i;
  if (newblock != currblock) {
    if (indexSteps != -1) {
      mcmc.step = possibleSteps[indexSteps]; // this is if possiblesteps is a diff length
    } else {
      mcmc.step = possibleSteps[newblock];
    }

    // if (mcmc.step.ll != mcmc.step.ll) {
    //   Rprintf("ll is nan!\n");
    // }
    components[currblock] = possibleBlocks[currblock];

    if (newblock == components.size()) {
      components.push_back(possibleBlocks[newblock]);
    } else {
      components[newblock] = possibleBlocks[newblock];
    }

    graph.updateNode(nodeId, newblock);

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
}
void fullPixelPass(Graph &graph, Partition &components, ParamsG &params, MCMC &mcmc)
{
  int i, j, maxComp;
  int currblock, newblock;
  double maxll;

  for (i = 0; i < params.N; i++) {
    // Rprintf("i:%d\n", i);
    currblock = graph.nodes[i].component;
    maxComp = components.size() + (components[currblock].size != graph.nodes[i].size);
    vector<Component> possibleBlocks = components; // this vector is not a partition!
    vector<MCMCStepG> possibleSteps(maxComp, mcmc.step);
    maxll = mcmc.step.ll;

    possibleBlocks[currblock].removeNode(graph.nodes[i]); // remove the node from current block

    // loop through all possible components
    for (j = 0; j < maxComp; j++) {
      if (j == components.size()) {
        Component newestBlock(graph.nodes[i]);
        possibleBlocks.push_back(newestBlock);
      } else if (j != currblock) {
        possibleBlocks[j].addNode(graph.nodes[i]);
      }

      possibleSteps[j].updateLogLik(params, graph, components, possibleBlocks[j],
                                    possibleBlocks[currblock], graph.nodes[i], j);

      if (possibleSteps[j].ll > maxll) {
        maxll = possibleSteps[j].ll;
      }
    }

    newblock = sampleLogLik(possibleSteps, maxll);
    updateComponents(params, mcmc, components, graph, possibleSteps, possibleBlocks, currblock,
                     newblock, i);
  }
}

void activePixelPass(Graph &graph, Partition &components, ParamsG &params, MCMC &mcmc)
{
  int i, j, k, index;
  int currblock, newblock, passtype;
  double maxll, u;

  if (params.p1 == 1) {
    passtype = 1;  // correct APP
  } else if (params.p1 == 0) {
    passtype = 2;  // modified APP
  } else {
    u = Rf_runif(0.0, 1.0);

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

    //    Rprintf("node: %d  active: %d\n", i+1, graph.nodes[i].active );
    currblock = graph.nodes[i].component;
    vector<Component> possibleBlocks = components; // this vector is not a partition!
    vector<MCMCStepG> possibleSteps;
    IntegerVector neighbors = graph.nodes[i].neighbors;
    maxll = mcmc.step.ll;
    possibleBlocks[currblock].removeNode(graph.nodes[i]); // remove the node from current block

    IntVec allBlocks; // keep track of the components we are trying out (in the order we tried them)
    index = 0; // the unique count of the block we are on

    if (graph.nodes[i].active == 2) {
      mcmc.type2pix[mcmc.k - 101]++;
    }

    //if (graph.nodes[i].active == 1 || (graph.nodes[i].active == 2 && params.boundaryType == 1)) {
    if (graph.nodes[i].active == 1 || passtype == 2) {
      // if (graph.nodes[i].active == 1) {
      // agree with some neighbors, not all
      IntVec blockTried(possibleBlocks.size(), 0);

      // loop through all possible components
      for (k = 0; k < neighbors.size(); k++) {
        j = graph.nodes[neighbors[k]].component;

        if (blockTried[j] == 1) {
          continue;
        }

        if (j != currblock) {
          possibleBlocks[j].addNode(graph.nodes[i]);
        }

        allBlocks.push_back(j);
        blockTried[j] = 1;

        possibleSteps.push_back(mcmc.step);
        possibleSteps[index].updateLogLik(params, graph, components, possibleBlocks[j],
                                          possibleBlocks[currblock], graph.nodes[i], j);

        if (possibleSteps[index].ll > maxll) {
          maxll = possibleSteps[index].ll;
        }

        index++;
      }
    } else { // active = 2; disagrees with all neighbors
      IntVec blocksToTry(possibleBlocks.size(), 1); // indicators of whether or not we try these blocks

      for (k = 0; k < neighbors.size(); k++) {
        blocksToTry[graph.nodes[neighbors[k]].component] = 0;
        //Rprintf("nb: %d (%d)  ", neighbors[k]+1,  graph.nodes[neighbors[k]].component);
      }

      //Rprintf("\n");
      // loop through all possible components
      for (j = 0; j <= components.size(); j++) {
        if (j == components.size()) { // consider making own block
          if (components[currblock].size != graph.nodes[i].size) {
            Component newestBlock(graph.nodes[i]);
            possibleBlocks.push_back(newestBlock);
            allBlocks.push_back(j);
          } else {
            continue;
          }
        } else if (blocksToTry[j] == 0) {
          continue;
        } else {
          if (j != currblock) {
            possibleBlocks[j].addNode(graph.nodes[i]);
          }

          allBlocks.push_back(j);
        }

        possibleSteps.push_back(mcmc.step);
        possibleSteps[index].updateLogLik(params, graph, components, possibleBlocks[j],
                                          possibleBlocks[currblock], graph.nodes[i], j);

        if (possibleSteps[index].ll > maxll) {
          maxll = possibleSteps[index].ll;
        }

        index++;
      }
    }

    index = sampleLogLik(possibleSteps, maxll);

    newblock = allBlocks[index];
    //    Rprintf("currblock:%d, newblock:%d\n", currblock, newblock);
    updateComponents(params, mcmc, components, graph, possibleSteps, possibleBlocks, currblock,
                     newblock, i, index);
    //    if (graph.nodes[i].active == 2) Rprintf("i: %d  currblock: %d  newblock: %d   ll:%0.4f\n", i, currblock, newblock, mcmc.step.ll);

  }

}

// this computes the change in boundary length given a new block for pixel (i,k);
// returns 2x the length

// [[Rcpp::export]]
SEXP rcpp_ppm(SEXP pdata, SEXP pid, SEXP padj, SEXP pmcmcreturn, SEXP pburnin, SEXP pmcmc, SEXP pa, SEXP pc,
              SEXP pmembs, SEXP pboundaryType, SEXP p1, SEXP pfreqAPP)
{

  NumericMatrix data(pdata);
  int mm = data.ncol();

  List adj(padj);
  NumericVector membInit(pmembs);

  int mcmcreturn = INTEGER_DATA(pmcmcreturn)[0];
  int burnin = INTEGER_DATA(pburnin)[0];
  int itsmcmc = INTEGER_DATA(pmcmc)[0];
  int bType = INTEGER_DATA(pboundaryType)[0];
  int MM = burnin + itsmcmc;
  // some helper variables
  int i, m, j;
  double wstar;
  int freqAPP = INTEGER_DATA(pfreqAPP)[0];
  // initialize my graph
  // Rprintf("Initializing objects\n");
  Graph graph(data, pid, membInit, adj);
  ParamsG params(NUMERIC_DATA(pc)[0], NUMERIC_DATA(pa)[0], graph.nodes.size(), 
                data.nrow(), bType, burnin,
                itsmcmc, NUMERIC_DATA(p1)[0], mm);
  int nn = params.N;
  NumericMatrix pmean(nn, mm);
  NumericMatrix ss(nn, mm);
  NumericMatrix pvar(nn, mm);
  NumericVector pboundary(nn);
  // graph.print();
  // params.print();
  Partition components;
    
  // Rprintf("Initializing graph\n");

  for (i = 0; i < nn; i++) {
    graph.nodes[i].calcActiveAndBound(graph.nodes);

    if ((int) membInit[i] >= components.size()) {
      Component newComp(graph.nodes[i]);
      components.push_back(newComp);
    } else {
      components[(int) membInit[i]].addNode(graph.nodes[i]);
    }
  }

  MCMC mcmc(components, graph, params);
  // printPartition(components);
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
  NumericMatrix results(MM2*nn2,mm);
  double tmpMean;
  
  /* --- END THINGS TO BE RETURNED TO R --- */
  GetRNGstate(); // Consider Dirk's comment on this.

  // Rprintf("Before 100 pixel passes loop\n");
  /* ----------------START THE BIG LOOP--------------------------- */
  // first 100 iterations, do complete pixel passes
  for (m = 0; m < 100; m++) {
    fullPixelPass(graph, components, params, mcmc);
    mcmc.addStep(params);
  }

  // Rprintf("After 100 full pixel passes\n");
  for (m = 0; m < MM; m++) {
    fullPixelPass(graph, components, params, mcmc);
    for (i = 0; i < freqAPP; i++) {
      activePixelPass(graph, components, params, mcmc);
    }

    mcmc.addStep(params);
    // Rprintf("m:%d\n", m);
    // mcmc.step.print();
    // printPartition(components);
    wstar = mcmc.wstarvals[mcmc.k-1];  
    // Rprintf("1037 k:%d wstar:%0.8f\n", mcmc.k-1, wstar);
    if (m >= burnin || mcmcreturn == 1) {
      // for calculating posterior estimate of error variance
      // if (m >= burnin) {
      //   pvar += (mcmc.step.W + mcmc.step.B * wstar) / (params.nn2 * params.mm - 3);
      // }
      for (i = 0; i < params.N; i++) {    
        if (m >= burnin)
          pboundary[i] += (graph.nodes[i].active > 0);
        if (mcmcreturn == 1)            
          membsa(i, m) = graph.nodes[i].component;
        for (j = 0; j < mm; j++) {
          tmpMean = (1 - wstar) * components[graph.nodes[i].component].mean[j]
                               + wstar * graph.mean;
                            
          if (m >= burnin) {
            pmean(i, j) += tmpMean;
            ss(i, j) += tmpMean * tmpMean;
          }
            
          if (mcmcreturn == 1) {
            results(m*params.N+i, j) = tmpMean;
          }            
        }     
      }
    }
    

  }

  // mcmc.step.print();

  // Rprintf("Begin post-processing\n");

  for (i = 0; i < nn; i++) {
    pboundary[i] /= itsmcmc;
    for (j = 0; j < mm; j++) {
      pmean(i, j) /= itsmcmc;
      pvar(i, j) = (ss(i, j)/ itsmcmc - pmean(i,j)*pmean(i,j))*(itsmcmc/(itsmcmc-1));
    }
  }

  // pvar /= itsmcmc;

  PutRNGstate();
  List z;
  z["posterior.mean"] = pmean;
  z["posterior.var"] = pvar;
  // z["ll"] = wrap(mcmc.ll);
  z["posterior.prob"] = pboundary;
  z["mcmc.rhos"] = membsa;
  z["mcmc.means"] = results;
  z["blocks"] = wrap(mcmc.Mvals);
  z["len"] = wrap(mcmc.step.len);
  // z["type2pix"] = wrap(mcmc.type2pix);
  // z["simErr"] = wrap(mcmc.simErr);
  // Rprintf("End printing\n");

  //  closeLogFile();

  return z;
}
/* END MAIN  */

