#ifndef BEST_CUT
#define BEST_CUT

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>

double *array;

int compare(const void *a, const void *b) {
  int ia = *(int *) a;
  int ib = *(int *) b;
  int ans = 0;
  if (array[ia] < array[ib]) {
    ans = -1;
  }
  else if (array[ia] > array[ib]) {
    ans = 1;
  }
  return ans;
}

bool check_ratio(int a, int b, double r) {
  bool ans = true;
  if (a > b) {
    ans = (a <= r*b);
  }
  else if (b > a) {
    ans = (b <= r*a);
  }
  return ans;
}

int *order_data(double *data, int n) {
  int i;
  int *ans = (int *)malloc(sizeof(int) * n);
  for (i = 0; i < n; i++) {
    ans[i] = i;
  }
  array = data;
  qsort(ans, n, sizeof(*ans), compare);
  return ans;
}

double get_cost_imbalance(double cost, double a, double b,
                          int centers_left, int centers_right) {
  double imbalance = (centers_left > centers_right) ?
                     (double) centers_left / centers_right :
                     (double) centers_right / centers_left;
  double factor = a * imbalance + b;
  return cost * factor;
}

void best_cut_single_dim(double *data, int *data_count, double *centers,
                         double *distances, int *dist_order, int n, int k,
                         double *r_p, bool check_imbalance,
                         double imbalance_factor, double *ans) {
  double r = r_p[0];
  int i, j, c, cur_c, ix, ic;
  int idx_data = 0;
  int idx_centers = 0;
  double nxt_cut;
  double best_cut;
  double cur_cost = 0;
  double best_cost;
  double best_cost_imbalance;
  double cur_cost_imbalance;
  double a = (k == 2) ? 0 : (imbalance_factor - 1) / (k - 2);
  double b = (k == 2) ? 1 : 1 - a;
  double old_data_cost;
  double max_cut;
  int max_center;
  int *left_data_mask = (int *)malloc(sizeof(int) * n);
  int *left_centers_mask = (int *)malloc(sizeof(int) * k);
  int *best_in_left = (int *)malloc(sizeof(int) * n);
  int *best_in_right = (int *)malloc(sizeof(int) * n);
  double *cur_costs = (double *)malloc(sizeof(double) * n);
  int *cur_centers = (int *)malloc(sizeof(int) * n);
  int *data_order = (int *)malloc(sizeof(int) * n);
  int *centers_order = (int *)malloc(sizeof(int) * k);

  // data order array
  for (i = 0; i < n; i++) {
    data_order[i] = i;
  }
  array = data;
  qsort(data_order, n, sizeof(*data_order), compare);

  // centers order array
  for (i = 0; i < k; i++) {
    centers_order[i] = i;
  }
  array = centers;
  qsort(centers_order, k, sizeof(*centers_order), compare);

  // next cut is the smallest cut that respects the ratio
  // max cut is largest value of center that respects the ratio
  ic = 0;
  while (!check_ratio(ic, k - ic, r)) {
    ic++;
    if (ic == k) {
      ic = 1;
      break;
    }
  }
  ic--;
  c = centers_order[ic];
  nxt_cut = centers[c];
  ic++;
  while (check_ratio(ic, k - ic, r)) {
    ic++;
    if (ic == k) {
      break;
    }
  }
  ic--;
  max_center = ic;
  c = centers_order[ic];
  max_cut = centers[c];
  ic++;

  // current costs are the best costs (no cuts yet)
  for (i = 0; i < n; i++) {
    c = dist_order[i * k];
    cur_centers[i] = c;
    cur_costs[i] = distances[i*k + c] * data_count[i];
    cur_cost += cur_costs[i];
  }

  // consider a single center is in the left
  c = centers_order[0];

  // vector to keep track of best center to the left
  // initially, the best center to the left is the only one there
  for (i = 0; i < n; i++) {
    best_in_left[i] = c;
  }

  // vector to keep track of index of centers that
  // have been tested to the right
  // initially, the best center to the right is the first one
  for (i = 0; i < n; i++) {
    best_in_right[i] = 0;
  }

  // set indices of data and center as first index in
  // which the element is to the right of the cut
  ix = data_order[idx_data];
  while ( (data[ix] <= nxt_cut) && idx_data < n) {
    idx_data++;
    if (idx_data < n) {
      ix = data_order[idx_data];
    }
  }

  while ( (centers[c] <= nxt_cut) && (idx_centers < k) ) {
    idx_centers++;
    if (idx_centers < k) {
      c = centers_order[idx_centers];
      // if a center is moved to the left during initialization,
      // check if it's the best center to the left for each datum
      if (centers[c] <= nxt_cut) {
        for (i = 0; i < n; i++) {
          cur_c = best_in_left[i];
          if (distances[i*k + cur_c] > distances[i*k + c]) {
            best_in_left[i] = c;
          }
        }
      }
    }
  }

  // if all centers moved to the left after the
  // first cut, return (no cut is possible)
  if (idx_centers == k) {
    ans[0] = -1;
    ans[1] = INFINITY;
    ans[2] = -1;
    ans[3] = -1;
    free(cur_costs);
    free(left_data_mask);
    free(left_centers_mask);
    free(best_in_left);
    free(best_in_right);
    free(cur_centers);
    free(data_order);
    free(centers_order);
    return;
  }

  // define masks
  for (i = 0; i < n; i++) {
    left_data_mask[i] = data[i] <= nxt_cut;
  }
  for (i = 0; i < k; i++) {
    left_centers_mask[i] = centers[i] <= nxt_cut;
  }

  // reassign data separated from their centers
  for (i = 0; i < n; i++) {
    cur_c = cur_centers[i];
    // if datum is to the left, best center is in best_in_left vector
    if (left_data_mask[i]) {
      c = best_in_left[i];
    }
    // if datum is to the right, find center to the right closest to datum
    // (going from the best_in_right vector)
    else {
      j = best_in_right[i];
      c = dist_order[i*k + j];
      while (left_centers_mask[c]) {
        j++;
        c = dist_order[i*k + j];
      }
      best_in_right[i] = j;
    }
    // if best center is different than current center, update cost
    if (c != cur_c) {
      old_data_cost = cur_costs[i];
      cur_costs[i] = distances[i*k + c] * data_count[i];
      cur_cost += (cur_costs[i] - old_data_cost);
      cur_centers[i] = c;
    }
  }

  // store initial cut and cost as best ones (if they're possible)
  if ((idx_centers != 0) && (idx_centers != k) &&
      (idx_data >= idx_centers) && ( (n - idx_data) >= (k - idx_centers) ) ) {
       best_cut = nxt_cut;
       best_cost = cur_cost;
       best_cost_imbalance = check_imbalance ?
                             get_cost_imbalance(cur_cost, a, b, idx_centers,
                               k - idx_centers) :
                             best_cost;
  }
  else {
    best_cut = -1;
    best_cost = INFINITY;
    best_cost_imbalance = INFINITY;
  }

  while ( (idx_data < n) && (idx_centers < k) ) {
    // find next cut and check if feasible
    ix = data_order[idx_data];
    ic = centers_order[idx_centers];
    if (data[ix] < centers[ic]) {
      nxt_cut = data[ix];
    }
    else {
      nxt_cut = centers[ic];
    }
    if (nxt_cut >= max_cut) {
      break;
    }

    // move data points to the left and assign them to best center to the left
    while ( (idx_data < n) && (data[ix] <= nxt_cut) ) {
      old_data_cost = cur_costs[ix];
      left_data_mask[ix] = 1;
      // find best center to the left via best_in_left vector
      c = best_in_left[ix];
      cur_centers[ix] = c;
      cur_costs[ix] = distances[ix*k + c] * data_count[ix];
      cur_cost += (cur_costs[ix] - old_data_cost);
      idx_data++;
      if (idx_data < n) {
          ix = data_order[idx_data];
      }
    }

    // move centers to the left and reassign data points
    while ( (idx_centers < k) && (centers[ic] <= nxt_cut) ) {
      left_centers_mask[ic] = 1;
      for (i = 0; i < n; i++) {
        old_data_cost = cur_costs[i];
        // update best_in_left vector
        cur_c = best_in_left[i];
        if (distances[i*k + ic] < distances[i*k + cur_c]) {
          best_in_left[i] = ic;
        }
        // if datum is to the left, assign it to center that moved to the left
        // if it's best in left
        if (left_data_mask[i]) {
          if (best_in_left[i] == ic) {
            cur_centers[i] = ic;
            cur_costs[i] = distances[i*k + ic] * data_count[i];
            cur_cost += (cur_costs[i] - old_data_cost);
          }
        }
        // if datum is to the right and its current center moved left, find
        // new best center (starting search from index in best_in_right)
        else if (cur_centers[i] == ic) {
          j = best_in_right[i];
          c = dist_order[i*k];
          while (left_centers_mask[c]) {
            j++;
            c = dist_order[i*k + j];
          }
          best_in_right[i] = j;
          cur_costs[i] = distances[i*k + c] * data_count[i];
          cur_cost += (cur_costs[i] - old_data_cost);
          cur_centers[i] = c;
        }
      }
      idx_centers++;

      if (idx_centers < k) {
        ic = centers_order[idx_centers];
      }
    }

    // check if cut is acceptable: must have at least as many centers
    // as data points on each side
    if ((idx_centers != 0) && (idx_centers != k)) {
        cur_cost_imbalance = check_imbalance ?
                             get_cost_imbalance(cur_cost, a, b, idx_centers,
                               k - idx_centers) :
                             cur_cost;
    }
    if ((idx_centers != 0) && (idx_centers != k) &&
        (idx_data >= idx_centers) && ( (n - idx_data) >= (k - idx_centers) ) &&
        (cur_cost_imbalance < best_cost_imbalance) ) {
         best_cut = nxt_cut;
         best_cost = cur_cost;
         best_cost_imbalance = cur_cost_imbalance;
    }
  }

  // if idx_centers > max_centers, return that no cut is possible
  // (chosen cut did not respect the separation ratio)
  if ( (max_center) && (idx_centers > max_center) ) {
    ans[0] = -1;
    ans[1] = INFINITY;
    ans[2] = -1;
    ans[3] = -1;
    free(cur_costs);
    free(left_data_mask);
    free(left_centers_mask);
    free(best_in_left);
    free(best_in_right);
    free(cur_centers);
    free(data_order);
    free(centers_order);
    return;
  }
  // return best cut and cost
  ans[0] = best_cut;
  ans[1] = best_cost;
  ans[2] = (double) idx_centers;
  ans[3] = (double) idx_data;

  free(cur_costs);
  free(left_data_mask);
  free(left_centers_mask);
  free(best_in_left);
  free(best_in_right);
  free(cur_centers);
  free(data_order);
  free(centers_order);
  return;
};

#endif
