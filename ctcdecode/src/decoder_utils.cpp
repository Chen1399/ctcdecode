#include "decoder_utils.h"

#include <algorithm>
#include <cmath>
#include <limits>

std::vector<std::pair<size_t, float>> get_pruned_log_probs(
    const std::vector<double> &prob_step,
    double cutoff_prob,
    size_t cutoff_top_n,
    int log_input) {
  size_t cutoff_len = prob_step.size();
  double log_cutoff_prob = log(cutoff_prob);

  std::vector<std::pair<int, double>> get_prob_idx;
  if(log_cutoff_prob < 0.0 || cutoff_top_n < cutoff_len){
    std::vector<std::pair<int, double>> prob_idx_sort;
    double max_half = std::log(0.5);
    double max_pre = 1;
    double max_sum = 0.0;
    int sort_id_set[cutoff_top_n];
    for(int i = 0; i < cutoff_top_n; i++){
      sort_id_set[i] = -1;
    }
    for(int i = 0; i < cutoff_top_n; i++){
      double max = -NUM_FLT_INF;
      int max_id = 0;
      for(int j = 0; j < cutoff_len; j++){
        if(prob_step[j] > max && prob_step[j] <= max_pre){
          if(prob_step[j] == max_pre){
            bool isPre = false;
            for(int k = 0; k < i; k++){
              if(sort_id_set[k] == j){
                isPre = true;
                break;
              }
            }
            if(isPre) continue;
          }
          max = prob_step[j];
          max_id = j;
          if(max >= max_half) break;
        } 
      }
      std::pair<int,double> sortMax;
      sortMax.first = max_id;
      sortMax.second = prob_step[max_id];
      prob_idx_sort.push_back(sortMax);
      max_sum += std::exp(max);
      max_half = std::log((1 - max_sum) / 2);
      max_pre = max;
      sort_id_set[i] = max_id;
    }


    if (log_cutoff_prob < 0.0){
      double cum_prob = log_input ? prob_idx_sort[0].second : log(prob_idx_sort[0].second);
      cutoff_len = 1;
      for (size_t i = 1; i < prob_idx_sort.size(); ++i) {
        if (cum_prob >= log_cutoff_prob) break;
        cum_prob = log_sum_exp(cum_prob, log_input ? prob_idx_sort[i].second : log(prob_idx_sort[i].second));
        cutoff_len += 1;
      }
    }else{
      cutoff_len = cutoff_top_n;
    }
     get_prob_idx = std::vector<std::pair<int, double>>(
        prob_idx_sort.begin(), prob_idx_sort.begin() + cutoff_len);
  }
  std::vector<std::pair<size_t, float>> log_prob_idx;
  for (size_t i = 0; i < cutoff_len; ++i) {
    log_prob_idx.push_back(std::pair<int, float>(
        get_prob_idx[i].first, log_input ? (float)(get_prob_idx[i].second) : (float)(log(get_prob_idx[i].second + NUM_FLT_MIN))));
  }
  return log_prob_idx;
}

std::vector<std::pair<double, Output>> get_beam_search_result(
    const std::vector<PathTrie *> &prefixes,
    size_t beam_size) {
  // allow for the post processing
  std::vector<PathTrie *> space_prefixes;
  if (space_prefixes.empty()) {
    for (size_t i = 0; i < beam_size && i < prefixes.size(); ++i) {
      space_prefixes.push_back(prefixes[i]);
    }
  }

  std::sort(space_prefixes.begin(), space_prefixes.end(), prefix_compare);
  std::vector<std::pair<double, Output>> output_vecs;
  for (size_t i = 0; i < beam_size && i < space_prefixes.size(); ++i) {
    std::vector<int> output;
    std::vector<int> timesteps;
    space_prefixes[i]->get_path_vec(output, timesteps);
    Output outputs;
    outputs.tokens = output;
    outputs.timesteps = timesteps;
    std::pair<double, Output> output_pair(-space_prefixes[i]->approx_ctc,
                                               outputs);
    output_vecs.emplace_back(output_pair);
  }

  return output_vecs;
}

size_t get_utf8_str_len(const std::string &str) {
  size_t str_len = 0;
  for (char c : str) {
    str_len += ((c & 0xc0) != 0x80);
  }
  return str_len;
}

std::vector<std::string> split_utf8_str(const std::string &str) {
  std::vector<std::string> result;
  std::string out_str;

  for (char c : str) {
    if ((c & 0xc0) != 0x80)  // new UTF-8 character
    {
      if (!out_str.empty()) {
        result.push_back(out_str);
        out_str.clear();
      }
    }

    out_str.append(1, c);
  }
  result.push_back(out_str);
  return result;
}

std::vector<std::string> split_str(const std::string &s,
                                   const std::string &delim) {
  std::vector<std::string> result;
  std::size_t start = 0, delim_len = delim.size();
  while (true) {
    std::size_t end = s.find(delim, start);
    if (end == std::string::npos) {
      if (start < s.size()) {
        result.push_back(s.substr(start));
      }
      break;
    }
    if (end > start) {
      result.push_back(s.substr(start, end - start));
    }
    start = end + delim_len;
  }
  return result;
}

bool prefix_compare(const PathTrie *x, const PathTrie *y) {
  if (x->score == y->score) {
    if (x->character == y->character) {
      return false;
    } else {
      return (x->character < y->character);
    }
  } else {
    return x->score > y->score;
  }
}

void add_word_to_fst(const std::vector<int> &word,
                     fst::StdVectorFst *dictionary) {
  if (dictionary->NumStates() == 0) {
    fst::StdVectorFst::StateId start = dictionary->AddState();
    assert(start == 0);
    dictionary->SetStart(start);
  }
  fst::StdVectorFst::StateId src = dictionary->Start();
  fst::StdVectorFst::StateId dst;
  for (auto c : word) {
    dst = dictionary->AddState();
    dictionary->AddArc(src, fst::StdArc(c, c, 0, dst));
    src = dst;
  }
  dictionary->SetFinal(dst, fst::StdArc::Weight::One());
}

bool add_word_to_dictionary(
    const std::string &word,
    const std::unordered_map<std::string, int> &char_map,
    bool add_space,
    int SPACE_ID,
    fst::StdVectorFst *dictionary) {
  auto characters = split_utf8_str(word);

  std::vector<int> int_word;

  for (auto &c : characters) {
    if (c == " ") {
      int_word.push_back(SPACE_ID);
    } else {
      auto int_c = char_map.find(c);
      if (int_c != char_map.end()) {
        int_word.push_back(int_c->second);
      } else {
        return false;  // return without adding
      }
    }
  }

  if (add_space) {
    int_word.push_back(SPACE_ID);
  }

  add_word_to_fst(int_word, dictionary);
  return true;  // return with successful adding
}
