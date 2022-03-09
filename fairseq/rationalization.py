"""Tools to perform combinatorial rationalization in Fairseq."""
import math
import torch
from fairseq import utils

import pdb


def decode_single_word(model, word_index, source=False):
  if source:
    dictionary = model.task.source_dictionary
  else:
    dictionary = model.task.target_dictionary
  if model.bpe is None:
    word_str = dictionary.symbols[word_index]
    if word_index == 2:
      return "<bos>"
    elif word_index < 2 or word_index == 3:
      return "<pad>"
    else:
      return word_str
  return model.bpe.decode(dictionary.string([word_index]))


def decode_sequence(model, tokens, source=False):
  return [decode_single_word(model, token, source) for token in tokens]


@torch.no_grad()
def rationalize_occupation_model(model,
                                 input_ids,
                                 years,
                                 educations=None,
                                 ethnicities=None,
                                 genders=None,
                                 locations=None,
                                 verbose=False,
                                 max_steps=1024,
                                 start_step=0,
                                 max_tokens_per_batch=4096):
  all_rationales = []
  log = {}
  num_tokens = len(input_ids)

  input_text = decode_sequence(model, input_ids, source=False)
  input_years_text = model.task._year_dictionary.string(years).split(" ")
  log['input_ids'] = list(input_ids.cpu().numpy())
  log['input_text'] = input_text
  log['input_years'] = input_years_text
  log['rationalization'] = []

  if verbose:
    print("All tokens: {}".format(input_text))
    print("All years: {}".format(input_years_text))
  
  all_positions = utils.make_positions(
    input_ids[None], model.model.decoder.padding_idx)
  
  # Perform greedy rationalization for each token in the sequence, starting
  # from `start_step`.
  for prev_token in range(start_step, num_tokens - 1):
    goal_word_text = input_text[prev_token + 1]
    token_log = {}
    token_log['target_position'] = prev_token + 1
    token_log['goal_job'] = goal_word_text
    token_log['log'] = []
  
    # Initialize the rationale. The rationale must always include the most
    # recent token.
    rationale = [prev_token]

    if verbose:
      print("Currently rationalizing token {}: '{}'".format(
        prev_token + 1, goal_word_text))
    
    for rationale_size in range(1, min(max_steps + 1, prev_token + 2)):
      if rationale_size == 1:
        # A rationale of size 1 can only include the most recent target token.
        decoder_out = model.model.decoder(
          prev_output_tokens=input_ids[prev_token:(prev_token + 1)][None],
          years=years[prev_token:(prev_token + 1)][None],
          educations=educations[prev_token:(prev_token + 1)][None],
          ethnicities=ethnicities[None],
          genders=genders[None],
          locations=locations[None],
          position_ids=all_positions[:, prev_token:(prev_token + 1)])
        best_probs = model.model.get_normalized_probs(
          decoder_out, log_probs=True, 
          two_stage=model.model.decoder.args.two_stage, 
          prev_tokens=input_ids[prev_token:(prev_token + 1)][None])[0, -1].exp()
        added_token_text = input_text[prev_token]
        added_token_position = prev_token
        added_year_text = input_years_text[prev_token]
        if verbose:
          added_token_string = ("Adding previous token to sequence: "
                                "'{}' (year: {})".format(added_token_text,
                                                         added_year_text))
      else:
        # Consider the current rationale + each target token
        candidates = [sorted(rationale + [x]) for x in range(prev_token + 1) 
                      if x not in rationale]
        candidate_input_ids = input_ids[[candidates]]
        candidate_years = years[[candidates]]
        candidate_educations = educations[[candidates]]
        candidate_ethnicities = ethnicities[None].repeat([len(candidates), 1])
        candidate_genders = genders[None].repeat([len(candidates), 1])
        candidate_locations = locations[None].repeat([len(candidates), 1])
        candidate_position_ids = all_positions[0, candidates]

        # Divide the candidates into batches, since all possible subsets may
        # not fit in memory if we pass them to the model at once.
        num_candidates, seq_len = candidate_input_ids.shape
        batch_size = math.floor(max_tokens_per_batch / seq_len)
        num_batches = math.ceil(num_candidates / batch_size)
        best_prob = -float("inf")
        for batch_ind in range(num_batches):
          batch_start_ind = batch_ind * batch_size
          batch_end_ind = (batch_ind + 1) * batch_size
          batch_input_ids = candidate_input_ids[batch_start_ind:batch_end_ind]
          batch_years = candidate_years[batch_start_ind:batch_end_ind]
          batch_educations = candidate_educations[
            batch_start_ind:batch_end_ind]
          batch_ethnicities = candidate_ethnicities[
            batch_start_ind:batch_end_ind]
          batch_genders = candidate_genders[batch_start_ind:batch_end_ind]
          batch_locations = candidate_locations[batch_start_ind:batch_end_ind]
          batch_position_ids = candidate_position_ids[
            batch_start_ind:batch_end_ind]
          batch_decoder_out = model.model.decoder(
            batch_input_ids,
            years=batch_years,
            educations=batch_educations,
            ethnicities=batch_ethnicities,
            genders=batch_genders,
            locations=batch_locations,
            position_ids=batch_position_ids)
          batch_probs = model.model.get_normalized_probs(
            batch_decoder_out, log_probs=True,
            two_stage=model.model.decoder.args.two_stage,
            prev_tokens=batch_input_ids).exp()[:, -1]
          true_token_probs = batch_probs[:, input_ids[prev_token + 1]]
          if batch_probs.max() > best_prob:
            best_prob = true_token_probs.max()
            best_token = true_token_probs.argmax() + batch_start_ind
            best_probs = batch_probs[true_token_probs.argmax()]
        
        best_token_position = set(candidates[best_token]) - set(rationale)
        best_token_position = best_token_position.pop()
        rationale.append(best_token_position)
        added_token = input_text[best_token_position]
        added_year_text = input_years_text[best_token_position]
        added_token_string = "Adding token: '{}' (year: {})".format(
          added_token, added_year_text)
        added_token_text = input_text[best_token_position]
        added_token_position = best_token_position
      
      predicted_word_id = best_probs.argmax().item()
      predicted_word_prob = best_probs.max().item()
      predicted_word_text = decode_single_word(
        model, predicted_word_id, source=False)
      top_2_word_id = best_probs.topk(2).indices[1].item()
      top_2_word_prob = best_probs.topk(2).values[1].item()
      top_2_word_text = decode_single_word(model, top_2_word_id, source=False)

      true_token_prob = best_probs[input_ids[prev_token + 1]].item()
      token_log['log'].append({
        "rationale_size": rationale_size,
        "added_token_position": added_token_position,
        "added_token_text": added_token_text,
        "added_year_text": added_year_text,
        "prediction": predicted_word_text,
        "prediction_prob": predicted_word_prob,
        "top_2_prediction": top_2_word_text,
        "top_2_prediction_prob": top_2_word_prob,
        "true_token_prob": true_token_prob,
      })
      if verbose:
        print("{}. This makes the top predicted job: '{}', and the second "
              "most-likely job: '{}'. P('{}') = {:.3f}".format(
                added_token_string, predicted_word_text, top_2_word_text,
                goal_word_text, true_token_prob))
      # Our combinatorial optimization is complete when the predicted token is
      # the true token or the second-most-likely token.
      if (torch.argmax(best_probs) == input_ids[prev_token + 1] or 
          top_2_word_id == input_ids[prev_token + 1]):
        if verbose:
          print("When predicting: '{}'".format(goal_word_text))
          print("  The rationale is: {}".format(
            ', '.join([input_text[x] + " [" + input_years_text[x] + "]" 
            for x in sorted(rationale)])))
          print("Finished with {} tokens.".format(rationale_size))
          print("..........")
        break
    # When we've finished rationalizing, add the rationale to the complete 
    # rationale list.
    all_rationales.append(rationale)
    token_log['rationale'] = rationale
    reached_argmax = (predicted_word_id == input_ids[prev_token + 1] or 
                      top_2_word_id == input_ids[prev_token + 1])
    token_log['reached_argmax'] = reached_argmax.item()
    log['rationalization'].append(token_log)
  
  log['all_rationales'] = all_rationales
  return all_rationales, log
