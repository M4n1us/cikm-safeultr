
import os, json, pdb
import pandas as pd
import numpy as np
import torch


class PlackettLuceModel:
    '''
    Plackett-Luce distribution
    '''
    
    def __init__(self, n_samples):
        '''
        Inputs:
            n_samples: #sampled rankings to generate per query
        '''
        self.n_samples = n_samples
        self.eps = 1e-08
    
    def __reverse_logcumsumexp(self, ranking_scores, mask, k):
        '''
        Custom implementation of logcumsumexp operation on ranking_scores with masking. 
        Returns document placement probability per rank, and the log-scores(for REINFORCE training)
        '''

        masked_scores = ranking_scores.masked_fill(~mask, float('-inf'))

        b_size, n_samp, n_docs = masked_scores.shape
        log_score = torch.zeros(b_size, n_samp, device=masked_scores.device)
        doc_prob_per_rank = torch.zeros_like(masked_scores)

        for i in range(k):
            s_i = masked_scores[:, :, i]

            denom_i = torch.logsumexp(masked_scores[:, :, i:], dim=-1)
            log_score_i = s_i - denom_i
            log_score += log_score_i

            doc_prob_per_rank[:, :, i] = torch.exp(log_score_i)
        return log_score, doc_prob_per_rank
        """
        # set log-scores to -INF at padded position
        #ranking_scores[~mask] = 0
        # log-normalizing factor for doc placement prob
        # need to sum from position-i to last position in the ranking. Cumsum gives score from pos-0 till pos-i, hence using flip
        #log_norm = torch.flip(torch.log(torch.cumsum(torch.exp(torch.flip(ranking_scores, [2])) + self.eps, -1)), [2])
        log_norm = torch.flip(torch.logcumsumexp(torch.flip(ranking_scores, [2]), -1), [2])
        # exponentiating log_norm to get the 'true' normalizing factor
        #norm = torch.exp(log_norm)
        # get placement prob. from log-scores
        #place_prob = torch.exp(ranking_scores)
        normalized_scores = torch.exp(ranking_scores - log_norm)
        # cummulative product to get the actual plackett-luce placement prob, which is product of terms till rank k, for all k. 
        # TO-DO: How to make them numerically stable for all kinds of inputs? 
        #doc_prob_per_rank = torch.exp(ranking_scores - log_norm)
        doc_prob_per_rank = torch.cumprod(normalized_scores, dim=-1)
        # resetting mask values to zero, to avoid nan in log-score computation
        #ranking_scores[~mask] = 0.
        # get-log-score from top-k elements
        log_score = torch.sum(ranking_scores[:, :, :k], dim=-1) - torch.sum(log_norm[:, :, :k], dim=-1)
        return log_score, doc_prob_per_rank
        """

    
    def prob_per_rank(self, ranking_scores, mask=None):
        '''
        Get placement prob. of document per rank.
        '''
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).expand_as(ranking_scores)
            ranking_scores = ranking_scores.masked_fill(~mask_expanded, float('-inf'))
        b_size, n_samp, n_docs = ranking_scores.shape
        doc_prob = torch.zeros_like(ranking_scores)

        for i in range(n_docs):
            denom_i = torch.logsumexp(ranking_scores[:, :, i:], dim=-1, keepdim=True)
            doc_prob[:, :, i] = torch.exp(ranking_scores[:, :, i] - denom_i)

        return doc_prob
        """
        log_norm = torch.flip(torch.logcumsumexp(torch.flip(ranking_scores, [2]), -1), [2])
        # exponentiating log_norm to get the 'true' normalizing factor
        #norm = torch.exp(log_norm)
        # get placement prob. from log-scores
        #place_prob = torch.exp(ranking_scores)
        normalized_scores = torch.exp(ranking_scores - log_norm)
        # cummulative product to get the actual plackett-luce placement prob, which is product of terms till rank k, for all k. 
        # TO-DO: How to make them numerically stable for all kinds of inputs? 
        #doc_prob_per_rank = torch.exp(ranking_scores - log_norm)
        doc_prob_per_rank = normalized_scores
        return doc_prob_per_rank
        """

        
    def sample(self, logits, mask, T=1):
        '''
        Input:
            logits: shape= #queriesInBatch * max_rank_length (with scores per document)
            Generate sampled rankings for the mini-batch of queries with logit scores per doc/item.
            mask: Mask per query. For each query, indicates the padding used 
        '''
        # Sampling via Gumbel distribution. 
        # Step1: Sample i.i.d. gumbel noise
        # Step2: Add the gumbel noise to the scores
        # Sort based on the noisy scores, take top-K. Results in a sampling from the PL model
        # T = temperature. T < 1 makes the sampling more deterministic, T > 1 makes it more uniform. 
        logits = logits/T
        self.size = logits.shape
        batch_size, n_docs = logits.shape
        logits_expanded = logits.unsqueeze(1).expand(batch_size, self.n_samples, n_docs)
        unif = torch.rand_like(logits_expanded)
        unif = torch.clamp(unif, min=1e-8, max=1 - 1e-8) 
        gumbel_noise = -torch.log(-torch.log(unif + self.eps))

        gumbel_scores = logits_expanded + gumbel_noise
        gumbel_scores = torch.clamp(gumbel_scores, -1e6, 1e6)

        mask_expanded = mask.unsqueeze(1).expand_as(gumbel_scores)
        gumbel_scores = gumbel_scores.masked_fill(~mask_expanded, float('-inf'))

        #gumbel_scores = gumbel_scores + (~mask.unsqueeze(1)) * -torch.tensor(1000000)
        ranking_scores, sampled_rankings = torch.sort(gumbel_scores, descending=True, dim=-1, stable=True)
        return (ranking_scores, sampled_rankings)

    def log_scores(self, ranking_scores, mask, k):
        '''
        Output: 
            computes log-score given samples    
            output size: #queriesInBatch
        '''
        mask_expanded = mask.unsqueeze(1).expand_as(ranking_scores)
        #mask1 = mask.unsqueeze(1).expand(mask.shape[0], self.n_samples, mask.shape[1])
        log_score, doc_prob_per_rank = self.__reverse_logcumsumexp(ranking_scores, mask_expanded, k)
        return log_score, doc_prob_per_rank        
