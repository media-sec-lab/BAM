import os 
import sys
import torch
import numpy as np

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def computer_precision_recall_fscore(pred, label):
    # predict class base pred value
    accuracy = accuracy_score(label,pred)
    precision, recall, fbeta_score, _ = precision_recall_fscore_support(label, pred, average='binary', zero_division=1.0)
    return accuracy, precision, recall, fbeta_score

def compute_acc(scores, label):
    """
        computer acc base model output and label
    :param scores: ndarray, dimension should be (sample_num, 2)
    :param label: ndarray, dimension should be (sanmple_num, 1)
    :return: acc
    """
    pred = scores.argmax(axis=1)
    num_total = len(label)
    num_correct = (pred == label).sum().item()
    acc = 100 * (num_correct / num_total)
    return acc

def compute_det_curve(target_scores, nontarget_scores):

    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    # 注意： 默认设置为预测分数比阈值大，则认为是目标样本，预测分数比阈值小，则认为是非目标样本
    tar_trial_sums = np.cumsum(labels)
    # tar_trial_sums数组的第i个元素的值表示为：选取比第i大的分数作为阈值时，比该分数小的样本数，即也是目标样本被错误分类的数量
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)
    # 同理，nontarget_trial_sums表示选取第i大的分数作为阈值时，比该分数大的样本数，即也是非目标样本被错误分类的数量
    # 注意： 当选取比第i大的分数作为阈值时，这个分数对应的样本也视为比阈值小
    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))  # false rejection rates
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))  # false acceptance rates
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))  # Thresholds are the sorted scores
    # 这里添加拼接的第一部分为了与FRR为0，FAR为1的情况对应
    return frr, far, thresholds

def compute_eer(pred, label):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    if isinstance(pred, torch.Tensor):
        pred, label = pred.cpu().numpy(), label.cpu().numpy()
    pred, label = pred.flatten(), label.flatten()
    target_scores = pred[label==1.]
    nontarget_scores = pred[label==0.]
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)   # 找当FAR和FRR最靠近的情况
    eer = np.mean((frr[min_index], far[min_index]))   # 在FAR和FRR之间使用线性插值求相等的EER
    return eer, thresholds[min_index]

def compute_tDCF(bonafide_score_cm, spoof_score_cm, Pfa_asv, Pmiss_asv, Pfa_spoof_asv, cost_model, print_cost):
    """
    Compute Tandem Detection Cost Function (t-DCF) [1] for a fixed ASV system.
    In brief, t-DCF returns a detection cost of a cascaded system of this form,
      Speech waveform -> [CM] -> [ASV] -> decision
    where CM stands for countermeasure and ASV for automatic speaker
    verification. The CM is therefore used as a 'gate' to decided whether or
    not the input speech sample should be passed onwards to the ASV system.
    Generally, both CM and ASV can do detection errors. Not all those errors
    are necessarily equally cost, and not all types of users are necessarily
    equally likely. The tandem t-DCF gives a principled with to compare
    different spoofing countermeasures under a detection cost function
    framework that takes that information into account.
    INPUTS:
      bonafide_score_cm   A vector of POSITIVE CLASS (bona fide or human)
                          detection scores obtained by executing a spoofing
                          countermeasure (CM) on some positive evaluation trials.
                          trial represents a bona fide case.
      spoof_score_cm      A vector of NEGATIVE CLASS (spoofing attack)
                          detection scores obtained by executing a spoofing
                          CM on some negative evaluation trials.
      Pfa_asv             False alarm (false acceptance) rate of the ASV
                          system that is evaluated in tandem with the CM.
                          Assumed to be in fractions, not percentages.
      Pmiss_asv           Miss (false rejection) rate of the ASV system that
                          is evaluated in tandem with the spoofing CM.
                          Assumed to be in fractions, not percentages.
      Pmiss_spoof_asv     Miss rate of spoof samples of the ASV system that
                          is evaluated in tandem with the spoofing CM. That
                          is, the fraction of spoof samples that were
                          rejected by the ASV system.
      cost_model          A struct that contains the parameters of t-DCF,
                          with the following fields.
                          Ptar        Prior probability of target speaker.
                          Pnon        Prior probability of nontarget speaker (zero-effort impostor)
                          Psoof       Prior probability of spoofing attack.
                          Cmiss       Cost of tandem system falsely rejecting target speaker.
                          Cfa         Cost of tandem system falsely accepting nontarget speaker.
                          Cfa_spoof   Cost of tandem system falsely accepting spoof.
      print_cost          Print a summary of the cost parameters and the
                          implied t-DCF cost function?
    OUTPUTS:
      tDCF_norm           Normalized t-DCF curve across the different CM
                          system operating points; see [2] for more details.
                          Normalized t-DCF > 1 indicates a useless
                          countermeasure (as the tandem system would do
                          better without it). min(tDCF_norm) will be the
                          minimum t-DCF used in ASVspoof 2019 [2].
      CM_thresholds       Vector of same size as tDCF_norm corresponding to
                          the CM threshold (operating point).
    NOTE:
    o     In relative terms, higher detection scores values are assumed to
          indicate stronger support for the bona fide hypothesis.
    o     You should provide real-valued soft scores, NOT hard decisions. The
          recommendation is that the scores are log-likelihood ratios (LLRs)
          from a bonafide-vs-spoof hypothesis based on some statistical model.
          This, however, is NOT required. The scores can have arbitrary range
          and scaling.
    o     Pfa_asv, Pmiss_asv, Pmiss_spoof_asv are in fractions, not percentages.
    References:
      [1] T. Kinnunen, H. Delgado, N. Evans,K.-A. Lee, V. Vestman, 
          A. Nautsch, M. Todisco, X. Wang, M. Sahidullah, J. Yamagishi, 
          and D.-A. Reynolds, "Tandem Assessment of Spoofing Countermeasures
          and Automatic Speaker Verification: Fundamentals," IEEE/ACM Transaction on
          Audio, Speech and Language Processing (TASLP).
      [2] ASVspoof 2019 challenge evaluation plan
          https://www.asvspoof.org/asvspoof2019/asvspoof2019_evaluation_plan.pdf
    """


    # Sanity check of cost parameters
    if cost_model['Cfa'] < 0 or cost_model['Cmiss'] < 0 or \
            cost_model['Cfa'] < 0 or cost_model['Cmiss'] < 0:
        print('WARNING: Usually the cost values should be positive!')

    if cost_model['Ptar'] < 0 or cost_model['Pnon'] < 0 or cost_model['Pspoof'] < 0 or \
            np.abs(cost_model['Ptar'] + cost_model['Pnon'] + cost_model['Pspoof'] - 1) > 1e-10:
        sys.exit('ERROR: Your prior probabilities should be positive and sum up to one.')

    # Unless we evaluate worst-case model, we need to have some spoof tests against asv
    if Pfa_spoof_asv is None:
        sys.exit('ERROR: you should provide false alarm rate of spoof tests against your ASV system.')

    # Sanity check of scores
    combined_scores = np.concatenate((bonafide_score_cm, spoof_score_cm))
    if np.isnan(combined_scores).any() or np.isinf(combined_scores).any():
        sys.exit('ERROR: Your scores contain nan or inf.')

    # Sanity check that inputs are scores and not decisions
    n_uniq = np.unique(combined_scores).size
    if n_uniq < 3:
        sys.exit('ERROR: You should provide soft CM scores - not binary decisions')

    # Obtain miss and false alarm rates of CM
    Pmiss_cm, Pfa_cm, CM_thresholds = compute_det_curve(bonafide_score_cm, spoof_score_cm)

    # Constants - see ASVspoof 2019 evaluation plan

    C0 = cost_model['Ptar'] * cost_model['Cmiss'] * Pmiss_asv + cost_model['Pnon']*cost_model['Cfa']*Pfa_asv
    C1 = cost_model['Ptar'] * cost_model['Cmiss'] - (cost_model['Ptar'] * cost_model['Cmiss'] * Pmiss_asv + cost_model['Pnon'] * cost_model['Cfa'] * Pfa_asv)
    C2 = cost_model['Pspoof'] * cost_model['Cfa_spoof'] * Pfa_spoof_asv;


    # Sanity check of the weights
    if C0 < 0 or C1 < 0 or C2 < 0:
        sys.exit('You should never see this error but I cannot evalute tDCF with negative weights - please check whether your ASV error rates are correctly computed?')

    # Obtain t-DCF curve for all thresholds
    tDCF = C0 + C1 * Pmiss_cm + C2 * Pfa_cm

    # Obtain default t-DCF
    tDCF_default = C0 + np.minimum(C1, C2)

    # Normalized t-DCF
    tDCF_norm = tDCF / tDCF_default

    # Everything should be fine if reaching here.
    if print_cost:

        print('t-DCF evaluation from [Nbona={}, Nspoof={}] trials\n'.format(bonafide_score_cm.size, spoof_score_cm.size))
        print('t-DCF MODEL')
        print('   Ptar         = {:8.5f} (Prior probability of target user)'.format(cost_model['Ptar']))
        print('   Pnon         = {:8.5f} (Prior probability of nontarget user)'.format(cost_model['Pnon']))
        print('   Pspoof       = {:8.5f} (Prior probability of spoofing attack)'.format(cost_model['Pspoof']))
        print('   Cfa          = {:8.5f} (Cost of tandem system falsely accepting a nontarget)'.format(cost_model['Cfa']))
        print('   Cmiss        = {:8.5f} (Cost of tandem system falsely rejecting target speaker)'.format(cost_model['Cmiss']))
        print('   Cfa_spoof    = {:8.5f} (Cost of tandem sysmte falsely accepting spoof)'.format(cost_model['Cfa_spoof']))
        print('\n   Implied normalized t-DCF function (depends on t-DCF parameters and ASV errors), t_CM=CM threshold)')
        print('   tDCF_norm(t_CM) = {:8.5f} + {:8.5f} x Pmiss_cm(t_CM) + {:8.5f} x Pfa_cm(t_CM)\n'.format(C0/tDCF_default, C1/tDCF_default, C2/tDCF_default))
        print('     * The optimum value is given by the first term (0.06273). This is the normalized t-DCF obtained with an error-free CM system.')
        print('     * The minimum normalized cost (minimum over all possible thresholds) is always <= 1.00.')
        print('')

    return tDCF_norm, CM_thresholds


if __name__ == '__main__':
    label = np.array([0,0,0,0,0,1,1,1,1,1])
    pred = np.array([1,0,0,0,0,0,0,1,1,1])
    precision, recall, fbeta_score, _ = precision_recall_fscore_support(label, pred, average='binary', zero_division=1.0)
    print('define of metric compute.')