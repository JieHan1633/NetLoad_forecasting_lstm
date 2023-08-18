import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

def get_loss_module(config):

    task = config['task']

    if (task == "imputation") or (task == "transduction"):
        print("============loss function is MaskedMSELoss")
        return MaskedMSELoss(reduction='none')  # outputs loss for each batch element

    if task == "classification":
        return NoFussCrossEntropyLoss(reduction='none')  # outputs loss for each batch sample

    if task == "regression": 
        if config['finetuneloss'] == 'pinball_regloss_CityDayTime':
            return pinball_regloss_CityDayTime(config['city'])
        if config['finetuneloss'] == 'pinball_regloss':
            return pinball_regloss(config['reg_weight'],config['strategy'])
        if config['finetuneloss'] == 'pinball_loss':
            return pinball_loss(config['strategy'])
        if config['finetuneloss'] == 'mse':
            return nn.MSELoss(reduction='none')  # outputs loss for each batch sample
        
    else:
        raise ValueError("Loss module for task '{}' does not exist".format(task))


def l2_reg_loss(model):
    """Returns the squared L2 norm of output layer of given model"""

    for name, param in model.named_parameters():
        if name == 'output_layer.weight':
            return torch.sum(torch.square(param))

class pinball_regloss_CityDayTime(nn.Module):
    def __init__(self, city):
        super().__init__() 
        self.city = city
    def forward(self, y_pred,y_true):
        
        # The function then computes a weighted sum of the MSE losses and the pinball loss. 
        # The weights are provided by the weights parameter.
    
        # The function can be used with either a single sample or multiple samples
        # One sample:  
        # y_true has shape (n_output,), such as (24,)
        # y_pred has shape (n_output, n_quantiles), such as (24, 11)
        # tau has shape (1, n_quantiles,), such as (1, 11), and is a vector of quantiles
    
        # example: tau = np.linspace(0, 1, 11)[None, :] # Shape tau to match y_pred dimensions
        
        # Multiple samples:
        # y_true has shape (n_output, n_samples), such as (24, 100)
        # y_pred has shape (n_output, n_quantiles, n_samples), such as (24, 11, 100)
        # tau has shape (1, n_quantiles, 1), such as (1, 11, 1), and is a vector of quantiles
        # example: tau = np.linspace(0, 1, 11)[None, :, None] # Reshape tau to match y_pred dimensions
        
        # Extract the daytime rows for the given city
        # Calculate pinball loss
        # Donalsonville: 6 - 21  20220101T10:00:00+0000 - 20220102T01:00:00+0000 
        # San Antonio: 6 - 21  20220101T11:00:00+0000 - 20220102T02:00:00+0000 
        # Amity: 5 - 22    2022-01-01 12:00:00+00:00 - 2022-01-02 05:00:00+00:00
        # Waianae: 5 - 20   20220101T15:00:00+0000 - 20220102T06:00:00+0000 
        if self.city == 'Donalsonville':
            y_pred = y_pred[:,7:21,:]
            y_true = y_true[:,7:21]
        elif self.city == 'SanAntonio':
            y_pred = y_pred[:,7:21, :]
            y_true = y_true[:,7:21]
        elif self.city == 'Amity':
            y_pred = y_pred[:,5:21, :]
            y_true = y_true[:,5:21]
        elif self.city == 'Waianae':
            y_pred = y_pred[:,6:19, :]
            y_true = y_true[:,6:19]
        else:
            print(f"Invalid city: {self.city}")
            y_pred = None
    
        # If y_true < y_pred, I = 1, else I = 0.   
        I = torch.where(y_true[:, :, None]<y_pred, 1, 0) 
        # I = np.less(y_true[:, None], y_pred).astype(int)
        # L(τ, y, ŷ) = (ŷ - y)(I(y ≤ ŷ) - τ)  
        #              (ŷ - y) * -τ for y > ŷ
        #              (ŷ - y) * (1 - τ) for y ≤ ŷ  
        # L(τ, y, ŷ) = (y - ŷ)(τ - I(y ≤ ŷ))  
        #              (y - ŷ) * τ for y > ŷ
        #              (y - ŷ) * (τ - 1) for y ≤ ŷ 
        weights = torch.Tensor(np.array([0.1]*11 + [8.0])).to(device)
        tau = torch.Tensor(np.reshape(np.arange(0,1.1,0.1),(1,11))).to(device)
        pinball_mean =  torch.mean(tau * (1 - I) * (y_true[:, :,None]-y_pred) + (tau - 1) * I * (y_true[:,:, None] - y_pred))
        # pinball_loss =  np.mean(tau * (1 - I) * (y_true[:, None]-y_pred) + (tau - 1) * I * (y_true[:, None] - y_pred))
        
        # pinball_sum = np.sum(tau * (1 - I) * (y_true[:, None]-y_pred) + (tau - 1) * I * (y_true[:, None] - y_pred))
        
        # MSE for each percentile prediction
        mse_loss = torch.mean((y_pred - y_true[:, :, None])**2, axis=[0,1]) 
        # if y_pred.ndim == 2: 
        #     mse_loss = np.mean((y_pred - y_true[:, None])**2, axis=0)
        # elif y_pred.ndim == 3:
        #     mse_loss = np.mean((y_pred - y_true[:, None])**2, axis=(0,2))
        
        # Compute the weighted sum of all losses
        # weights in the same order as the quantiles for predictions. 
        # The last weight in the weights list is applied to the pinball loss.
        pinball_regloss = torch.dot(weights[0:-1], mse_loss) + weights[-1] * pinball_mean
        
        return pinball_regloss, pinball_mean

class pinball_regloss(nn.Module):
    def __init__(self,reg_weight,strategy=None):
        super().__init__() 
        self.reg_weight = reg_weight
        self.strategy = strategy
    def forward(self, y_pred,y_true):
        I = torch.where(y_true[:, :, None]<y_pred, 1, 0) 
        # tau = torch.Tensor(np.reshape(np.arange(0,1.1,0.1),(1,11))).to(device)
        
        if self.strategy=="narrow":
            tau = torch.Tensor(np.reshape(np.array([0.09]+list(np.arange(0.1,1,0.1))+[0.91]),
                                          (1,11))).to(device)
        if self.strategy=="wide":
            tau = torch.Tensor(np.reshape(np.array([0.01]+list(np.arange(0.1,1,0.1))+[0.99]),
                                          (1,11))).to(device)
        if self.strategy=="none":
            tau = torch.Tensor(np.reshape(np.arange(0,1.1,0.1), (1,11))).to(device)
        
        pinball_sum = torch.sum(tau * (1 - I) * (y_true[:, :,None] - y_pred) +\
                                (tau - 1) * I * (y_true[:, :,None] - y_pred))
        pinball_mean =  torch.mean(tau * (1 - I) * (y_true[:, :,None] - y_pred) +\
                                   (tau - 1) * I * (y_true[:,:, None] - y_pred)) 
        weights = torch.Tensor(np.array([0.1]*11 + [self.reg_weight])).to(device) 
        # mse should be [11]
        mse_loss = torch.mean((y_pred - y_true[:, :, None])**2, axis=[0,1]) 
        
        pinball_loss = torch.dot(weights[0:-1], mse_loss) + weights[-1] * pinball_mean
        # print("mse_loss shape", mse_loss.shape)
        # print("weights shape", weights.shape) 
        return pinball_loss, pinball_sum
     
class pinball_loss(nn.Module):
    def __init__(self,strategy=None):
        super().__init__() 
        self.strategy = strategy
    def forward(self, y_pred,y_true): 
        # The function can be used with either a single sample or multiple samples 
        # One sample:  
        # y_true has shape (n_output,), such as (24,)
        # y_pred has shape (n_output, n_quantiles), such as (24, 11)
        # tau has shape (1, n_quantiles,), such as (1, 11), and is a vector of quantiles
        # example: tau = np.linspace(0, 1, 11)[None, :] # Shape tau to match y_pred dimensions
        
        # Multiple samples:
        # y_true has shape (n_output, n_samples), such as (24, 100)
        # y_pred has shape (n_output, n_quantiles, n_samples), such as (24, 11, 100)
        # tau has shape (1, n_quantiles, 1), such as (1, 11, 1), and is a vector of quantiles
        # example: tau = np.linspace(0, 1, 11)[None, :, None] # Reshape tau to match y_pred dimensions
        
        # If y_true < y_pred, I = 1, else I = 0.   
        I = torch.where(y_true[:, :, None]<y_pred, 1, 0)
        # print("target shape:",y_true.shape)
        # I = np.less(y_true[:, None], y_pred).astype(int)
        # L(τ, y, ŷ) = (ŷ - y)(I(y ≤ ŷ) - τ)  
        #              (ŷ - y) * -τ for y > ŷ
        #              (ŷ - y) * (1 - τ) for y ≤ ŷ  
        # L(τ, y, ŷ) = (y - ŷ)(τ - I(y ≤ ŷ))  
        #              (y - ŷ) * τ for y > ŷ
        #              (y - ŷ) * (τ - 1) for y ≤ ŷ 
        # pinball_sum = np.sum(tau * (1 - I) * (y_true[:, None]-y_pred) + (tau - 1) * I * (y_true[:, None] - y_pred))
        # pinball_mean =  np.mean(tau * (1 - I) * (y_true[:, None]-y_pred) + (tau - 1) * I * (y_true[:, None] - y_pred))
        if self.strategy=="narrow":
            tau = torch.Tensor(np.reshape(np.array([0.09]+list(np.arange(0.1,1,0.1))+[0.91]),
                                          (1,11))).to(device)
        if self.strategy=="wide":
            tau = torch.Tensor(np.reshape(np.array([0.01]+list(np.arange(0.1,1,0.1))+[0.99]),
                                          (1,11))).to(device)
        if self.strategy=='none':
            tau = torch.Tensor(np.reshape(np.arange(0,1.1,0.1), (1,11))).to(device)
            
        # tau = torch.Tensor(np.reshape(np.arange(0,1.1,0.1),(1,11))).to(device)
        pinball_sum = torch.sum(tau * (1 - I) * (y_true[:, :,None]-y_pred) + (tau - 1) * I * (y_true[:, :,None] - y_pred))
        pinball_mean =  torch.mean(tau * (1 - I) * (y_true[:, :,None]-y_pred) + (tau - 1) * I * (y_true[:,:, None] - y_pred))
        
        return pinball_mean, pinball_sum
  
class CRPS_loss(nn.Module):
    def __init__(self):
        super().__init__() 
    def forward(self, inp,target): 
        ### inp dtype: torch.Tensor
        ###     shape: (batch size, 24 hours' obs, 11 percentiles)
        ### target dtype: torch.Tensor
        ###     shape: (batch size, 24 hours' obs)
        ### inp and target should be on the same device
        n = len(target)
        loss = [] 
        fx_prob = torch.Tensor(np.reshape(list(range(0,110,10))*inp.shape[1],(inp.shape[1],11)))
        fx_prob = fx_prob.to(device)
        for i in range(n): 
            obs = target[i,:]
            fx = inp[i,:,:]
            tmp = continuous_ranked_probability_score(obs, fx, fx_prob) 
            loss.append(tmp) 
        loss = torch.Tensor(loss).to(device)
        # print("loss: ", loss)
        loss.requires_grad_()
        return loss

def continuous_ranked_probability_score(obs, fx, fx_prob):
    """Continuous Ranked Probability Score (CRPS).

    .. math::

        \\text{CRPS} = \\frac{1}{n} \\sum_{i=1}^n \\int_{-\\infty}^{\\infty}
        (F_i(x) - \\mathbf{1} \\{x \\geq y_i \\})^2 dx

    where :math:`F_i(x)` is the CDF of the forecast at time :math:`i`,
    :math:`y_i` is the observation at time :math:`i`, and :math:`\\mathbf{1}`
    is the indicator function that transforms the observation into a step
    function (1 if :math:`x \\geq y`, 0 if :math:`x < y`). In other words, the
    CRPS measures the difference between the forecast CDF and the empirical CDF
    of the observation. The CRPS has the same units as the observation. Lower
    CRPS values indicate more accurate forecasts, where a CRPS of 0 indicates a
    perfect forecast. [1]_ [2]_ [3]_

    Parameters
    ----------
    obs : (n,) array_like
        Observations (physical unit).
    fx : (n, d) array_like
        Forecasts (physical units) of the right-hand-side of a CDF with d
        intervals (d >= 2), e.g., fx = [10 MW, 20 MW, 30 MW] is interpreted as
        <= 10 MW, <= 20 MW, <= 30 MW.
    fx_prob : (n, d) array_like
        Probability [%] associated with the forecasts.

    Returns
    -------
    crps : float
        The Continuous Ranked Probability Score, with the same units as the
        observation.

    Raises
    ------
    ValueError
        If the forecasts have incorrect dimensions; either a) the forecasts are
        for a single sample (n=1) with d CDF intervals but are given as a 1D
        array with d values or b) the forecasts are given as 2D arrays (n,d)
        but do not contain at least 2 CDF intervals (i.e. d < 2).

    Notes
    -----
    The CRPS can be calculated analytically when the forecast CDF is of a
    continuous parametric distribution, e.g., Gaussian distribution. However,
    since the Solar Forecast Arbiter makes no assumptions regarding how a
    probabilistic forecast was generated, the CRPS is instead calculated using
    numerical integration of the discretized forecast CDF. Therefore, the
    accuracy of the CRPS calculation is limited by the precision of the
    forecast CDF. In practice, this means the forecast CDF should 1) consist of
    at least 10 intervals and 2) cover probabilities from 0% to 100%.

    References
    ----------
    .. [1] Matheson and Winkler (1976) "Scoring rules for continuous
           probability distributions." Management Science, vol. 22, pp.
           1087-1096. doi: 10.1287/mnsc.22.10.1087
    .. [2] Hersbach (2000) "Decomposition of the continuous ranked probability
           score for ensemble prediction systems." Weather Forecast, vol. 15,
           pp. 559-570. doi: 10.1175/1520-0434(2000)015<0559:DOTCRP>2.0.CO;2
    .. [3] Wilks (2019) "Statistical Methods in the Atmospheric Sciences", 4th
           ed. Oxford; Waltham, MA; Academic Press.

    """
    # match observations to fx shape: (n,) => (n, d)
    # if np.ndim(fx) < 2:
    if fx.dim()<2:
        raise ValueError("forecasts must be 2D arrays (expected (n,d), got"
                         f"{np.shape(fx)})")
    # elif np.shape(fx)[1] < 2:
    elif fx.shape[1] < 2:
        raise ValueError("forecasts must have d >= 2 CDF intervals "
                         f"(expected >= 2, got {np.shape(fx)[1]})")

    n = len(fx)

    # extend CDF min to ensure obs within forecast support
    # fx.shape = (n, d) ==> (n, d + 1)
    fx_min = torch.minimum(obs, fx[:, 0])
    fx = torch.hstack([fx_min[:, np.newaxis], fx])
    fx_prob = torch.hstack([torch.zeros([n, 1]).to(device), fx_prob])

    # extend CDF max to ensure obs within forecast support
    # fx.shape = (n, d + 1) ==> (n, d + 2)
    idx = (fx[:, -1] < obs)
    fx_max = torch.maximum(obs, fx[:, -1])
    # fx = torch.hstack([fx, fx_max[:, np.newaxis]])
    fx = torch.hstack([fx, fx_max[:, None]])
    fx_prob = torch.hstack([fx_prob, torch.full([n, 1], 100).to(device)])

    # indicator function:
    # - left of the obs is 0.0
    # - obs and right of the obs is 1.0
    o = torch.where(fx >= obs[:, None], 1.0, 0.0)

    # correct behavior when obs > max fx:
    # - should be 0 over range: max fx < x < obs
    o[idx, -1] = 0.0

    # forecast probabilities [unitless]
    f = fx_prob / 100.0

    # integrate along each sample, then average all samples
    crps = torch.mean(torch.trapezoid((f - o) ** 2, x=fx, axis=1))

    return crps


def crps_skill_score(obs, fx, fx_prob, ref, ref_prob):
    """CRPS skill score.

        CRPSS = 1 - CRPS_fx / CRPS_ref

    where CRPS_fx is the CPRS of the evaluated forecast and CRPS_ref is the
    CRPS of a reference forecast.

    Parameters
    ----------
    obs : (n,) array_like
        Observations (physical unit).
    fx : (n, d) array_like
        Forecasts (physical units) of the right-hand-side of a CDF with d
        intervals (d >= 2), e.g., fx = [10 MW, 20 MW, 30 MW] is interpreted as
        <= 10 MW, <= 20 MW, <= 30 MW.
    fx_prob : (n,) array_like
        Probability [%] associated with the forecasts.
    ref : (n, d) array_like
        Reference forecasts (physical units) of the right-hand-side of a CDF
        with d intervals (d >= 2), e.g., fx = [10 MW, 20 MW, 30 MW] is
        interpreted as <= 10 MW, <= 20 MW, <= 30 MW.
    ref_prob : (n,) array_like
        Probability [%] associated with the reference forecast.

    Returns
    -------
    skill : float
        The CRPS skill score [unitless].

    See Also
    --------
    :py:func:`solarforecastarbiter.metrics.probabilistic.continuous_ranked_probability_score`

    """

    if np.isscalar(ref):
        return np.nan
    else:
        crps_fx = continuous_ranked_probability_score(obs, fx, fx_prob)
        crps_ref = continuous_ranked_probability_score(obs, ref, ref_prob)

        if crps_fx == crps_ref:
            return 0.0
        elif crps_ref == 0.0:
            # avoid divide by zero
            return np.NINF
        else:
            return 1 - crps_fx / crps_ref
        
class NoFussCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    pytorch's CrossEntropyLoss is fussy: 1) needs Long (int64) targets only, and 2) only 1D.
    This function satisfies these requirements
    """

    def forward(self, inp, target):
        return F.cross_entropy(inp, target.long().squeeze(), weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)


class MaskedMSELoss(nn.Module):
    """ Masked MSE Loss
    """

    def __init__(self, reduction: str = 'mean'):

        super().__init__()

        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=self.reduction)

    def forward(self,
                y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        """Compute the loss between a target value and a prediction.

        Args:
            y_pred: Estimated values
            y_true: Target values
            mask: boolean tensor with 0s at places where values should be ignored and 1s where they should be considered

        Returns
        -------
        if reduction == 'none':
            (num_active,) Loss for each active batch element as a tensor with gradient attached.
        if reduction == 'mean':
            scalar mean loss over batch as a tensor with gradient attached.
        """

        # for this particular loss, one may also elementwise multiply y_pred and y_true with the inverted mask
        masked_pred = torch.masked_select(y_pred, mask)
        masked_true = torch.masked_select(y_true, mask)

        return self.mse_loss(masked_pred, masked_true)
